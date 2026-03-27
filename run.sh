#!/bin/bash
#
# run.sh — One-button experiment runner for parameter-golf.
#
# Usage:
#   ./run.sh                              # launch instance, run baseline, keep instance
#   ./run.sh --kill                        # terminate instance when done
#   ./run.sh --desc "my experiment"        # custom description for results log
#   ./run.sh MATRIX_LR=0.06 MLP_MULT=3    # pass env vars to training
#   ./run.sh --time 300                    # train for 5 min (default 600 = 10 min)
#   ./run.sh --type gpu_1x_gh200           # use GH200 instead of H100
#   ./run.sh --budget 20                   # warn if spend exceeds $20
#
# Examples:
#   ./run.sh MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 --desc "autoresearch hparams"
#   ./run.sh MOR_ENABLED=1 MOR_NUM_RECURSIONS=4 --desc "MoR 4x" --time 300
#   ./run.sh --kill --desc "quick test"
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config (secrets in .env) ───────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found. Create it with LAMBDA_API_KEY, LAMBDA_SSH_KEY, GITHUB_TOKEN, SMS_TO, BRANCH"
    exit 1
fi
source "$ENV_FILE"

AKEY="$LAMBDA_API_KEY"
API="https://cloud.lambdalabs.com/api/v1"
SSH_KEY="$LAMBDA_SSH_KEY"
SOPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"
GHTOK="$GITHUB_TOKEN"
STATE_FILE="$SCRIPT_DIR/.instance_state"
COST_LOG="$SCRIPT_DIR/cost_log.tsv"

# ── Defaults ──────────────────────────────────────────────────────────────
INSTANCE_TYPE="gpu_1x_h100_sxm5"
TRAIN_TIME=600
KILL_AFTER=false
DESC=""
BUDGET=0
ENV_VARS=""

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kill)       KILL_AFTER=true; shift ;;
        --desc)       DESC="$2"; shift 2 ;;
        --time)       TRAIN_TIME="$2"; shift 2 ;;
        --type)       INSTANCE_TYPE="$2"; shift 2 ;;
        --budget)     BUDGET="$2"; shift 2 ;;
        --help|-h)
            head -17 "$0" | tail -15
            exit 0 ;;
        *=*)          ENV_VARS="$ENV_VARS $1"; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

ENV_VARS="MAX_WALLCLOCK_SECONDS=$TRAIN_TIME ITERATIONS=50000 VAL_LOSS_EVERY=0 $ENV_VARS"
[[ -z "$DESC" ]] && DESC="run $(date '+%H:%M')"

# ── Helpers ───────────────────────────────────────────────────────────────
sms() {
    local msg="$1"
    echo "[SMS] Sending: $msg"
    if ! printf "Subject: pgolf\n\n%s" "$msg" | msmtp "5036802823@vtext.com"; then
        echo "[SMS] FAILED to send text!"
    fi
}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_instance_info() {
    if [[ -f "$STATE_FILE" ]]; then
        INSTANCE_ID=$(sed -n '1p' "$STATE_FILE")
        IP=$(sed -n '2p' "$STATE_FILE")
        LAUNCH_TIME=$(sed -n '3p' "$STATE_FILE")
        PRICE_CENTS=$(sed -n '4p' "$STATE_FILE")
        return 0
    fi
    return 1
}

save_instance_info() {
    echo "$INSTANCE_ID" > "$STATE_FILE"
    echo "$IP" >> "$STATE_FILE"
    echo "$(date +%s)" >> "$STATE_FILE"
    echo "$PRICE_CENTS" >> "$STATE_FILE"
}

estimate_cost() {
    if get_instance_info; then
        local now=$(date +%s)
        local elapsed_hrs=$(echo "($now - $LAUNCH_TIME) / 3600" | bc -l)
        local cost=$(echo "$elapsed_hrs * $PRICE_CENTS / 100" | bc -l)
        printf "%.2f" "$cost"
    else
        echo "0.00"
    fi
}

# ── Instance management ──────────────────────────────────────────────────

ensure_instance() {
    # Check if we already have a running instance
    if get_instance_info; then
        local status=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instances/$INSTANCE_ID" 2>/dev/null \
            | jq -r '.data.status // empty' 2>/dev/null || true)
        if [[ "$status" == "active" ]]; then
            local cost=$(estimate_cost)
            log "Reusing instance $INSTANCE_ID at $IP (running cost: \$$cost)"
            return 0
        fi
        log "Previous instance $INSTANCE_ID is $status, launching new one..."
        rm -f "$STATE_FILE"
    fi

    # Find capacity
    log "Looking for $INSTANCE_TYPE..."
    local region=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instance-types" \
        | jq -r ".data[\"$INSTANCE_TYPE\"].regions_with_capacity_available[0]?.name // empty" 2>/dev/null || true)

    if [[ -z "$region" ]]; then
        log "No $INSTANCE_TYPE available. Polling..."
        sms "No $INSTANCE_TYPE capacity. Polling..."
        while true; do
            region=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instance-types" \
                | jq -r ".data[\"$INSTANCE_TYPE\"].regions_with_capacity_available[0]?.name // empty" 2>/dev/null || true)
            [[ -n "$region" ]] && break
            sleep 120
        done
    fi

    # Get price
    PRICE_CENTS=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instance-types" \
        | jq -r ".data[\"$INSTANCE_TYPE\"].instance_type.price_cents_per_hour // 0" 2>/dev/null || echo 0)

    # Launch
    log "Launching $INSTANCE_TYPE in $region (\$$(echo "$PRICE_CENTS / 100" | bc -l | xargs printf '%.2f')/hr)..."
    local resp=$(curl -sf -X POST -H "Authorization: Bearer $AKEY" -H "Content-Type: application/json" \
        -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$INSTANCE_TYPE\",\"ssh_key_names\":[\"autor-ssh-1\"],\"name\":\"pgolf\"}" \
        "$API/instance-operations/launch")
    INSTANCE_ID=$(echo "$resp" | jq -r '.data.instance_ids[0] // empty')
    [[ -z "$INSTANCE_ID" ]] && { log "Launch failed: $resp"; exit 1; }
    log "Instance: $INSTANCE_ID"

    # Wait for active
    log "Booting..."
    while true; do
        local info=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instances/$INSTANCE_ID" 2>/dev/null || true)
        local st=$(echo "$info" | jq -r '.data.status // empty' 2>/dev/null || true)
        IP=$(echo "$info" | jq -r '.data.ip // empty' 2>/dev/null || true)
        [[ "$st" == "active" && -n "$IP" && "$IP" != "null" ]] && break
        [[ "$st" == "terminated" || "$st" == "error" || "$st" == "unhealthy" ]] && { log "FAILED: $st"; exit 1; }
        sleep 10
    done
    log "Active at $IP"
    save_instance_info

    # Wait for SSH
    for i in $(seq 1 30); do ssh $SOPTS ubuntu@$IP "echo ok" &>/dev/null && break; sleep 5; done
    log "SSH ready"

    # Setup
    log "Setting up..."
    ssh $SOPTS ubuntu@$IP "
        export PATH=\$HOME/.local/bin:\$PATH
        command -v uv &>/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; }
        [ -d ~/parameter-golf ] || git clone https://${GHTOK}@github.com/lhubbard011/parameter-golf.git ~/parameter-golf
        cd ~/parameter-golf && git fetch origin && git checkout $BRANCH 2>/dev/null || git checkout -b $BRANCH origin/$BRANCH
        git reset --hard origin/$BRANCH
        pip install -q sentencepiece numpy torch 2>/dev/null
        [ -d data/datasets/fineweb10B_sp1024 ] || python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
    " 2>&1
    log "Ready"
    sms "Instance up at $IP — \$$(echo "$PRICE_CENTS / 100" | bc -l | xargs printf '%.2f')/hr"
}

terminate_instance() {
    if get_instance_info; then
        local cost=$(estimate_cost)
        log "Terminating $INSTANCE_ID (total cost: \$$cost)"
        curl -sf -X POST -H "Authorization: Bearer $AKEY" -H "Content-Type: application/json" \
            -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" "$API/instance-operations/terminate" >/dev/null
        # Log cost
        echo -e "$(date '+%Y-%m-%d %H:%M')\t$INSTANCE_ID\t$cost\t$INSTANCE_TYPE" >> "$COST_LOG"
        rm -f "$STATE_FILE"
        sms "Instance terminated. Cost: \$$cost"
        log "Terminated"
    fi
}

# ── Budget check ──────────────────────────────────────────────────────────
check_budget() {
    if [[ "$BUDGET" -gt 0 ]] && get_instance_info; then
        local cost=$(estimate_cost)
        if (( $(echo "$cost > $BUDGET" | bc -l) )); then
            log "BUDGET EXCEEDED: \$$cost > \$$BUDGET — terminating!"
            sms "BUDGET EXCEEDED: \$$cost > \$$BUDGET — terminating instance!"
            terminate_instance
            exit 1
        fi
    fi
}

# ── Run experiment ────────────────────────────────────────────────────────
run_experiment() {
    local run_id="$(echo "$DESC" | tr ' ' '_' | tr -cd 'a-zA-Z0-9_-')"
    log "Experiment: $DESC"
    log "Env: $ENV_VARS"
    sms "STARTING: $DESC"

    # Sync code
    ssh $SOPTS ubuntu@$IP "cd ~/parameter-golf && git fetch origin && git reset --hard origin/$BRANCH" 2>&1

    # Train (output saved on remote, unbuffered)
    ssh $SOPTS ubuntu@$IP "
        cd ~/parameter-golf &&
        export $ENV_VARS RUN_ID=$run_id &&
        python3 -u train_gpt.py > run_${run_id}.log 2>&1
    "

    # Pull log
    scp $SOPTS ubuntu@$IP:~/parameter-golf/run_${run_id}.log ./run_${run_id}.log 2>/dev/null

    # Extract results
    local val_bpb=$(grep "val_bpb:" ./run_${run_id}.log 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || true)
    local val_loss=$(grep "val_loss:" ./run_${run_id}.log 2>/dev/null | tail -1 | grep -oP 'val_loss:\K[0-9.]+' || true)
    local steps=$(grep "stopping_early\|step:" ./run_${run_id}.log 2>/dev/null | tail -1 | grep -oP 'step:\K[0-9]+' || true)
    local params=$(grep "model_params:" ./run_${run_id}.log 2>/dev/null | grep -oP 'model_params:\K[0-9]+' || true)
    local artifact=$(grep "int8.*bytes" ./run_${run_id}.log 2>/dev/null | grep -oP 'int8\+zlib: \K[0-9]+' || true)

    if [[ -z "$val_bpb" ]]; then
        log "CRASH: $DESC"
        sms "CRASH: $DESC
$(tail -5 ./run_${run_id}.log 2>/dev/null)"
        python3 db.py add --project pgolf --desc "$DESC" --status crash 2>/dev/null || true
        return 1
    fi

    local params_m=$(echo "${params:-0}" | awk '{printf "%.1f", $1/1000000}')
    local artifact_mb=$(echo "${artifact:-0}" | awk '{printf "%.1f", $1/1048576}')
    local cost=$(estimate_cost)

    log "DONE: val_bpb=$val_bpb params=${params_m}M artifact=${artifact_mb}MB steps=$steps cost=\$$cost"

    # Log to TSV
    echo -e "$(date '+%Y-%m-%d %H:%M')\t${val_bpb}\t${val_loss}\t${artifact_mb}\t${steps}\t${DESC}" >> pgolf_results.tsv

    # Log to SQLite
    python3 db.py add --project pgolf --desc "$DESC" --val_bpb "$val_bpb" --val_loss "$val_loss" \
        --params_m "$params_m" --artifact_mb "$artifact_mb" --steps "$steps" --status run 2>/dev/null || true

    sms "DONE: $DESC
bpb: $val_bpb
params: ${params_m}M
artifact: ${artifact_mb}MB
steps: $steps
cost: \$$cost"
}

# ── Main ──────────────────────────────────────────────────────────────────
trap 'echo; log "Interrupted"; [[ "$KILL_AFTER" == true ]] && terminate_instance; exit 130' INT

log "========================================"
log "  Parameter Golf Runner"
log "  Desc: $DESC"
log "  Time: ${TRAIN_TIME}s"
[[ "$BUDGET" -gt 0 ]] && log "  Budget: \$$BUDGET"
[[ "$KILL_AFTER" == true ]] && log "  Will terminate after"
log "========================================"

ensure_instance
check_budget
run_experiment
check_budget

if [[ "$KILL_AFTER" == true ]]; then
    terminate_instance
else
    local cost=$(estimate_cost)
    log "Instance still running at $IP (cost so far: \$$cost)"
    log "Run './run.sh --kill' or 'run.sh <more experiments>' to continue"
fi
