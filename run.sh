#!/bin/bash
#
# run.sh — Run whatever is in train_gpt.py on a Lambda GPU.
#
# Usage:
#   ./run.sh "my experiment"           # launch/reuse instance, run train_gpt.py, download model
#   ./run.sh "my experiment" --kill     # terminate instance after
#   ./run.sh "my experiment" --type gpu_1x_gh200
#   ./run.sh "my experiment" --budget 10
#   ./kill.sh                           # emergency terminate
#
# Workflow:
#   1. Edit train_gpt.py with your changes
#   2. git commit
#   3. ./run.sh "description of what you changed"
#   4. Results logged, model downloaded, text sent
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .env 2>/dev/null || { echo "ERROR: .env not found"; exit 1; }

AKEY="$LAMBDA_API_KEY"
API="https://cloud.lambdalabs.com/api/v1"
SSH_KEY="$LAMBDA_SSH_KEY"
GHTOK="$GITHUB_TOKEN"
STATE_FILE="$SCRIPT_DIR/.instance_state"
COST_LOG="$SCRIPT_DIR/cost_log.tsv"

INSTANCE_TYPE="gpu_1x_h100_sxm5"
KILL_AFTER=false
BUDGET=0
DESC=""

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kill)    KILL_AFTER=true; shift ;;
        --type)    INSTANCE_TYPE="$2"; shift 2 ;;
        --budget)  BUDGET="$2"; shift 2 ;;
        --help|-h) head -15 "$0" | tail -13; exit 0 ;;
        *)         DESC="$1"; shift ;;
    esac
done
[[ -z "$DESC" ]] && { echo "Usage: ./run.sh \"description\" [--kill] [--type TYPE] [--budget N]"; exit 1; }

RUN_ID="$(echo "$DESC" | tr ' ' '_' | tr -cd 'a-zA-Z0-9_-')"

# ── Helpers ───────────────────────────────────────────────────────────────
sms() {
    echo "[SMS] $1"
    printf "Subject: pgolf\n\n%s" "$1" | msmtp "5036802823@vtext.com" || echo "[SMS] FAILED"
}
log() { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_run() { ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY" ubuntu@"$IP" "$@"; }

get_instance_info() {
    [[ -f "$STATE_FILE" ]] || return 1
    INSTANCE_ID=$(sed -n '1p' "$STATE_FILE")
    IP=$(sed -n '2p' "$STATE_FILE")
    LAUNCH_TIME=$(sed -n '3p' "$STATE_FILE")
    PRICE_CENTS=$(sed -n '4p' "$STATE_FILE")
}

save_instance_info() {
    printf '%s\n%s\n%s\n%s\n' "$INSTANCE_ID" "$IP" "$(date +%s)" "$PRICE_CENTS" > "$STATE_FILE"
}

estimate_cost() {
    get_instance_info 2>/dev/null || { echo "0.00"; return; }
    local hrs=$(echo "($(date +%s) - $LAUNCH_TIME) / 3600" | bc -l)
    printf "%.2f" "$(echo "$hrs * $PRICE_CENTS / 100" | bc -l)"
}

# ── Instance management ──────────────────────────────────────────────────
ensure_instance() {
    if get_instance_info; then
        local st=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instances/$INSTANCE_ID" \
            | jq -r '.data.status // empty' 2>/dev/null || true)
        if [[ "$st" == "active" ]]; then
            log "Reusing instance at $IP (cost: \$$(estimate_cost))"
            return 0
        fi
        rm -f "$STATE_FILE"
    fi

    log "Looking for $INSTANCE_TYPE..."
    local region=""
    while true; do
        region=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instance-types" \
            | jq -r ".data[\"$INSTANCE_TYPE\"].regions_with_capacity_available[0]?.name // empty" 2>/dev/null || true)
        [[ -n "$region" ]] && break
        log "No capacity. Polling..."
        sleep 120
    done

    PRICE_CENTS=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instance-types" \
        | jq -r ".data[\"$INSTANCE_TYPE\"].instance_type.price_cents_per_hour // 0" 2>/dev/null || echo 0)

    log "Launching in $region..."
    local resp=$(curl -sf -X POST -H "Authorization: Bearer $AKEY" -H "Content-Type: application/json" \
        -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$INSTANCE_TYPE\",\"ssh_key_names\":[\"autor-ssh-1\"],\"name\":\"pgolf\"}" \
        "$API/instance-operations/launch")
    INSTANCE_ID=$(echo "$resp" | jq -r '.data.instance_ids[0] // empty')
    [[ -z "$INSTANCE_ID" ]] && { log "Launch failed: $resp"; exit 1; }

    while true; do
        local info=$(curl -sf -H "Authorization: Bearer $AKEY" "$API/instances/$INSTANCE_ID" 2>/dev/null || true)
        local st=$(echo "$info" | jq -r '.data.status // empty' 2>/dev/null || true)
        IP=$(echo "$info" | jq -r '.data.ip // empty' 2>/dev/null || true)
        [[ "$st" == "active" && -n "$IP" && "$IP" != "null" ]] && break
        [[ "$st" == "terminated" || "$st" == "error" || "$st" == "unhealthy" ]] && { log "FAILED: $st"; exit 1; }
        sleep 10
    done
    save_instance_info
    for i in $(seq 1 30); do ssh_run "echo ok" &>/dev/null && break; sleep 5; done

    log "Setting up..."
    ssh_run "
        pip install -q sentencepiece numpy torch huggingface-hub datasets tqdm 2>/dev/null
        [ -d ~/parameter-golf ] || git clone https://${GHTOK}@github.com/lhubbard011/parameter-golf.git ~/parameter-golf
        cd ~/parameter-golf && git fetch origin && git checkout ${BRANCH} 2>/dev/null || git checkout -b ${BRANCH} origin/${BRANCH}
        [ -d data/datasets/fineweb10B_sp1024 ] || python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
    " 2>&1
    sms "Instance up at $IP"
}

terminate_instance() {
    get_instance_info 2>/dev/null || return
    local cost=$(estimate_cost)
    log "Terminating (cost: \$$cost)"
    curl -sf -X POST -H "Authorization: Bearer $AKEY" -H "Content-Type: application/json" \
        -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" "$API/instance-operations/terminate" >/dev/null
    echo -e "$(date '+%Y-%m-%d %H:%M')\t$INSTANCE_ID\t$cost\t$INSTANCE_TYPE" >> "$COST_LOG"
    rm -f "$STATE_FILE"
    sms "Terminated. Cost: \$$cost"
}

# ── Main ──────────────────────────────────────────────────────────────────
trap 'echo; log "Interrupted"; [[ "$KILL_AFTER" == true ]] && terminate_instance; exit 130' INT

log "========================================"
log "  $DESC"
log "========================================"

ensure_instance

if [[ "$BUDGET" -gt 0 ]]; then
    cost=$(estimate_cost)
    (( $(echo "$cost > $BUDGET" | bc -l) )) && { sms "BUDGET \$$cost > \$$BUDGET"; terminate_instance; exit 1; }
fi

sms "STARTING: $DESC"

# Sync committed code to remote
ssh_run "cd ~/parameter-golf && git fetch origin && git reset --hard origin/${BRANCH}" 2>&1

# Run whatever is in train_gpt.py
ssh_run "cd ~/parameter-golf && python3 -u train_gpt.py > run_${RUN_ID}.log 2>&1"

# Save + download model
ssh_run "cd ~/parameter-golf && cp final_model.int8.ptz models_${RUN_ID}.int8.ptz 2>/dev/null; cp final_model.pt models_${RUN_ID}.pt 2>/dev/null" 2>/dev/null
mkdir -p ./models
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY" \
    ubuntu@"$IP":~/parameter-golf/run_${RUN_ID}.log ./run_${RUN_ID}.log 2>/dev/null
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY" \
    ubuntu@"$IP":~/parameter-golf/models_${RUN_ID}.int8.ptz ./models/${RUN_ID}.int8.ptz 2>/dev/null

# Extract results
val_bpb=$(grep "val_bpb:" ./run_${RUN_ID}.log 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || true)
steps=$(grep "stopping_early\|step:" ./run_${RUN_ID}.log 2>/dev/null | tail -1 | grep -oP 'step:\K[0-9]+' || true)
params=$(grep "model_params:" ./run_${RUN_ID}.log 2>/dev/null | grep -oP 'model_params:\K[0-9]+' || true)
artifact_bytes=$(grep "int8.*bytes" ./run_${RUN_ID}.log 2>/dev/null | grep -oP 'int8\+zlib: \K[0-9]+' || true)

if [[ -z "$val_bpb" ]]; then
    log "CRASH"
    sms "CRASH: $DESC
$(tail -5 ./run_${RUN_ID}.log 2>/dev/null)"
    python3 db.py add --project pgolf --desc "$DESC" --status crash 2>/dev/null || true
else
    params_m=$(echo "${params:-0}" | awk '{printf "%.1f", $1/1000000}')
    artifact_mb=$(echo "${artifact_bytes:-0}" | awk '{printf "%.1f", $1/1048576}')
    cost=$(estimate_cost)
    log "DONE: bpb=$val_bpb params=${params_m}M artifact=${artifact_mb}MB steps=$steps"
    echo -e "$(date '+%Y-%m-%d %H:%M')\t${val_bpb}\t${params_m}\t${artifact_mb}\t${steps}\t${DESC}" >> pgolf_results.tsv
    python3 db.py add --project pgolf --desc "$DESC" --val_bpb "$val_bpb" --params_m "$params_m" \
        --artifact_mb "$artifact_mb" --steps "$steps" --status run 2>/dev/null || true
    sms "DONE: $DESC
bpb: $val_bpb
params: ${params_m}M
steps: $steps
cost: \$$cost"
fi

if [[ "$KILL_AFTER" == true ]]; then
    terminate_instance
else
    log "Instance still running at $IP (\$$(estimate_cost))"
fi
