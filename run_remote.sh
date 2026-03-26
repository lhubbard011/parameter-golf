#!/bin/bash
#
# run_remote.sh — Run pgolf training on remote GPU with reliable logging.
#
# Usage:
#   ./run_remote.sh [ENV_VARS...] [-- description]
#
# Examples:
#   ./run_remote.sh                                    # baseline
#   ./run_remote.sh MATRIX_LR=0.06 -- muon lr 0.06
#   ./run_remote.sh MATRIX_LR=0.06 WARMDOWN_ITERS=13000 -- autoresearch findings
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Instance config
IP=$(sed -n '2p' remote_instance 2>/dev/null || echo "")
SSH_KEY=~/Downloads/autor-ssh-1.pem
SOPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"
SMS_TO="5036802823@vtext.com"

sms() {
    local msg="$1"
    echo "[SMS] $msg"
    printf "Subject: pgolf\n\n%s" "$msg" | msmtp "$SMS_TO"
}

if [[ -z "$IP" ]]; then
    echo "ERROR: No instance IP. Write IP to remote_instance file."
    exit 1
fi

# Parse args: env vars before --, description after --
ENV_VARS=""
DESC="run"
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift; DESC="$*"; break
    fi
    ENV_VARS="$ENV_VARS $1"
    shift
done

# Always set these defaults
ENV_VARS="MAX_WALLCLOCK_SECONDS=600 ITERATIONS=50000 VAL_LOSS_EVERY=0 $ENV_VARS"

RUN_ID="$(echo "$DESC" | tr ' ' '_' | tr -cd 'a-zA-Z0-9_-')"

echo "[$(date '+%H:%M:%S')] Running: $DESC"
echo "[$(date '+%H:%M:%S')] Env: $ENV_VARS"

# ── TEXT: experiment starting ──────────────────────────────────────────────
sms "STARTING: $DESC"

# Sync code to remote
ssh $SOPTS ubuntu@$IP "cd ~/parameter-golf && git fetch origin && git reset --hard origin/autoresearch-findings" 2>&1

# Run training ON THE REMOTE with output saved to a remote log file.
# python3 -u = unbuffered, stdbuf -oL = line-buffered — so we can peek mid-run.
ssh $SOPTS ubuntu@$IP "
cd ~/parameter-golf &&
stdbuf -oL $ENV_VARS RUN_ID=$RUN_ID python3 -u train_gpt.py > run_${RUN_ID}.log 2>&1
echo \$? > run_${RUN_ID}.exitcode
"

# Pull the log back
scp $SOPTS ubuntu@$IP:~/parameter-golf/run_${RUN_ID}.log ./run_${RUN_ID}.log 2>/dev/null

# Extract results
VAL_BPB=$(grep "val_bpb:" ./run_${RUN_ID}.log 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || true)
VAL_LOSS=$(grep "val_loss:" ./run_${RUN_ID}.log 2>/dev/null | tail -1 | grep -oP 'val_loss:\K[0-9.]+' || true)
ARTIFACT=$(grep "compressed_artifact_bytes:" ./run_${RUN_ID}.log 2>/dev/null | grep -oP 'compressed_artifact_bytes:\K[0-9]+' || true)
STEPS=$(grep "stopping_early\|step:" ./run_${RUN_ID}.log 2>/dev/null | tail -1 | grep -oP 'step:\K[0-9]+' || true)

if [[ -z "$VAL_BPB" ]]; then
    echo "[$(date '+%H:%M:%S')] CRASH: $DESC"
    echo "Last 20 lines:"
    tail -20 ./run_${RUN_ID}.log
    # ── TEXT: crash ────────────────────────────────────────────────────────
    sms "CRASH: $DESC
$(tail -3 ./run_${RUN_ID}.log 2>/dev/null)"
    exit 1
fi

ARTIFACT_MB=$(echo "${ARTIFACT:-0}" | awk '{printf "%.2f", $1/1048576}')

echo "[$(date '+%H:%M:%S')] DONE: val_bpb=$VAL_BPB val_loss=$VAL_LOSS artifact=${ARTIFACT_MB}MB steps=$STEPS"
echo -e "$(date '+%Y-%m-%d %H:%M')\t${VAL_BPB}\t${VAL_LOSS}\t${ARTIFACT_MB}\t${STEPS}\t${DESC}" >> pgolf_results.tsv

# ── TEXT: results ──────────────────────────────────────────────────────────
sms "DONE: $DESC
bpb: $VAL_BPB
loss: $VAL_LOSS
artifact: ${ARTIFACT_MB}MB
steps: $STEPS"
