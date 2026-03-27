#!/bin/bash
# Run a batch of experiments on the existing instance.
# Data must already be downloaded.
set -euo pipefail
cd "$(dirname "$0")"
source .env

IP=$(sed -n '2p' .instance_state)
SSH_KEY="$LAMBDA_SSH_KEY"
SOPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"
BRANCH="autoresearch-findings"

sms() {
    echo "[SMS] $1"
    printf "Subject: pgolf batch\n\n%s" "$1" | msmtp "5036802823@vtext.com" || true
}

run_one() {
    local desc="$1"
    shift
    local env_vars="$*"
    echo ""
    echo "============================================"
    echo "  $desc"
    echo "  $env_vars"
    echo "============================================"
    sms "STARTING: $desc"

    local run_id="$(echo "$desc" | tr ' ' '_' | tr -cd 'a-zA-Z0-9_-')"

    ssh $SOPTS ubuntu@$IP "cd ~/parameter-golf && git fetch origin && git reset --hard origin/$BRANCH" 2>&1

    ssh $SOPTS ubuntu@$IP "
        cd ~/parameter-golf &&
        export $env_vars RUN_ID=$run_id &&
        python3 -u train_gpt.py > run_${run_id}.log 2>&1
    "

    scp $SOPTS ubuntu@$IP:~/parameter-golf/run_${run_id}.log ./run_${run_id}.log 2>/dev/null

    local val_bpb=$(grep "val_bpb:" ./run_${run_id}.log 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || true)
    local steps=$(grep "stopping_early\|step:" ./run_${run_id}.log 2>/dev/null | tail -1 | grep -oP 'step:\K[0-9]+' || true)
    local params=$(grep "model_params:" ./run_${run_id}.log 2>/dev/null | grep -oP 'model_params:\K[0-9]+' || true)

    if [[ -z "$val_bpb" ]]; then
        echo "CRASH: $desc"
        sms "CRASH: $desc — $(tail -3 ./run_${run_id}.log 2>/dev/null)"
        python3 db.py add --project pgolf_batch --desc "$desc" --status crash 2>/dev/null || true
        return 1
    fi

    local params_m=$(echo "${params:-0}" | awk '{printf "%.1f", $1/1000000}')
    echo "DONE: val_bpb=$val_bpb params=${params_m}M steps=$steps"
    sms "DONE: $desc — bpb=$val_bpb params=${params_m}M steps=$steps"
    python3 db.py add --project pgolf_batch --desc "$desc" --val_bpb "$val_bpb" --params_m "$params_m" --steps "$steps" --status run 2>/dev/null || true
    return 0
}

COMMON="MAX_WALLCLOCK_SECONDS=300 ITERATIONS=50000 VAL_LOSS_EVERY=0"

sms "Starting experiment batch on $IP"

# ── 1. Hybrid Mamba(0-3) + Attention(4-8) ────────────────────────────────
run_one "hybrid mamba(0-3)+attn(4-8)" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 HYBRID_ENABLED=1 MAMBA_LAYERS=0,1,2,3"

# ── 2. Hybrid Mamba(0-6) + Attention(7-8) — more Mamba ───────────────────
run_one "hybrid mamba(0-6)+attn(7-8)" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 HYBRID_ENABLED=1 MAMBA_LAYERS=0,1,2,3,4,5,6"

# ── 3. SpinQuant + int6 (Hadamard rotation before quantization) ──────────
run_one "spinquant int6" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 QUANT_BITS=6 SPIN_QUANT=1"

# ── 4. SpinQuant + int8 (see if rotation helps even at int8) ─────────────
run_one "spinquant int8" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 QUANT_BITS=8 SPIN_QUANT=1"

# ── 5. SWA + SpinQuant + int6 (combined) ─────────────────────────────────
run_one "swa + spinquant int6" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 QUANT_BITS=6 SPIN_QUANT=1 SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=50"

# ── 6. MoR 4x + SpinQuant int8 (depth recurrence + rotation quant) ──────
run_one "mor4x spinquant int8" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 MOR_ENABLED=1 MOR_NUM_RECURSIONS=4 SPIN_QUANT=1"

# ── 7. Baseline + sqrt warmdown (paper suggestion) ───────────────────────
# For this one we need to modify train_gpt.py — skip for now, just note it
run_one "baseline autoresearch hparams (control)" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3"

# ── 8. AdaFisher optimizer (if repo is cloned) ───────────────────────────
run_one "adafisher optimizer" \
    "$COMMON USE_ADAFISHER=1 TRAIN_BATCH_TOKENS=262144" || true

# ── 9. Larger model with MoR (use saved params for width) ────────────────
run_one "mor4x dim=640 heads=10 kv=5" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 MOR_ENABLED=1 MOR_NUM_RECURSIONS=4 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5"

# ── 10. 11 layers (like leaderboard top entries) ─────────────────────────
run_one "11 layers mlp3x" \
    "$COMMON MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 NUM_LAYERS=11"

sms "ALL 10 EXPERIMENTS DONE. Check db.py list --project pgolf_batch"
echo ""
echo "=== ALL DONE ==="
python3 db.py list --project pgolf_batch
