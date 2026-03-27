# Next GPU Session Queue

## Priority 1: SpinQuant validation (fixed implementation)
Test the corrected SpinQuant (fused rotation, no undo needed) on int6:
```bash
./run.sh MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 QUANT_BITS=6 SPIN_QUANT=1 --time 300 --desc "spinquant_v2 int6 fused rotation"
./run.sh MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 QUANT_BITS=8 SPIN_QUANT=1 --time 300 --desc "spinquant_v2 int8 fused rotation"
./run.sh MATRIX_LR=0.06 WARMDOWN_ITERS=13000 TRAIN_BATCH_TOKENS=262144 MLP_MULT=3 QUANT_BITS=6 SPIN_QUANT=1 SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=50 --time 300 --desc "spinquant_v2 int6 + SWA"
```

Compare int6 bpb with vs without rotation. Expected: rotation should reduce int6 penalty.

## Priority 2: Experiments that didn't run
- AdaFisher optimizer
- 14L dim=256 (super tiny extreme depth)
- MoR 10x (ultra depth recurrence)
- 11L + half batch (131K tokens — best arch + best training trick)
- Batch 131K tokens (quarter batch = 4x steps)

## Priority 3: Best config refinement  
- 12L dim=384 was best (1.4174) — try 13L dim=384, 12L dim=416
- 11L + int6 SpinQuant (best arch + best quant)
