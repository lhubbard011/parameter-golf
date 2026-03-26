# Parameter Golf Strategy — Autoresearch Findings

## Competition constraints
- **16MB** compressed artifact (int8 quantized + zlib)
- **10 minutes** on 8xH100s
- **Score**: val_bpb (bits per byte) on FineWeb validation
- Current #1: 1.1194 (LeakyReLU² + TTT + Parallel Muon)
- Baseline: 1.2244 (9L 512dim 1024vocab TiedEmbeddings 4 KV heads)

## Key differences from autoresearch
| | Autoresearch | Parameter Golf |
|---|---|---|
| GPU | 1x GH200 (96GB) | 8x H100 (80GB each) |
| Time | 5 min | 10 min |
| Model size | Unlimited | 16MB compressed |
| Vocab | 8192 BPE | 1024 BPE (baseline) |
| Data | climbmix-400b | FineWeb |
| Distributed | No | DDP across 8 GPUs |

## Autoresearch findings that transfer to pgolf

### HIGH CONFIDENCE (validated over 107 experiments)

1. **Half batch size = biggest single win** (-0.0085 bpb, 6% relative)
   - Doubling optimizer steps crushed everything else
   - pgolf implication: use smallest practical batch per GPU, maximize steps in 10 min
   - Current pgolf baseline: 524K tokens/step. Try 262K or even 131K.

2. **Warmdown ratio 0.67** (not 0.5)
   - Longer cooldown consistently better. pgolf baseline uses `warmdown_iters=1200` out of 20000
   - That's only 6% warmdown! Our finding says ~67% is optimal
   - pgolf implication: warmdown_iters should be ~13000 not 1200

3. **Muon LR 0.06** (not 0.04)
   - 50% higher matrix LR was optimal
   - pgolf baseline uses 0.04 — directly applicable

4. **MLP 5x expansion > 4x > 3x > 2x**
   - Bigger FFN helps. pgolf baseline uses 2x MLP (very small!)
   - Top leaderboard entries already moved to 3x. Try 4x or 5x.

5. **Weight decay 0.1** (not 0 or 0.2)
   - Moderate regularization. pgolf leaderboard shows WD=0.04 is common.
   - Try 0.1 for Muon weight decay.

6. **Final LR fraction 0.01** (not 0)
   - Don't decay learning rate all the way to zero
   - Easy win, 1 line change

7. **VE gate channels 64 > 32**
   - Wider gating for value embeddings
   - pgolf doesn't use VE currently, but if added, use 64

### MEDIUM CONFIDENCE (patterns from sweep)

8. **Depth 7 optimal** for our compute budget
   - With 10 min on 8xH100, more steps are possible → can support deeper
   - Leaderboard is at 10-11 layers. Our finding: don't go too deep for your budget

9. **Embedding LR 0.6 is a sweet spot**
   - pgolf uses 0.6 for embed_lr already — validated

10. **SSSL window pattern > all alternatives**
    - pgolf doesn't use sliding window in training (only eval). Consider adding.

11. **ReluSquared > SwiGLU at small scale**
    - Surprising: SwiGLU was worse. LeakyReLU² is #1 on pgolf leaderboard — consistent!

### THINGS THAT HURT (avoid)
- Auxiliary losses (relay heads, factorized loss): overhead killed throughput
- Bigger models without more steps: always worse
- Warmup: any warmup ratio hurt (0.02, 0.05, 0.1 all worse)
- Label smoothing: catastrophic (1.33 bpb)
- Weight tying at autoresearch scale: destroyed training (3.2 bpb)
  - BUT pgolf baseline DOES use tied embeddings successfully at 1024 vocab
  - Likely works because 1024 vocab << 8192 vocab (smaller embedding matrix)
- PaLM-style parallel attn+MLP: hurt
- Disabling QK-norm: hurt
- Cosine warmdown schedule: worse than linear

## Priority experiments for pgolf

### Round 1: Apply validated findings to pgolf baseline
1. Increase warmdown_iters from 1200 to 13000
2. Increase matrix_lr from 0.04 to 0.06
3. Add final_lr_frac 0.01
4. Reduce train_batch_tokens from 524K to 262K
5. Increase MLP from 2x to 3x then 4x
6. Add weight decay 0.1 for Muon

### Round 2: Combine with leaderboard techniques
7. Apply above to the leading PR stack (int6 QAT + EMA + etc)
8. Add VE with gate_channels=64
9. Try ReluSquared activation (matches leaderboard #1's LeakyReLU²)
10. Add sliding window attention pattern SSSL

### Round 3: Quantization (the pgolf meta-game)
The 16MB artifact limit means quantization IS the meta-game. Lower bits = more params = better model.

| Quant level | Bits/weight | Params in 16MB | Used by |
|---|---|---|---|
| int8 + zlib | ~6.5 effective | ~20M | Baseline |
| int6 + zstd-22 | ~5.0 effective | ~26M | Most top entries |
| int5 (MLP) + int6 (attn) | ~4.5 effective | ~28M | #6 entry |
| Ternary (1-bit) | ~1.0 effective | ~100M+ | #10 entry |

Experiments to run:
11. Replace zlib with zstd-22 (free compression improvement)
12. int6 quantization (per-row, [-32, 31] range)
13. QAT (quantization-aware training) — train with simulated quantization noise
14. Mixed quant: int5 for MLP, int6 for attention, fp16 for embeddings
15. int6 + larger model (use saved bytes for more layers/width)
16. SWA (Stochastic Weight Averaging) — smoother weights quantize better

### Round 4: Paper-based innovations (from research agent)
17. Sqrt warmdown shape
18. Tail weight averaging (EMA of final 1%)
19. Batch size warmup (small→large)
20. Smear module (from modded-nanogpt)
