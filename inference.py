#!/usr/bin/env python3
"""
Local inference for parameter-golf models.
Usage:
    python inference.py                          # interactive
    python inference.py "Once upon a time"       # from prompt
"""
import io
import sys
import zlib
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim, self.base = dim, base
        self._len = 0
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _build(self, seq_len, device, dtype):
        if seq_len <= self._len:
            return
        self._len = seq_len
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        self.cos = freqs.cos().to(dtype)
        self.sin = freqs.sin().to(dtype)

    def forward(self, x):
        self._build(x.size(-2), x.device, x.dtype)
        c, s = self.cos[:x.size(-2)], self.sin[:x.size(-2)]
        d = x.shape[-1] // 2
        return torch.cat([x[..., :d] * c + x[..., d:] * s, x[..., :d] * (-s) + x[..., d:] * c], -1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.c_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q), self.rotary(k)
        q = F.rms_norm(q, (q.size(-1),)) * self.q_gain[None, :, None, None]
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.fc = nn.Linear(dim, dim * mult, bias=False)
        self.proj = nn.Linear(dim * mult, dim, bias=False)
    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))

    def forward(self, x, x0):
        mix = self.resid_mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=1024, num_layers=9, model_dim=512, num_heads=8,
                 num_kv_heads=4, mlp_mult=3, logit_softcap=30.0, rope_base=10000.0,
                 qk_gain_init=1.5):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear = SmearGate(model_dim)
        num_enc = num_layers // 2
        num_dec = num_layers - num_enc
        self.num_encoder_layers = num_enc
        self.num_decoder_layers = num_dec
        self.skip_weights = nn.Parameter(torch.ones(min(num_enc, num_dec), model_dim))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i][None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits


def dequantize(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].float()
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu()
        orig = passthrough_orig_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(dtype=getattr(torch, orig))
        out[name] = out_t
    return out


def load_model(path, **kwargs):
    model = GPT(**kwargs)
    if str(path).endswith(".ptz"):
        with open(path, "rb") as f:
            obj = torch.load(io.BytesIO(zlib.decompress(f.read())), map_location="cpu")
        state = dequantize(obj)
    else:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=50):
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long)
    for _ in range(max_tokens):
        logits = model(idx[:, -1024:])[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, next_token], dim=1)
        print(tokenizer.decode([next_token.item()]), end="", flush=True)
    print()


def main():
    model_path = "final_model.int8.ptz"
    prompt = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model_path = arg.split("=", 1)[1]
        elif not arg.startswith("--"):
            prompt = arg

    print(f"Loading {model_path}...")
    model = load_model(model_path, mlp_mult=3)
    params = sum(p.numel() for p in model.parameters())
    print(f"{params:,} params loaded")

    tokenizer = spm.SentencePieceProcessor(model_file="data/tokenizers/fineweb_1024_bpe.model")

    if prompt:
        generate(model, tokenizer, prompt)
    else:
        print("Interactive mode. Ctrl+C to quit.\n")
        while True:
            try:
                p = input("> ")
                if p.strip():
                    generate(model, tokenizer, p)
                    print()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

if __name__ == "__main__":
    main()
