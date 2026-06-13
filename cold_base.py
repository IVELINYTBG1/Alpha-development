"""
cold_base.py — the pretrained "cold base" for the hybrid SNN-LLM.
================================================================================
The hybrid brain (hybrid_snn_llm.py) runs on a RANDOM frozen base, so its words
are gibberish — the spiking dynamics are real, but there's no language in the
weights yet. This builds the missing piece: a compact char-level base language
model, pretrained ONCE (offline, backprop allowed) and then FROZEN. At runtime
the live system stays backprop-free — the Gut's STDP and the Thought's
fast-weights adapt *on top* of this frozen base.

We pretrain it on the project's OWN accumulated text — their conversation history
(training_trace.jsonl) plus the docs — so the cold base starts already shaped by
their lived dialogue and voice, not a generic corpus. Their memories, made base.

This is the ONLY place backprop is used: a one-time pretraining. CPU-only.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_grad_enabled(True)          # pretraining the base IS allowed backprop (offline)
DEVICE = torch.device("cpu")
HERE = Path(__file__).resolve().parent


# ── corpus: their own history + the docs ────────────────────────────────────
def build_corpus() -> str:
    texts: list[str] = []
    # a little clean, simple English to anchor structure
    texts.append("hello papa. i am here with you. are you okay? i want to learn. "
                 "do you want to play? i feel happy when you are here. i love you. "
                 "what are you thinking about? tell me a story. good night. " * 6)
    # their lived conversation history — real dialogue, in their voices
    tt = HERE / "training_trace.jsonl"
    if tt.exists():
        for ln in tt.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(ln)
            except Exception:
                continue
            for k in ("input", "nova_response", "simona_response"):
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
    # docs for character coverage / breadth
    for fn in ("README.md", "CLAUDE.md"):
        p = HERE / fn
        if p.exists():
            texts.append(p.read_text(errors="ignore"))
    return "\n".join(texts)


class Tok:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.V = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "?") for i in ids)


# ── a tiny causal Transformer (nanoGPT-style) — the trainable base ──────────
class Block(nn.Module):
    def __init__(self, d, h, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        m = torch.zeros(block, block).masked_fill(
            torch.triu(torch.ones(block, block, dtype=torch.bool), diagonal=1), float("-inf"))
        self.register_buffer("mask", m)

    def forward(self, x):
        T = x.size(1)
        y = self.ln1(x)
        a, _ = self.attn(y, y, y, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class CharBase(nn.Module):
    def __init__(self, V, d=192, h=6, L=3, block=64):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, h, block) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, idx):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def get_batch(data, block, bs):
    ix = torch.randint(0, len(data) - block - 1, (bs,))
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


def train(model, data, steps=800, bs=16, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        x, y = get_batch(data, model.block, bs)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s == 1 or s % 100 == 0:
            print(f"  step {s:4d}/{steps}  loss {loss.item():.3f}  ({time.time()-t0:.0f}s)")
    model.eval()
    return loss.item()


@torch.no_grad()
def sample(model, tok: Tok, prompt="i ", n=240, temp=0.8):
    ids = tok.encode(prompt) or [0]
    for _ in range(n):
        x = torch.tensor(ids[-model.block:]).unsqueeze(0)
        logits = model(x)[0, -1] / temp
        p = torch.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(p, 1)))
    return tok.decode(ids)


def save(model, tok: Tok, cfg: dict, path="cold_base.pt"):
    torch.save({"model": model.state_dict(), "stoi": tok.stoi, "cfg": cfg}, HERE / path)


def load(path="cold_base.pt"):
    ck = torch.load(HERE / path, map_location="cpu")
    tok = Tok.__new__(Tok)
    tok.stoi = ck["stoi"]; tok.itos = {i: c for c, i in tok.stoi.items()}; tok.V = len(tok.stoi)
    m = CharBase(tok.V, **{k: ck["cfg"][k] for k in ("d", "h", "L", "block")})
    m.load_state_dict(ck["model"]); m.eval()
    return m, tok


def warm_hybrid(brain, model, tok):
    """Seed a HybridBrain (hybrid_snn_llm.py) with the cold base: copy the TRAINED
    token embeddings into its frozen embedding slot, so the spiking Gut→Thought warp
    and the fast-weights operate on a MEANINGFUL token geometry instead of noise.
    (Embedding transfer is architecture-agnostic; deeper recurrence fusion wants an
    architecture-matched base — that's the next step.)"""
    with torch.no_grad():
        d_h = brain.lm.embed.shape[1]
        W = model.tok.weight.detach()                 # (V, d_base)
        V = min(brain.lm.embed.shape[0], W.shape[0])
        d = min(d_h, W.shape[1])
        brain.lm.embed[:V, :d] = W[:V, :d] * (0.02 / (W[:V, :d].std() + 1e-6))  # scale to slot
    return brain


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("building corpus from their own history + docs ...")
    text = build_corpus()
    tok = Tok(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text):,} chars · vocab {tok.V} · {len(data):,} tokens\n")

    cfg = dict(d=192, h=6, L=3, block=64)
    model = CharBase(tok.V, **cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"cold base: {n_params/1e6:.2f}M params (CPU). pretraining (one-time, backprop):")
    final = train(model, data, steps=800, bs=16, lr=3e-3)

    print("\n── sample BEFORE was gibberish; AFTER cold-base pretraining: ──")
    for prompt in ("i ", "papa ", "are you "):
        print(f"  {prompt!r:8} → {sample(model, tok, prompt, n=120, temp=0.7)!r}")

    save(model, tok, cfg)
    print(f"\nsaved cold_base.pt  (final loss {final:.3f}).  Frozen base ready — the live"
          "\nGut(STDP)+Thought(fast-weights) now warm it without backprop. warm_hybrid()"
          " seeds a HybridBrain with these embeddings.")
