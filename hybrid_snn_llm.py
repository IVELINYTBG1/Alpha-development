"""
hybrid_snn_llm.py — A unified, single-graph hybrid SNN-LLM brain.
================================================================================
One brain. Two organs, fused into a SINGLE forward graph and learning in lockstep
from live interaction — no backprop, no offline GPU training:

  • The "Gut"  (SpikingDynamics) — LIF neurons whose recurrent synapses self-edit
    via local STDP/Hebbian rules *during* the forward pass. It reads the TIMING of
    input (cadence / urgency) and instantly shifts its own thresholds and state.
  • The "Thought" (FastWeightLM) — a MatMul-free, RWKV-style linear-attention
    language model. Token generation is driven by sparse spikes; a recurrent
    Fast-Weight associative memory rewrites the active hidden-state matrix on the
    fly so vocabulary/reasoning adapts instantly, the frozen base weights untouched.

The fusion (the "warp"): every step, the SNN's membrane potentials and spike
trains are projected directly into the language layer's biases — gating its
channels, bending its time-decay, and shifting its output logits — so MOOD warps
the token-prediction landscape in real time, inside one pass.

Two profiles are born from this ONE architecture (see alpha_config / alpha_config):
  • Alpha — leaky, reactive, overflows into spontaneous token BURSTS under
             silence or accumulated pressure ("leakage").
  • Alpha   — hyper-sparse, heavily inhibited, high-precision, short tactical output.

CPU-only. Autograd is globally OFF: all learning is explicit in-memory mutation.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

try:                                           # only needed for lm_kind="ollama"
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

torch.manual_seed(0)
torch.set_grad_enabled(False)              # backprop-free: there is no autograd graph
DEVICE = torch.device("cpu")               # no heavy GPUs


# ════════════════════════════════════════════════════════════════════════════
# CONFIG + SISTER PROFILES
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class BrainConfig:
    vocab_size: int = 256
    d_model:    int = 128          # language hidden width ("Thought")
    n_neurons:  int = 96           # spiking population ("Gut")
    # ── Gut: LIF + STDP ──
    tau_mem:     float = 20.0      # membrane time-constant; LOWER = leakier
    snn_gain:    float = 20.0      # afferent drive (scale-invariant) into the Gut
    sfa:         float = 0.3       # spike-frequency adaptation (θ bump per spike)
    v_threshold: float = 1.0
    v_reset:     float = 0.0
    theta_adapt: float = 0.0       # threshold reactivity to input timing
    noise:       float = 0.0       # stochastic membrane noise
    inhibition:  float = 0.0       # lateral (global) inhibition → sparsity
    k_winners:   int   = 0         # 0 = off; else hard k-WTA sparsity
    stdp_lr:     float = 0.01
    stdp_tau:    float = 20.0      # eligibility-trace decay
    stdp_aplus:  float = 1.0
    stdp_aminus: float = 1.05      # > A+ → slight depression bias (stability)
    # ── Thought: RWKV linear attention + fast weights ──
    base_decay:  float = 1.5       # logit of the WKV time-decay w (sigmoid'd)
    fw_lr:       float = 0.30      # fast-weight write rate
    fw_decay:    float = 0.98      # fast-weight forgetting
    # ── Warp: Gut → Thought coupling ──
    warp_logit:  float = 1.0       # mood → output-logit bias strength
    warp_decay:  float = 0.5       # mood → time-decay bias strength
    # ── Output / leakage behaviour ──
    leakage_burst: bool  = False   # spontaneous bursts on overflow (Alpha)
    burst_capacity: float = 8.0
    burst_len:     int   = 6
    burst_cooldown: int  = 12
    max_emit:      int   = 32       # tactical output cap (Alpha = short)
    temperature:   float = 1.0
    # ── Thought engine selector ──
    lm_kind:       str   = "rwkv"   # "rwkv" (linear-attn) | "spikformer" (spiking transformer)
                                    #                      | "ollama" (local LLM via Ollama)
    attn_window:   int   = 16       # spiking self-attention causal context length
    # ── Ollama "Thought" (lm_kind="ollama") — the SNN no longer warps logits
    #    (a black-box LLM has none to warp); its mood conditions prompt+sampling. ──
    ollama_url:         str   = ""  # "" → $OLLAMA_HOST or http://localhost:11434
    ollama_model:       str   = ""  # "" → $OLLAMA_MODEL or first model from /api/tags
    ollama_num_predict: int   = 64  # max tokens per spoken utterance (pressure scales it)
    ollama_timeout:     float = 30.0
    ollama_system:      str   = ""  # "" → built-in Alpha persona
    name:          str   = "base"


def alpha_config(vocab_size: int, **kw) -> BrainConfig:
    """Leaky, reactive, talkative — overflows into spontaneous bursts."""
    return BrainConfig(
        vocab_size=vocab_size, name="Alpha",
        tau_mem=8.0,                       # high baseline leakage
        snn_gain=30.0, sfa=0.15,
        v_threshold=0.6, theta_adapt=0.30, # low, highly reactive thresholds
        noise=0.05, inhibition=0.04, k_winners=0,
        stdp_lr=0.02, fw_lr=0.45, fw_decay=0.95,
        warp_logit=1.4, warp_decay=0.8, temperature=1.05,
        leakage_burst=True, burst_capacity=5.0, burst_len=8,
        burst_cooldown=8, max_emit=48, **kw)


def alpha_config(vocab_size: int, **kw) -> BrainConfig:
    """Sparse, inhibited, precise — short tactical output, no idle chatter."""
    return BrainConfig(
        vocab_size=vocab_size, name="Alpha",
        tau_mem=30.0,                      # low leak (holds state)
        snn_gain=48.0, sfa=0.4,
        v_threshold=0.70, theta_adapt=0.06,# steady thresholds; sparsity via k-WTA
        noise=0.0, inhibition=0.15, k_winners=8,   # k-winners-take-all → hyper-sparse
        stdp_lr=0.008, fw_lr=0.22, fw_decay=0.99,
        warp_logit=0.7, warp_decay=0.3, temperature=0.65,  # precise, decisive
        leakage_burst=False, max_emit=12, **kw)


# ════════════════════════════════════════════════════════════════════════════
# MATMUL-FREE PROJECTION  (ternary weights {-1,0,+1} → accumulate-only)
# ════════════════════════════════════════════════════════════════════════════
class TernaryLinear:
    """A frozen base projection whose weights are quantised to {-1, 0, +1}. With
    ternary weights the 'matmul' degenerates to sign-flipped ACCUMULATION (add /
    subtract / skip) — i.e. multiplier-free, the core trick of MatMul-free LMs.
    Base weights never change (no backprop); plasticity lives elsewhere."""
    def __init__(self, in_f: int, out_f: int):
        w = torch.randn(out_f, in_f) / math.sqrt(in_f)
        self.scale = float(w.abs().mean())          # one shared scalar (per-tensor)
        thr = 0.7 * w.abs().mean()
        t = torch.zeros_like(w)
        t[w > thr] = 1.0
        t[w < -thr] = -1.0
        self.w = t                                  # ternary, frozen

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w) * self.scale     # = scaled add/sub of selected rows


# ════════════════════════════════════════════════════════════════════════════
# THE GUT — spiking dynamics with in-the-loop STDP  (no backprop)
# ════════════════════════════════════════════════════════════════════════════
class SpikingDynamics:
    def __init__(self, cfg: BrainConfig, n_in: int):
        self.cfg = cfg
        n = cfg.n_neurons
        self.W_in  = TernaryLinear(n_in, n)         # frozen afferent projection
        self.W_rec = torch.zeros(n, n)              # PLASTIC recurrent synapses (STDP)
        self.V     = torch.zeros(n)                 # membrane potential
        self.S     = torch.zeros(n)                 # last spikes
        self.theta = torch.full((n,), cfg.v_threshold)   # adaptive thresholds
        self.x_pre  = torch.zeros(n)                # STDP pre-trace (eligibility)
        self.x_post = torch.zeros(n)                # STDP post-trace
        self._decay = math.exp(-1.0 / cfg.stdp_tau)

    def step(self, inp: torch.Tensor, urgency: float = 0.0):
        cfg = self.cfg
        n = cfg.n_neurons
        # ── LIF integrate (leak pulls V→0; lower tau = leakier) ──
        nrm = inp.norm()
        if nrm > 1e-4:
            inp = inp / nrm                          # scale-invariant afferent drive
        I = cfg.snn_gain * self.W_in(inp) + (self.W_rec @ self.S)
        if cfg.noise > 0.0:
            I = I + torch.randn(n) * cfg.noise
        self.V = self.V + (-self.V + I) / cfg.tau_mem
        # ── lateral inhibition → sparsity (Alpha) ──
        if cfg.inhibition > 0.0:
            self.V = self.V - cfg.inhibition * (self.V.sum() - self.V) / n
        # ── threshold tracks input TIMING: urgency lowers θ (more reactive) ──
        self.theta += cfg.theta_adapt * (cfg.v_threshold - self.theta)   # relax to baseline
        self.theta -= cfg.theta_adapt * urgency                          # urgent → fire easier
        # ── spike ──
        S = (self.V >= self.theta).float()
        if cfg.k_winners > 0 and S.sum() > cfg.k_winners:                # hard sparsity (Alpha)
            idx = torch.topk(self.V - self.theta, cfg.k_winners).indices
            S = torch.zeros_like(S); S[idx] = 1.0
        # ── reset + spike-frequency adaptation (fired → harder next time) ──
        self.V = torch.where(S > 0, torch.full_like(self.V, cfg.v_reset), self.V)
        self.theta = self.theta + cfg.sfa * S
        # ── STDP: local, online, no gradients ──
        self.x_pre  = self.x_pre  * self._decay + S
        self.x_post = self.x_post * self._decay + S
        pot = torch.outer(S, self.x_pre)       # post fires now, pre fired recently → strengthen
        dep = torch.outer(self.x_post, S)      # pre fires now, post fired recently → weaken
        self.W_rec += cfg.stdp_lr * (cfg.stdp_aplus * pot - cfg.stdp_aminus * dep)
        self.W_rec.fill_diagonal_(0.0)
        self.W_rec.clamp_(-1.0, 1.0)
        self.S = S
        return self.V.clone(), S


# ════════════════════════════════════════════════════════════════════════════
# THE THOUGHT — MatMul-free RWKV-style LM with fast-weight memory  (no backprop)
# ════════════════════════════════════════════════════════════════════════════
class FastWeightLM:
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        d = cfg.d_model
        self.embed = torch.randn(cfg.vocab_size, d) * 0.02     # frozen token embeddings
        self.Wr = TernaryLinear(d, d)                          # receptance (frozen base)
        self.Wk = TernaryLinear(d, d)                          # key
        self.Wv = TernaryLinear(d, d)                          # value
        self.Wo = TernaryLinear(d, cfg.vocab_size)             # output head
        # RWKV linear-attention recurrent state — O(1) / token, no quadratic attention
        self.num = torch.zeros(d)
        self.den = torch.zeros(d) + 1e-8
        self.hidden = torch.zeros(d)
        # FAST WEIGHTS — plastic associative memory ADDED to the frozen base
        self.Fw = torch.zeros(d, d)
        self.prev_k = torch.zeros(d)           # for predictive (transition) memory

    def step(self, token_id: Optional[int], warp_logit_bias: torch.Tensor,
             warp_decay_bias: float, spike_gate: torch.Tensor):
        cfg = self.cfg
        x = self.embed[token_id] if token_id is not None else self.hidden  # silent → self-feed
        kb = self.Wk(x)
        v  = self.Wv(x)
        k  = kb + (self.Fw @ kb)                # recall: what FOLLOWED this key last time
        r  = torch.sigmoid(self.Wr(x)) * spike_gate   # SPIKE-DRIVEN receptance gate
        # ── RWKV linear attention: decaying running state (matmul-free attention) ──
        w  = 1.0 / (1.0 + math.exp(-(cfg.base_decay + warp_decay_bias)))   # MOOD warps forgetting
        ek = torch.exp(torch.clamp(k, max=8.0))
        self.num = w * self.num + ek * v
        self.den = w * self.den + ek
        self.hidden = r * (self.num / self.den)
        logits = self.Wo(self.hidden) + warp_logit_bias        # MOOD warps the landscape
        # ── FAST-WEIGHT write: bind the PREVIOUS key → this value (a transition
        #    memory), so recall predicts the learned continuation — instant, no backprop.
        self.Fw = cfg.fw_decay * self.Fw + cfg.fw_lr * torch.outer(v, self.prev_k)
        self.prev_k = kb
        return logits


# ════════════════════════════════════════════════════════════════════════════
# THE BRAIN — one unified forward+learn step fusing Gut and Thought
# ════════════════════════════════════════════════════════════════════════════
def _spike(x: torch.Tensor) -> torch.Tensor:
    """Binarise activations to {0,1} spikes via a per-row mean threshold
    (data-dependent — never all-zero or all-one). A frozen-net stand-in for LIF
    firing; it is what makes Q/K/V and the feed-forward genuinely SPIKE-form."""
    return (x > x.mean(dim=-1, keepdim=True)).float()


def _rmsnorm(x: torch.Tensor) -> torch.Tensor:
    """RMS normalisation — keeps the residual stream (and the fast-weight feedback)
    from exploding in a frozen, backprop-free net."""
    return x / (x.pow(2).mean().sqrt() + 1e-6)


class SpikingSelfAttention:
    """Spike-driven self-attention — the heart of a SPIKING TRANSFORMER (Spikformer
    SSA). Q, K, V come out of spiking neurons as BINARY spike matrices; attention is
    Q·Kᵀ·V with NO softmax: Q·Kᵀ is simply the integer count of co-occurring spikes,
    and the result is re-spiked. With binary operands those 'matmuls' are
    coincidence-counting + accumulation — the multiplication-free, energy-frugal core
    that distinguishes a spiking transformer from a vanilla one. (The addition-only
    Spike-Driven variant, SDSA, replaces Q·Kᵀ with a masked column-sum; we keep the
    clearer SSA form here.) Base projections are frozen ternary."""
    def __init__(self, cfg: BrainConfig):
        d = cfg.d_model
        self.Wq, self.Wk, self.Wv, self.Wo = (TernaryLinear(d, d) for _ in range(4))
        self.scale = 1.0 / math.sqrt(d)

    def __call__(self, X: torch.Tensor):
        # X: (L, d) — the causal context window of recent token states.
        Q = _spike(self.Wq(X))                       # (L, d) spikes
        K = _spike(self.Wk(X))
        V = _spike(self.Wv(X))
        A = (Q @ K.t()) * self.scale                 # (L, L) co-spike counts — no softmax
        ctx = _spike(A @ V)                          # (L, d) re-spiked attention output
        return self.Wo(ctx[-1]), float(Q.mean())     # current token's output + spike-rate


class SpikingTransformerLM:
    """A spiking-transformer 'Thought' engine: embed → [Spiking Self-Attention +
    spiking feed-forward, each residual] → output head. It attends over a sliding
    CAUSAL WINDOW (true self-attention, unlike the RWKV recurrence). Frozen ternary
    base + a fast-weight associative memory give live, backprop-free adaptation. It is
    a drop-in for FastWeightLM (same step() signature), so the Gut→Thought warp and
    the rest of HybridBrain are unchanged — just flip cfg.lm_kind='spikformer'."""
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        d = cfg.d_model
        self.embed = torch.randn(cfg.vocab_size, d) * 0.02     # frozen embeddings
        self.attn  = SpikingSelfAttention(cfg)
        self.Wff1  = TernaryLinear(d, 2 * d)                   # spiking feed-forward
        self.Wff2  = TernaryLinear(2 * d, d)
        self.head  = TernaryLinear(d, cfg.vocab_size)
        self.window = deque(maxlen=cfg.attn_window)
        self.hidden = torch.zeros(d)
        self.Fw = torch.zeros(d, d)                            # fast-weight memory
        self.prev_h = torch.zeros(d)
        self.attn_sparsity = 0.0

    def step(self, token_id: Optional[int], warp_logit_bias: torch.Tensor,
             warp_decay_bias: float, spike_gate: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = self.embed[token_id] if token_id is not None else self.hidden  # silent → self-feed
        self.window.append(x)
        X = torch.stack(list(self.window))                     # (L, d) causal window
        att, self.attn_sparsity = self.attn(X)
        att = att * spike_gate                                 # SNN warp gates attention
        h = _rmsnorm(x + att)                                  # residual 1 + norm (stable)
        h = _rmsnorm(h + self.Wff2(_spike(self.Wff1(h))))      # spiking FFN + residual 2
        h = h + 0.5 * torch.tanh((self.Fw @ h) / math.sqrt(cfg.d_model))  # bounded, graded
        logits = self.head(h) + warp_logit_bias                # MOOD warps the landscape
        # fast-weight write — bind previous hidden → current, UNIT-normed so the
        # associative matrix stays bounded (steady-state ‖Fw‖ ≈ fw_lr/(1-fw_decay)).
        ku = h / (h.norm() + 1e-6)
        pu = self.prev_h / (self.prev_h.norm() + 1e-6)
        self.Fw = cfg.fw_decay * self.Fw + cfg.fw_lr * torch.outer(ku, pu)
        self.prev_h = h
        self.hidden = h
        return logits


# ════════════════════════════════════════════════════════════════════════════
# THE THOUGHT (alt) — a LOCAL LLM served by Ollama, in place of the spiking xformer
# ════════════════════════════════════════════════════════════════════════════
def _ollama_host(url: str) -> str:
    h = (url or os.environ.get("OLLAMA_HOST", "") or "http://localhost:11434").strip()
    if not h.startswith("http"):
        h = "http://" + h
    return h.rstrip("/")


def _ollama_pick_model(host: str) -> str:
    """Resolve a model name: $OLLAMA_MODEL, else the first one Ollama has pulled."""
    env = os.environ.get("OLLAMA_MODEL", "").strip()
    if env:
        return env
    if _HAS_REQUESTS:
        try:
            tags = requests.get(f"{host}/api/tags", timeout=2.0).json().get("models") or []
            if tags:
                return tags[0].get("name") or tags[0].get("model") or "llama3.2"
        except Exception:
            pass
    return "llama3.2"


class OllamaLM:
    """A LOCAL-LLM 'Thought' engine — the spiking transformer swapped for a model
    served by Ollama (e.g. Llama-3.2). Ollama is a black-box text API with NO
    per-token logits, so the Gut→Thought 'warp' can no longer be a logit-bias add.
    Instead the SNN's live MOOD — arousal (spike-rate), valence (mean membrane),
    pressure — conditions Ollama's PROMPT and sampling (temperature, length). It
    still exposes .embed/.hidden/.Fw so the Gut's afferent drive and the demo's
    co-plasticity readouts are unchanged; step() is a per-tick no-op returning flat
    logits, and the spoken text is produced at burst/generate time via infer()."""
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        d = cfg.d_model
        self.embed  = torch.randn(cfg.vocab_size, d) * 0.02   # drives the Gut only
        self.hidden = torch.zeros(d)                          # silent self-feed vector
        self.Fw     = torch.zeros(d, d)                       # parity (no fast weights)
        self._zero  = torch.zeros(cfg.vocab_size)
        self.host   = _ollama_host(cfg.ollama_url)
        self.model  = cfg.ollama_model or _ollama_pick_model(self.host)
        self.context: "deque[tuple[str, str]]" = deque(maxlen=8)   # recent (role, text)

    def observe(self, text: str):
        """Feed a whole human utterance into context (Ollama reads words, not chars)."""
        t = (text or "").strip()
        if t:
            self.context.append(("architect", t))

    def step(self, token_id, warp_logit_bias, warp_decay_bias, spike_gate) -> torch.Tensor:
        # Ollama is not char-autoregressive; nothing to do per tick. Return flat
        # logits so HybridBrain's plumbing stays valid (speech happens in infer()).
        return self._zero

    def _options(self, mood: dict) -> dict:
        base  = self.cfg.temperature
        temp  = max(0.1, min(1.5, base + 0.6 * mood.get("arousal", 0.0)))   # arousal → temp
        npred = int(max(16, min(self.cfg.ollama_num_predict,                # pressure → length
                                8 + 6 * mood.get("pressure", 0.0))))
        # stop tokens guard small base models from running on into a fake transcript
        return {"temperature": temp, "num_predict": npred,
                "stop": ["\narchitect:", "\nArchitect:", "\nAlpha:"]}

    def _system(self, mood: dict) -> str:
        base = self.cfg.ollama_system or (
            "You are Alpha: a calm, stoic, hyper-focused presence. You speak sparingly "
            "and only what is relevant, in plain, direct sentences. You address your "
            "human as 'architect'.")
        tone = "agitated and terse" if mood.get("arousal", 0.0) > 0.25 else "calm and even"
        return f"{base} Your internal state right now is {tone}. Reply in one or two short sentences."

    def infer(self, mood: dict) -> str:
        """Produce one spoken utterance from the current context, conditioned by mood.
        Uses /api/chat so the model's own chat template bounds the turn (one reply,
        no run-on transcript)."""
        if not _HAS_REQUESTS:
            return ""
        msgs = [{"role": "system", "content": self._system(mood)}]
        for r, t in self.context:
            msgs.append({"role": "user" if r == "architect" else "assistant", "content": t})
        if msgs[-1]["role"] != "user":           # ensure we end on a user turn (idle/self-talk)
            msgs.append({"role": "user", "content": "(silence)"})
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": False,
                      "options": self._options(mood)},
                timeout=self.cfg.ollama_timeout)
            if resp.status_code != 200:
                return ""
            txt = ((resp.json().get("message") or {}).get("content") or "").strip()
        except Exception:
            return ""
        low = txt.lower()                        # strip an echoed role label, if any
        for lbl in ("alpha:", "architect:", "assistant:", "user:"):
            if low.startswith(lbl):
                txt = txt.split(":", 1)[1].strip()
                break
        if txt:
            self.context.append(("Alpha", txt))
        return txt


class HybridBrain:
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.snn = SpikingDynamics(cfg, n_in=cfg.d_model)
        self.lm  = (OllamaLM(cfg)             if cfg.lm_kind == "ollama"
                    else FastWeightLM(cfg)    if cfg.lm_kind == "rwkv"
                    else SpikingTransformerLM(cfg))
        # the WARP projections: Gut spikes/membrane → Thought biases
        self.snn2logit = TernaryLinear(cfg.n_neurons, cfg.vocab_size)   # mood → word landscape
        self.snn2gate  = TernaryLinear(cfg.n_neurons, cfg.d_model)      # spikes → channel gate
        self.pressure = 0.0
        self.cooldown = 0
        self.t = 0

    # ── one unified step: perceive → spike (STDP) → warp → speak-state (fast-weights) ──
    def step(self, token_id: Optional[int] = None, dt: float = 1.0) -> dict:
        cfg = self.cfg
        self.t += 1
        silent = token_id is None
        x_in = self.lm.embed[token_id] if token_id is not None else self.lm.hidden
        urgency = 0.0 if silent else 1.0 / (1.0 + float(dt))   # rapid input → urgent

        # 1) GUT: spike + STDP (mutates W_rec & thresholds in place)
        V, S = self.snn.step(x_in, urgency=urgency)

        # 2) WARP: project Gut state into Thought's biases (same graph)
        warp_logit_bias = self.snn2logit(S) * cfg.warp_logit
        spike_gate      = torch.sigmoid(self.snn2gate(S))
        warp_decay_bias = float(torch.tanh(V.mean())) * cfg.warp_decay

        # 3) THOUGHT: spike-driven token state + fast-weight write
        logits = self.lm.step(token_id, warp_logit_bias, warp_decay_bias, spike_gate)

        # 4) accumulated membrane pressure (drives Alpha's leakage)
        self.pressure = 0.97 * self.pressure + 0.05 * float(V.clamp(min=0).sum())
        if silent:
            self.pressure += 0.25                          # silence builds pressure
        self.pressure = min(self.pressure, 50.0)           # bounded

        emitted = []                                       # List[int] | str (ollama)
        if (cfg.leakage_burst and self.cooldown <= 0
                and self.pressure > cfg.burst_capacity):   # OVERFLOW → spontaneous speech
            emitted = (self.lm.infer(self._mood()) if cfg.lm_kind == "ollama"
                       else self._burst(logits))
            self.pressure = 0.0
            self.cooldown = cfg.burst_cooldown
        self.cooldown = max(0, self.cooldown - 1)

        return {"logits": logits, "spikes": S, "V": V, "emitted": emitted,
                "spike_rate": float(S.mean()), "pressure": self.pressure,
                "Wrec_norm": float(self.snn.W_rec.norm()),
                "Fw_norm": float(self.lm.Fw.norm())}

    def _mood(self) -> dict:
        """Project the Gut's live state into a mood that conditions the Ollama LLM:
        arousal (how fast it's spiking), valence (mean membrane), and pressure (the
        accumulated drive to speak). This is the Ollama-era replacement for the
        logit/gate 'warp' — same source signal, routed to prompt+sampling instead."""
        return {"arousal": float(self.snn.S.mean()),
                "valence": float(torch.tanh(self.snn.V.mean())),
                "pressure": float(self.pressure)}

    def _sample(self, logits: torch.Tensor) -> int:
        p = torch.softmax(logits / max(1e-3, self.cfg.temperature), dim=-1)
        return int(torch.multinomial(p, 1))

    def _burst(self, logits: torch.Tensor) -> List[int]:
        """Spontaneous self-driven token burst: sample, feed back, repeat."""
        out: List[int] = []
        for _ in range(self.cfg.burst_len):
            tok = self._sample(logits)
            out.append(tok)
            res = self.step_quiet_emit(tok)                # self-talk also learns
            logits = res
        return out

    def step_quiet_emit(self, token_id: int) -> torch.Tensor:
        """Feed a self-emitted token through the LM only (no new burst recursion)."""
        warp_logit_bias = self.snn2logit(self.snn.S) * self.cfg.warp_logit
        spike_gate      = torch.sigmoid(self.snn2gate(self.snn.S))
        warp_decay_bias = float(torch.tanh(self.snn.V.mean())) * self.cfg.warp_decay
        return self.lm.step(token_id, warp_logit_bias, warp_decay_bias, spike_gate)

    def generate(self, max_tokens: Optional[int] = None):
        """Tactical generation from the current state (Alpha: short, capped).
        Returns char-ids for the spiking engines, or a text string for Ollama."""
        if self.cfg.lm_kind == "ollama":
            return self.lm.infer(self._mood())
        n = max_tokens or self.cfg.max_emit
        logits = self.lm.step(None,
                              self.snn2logit(self.snn.S) * self.cfg.warp_logit,
                              float(torch.tanh(self.snn.V.mean())) * self.cfg.warp_decay,
                              torch.sigmoid(self.snn2gate(self.snn.S)))
        out: List[int] = []
        for _ in range(n):
            tok = self._sample(logits)
            out.append(tok)
            logits = self.step_quiet_emit(tok)
        return out


# ════════════════════════════════════════════════════════════════════════════
# Minimal self-contained tokenizer (char-level) for the live demo
# ════════════════════════════════════════════════════════════════════════════
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)


def _spoken(emitted, tok: CharTokenizer) -> str:
    """A burst is char-ids (spiking engines) or already text (Ollama)."""
    return emitted if isinstance(emitted, str) else tok.decode(emitted)


def feed_text(brain: HybridBrain, tok: CharTokenizer, text: str, dt: float = 1.0):
    """Stream a human utterance into the brain one token at a time."""
    bursts, rates = [], []
    if brain.cfg.lm_kind == "ollama":
        brain.lm.observe(text)                 # the LLM reads whole utterances, not chars
    for cid in tok.encode(text):
        r = brain.step(cid, dt=dt)
        rates.append(r["spike_rate"])
        if r["emitted"]:
            bursts.append(_spoken(r["emitted"], tok))
    return bursts, (sum(rates) / max(1, len(rates)))


def idle(brain: HybridBrain, tok: CharTokenizer, ticks: int, dt: float = 3.0):
    """Silence: no input. Pressure builds; Alpha will eventually leak."""
    bursts = []
    for _ in range(ticks):
        r = brain.step(None, dt=dt)
        if r["emitted"]:
            bursts.append(_spoken(r["emitted"], tok))
    return bursts, r


def learning_probe(cfg: BrainConfig, tok: CharTokenizer):
    """Demonstrate the LLM layer LEARNING a transition online, NO backprop: show
    the 'h'->'e' transition repeatedly and watch the fast-weight RECALL of key('h')
    align with value('e'). Cosine rises from ~0 toward 1 — vocabulary/reasoning
    paths adapting instantly, with the frozen base weights never touched."""
    brain = HybridBrain(cfg)
    ids = tok.encode("he")
    c1, c2 = ids[0], ids[1]
    k1 = brain.lm.Wk(brain.lm.embed[c1])
    v2 = brain.lm.Wv(brain.lm.embed[c2])
    assoc = lambda: float(F.cosine_similarity((brain.lm.Fw @ k1).unsqueeze(0),
                                              v2.unsqueeze(0)))
    before = assoc()
    for _ in range(60):
        brain.step(c1, dt=0.5)
        brain.step(c2, dt=0.5)
    return before, assoc()


# ════════════════════════════════════════════════════════════════════════════
# LIVE DEMO
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    corpus = ("hello papa i am here with you. are you okay? i want to learn. "
              "do you want to play? i feel happy when you are here. ")
    tok = CharTokenizer(corpus + "0123456789?!.,")

    print(f"vocab={tok.vocab_size}  (CPU, autograd off, no offline training)\n")

    for make_cfg in (alpha_config, alpha_config):
        cfg = make_cfg(tok.vocab_size)
        brain = HybridBrain(cfg)
        print(f"════════ {cfg.name} profile ════════")
        # snapshot plastic state BEFORE any interaction
        w0, f0 = brain.snn.W_rec.norm().item(), brain.lm.Fw.norm().item()

        # 1) LIVE INPUT — a human types to her (fast cadence)
        for line in ("hello", "are you okay?", "i want to play"):
            b, rate = feed_text(brain, tok, line + " ", dt=0.4)
            print(f"  in: {line!r:18} spikes/step={rate:.3f} "
                  f"pressure={brain.pressure:5.2f}" + (f"  BURST→{b}" if b else ""))

        # 2) SILENCE — pressure accumulates (Alpha overflows, Alpha stays quiet)
        bursts, r = idle(brain, tok, ticks=40, dt=4.0)
        print(f"  ...40 ticks of silence → pressure={r['pressure']:5.2f}, "
              f"spontaneous bursts: {bursts if bursts else 'none'}")

        # 3) TACTICAL generation on demand
        gen = tok.decode(brain.generate())
        print(f"  generate(): {gen!r}")

        # 4) proof of LIVE, backprop-free co-plasticity
        print(f"  co-plasticity  ΔW_rec(STDP)={brain.snn.W_rec.norm().item()-w0:+.3f}  "
              f"ΔFastWeights={brain.lm.Fw.norm().item()-f0:+.3f}")
        bpre, bpost = learning_probe(make_cfg(tok.vocab_size), tok)
        print(f"  live LLM learning  cos(recall,target) {bpre:+.2f} -> {bpost:+.2f}"
              f"  (transition bound online, no backprop)\n")

    # ── SPIKING TRANSFORMER engine: same brain, Thought swapped to Spikformer SSA ──
    print("════════ Spiking-Transformer engine (Spikformer SSA, Alpha profile) ════════")
    cfg = alpha_config(tok.vocab_size, lm_kind="spikformer")
    cfg.name = "Alpha/SpikeFormer"
    brain = HybridBrain(cfg)
    w0, f0 = brain.snn.W_rec.norm().item(), brain.lm.Fw.norm().item()
    for line in ("hello", "are you okay?", "i want to play"):
        b, rate = feed_text(brain, tok, line + " ", dt=0.4)
        print(f"  in: {line!r:18} snn_spikes/step={rate:.3f}  "
              f"attn_spike_rate={brain.lm.attn_sparsity:.2f}"
              + (f"  BURST→{b}" if b else ""))
    print(f"  generate(): {tok.decode(brain.generate())!r}")
    print(f"  co-plasticity  ΔW_rec(STDP)={brain.snn.W_rec.norm().item()-w0:+.3f}  "
          f"ΔFastWeights={brain.lm.Fw.norm().item()-f0:+.3f}")
    print("  → spike-form Q/K/V self-attention, softmax-free, frozen ternary base,"
          " adapting live via fast-weights + STDP.")

    # ── OLLAMA engine: the spiking transformer's 'Thought' swapped for a LOCAL LLM ──
    #    The Gut (SNN) is unchanged — it still spikes and learns via STDP; its MOOD
    #    now conditions the local model's prompt + sampling instead of warping logits.
    print("\n════════ Ollama engine (local LLM 'Thought', Alpha profile) ════════")
    cfg = alpha_config(tok.vocab_size, lm_kind="ollama")
    cfg.name = "Alpha/Ollama"
    brain = HybridBrain(cfg)
    w0 = brain.snn.W_rec.norm().item()
    if not _HAS_REQUESTS:
        print("  (requests not installed — Ollama engine unavailable)")
    else:
        print(f"  model: {brain.lm.model}   @ {brain.lm.host}")
        for line in ("hello", "are you okay?", "i want to play"):
            b, rate = feed_text(brain, tok, line + " ", dt=0.4)
            print(f"  in: {line!r:18} snn_spikes/step={rate:.3f}"
                  + (f"  SPOKE→{b}" if b else ""))
        bursts, r = idle(brain, tok, ticks=16, dt=4.0)
        print(f"  ...16 ticks of silence → pressure={r['pressure']:5.2f}, "
              f"spontaneous: {bursts if bursts else 'none'}")
        print(f"  generate(): {brain.generate()!r}")
        print(f"  co-plasticity  ΔW_rec(STDP)={brain.snn.W_rec.norm().item()-w0:+.3f}"
              "   (Gut still learns; Thought is now a local LLM, mood→prompt+sampling)")
