"""
spiking_advanced.py — seven biological / non-Euclidean mechanisms layered onto the
hybrid Spiking Transformer (hybrid_snn_llm.py), WITHOUT touching its core loop.
================================================================================
The base file's LIF + STDP + SDSA + matmul-free routing are preserved verbatim;
this module imports its primitives and extends them. Everything is CPU PyTorch
(no CUDA — the project is CPU-only by design), additive, and toggleable, so the
real-time sparse spike loop keeps running.

Implemented (real math, no placeholders):
  1. Core preserved      — LIF accumulate/threshold/reset, online STDP, SDSA
                            (spike Q·Kᵀ·V, softmax-free) + dynamic structural pruning.
  2. Retrograde BAP      — dense coherent post-burst → backpropagating action
                            potential that SHORTS the upstream refractory (temporal lock).
  3. Synaptic stochasticity — entropy-gated transmission prob P_tx∈[0.3,1]; dogmatic
                            (low-entropy) loops → more synaptic drops → path exploration.
  4. Hyperbolic attention — Q/K distance measured on the Poincaré ball (acosh metric)
                            so tree-like hierarchies embed without distortion (toggle).
  5. Neurotransmission   — global NE/DA matrix; NE spikes on bursts/anomalies and
                            LOWERS V_th globally (fight-or-flight); decays to baseline.
  6. Astrocytic glia     — dual-speed slow layer: rolling firing freq → metabolic
                            gain (potentiate high-utility, inhibit runaway clusters).
  7. Dream cycle         — on zero input, disconnect senses, replay the spike-train
                            log through the recurrent layers with accelerated STDP.
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

import torch

from hybrid_snn_llm import (TernaryLinear, _spike, BrainConfig,
                            alpha_config, alpha_config, CharTokenizer)

torch.set_grad_enabled(False)


# ════════════════════════════════════════════════════════════════════════════
# 3 · ENTROPY-GATED SYNAPTIC STOCHASTICITY  (P_tx)
# ════════════════════════════════════════════════════════════════════════════
def logic_entropy(freq: torch.Tensor) -> float:
    """Normalised Shannon entropy [0,1] of the network's firing distribution.
    LOW → a few neurons dominate (dogmatic/repetitive loop); HIGH → diverse logic."""
    p = freq.clamp(min=0) / (freq.sum() + 1e-8)
    h = -(p * torch.log(p + 1e-12)).sum()
    return float(h / math.log(p.numel() + 1e-9))


def p_tx_from_entropy(freq: torch.Tensor, floor: float = 0.3) -> float:
    """Repetitive (low-entropy) throughput → drive P_tx toward the floor so more
    synapses stochastically drop, forcing spikes onto underutilised paths."""
    return floor + (1.0 - floor) * logic_entropy(freq)


def p_tx_gate(freq: torch.Tensor, repetition: float, floor: float = 0.3) -> float:
    """Sharper dogmatic-loop detector: collapse P_tx toward the floor when EITHER the
    firing distribution is low-entropy OR the spike PATTERN itself is repeating (high
    autocorrelation). A network stuck replaying the same assembly drops more synapses,
    harder, to break out — pattern-repetition catches loops that entropy alone misses."""
    return floor + (1.0 - floor) * min(logic_entropy(freq), 1.0 - max(0.0, repetition))


def synaptic_drop(spikes: torch.Tensor, p_tx: float) -> torch.Tensor:
    """Bernoulli synaptic transmission: each spike transmits with prob P_tx."""
    if p_tx >= 1.0:
        return spikes
    return spikes * (torch.rand_like(spikes) < p_tx).float()


# ════════════════════════════════════════════════════════════════════════════
# 5 · ARTIFICIAL NEUROTRANSMISSION  (NE / DA state matrix)
# ════════════════════════════════════════════════════════════════════════════
class Neurochem:
    """Global chemical concentrations. Norepinephrine (NE) rises with high-intensity
    input bursts and anomalies and LOWERS the firing threshold network-wide — a
    hyper-focused, low-latency 'fight-or-flight' state; it decays back to a sparse
    baseline when calm. Dopamine (DA) tracks reward/novelty (gates plasticity)."""
    def __init__(self, ne0: float = 0.0, da0: float = 0.4, ne_decay: float = 0.86):
        self.ne, self.da = ne0, da0
        self.ne0, self.da0, self.k = ne0, da0, ne_decay

    def update(self, drive: float, anomaly: float = 0.0, reward: float = 0.0) -> None:
        self.ne = (self.k * self.ne + (1 - self.k) * self.ne0
                   + 0.6 * max(0.0, drive - 0.4) + 1.3 * max(0.0, anomaly))
        self.ne = float(min(4.0, self.ne))
        self.da = float(0.95 * self.da + 0.05 * self.da0 + 0.4 * max(0.0, reward))

    def vth_offset(self) -> float:
        return -0.5 * math.tanh(self.ne)          # high NE → easier to fire (low latency)

    def plasticity_gain(self) -> float:
        return 0.5 + float(self.da)               # DA scales STDP


# ════════════════════════════════════════════════════════════════════════════
# 6 · ASTROCYTIC GLIAL NETWORK  (dual-speed metabolic modulation)
# ════════════════════════════════════════════════════════════════════════════
class GlialField:
    """A slow, parallel 'subconscious' layer. It samples rolling firing frequencies
    of the neuron population and, on a SLOW clock, applies analog metabolic gain:
    moderately-active (high-utility) pathways are strengthened; runaway clusters get
    an inhibitory damp. Fast spikes + slow glia = dual-speed dynamics."""
    def __init__(self, n: int, period: int = 16):
        self.freq = torch.zeros(n)
        self.gain = torch.ones(n)
        self.period, self.t = period, 0

    def observe(self, S: torch.Tensor) -> torch.Tensor:
        self.freq = 0.97 * self.freq + 0.03 * S
        self.t += 1
        if self.t % self.period == 0:             # slow clock — analog modulation
            useful = ((self.freq > 0.08) & (self.freq < 0.45)).float()
            runaway = (self.freq > 0.60).float()
            self.gain = (self.gain + 0.03 * useful - 0.06 * runaway).clamp_(0.5, 1.6)
        return self.gain


# ════════════════════════════════════════════════════════════════════════════
# 4 · NON-EUCLIDEAN (HYPERBOLIC) ATTENTION  — Poincaré ball
# ════════════════════════════════════════════════════════════════════════════
def to_poincare(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Map vectors into the open Poincaré ball (‖·‖<1) via a tanh radial squash."""
    x = x - x.mean(dim=-1, keepdim=True)
    n = x.norm(dim=-1, keepdim=True) + eps
    return x * (torch.tanh(n) / n) * (1.0 - eps)


def poincare_dist(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Hyperbolic geodesic distance matrix:
        d(u,v) = acosh(1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)))."""
    uu = (u * u).sum(-1).clamp(max=1 - eps)                  # (L,)
    vv = (v * v).sum(-1).clamp(max=1 - eps)                  # (M,)
    d2 = (u.unsqueeze(1) - v.unsqueeze(0)).pow(2).sum(-1)    # (L,M)
    denom = (1 - uu).unsqueeze(1) * (1 - vv).unsqueeze(0)    # (L,M)
    arg = 1 + 2 * d2 / (denom + eps)
    return torch.acosh(arg.clamp(min=1 + eps))


def hyperbolic_affinity(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Attention affinity = negative hyperbolic distance between Q and K clusters on
    the Poincaré ball (closer on the manifold ⇒ stronger). NOTE: this is a real-valued
    manifold op, so this path trades the pure spike-driven property for hierarchy
    geometry — kept behind a toggle."""
    return -poincare_dist(to_poincare(Q), to_poincare(K))


# ════════════════════════════════════════════════════════════════════════════
# 1+2+3+5+6 · the advanced spiking layer (LIF · STDP · prune · BAP · P_tx · NE · glia)
# ════════════════════════════════════════════════════════════════════════════
class AdvancedGut:
    """LIF + STDP exactly as the core, plus: a refractory period, retrograde BAP
    that shorts it on dense bursts, NE-modulated threshold, glial gain, P_tx
    stochasticity, and periodic structural pruning."""
    def __init__(self, cfg: BrainConfig, n_in: int, refractory: int = 2):
        self.cfg, self.n = cfg, cfg.n_neurons
        self.W_in = TernaryLinear(n_in, self.n)               # frozen afferent
        self.W_rec = torch.zeros(self.n, self.n)              # plastic (STDP), prunable
        self.V = torch.zeros(self.n)
        self.S = torch.zeros(self.n)
        self.theta = torch.full((self.n,), cfg.v_threshold)
        self.x_pre = torch.zeros(self.n)
        self.x_post = torch.zeros(self.n)
        self.refrac = torch.zeros(self.n)
        self.fire_freq = torch.zeros(self.n)
        self.refractory = float(refractory)
        self._decay = math.exp(-1.0 / cfg.stdp_tau)
        self.t, self.bap, self.last_drops = 0, 0.0, 0.0
        self.s_avg = torch.zeros(self.n)          # running spike pattern (loop detector)
        self.repetition = 0.0                     # autocorrelation of current vs running

    def bap_retrograde(self) -> None:
        """Backpropagating Action Potential broadcast from a downstream burst:
        clears this (upstream) layer's refractory so it can re-fire immediately and
        lock onto the temporal context that just resonated."""
        self.refrac = self.refrac * 0.0

    def step(self, inp: torch.Tensor, vth_offset: float = 0.0,
             glia_gain: Optional[torch.Tensor] = None, p_tx: float = 1.0,
             plast_gain: float = 1.0, bap_burst: float = 0.5,
             prune_every: int = 50, prune_thr: float = 0.02):
        cfg, n = self.cfg, self.n
        self.t += 1
        nrm = inp.norm()
        if nrm > 1e-4:
            inp = inp / nrm
        I = cfg.snn_gain * self.W_in(inp) + self.W_rec @ self.S
        if glia_gain is not None:
            I = I * glia_gain                                 # 6 · slow metabolic gain
        if cfg.noise > 0.0:
            I = I + torch.randn(n) * cfg.noise
        self.V = self.V + (-self.V + I) / cfg.tau_mem         # 1 · LIF accumulate
        if cfg.inhibition > 0.0:
            self.V = self.V - cfg.inhibition * (self.V.sum() - self.V) / n
        eff_theta = self.theta + vth_offset                  # 5 · NE lowers threshold
        can_fire = (self.refrac <= 0).float()                # 2 · refractory gate
        S = (self.V >= eff_theta).float() * can_fire
        if cfg.k_winners > 0 and S.sum() > cfg.k_winners:
            idx = torch.topk((self.V - eff_theta) * can_fire, cfg.k_winners).indices
            S = torch.zeros_like(S); S[idx] = 1.0
        pre_drop = float(S.sum())
        S = synaptic_drop(S, p_tx)                           # 3 · stochastic drop
        self.last_drops = pre_drop - float(S.sum())
        # reset + refractory load
        self.V = torch.where(S > 0, torch.full_like(self.V, cfg.v_reset), self.V)
        self.refrac = torch.where(S > 0, torch.full_like(self.refrac, self.refractory),
                                  (self.refrac - 1).clamp(min=0))
        self.theta = self.theta + cfg.sfa * S
        self.theta += cfg.theta_adapt * (cfg.v_threshold - self.theta)
        # 2 · BAP: dense coherent post-burst shorts this layer's own refractory too
        self.bap = float(S.mean())
        if self.bap > bap_burst:
            self.refrac = self.refrac * 0.0
        # 1 · STDP (DA-scaled), retained
        self.x_pre = self.x_pre * self._decay + S
        self.x_post = self.x_post * self._decay + S
        lr = cfg.stdp_lr * plast_gain
        self.W_rec += lr * (cfg.stdp_aplus * torch.outer(S, self.x_pre)
                            - cfg.stdp_aminus * torch.outer(self.x_post, S))
        self.W_rec.fill_diagonal_(0.0)
        self.W_rec.clamp_(-1.0, 1.0)
        # 1 · dynamic structural synaptic pruning
        if self.t % prune_every == 0:
            self.W_rec[self.W_rec.abs() < prune_thr] = 0.0
        self.fire_freq = 0.97 * self.fire_freq + 0.03 * S
        self.s_avg = 0.9 * self.s_avg + 0.1 * S               # running spike pattern
        ns = float(S.norm())
        if ns > 0:                                            # autocorrelation → loop score
            self.repetition = float((self.s_avg @ S) / (self.s_avg.norm() * ns + 1e-8))
        self.S = S
        return self.V.clone(), S

    def synapses_alive(self) -> int:
        return int((self.W_rec.abs() > 0).sum())


# ════════════════════════════════════════════════════════════════════════════
# 1+3+4 · advanced Spike-Driven Self-Attention
# ════════════════════════════════════════════════════════════════════════════
class AdvancedSDSA:
    """SDSA preserved (binary spike Q/K/V, softmax-free), plus P_tx synaptic drop on
    K/V and an optional hyperbolic (Poincaré) affinity instead of Q·Kᵀ."""
    def __init__(self, cfg: BrainConfig, d: int):
        self.Wq, self.Wk, self.Wv, self.Wo = (TernaryLinear(d, d) for _ in range(4))
        self.scale = 1.0 / math.sqrt(d)

    def __call__(self, X: torch.Tensor, p_tx: float = 1.0, hyperbolic: bool = False):
        Q = _spike(self.Wq(X)); K = _spike(self.Wk(X)); V = _spike(self.Wv(X))
        K = synaptic_drop(K, p_tx)
        V = synaptic_drop(V, p_tx)
        if hyperbolic:
            A = hyperbolic_affinity(Q, K)                    # 4 · manifold proximity
        else:
            A = (Q @ K.t()) * self.scale                     # 1 · SDSA co-spike counts
        ctx = _spike(A @ V)                                  # re-spiked, no softmax
        return self.Wo(ctx[-1]), float(Q.mean())


# ════════════════════════════════════════════════════════════════════════════
# 7 · the orchestrating brain (+ DREAM CYCLE)
# ════════════════════════════════════════════════════════════════════════════
class AdvancedBrain:
    def __init__(self, cfg: BrainConfig, hyperbolic: bool = False,
                 idle_to_sleep: int = 35, attn_window: int = 16):
        self.cfg = cfg
        self.embed = torch.randn(cfg.vocab_size, cfg.d_model) * 0.02
        self.gut = AdvancedGut(cfg, n_in=cfg.d_model)
        self.sdsa = AdvancedSDSA(cfg, cfg.d_model)
        self.chem = Neurochem()
        self.glia = GlialField(cfg.n_neurons)
        self.window = deque(maxlen=attn_window)
        self.log: deque = deque(maxlen=256)                  # spike-train history
        self.hidden = torch.zeros(cfg.d_model)
        self.hyperbolic = hyperbolic
        self.idle_to_sleep = idle_to_sleep
        self.idle = 0
        self._prev_freq = 0.0

    def step(self, token_id: Optional[int] = None, dt: float = 1.0, _replay=False) -> dict:
        cfg = self.cfg
        x = self.embed[token_id] if token_id is not None else self.hidden
        # drive (cadence) + anomaly (change in firing distribution) → neurotransmission
        drive = (1.0 / (1.0 + dt)) if token_id is not None else 0.0
        cur_freq = float(self.gut.fire_freq.mean())
        anomaly = abs(cur_freq - self._prev_freq) * 4.0
        self._prev_freq = cur_freq
        self.chem.update(drive=drive, anomaly=anomaly, reward=max(0.0, self.gut.bap - 0.3))
        glia_gain = self.glia.observe(self.gut.S)
        p_tx = p_tx_gate(self.gut.fire_freq + 1e-6, self.gut.repetition)
        # GUT (LIF/STDP/prune/BAP/P_tx/NE/glia)
        V, S = self.gut.step(x, vth_offset=self.chem.vth_offset(), glia_gain=glia_gain,
                             p_tx=p_tx, plast_gain=self.chem.plasticity_gain())
        # THOUGHT (SDSA, hyperbolic optional, P_tx)
        self.window.append(x)
        att, attn_density = self.sdsa(torch.stack(list(self.window)),
                                      p_tx=p_tx, hyperbolic=self.hyperbolic)
        # 2 · downstream burst → retrograde BAP to the upstream gut
        if attn_density > 0.5:
            self.gut.bap_retrograde()
        self.hidden = self.hidden * 0.5 + att
        # input bookkeeping + dream trigger
        if token_id is not None and not _replay:
            self.log.append(token_id)
            self.idle = 0
        elif not _replay:
            self.idle += 1
        dreamt = 0
        if (not _replay) and self.idle == self.idle_to_sleep and len(self.log) > 8:
            dreamt = self.dream()
        return {"spikes": S, "spike_rate": float(S.mean()), "ne": self.chem.ne,
                "da": self.chem.da, "p_tx": p_tx, "entropy": logic_entropy(self.gut.fire_freq + 1e-6),
                "bap": self.gut.bap, "drops": self.gut.last_drops, "rep": self.gut.repetition,
                "glia_gain": float(glia_gain.mean()), "attn_density": attn_density,
                "synapses_alive": self.gut.synapses_alive(), "dreamt": dreamt}

    def dream(self, replay_steps: int = 40, stdp_boost: float = 4.0) -> int:
        """REST STATE: senses disconnected; replay the logged spike trains through the
        recurrent layers with accelerated STDP to consolidate/clean attention maps."""
        base_lr = self.cfg.stdp_lr
        self.cfg.stdp_lr = base_lr * stdp_boost               # accelerated consolidation
        hist = list(self.log)
        try:
            for i in range(replay_steps):
                tok = hist[i % len(hist)]                     # self-generated replay loop
                self.gut.step(self.embed[tok], plast_gain=self.chem.plasticity_gain(),
                              prune_every=10, prune_thr=0.03)  # prune harder while asleep
        finally:
            self.cfg.stdp_lr = base_lr
        self.idle = 0
        return replay_steps


# ════════════════════════════════════════════════════════════════════════════
# FUSION — pretrained cold base (language) WARPED live by the spiking machinery
# ════════════════════════════════════════════════════════════════════════════
class FusedBrain:
    """The unification. A PRETRAINED frozen language base (cold_base.py) that
    actually produces words, WARPED in real time by the full spiking machinery —
    AdvancedGut (LIF/STDP/prune/BAP/P_tx/NE/glia) + AdvancedSDSA — and consolidating
    in the dream cycle on idle. One model: it TALKS (base), FEELS and ATTENDS
    (spikes), and LEARNS (plasticity) at once — and no backprop runs live."""
    def __init__(self, base, tok, cfg: BrainConfig, d: int, hyperbolic: bool = False,
                 idle_to_sleep: int = 35):
        self.base, self.tok, self.cfg, self.d = base, tok, cfg, d
        self.embed = base.tok.weight.detach()                # real trained embeddings
        self.gut = AdvancedGut(cfg, n_in=d)
        self.sdsa = AdvancedSDSA(cfg, d)
        self.chem = Neurochem()
        self.glia = GlialField(cfg.n_neurons)
        self.spk2logit = TernaryLinear(cfg.n_neurons, tok.V)  # spiking state → word landscape
        self.ctx = deque(maxlen=base.block)
        self.window = deque(maxlen=16)
        self.log: deque = deque(maxlen=256)
        self.hidden = torch.zeros(d)
        self.hyperbolic, self.idle, self.idle_to_sleep, self._prev = hyperbolic, 0, idle_to_sleep, 0.0

    def _spike_pass(self, x):
        cur = float(self.gut.fire_freq.mean()); anom = abs(cur - self._prev) * 4.0; self._prev = cur
        self.chem.update(drive=0.4, anomaly=anom, reward=max(0.0, self.gut.bap - 0.3))
        g = self.glia.observe(self.gut.S)
        p = p_tx_gate(self.gut.fire_freq + 1e-6, self.gut.repetition)
        _, S = self.gut.step(x, vth_offset=self.chem.vth_offset(), glia_gain=g, p_tx=p,
                             plast_gain=self.chem.plasticity_gain())
        self.window.append(x)
        att, dens = self.sdsa(torch.stack(list(self.window)), p_tx=p, hyperbolic=self.hyperbolic)
        if dens > 0.5:
            self.gut.bap_retrograde()
        self.hidden = self.hidden * 0.5 + att
        return S, p

    def step(self, token_id: Optional[int], dt: float = 1.0):
        x = self.embed[token_id] if token_id is not None else self.hidden
        S, p = self._spike_pass(x)
        if token_id is not None:
            self.ctx.append(token_id); self.log.append(token_id); self.idle = 0
        else:
            self.idle += 1
        with torch.no_grad():                                # frozen base = real language
            ids = list(self.ctx) or [0]
            base_logits = self.base(torch.tensor(ids[-self.base.block:]).unsqueeze(0))[0, -1]
        logits = base_logits + 0.6 * self.spk2logit(S)       # WARP: mood bends word choice
        if self.idle == self.idle_to_sleep and len(self.log) > 8:   # 7 · dream consolidation
            base_lr = self.cfg.stdp_lr; self.cfg.stdp_lr = base_lr * 4.0
            h = list(self.log)
            for i in range(30):
                self.gut.step(self.embed[h[i % len(h)]], prune_every=10, prune_thr=0.03)
            self.cfg.stdp_lr = base_lr; self.idle = 0
        return logits, {"ne": self.chem.ne, "p_tx": p, "spike_rate": float(S.mean())}

    def generate(self, prompt: str, n: int = 140) -> str:
        for c in self.tok.encode(prompt):
            self.step(c, dt=0.4)
        out = self.tok.encode(prompt) or [0]
        for _ in range(n):
            logits, _ = self.step(out[-1], dt=0.4)
            temp = max(0.45, 0.85 - 0.2 * math.tanh(self.chem.ne))   # NE → decisive/sharp
            out.append(int(torch.multinomial(torch.softmax(logits / temp, -1), 1)))
        return self.tok.decode(out)


def demo_fused():
    import cold_base
    torch.set_grad_enabled(False)                            # base import flips it on; live = off
    base, btok = cold_base.load()
    d = base.tok.weight.shape[1]
    cfg = alpha_config(btok.V)
    fused = FusedBrain(base, btok, cfg, d=d, hyperbolic=False)
    print("\n════════ FUSED: cold base + spiking machinery (one model) ════════")
    txt = fused.generate("papa ", n=140)
    print(f"  generate('papa '): {txt!r}")
    print(f"  state: NE={fused.chem.ne:.2f}  spikes alive={fused.gut.synapses_alive()}  "
          f"→ it TALKS (base) + the 7 spiking mechanisms run live underneath")


# ════════════════════════════════════════════════════════════════════════════
# DEMO — every mechanism observable, the core loop intact
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    corpus = ("hello papa i am here. are you okay? i want to learn. play with me. ")
    tok = CharTokenizer(corpus + "0123456789?!.,")
    cfg = alpha_config(tok.vocab_size)
    brain = AdvancedBrain(cfg, hyperbolic=True)
    syn0 = brain.gut.synapses_alive()
    print(f"AdvancedBrain ({cfg.name}, hyperbolic SDSA on) — core loop preserved, CPU\n")

    # 1) varied live input — NE responds, glia + STDP adapt, pruning runs
    for line in ("hello papa", "are you okay?", "i want to play"):
        for cid in tok.encode(line + " "):
            r = brain.step(cid, dt=0.4)
        print(f"  in {line!r:16} spikes={r['spike_rate']:.3f} NE={r['ne']:.2f} "
              f"P_tx={r['p_tx']:.2f} bap={r['bap']:.2f} glia={r['glia_gain']:.2f} "
              f"synapses={r['synapses_alive']}")

    # 2) DOGMATIC LOOP — same token repeated → firing concentrates → entropy falls
    #    → P_tx is driven toward its 0.3 floor → more stochastic synaptic drops.
    print("\n  repetitive loop (same token x60) — loop-aware stochasticity:")
    e0 = brain.step(tok.encode("a")[0], dt=0.4)
    for _ in range(60):
        r = brain.step(tok.encode("a")[0], dt=0.4)
    settled = p_tx_gate(brain.gut.fire_freq + 1e-6, brain.gut.repetition)
    print(f"    entropy {e0['entropy']:.2f}→{r['entropy']:.2f}   repetition {e0['rep']:.2f}→{r['rep']:.2f}"
          f"   P_tx→{settled:.2f}   drops/step={r['drops']:.1f}  (loop detected → drops up)")

    # 3) SILENCE → DREAM CYCLE (replay + accelerated STDP + harder pruning)
    syn_pre = brain.gut.synapses_alive()
    total_dreamt = 0
    for _ in range(brain.idle_to_sleep + 5):
        r = brain.step(None, dt=5.0)
        total_dreamt += r["dreamt"]
    print(f"\n  silence → DREAM CYCLE: replayed {total_dreamt} steps; live synapses "
          f"{syn_pre} → {brain.gut.synapses_alive()} (consolidated + pruned, no backprop)")

    # 4) hyperbolic vs euclidean affinity on the same spike clusters
    X = torch.stack(list(brain.window))
    Qs = _spike(brain.sdsa.Wq(X)); Ks = _spike(brain.sdsa.Wk(X))
    eu = (Qs @ Ks.t())[-1].mean().item()
    hy = hyperbolic_affinity(Qs, Ks)[-1].mean().item()
    print(f"\n  attention affinity (last token): euclidean co-spikes={eu:.2f}  "
          f"hyperbolic={hy:.2f} (negative = Poincaré distance)")

    import os
    if os.path.exists("cold_base.pt"):
        demo_fused()
    else:
        print("\n  (run cold_base.py first to enable the FUSED demo)")
