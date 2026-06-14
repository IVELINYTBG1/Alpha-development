"""
phase2.py — autonomous deliberation + dynamic structural adaptability for the
spiking transformer. Builds on spiking_advanced.py / hybrid_snn_llm.py WITHOUT
touching their core loop: imports the primitives and adds three modules.

  1. AUTONOMOUS DELIBERATION (inner monologue) — on a task, run an internal loop
     for N cycles with: Gaussian noise current per step (spontaneous baseline),
     per-channel delayed echo of the output back into the input (resonance, τ
     matrix), and strict HOMEOSTATIC plasticity (V_th tracks a moving average of
     firing rate → self-organised criticality, no runaway/epileptic loops).
  2. DYNAMIC TOPOLOGY (List-style auto-expansion) — Node Birth: when a cluster
     stays saturated (V_th breaches) for consecutive windows without resolving,
     allocate & APPEND new LIF neurons into the (growing) tensors, initialised to
     integrate without corrupting established representations.
  3. AXONAL GROWTH CONES (hyperbolic routing) — every node has a Poincaré-ball
     coordinate; synchronously-firing nodes emit an attractor; underutilised nodes
     migrate toward it and SPAWN new synapses to nodes that become near on the
     manifold — routing physically around dead-ends.

Nova: dense, precise, low-entropy growth. Simona: volatile, high-stochasticity.
CPU PyTorch, binary spikes, sub-second. No backprop in the live loop.
"""
from __future__ import annotations

import glob
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

import torch

from spiking_advanced import poincare_dist, to_poincare   # reuse the hyperbolic geometry

torch.set_grad_enabled(False)


# ════════════════════════════════════════════════════════════════════════════
@dataclass
class P2Config:
    d:           int = 32        # fixed I/O width (task + echo + readout)
    n0:          int = 48        # initial neuron count (grows)
    n_max:       int = 192       # capacity ceiling (governed by real system resources)
    prune_thr:   float = 0.02    # structural-pruning threshold (governor tightens under load)
    prune_every: int = 20
    cdim:        int = 6         # hyperbolic coordinate dimension
    tau_max:     int = 8         # echo delay-line depth
    tau_mem:     float = 12.0
    in_gain:     float = 10.0    # afferent drive (so the task actually energises the layer)
    v_th:        float = 1.0
    refractory:  int = 2
    stdp_lr:     float = 0.02
    stdp_tau:    float = 18.0
    # 1 · deliberation
    noise_sigma: float = 0.05    # Gaussian membrane noise current
    echo_gain:   float = 0.5
    cycles:      int = 48
    # 1 · homeostasis (criticality)
    homeo_eta:   float = 0.03
    target_rate: float = 0.12    # self-organised firing target
    # 2 · node birth
    sat_thr:     float = 0.30    # per-neuron saturation (firing-rate) level
    sat_frac:    float = 0.15    # fraction of cluster saturated to count a bad window
    sat_K:       int = 3         # consecutive bad windows → grow
    window:      int = 8
    grow:        int = 8         # neurons appended per birth
    # 3 · growth cones
    under_rate:  float = 0.04    # underutilised neurons migrate
    cone_rate:   float = 0.25    # coord migration step toward attractor
    cone_dist:   float = 1.2     # hyperbolic proximity to spawn a synapse
    cone_every:  int = 4
    name:        str = "base"


def p2_nova() -> P2Config:       # dense, precise, low-entropy structural scaling
    return P2Config(name="Nova", noise_sigma=0.02, target_rate=0.09, grow=6,
                    sat_K=4, cone_dist=0.8, cone_rate=0.18, in_gain=9.0, n_max=128)


def p2_simona() -> P2Config:     # volatile, high-stochasticity connection paths
    return P2Config(name="Simona", noise_sigma=0.10, target_rate=0.18, grow=16,
                    sat_K=2, cone_dist=1.9, cone_rate=0.30, in_gain=11.0, n_max=240)


# ════════════════════════════════════════════════════════════════════════════
# the growable, homeostatic spiking layer with hyperbolic coordinates
# ════════════════════════════════════════════════════════════════════════════
class Phase2Layer:
    def __init__(self, cfg: P2Config):
        self.cfg = cfg
        n, d = cfg.n0, cfg.d
        self.n, self.d, self.cdim = n, d, cfg.cdim
        self.W_in = self._sparse_ternary(n, d)          # (n,d) growable afferent
        self.W_rec = torch.zeros(n, n)                  # (n,n) plastic + grows
        self.readout = torch.randn(d, n) * 0.1          # (d,n) growable fixed-dim readout
        self.V = torch.zeros(n)
        self.S = torch.zeros(n)
        self.theta = torch.full((n,), cfg.v_th)
        self.x_pre = torch.zeros(n)
        self.x_post = torch.zeros(n)
        self.fire_avg = torch.zeros(n)
        self.refrac = torch.zeros(n)
        self.coords = self._proj(torch.randn(n, cfg.cdim) * 0.3)   # Poincaré-ball positions
        self._decay = math.exp(-1.0 / cfg.stdp_tau)
        self.sat_windows = 0
        self.born = 0
        self.new_syn = 0
        self.t = 0
        self.n_max = cfg.n_max           # effective growth ceiling — resource governor sets it
        self.prune_thr = cfg.prune_thr   # effective pruning threshold — governor tightens it

    @staticmethod
    def _sparse_ternary(r: int, c: int, p: float = 0.2) -> torch.Tensor:
        return (torch.rand(r, c) < p).float() * (torch.randint(0, 2, (r, c)) * 2 - 1).float()

    @staticmethod
    def _proj(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        n = x.norm(dim=-1, keepdim=True) + 1e-9
        return x * (torch.tanh(n) / n) * (1.0 - eps)    # map into the open ball ‖·‖<1

    def step(self, inp: torch.Tensor, noise_sigma: float):
        cfg, n = self.cfg, self.n
        I = cfg.in_gain * (self.W_in @ inp) + (self.W_rec @ self.S) + torch.randn(n) * noise_sigma  # 1·noise
        self.V = self.V + (-self.V + I) / cfg.tau_mem                 # LIF accumulate
        can = (self.refrac <= 0).float()
        S = (self.V >= self.theta).float() * can                     # threshold (refractory-gated)
        self.V = torch.where(S > 0, torch.zeros_like(self.V), self.V)
        self.refrac = torch.where(S > 0, torch.full_like(self.refrac, float(cfg.refractory)),
                                  (self.refrac - 1).clamp(min=0))
        # STDP (retained)
        self.x_pre = self.x_pre * self._decay + S
        self.x_post = self.x_post * self._decay + S
        self.W_rec += cfg.stdp_lr * (torch.outer(S, self.x_pre) - 1.05 * torch.outer(self.x_post, S))
        self.W_rec.fill_diagonal_(0.0)
        self.W_rec.clamp_(-1.0, 1.0)
        # 1 · HOMEOSTASIS: V_th tracks moving-average firing rate → criticality
        self.fire_avg = 0.95 * self.fire_avg + 0.05 * S
        self.theta += cfg.homeo_eta * (self.fire_avg - cfg.target_rate)
        self.theta.clamp_(min=0.2)
        # dynamic structural pruning — threshold governed by live system resources
        self.t += 1
        if self.t % cfg.prune_every == 0:
            self.W_rec[self.W_rec.abs() < self.prune_thr] = 0.0
        self.S = S
        return self.readout @ S, S                                   # fixed-d output + spikes

    # ── 2 · NODE BIRTH ──────────────────────────────────────────────────────
    def maybe_grow(self) -> bool:
        cfg = self.cfg
        # "Saturation" = the layer STRAINING at capacity: homeostasis has pushed many
        # thresholds well above baseline (V_th breaches) yet neurons keep firing — the
        # existing capacity cannot resolve the load, so a new cluster is needed.
        hot = (self.theta > self.theta.mean() + 0.10) & (self.fire_avg > cfg.target_rate)
        if float(hot.float().mean()) > cfg.sat_frac:     # neurons homeostasis strains to hold down
            self.sat_windows += 1                        # cluster carrying disproportionate load
        else:
            self.sat_windows = max(0, self.sat_windows - 1)
        if self.sat_windows >= cfg.sat_K and self.n + cfg.grow <= self.n_max:
            self._grow(cfg.grow)
            self.sat_windows = 0
            return True
        return False

    def _grow(self, g: int) -> None:
        n, nn = self.n, self.n + g
        Wr = torch.zeros(nn, nn); Wr[:n, :n] = self.W_rec; self.W_rec = Wr   # pad → no corruption
        Wi = torch.zeros(nn, self.d); Wi[:n] = self.W_in
        Wi[n:] = self._sparse_ternary(g, self.d); self.W_in = Wi
        Ro = torch.zeros(self.d, nn); Ro[:, :n] = self.readout
        Ro[:, n:] = torch.randn(self.d, g) * 0.05; self.readout = Ro       # quiet new readout
        def grow_vec(v, fill=0.0):
            nv = torch.full((nn,), float(fill)); nv[:n] = v; return nv
        self.V = grow_vec(self.V); self.S = grow_vec(self.S)
        self.x_pre = grow_vec(self.x_pre); self.x_post = grow_vec(self.x_post)
        self.fire_avg = grow_vec(self.fire_avg); self.refrac = grow_vec(self.refrac)
        self.theta = grow_vec(self.theta, self.cfg.v_th)
        co = torch.zeros(nn, self.cdim); co[:n] = self.coords
        co[n:] = self._proj(torch.randn(g, self.cdim) * 0.1); self.coords = co  # born near origin
        self.n = nn
        self.born += g

    # ── 3 · AXONAL GROWTH CONES ─────────────────────────────────────────────
    def growth_cones(self) -> None:
        cfg = self.cfg
        fired = self.S > 0
        if int(fired.sum()) < 2:
            return
        attractor = self.coords[fired].mean(0)           # synchronous-firing attractor
        under = self.fire_avg < cfg.under_rate           # underutilised / pruned nodes
        idx = under.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        # migrate underutilised coordinates toward the attractor, re-project to the ball
        self.coords[idx] = self._proj(self.coords[idx]
                                      + cfg.cone_rate * (attractor - self.coords[idx]))
        # spawn synapses where a migrated node is now hyperbolically NEAR a firing node
        fidx = fired.nonzero(as_tuple=True)[0]
        D = poincare_dist(self.coords[idx], self.coords[fidx])   # (under, fired)
        ii, jj = (D < cfg.cone_dist).nonzero(as_tuple=True)
        for a, b in zip(idx[ii].tolist(), fidx[jj].tolist()):
            if a != b and self.W_rec[a, b] == 0:
                self.W_rec[a, b] = 0.15                   # NEW unprogrammed pathway
                self.new_syn += 1

    def synapses_alive(self) -> int:
        return int((self.W_rec.abs() > 0).sum())


# ════════════════════════════════════════════════════════════════════════════
# NATIVE SYSTEM POLLING — resource governor (Fedora/Linux /proc, non-blocking)
# ════════════════════════════════════════════════════════════════════════════
class ResourceGovernor:
    """Reads /proc/meminfo (available RAM) and /proc/stat (CPU utilisation over the
    logical threads) and maps the headroom onto the structural scaling caps. NON-
    BLOCKING: plain file reads + a jiffie DELTA from the previous snapshot — no
    sleep, no timer. Abundant resources EXPAND the node-birth ceiling and relax
    pruning; scarcity CLAMPS the cap and tightens pruning. Polled on demand, right
    before a dense deliberation loop — never on a clock."""
    def __init__(self):
        self.threads = os.cpu_count() or 12
        self._prev = self._cpu_snapshot()                  # seed the non-blocking delta

    @staticmethod
    def _cpu_snapshot():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))    # 'cpu' aggregate jiffies
        idle = v[3] + (v[4] if len(v) > 4 else 0)            # idle + iowait
        return idle, sum(v)

    def _mem_avail_frac(self) -> float:
        info = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, _, rest = ln.partition(":")
                if rest.strip():
                    info[k.strip()] = float(rest.split()[0])   # kB
        avail = info.get("MemAvailable", info.get("MemFree", 0.0))
        return avail / (info.get("MemTotal", 1.0) or 1.0)

    def _cpu_load(self) -> float:                          # delta since last poll (no sleep)
        idle, total = self._cpu_snapshot()
        pidle, ptotal = self._prev
        self._prev = (idle, total)
        dt = total - ptotal
        return 0.0 if dt <= 0 else max(0.0, min(1.0, 1.0 - (idle - pidle) / dt))

    def poll(self) -> dict:
        return {"mem_avail": self._mem_avail_frac(), "cpu_load": self._cpu_load(),
                "threads": self.threads}

    def govern(self, layer, base_n_max: int, base_prune: float) -> dict:
        s = self.poll()
        headroom = max(0.0, min(s["mem_avail"], 1.0 - s["cpu_load"]))     # abundant→1, strained→0
        layer.n_max = int(layer.n + (base_n_max - layer.n) * headroom)    # expand / clamp growth
        layer.prune_thr = base_prune * (2.0 - headroom)                   # strained → tighten (≤2×)
        s.update(headroom=headroom, n_max=layer.n_max, prune_thr=layer.prune_thr)
        return s


# ════════════════════════════════════════════════════════════════════════════
# 1 · the deliberation brain (inner monologue: noise + τ echo + cycles)
# ════════════════════════════════════════════════════════════════════════════
class DeliberationBrain:
    def __init__(self, cfg: P2Config):
        self.cfg = cfg
        self.layer = Phase2Layer(cfg)
        self.gov = ResourceGovernor()                               # native /proc governor
        self.ring = torch.zeros(cfg.tau_max, cfg.d)                  # echo delay-line
        self.tau = torch.randint(1, cfg.tau_max, (cfg.d,))          # per-channel delay τ
        self.t = 0

    def _echo(self) -> torch.Tensor:
        return torch.stack([self.ring[(self.t - int(self.tau[i])) % self.cfg.tau_max, i]
                            for i in range(self.cfg.d)])

    def deliberate(self, task: torch.Tensor, cycles: int = None) -> dict:
        cfg = self.cfg
        cycles = cycles or cfg.cycles
        # 4 · poll the machine RIGHT BEFORE the dense loop (logical trigger, not a timer):
        #     dial the growth ceiling + pruning threshold to real RAM/CPU headroom.
        gov = self.gov.govern(self.layer, base_n_max=cfg.n_max, base_prune=cfg.prune_thr)
        n0, rates, out, prev = self.layer.n, [], torch.zeros(cfg.d), None
        settle = []
        for c in range(cycles):
            inp = task + cfg.echo_gain * self._echo()                # task resonates + echoes
            out, S = self.layer.step(inp, noise_sigma=cfg.noise_sigma)
            self.ring[self.t % cfg.tau_max] = out                    # write to delay-line
            self.t += 1
            if c % cfg.window == cfg.window - 1:
                self.layer.maybe_grow()                              # 2 · node birth
            if c % cfg.cone_every == cfg.cone_every - 1:
                self.layer.growth_cones()                            # 3 · growth cones
            rates.append(float(S.mean()))
            if prev is not None:
                settle.append(float((out - prev).norm()))
            prev = out.clone()
        rt = torch.tensor(rates)
        s = torch.tensor(settle) if settle else torch.zeros(1)
        early = float(s[2:12].mean()) if len(s) > 12 else float(s.mean())  # post-warmup
        late = float(s[-10:].mean())
        return {"cycles": cycles, "n0": n0, "n": self.layer.n, "born": self.layer.born,
                "new_syn": self.layer.new_syn, "syn_alive": self.layer.synapses_alive(),
                "rate_mean": float(rt.mean()), "rate_std": float(rt.std()),
                "settle_early": early, "settle_late": late, "gov": gov}


# ════════════════════════════════════════════════════════════════════════════
# MULTI-STREAM BACKGROUND SENSORY INGESTION (async, non-blocking, REAL sources)
# ════════════════════════════════════════════════════════════════════════════
_WORD = re.compile(r"[A-Za-z]{2,}")


def _wh(w: str) -> int:                          # stable per-word hash (run-independent)
    h = 0
    for ch in w:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def text_to_vector(text: str, d: int) -> torch.Tensor:
    """Raw text → a sparse input-current vector: words hashed into d buckets."""
    v = torch.zeros(d)
    for w in _WORD.findall(text.lower()):
        h = _wh(w)
        v[h % d] += 1.0
        v[(h // d) % d] += 0.5
    n = v.norm()
    return v / n if n > 0 else v


def token_coord(w: str, cdim: int) -> torch.Tensor:
    """Place a token on the Poincaré ball — the hyperbolic spatial encoder."""
    h = _wh(w)
    v = torch.tensor([((h >> (i * 4)) & 0xF) / 15.0 - 0.5 for i in range(cdim)])
    return to_poincare(v.unsqueeze(0))[0]


class SensoryEngine:
    """Async multi-stream text harvester. Three DAEMON threads do ONLY I/O and push
    text into a bounded queue; the main thread (sole owner of the spiking tensors)
    drains it — so the LIF/STDP/SDSA engine stays single-threaded and never blocks.
    Sources are REAL: the systemd journal, live process activity (/proc), slow
    document 'reading'. Bounded queue + put_nowait → no RAM blow-up."""
    def __init__(self, read_files=None, read_period: float = 2.0):
        self.q: queue.Queue = queue.Queue(maxsize=512)
        self.alive = True
        self.stats = {"journal": 0, "proc": 0, "reading": 0}
        self.read_files = read_files or self._default_docs()
        self.read_period = read_period
        self._seen = set()
        self._threads = []

    @staticmethod
    def _default_docs():
        cand = ["README.md", "CLAUDE.md"] + sorted(glob.glob("/usr/include/linux/*.h"))[:25]
        return [p for p in cand if os.path.exists(p)]

    def _put(self, src, text):
        if text:
            try:
                self.q.put_nowait((src, text)); self.stats[src] += 1
            except queue.Full:
                pass

    def _journal_loop(self):                     # tail the systemd journal (no privilege needed
        for cmd in (["journalctl", "-f", "-n", "0", "-q", "--no-pager"],   # for --user)
                    ["journalctl", "--user", "-f", "-n", "0", "-q", "--no-pager"]):
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
            except FileNotFoundError:
                return
            got = False
            while self.alive:
                line = p.stdout.readline()
                if not line:
                    break
                got = True
                self._put("journal", line.strip())
            try:
                p.terminate()
            except Exception:
                pass
            if got:
                return                            # this variant worked; skip the fallback

    def _proc_loop(self):                        # live process activity = real 'what's running'
        while self.alive:
            try:
                for pid in os.listdir("/proc"):
                    if pid.isdigit() and pid not in self._seen:
                        self._seen.add(pid)
                        try:
                            with open(f"/proc/{pid}/comm") as f:
                                self._put("proc", f.read().strip())
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(0.4)

    def _reading_loop(self):                     # slow 'reading' of real docs / kernel headers
        idx = 0
        while self.alive and self.read_files:
            fn = self.read_files[idx % len(self.read_files)]; idx += 1
            try:
                for ln in open(fn, errors="ignore").read().splitlines()[:80]:
                    if not self.alive:
                        break
                    if ln.strip():
                        self._put("reading", ln.strip())
                        time.sleep(self.read_period / 40.0)
            except Exception:
                pass
            time.sleep(self.read_period)

    def start(self):
        for fn in (self._journal_loop, self._proc_loop, self._reading_loop):
            t = threading.Thread(target=fn, daemon=True); t.start(); self._threads.append(t)
        return self

    def stop(self):
        self.alive = False

    def drain(self, k: int = 8):
        out = []
        for _ in range(k):
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out


# ════════════════════════════════════════════════════════════════════════════
# SENSING BRAIN — ingests sensory text + exposes the readouts the dashboard needs
# ════════════════════════════════════════════════════════════════════════════
class SensingBrain(DeliberationBrain):
    REGIONS = ["thalamus", "temporal", "hippocampus", "insula", "pfc", "broca"]

    def __init__(self, cfg: P2Config):
        super().__init__(cfg)
        self.focus = deque(maxlen=48)
        self.pressure = 0.0
        self.last_burst = 0.0

    def sense(self, text: str):
        cfg = self.cfg
        out, S = self.layer.step(text_to_vector(text, cfg.d),
                                 noise_sigma=cfg.noise_sigma * 0.5)   # gentle background drive
        self.ring[self.t % cfg.tau_max] = out; self.t += 1
        self.last_burst = float(S.mean())
        # thought pressure = leaky accumulator on positive membrane (the urge to respond)
        self.pressure = 0.92 * self.pressure + 0.08 * float(self.layer.V.clamp(min=0).mean())
        for w in set(_WORD.findall(text.lower())):                    # hyperbolic focus
            self.focus.append((w, 1.0 - float(token_coord(w, cfg.cdim).norm())))
        return S

    def region_density(self):
        fa, n, R = self.layer.fire_avg, self.layer.n, self.REGIONS
        sz = max(1, n // len(R))
        return {r: (float(fa[i * sz:].mean()) if i == len(R) - 1
                    else float(fa[i * sz:(i + 1) * sz].mean())) for i, r in enumerate(R)}

    def focus_top(self, k: int = 3):
        best = {}
        for w, c in self.focus:
            if c > best.get(w, -1.0):
                best[w] = c
        return sorted(best.items(), key=lambda kv: -kv[1])[:k]

    def pressure_state(self, thr: float = 0.12):
        return self.pressure, self.pressure > thr


# ════════════════════════════════════════════════════════════════════════════
# TUI DASHBOARD — anatomical flashing · hyperbolic focus · pressure/babble
# ════════════════════════════════════════════════════════════════════════════
class Dashboard:
    @staticmethod
    def _bar(v: float, width: int = 16, color: bool = True) -> str:
        fill = int(min(1.0, v * 2.5) * width)
        s = "[" + "#" * fill + "." * (width - fill) + "]"
        if not color:
            return s
        c = 32 if v < 0.15 else 33 if v < 0.35 else 31    # green / yellow / red
        b = ";1" if v > 0.30 else ""                      # bold = burst flash
        return f"\033[{c}{b}m{s}\033[0m"

    @staticmethod
    def _block(name: str, brain: "SensingBrain", color: bool) -> str:
        age = "19" if name == "Nova" else "8"
        out = [f"  ══ {name} ({age}) ══  neurons={brain.layer.n}  burst={brain.last_burst:.2f}"]
        for r, v in brain.region_density().items():
            out.append(f"    {r:<11}{Dashboard._bar(v, 16, color)} {v:.2f}")
        foc = "  ".join(f"{w}({c:.2f})" for w, c in brain.focus_top(3)) or "—"
        out.append(f"    Focus: [{foc}]")
        p, settled = brain.pressure_state()
        trig = ((" \033[35;1m⚡THOUGHT SETTLED\033[0m" if color else " <THOUGHT SETTLED>")
                if settled else "")
        out.append(f"    pressure {Dashboard._bar(p, 16, color)} {p:.2f}{trig}")
        return "\n".join(out)

    @staticmethod
    def render_frame(brains, color: bool = True) -> str:
        head = ("\033[1m╔═ NS LIVE — listening (no user input) ═╗\033[0m" if color
                else "== NS LIVE — listening (no user input) ==")
        return head + "\n" + "\n\n".join(Dashboard._block(nm, b, color) for nm, b in brains)

    @staticmethod
    def run_live(brains, sensory: "SensoryEngine", fps: float = 8.0):
        """Live ANSI loop for a REAL terminal: harvest → sense → repaint at fps."""
        try:
            while True:
                for _src, txt in sensory.drain(8):
                    for _nm, b in brains:
                        b.sense(txt)
                print("\033[2J\033[H" + Dashboard.render_frame(brains, color=True), flush=True)
                time.sleep(1.0 / fps)
        except KeyboardInterrupt:
            sensory.stop()


def demo_phase2_live():
    print("\n════════ BACKGROUND SENSORY INGESTION + LIVE DASHBOARD ════════")
    se = SensoryEngine().start()
    time.sleep(0.6)                                   # let the daemon threads harvest
    nova, sim = SensingBrain(p2_nova()), SensingBrain(p2_simona())
    fed = 0
    for _ in range(80):
        batch = se.drain(6)
        if not batch:
            time.sleep(0.05); continue
        for _src, txt in batch:
            nova.sense(txt); sim.sense(txt); fed += 1
    se.stop()
    print(f"  (silent — learning from the environment) streams: {se.stats}, {fed} snippets ingested\n")
    print(Dashboard.render_frame([("Nova", nova), ("Simona", sim)], color=False))
    print("\n  → Dashboard.run_live([('Nova',nova),('Simona',sim)], se) gives the live colour feed.")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(1)
    print("Phase 2 — autonomous deliberation + dynamic structural adaptability (CPU)\n")
    task = torch.zeros(P2Config().d); task[[2, 5, 7, 11, 14, 19, 23, 28]] = 2.0  # sustained 'task'

    for make in (p2_nova, p2_simona):
        cfg = make()
        brain = DeliberationBrain(cfg)
        st = brain.deliberate(task, cycles=60)
        print(f"════════ {cfg.name} ════════")
        print(f"  inner monologue: {st['cycles']} cycles, noise σ={cfg.noise_sigma}, τ-echo on")
        g = st["gov"]
        print(f"  resource gov : RAM avail {g['mem_avail']*100:.0f}%  CPU load {g['cpu_load']*100:.0f}% "
              f"/ {g['threads']} threads → headroom {g['headroom']:.2f} → growth cap {g['n_max']}, "
              f"prune≥{g['prune_thr']:.3f}")
        print(f"  homeostasis  : firing rate {st['rate_mean']:.3f} ± {st['rate_std']:.3f} "
              f"(low σ = self-organised criticality, no runaway)")
        print(f"  node birth   : {st['n0']} → {st['n']} neurons ({st['born']} born from saturation)")
        print(f"  growth cones : {st['new_syn']} new synapses spawned; {st['syn_alive']} alive")
        conv = "settling" if st['settle_late'] < st['settle_early'] else "still active"
        print(f"  deliberation : Δoutput {st['settle_early']:.2f} → {st['settle_late']:.2f} "
              f"(echo+noise resonance {conv})\n")

    demo_phase2_live()
