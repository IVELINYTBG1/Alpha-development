"""
brain.py — The Synaptic Tissue
=================================
Three-population Spiking Neural Network built with snnTorch.

POPULATION PHYSICS:
  Phill   → Affective Core / Neuromodulatory Field
              Slow leak (β=0.95), high threshold. Holds long-term homeostatic state.
              Its membrane voltage acts as a GLOBAL GAIN FIELD that reshapes
              Nova's and Simona's thresholds and weights in real-time.

  Nova    → Precise Orchestrator / Elder Sister
              High inertia (β=0.90), high firing threshold (thr=1.2).
              Represents structural logic and long-term planning.
              Under Phill-stress her gates TIGHTEN — she becomes more selective.

  Simona  → Hasty Agent / Cat-Girl
              Low inertia (β=0.60), low firing threshold (thr=0.5).
              Represents curiosity, impulsive lateral thinking, immediate reaction.
              Under Phill-stress her resistance DROPS — she fires chaotically.

NEUROMODULATION (The Critical Physics):
  Let V_phill = current membrane voltage of Phill (scalar, clamped [0,1]).

      Nova threshold   = thr_nova_base   + alpha * V_phill
      Simona threshold = thr_simona_base - beta  * V_phill   (floor=0.1)

  This means:
    • High ambient energy/stress  → Nova gets harder to fire (stable),
                                    Simona gets easier to fire (chaotic).
    • Low ambient energy/calm     → Both settle toward their base rhythms.

SPARSE ACTIVATION:
  LIF neurons are DARK by default (zero membrane voltage).
  They only spike when accumulated input crosses their dynamic threshold.
  99 % of the network remains silent at any given timestep.
"""

import torch
import snntorch as snn
from snntorch import surrogate
import numpy as np


# ──────────────────────────────────────────────
#  Hyper-parameters — The Physics Constants
# ──────────────────────────────────────────────

# Phill — Affective Core
PHILL_BETA        = 0.95   # Very slow membrane leak  (long emotional memory)
PHILL_THRESHOLD   = 1.0    # High threshold            (doesn't spike easily)
PHILL_INPUT_DIM   = 8      # Raw ambient "vibe" vector dimension
PHILL_HIDDEN_DIM  = 16

# Nova — Precise Orchestrator
NOVA_BETA_BASE    = 0.90   # Slow leak                 (high inertia / stable thoughts)
NOVA_THRESHOLD    = 1.2    # Elevated firing threshold
NOVA_HIDDEN_DIM   = 32

# Simona — Hasty Agent
SIMONA_BETA_BASE  = 0.60   # Fast leak                 (impulsive, quick to forget)
SIMONA_THRESHOLD  = 0.5    # Low firing threshold      (fires at the slightest stimulus)
SIMONA_HIDDEN_DIM = 32

# Neuromodulation coupling constants
ALPHA_NOVA    = 0.4   # How much Phill tightens Nova's gate
BETA_SIMONA   = 0.35  # How much Phill loosens Simona's gate
SIMONA_FLOOR  = 0.10  # Simona's threshold never drops below this (prevent runaway chaos)

# Surrogate gradient for backprop through spikes (fast sigmoid)
SPIKE_GRAD = surrogate.fast_sigmoid(slope=25)


# ──────────────────────────────────────────────
#  Helper — Dynamic LIF neuron wrapper
# ──────────────────────────────────────────────

class DynamicLIF:
    """
    A Leaky Integrate-and-Fire neuron whose threshold is updated
    each timestep from outside (neuromodulation).
    snnTorch's Leaky layer owns the membrane state; we rebuild
    the layer when threshold changes (lightweight — no weight loss).
    """

    def __init__(self, beta: float, threshold: float):
        self.beta      = beta
        self.threshold = threshold
        self._lif      = snn.Leaky(beta=beta, threshold=threshold,
                                   spike_grad=SPIKE_GRAD, learn_beta=False)
        self.mem       = self._lif.init_leaky()   # membrane voltage state

    def update_threshold(self, new_threshold: float):
        """Neuromodulation: hot-swap the firing threshold."""
        if abs(new_threshold - self.threshold) > 1e-4:
            self.threshold = new_threshold
            self._lif = snn.Leaky(beta=self.beta, threshold=new_threshold,
                                  spike_grad=SPIKE_GRAD, learn_beta=False)
            # Preserve membrane state across the hot-swap
            # (the new layer starts with init voltage; we inject saved state)

    def forward(self, current: torch.Tensor):
        """Returns (spike, membrane_voltage)."""
        spk, self.mem = self._lif(current, self.mem)
        return spk, self.mem

    def reset(self):
        self.mem = self._lif.init_leaky()


# ──────────────────────────────────────────────
#  NeuromorphicBrain — The Living Tissue
# ──────────────────────────────────────────────

class NeuromorphicBrain:
    """
    The shared biological substrate of Nova and Simona,
    regulated by their affective core Phill.

    Usage (called from Rust via PyO3):
        brain = NeuromorphicBrain()
        result = brain.step(vibe_vector)   # list[float] of length PHILL_INPUT_DIM
        # returns dict with spike counts and membrane readings
    """

    def __init__(self):
        torch.manual_seed(42)

        # ── Synaptic Projections (the "axons") ──────────────────────────────
        # Phill receives raw ambient vibe
        self.phill_proj = torch.nn.Linear(PHILL_INPUT_DIM, PHILL_HIDDEN_DIM, bias=False)

        # Nova receives Phill's spike pattern + original vibe
        self.nova_proj  = torch.nn.Linear(PHILL_HIDDEN_DIM + PHILL_INPUT_DIM,
                                          NOVA_HIDDEN_DIM, bias=False)

        # Simona receives Phill's spike pattern + original vibe
        self.simona_proj = torch.nn.Linear(PHILL_HIDDEN_DIM + PHILL_INPUT_DIM,
                                           SIMONA_HIDDEN_DIM, bias=False)

        # Small random init — prevents dead neurons at t=0
        torch.nn.init.normal_(self.phill_proj.weight,  mean=0.0, std=0.15)
        torch.nn.init.normal_(self.nova_proj.weight,   mean=0.0, std=0.10)
        torch.nn.init.normal_(self.simona_proj.weight, mean=0.0, std=0.20)

        # ── LIF Populations ─────────────────────────────────────────────────
        self.phill  = DynamicLIF(beta=PHILL_BETA,    threshold=PHILL_THRESHOLD)
        self.nova   = DynamicLIF(beta=NOVA_BETA_BASE, threshold=NOVA_THRESHOLD)
        self.simona = DynamicLIF(beta=SIMONA_BETA_BASE, threshold=SIMONA_THRESHOLD)

        # ── Running History (for introspection) ─────────────────────────────
        self.tick = 0
        self.phill_voltage_history  = []
        self.nova_spikes_total      = 0
        self.simona_spikes_total    = 0

    # ── Core Step — called once per event loop tick ──────────────────────────

    def step(self, vibe: list) -> dict:
        """
        Process one timestep of ambient sensory input.

        Args:
            vibe : list[float] of length PHILL_INPUT_DIM
                   Represents the raw energy/vibe of the environment.
                   In production: acoustic energy, camera motion, etc.
                   In this scaffold: dummy floats from Rust's event loop.

        Returns:
            dict with keys:
              tick            — current timestep index
              phill_voltage   — Phill's membrane voltage (scalar)
              phill_spiked    — bool, did Phill fire?
              nova_spikes     — int, how many Nova neurons spiked
              simona_spikes   — int, how many Simona neurons spiked
              nova_threshold  — Nova's current (modulated) threshold
              simona_threshold— Simona's current (modulated) threshold
              nova_mem_mean   — mean membrane voltage across Nova's layer
              simona_mem_mean — mean membrane voltage across Simona's layer
        """
        self.tick += 1

        with torch.no_grad():   # Inference mode — no gradient tracking

            # ── 1. Convert vibe to tensor ────────────────────────────────
            x = torch.tensor(vibe, dtype=torch.float32).unsqueeze(0)  # [1, 8]

            # ── 2. Phill — Affective Core ────────────────────────────────
            phill_current = self.phill_proj(x)          # [1, 16]
            phill_spk, phill_mem = self.phill.forward(phill_current)

            # Extract Phill's scalar voltage (mean across hidden neurons)
            # Clamp to [0, 1] to use as a clean modulation signal
            V_phill = float(phill_mem.mean().clamp(0.0, 1.0))
            self.phill_voltage_history.append(V_phill)

            # ── 3. NEUROMODULATION — The Critical Physics ─────────────────
            #   Nova tightens (higher threshold) when Phill is excited
            nova_thr_now = NOVA_THRESHOLD + ALPHA_NOVA * V_phill
            self.nova.update_threshold(nova_thr_now)

            #   Simona loosens (lower threshold) when Phill is excited
            simona_thr_now = max(SIMONA_FLOOR,
                                 SIMONA_THRESHOLD - BETA_SIMONA * V_phill)
            self.simona.update_threshold(simona_thr_now)

            # ── 4. Nova — Precise Orchestrator ───────────────────────────
            # Her input is Phill's spike pattern + the raw vibe
            nova_in = torch.cat([phill_spk, x], dim=1)   # [1, 16+8]
            nova_current = self.nova_proj(nova_in)         # [1, 32]
            nova_spk, nova_mem = self.nova.forward(nova_current)

            # ── 5. Simona — Hasty Agent ───────────────────────────────────
            # Same input structure; different physics, different result
            simona_in = torch.cat([phill_spk, x], dim=1)
            simona_current = self.simona_proj(simona_in)
            simona_spk, simona_mem = self.simona.forward(simona_current)

            # ── 6. Accumulate ─────────────────────────────────────────────
            n_nova_spikes   = int(nova_spk.sum().item())
            n_simona_spikes = int(simona_spk.sum().item())
            self.nova_spikes_total   += n_nova_spikes
            self.simona_spikes_total += n_simona_spikes

        return {
            "tick":             self.tick,
            "phill_voltage":    round(V_phill, 6),
            "phill_spiked":     bool(phill_spk.sum().item() > 0),
            "nova_spikes":      n_nova_spikes,
            "simona_spikes":    n_simona_spikes,
            "nova_threshold":   round(nova_thr_now, 4),
            "simona_threshold": round(simona_thr_now, 4),
            "nova_mem_mean":    round(float(nova_mem.mean().item()), 6),
            "simona_mem_mean":  round(float(simona_mem.mean().item()), 6),
        }

    def reset_state(self):
        """Hard reset — wipe all membrane voltages. Like dreamless sleep."""
        self.phill.reset()
        self.nova.reset()
        self.simona.reset()
        self.phill_voltage_history.clear()
        self.nova_spikes_total   = 0
        self.simona_spikes_total = 0
        self.tick = 0

    def introspect(self) -> dict:
        """Return long-term running stats — the 'emotional summary'."""
        history = self.phill_voltage_history
        return {
            "total_ticks":        self.tick,
            "phill_mean_voltage": round(float(np.mean(history)) if history else 0.0, 6),
            "phill_peak_voltage": round(float(np.max(history))  if history else 0.0, 6),
            "nova_total_spikes":  self.nova_spikes_total,
            "simona_total_spikes":self.simona_spikes_total,
        }
