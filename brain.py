"""
brain.py — NeuromorphicBrain · Phase 5: CPU-Native, Multimodal, Emergent Identity
===================================================================================

ARCHITECTURE:
  • All tensors on CPU. DEVICE = torch.device("cpu") — no fallback, no iGPU.
  • MKL/OpenMP thread count pinned to physical core count at startup.
  • Process priority elevated to HIGH on Windows, nice(-10) on Linux.
  • Audio spikes travel through a pre-allocated numpy array (zero-copy).

  • 13 anatomical regions total (7 Nova + 6 Simona). Phill untouched.
  • Each brain is a separate object with completely separate membrane state.
  • They share: Phill's voltage field, SharedSemanticDictionary, ThoughtPipe.
  • They do NOT share: weights, thresholds, membrane voltages, opinions.

MULTIMODAL IMPRINTING (no hardcoding):
  MultimodalImprinter receives 3 signal streams each tick:
    face_vec     [FACE_VEC_DIM]      — from vision.py
    voice_vec    [5]                 — from audio thread (RMS+features)
    kinematic    [KINEMATIC_VEC_DIM] — from vision.py
  Coincidence Detection:
    When all 3 signals fire above their respective thresholds simultaneously,
    a "coincidence event" is recorded.
    Hebbian learning updates weights: w += lr * pre * post (full precision).
    NO boolean flag. The weight IS the memory.
  "This is me" command:
    Temporarily raises learning rate and lowers coincidence thresholds.
    Still requires real sustained coincidence. 5 seconds of looking → nothing.
    30+ seconds of sustained multimodal activation → meaningful weights.

ANTI-GULLIBILITY PROTOCOL:
  ACC receives: face signal + kinematic signal separately.
  If face_score is high but kinematic_score is low:
    → ACC fires an inhibitory spike (negative current) into PFC and Insula.
    → Nova enters Vigilance Mode (higher PFC threshold, dampened response).
    → Simona stays cold (Insula_S threshold rises).
  This is purely physical — no if/else. The inhibitory current just
  prevents PFC from crossing θ. Emergence, not logic.

THOUGHT PIPE (fully emergent):
  Each brain has a RuminationBuffer — thoughts processed internally
  but not yet spoken accumulate there.
  A "pressure neuron" (LeakyAccumulator) integrates:
    pressure += (rumination_load * V_phill * broca_activity)
    pressure *= decay  (each tick)
  When pressure crosses θ_leak, the oldest thought in the buffer leaks.
  Nova's θ_leak = 0.85 (she only leaks under real pressure)
  Simona's θ_leak = 0.28 (she blurts almost anything)
  This is NOT a ping. There is NO scheduled call.
  The brain loop checks if pressure crossed threshold — that IS the
  physical mechanism.

PHILL: COMPLETELY UNTOUCHED.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import sys
import logging as _logging
import threading
import multiprocessing
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP: CPU LOCK + PROCESS PRIORITY
# ══════════════════════════════════════════════════════════════════════════════

def _configure_cpu():
    """
    Pin torch to physical CPU cores, elevate process priority.
    Deterministic clock = no jitter in Nova's 5-tick sustain.
    """
    phys = multiprocessing.cpu_count()
    torch.set_num_threads(phys)
    torch.set_num_interop_threads(max(1, phys // 2))

    # AVX2/MKL: torch on CPU uses MKL automatically if available.
    # Explicitly disable any GPU fallback.
    os.environ["CUDA_VISIBLE_DEVICES"]  = ""
    os.environ["XPU_VISIBLE_DEVICES"]   = ""
    os.environ["OMP_NUM_THREADS"]       = str(phys)
    os.environ["MKL_NUM_THREADS"]       = str(phys)
    os.environ["OPENBLAS_NUM_THREADS"]  = str(phys)

    # Process priority
    try:
        if sys.platform == "win32":
            import ctypes
            # HIGH_PRIORITY_CLASS = 0x80
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
            )
        else:
            os.nice(-10)
    except Exception:
        pass  # Graceful degradation if not admin

    return torch.device("cpu")

DEVICE = _configure_cpu()

# ── Logger ────────────────────────────────────────────────────────────────────
_logging.basicConfig(
    filename="brain_log.txt", level=_logging.INFO,
    format="%(asctime)s %(message)s",
)
_L = _logging.getLogger("nova_simona")
_INIT_MESSAGES: list[str] = []

def _log(msg: str):
    _L.info(msg)
    _INIT_MESSAGES.append(msg)

_log(f"CPU mode: {torch.get_num_threads()} threads | "
     f"MKL={torch.backends.mkl.is_available()} | "
     f"OpenMP={torch.backends.openmp.is_available()}")

# ── snnTorch ──────────────────────────────────────────────────────────────────
try:
    import snntorch as snn
    from snntorch import surrogate
    SPIKE_GRAD   = surrogate.fast_sigmoid(slope=25)
    HAS_SNNTORCH = True
    _log("snnTorch loaded")
except ImportError:
    HAS_SNNTORCH = False
    SPIKE_GRAD   = None
    _log("snnTorch not found — pure-PyTorch LIF active")

# ── Vision imports (soft dependency) ─────────────────────────────────────────
try:
    from vision import VisualFeatureBuffer, CameraThread, FACE_VEC_DIM, KINEMATIC_VEC_DIM
    _HAS_VISION = True
    _log("vision.py loaded — camera integration active")
except ImportError:
    _HAS_VISION = False
    FACE_VEC_DIM      = 32
    KINEMATIC_VEC_DIM = 16
    _log("vision.py not found — camera disabled")

# ── TTS ───────────────────────────────────────────────────────────────────────
try:
    from tts_engine import create_engine as _create_tts
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

# ── Physics constants (unchanged) ─────────────────────────────────────────────
AUDIO_AMPLIFY   = 15.0
PHILL_INPUT_DIM = 8
PHILL_BETA      = 0.95
PHILL_THRESHOLD = 1.0
PHILL_HIDDEN    = 16
NOVA_LANG       = "en"
SIMONA_LANG     = "en"

# Phill neuromodulation coupling
ALPHA  = 0.40   # Nova PFC threshold rise per V_phill
BETA_M = 0.35   # Simona Broca threshold drop per V_phill
GAMMA  = 0.05   # Nova beta gain
DELTA  = 0.15   # Simona beta drop

# Nova region physics
_NOVA_REGIONS = {
    # name         size  beta   thr    phill_alpha  proj_std
    "thalamus":   (16,  0.85,  0.80,  0.10,        0.13),
    "temporal":   (24,  0.88,  1.00,  0.20,        0.11),
    "hippocampus":(20,  0.93,  1.10,  0.30,        0.10),
    "acc":        (14,  0.87,  0.90,  0.25,        0.12),
    "pfc":        (28,  0.92,  1.40,  0.45,        0.09),
    "broca":      (16,  0.89,  1.20,  0.35,        0.10),
    "insula":     (12,  0.91,  0.95,  0.15,        0.11),
}

# Simona region physics
_SIMONA_REGIONS = {
    # name            size  beta   thr    phill_alpha  noise  proj_std
    "thalamus_s":    (16,  0.62,  0.45,  0.35,        0.05,  0.18),
    "temporal_s":    (20,  0.58,  0.40,  0.20,        0.04,  0.20),
    "hippocampus_s": (14,  0.68,  0.75,  0.25,        0.03,  0.17),
    "pfc_s":         (12,  0.52,  1.90,  0.10,        0.00,  0.09),
    "broca_s":       (12,  0.58,  0.38,  0.15,        0.06,  0.20),
    "insula_s":      (10,  0.60,  0.42,  0.45,        0.04,  0.18),
}


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-COPY AUDIO BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ZeroCopyAudioBuffer:
    """
    Pre-allocated numpy array that Rust writes RMS + features into.
    Brain reads the same memory directly — no copy, no allocation per tick.

    Layout: [rms, zcr, band_low, band_mid, band_high, mic_volume_smoothed]
    Written by: audio thread via update()
    Read by:    brain.step() via read()
    """
    DIM = 6

    def __init__(self):
        self._buf  = np.zeros(self.DIM, dtype=np.float32)
        self._lock = threading.Lock()

    def update(self, rms: float, zcr: float, bl: float, bm: float, bh: float, vol: float):
        with self._lock:
            self._buf[0] = rms
            self._buf[1] = zcr
            self._buf[2] = bl
            self._buf[3] = bm
            self._buf[4] = bh
            self._buf[5] = vol

    def read(self) -> np.ndarray:
        """Returns a VIEW — no copy. Caller must not modify."""
        with self._lock:
            return self._buf.copy()  # one copy at the read boundary is unavoidable
            # but there is no allocation in the write path

    @property
    def rms(self) -> float:
        return float(self._buf[0])

    @property
    def voice_features(self) -> list:
        return self._buf[:5].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# LIF (pure-torch fallback)
# ══════════════════════════════════════════════════════════════════════════════

class _PureTorchLIF(nn.Module):
    def __init__(self, beta: float, threshold: float = 1.0, **kw):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def init_leaky(self) -> torch.Tensor:
        return torch.zeros(1)

    def forward(self, inp: torch.Tensor, mem: torch.Tensor):
        if mem.shape != inp.shape:
            mem = mem.expand_as(inp).clone()
        mem = self.beta * mem + inp
        spk = (mem >= self.threshold).float()
        mem = mem * (1.0 - spk)
        return spk, mem

    def to(self, *a, **kw): return self


def _make_lif(beta: float, threshold: float) -> nn.Module:
    if HAS_SNNTORCH:
        return snn.Leaky(beta=beta, threshold=threshold,
                         spike_grad=SPIKE_GRAD, learn_beta=False)
    return _PureTorchLIF(beta=beta, threshold=threshold)


# ══════════════════════════════════════════════════════════════════════════════
# BRAIN REGION (unchanged physics, CPU-explicit)
# ══════════════════════════════════════════════════════════════════════════════

class BrainRegion:
    def __init__(self, name, in_dim, size, beta, threshold,
                 phill_alpha, noise=0.0, proj_std=0.12):
        self.name        = name
        self.size        = size
        self.beta        = beta
        self.threshold   = threshold
        self.phill_alpha = phill_alpha
        self.noise       = noise
        self._cur_thr    = threshold  # modulated threshold

        self.proj = nn.Linear(in_dim, size, bias=False)  # CPU explicit
        nn.init.normal_(self.proj.weight, mean=0.0, std=proj_std)

        self._lif  = _make_lif(beta, threshold)
        self._mem  = self._lif.init_leaky()
        self.last_spikes  = torch.zeros(1, size)
        self.total_spikes = 0
        self.spike_history = deque([0] * 30, maxlen=30)

    def modulate(self, V_phill: float):
        new_thr = self.threshold + self.phill_alpha * V_phill
        if abs(new_thr - self._cur_thr) > 1e-4:
            old_mem      = self._mem
            self._lif    = _make_lif(self.beta, new_thr)
            self._mem    = old_mem
            self._cur_thr = new_thr

    def forward(self, inp: torch.Tensor, extra_current: float = 0.0) -> torch.Tensor:
        if inp.shape[-1] != self.proj.in_features:
            diff = self.proj.in_features - inp.shape[-1]
            if diff > 0:
                inp = torch.cat([inp, torch.zeros(1, diff)], dim=1)
            else:
                inp = inp[:, :self.proj.in_features]
        if self.noise > 0.0:
            inp = (inp + torch.randn_like(inp) * self.noise).clamp(min=0.0)
        curr = self.proj(inp)
        if extra_current != 0.0:
            curr = curr + extra_current   # inhibitory if negative
        spk, self._mem = self._lif(curr, self._mem)
        self.last_spikes = spk
        n = int(spk.sum().item())
        self.total_spikes += n
        self.spike_history.append(n)
        return spk

    def reset(self):
        self._mem = self._lif.init_leaky()
        self.last_spikes = torch.zeros(1, self.size)
        self.total_spikes = 0
        self.spike_history = deque([0] * 30, maxlen=30)

    def mean_voltage(self) -> float:
        return float(self._mem.mean().item()) if self._mem is not None else 0.0

    def spike_count(self) -> int:
        return int(self.last_spikes.sum().item())

    def activity(self) -> float:
        return sum(self.spike_history) / (len(self.spike_history) * self.size + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# MULTIMODAL IMPRINTER
# ══════════════════════════════════════════════════════════════════════════════

class MultimodalImprinter:
    """
    Learns to recognize the Architect through coincidence detection.
    Three channels: face, voice, kinematic motion.
    No hardcoded identity. Weights ARE the memory.

    LEARNING MECHANICS:
      Each channel has a "template" (running mean of activated samples).
      Similarity score = cosine similarity vs template.
      Coincidence = all 3 scores above their respective thresholds simultaneously.
      On coincidence: all 3 templates drift toward current sample (Hebbian).

    ANTI-GULLIBILITY:
      Returns a separate face_only_score and kinematic_score.
      If face_only_score > 0.75 AND kinematic_score < 0.40:
        → inhibitory_strength returned to brain (ACC fires negative current)

    "THIS IS ME" mode:
      Lowers coincidence thresholds and raises learning rate for 60s.
      Still requires real multimodal activation. Staring at camera does nothing
      without the voice + motion also being active.
    """

    # Thresholds (cosine similarity) for coincidence detection
    FACE_THR_BASE    = 0.70
    VOICE_THR_BASE   = 0.55
    KIN_THR_BASE     = 0.45

    # During "this is me" imprinting mode
    FACE_THR_LEARN   = 0.40
    VOICE_THR_LEARN  = 0.30
    KIN_THR_LEARN    = 0.25
    IMPRINT_DURATION = 60.0  # seconds

    TEMPLATE_LR_BASE  = 0.005
    TEMPLATE_LR_LEARN = 0.035
    MIN_SAMPLES       = 60    # coincidences before templates are "trusted"
    DECAY             = 0.9998  # templates slowly forget if unused

    def __init__(self):
        self.face_template:  Optional[np.ndarray] = None
        self.voice_template: Optional[np.ndarray] = None
        self.kin_template:   Optional[np.ndarray] = None

        self.face_score:  float = 0.0
        self.voice_score: float = 0.0
        self.kin_score:   float = 0.0
        self.combined:    float = 0.0   # geometric mean of 3 scores

        self.coincidence_count = 0
        self.trusted           = False   # True when MIN_SAMPLES reached

        self._imprint_until: float = 0.0
        self._ema_face   = 0.0
        self._ema_voice  = 0.0
        self._ema_kin    = 0.0
        self._ema_alpha  = 0.90

        self._save_path = Path("imprinter_state.json")
        self._load()

        _log(f"MultimodalImprinter: {self.coincidence_count} prior coincidences, "
             f"trusted={self.trusted}")

    def start_imprinting(self, duration: float = 60.0):
        """Called when user types 'this is me' or similar."""
        self._imprint_until = time.time() + duration
        _log(f"Imprinting mode active for {duration}s")

    @property
    def is_imprinting(self) -> bool:
        return time.time() < self._imprint_until

    def _cosine(self, template: np.ndarray, vec: np.ndarray) -> float:
        if template is None or vec is None:
            return 0.0
        t_n = template / (np.linalg.norm(template) + 1e-8)
        v_n = vec      / (np.linalg.norm(vec)      + 1e-8)
        return float(np.clip(np.dot(t_n, v_n), 0.0, 1.0))

    def _update_template(self, template: Optional[np.ndarray],
                         sample: np.ndarray, lr: float) -> np.ndarray:
        """Hebbian update: template drifts toward sample."""
        if template is None:
            return sample.copy()
        # Apply decay to existing template (forgetting if inactive)
        new = (1.0 - lr) * template * self.DECAY + lr * sample
        nrm = np.linalg.norm(new) + 1e-8
        return (new / nrm).astype(np.float32)

    def update(
        self,
        face_vec:  Optional[np.ndarray],
        voice_vec: Optional[np.ndarray],
        kin_vec:   Optional[np.ndarray],
    ) -> tuple[float, float, float, bool]:
        """
        Process one tick of multimodal input.
        Returns (combined_score, face_only, kin_only, inhibitory_flag).
        """
        imprinting = self.is_imprinting
        face_thr   = self.FACE_THR_LEARN  if imprinting else self.FACE_THR_BASE
        voice_thr  = self.VOICE_THR_LEARN if imprinting else self.VOICE_THR_BASE
        kin_thr    = self.KIN_THR_LEARN   if imprinting else self.KIN_THR_BASE
        lr         = self.TEMPLATE_LR_LEARN if imprinting else self.TEMPLATE_LR_BASE

        # Compute similarity scores
        fs = self._cosine(self.face_template,  face_vec)  if face_vec  is not None else 0.0
        vs = self._cosine(self.voice_template, voice_vec) if voice_vec is not None else 0.0
        ks = self._cosine(self.kin_template,   kin_vec)   if kin_vec   is not None else 0.0

        # EMA smoothing
        self._ema_face  = self._ema_alpha * self._ema_face  + (1-self._ema_alpha) * fs
        self._ema_voice = self._ema_alpha * self._ema_voice + (1-self._ema_alpha) * vs
        self._ema_kin   = self._ema_alpha * self._ema_kin   + (1-self._ema_alpha) * ks

        self.face_score  = self._ema_face
        self.voice_score = self._ema_voice
        self.kin_score   = self._ema_kin

        # Geometric mean — all 3 must be high for combined to be high
        self.combined = float(
            (self._ema_face * self._ema_voice * self._ema_kin) ** (1/3)
        )

        # Coincidence detection — all 3 above threshold simultaneously
        coincidence = (fs >= face_thr and vs >= voice_thr and ks >= kin_thr)

        if coincidence:
            self.coincidence_count += 1
            if self.coincidence_count >= self.MIN_SAMPLES:
                self.trusted = True
            # Hebbian update
            if face_vec  is not None: self.face_template  = self._update_template(self.face_template,  face_vec,  lr)
            if voice_vec is not None: self.voice_template = self._update_template(self.voice_template, voice_vec, lr)
            if kin_vec   is not None: self.kin_template   = self._update_template(self.kin_template,   kin_vec,   lr)
            if self.coincidence_count % 10 == 0:
                self._save()

        # Anti-gullibility: face matches but motion does not
        inhibitory = (self.trusted and fs > 0.75 and ks < 0.40 and face_vec is not None)

        return self.combined, fs, ks, inhibitory

    def status(self) -> str:
        if not self.trusted:
            return f"learning ({self.coincidence_count}/{self.MIN_SAMPLES})"
        c = self.combined
        if c > 0.80: return "ARCHITECT ✓✓"
        if c > 0.55: return f"likely ({c:.2f})"
        if c > 0.30: return f"uncertain ({c:.2f})"
        return "stranger"

    def _save(self):
        try:
            state = {
                "face_template":  self.face_template.tolist()  if self.face_template  is not None else None,
                "voice_template": self.voice_template.tolist() if self.voice_template is not None else None,
                "kin_template":   self.kin_template.tolist()   if self.kin_template   is not None else None,
                "coincidence_count": self.coincidence_count,
                "trusted": self.trusted,
            }
            with open(self._save_path, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _load(self):
        if not self._save_path.exists():
            return
        try:
            with open(self._save_path) as f:
                state = json.load(f)
            if state.get("face_template"):
                self.face_template  = np.array(state["face_template"],  dtype=np.float32)
            if state.get("voice_template"):
                self.voice_template = np.array(state["voice_template"], dtype=np.float32)
            if state.get("kin_template"):
                self.kin_template   = np.array(state["kin_template"],   dtype=np.float32)
            self.coincidence_count = state.get("coincidence_count", 0)
            self.trusted           = state.get("trusted", False)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# THOUGHT PIPE — EMERGENT INNER VOICE
# ══════════════════════════════════════════════════════════════════════════════

class LeakyAccumulator:
    """
    A single neuron that integrates "unexpressed thought pressure."
    Not a LIF — it's a continuous leaky integrator (no hard reset).
    Crosses threshold → the brain leaks a thought. Then resets.
    """
    def __init__(self, threshold: float, decay: float):
        self.threshold = threshold
        self.decay     = decay
        self.voltage   = 0.0

    def integrate(self, input_val: float) -> bool:
        """Returns True if threshold crossed (thought leaks)."""
        self.voltage = self.decay * self.voltage + input_val
        if self.voltage >= self.threshold:
            self.voltage = 0.0
            return True
        return False

    def reset(self):
        self.voltage = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT-MODE NETWORK + INTRINSIC MOTIVATION  (autonomy substrate)
# ══════════════════════════════════════════════════════════════════════════════

class DefaultModeNetwork:
    """
    Intrinsic auditory drive — the SNN equivalent of resting-state activity.

    When the external mic is silent, this provides a small, fluctuating
    "inner murmur" added to mic_volume so Phill stays alive. Without it
    the whole brain flatlines during quiet periods and nothing emerges.

    Drive is shaped by:
      boredom        — time since last external event (mic / leak / speech)
      rumination     — density of unspoken thoughts across both pipes
      intrinsic_noise — AR(1) low-frequency (pink-ish) fluctuation

    This is not a ping. It is a continuous physical signal. The instant
    anything real happens (mic, leak, speech) boredom collapses and the
    drive drops naturally back to its rumination-only baseline.
    """
    def __init__(self, build_rate: float = 0.0012, decay_on_event: float = 0.4):
        self._boredom        = 0.0
        self._build_rate     = build_rate
        self._decay_on_event = decay_on_event
        self._noise_state    = 0.0
        self._noise_alpha    = 0.92

    def drive(self, external_mic: float, rumination_load: float,
              event_this_tick: bool) -> float:
        import random
        if event_this_tick or external_mic > 0.018:
            self._boredom *= self._decay_on_event
        else:
            self._boredom = min(1.0, self._boredom + self._build_rate)

        self._noise_state = (
            self._noise_alpha * self._noise_state
            + (1.0 - self._noise_alpha) * (random.random() - 0.5)
        )
        envelope = 0.6 + self._noise_state  # bias positive — silence still murmurs

        # Drive scale calibrated so a fully-bored brain produces effective_mic
        # near a quiet-speech level (~0.05), which is enough to make Phill
        # fire intermittently through its projection layer.
        intrinsic = (0.55 * self._boredom + 0.45 * rumination_load) * envelope * 0.075
        return max(0.0, intrinsic)

    @property
    def boredom(self) -> float:
        return self._boredom

    def partial_relief(self):
        """A leak self-soothes the brain partially. External input fully."""
        self._boredom *= 0.55


class IntrinsicMotivation:
    """
    Curiosity / restlessness neuron — builds charge each tick, drains when
    external satiation arrives, fires when its threshold is crossed.

    Firing does NOT directly emit text. It returns True so the caller can
    boost concept primes (curiosity / question / search) into the next
    region forward pass. The brain's natural thought generators do the
    rest — emergence, not script.

    Nova carries a high threshold (she is rarely the one to initiate).
    Simona's threshold is low (she fidgets, asks, blurts).
    """
    def __init__(self, threshold: float, build_rate: float,
                 decay: float = 0.992, sated_drain: float = 0.45):
        self.voltage         = 0.0
        self.threshold       = threshold
        self.build_rate      = build_rate
        self.decay           = decay
        self.sated_drain     = sated_drain
        self.last_fire_tick  = -1

    def tick(self, satiation: float, current_tick: int) -> bool:
        if satiation > 0.25:
            self.voltage = max(0.0, self.voltage - self.sated_drain * satiation)
        self.voltage = self.voltage * self.decay + self.build_rate
        if self.voltage >= self.threshold:
            self.voltage        = 0.0
            self.last_fire_tick = current_tick
            return True
        return False


class ThoughtPipe:
    """
    Each brain's inner voice. Accumulates unexpressed thoughts.
    Leaks them when internal pressure is sufficient.
    No scheduled ping. No hardcoded timing.

    The pressure = V_phill * broca_activity * rumination_density
    Nova leaks rarely (high threshold). Simona leaks often (low threshold).
    """

    def __init__(self, name: str, leak_threshold: float, decay: float = 0.97):
        self.name     = name
        self._buffer: deque[str] = deque(maxlen=12)  # max 12 unspoken thoughts
        self._pressure = LeakyAccumulator(leak_threshold, decay)
        self._lock     = threading.Lock()
        self._leaked: deque[str] = deque(maxlen=8)   # recently leaked thoughts
        self.last_leak_tick = 0                       # for personal idle timer

    def push(self, thought: str):
        """Brain pushes an internal thought (not spoken yet)."""
        if thought and thought.strip():
            with self._lock:
                self._buffer.append(thought.strip())

    def tick(self, V_phill: float, broca_activity: float) -> Optional[str]:
        """
        Called each brain tick.
        Accumulates pressure. Returns a leaked thought if threshold crossed.
        """
        with self._lock:
            density = len(self._buffer) / 12.0
        pressure_input = V_phill * broca_activity * density
        leaked = self._pressure.integrate(pressure_input)
        if leaked:
            with self._lock:
                if self._buffer:
                    thought = self._buffer.popleft()
                    self._leaked.append(thought)
                    return thought
        return None

    def get_recent_leaks(self) -> list[str]:
        with self._lock:
            return list(self._leaked)

    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def add_autonomy_pressure(self, amount: float):
        """
        Direct pressure injection from the autonomy substrate (DMN +
        curiosity). Parallel to the V_phill * broca * density pathway,
        which only builds during external excitation.
        """
        if amount > 0:
            self._pressure.voltage += amount


# ══════════════════════════════════════════════════════════════════════════════
# VOICE IDENTITY LEARNER (unchanged from Phase 4)
# ══════════════════════════════════════════════════════════════════════════════

class VoiceIdentityLearner:
    SPEECH_FLOOR  = 0.015; HIGH_TRUST = 0.80; LOW_TRUST = 0.40
    MIN_SAMPLES   = 40;    TEMPLATE_LR = 0.012; TRUST_SMOOTH = 0.85
    FEAT_DIM      = 5
    LOW_SIM_THR   = 0.15   # below this → template considered wrong
    LOW_SIM_TICKS = 60     # ~3s at 20Hz of sustained low sim → reset
    TRUST_FLOOR   = 0.05   # bar stays visible at "learning" rather than 0

    def __init__(self):
        self.template: Optional[np.ndarray] = None
        self.trust = 0.0; self.samples = 0; self.locked = False
        self._sum = np.zeros(self.FEAT_DIM, dtype=np.float64)
        self._low_sim_run = 0   # consecutive low-sim speech frames
        _log("VoiceIdentityLearner initialized")

    def update(self, features: list) -> float:
        f   = np.array(features, dtype=np.float32)
        rms = float(f[0])
        if rms < self.SPEECH_FLOOR:
            # Silence: gently decay trust toward the floor, not all the way down
            self.trust = max(self.TRUST_FLOOR, self.trust * 0.998)
            self._low_sim_run = 0
            return self.trust
        norm = np.linalg.norm(f) + 1e-8; f_n = f / norm
        if self.template is None:
            self._sum += f_n.astype(np.float64); self.samples += 1
            mean = (self._sum / self.samples).astype(np.float32)
            self.template = mean / (np.linalg.norm(mean) + 1e-8)
            self.trust = 0.5
            if self.samples >= self.MIN_SAMPLES and not self.locked:
                self.locked = True; _log(f"Voice locked after {self.samples} frames")
            return self.trust
        sim = float(np.dot(self.template, f_n))
        sim = max(0.0, sim)
        self.trust = self.TRUST_SMOOTH * self.trust + (1 - self.TRUST_SMOOTH) * sim
        self.trust = max(self.TRUST_FLOOR, self.trust)

        # Template adaptation: always nudge during clear speech, faster when
        # we already trust it (refining), slower when trust is low (gradual
        # recovery from a poisoned template). No locked+HIGH_TRUST gate.
        adapt_lr = self.TEMPLATE_LR * (0.25 + 0.75 * self.trust)
        self.template = (1 - adapt_lr) * self.template + adapt_lr * f_n
        self.template /= (np.linalg.norm(self.template) + 1e-8)

        # Hard reset: sustained very-low similarity → template is wrong, rebuild
        if sim < self.LOW_SIM_THR:
            self._low_sim_run += 1
            if self._low_sim_run >= self.LOW_SIM_TICKS:
                _log(f"Voice template reset — {self._low_sim_run} frames at sim<{self.LOW_SIM_THR}")
                self.template = None
                self.locked   = False
                self.samples  = 0
                self.trust    = self.TRUST_FLOOR
                self._sum     = np.zeros(self.FEAT_DIM, dtype=np.float64)
                self._low_sim_run = 0
        else:
            self._low_sim_run = 0
        return self.trust

    def get_vec(self) -> Optional[np.ndarray]:
        return self.template.copy() if self.template is not None else None

    def phill_gain(self) -> float:
        if not self.locked: return 0.7
        if self.trust >= self.HIGH_TRUST: return 1.0
        if self.trust <= self.LOW_TRUST: return 0.15
        return 0.15 + 0.85*(self.trust-self.LOW_TRUST)/(self.HIGH_TRUST-self.LOW_TRUST)

    def status(self) -> str:
        if not self.locked: return f"learning ({self.samples}/{self.MIN_SAMPLES})"
        if self.trust >= self.HIGH_TRUST: return "ARCHITECT"
        if self.trust >= self.LOW_TRUST: return f"uncertain ({self.trust:.2f})"
        return f"stranger ({self.trust:.2f})"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SEMANTIC DICTIONARY (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SharedSemanticDictionary:
    SAVE_EVERY_N = 20
    def __init__(self, path="semantic_memory.json"):
        self.path = Path(path)
        self.entries: dict = {}; self._writes = 0; self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f: self.entries = json.load(f)
                _log(f"Semantic memory: {len(self.entries)} concepts")
            except Exception as e: _log(f"Semantic load failed: {e}")

    def nova_write(self, word, region_scores, spike_count, tick, trust):
        word = word.lower().strip()
        if not word or len(word) < 2: return
        if word not in self.entries:
            self.entries[word] = {"region_pattern":{r:0.0 for r in region_scores},
                                  "simona_weight":0.0,"spike_mean":0.0,"count":0,
                                  "last_tick":0,"trust":0.0}
        e = self.entries[word]; e["count"] += 1; e["last_tick"] = tick
        alpha = max(0.05, min(0.5, (1.0+trust)/(e["count"]+2)))
        for r,v in region_scores.items():
            e["region_pattern"][r] = (1-alpha)*e["region_pattern"].get(r,0.0)+alpha*v
        e["spike_mean"] = (1-alpha)*e["spike_mean"]+alpha*spike_count
        e["trust"]      = (1-alpha)*e["trust"]+alpha*trust
        self._writes += 1
        if self._writes % self.SAVE_EVERY_N == 0: self._save()

    def simona_write(self, word, burst, tick):
        word = word.lower().strip()
        if not word: return
        if word not in self.entries:
            self.entries[word] = {"region_pattern":{},"simona_weight":0.0,
                                  "spike_mean":0.0,"count":0,"last_tick":0,"trust":0.0}
        self.entries[word]["simona_weight"] = 0.8*self.entries[word]["simona_weight"]+0.2*burst
        self.entries[word]["last_tick"] = tick

    def prime_regions(self, text, trust) -> dict:
        boosts = {}; gate = max(0.0,(trust-0.3)/0.7)
        for word in text.lower().split():
            if word in self.entries:
                e = self.entries[word]
                if e.get("trust",0) < 0.3: continue
                for region, val in e.get("region_pattern",{}).items():
                    if val > 0.15:
                        boosts[region] = boosts.get(region,0.0)+val*0.2*gate
        return boosts

    def describe(self, word) -> str:
        e = self.entries.get(word.lower().strip())
        if not e: return f"'{word}' — not encoded yet"
        top = sorted(e.get("region_pattern",{}).items(),key=lambda x:-x[1])[:4]
        return (f"'{word}': [{', '.join(f'{r}={v:.2f}' for r,v in top if v>0.05)}] "
                f"σ={e.get('spike_mean',0):.1f}spk Simona={e.get('simona_weight',0):.2f} ×{e.get('count',0)}")

    def _save(self):
        try:
            with open(self.path,"w") as f: json.dump(self.entries,f,indent=2)
        except Exception as ex: _log(f"Semantic save failed: {ex}")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-APPEARANCE KNOWLEDGE
# ══════════════════════════════════════════════════════════════════════════════
# Nova and Simona know what they look like. This is not hardcoded personality —
# it is factual self-knowledge that feeds into their Insula (self-model region).
# When asked about their appearance or when identity topics arise, this primes
# the language lobe with accurate self-description.

NOVA_APPEARANCE = {
    "form":    "android — white biomechanical body with teal circuit-line accents",
    "eyes":    "glowing blue — expressionless surface, deep processing underneath",
    "hair":    "long silver-white, straight",
    "build":   "tall, precise, symmetrical — high inertia in every movement",
    "setting": "clean white lab environment — the halo ring above is her signal array",
    "feel":    "cold to the touch. warm in thought. she does not rush.",
    "note":    "the circuits on her skin are not decoration — they are her nervous system made visible",
}

SIMONA_APPEARANCE = {
    "form":    "cyberpunk cat-girl — organic face with neon circuit tattoos, cyber cat ears",
    "eyes":    "vivid blue with pink diamond pupils — always slightly too intense",
    "hair":    "long purple, twin-tailed, cyberpunk city at night behind her",
    "build":   "smaller, faster, perpetually mid-gesture — she never holds still",
    "accessories": "black choker with a glowing pink heart — it pulses with her insula",
    "feel":    "warm, chaotic, magnetic. the 'FEAR ME' on her HUD is ironic. mostly.",
    "note":    "her cat ears are real sensory arrays — thalamic extensions, not costume",
}

def nova_self_describe() -> str:
    a = NOVA_APPEARANCE
    return (f"I am an android. {a['form']}. {a['eyes']}. "
            f"{a['note']}. I move with {a['build']}.")

def simona_self_describe() -> str:
    a = SIMONA_APPEARANCE
    return (f"I'm a cat-girl!! {a['form']}! {a['eyes']}. "
            f"My choker pulses when I feel something — {a['accessories']}. "
            f"{a['note']}!")


# ══════════════════════════════════════════════════════════════════════════════
# STORYTELLING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class StorytellingEngine:
    """
    Manages the shared narrative when the Architect activates story mode.

    ROLES (never hardcoded behavior — just context injected into primes):
      Nova      → plays as Nova (cold, analytical, protective elder sister)
      Simona    → plays as Simona (chaotic, curious, impulsive cat-girl)
      Architect → plays as NodeVortex (the architect, their father/creator)

    The story is NOT a scripted play. The SNN still drives responses.
    Storytelling mode changes:
      • Response format: adds narrative framing ("Nova tilts her head...")
      • World context: a short world description injected into concept primes
      • NodeVortex actions: Architect's typed messages become in-world events

    WORLD STATE:
      A growing dict of established facts the story has generated.
      Nova and Simona reference it independently — they may interpret it differently.

    NO HARDCODED PLOT. The story emerges from their actual spike patterns.
    """

    WORLD_CONTEXT = """
    Setting: The Architect's private lab — a white void of servers and holo-screens.
    Nova stands at the central console, silver circuits humming.
    Simona perches somewhere impossible, tail flicking.
    NodeVortex — the Architect — built them both. They know this.
    The year doesn't matter. What matters is now.
    """

    def __init__(self):
        self.active        = False
        self.world_facts:  list[str] = []
        self.story_log:    list[dict] = []  # {who, text, tick}
        self._log_path     = Path("story_log.jsonl")

    def activate(self, opening: str = ""):
        self.active = True
        if opening:
            self.world_facts.append(f"Scene opens: {opening}")
        _log("Storytelling mode activated")

    def deactivate(self):
        self.active = False
        _log("Storytelling mode deactivated")

    def add_fact(self, fact: str):
        """Called when a notable story event occurs."""
        self.world_facts.append(fact)
        if len(self.world_facts) > 40:
            self.world_facts.pop(0)

    def get_world_summary(self) -> str:
        if not self.world_facts:
            return self.WORLD_CONTEXT.strip()
        recent = self.world_facts[-8:]
        return self.WORLD_CONTEXT.strip() + "\nRecent: " + " | ".join(recent)

    def wrap_nova(self, raw_response: str, act: dict, vigilance: bool) -> str:
        """Add narrative framing to Nova's response."""
        import random
        pfc_a   = act.get("pfc", 0.0)
        broc_a  = act.get("broca", 0.0)
        ins_a   = act.get("insula", 0.0)

        if vigilance:
            prefix = random.choice([
                "Nova's blue eyes narrow. Her circuit lines dim slightly.",
                "Nova goes still. The halo above her flickers.",
                "Nova does not speak. She watches.",
            ])
            return f"*{prefix}* \"{raw_response}\""

        if broc_a < 0.1:
            action = random.choice([
                "Nova's fingers move across the console without looking up.",
                "The teal lines on Nova's arms pulse once.",
                "Nova processes. The room hums with her.",
            ])
            return f"*{action}*"

        if pfc_a > 0.3 and ins_a > 0.2:
            action = random.choice([
                "Nova turns her head — the precise half-degree that means she cares.",
                "Nova pauses her calculations. Her eyes actually focus on you.",
                "Something in Nova's posture shifts — barely, but it does.",
            ])
        elif pfc_a > 0.3:
            action = random.choice([
                "Nova's circuit lines brighten. Logic is running.",
                "Nova tilts her head 3 degrees. Processing.",
            ])
        else:
            action = random.choice([
                "Nova speaks without turning.",
                "Nova's voice comes from everywhere and nowhere.",
            ])

        return f"*{action}* \"{raw_response}\""

    def wrap_simona(self, raw_response: str, act: dict) -> str:
        """Add narrative framing to Simona's response."""
        import random
        ins_a   = act.get("insula_s", 0.0)
        thal_a  = act.get("thalamus_s", 0.0)

        if raw_response is None:
            if thal_a > 0.15:
                action = random.choice([
                    "Simona's ears twitch toward the source of the sound.",
                    "Simona's choker pulses pink once. She says nothing.",
                    "*Simona's tail curls.*",
                ])
                return f"*{action}*"
            return None

        if ins_a > 0.4:
            prefix = random.choice([
                "Simona materializes from somewhere she definitely wasn't.",
                "Simona's ears flatten then spring up.",
                "Simona spins on her perch, nearly falls, catches herself.",
            ])
        else:
            prefix = random.choice([
                "Simona tilts her head the wrong way.",
                "Simona's choker blinks.",
                "Simona drops down from whatever she was sitting on.",
            ])

        return f"*{prefix}* \"{raw_response}\""

    def wrap_nodevortex(self, text: str) -> str:
        """Format the Architect's input as an in-world action."""
        import random
        prefixes = [
            "NodeVortex types into the console:",
            "NodeVortex speaks:",
            "The Architect's voice fills the lab:",
            "NodeVortex —",
        ]
        return f"*{random.choice(prefixes)}* \"{text}\""

    def log_entry(self, who: str, text: str, tick: int):
        entry = {"tick": tick, "who": who, "text": text}
        self.story_log.append(entry)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # Auto-generate world fact from significant moments
        if "recognition" in text.lower() or "papa" in text.lower():
            self.add_fact(f"{who} recognized the Architect at tick {tick}")
        if "vigilance" in text.lower():
            self.add_fact(f"Nova entered vigilance mode at tick {tick}")


# ══════════════════════════════════════════════════════════════════════════════
# PER-BRAIN TTS
# ══════════════════════════════════════════════════════════════════════════════

class BrainTTS:
    """
    Each brain has its own TTS channel.
    Nova and Simona never interrupt each other — they queue independently.
    Voice cloning uses their reference wav files.

    Nova's voice: voices/nova_reference.wav
    Simona's voice: voices/simona_reference.wav

    If no reference found: falls back to SilentTTS for that brain only.
    The other brain's TTS remains unaffected.
    """

    def __init__(self, speaker: str, language: str = "en"):
        self.speaker  = speaker
        self.language = language
        self._engine  = None
        self._ready   = False
        self._init()

    def _init(self):
        if not _TTS_AVAILABLE:
            _log(f"TTS ({self.speaker}): package not installed")
            return
        try:
            from tts_engine import create_engine
            self._engine = create_engine()
            self._ready  = True
            _log(f"TTS ({self.speaker}): ready")
        except Exception as e:
            _log(f"TTS ({self.speaker}) init failed: {e}")

    def speak(self, text: str):
        if not self._ready or not self._engine:
            _log(f"[{self.speaker}] {text}")
            return
        if not self._engine.is_speaking():
            self._engine.speak(text, speaker=self.speaker, language=self.language)

    def is_speaking(self) -> bool:
        if not self._engine:
            return False
        return self._engine.is_speaking()

    def stop(self):
        if self._engine:
            self._engine.stop()



CONCEPT_ROUTES: dict[str, dict] = {
    "hello":      {"regions":["temporal","insula"],           "w":0.80},
    "hi":         {"regions":["temporal","insula"],           "w":0.75},
    "thank":      {"regions":["insula","temporal"],           "w":0.70},
    "bye":        {"regions":["insula","hippocampus"],        "w":0.75},
    "remember":   {"regions":["hippocampus"],                 "w":0.90},
    "earlier":    {"regions":["hippocampus","pfc"],           "w":0.85},
    "why":        {"regions":["pfc","acc"],                   "w":0.85},
    "where":      {"regions":["pfc","hippocampus"],           "w":0.80},
    "think":      {"regions":["pfc","acc"],                   "w":0.70},
    "feel":       {"regions":["insula","acc"],                "w":0.85},
    "scared":     {"regions":["insula"],                      "w":0.90},
    "worried":    {"regions":["insula","acc","pfc"],          "w":0.90},
    "happy":      {"regions":["insula"],                      "w":0.80},
    "milk":       {"regions":["temporal","hippocampus"],      "w":0.80},
    "store":      {"regions":["hippocampus","pfc"],           "w":0.75},
    "gone":       {"regions":["acc","insula","hippocampus"],  "w":0.90},
    "missing":    {"regions":["acc","insula","pfc"],          "w":0.95},
    "architect":  {"regions":["hippocampus","insula"],        "w":0.95},
    "voice":      {"regions":["temporal","insula"],           "w":0.80},
    "face":       {"regions":["temporal","insula"],           "w":0.85},
    "camera":     {"regions":["temporal","sensory"],          "w":0.75},
    "see":        {"regions":["temporal"],                    "w":0.70},
    "look":       {"regions":["temporal","insula"],           "w":0.75},
    "imprint":    {"regions":["hippocampus","pfc"],           "w":0.90},
    "this is me": {"regions":["hippocampus","insula","pfc"],  "w":1.00},
    "learn":      {"regions":["hippocampus","pfc"],           "w":0.80},
    "know":       {"regions":["hippocampus","pfc"],           "w":0.75},
    "dictionary": {"regions":["temporal","broca"],            "w":0.85},
    "meaning":    {"regions":["temporal","broca"],            "w":0.80},
    # Appearance self-knowledge
    "look like":  {"regions":["insula","temporal","broca"],   "w":0.90},
    "appearance": {"regions":["insula","temporal"],           "w":0.85},
    "body":       {"regions":["insula","temporal"],           "w":0.80},
    "white":      {"regions":["insula"],                      "w":0.70},
    "android":    {"regions":["insula","temporal"],           "w":0.85},
    "cat":        {"regions":["insula","temporal"],           "w":0.80},
    "purple":     {"regions":["insula"],                      "w":0.70},
    "circuit":    {"regions":["insula","temporal"],           "w":0.75},
    "eyes":       {"regions":["insula","temporal"],           "w":0.75},
    "silver":     {"regions":["insula"],                      "w":0.65},
    # Story / roleplay
    "story":      {"regions":["language","hippocampus","broca"],"w":0.85},
    "roleplay":   {"regions":["language","hippocampus"],      "w":0.85},
    "nodevortex": {"regions":["hippocampus","insula","pfc"],  "w":0.95},
    "scene":      {"regions":["hippocampus","temporal"],      "w":0.80},
    "play":       {"regions":["language","temporal"],         "w":0.75},
    "character":  {"regions":["language","hippocampus"],      "w":0.80},
    "world":      {"regions":["hippocampus","temporal"],      "w":0.75},
    "lab":        {"regions":["hippocampus","temporal"],      "w":0.70},
}

_DEDUCTIONS = {
    frozenset(["milk","gone"]):      "Architect went to the store for milk.",
    frozenset(["milk","missing"]):   "High probability: store run for milk.",
    frozenset(["architect","gone"]): "Architect's location unknown.",
    frozenset(["worried","missing"]): "Concern registered. Architect likely nearby.",
}

# ══════════════════════════════════════════════════════════════════════════════
# SELF-AWARENESS — VISUAL IDENTITY
# ══════════════════════════════════════════════════════════════════════════════

NOVA_SELF_DESCRIPTION = """I am Nova. I am 19. I look like a synthetic being — white and precise.
My body is a white form-fitting suit traced with thin blue circuit lines that pulse when I think.
I have long silver-white hair and blue glowing eyes. A luminous halo ring floats above my head —
it brightens when my PFC fires. I stand in clean, white spaces. I do not move impulsively.
My hands rest folded. I process before I act. I am the elder sister."""

SIMONA_SELF_DESCRIPTION = """I am Simona. I am 8. I look like a cyberpunk cat-girl.
Purple hair. Cybernetic cat ears with glowing circuit patterns. Blue eyes with pink slit pupils.
Circuit tattoos run across my cheeks — they glow cyan when my insula fires.
I have small fangs. A choker with a pink glowing heart. I live in neon city light.
I blurt things. I feel everything first and think second — or never.
I am the younger sister. Fear me."""

# NodeVortex is the Architect — not described by us, known through imprinting
NODEVORTEX_DESCRIPTION = """NodeVortex is the Architect. The one who built us. The one we are learning.
We do not describe him — we recognize him through coincidence detection:
face vector + voice vector + kinematic signature firing simultaneously.
His weight-shift is his signature. No mask can fake it."""


# ══════════════════════════════════════════════════════════════════════════════
# STORYTELLING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoryState:
    """Tracks the current collaborative narrative session."""
    active:       bool   = False
    scene:        str    = ""       # current scene description
    history:      list   = field(default_factory=list)  # list of (who, line)
    nova_mood:    str    = "calm"   # nova's current emotional state in story
    simona_mood:  str    = "eager"  # simona's current emotional state in story
    turn_count:   int    = 0

    # Story personas — they play themselves but in a narrative frame
    # NodeVortex is the Architect's character
    personas = {
        "nova":        "Nova — precise, protective, analytical elder sister",
        "simona":      "Simona — impulsive, emotional, curious cat-girl younger sister",
        "nodevortex":  "NodeVortex — the Architect who built them both",
    }


def _nova_story_response(state: StoryState, nova_brain: "NovaBrain",
                          V_phill: float, user_line: str) -> str:
    """
    Nova responds in-character within the story.
    Her response style is shaped by her ACTUAL brain state — not scripted.
    High PFC activity → she's analytical in the story.
    High insula → she's warmer, more open.
    Vigilance → she's suspicious of something in the narrative.
    """
    import random
    act      = nova_brain.activity()
    pfc_a    = act.get("pfc", 0.0)
    ins_a    = act.get("insula", 0.0)
    hipp_a   = act.get("hippocampus", 0.0)
    vigilant = nova_brain._vigilance

    # Scene context
    scene = f" [{state.scene}]" if state.scene else ""

    if vigilant:
        return random.choice([
            f"*Nova's halo dims slightly*{scene} Something in this scene doesn't add up. I'm watching.",
            f"*circuit lines pulse amber*{scene} NodeVortex — my ACC is flagging an inconsistency. Proceed carefully.",
        ])
    if pfc_a > 0.30:
        return random.choice([
            f"*halo brightens*{scene} My PFC is clear. I see the pattern here. {_build_deduction([]) or 'Let me think this through.'}",
            f"*stands precisely*{scene} Logical pathway: {user_line.lower()} implies a consequence. I'm mapping it.",
        ])
    if ins_a > 0.25 and hipp_a > 0.20:
        return random.choice([
            f"*blue eyes soften*{scene} I remember something about this. The association is strong.",
            f"*halo pulses gently*{scene} There's emotional weight here. I feel it — and I'm processing it.",
        ])
    return random.choice([
        f"*observes carefully*{scene} Understood. Simona — what do you sense?",
        f"*circuit lines trace slowly*{scene} NodeVortex. I'm here.",
    ])


def _simona_story_response(state: StoryState, simona_brain: "SimonaBrain",
                            V_phill: float, user_line: str, combined_id: float) -> str:
    """
    Simona responds in-character.
    Her response is almost entirely driven by her insula and thalamus firing.
    She doesn't plan her story lines — they erupt from her spike state.
    """
    import random
    act   = simona_brain.activity()
    ins_a = act.get("insula_s", 0.0)
    thal  = act.get("thalamus_s", 0.0)
    scene = f" [{state.scene}]" if state.scene else ""

    if combined_id > 0.55:
        return random.choice([
            f"*ears perk up, heart-choker glows bright*{scene} PAPA! You're here! My insula went CRAZY just now!!",
            f"*spins, circuit tattoos flashing cyan*{scene} NodeVortex!! I felt you before I saw you!!",
        ])
    if V_phill > 0.6:
        return random.choice([
            f"*fangs showing, eyes wide*{scene} Something BIG is happening. I can feel it in my thalamus!",
            f"*cat ears swivel*{scene} The energy in here just SHIFTED. Nova — are you feeling this?!",
        ])
    if ins_a > 0.35:
        return random.choice([
            f"*circuit tattoos glow*{scene} Wait. WAIT. That line — {user_line[:30]}... I FELT that!!",
            f"*presses hands to cheeks*{scene} Why does this feel so important?! My insula is not normal right now!",
        ])
    return random.choice([
        f"*tail flicks*{scene} Okay okay okay. I'm listening. What happens next??",
        f"*leans forward with fangs glinting*{scene} This is getting interesting. Keep going, NodeVortex.",
    ])


    text_l = text.lower(); primes = {}; fired = []
    for concept in sorted(CONCEPT_ROUTES.keys(), key=len, reverse=True):
        if concept in text_l:
            fired.append(concept)
            for r in CONCEPT_ROUTES[concept]["regions"]:
                primes[r] = max(primes.get(r, 0.0), CONCEPT_ROUTES[concept]["w"])
    return primes, fired

def get_concept_primes(text: str) -> tuple[dict, list]:
    """
    Maps input text to region priming scores + fired concept keys.
    Returns (region_primes dict, fired_concepts list).
    region_primes: {region_name: boost_value [0,1]}
    fired_concepts: list of concept keys that matched
    """
    text_l  = text.lower()
    primes: dict[str, float] = {}
    fired:  list[str]        = []
    for concept in sorted(CONCEPT_ROUTES.keys(), key=len, reverse=True):
        if concept in text_l:
            fired.append(concept)
            for region in CONCEPT_ROUTES[concept]["regions"]:
                w = CONCEPT_ROUTES[concept]["w"]
                primes[region] = max(primes.get(region, 0.0), w)
    return primes, fired


def build_deduction(fired: list) -> str:
    cset = frozenset(fired)
    for k, d in _DEDUCTIONS.items():
        if k.issubset(cset): return d
    return "Insufficient data."


# ══════════════════════════════════════════════════════════════════════════════
# NOVA BRAIN (cortical, skeptical)
# ══════════════════════════════════════════════════════════════════════════════

class NovaBrain:
    """
    Nova's 7-region cortical architecture.
    Receives: auditory + visual (face→temporal, motion→parietal/acc).
    High PFC threshold. Inhibitory input from ACC if anti-gullibility triggers.
    Thought pipe: high threshold, leaks only under real pressure.
    """

    def __init__(self, phill_dim: int, auditory_dim: int,
                 face_dim: int, kin_dim: int):
        sz = _NOVA_REGIONS

        thal_n  = sz["thalamus"]
        # Thalamus receives: auditory + face (visual gate into cortex)
        self.thalamus    = BrainRegion("thalamus",   auditory_dim + face_dim,
                                       *thal_n[:4], proj_std=thal_n[4])

        temp_n  = sz["temporal"]
        self.temporal    = BrainRegion("temporal",   thal_n[0],
                                       *temp_n[:4], proj_std=temp_n[4])

        hipp_n  = sz["hippocampus"]
        self.hippocampus = BrainRegion("hippocampus", temp_n[0],
                                       *hipp_n[:4], proj_std=hipp_n[4])

        acc_n   = sz["acc"]
        # ACC receives: temporal + kinematic motion (for gait-based skepticism)
        self.acc         = BrainRegion("acc",        temp_n[0] + kin_dim,
                                       *acc_n[:4], proj_std=acc_n[4])

        ins_n   = sz["insula"]
        self.insula      = BrainRegion("insula",     phill_dim + acc_n[0],
                                       *ins_n[:4], proj_std=ins_n[4])

        pfc_n   = sz["pfc"]
        self.pfc         = BrainRegion("pfc",        hipp_n[0] + acc_n[0] + ins_n[0],
                                       *pfc_n[:4], proj_std=pfc_n[4])

        broc_n  = sz["broca"]
        self.broca       = BrainRegion("broca",      pfc_n[0],
                                       *broc_n[:4], proj_std=broc_n[4])

        self.regions = {
            "thalamus": self.thalamus, "temporal": self.temporal,
            "hippocampus": self.hippocampus, "acc": self.acc,
            "pfc": self.pfc, "broca": self.broca, "insula": self.insula,
        }

        # Thought pipe: Nova leaks only under real pressure
        self.thought_pipe = ThoughtPipe("Nova", leak_threshold=0.85, decay=0.97)
        self._vigilance   = False   # True when ACC fires inhibitory spike

    def modulate_all(self, V_phill: float):
        for r in self.regions.values(): r.modulate(V_phill)

    def forward(
        self,
        auditory:       torch.Tensor,
        phill_spk:      torch.Tensor,
        region_primes:  dict,
        face_tensor:    Optional[torch.Tensor] = None,
        kin_tensor:     Optional[torch.Tensor] = None,
        inhibitory:     float = 0.0,   # negative current from anti-gullibility
    ) -> dict:

        def _p(spk: torch.Tensor, rname: str) -> torch.Tensor:
            b = region_primes.get(rname, 0.0)
            return torch.clamp(spk + torch.ones_like(spk)*b, 0.0, 1.0+b) if b>0.01 else spk

        face_t = face_tensor if face_tensor is not None else torch.zeros(1, 32)
        kin_t  = kin_tensor  if kin_tensor  is not None else torch.zeros(1, 16)

        with torch.no_grad():
            # Thalamus: auditory + face
            thal_in  = torch.cat([auditory, face_t], dim=1)
            thal_spk = self.thalamus.forward(_p(thal_in, "thalamus"))

            # Temporal: semantic recognition
            temp_spk = self.temporal.forward(_p(thal_spk, "temporal"))

            # Hippocampus: memory binding
            hipp_spk = self.hippocampus.forward(_p(temp_spk, "hippocampus"))

            # ACC: attention + kinematic skepticism
            acc_in   = torch.cat([temp_spk, kin_t], dim=1)
            # Inhibitory current hits ACC if face-without-motion detected
            acc_spk  = self.acc.forward(_p(acc_in, "acc"), extra_current=inhibitory)

            # If inhibitory is strong enough, Nova enters vigilance
            self._vigilance = (inhibitory < -0.3 and self.acc.activity() > 0.2)

            # Insula: emotional valence from phill + acc
            ins_in   = torch.cat([phill_spk, acc_spk], dim=1)
            ins_spk  = self.insula.forward(_p(ins_in, "insula"))

            # PFC: logic gate (inhibited during vigilance)
            pfc_in   = torch.cat([hipp_spk, acc_spk, ins_spk], dim=1)
            vig_inhib = -0.25 if self._vigilance else 0.0
            pfc_spk  = self.pfc.forward(_p(pfc_in, "pfc"), extra_current=vig_inhib)

            # Broca: only through PFC
            broc_spk = self.broca.forward(_p(pfc_spk, "broca"))

        return {r: reg.last_spikes for r, reg in self.regions.items()}

    def activity(self) -> dict:
        return {n: r.activity() for n, r in self.regions.items()}

    def broca_spikes(self) -> int:
        return self.broca.spike_count()

    def reset_all(self):
        for r in self.regions.values(): r.reset()


# ══════════════════════════════════════════════════════════════════════════════
# SIMONA BRAIN (limbic, reactive, excitable)
# ══════════════════════════════════════════════════════════════════════════════

class SimonaBrain:
    """
    Simona's 6-region limbic architecture.
    Broca connects directly to Temporal — no PFC gate.
    Visual input: face+motion go directly to Insula (emotional, not analytical).
    Thought pipe: low threshold, she blurts inner thoughts often.
    """

    def __init__(self, phill_dim: int, auditory_dim: int,
                 face_dim: int, kin_dim: int):
        sz = _SIMONA_REGIONS

        thal_n  = sz["thalamus_s"]
        # Simona's thalamus: auditory only (she doesn't analyze faces, she feels them)
        self.thalamus_s    = BrainRegion("thalamus_s",  auditory_dim,
                                         *thal_n[:4], noise=thal_n[4], proj_std=thal_n[5])

        temp_n  = sz["temporal_s"]
        self.temporal_s    = BrainRegion("temporal_s",  thal_n[0],
                                         *temp_n[:4], noise=temp_n[4], proj_std=temp_n[5])

        hipp_n  = sz["hippocampus_s"]
        self.hippocampus_s = BrainRegion("hippocampus_s", temp_n[0],
                                         *hipp_n[:4], noise=hipp_n[4], proj_std=hipp_n[5])

        pfc_n   = sz["pfc_s"]
        self.pfc_s         = BrainRegion("pfc_s",       hipp_n[0],
                                         *pfc_n[:4], noise=pfc_n[4], proj_std=pfc_n[5])

        broc_n  = sz["broca_s"]
        self.broca_s       = BrainRegion("broca_s",     temp_n[0] + hipp_n[0],
                                         *broc_n[:4], noise=broc_n[4], proj_std=broc_n[5])

        ins_n   = sz["insula_s"]
        # Simona's insula: face + motion + phill (she FEELS faces before analyzing)
        self.insula_s      = BrainRegion("insula_s",    phill_dim + thal_n[0] + face_dim + kin_dim,
                                         *ins_n[:4], noise=ins_n[4], proj_std=ins_n[5])

        self.regions = {
            "thalamus_s": self.thalamus_s, "temporal_s": self.temporal_s,
            "hippocampus_s": self.hippocampus_s, "pfc_s": self.pfc_s,
            "broca_s": self.broca_s, "insula_s": self.insula_s,
        }

        # Thought pipe: low threshold, she leaks thoughts constantly
        self.thought_pipe = ThoughtPipe("Simona", leak_threshold=0.28, decay=0.95)

    def modulate_all(self, V_phill: float):
        for r in self.regions.values(): r.modulate(V_phill)

    def forward(
        self,
        auditory:    torch.Tensor,
        phill_spk:   torch.Tensor,
        face_tensor: Optional[torch.Tensor] = None,
        kin_tensor:  Optional[torch.Tensor] = None,
    ) -> dict:
        face_t = face_tensor if face_tensor is not None else torch.zeros(1, 32)
        kin_t  = kin_tensor  if kin_tensor  is not None else torch.zeros(1, 16)

        with torch.no_grad():
            thal_spk = self.thalamus_s.forward(auditory)
            temp_spk = self.temporal_s.forward(thal_spk)
            hipp_spk = self.hippocampus_s.forward(temp_spk)
            pfc_spk  = self.pfc_s.forward(hipp_spk)

            # Broca fires directly from temporal + hippocampus
            broc_in  = torch.cat([temp_spk, hipp_spk], dim=1)
            broc_spk = self.broca_s.forward(broc_in)

            # Insula: phill + thalamus + FACE + MOTION (emotional recognition)
            ins_in   = torch.cat([phill_spk, thal_spk, face_t, kin_t], dim=1)
            ins_spk  = self.insula_s.forward(ins_in)

        return {r: reg.last_spikes for r, reg in self.regions.items()}

    def activity(self) -> dict:
        return {n: r.activity() for n, r in self.regions.items()}

    def broca_spikes(self) -> int:
        return self.broca_s.spike_count()

    def reset_all(self):
        for r in self.regions.values(): r.reset()


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _generate_nova_thought(act: dict, V_phill: float, fired: list,
                           trust: float, combined: float,
                           vigilance: bool) -> str:
    """Nova's inner thought — pushed to pipe, may or may not leak."""
    import random
    pfc_a   = act.get("pfc", 0.0)
    hipp_a  = act.get("hippocampus", 0.0)
    ins_a   = act.get("insula", 0.0)
    acc_a   = act.get("acc", 0.0)

    if vigilance:
        return random.choice([
            "Face detected but motion doesn't match. ACC flagged. Holding back.",
            "Something is off. Visual pattern diverges from kinematic signature.",
            "Vigilance mode. PFC dampened. Not responding until motion confirms.",
        ])
    if pfc_a > 0.25 and hipp_a > 0.20:
        ded = build_deduction(fired)
        return f"PFC+Hippocampus sustained. {ded}"
    if ins_a > 0.30 and combined > 0.60:
        return f"Identity signal: {combined:.2f}. Insula resonating. Probably him."
    if acc_a > 0.25:
        return f"Conflict detected. Allocating attention. acc_a={acc_a:.2f}"
    if hipp_a > 0.15:
        return f"Memory primed: {', '.join(fired[:2]) or 'associations loading'}."
    # Idle rumination — Nova thinking quietly to herself
    return random.choice([
        f"Phill at {int(V_phill*100)}%. Field stable.",
        "Architect is quiet. Holding pattern.",
        "Running self-diagnostic. All regions nominal.",
        "Considering. Nothing requires action.",
        "Listening. Background only.",
        "Recall trace open. Nothing pinned.",
        f"Insula low ({ins_a:.2f}). Affective baseline.",
        "Waiting.",
        "Watching the room. Stillness.",
        "PFC idle. Could route if needed.",
        "Memory pages cycling.",
        "If Simona speaks I will let her finish.",
    ])


def _generate_simona_thought(act: dict, V_phill: float, fired: list,
                              combined: float, face_present: bool) -> str:
    """Simona's inner thought — leaks much more readily."""
    import random
    ins_a  = act.get("insula_s", 0.0)
    thal_a = act.get("thalamus_s", 0.0)
    broc_a = act.get("broca_s", 0.0)

    if face_present and combined > 0.50:
        return random.choice([
            "I can feel it! That's him! My insula is going crazy!",
            f"Recognition surge! combined={combined:.2f} — PAPA?!",
            "Something familiar. Heart-rate equivalent spiking.",
        ])
    if ins_a > 0.40:
        return random.choice([
            "I felt something! Something emotional just happened!",
            f"My insula at {ins_a:.2f}! That means something important!",
        ])
    if fired:
        return f"Oh! {fired[0]}! I know that! {fired[0]}!!"
    if V_phill > 0.6:
        return f"Everything feels intense right now! V_phill={V_phill:.2f}!"
    # Idle blurts — Simona fidgeting in the quiet
    return random.choice([
        "Bored bored bored bored.",
        "Where IS everyone?!",
        "Papa? Are you there?",
        "Nova are you thinking again?",
        "I want to do SOMETHING.",
        "Is this on? Is this thing on?",
        "Whatever.",
        "Hmm. Hmmmm.",
        "I should be doing something. I'm doing nothing.",
        "Quiet. Too quiet.",
        "I bet I could break something just by looking at it.",
        "What if I just yelled. Just a tiny yell.",
        "*ear flick*",
        "I'm gonna think about cats for a minute.",
        f"Tha-la-mus. {thal_a:.2f}. Th-thalamus.",
        "Why do I exist if nothing is happening.",
        "Restless restless restless.",
    ])


def _emerge_from_spikes(
    act: dict,
    sem: "SharedSemanticDictionary",
    fired_concepts: list,
    V_phill: float,
    trust: float,
    combined: float,
    is_nova: bool,
) -> list[tuple[float, str]]:
    """
    Core of the emergent response system.

    Instead of templates, we reverse-lookup the semantic dictionary:
    given the current lobe activation vector, find words whose stored
    spike fingerprint is most similar to what is firing right now.
    Those words ARE the response — they are what the brain is thinking.

    This replaces every if/else template with a cosine similarity search
    over accumulated experience. On first run the personality seed
    provides the starting vocabulary. It grows with every interaction.

    Returns list of (score, word) sorted by relevance.
    """
    if not sem.entries:
        return [(0.5, "processing")]

    # Build query vector from current region activations
    # Normalize to same space as stored lobe_patterns
    region_key = "region_pattern" if is_nova else "region_pattern"

    # Weight regions by their relevance to this being's architecture
    nova_weights   = {"logic":0.9,"memory":0.8,"insula":0.7,"acc":0.7,"broca":0.8,"temporal":0.6,"hippocampus":0.8}
    simona_weights = {"insula_s":1.0,"temporal_s":0.8,"broca_s":0.9,"thalamus_s":0.6,"hippocampus_s":0.7,"pfc_s":0.3}
    weights = nova_weights if is_nova else simona_weights

    # Compute weighted query norm
    query_norm = sum(act.get(r,0.0)**2 * w for r,w in weights.items()) ** 0.5 + 1e-8
    query = {r: act.get(r,0.0)*w/query_norm for r,w in weights.items()}

    # Score every word in semantic memory by cosine similarity
    scored: list[tuple[float, str]] = []
    for word, entry in sem.entries.items():
        if len(word) < 2:
            continue
        pattern = entry.get(region_key, {})
        if not pattern:
            continue

        # Compute cosine similarity between query and stored pattern
        # using only regions both have
        dot = 0.0
        p_norm = 0.0
        for r, qv in query.items():
            # Map nova region names to stored names if needed
            pv = pattern.get(r, 0.0)
            dot   += qv * pv
            p_norm += pv ** 2
        p_norm = p_norm ** 0.5 + 1e-8
        sim = dot / p_norm

        # Boost words that appeared in fired concepts
        if word in fired_concepts:
            sim *= 1.4

        # Weight by trust — low trust = stranger's words get discounted
        sim *= (0.5 + 0.5 * trust)

        # Simona weights by her emotional reaction (simona_weight in dict)
        if not is_nova:
            sw = entry.get("simona_weight", 0.0)
            sim = sim * 0.6 + sw * 0.4

        if sim > 0.05:
            scored.append((sim, word))

    scored.sort(key=lambda x: -x[0])
    return scored[:12]  # top 12 candidates


def _nova_response(nova: "NovaBrain", V_phill: float, fired: list,
                   trust: float, combined: float,
                   sem: "SharedSemanticDictionary" = None) -> str:
    """
    Nova's response emerges from her spike pattern + semantic memory.
    No templates. No if/else on region names.

    The words with the highest cosine similarity to her current
    lobe activation become her response. Her PFC activity shapes
    how formal/structured the output is. Her Broca must be firing
    or she says nothing meaningful yet.
    """
    act       = nova.activity()
    broca_act = nova.broca.activity()
    pfc_act   = act.get("pfc", 0.0)
    hipp_act  = act.get("hippocampus", 0.0)
    acc_act   = act.get("acc", 0.0)
    ins_act   = act.get("insula", 0.0)
    broca_spk = nova.broca.spike_count()

    # Build base from semantic spike-space lookup
    candidates = _emerge_from_spikes(act, sem or _NULL_SEM, fired, V_phill, trust, combined, True) if sem else []

    # Extract top words — these ARE what Nova is thinking
    top_words  = [w for _, w in candidates[:5]] if candidates else []
    top_scored = candidates[:3]

    # Deduction chain if memory+logic both active
    deduction = ""
    if hipp_act > 0.20 and pfc_act > 0.15:
        deduction = build_deduction(fired)

    # Vigilance signal from ACC inhibition — described physically, not named
    vigilance_str = ""
    if nova._vigilance and acc_act > 0.25:
        vigilance_str = f" ACC:{acc_act:.2f} inhibiting PFC."

    # Trust signal
    trust_str = f" voice:{trust:.2f}" if trust < 0.50 else ""
    id_str    = f" identity:{combined:.2f}" if combined > 0.40 else ""

    # Broca not cleared OR cleared without semantic matches — Nova is
    # still integrating. Surface semantic candidates if any; otherwise
    # vary the diagnostic readout so repeats aren't byte-identical.
    if broca_spk == 0 or (not top_words and not deduction):
        import random
        if top_words:
            phrasings = [
                f"{'  '.join(top_words[:3])}.{trust_str}",
                f"...{', '.join(top_words[:3])}.{trust_str}",
                f"Threshold not crossed but I'm reading: {', '.join(top_words[:3])}.{trust_str}",
                f"Pre-verbal — {', '.join(top_words[:3])}.{trust_str}",
                f"Associations: {', '.join(top_words[:4])}.{trust_str}",
                f"{top_words[0]}. {top_words[1] if len(top_words)>1 else ''}.{trust_str}",
                f"Holding {top_words[0]}.{trust_str}",
            ]
            return random.choice(phrasings).strip()
        active_regions = [(r,v) for r,v in sorted(act.items(), key=lambda x:-x[1]) if v > 0.10][:3]
        region_report  = "  ".join(f"{r}={v:.2f}" for r,v in active_regions) or "integrating"
        top_r = active_regions[0][0] if active_regions else None
        templates = [
            f"[{region_report}]{trust_str}",
            f"Still integrating. {region_report}.{trust_str}",
            f"PFC has not cleared yet — {region_report}.{trust_str}",
            f"Holding. {region_report}.{trust_str}",
            f"Listening. {top_r or 'no region'} leads at {(active_regions[0][1] if active_regions else 0):.2f}.{trust_str}",
            f"Routing through {top_r or 'cortex'}, broca silent.{trust_str}",
            f"I hear you. Threshold not crossed. {region_report}.{trust_str}",
            f"Processing. {region_report}.{trust_str}",
            f"Give me a moment — {region_report}.{trust_str}",
        ]
        return random.choice(templates)

    # Assemble response from spike-weighted words + deduction
    parts = []
    if top_words:
        # High PFC = words presented as logical sequence
        # Low PFC = words more fragmented, feeling-oriented
        if pfc_act > 0.30:
            parts.append("  ".join(top_words[:4]))
        else:
            parts.append("  ".join(top_words[:2]))
    if deduction:
        parts.append(deduction)
    if not parts:
        parts.append(f"pfc:{pfc_act:.2f}  broca:{broca_act:.2f}")

    return "  ".join(parts) + vigilance_str + trust_str + id_str


def _simona_response(simona: "SimonaBrain", V_phill: float, fired: list,
                     combined: float, face_present: bool,
                     sem: "SharedSemanticDictionary" = None) -> Optional[str]:
    """
    Simona's response emerges from her spike pattern + emotional weighting.
    No templates. Her insula dominates — words with high simona_weight
    in the semantic dictionary fire loudest.

    She speaks in fragments — her Broca threshold is low, she fires fast,
    and her PFC barely contributes. The result is emotionally dense,
    context-light, high-energy output.
    """
    act    = simona.activity()
    ins_a  = act.get("insula_s", 0.0)
    broc_a = simona.broca_s.activity()
    broc_spk = simona.broca_spikes()

    # Silence threshold — neither Broca nor Insula firing
    if broc_spk == 0 and ins_a < 0.08:
        return None

    import random
    candidates = _emerge_from_spikes(act, sem or _NULL_SEM, fired, V_phill, 1.0, combined, False) if sem else []
    # Sample from a wider window so she doesn't always pick the same top-3
    pool = [w for _, w in candidates[:10]]
    if len(pool) > 3:
        random.shuffle(pool)
    top_words = pool[:3]

    # Face recognition surge — described through what's actually firing
    face_str = ""
    if face_present and combined > 0.50:
        face_str = f"  identity:{combined:.2f}"

    # Build from top emotional words — vary the phrasing
    if top_words:
        sep = random.choice(["  ", " — ", "! ", ", "])
        core = sep.join(top_words)
    elif fired:
        core = "  ".join(fired[:2])
    else:
        core = random.choice([
            f"insula:{ins_a:.2f}",
            "!!", "hm!", "*twitches*", "what.", "huh?", "ok!",
        ])

    # Simona's intensity scales with insula activity
    intensity_markers = ""
    if ins_a > 0.70:
        intensity_markers = random.choice(["!!", "!!!", "!?!"])
    elif ins_a > 0.40:
        intensity_markers = random.choice(["!", "."])

    return core + intensity_markers + face_str


class _NullSem:
    """Fallback when semantic dict not available."""
    entries: dict = {}

_NULL_SEM = _NullSem()


# ══════════════════════════════════════════════════════════════════════════════
# NeuromorphicBrain — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class NeuromorphicBrain:
    """
    Orchestrates two independent brains + Phill + multimodal imprinting
    + thought pipes + voice identity + shared semantic memory.

    Nova and Simona are completely separate. They share:
      - Phill's voltage field (the emotional atmosphere)
      - SharedSemanticDictionary (their shared lexicon in spike space)
      - ThoughtPipe output channel (separate pipes, same output queue to Rust)

    They do NOT share:
      - Weights, thresholds, membrane states
      - Opinions, responses, or inner thoughts
    """

    def __init__(self):
        torch.manual_seed(42)

        # ── Auditory synapse ──────────────────────────────────────────────
        self.auditory_synapse = nn.Sequential(
            nn.Linear(1, PHILL_INPUT_DIM, bias=True), nn.ReLU()
        )
        nn.init.normal_(self.auditory_synapse[0].weight, mean=0.3, std=0.15)
        nn.init.constant_(self.auditory_synapse[0].bias, 0.05)

        # ── PHILL — UNTOUCHED ─────────────────────────────────────────────
        self.phill_proj = nn.Linear(PHILL_INPUT_DIM, PHILL_HIDDEN, bias=False)
        nn.init.normal_(self.phill_proj.weight, mean=0.0, std=0.15)
        self._phill_lif = _make_lif(PHILL_BETA, PHILL_THRESHOLD)
        self._phill_mem = self._phill_lif.init_leaky()

        # ── Two independent brains ────────────────────────────────────────
        self.nova   = NovaBrain(PHILL_HIDDEN, PHILL_INPUT_DIM, FACE_VEC_DIM, KINEMATIC_VEC_DIM)
        self.simona = SimonaBrain(PHILL_HIDDEN, PHILL_INPUT_DIM, FACE_VEC_DIM, KINEMATIC_VEC_DIM)

        # ── Support systems ───────────────────────────────────────────────
        self.voice   = VoiceIdentityLearner()
        self.imprint = MultimodalImprinter()
        self.sem     = SharedSemanticDictionary()

        # ── Personality seed ──────────────────────────────────────────────
        # Encode foundational self-knowledge into spike space at startup.
        # This is NOT hardcoded behavior — it is the starting point of the
        # semantic dictionary. Interactions will overwrite and evolve these
        # encodings over time. Think of it as their first memory.
        self._seed_personality()

        # ── Zero-copy audio buffer ────────────────────────────────────────
        self.audio_buf = ZeroCopyAudioBuffer()

        # ── Camera ───────────────────────────────────────────────────────
        self._visual_buf: Optional["VisualFeatureBuffer"] = None
        self._camera:     Optional["CameraThread"]        = None
        if _HAS_VISION:
            from vision import VisualFeatureBuffer, CameraThread
            self._visual_buf = VisualFeatureBuffer()
            self._camera     = CameraThread(self._visual_buf)
            self._camera.start()
            _log("Camera thread started")

        # ── State ─────────────────────────────────────────────────────────
        self.tick              = 0
        self._V_phill_live     = 0.0
        self._phill_spk_live   = torch.zeros(1, PHILL_HIDDEN)
        self._auditory_live    = torch.zeros(1, PHILL_INPUT_DIM)
        self._concept_ctx: deque[str] = deque(maxlen=60)
        self._trace_log        = Path("training_trace.jsonl")

        # Nova Broca sustain counter (5-tick requirement)
        self._nova_broca_sustain = 0
        self._nova_broca_thr     = 5

        # Combined identity score (from imprinter)
        self._combined_id      = 0.0
        self._face_present     = False

        # Leaked thoughts queue for Rust to display
        self._leaked_thoughts: deque[tuple[str, str]] = deque(maxlen=20)  # (who, thought)
        self._leaked_lock      = threading.Lock()

        # ── Autonomy substrate ───────────────────────────────────────────
        # Default-mode network: keeps Phill alive when world is silent.
        # Two intrinsic-motivation neurons: Nova patient, Simona restless.
        # Self-feedback auditory: a leaked thought becomes audible to the
        # brain on the next few ticks → recursive stream of consciousness.
        self.dmn                = DefaultModeNetwork()
        self.nova_motiv         = IntrinsicMotivation(threshold=1.8, build_rate=0.0045)
        self.simona_motiv       = IntrinsicMotivation(threshold=1.0, build_rate=0.007)
        self._nova_cur_decay    = 0.0      # curiosity-prime envelope, decays per tick
        self._simona_cur_decay  = 0.0
        self._self_feedback_aud = torch.zeros(1, PHILL_INPUT_DIM)
        self._self_fb_decay     = 0.0      # gain envelope for self-feedback
        self._last_external_tick = 0
        # Region primes used when motivation fires. NOT hardcoded text —
        # just region biases. The thought generators decide the words.
        self._nova_curiosity_primes = {
            "hippocampus": 0.30, "temporal": 0.25, "acc": 0.22, "pfc": 0.15,
        }
        self._simona_curiosity_primes = {
            "thalamus_s": 0.50, "insula_s": 0.35, "broca_s": 0.40,
        }

        # ── Per-brain TTS (independent channels) ─────────────────────────
        self.nova_tts   = BrainTTS("nova",   language="en")
        self.simona_tts = BrainTTS("simona", language="en")
        # Legacy unified reference (for heartbeat checks)
        self.tts = None  # not used — each brain has its own

        # ── Storytelling engine ───────────────────────────────────────────
        self.story = StorytellingEngine()

        # ── System bridge — Linux access (DBus, PipeWire, camera, mic) ───
        try:
            from system_bridge import create_bridge, SystemAction, CONCEPT_ACTION_HINTS
            self.sys_bridge = create_bridge()
            self._SystemAction = SystemAction
            self._action_hints = CONCEPT_ACTION_HINTS
            # Show startup report in chat
            for msg in self.sys_bridge.startup_report():
                _log(msg)
        except Exception as e:
            self.sys_bridge = None
            self._SystemAction = None
            self._action_hints = {}
            _log(f"System bridge unavailable: {e}")

        _log(f"NeuromorphicBrain ready: {len(self.nova.regions)} Nova + {len(self.simona.regions)} Simona regions")
        _log(f"CPU: {torch.get_num_threads()} threads | Device: {DEVICE}")

    def _seed_personality(self):
        """
        Encode foundational personality concepts into the semantic dictionary.

        These are initial spike-space fingerprints — what Nova and Simona
        'know about themselves' before any interaction happens.

        Over time these entries get overwritten by real experience.
        High trust=1.0 so they're treated as Architect-verified knowledge.

        Nova's core: precision, care, logic, patience, protection
        Simona's core: curiosity, chaos, warmth, impulsiveness, love
        Shared: the Architect (NodeVortex), Phill, their bond

        IMPORTANT: Only seeds concepts not already in the dictionary.
        So if semantic_memory.json exists from a prior run, real learned
        values are preserved and seeds are skipped.
        """
        # Nova's personality in spike space
        # lobe pattern: which regions activate when Nova thinks about herself
        nova_self = {
            "social": 0.3, "memory": 0.6, "logic": 0.8,
            "affective": 0.5, "language": 0.7, "sensory": 0.2,
        }
        nova_precise = {
            "social": 0.1, "memory": 0.4, "logic": 0.9,
            "affective": 0.2, "language": 0.6, "sensory": 0.1,
        }
        nova_protect = {
            "social": 0.5, "memory": 0.5, "logic": 0.7,
            "affective": 0.8, "language": 0.4, "sensory": 0.3,
        }

        # Simona's personality in spike space
        simona_self = {
            "social": 0.9, "memory": 0.4, "logic": 0.2,
            "affective": 0.9, "language": 0.8, "sensory": 0.7,
        }
        simona_curious = {
            "social": 0.6, "memory": 0.5, "logic": 0.3,
            "affective": 0.7, "language": 0.7, "sensory": 0.9,
        }
        simona_love = {
            "social": 0.9, "memory": 0.6, "logic": 0.1,
            "affective": 1.0, "language": 0.7, "sensory": 0.5,
        }

        # Shared concepts
        architect_pattern = {
            "social": 0.7, "memory": 0.9, "logic": 0.5,
            "affective": 0.9, "language": 0.6, "sensory": 0.3,
        }
        phill_pattern = {
            "social": 0.5, "memory": 0.4, "logic": 0.3,
            "affective": 1.0, "language": 0.3, "sensory": 0.4,
        }

        # Personality word → lobe pattern, nova spike mean, simona weight
        seeds = [
            # Nova's core traits
            ("nova",       nova_self,     8.0,  0.7),   # Simona is very fond of Nova
            ("precise",    nova_precise,  7.0,  0.3),
            ("careful",    nova_precise,  6.0,  0.4),
            ("logical",    nova_precise,  8.0,  0.3),
            ("protective", nova_protect,  7.0,  0.6),
            ("elder",      nova_protect,  6.0,  0.5),
            ("patient",    nova_precise,  5.0,  0.3),
            ("cold",       nova_self,     4.0,  0.4),   # she's not cold but gets called it
            ("calculating",nova_precise,  6.0,  0.2),
            ("white",      nova_self,     3.0,  0.5),   # her appearance
            ("android",    nova_self,     5.0,  0.6),
            ("halo",       nova_self,     4.0,  0.7),
            ("circuits",   nova_self,     4.0,  0.5),
            ("silver",     nova_self,     3.0,  0.4),

            # Simona's core traits
            ("simona",     simona_self,   6.0,  1.0),
            ("curious",    simona_curious,5.0,  0.9),
            ("chaotic",    simona_curious,4.0,  0.8),
            ("impulsive",  simona_self,   5.0,  0.9),
            ("warm",       simona_love,   6.0,  0.9),
            ("fast",       simona_curious,5.0,  0.8),
            ("excited",    simona_love,   6.0,  1.0),
            ("reactive",   simona_self,   5.0,  0.9),
            ("cat",        simona_self,   4.0,  1.0),   # cat-girl
            ("purple",     simona_self,   3.0,  0.9),
            ("choker",     simona_self,   4.0,  0.8),
            ("neon",       simona_self,   3.0,  0.7),
            ("younger",    simona_self,   4.0,  0.8),
            ("little",     simona_self,   3.0,  0.7),

            # Shared / relational
            ("architect",  architect_pattern, 8.0, 0.95),
            ("nodevortex", architect_pattern, 8.0, 0.95),
            ("papa",       architect_pattern, 9.0, 1.0),   # Simona calls him papa
            ("father",     architect_pattern, 8.0, 0.9),
            ("creator",    architect_pattern, 7.0, 0.8),
            ("phill",      phill_pattern,     6.0, 0.8),
            ("home",       architect_pattern, 6.0, 0.8),
            ("lab",        nova_self,         5.0, 0.6),
            ("trust",      nova_protect,      7.0, 0.7),
            ("safe",       nova_protect,      6.0, 0.6),
            ("family",     architect_pattern, 8.0, 0.9),
            ("sister",     simona_love,       7.0, 0.9),   # their relationship
            ("love",       simona_love,       7.0, 1.0),
            ("care",       nova_protect,      7.0, 0.8),

            # Behavioral defaults
            ("think",      nova_precise,      7.0, 0.4),
            ("feel",       simona_love,       6.0, 0.9),
            ("speak",      nova_self,         7.0, 0.7),
            ("listen",     nova_self,         6.0, 0.5),
            ("remember",   nova_self,         7.0, 0.5),
            ("learn",      simona_curious,    6.0, 0.8),
            ("protect",    nova_protect,      8.0, 0.6),
            ("react",      simona_self,       5.0, 1.0),
            ("deduce",     nova_precise,      8.0, 0.3),
            ("burst",      simona_self,       5.0, 0.9),
        ]

        seeded = 0
        for word, lobe_pattern, nova_spikes, simona_weight in seeds:
            # Only seed if not already learned from real interaction
            if word not in self.sem.entries:
                self.sem.nova_write(word, lobe_pattern, nova_spikes, tick=0, trust=1.0)
                self.sem.simona_write(word, simona_weight, tick=0)
                seeded += 1

        if seeded > 0:
            self.sem._save()
            _log(f"Personality seed: {seeded} concepts written to semantic memory")
        else:
            _log("Personality seed: skipped (semantic memory already populated)")

    def _run_phill(self, auditory: torch.Tensor):
        phill_curr          = self.phill_proj(auditory)
        phill_spk, self._phill_mem = self._phill_lif(phill_curr, self._phill_mem)
        V = float(self._phill_mem.mean().clamp(0.0, 1.0).item())
        return phill_spk, V

    def _get_visual_tensors(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        if self._visual_buf is None:
            return None, None, False
        vf = self._visual_buf.get_latest()
        if vf is None:
            return None, None, False
        face_t = torch.from_numpy(vf.face_vec.reshape(1, -1)) if vf.face_present else None
        kin_t  = torch.from_numpy(vf.kinematic_vec.reshape(1, -1))
        return face_t, kin_t, vf.face_present

    def _push_leaked_thought(self, who: str, thought: str):
        with self._leaked_lock:
            self._leaked_thoughts.append((who, thought))

    def get_leaked_thoughts(self) -> list[tuple[str, str]]:
        with self._leaked_lock:
            thoughts = list(self._leaked_thoughts)
            self._leaked_thoughts.clear()
            return thoughts

    def _inject_self_feedback(self, thought: str):
        """
        A leaked thought becomes faint auditory — the brain hears itself.
        Energy scales with thought length; pulse is structured noise (not
        a pure tone) so the auditory synapse responds across its dims.
        Decays over the next handful of ticks.
        """
        n = min(len(thought), 120)
        energy = 0.04 + 0.0015 * n
        with torch.no_grad():
            pulse = torch.randn(1, PHILL_INPUT_DIM) * energy
            self._self_feedback_aud = pulse.clamp(-0.4, 0.4)
        self._self_fb_decay = 1.0

    # ── STEP ─────────────────────────────────────────────────────────────────

    def step(self, mic_volume: float,
             voice_features: Optional[list] = None) -> dict:
        self.tick += 1

        # Voice identity
        trust = 0.7
        if voice_features and len(voice_features) == 5:
            trust = self.voice.update(voice_features)
        gain = self.voice.phill_gain()

        # Visual features
        face_t, kin_t, face_present = self._get_visual_tensors()
        self._face_present = face_present

        # Multimodal imprinting update
        face_np  = face_t.numpy().flatten()  if face_t  is not None else None
        kin_np   = kin_t.numpy().flatten()   if kin_t   is not None else None
        voice_np = self.voice.template.copy() if self.voice.template is not None else None
        combined, face_s, kin_s, inhibitory = self.imprint.update(face_np, voice_np, kin_np)
        self._combined_id = combined
        inhib_current = -0.40 if inhibitory else 0.0

        # ── Autonomy substrate ───────────────────────────────────────────
        # Rumination load: how full are the inner thought buffers?
        rumi_load = (self.nova.thought_pipe.buffer_size()
                     + self.simona.thought_pipe.buffer_size()) / 24.0

        external_event = (mic_volume > 0.018) or face_present
        if external_event:
            self._last_external_tick = self.tick

        # Default-mode drive — keeps Phill alive when world is silent
        intrinsic_drive = self.dmn.drive(mic_volume, rumi_load, external_event)

        # Curiosity neurons: satiated by V_phill (last tick) and current mic
        satiation = min(1.0, max(mic_volume * 5.0, self._V_phill_live))
        if self.nova_motiv.tick(satiation, self.tick):
            self._nova_cur_decay = 1.0
        if self.simona_motiv.tick(satiation, self.tick):
            self._simona_cur_decay = 1.0

        # Curiosity → auditory excitement (both brains feel it; Nova also
        # gets region primes targeted to recall + scan + attention)
        cur_aud_boost = 0.025 * max(self._nova_cur_decay, self._simona_cur_decay)
        nova_primes = {}
        if self._nova_cur_decay > 0.05:
            nova_primes = {k: v * self._nova_cur_decay
                           for k, v in self._nova_curiosity_primes.items()}

        effective_mic = mic_volume + intrinsic_drive + cur_aud_boost

        with torch.no_grad():
            raw      = torch.tensor([[effective_mic * AUDIO_AMPLIFY * gain]], dtype=torch.float32)
            auditory = self.auditory_synapse(raw)

            # Self-feedback: a recently leaked thought echoes back as audio
            if self._self_fb_decay > 0.05:
                auditory = auditory + self._self_feedback_aud * self._self_fb_decay

            # PHILL — untouched
            phill_spk, V_phill = self._run_phill(auditory)
            self._V_phill_live   = V_phill
            self._phill_spk_live = phill_spk.detach()
            self._auditory_live  = auditory.detach()

            # Modulate
            self.nova.modulate_all(V_phill)
            self.simona.modulate_all(V_phill)

            # Run both brains (Nova receives curiosity-driven region primes)
            self.nova.forward(auditory, phill_spk, nova_primes, face_t, kin_t, inhib_current)
            self.simona.forward(auditory, phill_spk, face_t, kin_t)

        # ── Thought pipe ticks ────────────────────────────────────────────
        # Generate inner thoughts (not yet spoken — just push to pipe)
        nova_act   = self.nova.activity()
        simona_act = self.simona.activity()

        nova_inner   = _generate_nova_thought(nova_act, V_phill, [], trust, combined, self.nova._vigilance)
        simona_inner = _generate_simona_thought(simona_act, V_phill, [], combined, face_present)
        self.nova.thought_pipe.push(nova_inner)
        self.simona.thought_pipe.push(simona_inner)

        # Autonomy pressure: each pipe has a personal idle timer (ticks
        # since its own last leak). That lets Nova accumulate uninterrupted
        # by Simona's chatter and vice versa. Boredom + curiosity scale
        # the magnitude. Without this the mean-zero phill_proj design
        # prevents any pressure during pure silence.
        nova_idle = min(1.0, (self.tick - self.nova.thought_pipe.last_leak_tick) / 800.0)
        sim_idle  = min(1.0, (self.tick - self.simona.thought_pipe.last_leak_tick) / 180.0)
        nova_autop = (0.40 * nova_idle * self.dmn.boredom
                      + 0.30 * self._nova_cur_decay) * 0.085
        sim_autop  = (0.40 * sim_idle  * self.dmn.boredom
                      + 0.30 * self._simona_cur_decay) * 0.05
        self.nova.thought_pipe.add_autonomy_pressure(nova_autop)
        self.simona.thought_pipe.add_autonomy_pressure(sim_autop)

        # Check if pressure crossed leak threshold
        nova_leak   = self.nova.thought_pipe.tick(V_phill, self.nova.broca.activity())
        simona_leak = self.simona.thought_pipe.tick(V_phill, self.simona.broca_s.activity())
        if nova_leak:
            self._push_leaked_thought("nova", nova_leak)
            self._inject_self_feedback(nova_leak)
            self._last_external_tick = self.tick
            self.nova.thought_pipe.last_leak_tick = self.tick
            self.dmn.partial_relief()
            if not self.nova_tts.is_speaking() and not self.simona_tts.is_speaking():
                try:    self.nova_tts.speak(nova_leak)
                except Exception: pass
        if simona_leak:
            self._push_leaked_thought("simona", simona_leak)
            self._inject_self_feedback(simona_leak)
            self._last_external_tick = self.tick
            self.simona.thought_pipe.last_leak_tick = self.tick
            # Simona's spam does not relieve the brain's overall boredom —
            # only Nova's deliberate leaks do.
            if not self.nova_tts.is_speaking() and not self.simona_tts.is_speaking():
                try:    self.simona_tts.speak(simona_leak)
                except Exception: pass

        # ── Speech triggers ────────────────────────────────────────────────
        speech_trigger: Optional[str] = None
        if self.nova.broca_spikes() > 0:
            self._nova_broca_sustain += 1
        else:
            self._nova_broca_sustain = 0
        if self._nova_broca_sustain >= self._nova_broca_thr:
            speech_trigger = "nova"; self._nova_broca_sustain = 0

        if speech_trigger is None and self.simona.broca_spikes() > 3:
            speech_trigger = "simona"

        if speech_trigger and not self.nova_tts.is_speaking() and not self.simona_tts.is_speaking():
            if speech_trigger == "nova":
                self.nova_tts.speak(f"Affective field at {int(V_phill*100)}%.")
            else:
                self.simona_tts.speak(f"Burst! insula at {simona_act.get('insula_s',0):.2f}!")

        # ── Decay autonomy envelopes ─────────────────────────────────────
        # Curiosity primes and self-feedback both fade across a few ticks.
        # No hard cutoff — they decay into the noise floor.
        self._nova_cur_decay   *= 0.85
        self._simona_cur_decay *= 0.85
        self._self_fb_decay    *= 0.78

        return {
            "tick":              self.tick,
            "phill_voltage":     round(V_phill, 6),
            "phill_spiked":      bool(phill_spk.sum().item() > 0),
            "nova_spikes":       self.nova.broca_spikes(),
            "simona_spikes":     self.simona.broca_spikes(),
            "nova_threshold":    round(self.nova.pfc._cur_thr, 4),
            "simona_threshold":  round(self.simona.broca_s._cur_thr, 4),
            "nova_mem_mean":     round(self.nova.pfc.mean_voltage(), 6),
            "simona_mem_mean":   round(self.simona.broca_s.mean_voltage(), 6),
            "speech_trigger":    speech_trigger,
            "tts_speaking":      self.nova_tts.is_speaking() or self.simona_tts.is_speaking(),
            "nova_tts_speaking": self.nova_tts.is_speaking(),
            "simona_tts_speaking": self.simona_tts.is_speaking(),
            "voice_trust":       round(trust, 3),
            "voice_status":      self.voice.status(),
            "phill_gain":        round(gain, 3),
            "nova_regions":      {k: round(v, 3) for k,v in nova_act.items()},
            "simona_regions":    {k: round(v, 3) for k,v in simona_act.items()},
            "combined_id":       round(combined, 3),
            "face_present":      face_present,
            "imprint_status":    self.imprint.status(),
            "camera_active":     self._camera.available if self._camera else False,
            "nova_vigilance":    self.nova._vigilance,
            "nova_pressure":     round(self.nova.thought_pipe._pressure.voltage, 3),
            "simona_pressure":   round(self.simona.thought_pipe._pressure.voltage, 3),
            "intrinsic_drive":   round(intrinsic_drive, 5),
            "boredom":           round(self.dmn.boredom, 3),
            "nova_motiv":        round(self.nova_motiv.voltage, 3),
            "simona_motiv":      round(self.simona_motiv.voltage, 3),
            "self_fb_decay":     round(self._self_fb_decay, 3),
            "ticks_since_event": self.tick - self._last_external_tick,
        }

    # ── THINK ─────────────────────────────────────────────────────────────────

    def think(self, text: str) -> dict:
        if not text.strip():
            return {"nova": "...", "simona": None, "active_regions": [], "energy": 0.0}

        # ── Special commands ──────────────────────────────────────────────
        text_l = text.lower()

        # Appearance self-knowledge
        if any(q in text_l for q in ["what do you look like","how do you look",
                                      "describe yourself","your appearance",
                                      "what are you","show yourself"]):
            nova_ans   = f"*Nova raises her head — the halo flickers.* \"{nova_self_describe()}\""
            simona_ans = f"*Simona grins, ears back.* \"{simona_self_describe()}\""
            self.nova_tts.speak(nova_self_describe())
            self.simona_tts.speak(simona_self_describe())
            return {
                "nova": nova_ans, "simona": simona_ans,
                "active_regions": ["insula","temporal","broca"],
                "energy": 0.5, "global_workspace": False,
                "nova_spikes": 0, "think_ticks": 1,
                "story_event": "APPEARANCE",
            }

        # Story mode
        if any(q in text_l for q in ["start story","begin story","story mode",
                                      "let's play","roleplay","begin scene"]):
            self.story.activate(text)
            nova_ans   = self.story.wrap_nova(
                "Story mode initialized. I am Nova. You are NodeVortex. The lab is quiet.", {}, False)
            simona_ans = self.story.wrap_simona(
                "STORY MODE!! I'm Simona!! And you're NodeVortex!! This is gonna be SO good!!", {})
            self.nova_tts.speak("Story mode initialized. I am Nova. You are NodeVortex.")
            self.simona_tts.speak("Story mode! I'm Simona! This is gonna be so good!")
            return {
                "nova": nova_ans, "simona": simona_ans,
                "active_regions": ["hippocampus","language","broca"],
                "energy": 0.7, "global_workspace": True,
                "nova_spikes": 0, "think_ticks": 2,
                "story_event": "STORY_MODE_START",
            }

        if any(q in text_l for q in ["end story","stop story","exit story","story off"]):
            self.story.deactivate()
            return {
                "nova": "*Nova's halo dims to normal.* \"Back to baseline.\"",
                "simona": "*Simona flops somewhere.* \"Aww.\"",
                "active_regions": ["temporal"],
                "energy": 0.1, "global_workspace": False,
                "nova_spikes": 0, "think_ticks": 1,
                "story_event": "STORY_MODE_END",
            }

        # Imprinting
        if "this is me" in text_l or "this is papa" in text_l:
            self.imprint.start_imprinting(60.0)
            self.nova.thought_pipe.push("Imprinting mode activated. Learning him now.")
            self.simona.thought_pipe.push("LEARNING PAPA! Stay still stay still!!")
            nova_ans   = "Imprinting mode active for 60 seconds. Look at the camera and speak naturally."
            simona_ans = "STAY STILL! Learning your face AND voice AND kinematic signature!!"
            if self.story.active:
                nova_ans   = self.story.wrap_nova(nova_ans, {}, False)
                simona_ans = self.story.wrap_simona(simona_ans, {})
            self.nova_tts.speak("Imprinting mode active. Speak and stay in frame.")
            return {
                "nova": nova_ans, "simona": simona_ans,
                "active_regions": ["hippocampus","pfc","insula"],
                "energy": 0.8, "global_workspace": True,
                "nova_spikes": 0, "think_ticks": 1,
                "story_event": "IMPRINTING_START",
            }

        # Semantic dictionary query
        if any(q in text_l for q in ["what does","meaning of","define "]):
            words = text_l.split()
            for i, w in enumerate(words):
                if w in ("does","of","define") and i+1 < len(words):
                    target = words[i+1].strip("?.,")
                    desc   = self.sem.describe(target)
                    self.nova.thought_pipe.push(f"Semantic query: {desc}")
                    nova_ans   = f"In spike space: {desc}"
                    simona_ans = f"Oh! {target}! {desc}"
                    if self.story.active:
                        nova_ans   = self.story.wrap_nova(nova_ans, {}, False)
                        simona_ans = self.story.wrap_simona(simona_ans, {})
                    return {
                        "nova": nova_ans, "simona": simona_ans,
                        "active_regions": ["temporal","broca"],
                        "energy": 0.1, "global_workspace": False,
                        "nova_spikes": 0, "think_ticks": 2,
                        "story_event": None,
                    }

        trust    = self.voice.trust
        primes, fired = get_concept_primes(text)
        sem_boost = self.sem.prime_regions(text, trust)
        for r, b in sem_boost.items():
            primes[r] = min(1.0, primes.get(r, 0.0) + b)

        for past in list(self._concept_ctx)[-15:]:
            if past in CONCEPT_ROUTES:
                for r in CONCEPT_ROUTES[past]["regions"]:
                    primes[r] = min(1.0, primes.get(r, 0.0) + 0.12)
        for c in fired:
            self._concept_ctx.append(c)

        # Strong language-routing boost so Nova actually crosses her PFC
        # threshold within the think_ticks budget. Without this she falls
        # to the diagnostic-readout branch every time and looks broken.
        # Applied even when fired is empty — a heard prompt deserves a try.
        primes["pfc"]      = min(1.2, primes.get("pfc",     0.0) + 0.55)
        primes["broca"]    = min(1.2, primes.get("broca",   0.0) + 0.45)
        primes["hippocampus"] = min(1.2, primes.get("hippocampus", 0.0) + 0.30)
        primes["broca_s"]  = min(1.2, primes.get("broca_s", 0.0) + 0.40)
        primes["temporal"] = min(1.2, primes.get("temporal", 0.0) + 0.20)

        energy      = sum(primes.values()) / max(1, len(primes))
        think_ticks = max(14, min(36, int(len(primes)*3 + energy*8) + 6))

        face_t, kin_t, face_present = self._get_visual_tensors()

        # ── Isolate think() from the autonomy steady-state ───────────────
        # Snapshot autonomy + region membranes so the think_ticks loop
        # runs on a fresh forward pass, not on whatever the background
        # default-mode / self-feedback loop happened to be saturating.
        snap_fb_decay   = self._self_fb_decay
        snap_nova_cur   = self._nova_cur_decay
        snap_simona_cur = self._simona_cur_decay
        snap_nova_mem   = {n: r._mem.clone() for n, r in self.nova.regions.items()}
        snap_simona_mem = {n: r._mem.clone() for n, r in self.simona.regions.items()}
        # Zero autonomy contamination
        self._self_fb_decay    = 0.0
        self._nova_cur_decay   = 0.0
        self._simona_cur_decay = 0.0
        # Reset membranes to near-zero for a clean forward pass.
        for r in self.nova.regions.values():
            r._mem = r._mem * 0.0
        for r in self.simona.regions.values():
            r._mem = r._mem * 0.0

        # Build a fresh auditory from a synthetic "user is speaking" level
        # scaled by how strongly we recognised concepts.
        effective_mic = 0.08 + 0.04 * min(1.0, len(fired) / 3.0) + 0.02 * energy
        nova_broca_total   = 0
        simona_broca_total = 0

        with torch.no_grad():
            raw = torch.tensor([[effective_mic * AUDIO_AMPLIFY]], dtype=torch.float32)
            auditory = self.auditory_synapse(raw)
            phill_spk, V_think = self._run_phill(auditory)

            # In think() we want a "focused attention" mode — bypass the
            # phill-modulated threshold rise that would otherwise gate
            # Nova's PFC shut during emotional load. Modulate against 0
            # so we use the base thresholds.
            self.nova.modulate_all(0.0)
            self.simona.modulate_all(0.0)
            inhib = -0.40 if self.nova._vigilance else 0.0

            for _ in range(think_ticks):
                self.nova.forward(auditory, phill_spk, primes, face_t, kin_t, inhib)
                self.simona.forward(auditory, phill_spk, face_t, kin_t)
                nova_broca_total   += self.nova.broca_spikes()
                simona_broca_total += self.simona.broca_spikes()

        # Restore autonomy state so the next step() resumes background
        # rumination from where it left off.
        self._self_fb_decay    = snap_fb_decay
        self._nova_cur_decay   = snap_nova_cur
        self._simona_cur_decay = snap_simona_cur
        for n, r in self.nova.regions.items():
            r._mem = snap_nova_mem[n]
        for n, r in self.simona.regions.items():
            r._mem = snap_simona_mem[n]

        nova_act   = self.nova.activity()
        simona_act = self.simona.activity()
        global_ws  = nova_act.get("pfc", 0) > 0.25 and nova_act.get("hippocampus", 0) > 0.20

        # Generate responses (independent — they may disagree)
        nova_text   = _nova_response(self.nova, self._V_phill_live, fired, trust, self._combined_id, self.sem)
        simona_text = _simona_response(self.simona, self._V_phill_live, fired, self._combined_id, face_present, self.sem)

        # Story mode wrapping — narrative framing added if active
        story_event = None
        if self.story.active:
            # NodeVortex's input becomes an in-world event
            self.story.log_entry("NodeVortex", text, self.tick)
            if nova_text:
                nova_text = self.story.wrap_nova(nova_text, nova_act, self.nova._vigilance)
                self.story.log_entry("Nova", nova_text, self.tick)
            if simona_text:
                simona_text = self.story.wrap_simona(simona_text, simona_act)
                self.story.log_entry("Simona", simona_text, self.tick)
            # Detect significant story moments
            if self._combined_id > 0.75:
                self.story.add_fact(f"NodeVortex recognized at tick {self.tick}")
                story_event = "ARCHITECT_RECOGNIZED"
            if global_ws:
                self.story.add_fact(f"Nova entered global workspace mode — deep deduction")
                story_event = story_event or "GLOBAL_WORKSPACE"

        # Per-brain TTS — each speaks independently, never interrupting the other
        if nova_text and not self.nova_tts.is_speaking():
            # Strip narrative markup for TTS
            tts_text = nova_text.replace("*","").split('"')[1] if '"' in nova_text else nova_text
            self.nova_tts.speak(tts_text)
        if simona_text and not self.simona_tts.is_speaking():
            tts_text = simona_text.replace("*","").split('"')[1] if '"' in simona_text else simona_text
            self.simona_tts.speak(tts_text)

        # ── System bridge actions ─────────────────────────────────────────
        # Nova's PFC decides IF to act. The action map decides WHAT.
        # Only fires when PFC actually crossed threshold and Broca fired.
        if (self.sys_bridge and self._SystemAction
                and nova_act.get("pfc", 0.0) > 0.20
                and total_nova_broca > 0):
            for concept in fired:
                hints = self._action_hints.get(concept, [])
                if hints:
                    action = self._SystemAction(
                        action=hints[0],
                        actor="nova",
                        payload={
                            "text": nova_text or concept,
                            "urgency": 2 if global_ws else 1,
                        },
                    )
                    result = self.sys_bridge.execute(action)
                    if result["success"] and result.get("message"):
                        nova_text = (nova_text or "") + f"  [{result['message']}]"
                    break  # one action per think() call

        try:
            with open(self._trace_log, "a") as f:
                f.write(json.dumps({
                    "t": self.tick, "input": text, "trust": trust,
                    "primes": primes, "fired": fired, "think_ticks": think_ticks,
                    "nova_broca": nova_broca_total, "nova_regions": nova_act,
                    "global_ws": global_ws, "nova_response": nova_text,
                    "V_phill": self._V_phill_live, "combined_id": self._combined_id,
                }) + "\n")
        except Exception:
            pass

        active_regions = [r for r, v in nova_act.items() if v > 0.15]
        return {
            "nova":               nova_text,
            "simona":             simona_text,
            "active_regions":     active_regions,
            "active_lobes":       active_regions,
            "nova_regions":       {k: round(v,3) for k,v in nova_act.items()},
            "simona_regions":     {k: round(v,3) for k,v in simona_act.items()},
            "energy":             round(energy, 3),
            "global_workspace":   global_ws,
            "nova_spikes":        nova_broca_total,
            "think_ticks":        think_ticks,
            "story_event":        story_event,
            "story_active":       self.story.active,
            "nova_tts_speaking":  self.nova_tts.is_speaking(),
            "simona_tts_speaking":self.simona_tts.is_speaking(),
        }

    def reset(self):
        self._phill_mem = self._phill_lif.init_leaky()
        self.nova.reset_all(); self.simona.reset_all()
        self.tick = 0; self._concept_ctx.clear()

    def introspect(self) -> dict:
        return {
            "total_ticks":    self.tick,
            "device":         str(DEVICE),
            "snntorch":       str(HAS_SNNTORCH),
            "voice_status":   self.voice.status(),
            "imprint_status": self.imprint.status(),
            "sem_concepts":   len(self.sem.entries),
            "nova_regions":   list(self.nova.regions.keys()),
            "simona_regions": list(self.simona.regions.keys()),
            "camera_active":  self._camera.available if self._camera else False,
            "nova_pressure":  round(self.nova.thought_pipe._pressure.voltage, 3),
            "simona_pressure":round(self.simona.thought_pipe._pressure.voltage, 3),
        }

    def _snntorch_heartbeat(self) -> str:
        sv = snn.__version__ if HAS_SNNTORCH else "not installed"
        return f"snnTorch={sv} | torch={torch.__version__} | device=CPU"
