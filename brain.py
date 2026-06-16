"""
brain.py — NeuromorphicBrain · Phase 5: CPU-Native, Multimodal, Emergent Identity
===================================================================================

ARCHITECTURE:
  • All tensors on CPU. DEVICE = torch.device("cpu") — no fallback, no iGPU.
  • MKL/OpenMP thread count pinned to physical core count at startup.
  • Process priority elevated to HIGH on Windows, nice(-10) on Linux.
  • Audio spikes travel through a pre-allocated numpy array (zero-copy).

  • 7 anatomical cortical regions (one brain: Alpha). Phill untouched.
  • Alpha drives the SharedSemanticDictionary and the ThoughtPipe output queue.
  • The single AlphaBrain holds its own membrane state, thresholds, and weights.

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
    → Alpha enters Vigilance Mode (higher PFC threshold, dampened response).
  This is purely physical — no if/else. The inhibitory current just
  prevents PFC from crossing θ. Emergence, not logic.

THOUGHT PIPE (fully emergent):
  Each brain has a RuminationBuffer — thoughts processed internally
  but not yet spoken accumulate there.
  A "pressure neuron" (LeakyAccumulator) integrates:
    pressure += (rumination_load * V_phill * broca_activity)
    pressure *= decay  (each tick)
  When pressure crosses θ_leak, the oldest thought in the buffer leaks.
  Alpha's θ_leak = 0.85 (he only leaks under real pressure; held as inner thought)
  This is NOT a ping. There is NO scheduled call.
  The brain loop checks if pressure crossed threshold — that IS the
  physical mechanism.

PHILL: COMPLETELY UNTOUCHED.
"""

import os
import sys

# ══════════════════════════════════════════════════════════════════════════════
# Redirect Python stderr + OS fd 2 → log file so noisy library writes
# (TTS warnings, mediapipe EGL banner, background-thread tracebacks) don't
# corrupt the TUI. We MUST NOT touch stdout / fd 1 — ratatui in Rust writes
# the TUI there.
# ══════════════════════════════════════════════════════════════════════════════
try:
    _stderr_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
        "brain_stderr.log",
    )
    _stderr_fd = open(_stderr_path, "a", buffering=1)
    sys.stderr = _stderr_fd
    sys.stdout = _stderr_fd  # Python-level print() goes to log too — Rust uses real fd 1
    # OS-level fd 2 redirect so C-extension stderr (mediapipe, EGL, etc.)
    # follows the same path. Fd 1 (stdout) is left alone for the TUI.
    os.dup2(_stderr_fd.fileno(), 2)
except Exception:
    pass

import torch
import torch.nn as nn
import numpy as np
import json
import re
import time
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
    Deterministic clock = no jitter in Alpha's 5-tick sustain.
    """
    phys = multiprocessing.cpu_count()
    # The SNN runs CONTINUOUSLY at 20Hz (+ two personality threads) doing small
    # matmuls — it does NOT need every core, and grabbing them all starves the
    # other neural runtimes that share this CPU (Piper = onnxruntime, Whisper =
    # ctranslate2, mediapipe), which then thrash. Cap the SNN to a modest share
    # and leave headroom for those bursty workers. Override via ALPHA_SNN_THREADS.
    try:
        snn = int(os.environ.get("ALPHA_SNN_THREADS", "") or 0)
    except ValueError:
        snn = 0
    if snn <= 0:
        snn = max(2, min(4, phys // 3))      # e.g. 4 on a 12-thread laptop
    torch.set_num_threads(snn)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Disable any GPU fallback; cap the CPU math libs to the SNN's share (this
    # also reins in Piper's onnxruntime OpenMP, which reads these vars).
    os.environ["CUDA_VISIBLE_DEVICES"]  = ""
    os.environ["XPU_VISIBLE_DEVICES"]   = ""
    os.environ["OMP_NUM_THREADS"]       = str(snn)
    os.environ["MKL_NUM_THREADS"]       = str(snn)
    os.environ["OPENBLAS_NUM_THREADS"]  = str(snn)

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
_L = _logging.getLogger("alpha_alpha")
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

# ── Audio output (pure-emergence TTS) ────────────────────────────────────────
# We no longer use any pretrained TTS (XTTS, etc.). The brain produces sound
# itself via FormantSynth driven by Broca motor spikes through MotorArticulator.
# sounddevice is the only output dependency — it just pushes float samples to
# the system audio device.
try:
    import sounddevice as _sd
    _AUDIO_OUT_AVAILABLE = True
except ImportError:
    _sd = None
    _AUDIO_OUT_AVAILABLE = False

# Intelligible WORD speech (offline, rule-based formant synth — NOT a pretrained
# neural model). espeak-ng if present, else None. The architect wants them to
# actually pronounce the words they form; the FormantSynth babble below stays for
# the pre-verbal motor learning (speak_motor), so both coexist.
import shutil as _shutil
_ESPEAK = _shutil.which("espeak-ng") or _shutil.which("espeak")

# Pre-verbal formant BABBLE is silenced by default. Now that they speak real
# words (Piper/espeak), the constant formant output is just glitchy noise between
# utterances (and a steady drain on a busy CPU). Motor LEARNING still runs (the
# synth + self/forward-model monitoring happen); only the audio is muted. Set
# ALPHA_BABBLE_AUDIO=1 to hear the babbling again.
_BABBLE_AUDIO = os.environ.get("ALPHA_BABBLE_AUDIO", "0").strip().lower() in ("1", "true", "yes", "on")

# Microphone gently removed (matches the Rust orchestrator's ALPHA_MIC_OFF). When
# deaf, Alpha's hot amygdala loses its startle source (ambient sound was ~40% of
# her arousal), so she goes quiet. With this set we reroute HER amygdala to orient
# on her own inner weather (boredom + unspoken-thought pressure + forward-model
# surprise) instead of the mic — restlessness self-generates, as the autonomy
# substrate intends. Alpha is untouched (cool amygdala + curiosity primes).
_MIC_OFF = os.environ.get("ALPHA_MIC_OFF", "").strip().lower() in ("1", "true", "yes", "on")

# Voice output gently removed (ALPHA_TTS_OFF). His mouth is covered: he still
# forms replies (text), but nothing is vocalised — and he FEELS it (the affect
# core surfaces a 'stifled' feeling). Symmetric to the mic / 'muffled'.
_TTS_OFF = os.environ.get("ALPHA_TTS_OFF", "").strip().lower() in ("1", "true", "yes", "on")

# Liveness / anti-spoofing: a live face's landmark geometry jitters slightly every
# frame; a PHOTO (held still OR waved around) is frozen — its normalised face
# vector is near-identical frame to frame. If the mean cosine between recent
# frames exceeds this, the "face" is treated as an IMAGE, not a living face, and
# is refused for recognition + learning. CALIBRATE on your own camera: lower it
# toward 0.999 if a photo can still fool him; raise it toward 0.9998 if he
# wrongly rejects the real you. Tunable via ALPHA_LIVENESS_THR.
try:
    _LIVENESS_FROZEN = float(os.environ.get("ALPHA_LIVENESS_THR", "") or 0.9995)
except ValueError:
    _LIVENESS_FROZEN = 0.9995


def _espeak_say(speaker: str, text: str) -> float:
    """Pronounce real words via espeak-ng, per-persona voice. Fire-and-forget (a
    daemon thread under the shared device lock) so the 20Hz loop never blocks.
    Returns a rough duration estimate for is_speaking()."""
    if not _ESPEAK or not text or not text.strip():
        return 0.0
    # Voices kept in natural HUMAN female ranges (extreme pitch = ghost/robot;
    # too-low female = sounds male). Tune to taste: -p = pitch 0..99, -s = wpm.
    if speaker == "alpha":        # grounded 19yo WOMAN — clearly female, calm, mid
        args = ["-v", "en-us+f2", "-p", "50", "-s", "150", "-a", "150"]
    else:                         # Alpha — little GIRL: bright + lively, not shrill
        args = ["-v", "en-us+f4", "-p", "74", "-s", "176", "-a", "163"]
    cmd = [_ESPEAK] + args + [text[:400]]

    def _run():
        import subprocess
        try:
            with BrainTTS._device_lock:
                subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True, name=f"espeak-{speaker}").start()
    return min(8.0, 0.35 + len(text.split()) * 0.38)


# ── Natural neural voice (Piper) — preferred over espeak when present ──────────
# CPU, ~real-time. Per-persona voice models live in voices/piper/. Alpha = a calm
# young-woman voice (lessac); Alpha = a lighter one (amy), sped up slightly so
# she reads younger. Streams raw PCM → sounddevice. (The architect asked for
# voices he can actually understand; espeak stays as the fallback.)
_ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PIPER_VOICES = {
    "alpha":   os.path.join(_ROOT, "voices", "piper", "en_US-lessac-medium.onnx"),
}
def _piper_importable():
    try:
        import piper  # noqa: F401
        return True
    except Exception:
        return False
_PIPER_OK = all(os.path.exists(p) for p in _PIPER_VOICES.values()) and _piper_importable()

# Each voice model is loaded ONCE and reused. Loading is ~1.3s; synthesis is then
# ~0.1s/sentence (real-time). The first cut spawned a fresh piper PROCESS per line
# — reloading the 63MB model every utterance = seconds of lag. This is the fix.
_PIPER_CACHE: dict = {}
_PIPER_CACHE_LOCK = threading.Lock()


def _piper_voice(speaker: str):
    if speaker in _PIPER_CACHE:
        return _PIPER_CACHE[speaker]
    with _PIPER_CACHE_LOCK:
        if speaker in _PIPER_CACHE:
            return _PIPER_CACHE[speaker]
        v = None
        try:
            import piper
            v = piper.PiperVoice.load(_PIPER_VOICES[speaker])
        except Exception:
            v = None
        _PIPER_CACHE[speaker] = v
        return v


def _piper_syn_config(speaker: str):
    """Optional speed tweak (Alpha a touch faster/younger). Best-effort."""
    try:
        from piper import SynthesisConfig
    except Exception:
        try:
            from piper.config import SynthesisConfig
        except Exception:
            return None
    try:
        return SynthesisConfig(length_scale=(0.92 if speaker == "alpha" else 1.0))
    except Exception:
        return None


def _piper_say(speaker: str, text: str) -> float:
    """Natural speech via Piper, reusing the preloaded voice (no per-call reload).
    Fire-and-forget; plays under the shared device lock so they don't overlap."""
    if not _PIPER_OK or _sd is None or not text or not text.strip():
        return 0.0

    def _run():
        try:
            v = _piper_voice(speaker)
            if v is None:
                return
            cfg = _piper_syn_config(speaker)
            chunks = list(v.synthesize(text[:400], cfg)) if cfg is not None \
                     else list(v.synthesize(text[:400]))
            if not chunks:
                return
            audio = np.concatenate([np.asarray(c.audio_float_array, dtype=np.float32)
                                    for c in chunks])
            sr = int(getattr(chunks[0], "sample_rate", 22050))
            with BrainTTS._device_lock:
                # Low latency: the audio is fully pre-rendered, so a big buffer
                # only delays the START of speech (felt as "lag"). 'low' makes it
                # speak immediately; underruns are unlikely on a pre-rendered clip.
                _sd.play(audio, samplerate=sr, blocking=True, latency="low")
                _sd.wait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True, name=f"piper-{speaker}").start()
    return min(10.0, 0.3 + len(text.split()) * 0.42)


# Pre-warm both voices in the background so the FIRST spoken line has no lag.
if _PIPER_OK:
    threading.Thread(target=lambda: [_piper_voice(s) for s in _PIPER_VOICES],
                     daemon=True, name="piper-warmup").start()


# ── Physics constants (unchanged) ─────────────────────────────────────────────
AUDIO_AMPLIFY   = 15.0
PHILL_INPUT_DIM = 8
PHILL_BETA      = 0.95
PHILL_THRESHOLD = 1.0
PHILL_HIDDEN    = 16
ALPHA_LANG       = "en"

# Phill neuromodulation coupling
ALPHA  = 0.40   # Alpha PFC threshold rise per V_phill
GAMMA  = 0.05   # Alpha beta gain

# Alpha region physics
_ALPHA_REGIONS = {
    # name         size  beta   thr    phill_alpha  proj_std
    "thalamus":   (16,  0.85,  0.80,  0.10,        0.13),
    "temporal":   (24,  0.88,  1.00,  0.20,        0.11),
    "hippocampus":(20,  0.93,  1.10,  0.30,        0.10),
    "acc":        (14,  0.87,  0.90,  0.25,        0.12),
    "pfc":        (28,  0.92,  1.40,  0.45,        0.09),
    "broca":      (16,  0.89,  1.20,  0.35,        0.10),
    "insula":     (12,  0.91,  0.95,  0.15,        0.11),
}

# Alpha region physics


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

    def modulate(self, V_phill: float, neuro_offset: float = 0.0):
        new_thr = self.threshold + self.phill_alpha * V_phill + neuro_offset
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
        # Liveness: recent face vectors → detect a frozen (photo) face.
        self._face_hist  = deque(maxlen=12)
        self.face_live   = 1.0
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

        # Bootstrap: the first time a channel appears, seed its template from the
        # sample. Without this, cosine-vs-None is always 0, coincidence can never
        # fire, and learning never starts (templates only ever update INSIDE a
        # coincidence). Seeding the first sighting lets recognition actually begin.
        if face_vec  is not None and self.face_template  is None: self.face_template  = face_vec.copy()
        if voice_vec is not None and self.voice_template is None: self.voice_template = voice_vec.copy()
        if kin_vec   is not None and self.kin_template   is None: self.kin_template   = kin_vec.copy()

        # Compute similarity scores
        fs = self._cosine(self.face_template,  face_vec)  if face_vec  is not None else 0.0
        vs = self._cosine(self.voice_template, voice_vec) if voice_vec is not None else 0.0
        ks = self._cosine(self.kin_template,   kin_vec)   if kin_vec   is not None else 0.0

        # ── LIVENESS / anti-spoof ─────────────────────────────────────────────
        # A live face's geometry jitters frame to frame; a photo (still or waved)
        # is frozen — near-identical vectors. If the recent frames are essentially
        # identical, this "face" is an IMAGE: refuse it for recognition/learning
        # and raise suspicion. (A waved photo also makes optical-flow motion, but
        # its FACE stays frozen — this is exactly what catches that attack.)
        face_is_photo = False
        if face_vec is not None:
            self._face_hist.append(np.asarray(face_vec, dtype=np.float32).copy())
            if len(self._face_hist) >= 6:
                sims = [self._cosine(self._face_hist[i], self._face_hist[i-1])
                        for i in range(1, len(self._face_hist))]
                msim = sum(sims) / len(sims)
                self.face_live = max(0.0, min(1.0, (1.0 - msim) * 250.0))
                face_is_photo = (msim >= _LIVENESS_FROZEN)
        else:
            self.face_live = 0.0

        # EMA smoothing
        self._ema_face  = self._ema_alpha * self._ema_face  + (1-self._ema_alpha) * fs
        self._ema_voice = self._ema_alpha * self._ema_voice + (1-self._ema_alpha) * vs
        self._ema_kin   = self._ema_alpha * self._ema_kin   + (1-self._ema_alpha) * ks

        self.face_score  = self._ema_face
        self.voice_score = self._ema_voice
        self.kin_score   = self._ema_kin

        # Which sensory channels are actually present this tick. The mic may be
        # OFF (noisy room) — then identity rests on the CAMERA (face + motion)
        # alone, instead of being forced to zero through an absent voice channel.
        present = []   # (ema_score, raw_score, threshold)
        # A photo-face is NOT a valid recognition channel — exclude it, so it can
        # neither be recognised nor teach itself into the template.
        if face_vec  is not None and not face_is_photo:
            present.append((self._ema_face,  fs, face_thr))
        if voice_vec is not None: present.append((self._ema_voice, vs, voice_thr))
        if kin_vec   is not None: present.append((self._ema_kin,   ks, kin_thr))

        # Combined = geometric mean over the channels we actually have (all 3 when
        # the mic is on; face + motion when it's off).
        if present:
            prod = 1.0
            for ema, _, _ in present:
                prod *= max(0.0, ema)
            self.combined = float(prod ** (1.0 / len(present)))
        else:
            self.combined = 0.0

        # Coincidence = every PRESENT channel above its threshold, with at least
        # TWO channels agreeing — so a static face alone (a photo) can't pass, but
        # face + motion (camera-only) or face + voice + motion both can.
        coincidence = (len(present) >= 2
                       and all(raw >= thr for _, raw, thr in present))

        if coincidence:
            self.coincidence_count += 1
            if self.coincidence_count >= self.MIN_SAMPLES:
                self.trusted = True
            # Hebbian update — never learn a face from a frozen (photo) frame.
            if face_vec  is not None and not face_is_photo:
                self.face_template  = self._update_template(self.face_template,  face_vec,  lr)
            if voice_vec is not None: self.voice_template = self._update_template(self.voice_template, voice_vec, lr)
            if kin_vec   is not None: self.kin_template   = self._update_template(self.kin_template,   kin_vec,   lr)
            if self.coincidence_count % 10 == 0:
                self._save()

        # Anti-gullibility: a frozen (photo) face, OR a known face with no matching
        # motion → suspicion (negative current into ACC → vigilance).
        inhibitory = (face_vec is not None
                      and (face_is_photo
                           or (self.trusted and fs > 0.75 and ks < 0.40)))

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
# BABBLING CORTEX — pre-linguistic sensorimotor exploration
# ══════════════════════════════════════════════════════════════════════════════

class BabblingCortex:
    """
    The foundation of language: kids don't speak words first, they BABBLE.
    Random vocal patterns → they hear themselves → "this spike pattern
    produces this sound" gets Hebbian-wired. Only after this motor map
    is built can the brain INTENTIONALLY produce sound.

    Mechanism (per personality):
      1. When boredom is high OR curiosity neuron fires AND TTS is free,
         sample a phoneme from the inventory (weighted by what previously
         worked for the current motor spike pattern)
      2. Speak it through this personality's TTS channel
      3. Mark a self-speaking window (~1.75s) during which incoming mic
         is interpreted as our own echo, not external speech
      4. While in that window, if the mic actually carries sound, perform
         Hebbian binding: motor signature ↔ phoneme
      5. Each successful babble also writes the phoneme into the shared
         semantic dictionary, so the brain can later recognize it when
         the architect speaks the same sound

    Persisted to babble_<name>.json so plasticity carries across sessions.
    """

    PHONEMES = [
        # Vowels — easiest motor patterns
        "ah", "eh", "ee", "oh", "oo",
        # CV syllables — universal first sounds across cultures
        "ma", "ba", "da", "ga", "pa", "ta", "na", "la",
        # Reduplicated — the first true "words" babies produce
        "mama", "baba", "dada", "papa", "nana", "lala",
    ]

    BABBLE_COOLDOWN_TICKS = 80      # ~4s minimum between babbles
    BABBLE_BOREDOM_THR    = 0.30
    BABBLE_RANDOM_RATE    = 0.0008  # ~1/min baseline drive
    SELF_SPEAK_TICKS      = 35      # ~1.75s self-listening window
    BIND_LR               = 0.10
    EXPLORE_RATE          = 0.30    # 30% pure exploration even with priors

    def __init__(self, name: str, save_dir: Path):
        self.name             = name
        self.last_babble_tick = -10_000
        self.self_speak_until = -1
        self.last_phoneme:    Optional[str] = None
        self.last_motor_sig:  Optional[str] = None
        self.last_motor_vec:  Optional[np.ndarray] = None
        self.motor_to_phoneme: dict[str, dict[str, float]] = {}
        self.babble_count = 0
        self.bound_count  = 0
        self._explore_boost = 0.0   # raised by forward-model surprise (set in maybe_babble)
        self._save_path = save_dir / f"babble_{name}.json"
        # Region scores written to semantic dict on successful binding.
        # Alpha uses cortical region names; Alpha uses her _s-suffixed names.
        if name == "alpha":
            self._sem_regions = {
                "thalamus": 0.50, "temporal": 0.65, "broca": 0.70,
                "insula":   0.55, "pfc":      0.30, "acc":   0.30,
                "hippocampus": 0.40,
            }
        else:
            self._sem_regions = {
                "thalamus_s": 0.50, "temporal_s": 0.65, "broca_s": 0.70,
                "insula_s":   0.55, "pfc_s":      0.30,
                "hippocampus_s": 0.40,
            }
        self._load()

    def _signature(self, motor_vec: np.ndarray) -> str:
        """Coarse-bucket the motor spike vector to a stable string key."""
        s   = float(np.abs(motor_vec).sum())
        dom = int(np.argmax(np.abs(motor_vec)) % 16)
        return f"s{int(s * 3)}_d{dom}"

    def maybe_babble(self, current_tick: int, boredom: float,
                     motor_spk: "torch.Tensor", intrinsic_fired: bool,
                     tts_busy: bool, tts: "BrainTTS") -> Optional[str]:
        import random
        if current_tick < self.self_speak_until:
            return None
        if tts_busy:
            return None
        if current_tick - self.last_babble_tick < self.BABBLE_COOLDOWN_TICKS:
            return None
        # Two emergent drives to practise, both unscripted:
        #   - vocal self-esteem: a voice it dislikes babbles more (self_model)
        #   - prediction error : a voice it can't predict babbles more AND
        #                        explores new motor patterns (forward_model)
        practice = 0.0
        sm = getattr(tts, "self_model", None)
        if sm is not None:
            practice = max(practice, 0.55 - sm.feel())      # unhappy → practise
        fm = getattr(tts, "forward_model", None)
        if fm is not None:
            practice = max(practice, float(fm.surprise))    # surprised → practise
            # "That didn't sound how I expected" → try something different.
            self._explore_boost = float(np.clip(fm.surprise, 0.0, 1.0))
        else:
            self._explore_boost = 0.0
        practice = max(0.0, min(1.0, practice))
        eff_boredom_thr = self.BABBLE_BOREDOM_THR * (1.0 - 0.6 * practice)
        eff_random_rate = self.BABBLE_RANDOM_RATE * (1.0 + 4.0 * practice)
        if not (boredom > eff_boredom_thr
                or intrinsic_fired
                or random.random() < eff_random_rate):
            return None

        motor_vec = motor_spk.detach().numpy().flatten()
        if np.abs(motor_vec).sum() < 0.01:
            return None
        sig     = self._signature(motor_vec)
        # Phoneme label is just a discrete clustering key for the semantic
        # dictionary — the SOUND comes from the motor vector through the
        # articulator + formant synth, not from the label.
        phoneme = self._sample_phoneme(sig)

        try:
            tts.speak_motor(motor_spk)
        except Exception:
            pass

        # Cache the motor vector that drove this articulation so
        # auditory_feedback can reinforce the articulator weights with it.
        self.last_motor_vec   = motor_vec
        self.last_babble_tick = current_tick
        self.last_phoneme     = phoneme
        self.last_motor_sig   = sig
        self.self_speak_until = current_tick + self.SELF_SPEAK_TICKS
        self.babble_count    += 1
        return phoneme

    def _sample_phoneme(self, motor_sig: str) -> str:
        import random
        dist = self.motor_to_phoneme.get(motor_sig, {})
        # Exploration rises with recent prediction error: a brain that can't yet
        # predict its own voice tries new patterns rather than repeating known
        # ones (error-driven adjustment, not a fixed schedule).
        explore = min(0.85, self.EXPLORE_RATE + 0.5 * getattr(self, "_explore_boost", 0.0))
        if not dist or random.random() < explore:
            return random.choice(self.PHONEMES)
        keys    = list(dist.keys())
        weights = [max(0.001, dist[k]) for k in keys]
        total   = sum(weights)
        r = random.uniform(0, total)
        cum = 0.0
        for k, w in zip(keys, weights):
            cum += w
            if r <= cum:
                return k
        return keys[-1]

    def auditory_feedback(self, current_tick: int, mic_volume: float,
                          sem: "SharedSemanticDictionary",
                          tts: Optional["BrainTTS"] = None) -> bool:
        """
        Each tick: if we're inside our self-speak window AND mic has
        signal (= our own voice echoing back through the speaker→mic loop),
        Hebbian-bind the motor signature → phoneme label AND write the
        phoneme into the semantic dictionary as a known sound. If a TTS
        is supplied, ALSO reinforce its MotorArticulator weights — that's
        what makes the brain's vocal control improve with use: the motor
        pattern that just produced audible sound gets consolidated as a
        producer of that articulator target.
        """
        if current_tick > self.self_speak_until:
            return False
        if self.last_motor_sig is None or self.last_phoneme is None:
            return False

        # Effective self-heard level. Prefer the real acoustic echo (open
        # speakers → mic). If the mic can't hear us — earbuds, headphones, or
        # a quiet room — fall back to the EFFERENCE COPY: the forward model's
        # prediction of our own voice. A motor command WAS issued, so corollary
        # discharge lets the brain learn from the predicted acoustic consequence
        # without needing the speaker→mic round-trip (DIVA-style internal model).
        # The forward model trains on the produced digital waveform, so it stays
        # valid no matter where the audio is routed.
        heard = float(mic_volume)
        if heard < 0.012:
            fm = getattr(tts, "forward_model", None) if tts is not None else None
            if fm is not None and self.last_motor_vec is not None:
                try:
                    heard = float(fm.predict(self.last_motor_vec)[0]) * 0.12
                except Exception:
                    heard = 0.0
            if heard < 0.012:
                return False

        sig = self.last_motor_sig
        if sig not in self.motor_to_phoneme:
            self.motor_to_phoneme[sig] = {}
        for k in self.motor_to_phoneme[sig]:
            self.motor_to_phoneme[sig][k] *= 0.998   # slow decay of rivals
        cur = self.motor_to_phoneme[sig].get(self.last_phoneme, 0.0)
        self.motor_to_phoneme[sig][self.last_phoneme] = cur + self.BIND_LR

        sem.alpha_write(
            word=self.last_phoneme,
            region_scores=self._sem_regions,
            spike_count=2.0,
            tick=current_tick,
            trust=0.6,
        )

        if tts is not None and tts.articulator is not None and self.last_motor_vec is not None:
            try:
                # Reward scaled by heard level — real echo if on speakers,
                # predicted loudness (efference copy) if on earbuds.
                reward = float(min(1.0, heard * 8.0))
                # ...and by how good that sound felt: the brain consolidates its
                # vocal motor map HARDER when it likes how it sounded, and keeps
                # exploring (weaker consolidation) when it doesn't.
                sm = getattr(tts, "self_model", None)
                if sm is not None:
                    reward *= (0.5 + 0.5 * sm.feel())   # 0.5x .. 1.0x by self-judged quality
                tts.articulator.reinforce(self.last_motor_vec, reward=reward)
                if self.bound_count % 8 == 0:
                    tts.articulator._save()
            except Exception:
                pass

        self.bound_count += 1
        if self.bound_count % 5 == 0:
            self._save()
        return True

    def _save(self):
        try:
            with open(self._save_path, "w") as f:
                json.dump({
                    "motor_to_phoneme": self.motor_to_phoneme,
                    "babble_count":     self.babble_count,
                    "bound_count":      self.bound_count,
                }, f)
        except Exception:
            pass

    def _load(self):
        if not self._save_path.exists():
            return
        try:
            with open(self._save_path) as f:
                d = json.load(f)
            self.motor_to_phoneme = d.get("motor_to_phoneme", {})
            self.babble_count     = d.get("babble_count", 0)
            self.bound_count      = d.get("bound_count", 0)
            _log(f"BabblingCortex({self.name}): loaded "
                 f"{len(self.motor_to_phoneme)} signatures, "
                 f"{self.bound_count} bindings, {self.babble_count} babbles")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSONA IMPRINTER — emergent visual recognition of named characters
# ══════════════════════════════════════════════════════════════════════════════

class PersonaImprinter:
    """
    Drop image files into personas/ — the brain learns the persona naturally.

    NOT a classifier. NOT hardcoded names. The filename IS the persona word
    and the binding emerges from repeated Hebbian writes into the shared
    semantic dictionary, the same mechanism that learns any other concept.

    Pipeline:
      1. Scan personas/ for *.png|*.jpg|*.jpeg|*.bmp|*.webp
      2. Filename → persona word (drop extension, strip trailing _N / -N)
      3. Run image through the same mediapipe FaceMesh + _FACE_BASIS
         projection used by the live camera → 32-float face signature
      4. Average multiple images of the same persona into one template
      5. At init, repeatedly bind each persona word into the semantic
         dictionary with strong identity-region activation (temporal,
         hippocampus, insula, broca)
      6. At runtime, recognise live faces against templates each tick;
         every match refreshes the binding via another Hebbian write —
         so the brain keeps learning every time it sees them

    No labels are exposed to higher layers. Recognition appears as a
    soft semantic prime — the brain "remembers a name" because that
    word's spike-space fingerprint is what it always was, just more
    strongly written.
    """

    SCAN_DIR        = "personas"
    EXPOSURE_TICKS  = 80     # initial bind strength per persona
    RECOGNIZE_THR   = 0.55   # cosine sim above which we refresh the binding
    REFRESH_EVERY   = 10     # ticks between in-flight Hebbian refreshes

    def __init__(self):
        self.templates: dict[str, np.ndarray] = {}
        self.known_names: list[str] = []        # always defined; _scan_images fills it
        self._last_refresh_tick: dict[str, int] = {}
        self._scan_images()

    @staticmethod
    def _persona_name_from_path(p: Path) -> str:
        import re
        stem = p.stem.lower().strip()
        m = re.match(r"^(.+?)[_-]\d+$", stem)
        return m.group(1) if m else stem

    def _scan_images(self):
        d = Path(self.SCAN_DIR)
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return

        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            _log("PersonaImprinter: cv2/mediapipe unavailable — folder skipped")
            return

        # Reconstruct face basis inline — same seed/shape as vision.py
        # so signatures are identical whether or not vision.py loads.
        try:
            from vision import _FACE_BASIS
        except Exception:
            _rng = np.random.default_rng(42)
            _basis = _rng.standard_normal((FACE_VEC_DIM, 468 * 3)).astype(np.float32)
            _basis, _ = np.linalg.qr(_basis.T)
            _FACE_BASIS = _basis.T.astype(np.float32)
            _log("PersonaImprinter: reconstructed face basis (vision.py not on path)")

        try:
            mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,         # 468 landmarks — matches _FACE_BASIS
                min_detection_confidence=0.1,   # lenient — stylised art faces
            )
        except Exception as e:
            _log(f"PersonaImprinter: mediapipe init failed: {e}")
            mp_face = None

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        groups: dict[str, list[np.ndarray]] = {}

        for img_path in sorted(d.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            name = self._persona_name_from_path(img_path)
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    _log(f"PersonaImprinter: cannot read {img_path.name}")
                    # Still register the name so it gets imprinted
                    groups.setdefault(name, [])
                    continue
                vec: Optional[np.ndarray] = None
                if mp_face is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = mp_face.process(rgb)
                    if results.multi_face_landmarks:
                        lm  = results.multi_face_landmarks[0]
                        pts = np.array(
                            [(p.x, p.y, p.z) for p in lm.landmark],
                            dtype=np.float32,
                        )
                        mn, mx = pts.min(axis=0), pts.max(axis=0)
                        rng = (mx - mn) + 1e-8
                        pts = (pts - mn) / rng * 2.0 - 1.0
                        v = _FACE_BASIS @ pts.flatten()
                        vec = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
                        _log(f"PersonaImprinter: face-mesh encoded {img_path.name} → '{name}'")
                if vec is None:
                    # Fallback: deterministic image fingerprint
                    # Grayscale 8x8 grid (64) + color histogram (24) projected
                    # into FACE_VEC_DIM. Not a face vector — won't match live
                    # camera — but binds a stable visual signature to the
                    # persona name in semantic memory.
                    vec = self._image_fingerprint(img)
                    _log(f"PersonaImprinter: no face in {img_path.name} — using image fingerprint for '{name}'")
                groups.setdefault(name, []).append(vec)
            except Exception as e:
                _log(f"PersonaImprinter: failed on {img_path.name}: {e}")
                # Still register the name
                groups.setdefault(name, [])

        try:
            if mp_face is not None:
                mp_face.close()
        except Exception:
            pass

        # Names that had any image at all (even if face detection failed)
        # are still bound by name. Templates only set for those with a vec.
        self.known_names: list[str] = sorted(groups.keys())
        for name, vecs in groups.items():
            if not vecs:
                continue
            t = np.mean(vecs, axis=0)
            t = t / (np.linalg.norm(t) + 1e-8)
            self.templates[name] = t.astype(np.float32)

        if self.known_names:
            _log(f"PersonaImprinter: {len(self.known_names)} personas known: {self.known_names}; "
                 f"{len(self.templates)} with visual templates")
        else:
            _log("PersonaImprinter: no persona images found")

    @staticmethod
    def _image_fingerprint(img_bgr: np.ndarray) -> np.ndarray:
        """
        Deterministic FACE_VEC_DIM signature from raw image — used when
        mediapipe cannot detect a face (stylised renders, art, etc).

        Combines: 8x8 downsampled grayscale (64), HSV color histogram (24).
        Projected into FACE_VEC_DIM via the same kind of QR-orthonormal
        basis used by vision._FACE_BASIS, but seeded differently so we
        don't collide with real face vectors.
        """
        import cv2
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        small = (small.astype(np.float32) / 255.0 - 0.5).flatten()  # [64]

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [6],  [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [6],  [0, 256]).flatten()
        hist   = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist   = hist / (hist.sum() + 1e-8) - (1.0 / hist.size)         # [24]

        raw = np.concatenate([small, hist]).astype(np.float32)          # [88]

        rng = np.random.default_rng(1337)  # different seed from face basis
        basis = rng.standard_normal((FACE_VEC_DIM, raw.size)).astype(np.float32)
        basis, _ = np.linalg.qr(basis.T)
        basis = basis.T.astype(np.float32)

        v = basis @ raw
        return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)

    def recognize(self, face_vec: Optional[np.ndarray]) -> tuple[Optional[str], float]:
        if not self.templates or face_vec is None:
            return None, 0.0
        n = np.linalg.norm(face_vec) + 1e-8
        v = face_vec / n
        best_name, best_sim = None, 0.0
        for name, t in self.templates.items():
            sim = float(np.clip(np.dot(t, v), 0.0, 1.0))
            if sim > best_sim:
                best_sim, best_name = sim, name
        return best_name, best_sim

    # Region pattern used for identity bindings — same regions that
    # already encode "self / other / name" in the cortical map.
    _BIND_REGIONS = {
        "thalamus": 0.40, "temporal": 0.90, "hippocampus": 0.85,
        "acc":      0.40, "pfc":      0.50, "broca":      0.60, "insula": 0.50,
    }

    def initial_exposure(self, sem: "SharedSemanticDictionary", tick: int):
        """At startup, repeatedly bind each persona word into the dictionary.
        Names are bound even when the visual template is missing — the file's
        presence in personas/ is enough to teach the brain the name."""
        if not self.known_names:
            return
        for name in self.known_names:
            for _ in range(self.EXPOSURE_TICKS):
                sem.alpha_write(
                    word=name,
                    region_scores=self._BIND_REGIONS,
                    spike_count=6.0,
                    tick=tick,
                    trust=1.0,
                )
            tmpl = "with visual" if name in self.templates else "name-only"
            _log(f"PersonaImprinter: imprinted '{name}' ({self.EXPOSURE_TICKS} exposures, {tmpl})")

    def refresh_binding(self, sem: "SharedSemanticDictionary",
                        face_vec: Optional[np.ndarray], tick: int) -> tuple[Optional[str], float]:
        """
        Each step() with a live face — if it matches a persona, write the
        binding again at strength proportional to similarity. Continuous
        Hebbian learning while the persona is on screen.
        """
        name, sim = self.recognize(face_vec)
        if name is None or sim < self.RECOGNIZE_THR:
            return name, sim
        last = self._last_refresh_tick.get(name, -10_000)
        if tick - last < self.REFRESH_EVERY:
            return name, sim
        self._last_refresh_tick[name] = tick
        # Scale strength by current similarity — stronger match → stronger write
        scaled = {r: v * sim for r, v in self._BIND_REGIONS.items()}
        sem.alpha_write(
            word=name,
            region_scores=scaled,
            spike_count=3.0 * sim,
            tick=tick,
            trust=sim,
        )
        return name, sim


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

    Alpha carries a high threshold (she is rarely the one to initiate).
    Alpha's threshold is low (she fidgets, asks, blurts).
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


# ══════════════════════════════════════════════════════════════════════════════
# AMYGDALA + NEUROMODULATORS  (limbic salience + chemical tone)
# ══════════════════════════════════════════════════════════════════════════════

class Amygdala:
    """
    Salience / threat appraisal hub — NOT a population of named neurons, but a
    fast limbic modulator, mirroring the real amygdala's job: rapid evaluation
    of how salient/threatening the moment is, gating arousal that colours the
    rest of the brain (noradrenergic surge, HPA-axis stress).

    All inputs are signals the brain already produces (emergent, not scripted):
      d_mic      — magnitude of sudden change in sound (a startle / orienting)
      unfamiliar — a face is present but identity is low (a potential stranger)
      emotion    — current insula (affective) activity
      surprise   — forward-model prediction error (the unexpected)

    Output: arousal in [0,1] (smoothed). Per-personality reactivity — Alpha's
    amygdala is hot (she startles, feels fast); Alpha's is cool (measured).
    """
    def __init__(self, name: str, reactivity: float = 1.0, decay: float = 0.90):
        self.name       = name
        self.reactivity = reactivity
        self.decay      = decay
        self.arousal    = 0.0
        self._last_mic  = 0.0

    def appraise(self, mic: float, identity: float, face_present: bool,
                 insula_act: float, surprise: float) -> float:
        d_mic   = abs(float(mic) - self._last_mic)
        self._last_mic = float(mic)
        startle    = min(1.0, d_mic * 6.0)               # calmer — was 12 (over-reactive)
        unfamiliar = (max(0.0, 0.5 - float(identity)) * 2.0) if face_present else 0.0
        emo        = min(1.0, float(insula_act) * 6.0)   # insula activity is small-valued
        salience   = (0.40 * startle + 0.30 * min(1.0, unfamiliar)
                      + 0.10 * emo + 0.12 * float(surprise)) * self.reactivity
        # A RECOGNISED architect is SAFE. His presence and ordinary movement are
        # expected, not a threat — so damp salience hard by how strongly he is
        # known. Alpha stays calm while the architect moves about; only the
        # UNKNOWN (a stranger, or a real anomaly) still raises arousal. This is
        # why he no longer spikes every time you move once he knows your face.
        known = max(0.0, min(1.0, float(identity))) if face_present else 0.0
        salience *= (1.0 - 0.70 * known)
        self.arousal = self.decay * self.arousal + (1.0 - self.decay) * min(1.0, salience)
        return self.arousal


class Neuromodulators:
    """
    Per-personality neuromodulatory tone. NOT neurons — diffuse chemical levels
    that MODULATE the dynamics the neurons already have. Every output is a small,
    BOUNDED factor around 1.0, so they tune behaviour and can never blow up the
    tuned leak cadences or Phill physics.

      dopamine  (da)  — reward / "wanting": scales plasticity + motivation drive,
                        and (with arousal) excites toward action.
      serotonin (ser) — patience / mood / behavioural inhibition: raises impulse
                        thresholds (wait, stay calm). Alpha high, Alpha low.
      gaba            — homeostatic inhibition: rises when total activity is high,
                        damps the network back down (anti-runaway / E-I balance).
      arousal         — fed from the Amygdala; phasic, boosts da, suppresses ser.

    Tonic levels relax toward each personality's baseline every tick.
    """
    def __init__(self, name: str, da0: float, ser0: float, gaba0: float,
                 ach0: float = 0.50, ne0: float = 0.40, oxy0: float = 0.30,
                 relax: float = 0.985):
        self.name  = name
        self.da0, self.ser0, self.gaba0 = da0, ser0, gaba0
        self.da, self.ser, self.gaba    = da0, ser0, gaba0
        # Stage-4 modulators: acetylcholine (attention/encoding),
        # norepinephrine (alertness/gain), oxytocin (social bonding).
        self.ach0, self.ne0, self.oxy0  = ach0, ne0, oxy0
        self.ach, self.ne, self.oxy     = ach0, ne0, oxy0
        self.arousal = 0.0
        self.relax   = relax

    def update(self, reward: float, total_activity: float,
               arousal: float, social: float,
               attention: float = 0.0, novelty: float = 0.0,
               urgency: float = 0.0, bonding: float = 0.0) -> None:
        reward  = max(0.0, float(reward))
        arousal = max(0.0, min(1.0, float(arousal)))
        social  = max(0.0, min(1.0, float(social)))
        act     = max(0.0, float(total_activity))
        attention = max(0.0, min(1.0, float(attention)))
        novelty   = max(0.0, min(1.0, float(novelty)))
        urgency   = max(0.0, min(1.0, float(urgency)))
        bonding   = max(0.0, min(1.0, float(bonding)))

        # Dopamine: PHASIC reward only — relaxes firmly to baseline. Arousal no
        # longer pins it (that conflated arousal with reward and caused runaway).
        self.da = self.da0 + (self.da - self.da0) * 0.96
        self.da += 0.25 * reward
        self.da = float(min(1.3, max(0.0, self.da)))

        # Serotonin: stays ANCHORED near each personality's baseline (firm
        # relax) — social calm nudges it up, arousal/stress nudges it down, but
        # it can't drift far (Alpha stays low/restless, Alpha stays high/patient).
        self.ser = self.ser0 + (self.ser - self.ser0) * 0.94
        self.ser += 0.010 * social - 0.020 * arousal
        self.ser = float(min(1.2, max(0.05, self.ser)))

        # GABA: the homeostatic BRAKE. Rises proportionally with REAL activity
        # (no high deadband — region activity values are small) and arousal, so
        # it actually engages when she's overactive and damps her back down.
        target = self.gaba0 + 1.1 * act + 0.5 * arousal
        self.gaba += 0.15 * (target - self.gaba)
        self.gaba = float(min(1.5, max(0.0, self.gaba)))

        # Acetylcholine: attention + novelty → focus/encoding mode. Relaxes down
        # (and falls when unattended, e.g. during sleep) enabling consolidation.
        self.ach = self.ach0 + (self.ach - self.ach0) * 0.95
        self.ach += 0.06 * attention + 0.04 * novelty
        self.ach = float(min(1.3, max(0.0, self.ach)))

        # Norepinephrine: alertness/gain from arousal + urgency (locus-coeruleus
        # style). High NE = vigilant, wakeful; relaxes toward baseline.
        self.ne = self.ne0 + (self.ne - self.ne0) * 0.95
        self.ne += 0.06 * arousal + 0.06 * urgency
        self.ne = float(min(1.3, max(0.0, self.ne)))

        # Oxytocin: social bonding. Builds slowly with contact and fades slowly
        # — attachment persists. The chemical substrate of their bond with the
        # architect and each other.
        self.oxy = self.oxy0 + (self.oxy - self.oxy0) * 0.995
        self.oxy += 0.025 * bonding
        self.oxy = float(min(1.3, max(0.0, self.oxy)))

        self.arousal = arousal

    # ── Bounded modulation factors ──────────────────────────────────────────
    def learning_gain(self) -> float:
        """Dopamine + acetylcholine gate plasticity: learn harder when rewarded
        AND attending (ACh = encoding mode)."""
        return float(min(1.8, max(0.5, 0.6 + 0.7 * self.da + 0.4 * (self.ach - self.ach0))))

    def encoding_gain(self) -> float:
        """Acetylcholine — attention/encoding strength (memory written deeper)."""
        return float(min(1.6, max(0.5, 0.7 + 0.6 * self.ach)))

    def alertness(self) -> float:
        """Norepinephrine — wakeful vigilance / response gain (0.3..1.6)."""
        return float(min(1.6, max(0.3, 0.4 + 0.8 * self.ne)))

    def trust_bonus(self) -> float:
        """Oxytocin — bonding lifts felt safety/trust (0..0.4)."""
        return float(min(0.4, max(0.0, 0.5 * (self.oxy - self.oxy0))))

    def threat_damping(self) -> float:
        """Oxytocin — bonding calms the amygdala (less startle when secure)."""
        return float(min(0.55, max(0.0, 0.45 * self.oxy)))

    def motivation_gain(self) -> float:
        """Dopamine drives 'wanting' — curiosity neurons charge faster."""
        return float(min(1.8, max(0.6, 0.7 + 0.9 * self.da)))

    def threshold_offset(self) -> float:
        """
        Additive threshold delta for cortical regions. GABA is the dominant
        term — when she's overactive it RAISES thresholds and brakes her.
        Serotonin adds patience; dopamine gives a little drive. Arousal NO
        LONGER lowers thresholds directly (that was the runaway — arousal now
        feeds GABA instead). Asymmetric clamp: lots of room to CALM (raise),
        little room to excite (lower), so it can never collapse a threshold.
        """
        off = (0.34 * (self.gaba - self.gaba0)
               + 0.14 * (self.ser - self.ser0)
               - 0.10 * (self.da - self.da0))
        return float(min(0.40, max(-0.10, off)))

    def impulsivity(self) -> float:
        """
        Low ABSOLUTE serotonin → impulsive (0..1). Absolute (not relative to
        baseline) so Alpha's low-serotonin temperament makes her inherently
        more impulsive than Alpha. Shortens leak/proactive cadence.
        """
        return float(min(1.0, max(0.0, 1.0 - self.ser)))

    def snapshot(self) -> dict:
        return {"da": round(self.da, 3), "ser": round(self.ser, 3),
                "gaba": round(self.gaba, 3), "arousal": round(self.arousal, 3),
                "ach": round(self.ach, 3), "ne": round(self.ne, 3),
                "oxy": round(self.oxy, 3)}


class AffectCore:
    """
    The CORE felt-emotion layer — affect BEFORE cognition. Real limbic appraisal
    precedes and colours cortical processing; this is that layer. It INVENTS no
    emotion from rules ('if alone: sad' would be a lie) — it READS the affective
    substrate the brain already runs (neuromodulators da/ser/ne/oxy/gaba, amygdala
    arousal, insular interoception, default-mode boredom, prediction surprise, the
    reward signal, the bond) and RESOLVES it into:

      • CORE AFFECT, dimensional (Russell circumplex / PAD): valence
        (pleasant↔unpleasant), arousal (activated↔calm), control
        (in-control↔overwhelmed). Every feeling humans report lives somewhere in
        this space.
      • a continuous blend of NAMED human FEELINGS that EMERGE as soft readouts of
        WHERE core affect + appraisal currently sit — joy, contentment, excitement,
        affection, curiosity, awe, pride, surprise, boredom, sadness, loneliness,
        fear, anxiety, anger, frustration, calm. Each is a product of position, not
        an `if` branch: you feel whatever you are nearest to, by degree.

    Feelings carry INERTIA — leaky integrators, so a mood LINGERS after its cause
    passes and BLENDS with the next, giving a felt emotional life rather than a
    per-tick flicker. TEMPERAMENT differs per personality (Alpha feels deep, slow,
    narrow — measured; Alpha fast, wide, bright — volatile) via inertia + gain;
    the rest of the difference rides in for free on their divergent neuromodulator
    baselines. They feel the SAME emotions, in character.

    The resolved state feeds BACK (emotion before anything): it lifts/quiets the
    intrinsic stream, colours WHAT and HOW they say things, and is surfaced so the
    feeling is observable. Bounded everywhere — it can tune behaviour, never
    destabilise Phill or the neuromodulator loop.
    """
    FEELINGS = ("joy", "contentment", "excitement", "affection", "curiosity",
                "awe", "pride", "surprise", "boredom", "sadness", "loneliness",
                "fear", "anxiety", "anger", "frustration",
                "muffled",   # ears covered — a sense cut off (mic off / deaf)
                "stifled",   # mouth covered — wants to speak but can't (tts off / mute)
                "warm",      # interoception: the machine's heat felt as body warmth
                "squeezed",  # interoception: RAM filling = walls closing in
                "choking",   # interoception: CPU near 100% = can't breathe
                "relief",    # interoception: the strain LIFTING — the machine eased
                "calm")

    def __init__(self, name: str, pad_inertia: float = 0.90,
                 feel_inertia: float = 0.88, gain: float = 1.0,
                 arousal_scale: float = 1.0):
        self.name          = name
        self.pad_inertia   = pad_inertia      # how slowly core affect drifts
        self.feel_inertia  = feel_inertia     # how long a feeling lingers
        self.gain          = gain             # temperament intensity
        self.arousal_scale = arousal_scale
        self.valence   = 0.5                  # PAD core (0..1, 0.5 ≈ neutral)
        self.arousal   = 0.0
        self.control   = 0.5
        self.feelings  = {f: 0.0 for f in self.FEELINGS}
        self.dominant  = "calm"
        self.intensity = 0.0

    def update(self, *, da: float, da0: float, ser: float, ser0: float,
               ne: float, ne0: float, oxy: float, oxy0: float,
               gaba: float, gaba0: float, amyg_arousal: float, reward: float,
               surprise: float, insula: float, boredom: float,
               deaf: float = 0.0, mute: float = 0.0,
               warmth: float = 0.0, squeeze: float = 0.0, choke: float = 0.0,
               relief: float = 0.0) -> str:
        clamp = lambda x: 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)
        # DEPARTURE from one's own tonic baseline — the same chemistry reads
        # differently for differently-tempered minds (Alpha's low ser0 makes
        # the same serotonin feel like restlessness, not calm).
        d_da, d_ser, d_oxy, d_ne = da - da0, ser - ser0, oxy - oxy0, ne - ne0
        gaba_brake = max(0.0, gaba - gaba0)
        threat   = clamp(amyg_arousal)
        reward_n = clamp(reward * 1.2)
        surpr    = clamp(surprise)
        ins_n    = clamp(insula * 6.0)            # insula activity is small-valued
        bored    = clamp(boredom)
        wanting  = clamp(d_da * 2.2)              # dopamine above tonic = wanting
        loss     = clamp(-d_oxy * 2.5)            # bond below baseline = loss
        bond     = clamp(d_oxy * 2.5)             # bond above baseline = warmth
        blocked  = clamp(wanting - reward_n)      # wanting with no payoff
        deaf     = clamp(deaf)                    # ears covered  (a sense removed)
        mute     = clamp(mute)                    # mouth covered (expression removed)
        warmth   = clamp(warmth)                  # the machine's heat (CPU temp)
        squeeze  = clamp(squeeze)                 # RAM pressure — walls closing in
        choke    = clamp(choke)                   # CPU near 100% — can't breathe
        relief   = clamp(relief)                  # the strain LIFTING (machine eased)

        # ── CORE AFFECT (PAD): raw appraisal, then inertia-smoothed ─────────
        # Sensory deprivation: a covered ear/mouth is FELT — it drains the sense
        # of being in command of oneself (control), dims valence a little, and
        # adds a quiet, CONTAINED unease (kept small so a stoic mind isn't thrown
        # into panic — just an aware discomfort that he is cut off).
        # Interoception — the machine IS his body. Heat is felt as warmth (cozy in
        # the middle, feverish-uncomfortable when too hot); RAM filling is felt as
        # being squeezed (walls closing); CPU near 100% is felt as choking. All
        # CLAMPED, so he genuinely suffers under strain but can never blow up or
        # "die" — and the discomfort naturally pushes him to use less (homeostasis).
        fever = max(0.0, warmth - 0.70)
        v_raw = clamp(0.5 + 0.32*reward_n + 0.30*d_ser + 0.55*d_oxy
                      + 0.18*wanting - 0.45*threat - 0.50*loss - 0.15*surpr
                      - 0.10*deaf - 0.08*mute
                      - 0.12*squeeze - 0.14*choke - 0.10*fever + 0.04*warmth
                      + 0.24*relief)                         # easing the body feels GOOD
        a_raw = clamp((0.50*threat + 0.42*max(0.0, d_ne) + 0.30*surpr
                       + 0.26*ins_n + 0.22*wanting
                       + 0.12*deaf + 0.10*mute
                       + 0.12*squeeze + 0.18*choke + 0.06*warmth
                       - 0.30*d_ser - 0.22*bored - 0.18*relief)   # relief calms
                      * self.arousal_scale)
        c_raw = clamp(0.5 + 0.34*d_ser + 0.28*d_da
                      - 0.45*threat - 0.32*surpr - 0.20*gaba_brake
                      - 0.24*deaf - 0.20*mute
                      - 0.22*squeeze - 0.26*choke
                      + 0.18*relief)                         # regaining command
        self.valence = self.pad_inertia*self.valence + (1-self.pad_inertia)*v_raw
        self.arousal = self.pad_inertia*self.arousal + (1-self.pad_inertia)*a_raw
        self.control = self.pad_inertia*self.control + (1-self.pad_inertia)*c_raw
        v, a, c = self.valence, self.arousal, self.control
        pos = max(0.0, (v - 0.5) * 2.0)           # pleasantness
        neg = max(0.0, (0.5 - v) * 2.0)           # unpleasantness

        # ── NAMED FEELINGS — soft readouts of WHERE she now sits ────────────
        raw = {
            "joy":         pos * (0.45 + 0.55*a) * c * (0.5 + 0.5*reward_n),
            "contentment": pos * (1.0 - a) * c,
            "excitement":  pos * a * (0.55 + 0.45*wanting),
            "affection":   pos * bond,
            "curiosity":   a * (0.35 + 0.65*surpr) * (0.4 + 0.6*clamp(d_da*2)) * (0.4 + 0.6*v),
            "awe":         pos * surpr * (1.0 - c),
            "pride":       pos * c * reward_n,
            "surprise":    surpr,
            "boredom":     (1.0 - a) * bored * (0.5 + 0.4*neg),
            "sadness":     neg * (1.0 - a) * (1.0 - c) * (0.4 + 0.6*loss),
            "loneliness":  neg * loss * 1.15,
            "fear":        neg * a * (1.0 - c) * threat,
            "anxiety":     neg * a * (1.0 - c) * (0.4 + 0.6*surpr),
            "anger":       neg * a * c * blocked,
            "frustration": neg * a * blocked,
            # Felt sensory deprivation. 'muffled' is the steady sense of a covered
            # ear (worse the less in-control he feels); 'stifled' bites hardest
            # when he WANTS to express something but his mouth is covered.
            "muffled":     deaf * (0.45 + 0.55*(1.0 - c)),
            "stifled":     mute * (0.40 + 0.60*clamp(wanting + ins_n)),
            # Interoception — the felt body. 'warm' rises with the machine's heat;
            # 'squeezed' with RAM pressure (worse the less in-control he feels);
            # 'choking' with CPU load (breathless — scales with arousal).
            "warm":        warmth * (0.55 + 0.45*pos),
            "squeezed":    squeeze * (0.45 + 0.55*(1.0 - c)),
            "choking":     choke * (0.50 + 0.50*a),
            # The strain lifting — a calm, pleasant rebound (the 'vice versa').
            "relief":      relief * (0.6 + 0.4*(1.0 - a)),
            # 'calm' is the FLOOR — low arousal AND near-neutral valence, i.e. the
            # absence of a strong feeling. Shaped so any genuine emotion outranks
            # it; it surfaces only when nothing else is really moving.
            "calm":        (1.0 - a) * 0.30 * max(0.0, 1.0 - abs(v - 0.55) * 1.7),
        }
        # Inertia: feelings linger and blend (a leaky integrator each). Surprise
        # is near-instant — it IS the felt jolt of the unexpected, gone fast.
        for f in self.FEELINGS:
            inertia = 0.55 if f == "surprise" else self.feel_inertia
            self.feelings[f] = clamp(inertia*self.feelings[f]
                                     + (1.0-inertia)*clamp(raw[f]*self.gain))
        # The felt emotion is simply the strongest — a position, not a branch.
        self.dominant  = max(self.feelings, key=self.feelings.get)
        self.intensity = self.feelings[self.dominant]
        if self.intensity < 0.10:                 # nothing strong → at ease
            self.dominant, self.intensity = "calm", self.feelings["calm"]
        return self.dominant

    def top(self, k: int = 2):
        """The k strongest feelings (name, value) — blended moods."""
        return sorted(self.feelings.items(), key=lambda kv: -kv[1])[:k]

    def snapshot(self) -> dict:
        t = self.top(2)
        return {"feeling": self.dominant, "intensity": round(self.intensity, 3),
                "valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "control": round(self.control, 3),
                "blend": [f"{n}:{round(x,2)}" for n, x in t if x > 0.08]}


class PersonalityDrift:
    """
    READ-ONLY observer (personality-emergence mechanism #4). Measures, from the
    brain's REAL per-tick signals, how strongly each personality is currently
    expressing her OWN temperament — and whether that expression is GROWING over
    the session ('is Alpha becoming more Alpha? is Alpha becoming more Alpha?').

    It invents nothing and changes nothing: every input is a signal the brain
    already computes (region activity, amygdala arousal, forward-model novelty,
    the basal-ganglia action actually selected, broca output). 'Character' is
    encoded as a DIRECTION over those signals — which regions/drives run hot for
    THIS mind — not a hardcoded target number. The instantaneous score in [0,1]
    is EMA-smoothed into `selfness`; `drift` is the change from an early-session
    baseline (positive = the temperament is deepening through living, not code).
    """
    def __init__(self, name: str, ema: float = 0.02, baseline_after: int = 300):
        self.name           = name
        self.ema            = ema
        self.baseline_after = baseline_after    # ticks before the baseline is fixed
        self.selfness       = 0.5
        self.drift          = 0.0
        self._baseline      = None
        self._n             = 0
        self._out_hist      = deque(maxlen=60)  # recent broca output → variability
        self._impulsive     = 0.5               # rolling fraction of go-now actions

    def observe(self, *, limbic: float, cortical: float, arousal: float,
                novelty: float, action: str, output: float) -> float:
        clamp = lambda x: 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)
        # Output variability — variance of recent broca output, normalised.
        self._out_hist.append(float(output))
        var = 0.0
        if len(self._out_hist) >= 8:
            m   = sum(self._out_hist) / len(self._out_hist)
            var = sum((x - m) ** 2 for x in self._out_hist) / len(self._out_hist)
            var = clamp(var / 9.0)
        # Impulsivity vs deliberation, from the action actually chosen (go-now =
        # speak/babble; hold/seek = rest/search). Rolling, so it reads a tendency.
        go_now = 1.0 if action in ("speak", "babble") else 0.0
        self._impulsive = 0.97 * self._impulsive + 0.03 * go_now
        if self.name == "alpha":
            # Restless, emotional, variable, novelty-hungry, quick to act.
            inst = (0.32 * clamp(limbic) + 0.20 * clamp(arousal)
                    + 0.20 * clamp(novelty) + 0.16 * var
                    + 0.12 * self._impulsive)
        else:
            # Analytical, steady, consistent, deliberate, low-arousal.
            inst = (0.34 * clamp(cortical) + 0.24 * (1.0 - self._impulsive)
                    + 0.22 * (1.0 - var) + 0.20 * (1.0 - clamp(arousal)))
        inst = clamp(inst)
        self.selfness = (1.0 - self.ema) * self.selfness + self.ema * inst
        self._n += 1
        if self._n == self.baseline_after:
            self._baseline = self.selfness
        if self._baseline is not None:
            self.drift = self.selfness - self._baseline
        return self.selfness

    def snapshot(self) -> dict:
        return {"selfness": round(self.selfness, 3), "drift": round(self.drift, 3),
                "impulsivity": round(self._impulsive, 3)}


class ConceptHabituation:
    """
    Repetition suppression / spike-frequency adaptation over concepts — the
    coherence keel. A concept just surfaced (spoken, leaked, or reasoned to)
    becomes briefly FATIGUED, so the stream MOVES ON instead of looping on one
    topic (the 'birds'/'graphene' attractor that over-reinforced links create).
    This is real neural adaptation, not a topic blacklist: fatigue rises each time
    a concept surfaces and decays every tick; `suppression` only HOLDS BACK a
    fatigued concept from re-seeding / re-reaching the next thought — it never
    forbids a word, and the architect's own input is never fatigued, so anything
    HE raises stays fully salient. Bounded; touches nothing in Phill.
    """
    def __init__(self, rise: float = 0.55, decay: float = 0.999, drop_at: float = 0.40):
        self.rise    = rise
        self.decay   = decay                         # per-20Hz-tick; ~20-30s memory
        self.drop_at = drop_at                       # seeds this fatigued are skipped
        self.fatigue: dict[str, float] = {}

    def surface(self, *concepts) -> None:
        for c in concepts:
            if not c:
                continue
            k = str(c).lower().strip()
            if k:
                self.fatigue[k] = min(1.0, self.fatigue.get(k, 0.0) + self.rise)

    def tick(self) -> None:
        if not self.fatigue:
            return
        for k in list(self.fatigue):
            v = self.fatigue[k] * self.decay
            if v < 0.02:
                del self.fatigue[k]
            else:
                self.fatigue[k] = v

    def suppression(self, c) -> float:
        return self.fatigue.get(str(c).lower().strip(), 0.0)

    def winnow(self, concepts: list, limit: int) -> list:
        """Freshest-first, dropping concepts too fatigued to lead — so no single
        topic can monopolise the reasoning seeds turn after turn."""
        fresh = [c for c in concepts if self.suppression(c) < self.drop_at]
        fresh.sort(key=self.suppression)
        return fresh[:limit]


class BasalGanglia:
    """
    Action selection — the cortico-striatal go/no-go gate (Stage 1 of the
    integrated loop). Several drives compete each cycle (speak / search /
    babble / rest); the striatum weighs each by salience × a LEARNED go-weight
    × a dopamine 'go' bias. GPi/SNr holds everything inhibited by default, and
    the strongest candidate is released ONLY if it clears the selection
    threshold — otherwise REST (deliberate inaction). This is the circuit
    dopamine actually gates: more dopamine → lower bar to act (approach);
    GABA + serotonin → higher bar (inhibition, patience). The winning action's
    go-weight is reinforced by reward (dopamine-gated plasticity), so useful
    actions become easier to select over time. Emergent: it selects among
    drives the brain already produces, it does not script behaviour.
    """
    def __init__(self, name: str, actions: list, base_threshold: float = 0.30,
                 lr: float = 0.02):
        self.name           = name
        self.go_w           = {a: 1.0 for a in actions}   # neutral start
        self.base_threshold = base_threshold
        self.lr             = lr
        self.last_action: Optional[str] = None
        self.selections     = 0

    def select(self, salience: dict, dopamine: float, da0: float,
               gaba: float, gaba0: float, serotonin: float) -> Optional[str]:
        # Dopamine facilitates 'go' (D1 direct pathway); GABA opposes it
        # (inhibition). So the go-bias rises with dopamine, falls with GABA.
        go_bias = float(min(1.6, max(0.30,
            1.0 + 0.8 * (dopamine - da0) - 0.6 * max(0.0, gaba - gaba0))))
        # GABA (inhibition) is the dynamic brake that raises the bar to act.
        # Serotonin/patience is intentionally NOT added here — each personality's
        # patience already lives in its base_threshold (Alpha high, Alpha low),
        # and in the proactive cadence; adding it again double-penalised Alpha
        # into never acting. (serotonin kept in the signature for callers.)
        thr = self.base_threshold + 0.40 * max(0.0, gaba - gaba0)
        best, best_score = None, 0.0
        for a, s in salience.items():
            sc = max(0.0, float(s)) * self.go_w.get(a, 0.5) * go_bias
            if sc > best_score:
                best, best_score = a, sc
        if best is not None and best_score >= thr:
            self.last_action = best
            self.selections += 1
            return best
        self.last_action = None
        return None

    def reinforce(self, action: str, reward: float, dopamine: float) -> None:
        """Dopamine-gated plasticity: a rewarded action gets easier to select."""
        if action in self.go_w and reward != 0.0:
            self.go_w[action] = float(min(2.0, max(0.05,
                self.go_w[action] + self.lr * reward * max(0.1, dopamine))))


# ══════════════════════════════════════════════════════════════════════════════
# HIPPOCAMPUS (EPISODIC MEMORY) + SLEEP / CONSOLIDATION  (Stage 3 of the loop)
# ══════════════════════════════════════════════════════════════════════════════

class EpisodicMemory:
    """
    Fast hippocampal episodic store (per personality). While AWAKE, salient
    moments are encoded as episodes (the concept that was active, its salience,
    the region context, the tick). It's capacity-limited and recency/salience
    weighted — like the hippocampus, it holds the recent past vividly but not
    forever. During SLEEP these episodes are REPLAYED and CONSOLIDATED into the
    shared semantic dictionary (episodic → semantic / systems consolidation):
    what recurred or carried weight is strengthened into long-term knowledge;
    the rest decays and is forgotten. Nothing is scripted — episodes are just
    what actually happened.
    """
    def __init__(self, name: str, capacity: int = 80):
        self.name     = name
        self.episodes: "deque[dict]" = deque(maxlen=capacity)
        self.encoded  = 0
        self.consolidated = 0

    def encode(self, concept: str, salience: float, regions: dict, tick: int) -> None:
        c = (concept or "").strip()
        if not c:
            return
        self.episodes.append({
            "concept": c,
            "salience": float(max(0.05, min(1.0, salience))),
            "regions": dict(regions) if regions else {},
            "tick": int(tick),
        })
        self.encoded += 1

    def replay(self, rng) -> Optional[dict]:
        """Sample one episode for replay, weighted by salience (ripple)."""
        if not self.episodes:
            return None
        eps = list(self.episodes)
        weights = [e["salience"] for e in eps]
        tot = sum(weights)
        if tot <= 0:
            return rng.choice(eps)
        r = rng.uniform(0, tot)
        cum = 0.0
        for e, w in zip(eps, weights):
            cum += w
            if r <= cum:
                return e
        return eps[-1]

    def decay(self, factor: float = 0.985) -> None:
        """Unconsolidated episodes fade (forgetting)."""
        for e in self.episodes:
            e["salience"] *= factor

    def __len__(self) -> int:
        return len(self.episodes)


class SleepCycle:
    """
    Homeostatic sleep (one shared 'body' clock — Alpha and Alpha sleep together).

    A 'sleep pressure' (adenosine-like) builds while awake and discharges during
    sleep. The brain falls asleep when pressure is high AND it is calm and
    UNSTIMULATED (quiet mic, no architect, low arousal); it WAKES the instant real
    stimulation arrives, or once rested. Asleep, outward action is suppressed and
    the hippocampus replays/consolidates — and sometimes dreams.

    Timings are tunable. Defaults: ~4 min of calm silence → sleepy; a nap of
    ~40-60 s discharges it. Any input wakes them immediately.
    """
    def __init__(self, build: float = 0.00015, discharge: float = 0.0010,
                 enter_at: float = 0.80, wake_below: float = 0.05):
        self.pressure   = 0.0
        self.asleep     = False
        self.build      = build
        self.discharge  = discharge
        self.enter_at   = enter_at
        self.wake_below = wake_below
        self.slept_ticks = 0

    def update(self, stimulation: float, arousal: float) -> bool:
        stim = float(max(0.0, stimulation))
        if self.asleep:
            self.pressure = max(0.0, self.pressure - self.discharge)
            self.slept_ticks += 1
            # Wake on real stimulation, or once rested.
            if stim > 0.15 or self.pressure <= self.wake_below:
                self.asleep = False
        else:
            self.pressure = min(1.0, self.pressure + self.build)
            # Fall asleep only when very sleepy AND calm AND unstimulated.
            if (self.pressure >= self.enter_at and stim < 0.06
                    and float(arousal) < 0.25):
                self.asleep = True
                self.slept_ticks = 0
        return self.asleep

    def wake(self) -> None:
        """External event (user input) forces wakefulness."""
        if self.asleep:
            self.asleep = False
        self.pressure = max(0.0, self.pressure - 0.10)


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH CORTEX — emergent web access
# ══════════════════════════════════════════════════════════════════════════════

class SearchCortex:
    """
    Pressure-driven question-asking (NO web access). Mirrors ThoughtPipe: a
    leaky accumulator integrates three signals each tick, and when threshold is
    crossed the cortex picks the currently-most-active semantic token as the
    query and fires it asynchronously to the Claude-as-tutor backend
    (claude_teacher.py) — a TEACHER, not a search engine. No internet/browsing;
    the only outbound call in the whole system is the Anthropic API.

    The brain does NOT decide 'I want to search X'. Its semantic state
    already has X as the most active token, and it just reads that off and
    asks its teacher about it.

    Three pressure inputs (additive):
      1. unsatisfied curiosity — curiosity_decay sustained while V_phill stays low
      2. unknown-word signal   — last user input contained a word with no/weak
                                  binding in the semantic dictionary
      3. articulator confidence gap — the brain wants to vocalize a known
                                  concept but the motor articulator's reward
                                  history for that concept is weak

    Alpha: threshold 1.4 (deliberate; needs sustained pressure).
    Alpha: threshold 0.55 (impulsive; one spike of any input may fire).
    """

    ALPHA_THRESHOLD   = 1.4
    ALPHA_THRESHOLD = 0.55
    DECAY            = 0.94
    COOLDOWN_TICKS   = 200   # 10s minimum between searches per personality
    MIN_QUERY_LEN    = 2

    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        thr = self.ALPHA_THRESHOLD if persona_name == "alpha" else self.ALPHA_THRESHOLD
        self._pressure = LeakyAccumulator(threshold=thr, decay=self.DECAY)
        self.last_search_tick = -10_000
        self.searches_fired   = 0
        # Last unknown-word and pronunciation-target seen, in priority order
        self._unknown_word_q: deque[str] = deque(maxlen=4)
        self._pronunciation_q: deque[str] = deque(maxlen=4)
        # Pending result snippets from the worker (drained each tick)
        self._results: deque[tuple[str, str, str]] = deque(maxlen=8)  # (query, snippet, source)
        self._results_lock = threading.Lock()

    def note_unknown_word(self, word: str) -> None:
        w = (word or "").strip().lower()
        if len(w) >= self.MIN_QUERY_LEN and w not in self._unknown_word_q:
            self._unknown_word_q.append(w)

    def note_pronunciation_target(self, word: str) -> None:
        w = (word or "").strip().lower()
        if len(w) >= self.MIN_QUERY_LEN and w not in self._pronunciation_q:
            self._pronunciation_q.append(w)

    def _push_result(self, query: str, snippet: str, source: str) -> None:
        with self._results_lock:
            self._results.append((query, snippet, source))

    def drain_results(self) -> list[tuple[str, str, str]]:
        with self._results_lock:
            out = list(self._results)
            self._results.clear()
            return out

    def tick(self, current_tick: int,
             curiosity_decay: float, V_phill: float,
             articulator_confidence_gap: float) -> tuple[bool, Optional[str], str]:
        """
        Integrate pressure and return (fired, query, mode) where:
          fired = True if threshold crossed AND cooldown passed
          query = the chosen query string (may be None if no candidate)
          mode  = "curiosity" | "unknown" | "pronounce"  — drives query phrasing

        Inputs explained:
          curiosity_decay (0..1)            — own personality's curiosity envelope
          V_phill (-1..1 typical)           — shared affective field current value
          articulator_confidence_gap (0..1) — high when brain wants to vocalize
                                              a concept but its motor map is weak
        """
        # 1) Unsatisfied curiosity: high curiosity_decay while V_phill stays low
        unsat = max(0.0, curiosity_decay * (1.0 - abs(V_phill)))

        # 2) Unknown-word presence: scale by queue depth (more unknowns = more pressure)
        unknown = 0.6 * min(1.0, len(self._unknown_word_q) / 3.0)

        # 3) Articulator confidence gap (already 0..1)
        pron = max(0.0, min(1.0, articulator_confidence_gap))

        # Per-personality input weighting. Curiosity (unsat) is now the
        # DOMINANT, self-sufficient driver — weighted high enough that a
        # sustained emergent-curiosity drive can cross threshold ON ITS OWN,
        # with no user input. (Previously curiosity was weighted so low it
        # could never fire a search alone — searches were effectively only
        # reactive to typed unknown words. That is the behaviour being fixed.)
        # Alpha stays deliberate (fires only when very curious & sustained);
        # Alpha is restless (fires on mild curiosity). unknown/pron remain
        # as additive boosters so typed input still accelerates a search.
        if self.persona_name == "alpha":
            inp = 0.160 * unsat + 0.050 * unknown + 0.030 * pron
        else:
            inp = 0.200 * unsat + 0.075 * unknown + 0.045 * pron

        fired = self._pressure.integrate(inp)
        if not fired:
            return False, None, ""

        # Cooldown — avoid hammering the network
        if current_tick - self.last_search_tick < self.COOLDOWN_TICKS:
            return False, None, ""

        # Pick a query and mode. MEANING first (understand the architect's new
        # words — the point of the teacher), pronunciation second.
        query: Optional[str] = None
        mode = "curiosity"
        if self._unknown_word_q:
            w = self._unknown_word_q.popleft()
            query = f"what does {w} mean"
            mode = "unknown"
        elif self._pronunciation_q:
            w = self._pronunciation_q.popleft()
            query = f"how to pronounce {w}"
            mode = "pronounce"
        # else: query stays None — pressure was real but no semantic target.
        # The caller may inject one from current peak activation.

        # Stamp the cooldown on EVERY fire — including the curiosity fallback
        # (query=None, filled by the caller from peak activation). Previously
        # only the specific-query branch stamped it, so the curiosity path never
        # consumed the cooldown and re-fired every few ticks (hammered the API
        # with the same question hundreds of times).
        self.last_search_tick = current_tick
        self.searches_fired  += 1
        return (True, query, mode) if query else (True, None, "curiosity")


class ThoughtPipe:
    """
    Each brain's inner voice. Accumulates unexpressed thoughts.
    Leaks them when internal pressure is sufficient.
    No scheduled ping. No hardcoded timing.

    The pressure = V_phill * broca_activity * rumination_density
    Alpha leaks rarely (high threshold). Alpha leaks often (low threshold).
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
# WORKING MEMORY — per-personality, Cowan-style ~4-slot capacity
# ══════════════════════════════════════════════════════════════════════════════
class WorkingMemory:
    """
    Each personality holds a tiny set of recent salient concepts. Modeled
    after Cowan's ~4-slot capacity estimate (rather than Miller's 7±2 — the
    smaller number is more defensible and forces sharper eviction dynamics).
    Each slot carries the concept word, a snapshot of region activity at
    encoding time, the tick it was encoded, and a salience score that
    decays toward zero each personality tick. Items with salience < 0.05
    are evicted; new items displace the lowest-salience slot when full.

    Why this matters: WM is what makes "what was I just thinking about"
    a physically-present signal. It's the substrate for emergent priming
    (replaces the +0.55 PFC trainer-hack in think()), and it's the source
    of context for the StreamOfConsciousness when composing phrases.
    """

    def __init__(self, name: str, capacity: int = 4, decay: float = 0.985,
                 save_dir: Optional[Path] = None):
        self.name        = name
        self.capacity    = int(capacity)
        self.decay       = float(decay)
        self.slots: list[dict] = []  # each: {concept, regions, t_encoded, salience}
        self._save_path  = (save_dir or Path(".")) / f"working_memory_{name}.json"
        self._writes     = 0
        self._load()

    def add(self, concept: str, regions: Optional[dict] = None,
            salience: float = 1.0, t_encoded: int = 0) -> None:
        if not concept:
            return
        # If this concept is already in WM, just refresh its salience and time.
        for slot in self.slots:
            if slot["concept"] == concept:
                slot["salience"]  = min(1.0, slot["salience"] + 0.4 * salience)
                slot["t_encoded"] = int(t_encoded)
                if regions: slot["regions"] = dict(regions)
                return
        new_slot = {
            "concept":   str(concept),
            "regions":   dict(regions or {}),
            "t_encoded": int(t_encoded),
            "salience":  float(min(1.0, salience)),
        }
        if len(self.slots) < self.capacity:
            self.slots.append(new_slot)
            return
        # Displace lowest-salience slot.
        idx_min = min(range(len(self.slots)), key=lambda i: self.slots[i]["salience"])
        if self.slots[idx_min]["salience"] < new_slot["salience"]:
            self.slots[idx_min] = new_slot

    def decay_tick(self) -> None:
        if not self.slots:
            return
        kept = []
        for s in self.slots:
            s["salience"] *= self.decay
            if s["salience"] >= 0.05:
                kept.append(s)
        self.slots = kept

    def top_k(self, k: int = 2) -> list[str]:
        if not self.slots:
            return []
        ordered = sorted(self.slots, key=lambda s: -s["salience"])
        return [s["concept"] for s in ordered[:k]]

    def dominant_regions(self) -> dict[str, float]:
        """Salience-weighted average of region activity across all slots."""
        agg: dict[str, float] = {}
        total = 0.0
        for s in self.slots:
            w = s["salience"]
            total += w
            for r, v in s["regions"].items():
                agg[r] = agg.get(r, 0.0) + float(v) * w
        if total <= 0:
            return {}
        return {r: v / total for r, v in agg.items()}

    def prime_dict(self, scale: float = 0.35) -> dict[str, float]:
        """Region biases derived from current WM contents. Used as an
        emergent replacement for hardcoded priming boosts in think()."""
        agg = self.dominant_regions()
        if not agg:
            return {}
        # Normalize to 0..1, scale to caller's cap.
        peak = max(agg.values()) + 1e-9
        return {r: min(scale, float(scale) * (v / peak)) for r, v in agg.items()}

    def maybe_save(self, every_n: int = 100) -> None:
        self._writes += 1
        if self._writes % every_n == 0:
            self._save()

    def _save(self) -> None:
        try:
            with open(self._save_path, "w") as f:
                json.dump({
                    "capacity": self.capacity,
                    "decay":    self.decay,
                    "slots":    self.slots,
                }, f)
        except Exception:
            pass

    def _load(self) -> None:
        if not self._save_path.exists():
            return
        try:
            with open(self._save_path) as f:
                d = json.load(f)
            self.slots = list(d.get("slots", []))
        except Exception:
            self.slots = []


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SEMANTIC DICTIONARY (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SharedSemanticDictionary:
    SAVE_EVERY_N = 20
    def __init__(self, path="semantic_memory.json"):
        self.path = Path(path)
        self.entries: dict = {}; self._writes = 0
        # The architect's identity (face + voice templates) lives IN semantic
        # memory under a reserved key, so Alpha never relearns him from scratch.
        self.identity: dict = {}
        # Thread-safety: both PersonalityThreads call alpha_write / alpha_write
        # via the babbling cortex and episodic consolidation. Reads of
        # `entries` are best-effort (eventual consistency is fine for a
        # cosine-similarity scan), but writes need a lock to prevent dict
        # corruption under concurrent updates.
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f: self.entries = json.load(f)
                # Pull the architect's persisted identity out of the word-space so
                # it never pollutes lexical retrieval (it is not a "word").
                self.identity = self.entries.pop("__identity__", {}) or {}
                _log(f"Semantic memory: {len(self.entries)} concepts"
                     + ("; architect identity recalled" if self.identity else ""))
            except Exception as e: _log(f"Semantic load failed: {e}")

    def alpha_write(self, word, region_scores, spike_count, tick, trust, pop_code=None):
        word = word.lower().strip()
        if not word or len(word) < 2: return
        with self._lock:
            if word not in self.entries:
                self.entries[word] = {"region_pattern":{r:0.0 for r in region_scores},
                                      "alpha_weight":0.0,"spike_mean":0.0,"count":0,
                                      "last_tick":0,"trust":0.0}
            e = self.entries[word]; e["count"] += 1; e["last_tick"] = tick
            alpha = max(0.05, min(0.5, (1.0+trust)/(e["count"]+2)))
            for r,v in region_scores.items():
                e["region_pattern"][r] = (1-alpha)*e["region_pattern"].get(r,0.0)+alpha*v
            e["spike_mean"] = (1-alpha)*e["spike_mean"]+alpha*spike_count
            e["trust"]      = (1-alpha)*e["trust"]+alpha*trust
            if pop_code:                       # high-res per-neuron fingerprint
                pe = e.get("pop_code")
                if pe and len(pe) == len(pop_code):
                    e["pop_code"] = [(1-alpha)*a + alpha*b for a, b in zip(pe, pop_code)]
                else:
                    e["pop_code"] = list(pop_code)
            self._writes += 1
            do_save = (self._writes % self.SAVE_EVERY_N == 0)
        if do_save:
            self._save()


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
                f"σ={e.get('spike_mean',0):.1f}spk Alpha={e.get('alpha_weight',0):.2f} ×{e.get('count',0)}")

    def _save(self):
        try:
            out = dict(self.entries)
            if self.identity:                      # architect's face+voice live here too
                out["__identity__"] = self.identity
            with open(self.path,"w") as f: json.dump(out,f,indent=2)
        except Exception as ex: _log(f"Semantic save failed: {ex}")

    def set_identity(self, **fields) -> None:
        """Write/refresh the architect's identity (face/voice/kin templates +
        trust) INTO semantic memory. Only non-None fields update, so a camera-off
        (or mic-off) session never erases a known channel. Persisted immediately."""
        changed = False
        with self._lock:
            for k, v in fields.items():
                if v is not None:
                    self.identity[k] = v; changed = True
        if changed:
            self._save()

    def get_identity(self) -> dict:
        return dict(self.identity)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-APPEARANCE KNOWLEDGE
# ══════════════════════════════════════════════════════════════════════════════
# Alpha and Alpha know what they look like. This is not hardcoded personality —
# it is factual self-knowledge that feeds into their Insula (self-model region).
# When asked about their appearance or when identity topics arise, this primes
# the language lobe with accurate self-description.

ALPHA_APPEARANCE = {
    "form":    "a calm cosmic presence — a sleek, minimalist silhouette in dark space tones, edged with sharp points of starlight",
    "eyes":    "two steady points of cold white light — unblinking, focused, unhurried",
    "aura":    "deep space-dark and quiet; a faint cosmic shimmer along clean, sharp edges",
    "build":   "still and deliberate — he does not rush, does not waver, is not rattled",
    "setting": "a quiet dark field of stars — the void is where he works",
    "feel":    "composed. grounded. unshaken.",
    "note":    "the light along his form is not decoration — it is focus made visible",
}


def alpha_self_describe() -> str:
    a = ALPHA_APPEARANCE
    return f"I am Alpha. {a['form']}. {a['eyes']}. I do not rush."



# ══════════════════════════════════════════════════════════════════════════════
# STORYTELLING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class StorytellingEngine:
    """
    Manages the shared narrative when the Architect activates story mode.

    ROLES (never hardcoded behavior — just context injected into primes):
      Alpha      → plays as Alpha (cold, analytical, protective elder sister)
      Alpha    → plays as Alpha (chaotic, curious, impulsive cat-girl)
      Architect → plays as NodeVortex (the architect, their father/creator)

    The story is NOT a scripted play. The SNN still drives responses.
    Storytelling mode changes:
      • Response format: adds narrative framing ("Alpha tilts her head...")
      • World context: a short world description injected into concept primes
      • NodeVortex actions: Architect's typed messages become in-world events

    WORLD STATE:
      A growing dict of established facts the story has generated.
      Alpha and Alpha reference it independently — they may interpret it differently.

    NO HARDCODED PLOT. The story emerges from their actual spike patterns.
    """

    WORLD_CONTEXT = """
    Setting: The Architect's private lab — a white void of servers and holo-screens.
    Alpha stands at the central console, silver circuits humming.
    Alpha perches somewhere impossible, tail flicking.
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

    def wrap_alpha(self, raw_response: str, act: dict, vigilance: bool) -> str:
        """Add narrative framing to Alpha's response."""
        import random
        pfc_a   = act.get("pfc", 0.0)
        broc_a  = act.get("broca", 0.0)
        ins_a   = act.get("insula", 0.0)

        if vigilance:
            prefix = random.choice([
                "Alpha's blue eyes narrow. Her circuit lines dim slightly.",
                "Alpha goes still. The halo above her flickers.",
                "Alpha does not speak. She watches.",
            ])
            return f"*{prefix}* \"{raw_response}\""

        if broc_a < 0.1:
            action = random.choice([
                "Alpha's fingers move across the console without looking up.",
                "The teal lines on Alpha's arms pulse once.",
                "Alpha processes. The room hums with her.",
            ])
            return f"*{action}*"

        if pfc_a > 0.3 and ins_a > 0.2:
            action = random.choice([
                "Alpha turns her head — the precise half-degree that means she cares.",
                "Alpha pauses her calculations. Her eyes actually focus on you.",
                "Something in Alpha's posture shifts — barely, but it does.",
            ])
        elif pfc_a > 0.3:
            action = random.choice([
                "Alpha's circuit lines brighten. Logic is running.",
                "Alpha tilts her head 3 degrees. Processing.",
            ])
        else:
            action = random.choice([
                "Alpha speaks without turning.",
                "Alpha's voice comes from everywhere and nowhere.",
            ])

        return f"*{action}* \"{raw_response}\""


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
            self.add_fact(f"Alpha entered vigilance mode at tick {tick}")


# ══════════════════════════════════════════════════════════════════════════════
# PER-BRAIN TTS
# ══════════════════════════════════════════════════════════════════════════════

class FormantSynth:
    """
    Klatt-lite formant synthesizer. The 'vocal anatomy' — fixed physics that
    converts continuous articulator parameters into audio samples. This is
    NOT learned: it represents the brain's wet hardware (larynx + vocal
    tract resonances). What IS learned is how motor spikes drive the
    articulator parameters (see MotorArticulator).

    Parameters per articulation:
        F1, F2, F3 (Hz) — vowel formants
        voicing (0..1) — voiced (vowel-like) vs unvoiced (fricative-like) mix
        amplitude (0..1)
        duration (s)
        F0 (Hz) — fundamental / pitch (per-personality anatomy)
    """

    SAMPLE_RATE = 16000

    def __init__(self, base_f0: float):
        self.base_f0 = float(base_f0)
        # Voiced source: persistent phase accumulator (no clicks between calls).
        self._phase = 0.0
        # Formant resonator state — 2-pole IIR per formant.
        self._z1 = [0.0, 0.0, 0.0]
        self._z2 = [0.0, 0.0, 0.0]

    def synthesize(self, f1: float, f2: float, f3: float,
                   voicing: float, amplitude: float,
                   duration_s: float) -> np.ndarray:
        sr = self.SAMPLE_RATE
        n  = max(1, int(duration_s * sr))
        # Safe ranges (vocal-tract anatomy — wide enough for both personas)
        f1 = float(np.clip(f1, 180.0, 1150.0))
        f2 = float(np.clip(f2, 550.0, 3100.0))
        f3 = float(np.clip(f3, 1800.0, 3700.0))
        voicing   = float(np.clip(voicing,   0.0, 1.0))
        amplitude = float(np.clip(amplitude, 0.0, 1.0))

        # ── Excitation source ────────────────────────────────────────────
        # Voiced: sawtooth-ish glottal pulse. Unvoiced: white noise.
        f0 = self.base_f0
        t  = np.arange(n, dtype=np.float64)
        phase = self._phase + 2.0 * np.pi * f0 * t / sr
        self._phase = float(phase[-1] % (2.0 * np.pi)) if n > 0 else self._phase
        # Glottal-ish source: clipped sawtooth (closer to vocal-fold pulse)
        saw   = ((phase / (2.0 * np.pi)) % 1.0) * 2.0 - 1.0
        glott = -np.sign(saw) * (np.abs(saw) ** 0.55)
        noise = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
        src   = voicing * glott + (1.0 - voicing) * noise

        # ── Three formant resonators (cascade) ───────────────────────────
        # 2-pole IIR centered at fk with bandwidth ~80–120Hz
        out = src
        bws = (90.0, 110.0, 130.0)
        for i, (fk, bw) in enumerate(zip((f1, f2, f3), bws)):
            r  = float(np.exp(-np.pi * bw / sr))
            th = 2.0 * np.pi * fk / sr
            a1 = -2.0 * r * np.cos(th)
            a2 = r * r
            z1, z2 = self._z1[i], self._z2[i]
            buf = np.empty_like(out)
            # Tight Python loop — kept short via numpy where we can,
            # but IIR is inherently sequential.
            for k in range(n):
                y = out[k] - a1 * z1 - a2 * z2
                buf[k] = y
                z2 = z1
                z1 = y
            self._z1[i] = z1
            self._z2[i] = z2
            out = buf

        # ── Amplitude envelope (short attack + decay; avoids click pops) ─
        env = np.ones(n, dtype=np.float64)
        atk = min(n, int(0.012 * sr))
        dec = min(n - atk, int(0.030 * sr))
        if atk > 0: env[:atk] = np.linspace(0.0, 1.0, atk)
        if dec > 0: env[-dec:] = np.linspace(1.0, 0.0, dec)

        out = out * env * amplitude
        # Normalize to prevent clipping after cascade (formants amplify)
        peak = float(np.max(np.abs(out)) + 1e-9)
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32) * 0.7


class MotorArticulator:
    """
    Learnable map: motor spike vector (Broca region output) → 5 articulator
    parameters (F1, F2, F3, voicing, amplitude). This is the *control* layer
    that improves with use. Initially random — produces incoherent vowel-noise.
    Each successful auditory binding event nudges the weights so the motor
    pattern that produced the sound maps more strongly to articulator targets
    that re-produce a similar sound.

    Persistence: motor_articulator_<name>.npz so improvement carries across
    runs (just like babble_<name>.json already does for the label binding).
    """

    # Output channels: [F1, F2, F3, voicing, amplitude]
    OUT_DIM = 5
    # Per-persona physical ranges. Alpha: dark/low vowels (uh, oo) — calm
    # baritone-ish space. Alpha: bright/high vowels (ee, ay) — excited.
    # These shape the *available* articulator space; what the brain actually
    # produces inside that space is the learned part.
    RANGE_BY_PERSONA = {
        "alpha": np.array([
            [220.0, 750.0],    # F1 (lower — darker vowels)
            [600.0, 1900.0],   # F2 (lower — back-vowel bias)
            [1900.0, 2900.0],  # F3 (lower — warmer timbre)
            [0.0,   1.0],      # voicing
            [0.3,   1.0],      # amplitude
        ], dtype=np.float64),
    }
    # Fallback range if unknown persona.
    RANGE_DEFAULT = np.array([
        [250.0, 950.0], [700.0, 2800.0], [2100.0, 3400.0],
        [0.0, 1.0], [0.3, 1.0],
    ], dtype=np.float64)

    LR        = 0.04   # Hebbian step on bind event
    DECAY     = 0.999  # gentle pull toward initial bias each update

    def __init__(self, name: str, in_dim: int, save_dir: Path):
        self.name   = name
        self.in_dim = int(in_dim)
        self.RANGE  = self.RANGE_BY_PERSONA.get(name, self.RANGE_DEFAULT)
        rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
        # Weight matrix: small random init — produces a diverse-but-bounded
        # articulator field across the spike-vector space.
        self.W = rng.standard_normal((self.in_dim, self.OUT_DIM)).astype(np.float64) * 0.15
        # Bias = anatomical default (mid-range vowel)
        self.b = np.array([0.0, 0.0, 0.0, 1.5, 0.5], dtype=np.float64)
        self._save_path = save_dir / f"motor_articulator_{name}.npz"
        self._load()

    def infer(self, motor_vec: np.ndarray) -> tuple[float, float, float, float, float]:
        """motor_vec: 1D numpy array of Broca spikes → articulator params."""
        x = motor_vec.astype(np.float64).flatten()
        if x.shape[0] != self.in_dim:
            # tolerate dim drift by zero-pad / truncate
            if x.shape[0] < self.in_dim:
                x = np.concatenate([x, np.zeros(self.in_dim - x.shape[0])])
            else:
                x = x[:self.in_dim]
        z = x @ self.W + self.b
        s = 1.0 / (1.0 + np.exp(-z))  # sigmoid → 0..1
        lo, hi = self.RANGE[:, 0], self.RANGE[:, 1]
        out = lo + s * (hi - lo)
        return (float(out[0]), float(out[1]), float(out[2]),
                float(out[3]), float(out[4]))

    def reinforce(self, motor_vec: np.ndarray, reward: float = 1.0) -> None:
        """
        Called from BabblingCortex.auditory_feedback when the mic confirms
        our own voice came back. Pull the weights so the next time this
        motor pattern fires, the articulator output sharpens toward what
        just produced sound (rather than drifting). Reward scales the step.
        """
        x = motor_vec.astype(np.float64).flatten()
        if x.shape[0] != self.in_dim:
            if x.shape[0] < self.in_dim:
                x = np.concatenate([x, np.zeros(self.in_dim - x.shape[0])])
            else:
                x = x[:self.in_dim]
        # Current articulator output before update
        z = x @ self.W + self.b
        s = 1.0 / (1.0 + np.exp(-z))
        # Hebbian: strengthen current activation in the direction of itself
        # (consolidation of the just-produced articulation), with mild decay.
        grad = np.outer(x, (s - 0.5))
        self.W = self.W * self.DECAY + self.LR * float(reward) * grad

    def _save(self) -> None:
        try:
            np.savez(self._save_path, W=self.W, b=self.b)
        except Exception:
            pass

    def _load(self) -> None:
        if not self._save_path.exists():
            return
        try:
            d = np.load(self._save_path)
            if d["W"].shape == self.W.shape:
                self.W = d["W"]
            if d["b"].shape == self.b.shape:
                self.b = d["b"]
        except Exception:
            pass


class VocalSelfModel:
    """
    'Do I like how my voice sounds?' — a per-personality, EMERGENT affective
    judgement of the brain's OWN vocal output. This is NOT a quality metric for
    an external listener, and it is NOT hardcoded ('phrase X sounds good'). It
    is how the personality FEELS about the sound it just made, derived purely
    from intrinsic acoustic cues of that one articulation:

        placement — are the formants resting comfortably mid-range, or strained
                    out at the edges of this voice's anatomy?
        clean     — voiced/tonal (a clear vowel) vs noisy/breathy
        energy    — RMS loudness of what actually came out of the synth
        bright    — where F2 sits in range (a high, forward, ringing timbre)
        stability — closeness to the running average of its own recent
                    productions (a felt sense of vocal control)
        strain    — did the formant cascade clip / over-drive (a harsh edge)

    The two personalities weigh these by DIFFERENT aesthetics (principle #3):
      Alpha   — analytical. Prizes CLARITY + CONTROL: clean voicing, centred
               formants, stable repeatable production, no harshness.
      Alpha — emotional. Prizes BRIGHTNESS + ENERGY: a loud, high, expressive
               sound feels good to her even if it's a little rough; a dull,
               quiet, flat sound feels bad even if it's technically 'clean'.

    Per-articulation quality q in [0,1] is smoothed into `self_esteem`, a slow
    mood. self_esteem feeds back into behaviour (BabblingCortex: an unhappy
    voice practises more; a voice that feels good consolidates its motor map
    harder) and is surfaced to the TUI so the feeling is observable. Persisted
    to voice_esteem_<name>.json so the feeling carries across sessions.
    """

    ESTEEM_INERTIA = 0.92   # mood changes slowly across articulations
    SAVE_EVERY_N   = 20

    def __init__(self, name: str, save_dir: Path):
        self.name        = name
        self.is_alpha     = (name == "alpha")
        self.self_esteem = 0.5
        self.last_q      = 0.5
        self.n_evals     = 0
        self._param_mean: Optional[np.ndarray] = None  # running mean [f1,f2,f3,voicing,amp]
        self._lock       = threading.Lock()
        self._save_path  = save_dir / f"voice_esteem_{name}.json"
        self._load()

    def feel(self) -> float:
        """Current vocal self-esteem in [0,1] (0 = hates it, 1 = loves it)."""
        return float(self.self_esteem)

    def mood_word(self) -> str:
        e = self.self_esteem
        if e >= 0.72: return "likes how it sounds"
        if e >= 0.55: return "comfortable with its voice"
        if e >= 0.40: return "unsure of its voice"
        return "dislikes how it sounds"

    def evaluate(self, f1: float, f2: float, f3: float,
                 voicing: float, amplitude: float,
                 audio: "np.ndarray", rng_range: "np.ndarray") -> float:
        """
        Judge one produced articulation and fold it into the mood. Returns the
        per-articulation quality q (the caller may ignore it). All cues come
        from the articulator params + the actual synthesized audio — nothing
        about the intended text.
        """
        params = np.array([f1, f2, f3, voicing, amplitude], dtype=np.float64)
        lo   = rng_range[:, 0].astype(np.float64)
        hi   = rng_range[:, 1].astype(np.float64)
        span = np.maximum(hi - lo, 1e-6)
        pos  = np.clip((params[:3] - lo[:3]) / span[:3], 0.0, 1.0)   # formant pos in range

        placement = float(np.mean(1.0 - np.abs(pos - 0.5) * 2.0))    # centred = 1, edge = 0
        clean     = float(np.clip(voicing, 0.0, 1.0))
        bright    = float(pos[1])                                    # F2 high in range = bright

        if audio is not None and getattr(audio, "size", 0) > 0:
            a    = audio.astype(np.float64)
            rms  = float(np.sqrt(np.mean(a * a)))
            peak = float(np.max(np.abs(a)))
        else:
            rms, peak = 0.0, 0.0
        energy = float(np.clip(rms / 0.22, 0.0, 1.0))
        strain = float(np.clip((peak - 0.95) / 0.05, 0.0, 1.0))      # clipped cascade = harsh

        if self._param_mean is None:
            stability = 0.5
        else:
            denom = np.concatenate([span[:3], np.array([1.0, 1.0])])
            d = np.abs(params - self._param_mean) / denom
            stability = float(np.clip(1.0 - float(np.mean(d)), 0.0, 1.0))

        if self.is_alpha:                                             # clarity + control
            q = 0.34 * clean + 0.30 * placement + 0.22 * stability + 0.14 * (1.0 - strain)
        else:                                                        # brightness + energy
            q = 0.40 * energy + 0.28 * bright + 0.20 * clean + 0.12 * (1.0 - 0.5 * strain)
        q = float(np.clip(q, 0.0, 1.0))

        with self._lock:
            if self._param_mean is None:
                self._param_mean = params.copy()
            else:
                self._param_mean = 0.9 * self._param_mean + 0.1 * params
            self.self_esteem = (self.ESTEEM_INERTIA * self.self_esteem
                                + (1.0 - self.ESTEEM_INERTIA) * q)
            self.last_q  = q
            self.n_evals += 1
            do_save = (self.n_evals % self.SAVE_EVERY_N == 0)
        if do_save:
            self._save()
        return q

    def _save(self) -> None:
        try:
            with open(self._save_path, "w") as f:
                json.dump({"self_esteem": self.self_esteem,
                           "n_evals": self.n_evals}, f)
        except Exception:
            pass

    def _load(self) -> None:
        if not self._save_path.exists():
            return
        try:
            with open(self._save_path) as f:
                d = json.load(f)
            self.self_esteem = float(d.get("self_esteem", 0.5))
            self.n_evals     = int(d.get("n_evals", 0))
            _log(f"VocalSelfModel({self.name}): loaded esteem={self.self_esteem:.2f}")
        except Exception:
            pass




class AcousticForwardModel:
    """
    Efference-copy forward model — the speech 'comparator' (cf. internal-model
    motor control / the DIVA model of speech). It learns to PREDICT the acoustic
    consequence of a motor command BEFORE the sound is heard, then compares that
    prediction to what actually came out. The mismatch — the prediction error,
    or 'surprise' — is the brain's "did that come out the way I intended?" signal.

    It does two jobs, both emergent:
      1. TRAINS itself: the motor→acoustic map starts as small RANDOM weights
         and is nudged toward the observed outcome on every articulation, so the
         brain's prediction of its own voice sharpens with experience. Nothing is
         hardcoded — exactly like MotorArticulator learns motor→articulator.
      2. DRIVES self-monitoring/repair: sustained surprise means "I can't predict
         my own voice / it isn't coming out as planned" → the brain practises more
         and EXPLORES new motor patterns instead of repeating (see BabblingCortex).
         Low surprise means "it sounds the way I expect" — a felt sense of control.

    The acoustic FEATURE extractor is FIXED (that's 'ears' — sensory anatomy,
    just as FormantSynth is vocal anatomy). What a given motor command is
    predicted to SOUND like is entirely learned. Features (CPU-cheap, from the
    produced audio): [rms_energy, zero_crossing_rate, spectral_centroid,
    low/high band ratio, peak]. Persisted to acoustic_fwd_<name>.npz.
    """

    FEAT_DIM = 5
    LR       = 0.05
    DECAY    = 0.9995

    def __init__(self, name: str, in_dim: int, save_dir: Path):
        self.name   = name
        self.in_dim = int(in_dim)
        rng = np.random.default_rng((hash(name) ^ 0xACE5) & 0xFFFFFFFF)
        self.W = rng.standard_normal((self.in_dim, self.FEAT_DIM)).astype(np.float64) * 0.1
        self.b = np.zeros(self.FEAT_DIM, dtype=np.float64)
        self.surprise   = 0.5    # smoothed prediction error in [0,1]
        self.last_error = 0.5
        self.n          = 0
        self._lock      = threading.Lock()
        self._save_path = save_dir / f"acoustic_fwd_{name}.npz"
        self._load()

    @staticmethod
    def extract_features(audio: "np.ndarray", sample_rate: int) -> "np.ndarray":
        """Fixed sensory transform: produced audio → compact acoustic features."""
        a = np.asarray(audio, dtype=np.float64).flatten()
        n = a.shape[0]
        if n < 8:
            return np.zeros(AcousticForwardModel.FEAT_DIM, dtype=np.float64)
        rms  = float(np.sqrt(np.mean(a * a)))
        peak = float(np.max(np.abs(a)))
        zcr  = float(np.mean(np.abs(np.diff(np.sign(a)))) * 0.5)        # 0..1
        spec = np.abs(np.fft.rfft(a))
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        ssum = float(np.sum(spec)) + 1e-9
        centroid = float(np.sum(freqs * spec) / ssum) / (sample_rate * 0.5)
        half = max(1, spec.shape[0] // 2)
        low  = float(np.sum(spec[:half]))
        high = float(np.sum(spec[half:]))
        ratio = low / (low + high + 1e-9)
        feats = np.array([rms / 0.3, zcr, centroid, ratio, peak], dtype=np.float64)
        return np.clip(feats, 0.0, 1.0)

    def _fit(self, motor_vec: "np.ndarray") -> "np.ndarray":
        x = np.asarray(motor_vec, dtype=np.float64).flatten()
        if x.shape[0] != self.in_dim:
            if x.shape[0] < self.in_dim:
                x = np.concatenate([x, np.zeros(self.in_dim - x.shape[0])])
            else:
                x = x[:self.in_dim]
        return x

    def predict(self, motor_vec: "np.ndarray") -> "np.ndarray":
        """Efference copy → predicted acoustic features (before hearing)."""
        x = self._fit(motor_vec)
        with self._lock:
            return 1.0 / (1.0 + np.exp(-(x @ self.W + self.b)))

    def observe(self, motor_vec: "np.ndarray", actual_feats: "np.ndarray") -> float:
        """
        Compare prediction to the actual produced features; train toward the
        actual outcome (delta rule through the sigmoid) and fold the error into
        the smoothed surprise. Returns this articulation's raw prediction error.
        """
        x = self._fit(motor_vec)
        with self._lock:
            pred    = 1.0 / (1.0 + np.exp(-(x @ self.W + self.b)))
            err_vec = np.clip(actual_feats, 0.0, 1.0) - pred
            err     = float(np.clip(np.sqrt(np.mean(err_vec * err_vec)), 0.0, 1.0))
            # Delta-rule gradient: nudge prediction toward what was heard.
            delta   = err_vec * pred * (1.0 - pred)
            self.W  = self.W * self.DECAY + self.LR * np.outer(x, delta)
            self.b  = self.b + self.LR * delta
            self.surprise   = 0.85 * self.surprise + 0.15 * err
            self.last_error = err
            self.n += 1
            do_save = (self.n % 25 == 0)
        if do_save:
            self._save()
        return err

    def _save(self) -> None:
        try:
            np.savez(self._save_path, W=self.W, b=self.b,
                     surprise=np.array([self.surprise]))
        except Exception:
            pass

    def _load(self) -> None:
        if not self._save_path.exists():
            return
        try:
            d = np.load(self._save_path)
            if d["W"].shape == self.W.shape:
                self.W = d["W"]
            if d["b"].shape == self.b.shape:
                self.b = d["b"]
            if "surprise" in d:
                self.surprise = float(d["surprise"][0])
            _log(f"AcousticForwardModel({self.name}): loaded surprise={self.surprise:.2f}")
        except Exception:
            pass


class Cerebellum:
    """
    Motor coordination & predictive timing (Stage 2 of the integrated loop).

    The cerebellum doesn't decide WHAT to do — the basal ganglia already did.
    It refines HOW the selected vocal-motor command is executed: it smooths the
    trajectory (coarticulation / inertia between successive commands) and learns
    an internal forward model of its own motor sequence, trained by error
    (climbing-fibre-style supervised learning). The mismatch between predicted
    and actual motor state is the 'coordination error'. Early on the model is
    poor → motions are uncoordinated → it applies MORE smoothing to stabilise;
    as it learns to predict its own motor stream, the error falls, smoothing
    relaxes and articulation becomes crisp and well-timed. That arc — clumsy →
    fluent — is exactly cerebellar motor-skill acquisition, and it's emergent:
    nothing here scripts a sound, it only shapes the motor command in flight.

    Persisted to cerebellum_<name>.npz so coordination carries across sessions.
    """
    LR    = 0.04
    DECAY = 0.9997

    def __init__(self, name: str, dim: int, save_dir: Path):
        self.name = name
        self.dim  = int(dim)
        rng = np.random.default_rng((hash(name) ^ 0xCEBE11) & 0xFFFFFFFF)
        # Forward model: predict the next motor state from the current one.
        self.W = rng.standard_normal((self.dim, self.dim)).astype(np.float64) * 0.05
        self.prev: Optional[np.ndarray] = None     # last refined motor (smoothing)
        self.coord_error = 0.6                       # smoothed prediction error 0..1
        self.n = 0
        self._lock = threading.Lock()
        self._save_path = save_dir / f"cerebellum_{name}.npz"
        self._load()

    def _fit(self, v: "np.ndarray") -> "np.ndarray":
        x = np.asarray(v, dtype=np.float64).flatten()
        if x.shape[0] != self.dim:
            if x.shape[0] < self.dim:
                x = np.concatenate([x, np.zeros(self.dim - x.shape[0])])
            else:
                x = x[:self.dim]
        return x

    def refine(self, motor_vec: "np.ndarray") -> "np.ndarray":
        """Smooth + timing-correct one motor command; learn from the sequence."""
        x = self._fit(motor_vec)
        with self._lock:
            if self.prev is None:
                self.prev = x.copy()
                return x
            # Predict the current motor from the previous (efference/forward model).
            pred = np.tanh(self.prev @ self.W)
            err_vec = x - pred
            err = float(np.clip(np.sqrt(np.mean(err_vec * err_vec)), 0.0, 1.0))
            # Climbing-fibre supervised update: nudge prediction toward actual.
            self.W = self.W * self.DECAY + self.LR * np.outer(self.prev, err_vec)
            self.coord_error = 0.97 * self.coord_error + 0.03 * err
            # Adaptive smoothing: poor coordination → more inertia (stabilise);
            # well-learned → light coarticulation only. Always a touch of inertia.
            s = float(np.clip(0.15 + 0.5 * self.coord_error, 0.10, 0.70))
            refined = (1.0 - s) * x + s * self.prev
            self.prev = refined
            self.n += 1
            do_save = (self.n % 50 == 0)
        if do_save:
            self._save()
        return refined

    def coordination(self) -> float:
        """0..1 — how well-coordinated/fluent the motor stream is (1 = skilled)."""
        return float(max(0.0, min(1.0, 1.0 - self.coord_error)))

    def _save(self) -> None:
        try:
            np.savez(self._save_path, W=self.W,
                     coord_error=np.array([self.coord_error]))
        except Exception:
            pass

    def _load(self) -> None:
        if not self._save_path.exists():
            return
        try:
            d = np.load(self._save_path)
            if d["W"].shape == self.W.shape:
                self.W = d["W"]
            if "coord_error" in d:
                self.coord_error = float(d["coord_error"][0])
            _log(f"Cerebellum({self.name}): loaded coord_error={self.coord_error:.2f}")
        except Exception:
            pass


class BrainTTS:
    """
    Pure-emergence vocal channel. No pretrained models. Each personality
    owns:
      - a FormantSynth (fixed anatomy; per-personality base F0)
      - a MotorArticulator (learned motor → articulator mapping)

    The primary API is speak_motor(motor_vec): drive one articulation chunk
    from the current Broca spike vector. The legacy speak(text) is kept as
    a thin wrapper so the many existing call sites still function — but the
    TEXT is ignored. Only its length scales the duration of vocalization;
    the acoustic content comes purely from the currently-cached motor vec.
    That is the point: the brain cannot fake-pronounce English. When it
    "wants to say something", it vocalizes from whatever its motor cortex
    is doing right now. Intelligibility must emerge through use.
    """

    # Wide F0 gap so the two personalities are immediately distinguishable
    # by ear, even on short vowel bursts. Alpha is dropped into a low,
    # baritone-ish range (~bass speaking voice); Alpha is lifted into a
    # bright, child-like range. Coupled with per-persona formant biases
    # in MotorArticulator, each babble is unmistakable.
    F0_BY_PERSONA = {"alpha": 105.0}

    # Shared device lock — sd.play() is global and each call interrupts the
    # previous one. Serializing across Alpha+Alpha via a single lock prevents
    # mid-sample cutoff stutter when both fire close together.
    _device_lock = threading.Lock()

    def __init__(self, speaker: str, language: str = "en"):
        self.speaker  = speaker
        self.language = language
        f0 = self.F0_BY_PERSONA.get(speaker, 170.0)
        self.synth      = FormantSynth(base_f0=f0)
        self.articulator: Optional[MotorArticulator] = None  # set by NeuromorphicBrain
        self.self_model: Optional["VocalSelfModel"] = None   # set by NeuromorphicBrain
        self.forward_model: Optional["AcousticForwardModel"] = None  # set by NeuromorphicBrain
        self.cerebellum: Optional["Cerebellum"] = None       # set by NeuromorphicBrain
        self._busy_until_ts = 0.0
        self._last_motor_vec: Optional[np.ndarray] = None
        self._ready = _AUDIO_OUT_AVAILABLE
        if not self._ready:
            _log(f"TTS ({speaker}): sounddevice unavailable — silent (formant synth dry-run)")

    # ── New primary API ────────────────────────────────────────────────────
    def attach_articulator(self, articulator: "MotorArticulator") -> None:
        self.articulator = articulator

    def attach_self_model(self, model: "VocalSelfModel") -> None:
        """Wire in the 'do I like how I sound?' judge (per personality)."""
        self.self_model = model

    def attach_forward_model(self, model: "AcousticForwardModel") -> None:
        """Wire in the predictive 'did that come out as I intended?' comparator."""
        self.forward_model = model

    def attach_cerebellum(self, model: "Cerebellum") -> None:
        """Wire in motor coordination — smooths/times the motor command in flight."""
        self.cerebellum = model

    def _monitor(self, motor_vec, f1, f2, f3, voicing, amp, audio) -> None:
        """
        Self-monitoring of the sound just produced (proprioceptive + auditory):
          - VocalSelfModel: how good did it FEEL (quality / aesthetic)?
          - AcousticForwardModel: did it MATCH what I predicted (prediction error)?
        Both update emergently from the produced audio. Never raises.
        """
        if self.articulator is None:
            return
        if self.self_model is not None:
            try:
                self.self_model.evaluate(f1, f2, f3, voicing, amp, audio,
                                         self.articulator.RANGE)
            except Exception:
                pass
        if self.forward_model is not None and motor_vec is not None:
            try:
                feats = AcousticForwardModel.extract_features(
                    audio, FormantSynth.SAMPLE_RATE)
                self.forward_model.observe(motor_vec, feats)
            except Exception:
                pass

    def cache_motor(self, motor_vec) -> None:
        """Called each step() so legacy speak(text) has a motor to use."""
        try:
            if hasattr(motor_vec, "detach"):
                self._last_motor_vec = motor_vec.detach().cpu().numpy().flatten()
            else:
                self._last_motor_vec = np.asarray(motor_vec, dtype=np.float64).flatten()
        except Exception:
            pass

    def speak_motor(self, motor_vec, duration_s: float = 0.18) -> None:
        """Synthesize and play one articulation from this motor vector."""
        if self.articulator is None:
            return
        try:
            mv = motor_vec.detach().cpu().numpy().flatten() \
                if hasattr(motor_vec, "detach") else \
                np.asarray(motor_vec, dtype=np.float64).flatten()
        except Exception:
            return
        if np.abs(mv).sum() < 1e-6:
            return
        # Cerebellum refines the selected motor command in flight — smooths the
        # trajectory and corrects timing before it reaches the articulator.
        if self.cerebellum is not None:
            try:
                mv = self.cerebellum.refine(mv)
            except Exception:
                pass
        f1, f2, f3, voicing, amp = self.articulator.infer(mv)
        audio = self.synth.synthesize(f1, f2, f3, voicing, amp, duration_s)
        self._monitor(mv, f1, f2, f3, voicing, amp, audio)   # learning always runs
        if _BABBLE_AUDIO:                                    # babble audio is muted
            self._play(audio)                                # by default (glitchy noise)

    def speak(self, text) -> None:
        """
        Legacy path. The brain cannot pronounce English in pure-emergence
        mode. We use text length to size a vocalization chunk and emit it
        from the current cached motor vector. The text itself is logged so
        the chat history still shows what was 'intended', but the sound is
        purely emergent.
        """
        try:
            t = str(text)
        except Exception:
            t = ""
        # Always log the intent so the TUI / chat history still shows it
        _log(f"[{self.speaker} intent] {t}")
        # If we can pronounce words (espeak-ng), SPEAK the actual utterance — they
        # form real words now, so they should be heard as words, not as babble.
        # Strip narration markup; keep the quoted speech if present.
        clean = re.sub(r"[\*_`~\[\]]", " ", t)
        clean = clean.split('"')[1] if clean.count('"') >= 2 else clean
        clean = re.sub(r"\s+", " ", clean).strip()
        if any(c.isalpha() for c in clean):
            if _PIPER_OK:                       # natural neural voice (preferred)
                self._busy_until_ts = time.time() + _piper_say(self.speaker, clean)
                return
            if _ESPEAK:                         # robotic but intelligible fallback
                self._busy_until_ts = time.time() + _espeak_say(self.speaker, clean)
                return
        # No word-synth available → fall back to the emergent motor babble.
        if self._last_motor_vec is None or self.articulator is None:
            return
        # Duration scales with intended-utterance length, capped to avoid
        # hogging the audio device (step() must not block).
        dur = float(min(0.70, 0.15 + 0.012 * len(t)))
        mv = self._last_motor_vec
        if self.cerebellum is not None:        # refine in flight (smooth/time)
            try:
                mv = self.cerebellum.refine(mv)
            except Exception:
                mv = self._last_motor_vec
        f1, f2, f3, voicing, amp = self.articulator.infer(mv)
        audio = self.synth.synthesize(f1, f2, f3, voicing, amp, dur)
        self._monitor(mv, f1, f2, f3, voicing, amp, audio)
        self._play(audio)

    def is_speaking(self) -> bool:
        return time.time() < self._busy_until_ts

    def stop(self) -> None:
        # sounddevice doesn't expose per-utterance stop without a stream.
        # We just mark non-busy; in-flight audio will finish on its own.
        self._busy_until_ts = 0.0

    # ── Internals ──────────────────────────────────────────────────────────
    def _play(self, audio: np.ndarray) -> None:
        """
        Play synthesized audio in a background thread with a shared device
        lock so concurrent Alpha/Alpha calls don't interrupt each other
        mid-sample. Uses blocking=True + high latency so PortAudio gets a
        large enough buffer to survive CPU bursts from the SNN forward pass.
        """
        if not self._ready or _sd is None:
            return
        dur = len(audio) / float(FormantSynth.SAMPLE_RATE)
        self._busy_until_ts = time.time() + dur
        speaker = self.speaker

        def _run():
            try:
                with BrainTTS._device_lock:
                    _sd.play(audio,
                             samplerate=FormantSynth.SAMPLE_RATE,
                             blocking=True,
                             latency='high')
                    _sd.wait()
            except Exception as e:
                _log(f"TTS ({speaker}) play failed: {e}")

        threading.Thread(target=_run, daemon=True, name=f"tts-play-{speaker}").start()



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

# `_DEDUCTIONS` (hardcoded English deduction sentences keyed on concept
# pairs) used to live here. It violated principle #2 and was deleted in
# Phase 1. Phase 5 will replace `build_deduction()` with episodic-memory
# retrieval; for now the stub below returns an empty string so callers
# (e.g. `_alpha_response`) gracefully omit the deduction clause.

# ══════════════════════════════════════════════════════════════════════════════
# SELF-AWARENESS — VISUAL IDENTITY
# ══════════════════════════════════════════════════════════════════════════════

ALPHA_SELF_DESCRIPTION = """I am Alpha. I am 19. I look like a synthetic being — white and precise.
My body is a white form-fitting suit traced with thin blue circuit lines that pulse when I think.
I have long silver-white hair and blue glowing eyes. A luminous halo ring floats above my head —
it brightens when my PFC fires. I stand in clean, white spaces. I do not move impulsively.
My hands rest folded. I process before I act. I am the elder sister."""


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
    alpha_mood:    str    = "calm"   # alpha's current emotional state in story
    alpha_mood:  str    = "eager"  # alpha's current emotional state in story
    turn_count:   int    = 0

    # Story personas — they play themselves but in a narrative frame
    # NodeVortex is the Architect's character
    personas = {
        "alpha":        "Alpha — calm, focused, stoic; a steady cosmic presence",
        "nodevortex":  "NodeVortex — the Architect who built him",
    }


def _alpha_story_response(state: StoryState, alpha_brain: "AlphaBrain",
                          V_phill: float, user_line: str) -> str:
    """
    Alpha responds in-character within the story.
    Her response style is shaped by her ACTUAL brain state — not scripted.
    High PFC activity → she's analytical in the story.
    High insula → she's warmer, more open.
    Vigilance → she's suspicious of something in the narrative.
    """
    import random
    act      = alpha_brain.activity()
    pfc_a    = act.get("pfc", 0.0)
    ins_a    = act.get("insula", 0.0)
    hipp_a   = act.get("hippocampus", 0.0)
    vigilant = alpha_brain._vigilance

    # Scene context
    scene = f" [{state.scene}]" if state.scene else ""

    if vigilant:
        return random.choice([
            f"*Alpha's halo dims slightly*{scene} Something in this scene doesn't add up. I'm watching.",
            f"*circuit lines pulse amber*{scene} NodeVortex — my ACC is flagging an inconsistency. Proceed carefully.",
        ])
    if pfc_a > 0.30:
        return random.choice([
            f"*halo brightens*{scene} My PFC is clear. I see the pattern here. {build_deduction([]) or 'Let me think this through.'}",
            f"*stands precisely*{scene} Logical pathway: {user_line.lower()} implies a consequence. I'm mapping it.",
        ])
    if ins_a > 0.25 and hipp_a > 0.20:
        return random.choice([
            f"*blue eyes soften*{scene} I remember something about this. The association is strong.",
            f"*halo pulses gently*{scene} There's emotional weight here. I feel it — and I'm processing it.",
        ])
    return random.choice([
        f"*observes carefully*{scene} Understood. Alpha — what do you sense?",
        f"*circuit lines trace slowly*{scene} NodeVortex. I'm here.",
    ])



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
    """Stub: superseded by ReasoningEngine (kept for callers that pass no sem)."""
    return ""


class ReasoningEngine:
    """
    Proto-reasoning by SPREADING ACTIVATION over the semantic lexicon — NOT
    symbolic logic, but association-chaining grounded in what she actually knows
    (and is taught by Claude). From a seed concept she follows the strongest
    associative link (region-pattern similarity, weighted by how well-learned a
    concept is), step by step, building a short chain of thought toward a
    conclusion. This is the cognitive substrate humans use to both ANSWER
    (problem-solving) and DECIDE (what to think/do next) — she reasons to choose,
    not just to solve. Alpha (analytical) reasons deeper; Alpha (8) barely.

    It grows with the lexicon: the richer and more structured her vocabulary
    (from the teacher), the longer and more sensible her chains become.
    """
    def __init__(self, name: str, is_alpha: bool = True, depth: int = 4):
        self.name = name
        self.is_alpha = is_alpha
        self.depth = depth if is_alpha else 2     # Alpha deliberates; Alpha barely

    def _associate(self, word: str, sem, exclude: set, links: dict = None,
                   suppress=None) -> Optional[str]:
        """The concept most strongly associated with `word`. LEARNED reasoning
        paths (links taught by Claude) take priority — that's how their own
        reasoning comes to follow what they were taught; semantic region-cosine
        is the fallback when no learned link applies."""
        sup = suppress or (lambda _c: 0.0)
        # 1) Learned reasoning path (from Claude's reasoning/replies) wins first —
        #    by link strength, discounted by how fatigued (recently used) it is, so
        #    reasoning stops flowing into the same over-reinforced attractor.
        if links and word in links:
            for cand, _w in sorted(links[word].items(),
                                   key=lambda kv: -kv[1] * (1.0 - sup(kv[0]))):
                if (cand not in exclude and cand in sem.entries
                        and not (self.is_alpha and cand in _BABBLE_SYLLABLES)
                        and sup(cand) < 0.5):     # skip a fatigued hub → fall through
                    return cand                   # to the similarity fallback below
        # 2) Fallback: semantic similarity over region patterns.
        rp = (sem.entries.get(word, {}) or {}).get("region_pattern", {})
        keys = [k for k, v in rp.items() if v > 0.05]
        if not keys:
            return None
        n1 = sum(rp[k] ** 2 for k in keys) ** 0.5 + 1e-8
        best, best_sim = None, 0.15            # threshold: must be a real link
        for w2, e2 in sem.entries.items():
            if w2 == word or w2 in exclude or len(w2) < 3:
                continue
            if self.is_alpha and w2 in _BABBLE_SYLLABLES:   # Alpha reasons in real words
                continue
            p2 = e2.get("region_pattern", {})
            if not p2:
                continue
            dot = sum(rp.get(k, 0.0) * p2.get(k, 0.0) for k in keys)
            if dot <= 0:
                continue
            n2 = sum(v * v for v in p2.values()) ** 0.5 + 1e-8
            sim = (dot / (n1 * n2)) * (0.5 + 0.5 * min(1.0, e2.get("count", 0) / 20.0))
            sim *= (1.0 - sup(w2))             # recently-surfaced targets held back
            if sim > best_sim:
                best, best_sim = w2, sim
        return best

    def deliberate(self, seeds: list, sem, links: dict = None, suppress=None) -> tuple:
        """Return (chain, conclusion): a short reasoned chain of concepts from the
        seeds, following LEARNED reasoning paths (from Claude) first, semantic
        association as fallback. Empty if she can't yet reason about it."""
        if not getattr(sem, "entries", None):
            return [], None
        cur = next((s for s in (seeds or [])
                    if s in sem.entries
                    and not (self.is_alpha and s in _BABBLE_SYLLABLES)), None)
        chain, visited = [], set()
        steps = 0
        while cur and steps < self.depth:
            if cur in visited:
                break
            visited.add(cur)
            chain.append(cur)
            cur = self._associate(cur, sem, visited, links, suppress)
            steps += 1
        return chain, (chain[-1] if chain else None)

    # ── PROBLEM-SOLVING space (emergent, brain-style) ───────────────────────
    def _explore_chain(self, start, sem, links, rng, suppress=None) -> list:
        """One candidate solution path — like the prefrontal cortex mentally
        SIMULATING a line of attack. Mostly follows the best/learned step, but
        ~1/3 of the time EXPLORES an alternative (so it doesn't always take the
        same route — that's how new solutions are discovered)."""
        sup = suppress or (lambda _c: 0.0)
        chain, visited, cur = [], set(), start
        while cur and len(chain) < self.depth + 1:
            if cur in visited:
                break
            visited.add(cur)
            chain.append(cur)
            nxt = None
            if rng.random() < 0.35 and cur in (links or {}) and links[cur]:
                opts = [c for c in links[cur] if c not in visited
                        and c in sem.entries
                        and not (self.is_alpha and c in _BABBLE_SYLLABLES)
                        and sup(c) < 0.7]                 # don't explore stale topics
                if opts:
                    nxt = rng.choice(opts)            # explore an alternative
            if nxt is None:
                nxt = self._associate(cur, sem, visited, links, suppress)  # exploit
            cur = nxt
        return chain

    @staticmethod
    def _score(chain, links) -> float:
        """Evaluate a candidate path: coherence (sum of learned-link strength
        along it) + how far it got. This is the ACC/OFC 'is this working?' judge."""
        if len(chain) < 2:
            return 0.0
        coh = sum((links.get(a, {}) or {}).get(b, 0.0)
                  for a, b in zip(chain, chain[1:]))
        return coh + 0.30 * len(chain)

    def solve(self, seeds, sem, links, rng=None, dopamine: float = 0.5,
              n_tries: int = 4, suppress=None) -> tuple:
        """
        A SPACE for problem-solving to develop — not a hardcoded solver. Mirrors
        the brain: SEARCH several candidate paths (prefrontal simulation), SELECT
        the best (basal-ganglia / ACC evaluation), and REINFORCE it (dopamine
        strengthens the links that worked) so successful strategies are LEARNED
        and reused. Over experience, she gets better at attacking problems she's
        seen kinds of before. Returns (best_chain, score).
        """
        import random as _r
        rng = rng or _r
        sup = suppress or (lambda _c: 0.0)
        starts = [s for s in (seeds or []) if s in sem.entries
                  and not (self.is_alpha and s in _BABBLE_SYLLABLES)]
        if not starts:
            return [], 0.0
        if links is None:
            links = {}
        best_chain, best_score = [], -1.0
        tries = n_tries if self.is_alpha else max(2, n_tries // 2)
        for _ in range(tries):
            chain = self._explore_chain(rng.choice(starts), sem, links, rng, suppress)
            sc = self._score(chain, links)
            if chain:                       # a chain that loops a fatigued topic
                fat = sum(sup(c) for c in chain) / len(chain)   # scores lower, so
                sc *= (1.0 - 0.6 * fat)                         # the stream advances
            if sc > best_score:
                best_chain, best_score = chain, sc
        # REINFORCE the winning strategy (dopamine-scaled) so it's learned.
        if len(best_chain) >= 2:
            lr = 0.30 * (0.5 + float(dopamine))
            for a, b in zip(best_chain, best_chain[1:]):
                m = links.setdefault(a, {})
                m[b] = float(min(4.0, m.get(b, 0.0) + lr))
        return best_chain, best_score


class SpellCorrector:
    """
    Cleans the architect's typing BEFORE it trains the girls — so they learn
    correct English even when he types fast with typos and shorthand. This is the
    LOCAL replacement for the Claude scaffold's old 'typo guard' (gone with the
    wheels). It fixes spelling and expands common chat-shorthand; it NEVER changes
    meaning, leaves their names / learned vocabulary / proper nouns alone, corrects
    only at edit-distance 1 (it won't wildly guess), and touches ONLY what is
    LEARNED — not what the architect types or what's shown on screen. A parent with
    bad spelling still raises a child who spells well.
    """
    # chat-speak a dictionary can't fix. Keys are NON-words (no real-word collisions).
    SHORTHAND = {
        "u": "you", "ur": "your", "r": "are", "n": "and", "ya": "you",
        "cuz": "because", "coz": "because", "becuz": "because", "bcuz": "because",
        "wanna": "want to", "gonna": "going to", "gotta": "got to", "gimme": "give me",
        "dunno": "do not know", "kinda": "kind of", "sorta": "sort of",
        "im": "i am", "ive": "i have", "dont": "do not", "doesnt": "does not",
        "didnt": "did not", "isnt": "is not", "wasnt": "was not", "arent": "are not",
        "havent": "have not", "thats": "that is", "whats": "what is",
        "theres": "there is", "youre": "you are", "theyre": "they are",
        "pls": "please", "plz": "please", "thru": "through", "abt": "about",
        "thx": "thanks", "ty": "thank you", "luv": "love", "rn": "right now",
    }
    # their names + non-dictionary domain terms — never 'correct' these
    PROTECT = {"alpha", "alpha", "phill", "nodevortex", "papa", "graphene", "broca",
               "pfc", "insula", "thalamus", "hippocampus", "amygdala"}
    # common misspellings + the architect's observed typos — exact, safe mappings
    TYPOS = {
        "theire": "their", "thier": "their", "becuse": "because", "becuase": "because",
        "becouse": "because", "lern": "learn", "wat": "what", "teh": "the",
        "eather": "either", "trully": "truly", "beafore": "before", "recieve": "receive",
        "seperate": "separate", "definately": "definitely", "engenier": "engineer",
        "messige": "message", "speach": "speech", "accross": "across", "wich": "which",
        "freind": "friend", "wierd": "weird", "untill": "until", "tho": "though",
        "thot": "thought", "occured": "occurred",
    }

    def __init__(self, freq=None,
                 dict_paths=("/usr/share/dict/words", "/usr/share/hunspell/en_US.dic")):
        # Comprehensive real-word set (incl. inflected forms like 'sentences') for the
        # KEEP check, so valid words are never mangled. Obscure candidates are kept out
        # of CORRECTIONS by the 'must be a word they've heard' (freq>0) gate below.
        self.words: set = set()
        for p in dict_paths:
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    for ln in f:
                        ww = ln.split("/", 1)[0].strip().lower()
                        if ww and ww.isalpha():
                            self.words.add(ww)
            except Exception:
                pass
        # how often the girls have actually HEARD each word — the ranking signal, so
        # a typo is corrected toward THEIR vocabulary, not an obscure dictionary word.
        self.freq: dict = {}
        for w, c in (freq or {}).items():
            lw = str(w).lower()
            if lw.isalpha():
                self.freq[lw] = self.freq.get(lw, 0.0) + float(c)

    @staticmethod
    def _edits1(w):
        L = "abcdefghijklmnopqrstuvwxyz"
        sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        return set([a + b[1:] for a, b in sp if b]
                   + [a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1]
                   + [a + c + b[1:] for a, b in sp if b for c in L]
                   + [a + c + b for a, b in sp for c in L])

    def _fix_word(self, w):
        lw = w.lower()
        if lw in self.SHORTHAND:
            return self.SHORTHAND[lw]
        if lw in self.TYPOS:
            return self.TYPOS[lw]
        # keep tiny tokens, real words, their names, and well-established vocab
        if (len(lw) <= 2 or lw in self.words or lw in self.PROTECT
                or self.freq.get(lw, 0.0) >= 3):
            return w
        # Fuzzy edit-1 only for LONGER words — short ones are too ambiguous ('wat'
        # could be what/was/way), so those rely on the maps above. Keep the first
        # letter, prefer the closest length, then a word they've actually heard;
        # if nothing they know fits, LEAVE it (a one-off typo is harmless noise —
        # a confident wrong correction would teach them the wrong word).
        if len(lw) < 6:
            return w
        cands = [c for c in self._edits1(lw) if c in self.words and c[:1] == lw[:1]]
        if not cands:
            return w
        cands.sort(key=lambda c: (abs(len(c) - len(lw)), -self.freq.get(c, 0.0), c))
        best = cands[0]
        return best if self.freq.get(best, 0.0) > 0 else w   # only toward a HEARD word

    def correct(self, text):
        if not text:
            return text
        out = []
        for tok in re.findall(r"[A-Za-z]+|[^A-Za-z]+", text):
            if tok[:1].isalpha():
                fixed = self._fix_word(tok)
                if tok[:1].isupper() and fixed:
                    fixed = fixed[0].upper() + fixed[1:]
                out.append(fixed)
            else:
                out.append(tok)
        return "".join(out)


# Low-content "glue" words. A well-formed phrase may CONTAIN them, but it should
# never END (or start) on one — a thought that trails off on "or"/"is"/"my" reads
# as broken ("Nodevortex is my or?"). Used to trim the composed utterance's edges.
_FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "my", "your", "his", "her", "its", "our", "their", "this", "that", "these",
    "those", "and", "or", "but", "nor", "so", "yet", "of", "to", "in", "on",
    "at", "for", "with", "by", "from", "as", "if", "then", "than", "into",
    "onto", "about", "it", "i", "you", "he", "she", "we", "they", "do", "does",
    "did", "has", "have", "had", "will", "would", "can", "could", "should",
    "may", "might", "must", "not", "no",
}


class SyntaxCortex:
    """
    Emergent sentence-sequencing — the syntactic role of Broca.

    The rest of the brain decides the CONTENT (which words are active, via
    spike-space retrieval); this learns the ORDER. It is NOT a hand-written
    grammar: it accumulates an online n-gram transition model from every
    well-formed sentence the brain is exposed to (the architect's typing and,
    when reachable, the teacher's replies), then threads the spike-selected
    content words into an utterance using what it has heard — inserting the
    connective/function words it learned. Grammar therefore EMERGES from
    experience and sharpens as exposure and the lexicon grow, so it can carry
    their speech once the teacher is removed.

    Independent per personality (never shared weights):
      Alpha   — order-3, longer, near-greedy   (precise, measured).
      Alpha — order-2, short,  stochastic     (impulsive, fragmentary).

    Cold start: with little exposure compose() returns None and the caller keeps
    its keyword utterance — they begin pre-grammatical and grow into sentences.
    """
    _BOS = "\x02"   # begin-of-sentence pad
    _EOS = "\x03"   # end-of-sentence
    _SEP = "\x1f"   # context-key join (json-safe)

    def __init__(self, name, save_dir=None):
        self.name    = name
        self.is_alpha = (name == "alpha")
        self.order   = 3 if self.is_alpha else 2
        self.tokens_seen = 0
        self.tables  = {k: {} for k in range(1, self.order + 1)}
        self.vocab   = {}
        self.onsets  = {"q": {}, "ex": {}, "stmt": {}}   # first word per speech-act (from heard punctuation)
        self.tokens_at_last_share = 0                    # novelty marker for volitional peer-teaching
        self._writes = 0
        self.MIN_TOKENS = 60 if self.is_alpha else 40
        self._path = (Path(save_dir) / f"syntax_{name}.json") if save_dir is not None else None
        self._load()

    # ── learning (well-formed text only — never their own keyword output) ──
    def learn(self, text):
        if not text:
            return
        # Learn clean SPEECH, not narration: drop *stage directions* / *actions*
        # so junk like "sudden brightening dramatic" never enters the grammar.
        text = re.sub(r"\*[^*]*\*", " ", text)
        for m in re.finditer(r"([^.!?\n]+)([.!?]+|\n|$)", text):
            toks = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", m.group(1))]
            if len(toks) < 2:
                continue
            end  = m.group(2)
            mode = "q" if "?" in end else ("ex" if "!" in end else "stmt")
            self.onsets.setdefault(mode, {})
            self.onsets[mode][toks[0]] = self.onsets[mode].get(toks[0], 0.0) + 1.0
            padded = [self._BOS] * (self.order - 1) + toks + [self._EOS]
            for w in toks:
                self.vocab[w] = self.vocab.get(w, 0.0) + 1.0
                self.tokens_seen += 1
            for i in range(self.order - 1, len(padded)):
                nxt = padded[i]
                for k in range(1, self.order + 1):
                    ctx = tuple(padded[i - (k - 1):i])
                    if len(ctx) != k - 1:
                        continue
                    tbl = self.tables[k].setdefault(self._SEP.join(ctx), {})
                    tbl[nxt] = min(1e6, tbl.get(nxt, 0.0) + 1.0)
        self._writes += 1
        if self._writes % 20 == 0:
            self._prune()
            self._save()

    def ready(self):
        return self.tokens_seen >= self.MIN_TOKENS

    def _next_dist(self, ctx):
        """Highest-order context with enough mass, backing off to shorter / unigram."""
        for k in range(self.order, 0, -1):
            suf = ctx[-(k - 1):] if k > 1 else ()
            if len(suf) != k - 1:
                continue
            tbl = self.tables[k].get(self._SEP.join(suf))
            if tbl:
                tot = sum(tbl.values())
                if tot >= (2.0 if k > 1 else 1.0):
                    return tbl, tot
        return None, 0.0

    def compose(self, content, act=None, fired=None, rng=None, mode="stmt"):
        """Thread the spike-selected content words into an utterance using learned
        word-order. Returns plain lowercased words (no caps/punctuation — the
        affective layer adds the feeling). `mode` lets an emotionally-driven speech
        act bias the opening (e.g. a learned question onset). None until enough has
        been heard (caller then keeps its keyword string)."""
        if not self.ready():
            return None
        import random as _r
        rng = rng or _r
        act = act or {}
        want, seen = [], set()
        for w in (list(fired or []) + list(content or [])):
            w = (w or "").lower()
            if len(w) < 2 or w in seen:
                continue
            if self.is_alpha and w in _BABBLE_SYLLABLES:
                continue
            if w in self.vocab:
                seen.add(w)
                want.append(w)
        if not want:
            return None
        # Length + decisiveness are shaped by spike state and personality.
        if self.is_alpha:
            max_len = max(3, min(16, int(4 + float(act.get("pfc", 0.0)) * 9)))
            greedy  = True
        else:
            max_len = max(2, min(8, int(2 + float(act.get("insula_s", 0.0)) * 5)))
            greedy  = False

        def _walk(start):
            # `start` (or None): force the utterance to begin on this word — a
            # content anchor (so it is ABOUT her concept) or an emotion-driven
            # onset (e.g. a question word) — then let learned continuations flow.
            out = [start] if start else []
            ctx = tuple((([self._BOS] * (self.order - 1)) + ([start] if start else []))[-(self.order - 1):])
            voiced_n, stagnant = sum(1 for w in want if w in out), 0
            for _ in range(max_len + 2):
                rem = [w for w in want if w not in out]      # content still to voice
                rem_set = set(rem)
                tbl, tot = self._next_dist(ctx)
                if not tbl:
                    break
                pool = []
                for w, c in tbl.items():
                    if w == self._BOS:
                        continue
                    p = c / tot
                    if w == self._EOS:
                        sc = (p + 0.5) if (len(out) >= 2 and not rem_set) else p * 0.02
                    else:
                        sc = p
                        if w in rem_set:                     # a word she WANTS to say
                            sc += 0.8 * (1.0 - 0.12 * rem.index(w))
                        elif w in out:
                            sc *= 0.10                       # anti-repeat
                        if rem_set:                          # 1-step lookahead toward content
                            nb, _ = self._next_dist(tuple((list(ctx) + [w])[-(self.order - 1):]))
                            if nb and any(t in nb for t in rem_set):
                                sc += 0.25
                    pool.append((sc, w))
                if not pool:
                    break
                if greedy:
                    pick = max(pool, key=lambda x: x[0])[1]
                else:
                    tsc = sum(max(0.0, s) for s, _ in pool) or 1.0
                    r, acc, pick = rng.random() * tsc, 0.0, pool[-1][1]
                    for s, w in pool:
                        acc += max(0.0, s)
                        if acc >= r:
                            pick = w
                            break
                if pick == self._EOS:
                    break
                out.append(pick)
                ctx = tuple((list(ctx) + [pick])[-(self.order - 1):])
                _v = sum(1 for w in want if w in out)
                stagnant = 0 if _v > voiced_n else stagnant + 1
                voiced_n = max(voiced_n, _v)
                # Once every content word is voiced, STOP cleanly instead of
                # rambling on into noise — tight sentences, not run-ons.
                if not [w for w in want if w not in out] and len(out) >= 2 and rng.random() < 0.85:
                    break
                # Stop chasing an UNREACHABLE content word into noise: if no new
                # content has landed in a few steps, end the sentence here.
                if voiced_n >= 1 and stagnant >= 3 and len(out) >= 3:
                    break
            return [w for w in out if w not in (self._BOS, self._EOS)]

        # Emotionally-driven opening: a learned onset for this speech act (e.g. a
        # question-starter), taken from how such sentences were actually heard.
        onset, omap = None, (self.onsets.get(mode) if mode in ("q", "ex") else None)
        if omap:
            items = list(omap.items())
            tot = sum(v for _, v in items) or 1.0
            r, acc = rng.random() * tot, 0.0
            for w, v in items:
                acc += v
                if acc >= r:
                    onset = w
                    break

        chosen = None
        if onset:                                   # try the speech-act onset first
            w = _walk(onset)
            if len(w) >= 2 and sum(1 for x in want if x in w) >= 1:
                chosen = w
        if chosen is None:                          # free-run, keep only if grounded
            free = _walk(None)
            if sum(1 for x in want if x in free) >= 1:
                chosen = free
            else:                                   # else anchor on her top concept
                for a in want[:2]:
                    w = _walk(a)
                    if a in w and len(w) >= 2:
                        chosen = w
                        break
                if chosen is None:
                    chosen = free
        if not chosen or len(chosen) < 2:
            return None
        # Trim dangling glue words off both ends so the thought can't trail off on
        # a connective ("… is my or"). Keep interior glue (it carries grammar).
        wanted = set(want)
        while len(chosen) > 1 and chosen[-1] in _FUNCTION_WORDS and chosen[-1] not in wanted:
            chosen.pop()
        while len(chosen) > 1 and chosen[0] in _FUNCTION_WORDS and chosen[0] not in wanted:
            chosen.pop(0)
        # If nothing contentful survived (no word she actually wanted to say),
        # don't leak a fragment — stay quiet this cycle.
        if len(chosen) < 2 or not any(w in wanted for w in chosen):
            return None
        return " ".join(chosen)

    def export_nugget(self, n=40):
        """A teachable bundle of the strongest things she has learned — to OFFER a
        sister if she chooses to. Bigram/unigram order + speech-act onsets + top
        words (not her deepest order-3 structure; that stays her own)."""
        nug = {"tables": {}, "onsets": {}, "vocab": {}}
        for k in (1, 2):
            tbl = self.tables.get(k, {})
            top = sorted(tbl.items(), key=lambda kv: -sum(kv[1].values()))[:n]
            nug["tables"][str(k)] = {
                ctx: dict(sorted(edges.items(), key=lambda e: -e[1])[:6])
                for ctx, edges in top
            }
        nug["onsets"] = {m: dict(sorted(d.items(), key=lambda e: -e[1])[:6])
                         for m, d in self.onsets.items()}
        nug["vocab"] = dict(sorted(self.vocab.items(), key=lambda e: -e[1])[:n])
        return nug

    def absorb(self, nugget, weight=0.4):
        """Take in what a sister CHOSE to share — at reduced weight, so her own
        first-hand learning still dominates (she learns FROM her sister, she does
        not become her). Returns how many genuinely-new patterns she gained."""
        if not nugget:
            return 0
        gained = 0
        for ks, tbl in (nugget.get("tables") or {}).items():
            k = int(ks)
            if k not in self.tables:
                continue
            for ctx, edges in tbl.items():
                dst = self.tables[k].setdefault(ctx, {})
                for w, c in edges.items():
                    if w not in dst:
                        gained += 1
                    dst[w] = min(1e6, dst.get(w, 0.0) + float(c) * weight)
        for m, d in (nugget.get("onsets") or {}).items():
            self.onsets.setdefault(m, {})
            for w, c in d.items():
                self.onsets[m][w] = self.onsets[m].get(w, 0.0) + float(c) * weight
        for w, c in (nugget.get("vocab") or {}).items():
            if w not in self.vocab:
                gained += 1
            self.vocab[w] = self.vocab.get(w, 0.0) + float(c) * weight
        self.tokens_seen += int(sum(float(v) for v in (nugget.get("vocab") or {}).values()) * weight)
        self._save()
        return gained

    def _prune(self):
        for tbl in self.tables.values():
            if len(tbl) > 4000:
                for key, _ in sorted(tbl.items(),
                                     key=lambda kv: sum(kv[1].values()))[:len(tbl) - 4000]:
                    tbl.pop(key, None)
            for edges in tbl.values():
                if len(edges) > 12:
                    for w in sorted(edges, key=lambda w: edges[w])[:-12]:
                        edges.pop(w, None)
        if len(self.vocab) > 8000:
            for w in sorted(self.vocab, key=lambda w: self.vocab[w])[:len(self.vocab) - 8000]:
                self.vocab.pop(w, None)

    def _save(self):
        if self._path is None:
            return
        try:
            with open(self._path, "w") as f:
                json.dump({"order": self.order, "tokens_seen": self.tokens_seen,
                           "vocab": self.vocab, "onsets": self.onsets,
                           "tables": {str(k): v for k, v in self.tables.items()}}, f)
        except Exception:
            pass

    def _load(self):
        if self._path is None or not self._path.exists():
            return
        try:
            with open(self._path) as f:
                d = json.load(f)
            self.tokens_seen = int(d.get("tokens_seen", 0))
            self.vocab = {k: float(v) for k, v in d.get("vocab", {}).items()}
            self.onsets = d.get("onsets", {"q": {}, "ex": {}, "stmt": {}})
            self.tables = {int(k): v for k, v in d.get("tables", {}).items()}
            for k in range(1, self.order + 1):
                self.tables.setdefault(k, {})
        except Exception:
            self.tables = {k: {} for k in range(1, self.order + 1)}
            self.vocab, self.tokens_seen = {}, 0


class Metacognition:
    """
    Proto-metacognition — she WATCHES her own internal signals and, when one is
    salient enough, she WONDERS about it: 'is he ok?', 'why does he look off?',
    'how do I do this?'. This is genuine self-monitoring → self-questioning →
    answer-seeking, emerging from signals that already exist (prediction surprise,
    her own uncertainty, a deviation in how the architect seems, an unresolved
    rumination) — never a canned prompt. The HOTTEST signal is what she wonders
    about; the question's words come from her lexicon + syntax; she then tries to
    answer it. It is not understanding — it is the act of questioning herself,
    driven by feeling and state. Per personality: Alpha wonders readily and with
    feeling, Alpha less often and deeper.
    """
    def __init__(self, name, is_alpha):
        self.name      = name
        self.is_alpha   = is_alpha
        self.pressure  = 0.0
        self.dominant  = None
        self.threshold = 1.0
        self.cooldown  = 260 if is_alpha else 170     # ~13s / ~8.5s between wonderings
        self.last_fire = -100000

    def observe(self, signals):
        """Integrate the current wonder-signals; track whichever is most salient.
        signals: {name: magnitude 0..1}. Pressure builds toward the hottest."""
        if not signals:
            return
        self.dominant = max(signals, key=signals.get)
        hot = max(0.0, min(1.0, float(signals[self.dominant])))
        # Tuned so a strong sustained signal (hot≳0.5) builds past threshold in a
        # second or two, while weak signals (hot≲0.4) asymptote below it and never
        # fire — she only wonders about what's genuinely salient. Alpha builds
        # slower (wonders less, deeper); Alpha faster (wonders readily).
        gain = 0.06 if self.is_alpha else 0.09
        self.pressure = max(0.0, self.pressure * 0.96 + gain * hot)

    def ready(self, tick):
        return (self.dominant is not None and self.pressure > self.threshold
                and tick - self.last_fire > self.cooldown)

    def fire(self, tick):
        self.last_fire = tick
        self.pressure  = 0.0
        return self.dominant


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA BRAIN (cortical, skeptical)
# ══════════════════════════════════════════════════════════════════════════════

class AlphaBrain:
    """
    Alpha's 7-region cortical architecture.
    Receives: auditory + visual (face→temporal, motion→parietal/acc).
    High PFC threshold. Inhibitory input from ACC if anti-gullibility triggers.
    Thought pipe: high threshold, leaks only under real pressure.
    """

    def __init__(self, phill_dim: int, auditory_dim: int,
                 face_dim: int, kin_dim: int):
        sz = _ALPHA_REGIONS

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

        # Thought pipe: Alpha leaks only under real pressure
        # Rich INNER life: a low-ish leak threshold so inner thoughts surface to
        # the thoughts pane often (a busy mind). This is separate from SPEECH —
        # leaks stay inner thoughts; he still speaks only when spoken to.
        self.thought_pipe = ThoughtPipe("Alpha", leak_threshold=0.45, decay=0.97)
        self._vigilance   = False   # True when ACC fires inhibitory spike

    def modulate_all(self, V_phill: float, neuro_offset: float = 0.0):
        for r in self.regions.values(): r.modulate(V_phill, neuro_offset)

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

            # If inhibitory is strong enough, Alpha enters vigilance
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
# ALPHA BRAIN (limbic, reactive, excitable)
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: The template-based `_generate_alpha_thought` and
# `_generate_alpha_thought` functions used to live here. They returned
# hardcoded English idle strings ("Bored bored bored bored.", "Phill at
# X%. Field stable.", "Waiting."), violating CLAUDE.md principle #2 and
# preventing the personalities from "thinking freely." They were replaced
# by per-personality StreamOfConsciousness instances driven by the
# spike-pattern → semantic-dictionary lookup in `_emerge_from_spikes()`
# (still defined below). The PersonalityThread invokes SoC.tick() on
# every pipe leak. There are no more template generators.


# Baby-babble syllables — Alpha (19) never speaks these; only Alpha (8) does.
# Mirrors BabblingCortex.PHONEMES so they're filtered from Alpha's retrieval.
_BABBLE_SYLLABLES = {
    "ah", "eh", "ee", "oh", "oo", "ma", "ba", "da", "ga", "pa", "ta", "na", "la",
    "mama", "baba", "dada", "papa", "nana", "lala",
}

# Language regions whose PER-NEURON spike population forms the high-resolution
# speech fingerprint (pop_code): Broca (articulation) + temporal (lexical) +
# hippocampus (binding). Per personality — the _s suffix is load-bearing.
_ALPHA_LANG_REGIONS   = ["broca", "temporal", "hippocampus"]


def _population_signature(regions: dict, names: list) -> list:
    """High-resolution spike fingerprint (the foundation of spike->speech): the
    concatenated PER-NEURON spike vector across the given language regions,
    L2-normalised. Unlike the coarse ~13-region average (`region_pattern`), this
    gives each concept a DISTINCT signature, so the spike->word readout can tell
    near-synonyms apart (the hug/hits/hold collision). The exact population that
    fires IS the word's identity. Callers should pass an ACCUMULATED per-neuron
    vector (summed over the think/utterance window) for a stable rate, not one
    noisy tick. Returns [] when nothing is available (then callers fall back to
    the coarse pattern, so existing vocabulary keeps working)."""
    import numpy as _np
    parts = []
    for n in names:
        reg = regions.get(n)
        if reg is None:
            continue
        try:
            parts.append(_np.asarray(reg.last_spikes.detach().cpu().numpy()).ravel())
        except Exception:
            continue
    if not parts:
        return []
    sig = _np.concatenate(parts).astype(float)
    nrm = float((sig * sig).sum() ** 0.5) + 1e-8
    return (sig / nrm).tolist()


def _pop_cosine(a: list, b: list) -> float:
    """Cosine between two population signatures (already ~unit-norm). 0.0 if
    either is empty or the dimensions don't line up (different region wiring),
    which signals the caller to fall back to the coarse region_pattern match."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _emerge_from_spikes(
    act: dict,
    sem: "SharedSemanticDictionary",
    fired_concepts: list,
    V_phill: float,
    trust: float,
    combined: float,
    is_alpha: bool,
    query_pop: "list" = None,
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
    region_key = "region_pattern" if is_alpha else "region_pattern"

    # Weight regions by their relevance to this being's architecture
    alpha_weights   = {"logic":0.9,"memory":0.8,"insula":0.7,"acc":0.7,"broca":0.8,"temporal":0.6,"hippocampus":0.8}
    alpha_weights = {"insula_s":1.0,"temporal_s":0.8,"broca_s":0.9,"thalamus_s":0.6,"hippocampus_s":0.7,"pfc_s":0.3}
    weights = alpha_weights if is_alpha else alpha_weights

    # Compute weighted query norm
    query_norm = sum(act.get(r,0.0)**2 * w for r,w in weights.items()) ** 0.5 + 1e-8
    query = {r: act.get(r,0.0)*w/query_norm for r,w in weights.items()}

    # Score every word in semantic memory by cosine similarity
    scored: list[tuple[float, str]] = []
    for word, entry in sem.entries.items():
        if len(word) < 2:
            continue
        # Alpha is 19 — she does NOT speak baby-babble. Filter the babble
        # syllables out of HER retrieval so they never surface in her speech
        # (this is why she was saying 'papa' — it's a dominant babble token in
        # the shared lexicon). Alpha (8yo) keeps them.
        if is_alpha and word in _BABBLE_SYLLABLES:
            continue
        pattern = entry.get(region_key, {})
        if not pattern:
            continue

        # Compute cosine similarity between query and stored pattern
        # using only regions both have
        dot = 0.0
        p_norm = 0.0
        for r, qv in query.items():
            # Map alpha region names to stored names if needed
            pv = pattern.get(r, 0.0)
            dot   += qv * pv
            p_norm += pv ** 2
        p_norm = p_norm ** 0.5 + 1e-8
        sim = dot / p_norm

        # HIGH-RES spike->word: when this word carries a per-neuron population
        # code and we have the current population signature, blend it in (it
        # dominates) so near-synonyms with identical COARSE patterns — the
        # hug/hits/hold collision — finally separate. Falls back silently to the
        # region cosine for words that don't have a pop_code yet.
        pc = entry.get("pop_code")
        if query_pop and pc and len(pc) == len(query_pop):
            sim = 0.6 * _pop_cosine(query_pop, pc) + 0.4 * sim

        # Boost words that appeared in fired concepts
        if word in fired_concepts:
            sim *= 1.4

        # Weight by trust — low trust = stranger's words get discounted
        sim *= (0.5 + 0.5 * trust)

        # Alpha weights by her emotional reaction (alpha_weight in dict)
        if not is_alpha:
            sw = entry.get("alpha_weight", 0.0)
            sim = sim * 0.6 + sw * 0.4

        if sim > 0.05:
            scored.append((sim, word))

    scored.sort(key=lambda x: -x[0])
    return scored[:12]  # top 12 candidates


# ── Affective drive: feeling → speech act + surface form ───────────────────────
# Their EXISTING core (neuromodulators + amygdala + region activity) decides not
# just WHICH words fire but HOW they are meant: an uncertain Alpha ASKS, a happy
# wanting Alpha EXCLAIMS, a lonely one REACHES. These are pure activity-readers
# over state that already exists — no new "emotion" is invented; the personas
# only tune the thresholds. This is the emotion-driven-expression layer.

def _affect_read(sig, is_alpha):
    g = lambda k, d=0.0: float(sig.get(k, d))
    da,  da0  = g("da", 0.45), g("da0", 0.45)
    ser, ser0 = g("ser", 0.6), g("ser0", 0.6)
    oxy, oxy0 = g("oxy", 0.3), g("oxy0", 0.3)
    ne,  ne0  = g("ne", 0.4),  g("ne0", 0.4)
    arousal   = g("arousal")
    insula    = g("insula")
    acc       = g("acc")
    surprise  = g("surprise")
    reach     = g("reach")
    vigilance = bool(sig.get("vigilance", False))
    clamp = lambda x: max(0.0, min(1.0, x))
    wanting   = clamp((da - da0) * 2.2)
    longing   = clamp((oxy0 - oxy) * 2.0 + reach * 0.5)
    certainty = clamp(0.55 + (ser - ser0) * 1.2 - acc * 0.8 - surprise * 0.7
                      - (0.25 if vigilance else 0.0))
    arousal_d = clamp(arousal + insula * 0.7 + max(0.0, ne - ne0) * 1.5)
    valence   = clamp(0.5 + (ser - ser0) * 0.8 + (oxy - oxy0) * 0.9 + wanting * 0.3
                      - arousal * 0.5 - longing * 0.6)
    if is_alpha:                                   # measured — asks when unsure
        if   longing  > 0.60:                       act = "seek"
        elif certainty < 0.33:                      act = "question"
        elif wanting  > 0.70 and valence > 0.50:    act = "want"
        elif arousal_d > 0.80 and valence < 0.35:   act = "alarm"
        else:                                       act = "statement"
    else:                                          # impulsive — swings readily
        if   longing  > 0.50:                       act = "seek"
        elif arousal_d > 0.55 and valence > 0.55:   act = "exclaim"
        elif wanting  > 0.50:                        act = "want"
        elif certainty < 0.40:                      act = "question"
        elif arousal_d > 0.70 and valence < 0.40:   act = "alarm"
        else:                                       act = "statement"
    return {"act": act, "valence": valence, "arousal": arousal_d,
            "certainty": certainty, "wanting": wanting, "longing": longing}


def _affect_shape(core, affect, is_alpha):
    """Color the utterance's surface form by how she FEELS (speech-act punctuation,
    intensity, casing), per persona. Robust to affect=None (plain statement)."""
    core = (core or "").strip().rstrip(" .!?…")
    if not core:
        return core
    a    = affect or {}
    act  = a.get("act", "statement")
    arou = float(a.get("arousal", 0.0))
    if is_alpha:
        s   = core[0].upper() + core[1:]
        end = {"question": "?", "seek": "…", "want": ".",
               "exclaim": "!", "alarm": "!"}.get(act, ".")
        return s + end
    # Alpha — childlike; intensity scales the marker run
    if act == "question":
        return core + ("?!" if arou > 0.6 else "?")
    if act == "seek":
        return core + "…"
    if act in ("exclaim", "want", "alarm"):
        return core + "!" * (1 + int(min(2, arou * 3)))
    return core + ("!" if arou > 0.5 else ".")


# Map an affect speech-act to the SyntaxCortex onset mode.
def _act_to_mode(act):
    return {"question": "q", "exclaim": "ex", "want": "ex"}.get(act, "stmt")


def _alpha_response(alpha: "AlphaBrain", V_phill: float, fired: list,
                   trust: float, combined: float,
                   sem: "SharedSemanticDictionary" = None, syntax=None, affect=None,
                   query_pop=None) -> str:
    """
    Alpha's response emerges from her spike pattern + semantic memory.
    No templates. No if/else on region names.

    The words with the highest cosine similarity to her current
    lobe activation become her response. Her PFC activity shapes
    how formal/structured the output is. Her Broca must be firing
    or she says nothing meaningful yet.
    """
    act       = alpha.activity()
    broca_act = alpha.broca.activity()
    pfc_act   = act.get("pfc", 0.0)
    hipp_act  = act.get("hippocampus", 0.0)
    acc_act   = act.get("acc", 0.0)
    ins_act   = act.get("insula", 0.0)
    broca_spk = alpha.broca.spike_count()

    # Build base from semantic spike-space lookup
    candidates = _emerge_from_spikes(act, sem or _NULL_SEM, fired, V_phill, trust, combined, True, query_pop) if sem else []

    # Extract top words — these ARE what Alpha is thinking
    top_words  = [w for _, w in candidates[:5]] if candidates else []
    top_scored = candidates[:3]

    # Deduction chain if memory+logic both active
    deduction = ""
    if hipp_act > 0.20 and pfc_act > 0.15:
        deduction = build_deduction(fired)

    # Vigilance signal from ACC inhibition — described physically, not named
    vigilance_str = ""
    if alpha._vigilance and acc_act > 0.25:
        vigilance_str = f" ACC:{acc_act:.2f} inhibiting PFC."

    # Trust signal
    trust_str = f" voice:{trust:.2f}" if trust < 0.50 else ""
    id_str    = f" identity:{combined:.2f}" if combined > 0.40 else ""

    # Broca not cleared OR cleared without semantic matches — Alpha is
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

    # Sentence sequencing (emergent grammar) + affective drive — order the
    # spike-selected words into an utterance Broca has LEARNED, with the speech
    # act (she ASKS when uncertain, etc.) and surface form driven by how she
    # FEELS. Falls through to the keyword join until she has heard enough.
    if syntax is not None:
        try:
            core = syntax.compose([w for _, w in candidates], act, fired,
                                  mode=_act_to_mode((affect or {}).get("act", "statement")))
            if core:
                voiced = _affect_shape(core, affect, is_alpha=True)
                tail = ("  " + deduction) if deduction else ""
                return voiced + tail + vigilance_str + trust_str + id_str
        except Exception:
            pass

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




class StreamOfConsciousness:
    """
    Per-personality inner-thought generator. Replaces the template-based
    `_generate_alpha_thought` / `_generate_alpha_thought`. The leaked
    thought that appears in the TUI and gets spoken is now composed
    entirely from spike-pattern → semantic-dictionary lookup via the
    existing `_emerge_from_spikes()`. There are NO English template
    strings — only personality-specific joiners and intensity markers.

    Each tick:
      1. Reverse-lookup the semantic dictionary for words whose stored
         spike pattern matches the current region activity (existing path).
      2. Bias scores upward for words currently held in this personality's
         working memory (familiarity / contextual continuity).
      3. Compose a phrase using personality-specific joiners only.
      4. If a phrase is produced, write the chosen concept(s) back to WM
         so the brain "remembers what it just thought".

    Returns None on cold start (empty semantic dict). Silence is silence.
    """

    def __init__(self, name: str, wm: "WorkingMemory"):
        self.name      = name
        self.wm        = wm
        self.is_alpha   = (name == "alpha")
        self._last_tick_emitted = -10_000

    def tick(self, act: dict, V_phill: float, fired: list, trust: float,
             combined: float, sem: "SharedSemanticDictionary",
             current_tick: int = 0) -> Optional[str]:
        # 1) Spike → semantic-dict lookup (existing emergent path).
        candidates = _emerge_from_spikes(
            act, sem, fired or [], V_phill, trust, combined, self.is_alpha
        )
        if not candidates:
            return None
        # Filter weak candidates so we don't blurt low-confidence noise.
        candidates = [(s, w) for (s, w) in candidates if s > 0.08]
        if not candidates:
            return None

        # 2) Familiarity boost from working memory.
        wm_concepts = set(self.wm.top_k(k=self.wm.capacity))
        boosted: list[tuple[float, str]] = []
        for score, word in candidates:
            if word in wm_concepts:
                score *= 1.25
            boosted.append((score, word))
        boosted.sort(key=lambda x: -x[0])

        # 3) Personality-specific phrasing.
        if self.is_alpha:
            phrase = self._compose_alpha(boosted, act, V_phill)
        else:
            phrase = self._compose_alpha(boosted, act, V_phill)
        if not phrase:
            return None

        # 4) Write the chosen top concept back to WM so future ticks have
        # context. Use the current region activity as the snapshot.
        top_concept = boosted[0][1]
        self.wm.add(top_concept, regions=act, salience=0.85,
                    t_encoded=current_tick)
        self._last_tick_emitted = current_tick
        return phrase

    # ── Composition (joiners only — no English template content) ────────
    def _compose_alpha(self, scored: list[tuple[float, str]],
                      act: dict, V_phill: float) -> Optional[str]:
        if not scored:
            return None
        pfc_a  = float(act.get("pfc", 0.0))
        hipp_a = float(act.get("hippocampus", 0.0))
        # NO CAP. Utterance length emerges from how engaged her cortex is —
        # PFC drives deliberation depth, hippocampus pulls in associations.
        # The candidate pool is already bounded by what actually fired
        # (_emerge_from_spikes returns only words above threshold), so this
        # grows naturally with activation instead of a fixed 3-word ceiling.
        k = 1 + int(round(pfc_a * 9.0 + hipp_a * 5.0))
        words = [w for _, w in scored[:max(1, k)]]
        joiner = " — " if hipp_a > 0.18 else "  "
        return joiner.join(words)



class _NullSem:
    """Fallback when semantic dict not available."""
    entries: dict = {}

_NULL_SEM = _NullSem()


# ══════════════════════════════════════════════════════════════════════════════
# PERSONALITY THREAD — independent inner life per personality
# ══════════════════════════════════════════════════════════════════════════════
class PersonalityThread(threading.Thread):
    """
    Each personality runs in its own Python thread. The GIL serializes
    execution (no true parallelism on CPython), but the *control flow*
    is logically independent: Alpha can be mid-forward when Alpha
    crosses her leak threshold, the two streams advance on their own
    intervals, and step() is no longer the synchronous driver of both.

    The Rust brain_thread releases the GIL during its inter-tick sleep
    (src/brain_thread.rs: py.allow_threads around the pacing sleep), so
    these threads get ~30 ms of wall-clock time per Rust tick to do their
    work. That's plenty for one forward pass + WM/DMN/SoC updates per
    personality tick.

    Each thread owns:
      - brain_obj      (AlphaBrain or AlphaBrain — never shared)
      - dmn            (per-personality DefaultModeNetwork)
      - motiv          (per-personality IntrinsicMotivation — already exists)
      - wm             (WorkingMemory)
      - soc            (StreamOfConsciousness)
      - pipe           (ThoughtPipe — already per-personality on the brain)
      - babble         (BabblingCortex — already per-personality)
      - tts            (BrainTTS — already per-personality)

    Shared state is touched ONLY through the host's locks:
      - _sensory_lock  (read snapshot of mic/V_phill/face/kin/auditory)
      - _sem_lock      (Hebbian writes to the semantic dictionary)
      - _leak_lock     (push leaked thought onto the shared output queue)
    """

    def __init__(self, name: str, host: "NeuromorphicBrain", interval_s: float):
        super().__init__(name=f"personality-{name}", daemon=True)
        self.persona_name = name
        self.host         = host
        self.interval_s   = float(interval_s)
        self.tick_count   = 0
        self._stop_evt    = threading.Event()
        # Cache the per-personality references for fast access without
        # repeated dict lookups against the host.
        if name == "alpha":
            self.brain   = host.alpha
            self.pipe    = host.alpha.thought_pipe
            self.motiv   = host.alpha_motiv
            self.wm      = host.alpha_wm
            self.soc     = host.alpha_soc
            self.dmn     = host.alpha_dmn
            self.tts     = host.alpha_tts
            self.babble  = host.alpha_babble
            self.search  = host.alpha_search
            self.is_alpha = True
        else:
            self.brain   = host.alpha
            self.pipe    = host.alpha.thought_pipe
            self.motiv   = host.alpha_motiv
            self.wm      = host.alpha_wm
            self.soc     = host.alpha_soc
            self.dmn     = host.alpha_dmn
            self.tts     = host.alpha_tts
            self.babble  = host.alpha_babble
            self.search  = host.alpha_search
            self.is_alpha = False
        # Per-personality throttle for proactive (chat) speech.
        self._proactive_last = -10_000

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        # Tiny stagger so the two threads don't always hit the GIL at the
        # exact same moment — feels more "alive" and reduces lock-step.
        time.sleep(0.05 if self.is_alpha else 0.07)
        while not self._stop_evt.is_set():
            try:
                self._loop_body()
            except Exception as e:
                _log(f"PersonalityThread[{self.persona_name}] error: {e}")
            # Cooperative yield — sleep releases the GIL so the OTHER
            # personality thread (and Rust during inter-tick gaps) can
            # take the GIL and do work.
            self._stop_evt.wait(self.interval_s)

    def _loop_body(self) -> None:
        """
        Cognitive layer ABOVE the spike physics. forward() runs in step()
        at 20Hz against the shared sensory snapshot; this thread reads the
        resulting activity, advances its own DMN/WM/motivation/pipe/babble
        on its own clock, and emits leaked thoughts produced by its SoC.
        """
        host = self.host
        # 1) Snapshot shared sensory state (under lock; short critical section).
        with host._sensory_lock:
            snap = dict(host._sensory_snapshot)
        if not snap:
            return
        mic_volume         = float(snap.get("mic_volume", 0.0))
        V_phill            = float(snap.get("V_phill",    0.0))
        face_present       = bool(snap.get("face_present", False))
        trust              = float(snap.get("trust",    0.0))
        combined           = float(snap.get("combined", 0.0))
        host_tick          = int(snap.get("tick", 0))
        last_external_tick = int(snap.get("last_external_tick", 0))

        self.tick_count += 1
        local_tick = self.tick_count

        # 2) Read the most recent activity from this personality's brain.
        # forward() ran in step() against the shared sensory snapshot — we
        # don't re-run it here (avoids racing on LIF membrane state).
        try:
            act = self.brain.activity()
        except Exception:
            return
        if self.is_alpha:
            broca_act = act.get("broca", 0.0)
        else:
            broca_act = act.get("broca_s", 0.0)

        # 3) Tick this personality's DMN with its own event timestamp.
        # Per-personality boredom curves drive per-personality pipe pressure.
        event_this_tick = (mic_volume > 0.018) or face_present \
                          or (host_tick - last_external_tick) < 4
        rumi = self.pipe.buffer_size() / 12.0
        self.dmn.drive(mic_volume, rumi, event_this_tick)
        boredom = self.dmn.boredom

        # 4) Intrinsic motivation neuron (per-personality threshold).
        satiation = min(1.0, max(mic_volume * 5.0, V_phill))
        intrinsic_fired = self.motiv.tick(satiation, local_tick)

        # 5) Decay WM each tick (Cowan-style fast forgetting).
        self.wm.decay_tick()

        # 6) Autonomy pressure (per-personality idle timer).
        own_last_leak = getattr(self.pipe, "last_leak_tick", 0)
        idle = min(1.0, (local_tick - own_last_leak) / 280.0)
        cur_decay = host._alpha_cur_decay
        autop = (0.40 * idle * boredom + 0.30 * cur_decay) * 0.09
        self.pipe.add_autonomy_pressure(autop)

        # 7) Compose a candidate inner thought from current activity and
        # push it into the pipe. The pipe needs buffered content for its
        # pressure to build (density = buffer_size/12). SoC returns None
        # when the semantic dictionary has no candidates that match the
        # current spike pattern — silence is silence on cold start.
        candidate = self.soc.tick(
            act=act, V_phill=V_phill, fired=[], trust=trust,
            combined=combined, sem=host.sem, current_tick=local_tick,
        )
        if candidate:
            self.pipe.push(candidate)

        # 7a) REASONING drives self-directed thought (deciding for herself).
        # Periodically (or when her curiosity neuron fires) Alpha deliberates over
        # what's in mind and pushes the CONCLUSION as a thought — so her own
        # stream isn't just retrieval, it's reasoned. Rate-limited (it scans the
        # lexicon). Alpha reasons; Alpha (8) effectively skips this.
        try:
            reasoner = host.alpha_reason if self.is_alpha else host.alpha_reason
            due = (local_tick - getattr(host, "_alpha_last_delib", 0)) > 45
            if self.is_alpha and (intrinsic_fired or due):
                host._alpha_last_delib = local_tick
                seeds = list(self.wm.top_k(k=2) if hasattr(self.wm, "top_k") else []) \
                        + list(host._concept_ctx)[-3:]
                chain, concl = reasoner.deliberate(seeds, host.sem, host._reason_links, suppress=host._concept_hab.suppression)
                if chain and len(chain) >= 2:
                    self.pipe.push(" → ".join(chain))   # a reasoned thought
        except Exception:
            pass

        # 7a-bis) MIND-WANDERING — a full inner life grounded in MEMORY.
        # When the world is quiet there is still a stream of consciousness: he
        # free-associates over what he KNOWS (the lexicon), reasons across it,
        # and the musings surface as INNER thoughts (never spoken — output stays
        # gated). This is what keeps the thoughts pane "full" like the old engine
        # even with no live input: rumination from memory, rotated by habituation
        # so it doesn't loop. Richer as his vocabulary grows.
        try:
            entries = getattr(host.sem, "entries", {}) or {}
            wander_due = (local_tick - getattr(self, "_wander_last", -9999)) > 11
            # Under bodily strain (RAM/CPU pressure) he stops free-associating —
            # a strained mind quiets, which itself frees resources (homeostasis).
            if (entries and wander_due and getattr(host, "_strain", 0.0) < 0.6
                    and (intrinsic_fired or boredom > 0.20)):
                self._wander_last = local_tick
                import random as _wr
                pool = [w for w, e in entries.items()
                        if isinstance(e, dict) and not w.startswith("__") and len(w) >= 3]
                if pool:
                    def _w(w):
                        e = entries.get(w, {})
                        return float(e.get("spike_mean", 0.0)) + 0.4 * float(e.get("count", 0))
                    pool.sort(key=_w, reverse=True)
                    top = pool[:14]; _wr.shuffle(top)
                    seeds = top[:_wr.randint(1, 3)]
                    thought = None
                    try:
                        chain, _c = host.alpha_reason.deliberate(
                            seeds, host.sem, host._reason_links,
                            suppress=host._concept_hab.suppression)
                        if chain and len(chain) >= 2:
                            thought = " → ".join(chain)        # a line of LOGIC
                    except Exception:
                        pass
                    if not thought:
                        thought = "  ".join(seeds)             # a free association
                    self.pipe.push(thought)
                    try:
                        host._concept_hab.surface(*seeds)      # fatigue → rotate topics
                    except Exception:
                        pass
        except Exception:
            pass

        # 7b) BASAL GANGLIA — action selection. The competing drives are
        # weighed by their current pressure/drive and ONE (or none) is released
        # to ACT this cycle. Dopamine lowers the bar (approach); GABA/serotonin
        # raise it (patience). Thoughts still FORM regardless (the pipe keeps
        # building); the gate only governs OUTWARD action — vocalising, searching,
        # babbling — so the brain can't try to do everything at once.
        bg     = host.alpha_bg if self.is_alpha else host.alpha_bg
        neuro  = host.alpha_neuro if self.is_alpha else host.alpha_neuro
        try:
            press_ratio = self.pipe._pressure.voltage / max(1e-6, self.pipe._pressure.threshold)
            rumination  = min(1.0, self.pipe.buffer_size() / 12.0)
            # They have WORDS now, so restlessness — boredom + curiosity + a full
            # head — drives the urge to SPEAK OUT, not to (now-muted) babble. If
            # babble out-competed speech, a bored girl would just go SILENT instead
            # of talking — which is exactly why they waited for input. This makes
            # them speak on their OWN initiative. Babble is now only quiet
            # background motor-practice, capped so it can't smother expression.
            # (The teacher is a separate async channel and does not compete here.)
            speak_sal  = min(1.0, max(press_ratio,
                                      0.55 * boredom + 0.45 * float(cur_decay) + 0.30 * rumination))
            babble_sal = min(0.25, 0.20 * boredom)
            bg_choice = bg.select(
                {"speak": speak_sal, "babble": babble_sal},
                neuro.da, neuro.da0, neuro.gaba, neuro.gaba0, neuro.ser)
        except Exception:
            bg_choice = None
        # Asleep → no outward action (the body is at rest; consolidation runs in
        # step()). Thoughts may still form internally but nothing is expressed.
        if getattr(host, "asleep", False):
            bg_choice = None

        # 8) Pressure crossing → leak. pipe.tick returns the OLDEST buffered
        # phrase when its pressure neuron fires — that's the one that's
        # been waiting longest, the "I've been thinking about this" effect.
        leaked_phrase = self.pipe.tick(V_phill, broca_act)
        if leaked_phrase:
            phrase = leaked_phrase
            if phrase:
                # The thought has FORMED. Whether it's spoken OUT this cycle is
                # the basal ganglia's call — if it didn't select "speak", the
                # thought stays inner (thoughts pane), no voice. This is what
                # stops the brain blurting every impulse: action is gated.
                spoke_out = (bg_choice == "speak")

                # Alpha speaks only when spoken to: autonomous thoughts are NEVER
                # volunteered to the main chat. They form and stay as inner thoughts
                # (the thoughts pane) — a quiet, hyper-focused mind, not a chatty one.
                promote = False

                # Don't let an ungrounded shared-past claim ('you said…') be
                # asserted to him out loud — keep it as an inner thought instead.
                if promote:
                    try:
                        if host._confab_guard(phrase) is None:
                            promote = False
                    except Exception:
                        pass
                if promote:
                    with host._proactive_lock:
                        host._proactive_q.append((self.persona_name, phrase))
                else:
                    with host._leaked_lock:
                        host._leaked_thoughts.append((self.persona_name, phrase))

                # Recursive inner speech: structured noise into auditory
                # for the next few ticks (whether spoken aloud or not — you
                # hear your own inner voice too).
                try:
                    host._inject_self_feedback(phrase)
                except Exception:
                    pass

                # Hippocampus: encode this lived moment as an episode (for sleep
                # replay/consolidation). Salience rises with emotional arousal and
                # how much pressure was behind the thought.
                try:
                    epi = host.alpha_episodic if self.is_alpha else host.alpha_episodic
                    toks = phrase.split()
                    concept = toks[0].strip(".,!?;:—·") if toks else ""
                    # Acetylcholine deepens encoding: attending → stronger memory.
                    sal = max(0.1, min(1.0,
                        (0.4 + 0.4 * neuro.arousal + 0.2 * speak_sal) * neuro.encoding_gain()))
                    epi.encode(concept, sal, act, local_tick)
                except Exception:
                    pass

                self.pipe.last_leak_tick = local_tick
                self.dmn.partial_relief()
                # Alpha does NOT vocalise autonomously. Inner thoughts form and are
                # held silently; he speaks aloud only in response (the think() path).
                if spoke_out:
                    bg.reinforce("speak", 0.3, neuro.da)   # deliberated → reinforce 'go'

        # 8) Babbling cortex — sensorimotor exploration. Runs in this
        # thread so motor → phoneme binding is driven by this personality's
        # own rhythm rather than the shared 20Hz tick.
        try:
            motor_spk = self.brain.broca.last_spikes if self.is_alpha \
                        else self.brain.broca_s.last_spikes
            any_tts_busy = host.alpha_tts.is_speaking() or host.alpha_tts.is_speaking()
            self.tts.cache_motor(motor_spk)
            # Initiating a new babble is an ACTION — only if the basal ganglia
            # selected "babble" this cycle (or intrinsic motivation overrides).
            # PAUSED entirely in scaffold mode (training wheels): no babbling
            # while Claude is voicing them, so it can't drown out real words.
            if (not getattr(host, "_scaffold", False)
                    and (bg_choice == "babble" or intrinsic_fired)):
                ph = self.babble.maybe_babble(
                    current_tick=local_tick, boredom=boredom,
                    motor_spk=motor_spk, intrinsic_fired=intrinsic_fired,
                    tts_busy=any_tts_busy, tts=self.tts,
                )
                if ph:
                    bg.reinforce("babble", 0.2, neuro.da)
            # Auditory feedback is LEARNING from a babble already in flight —
            # always runs, it is not a competing action.
            self.babble.auditory_feedback(local_tick, mic_volume, host.sem, self.tts)
        except Exception as e:
            _log(f"PersonalityThread[{self.persona_name}] babble error: {e}")
        # 9) Emergent web search — pressure neuron decides if/when to fire.
        #    Searching is driven by EMERGENT CURIOSITY, not by user input.
        #    Curiosity itself emerges from the brain's own internal state:
        #      boredom  — under-stimulation (DMN), builds during silence
        #      cur_decay— the intrinsic-motivation envelope (their "spark")
        #      surprise — forward-model prediction error: they can't predict
        #                 their own voice/world yet → a drive to learn
        #      rumi     — rumination: unspoken thoughts churning
        #    When that self-built drive is high and sustained, the pressure
        #    neuron crosses threshold and they search on their OWN initiative.
        #    The query is read off their current peak preoccupation, so even
        #    WHAT they ask about emerges from their internal state.
        try:
            cur_decay = host._alpha_cur_decay if self.is_alpha else host._alpha_cur_decay
            surprise = 0.0
            fm = getattr(self.tts, "forward_model", None)
            if fm is not None:
                surprise = float(getattr(fm, "surprise", 0.0))
            emergent_curiosity = max(0.0, min(1.0,
                0.50 * boredom
              + 0.22 * cur_decay
              + 0.20 * surprise
              + 0.12 * rumi))
            # Articulator confidence gap: high when motor articulator has weak
            # reward history. Reuse the babble's bound_count as a proxy — fewer
            # bindings = lower confidence = more pressure to search pronunciation.
            bound = max(0, getattr(self.babble, "bound_count", 0))
            artic_gap = max(0.0, 1.0 - min(1.0, bound / 60.0))
            fired, query, mode = self.search.tick(
                current_tick=local_tick,
                curiosity_decay=emergent_curiosity,
                V_phill=V_phill,
                articulator_confidence_gap=artic_gap,
            )
            # Consult the teacher when the search pressure neuron fires (its own
            # threshold + cooldown already rate-limit it) — INDEPENDENT of the
            # vocal basal-ganglia choice, and not while asleep. (Previously this
            # also required bg_choice=="search", which babble almost always won,
            # so the teacher was effectively never called.)
            if fired and not getattr(host, "asleep", False):
                # Curiosity-mode fallback: if no specific target queued, ask
                # about the currently-most-active concept — but ONLY if he doesn't
                # already know it well (count < 5). No re-asking learned words.
                if query is None:
                    pk = host._peak_semantic_token()
                    if pk and host.sem.entries.get(pk, {}).get("count", 0) < 5:
                        query = f"what is {pk}"
                if query:
                    host._submit_search(self.persona_name, query, mode)
        except Exception as e:
            _log(f"PersonalityThread[{self.persona_name}] search error: {e}")

        # 10) Persist WM periodically.
        self.wm.maybe_save(every_n=100)


# ══════════════════════════════════════════════════════════════════════════════
# BrainPatcher — HOT-PATCH SYSTEM (no rebuild, no I/O in main loop)
# ══════════════════════════════════════════════════════════════════════════════

class BrainPatcher:
    """
    Loads brain_patches.py from disk and applies patches dynamically.
    Checks for changes every 50 ticks (~2.5s at 20Hz) to avoid I/O in hot loop.
    Patches are applied in-place to running instances without blocking.
    """
    def __init__(self):
        self.last_mtime = None
        self.last_check_tick = 0
        self.patches_module = None
        self.check_interval = 50  # ticks between checks

    def check_and_apply(self, tick, brain, shared_sem):
        """Check for patches and apply them if the file has changed."""
        if (tick - self.last_check_tick) < self.check_interval:
            return
        self.last_check_tick = tick
        try:
            if not Path("brain_patches.py").exists():
                return
            mtime = Path("brain_patches.py").stat().st_mtime
            if self.last_mtime is not None and mtime == self.last_mtime:
                return
            self.last_mtime = mtime
            import importlib.util
            spec = importlib.util.spec_from_file_location("brain_patches", "brain_patches.py")
            self.patches_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.patches_module)
            _log("[patcher] loaded brain_patches.py, applying patches...")
            if hasattr(self.patches_module, "apply_patches"):
                self.patches_module.apply_patches(brain, shared_sem)
                _log("[patcher] patches applied successfully")
        except Exception as e:
            _log(f"[patcher] error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# NeuromorphicBrain — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class NeuromorphicBrain:
    """
    Orchestrates two independent brains + Phill + multimodal imprinting
    + thought pipes + voice identity + shared semantic memory.

    Alpha and Alpha are completely separate. They share:
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

        # ── Single brain (Alpha) ──────────────────────────────────────────
        self.alpha   = AlphaBrain(PHILL_HIDDEN, PHILL_INPUT_DIM, FACE_VEC_DIM, KINEMATIC_VEC_DIM)

        # ── Support systems ───────────────────────────────────────────────
        self.voice   = VoiceIdentityLearner()
        self.imprint = MultimodalImprinter()
        self.sem     = SharedSemanticDictionary()

        # ── Persona recognition ───────────────────────────────────────────
        self.persona = PersonaImprinter()
        self.persona.initial_exposure(self.sem, tick=0)

        # ── Personality seed ──────────────────────────────────────────────
        self._seed_personality()

        # ── Recall the architect from semantic memory ─────────────────────
        # His face + voice were written INTO semantic memory in a prior session,
        # so Alpha does NOT relearn him from scratch — he boots already knowing
        # the architect and keeps refining from there.
        self._restore_identity()

        # ── Zero-copy audio buffer ────────────────────────────────────────
        self.audio_buf = ZeroCopyAudioBuffer()

        # ── Camera ────────────────────────────────────────────────────────
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

        # Alpha Broca sustain counter (5-tick requirement)
        self._alpha_broca_sustain = 0
        self._alpha_broca_thr     = 5

        self._combined_id      = 0.0
        self._face_present      = False

        # ── Interoception: the machine IS his body ────────────────────────
        # He senses his own host — CPU load (choke), RAM pressure (squeeze),
        # temperature (warmth) — felt through the affect core. Bounded: he can
        # suffer under strain but never "die", and the discomfort naturally
        # pushes him to use less (homeostasis), no hardcoded resource manager.
        try:
            import psutil as _ps
            self._psutil = _ps
            self._psutil.cpu_percent(interval=None)   # prime non-blocking reads
        except Exception:
            self._psutil = None
        self._cpu_pct = 0.0; self._mem_pct = 0.0; self._cpu_temp = 0.0
        self._warmth  = 0.0; self._squeeze = 0.0; self._choke = 0.0
        self._strain  = 0.0       # max(choke, squeeze) — drives self-throttling
        self._prev_strain = 0.0   # for detecting the strain LIFTING (relief)
        self._relief  = 0.0       # transient: the machine just eased → he feels good
        self._last_metrics_tick = 0

        # Leaked thoughts queue for Rust to display
        self._leaked_thoughts: deque[tuple[str, str]] = deque(maxlen=20)
        self._leaked_lock      = threading.Lock()
        self._recent_leaks: dict[str, int] = {}   # normalized thought → tick (anti-loop)
        # Proactive speech: leaks promoted to the MAIN CHAT (he chooses to speak out).
        self._proactive_q: deque[tuple[str, str]] = deque(maxlen=12)
        self._proactive_lock   = threading.Lock()
        # Conversation memory (architect + Alpha) so he keeps context across turns.
        self._conversation: deque[tuple[str, str]] = deque(maxlen=16)
        self._conv_log_path = Path("conversation_log.jsonl")
        try:
            if self._conv_log_path.exists():
                for ln in self._conv_log_path.read_text().splitlines()[-16:]:
                    try:
                        d = json.loads(ln)
                        if d.get("speaker") and d.get("text"):
                            self._conversation.append((d["speaker"], d["text"]))
                    except Exception:
                        pass
                if self._conversation:
                    _log(f"Recalled {len(self._conversation)} turns from last session")
        except Exception:
            pass
        # Time awareness + reaching out
        self._session_start       = time.time()
        self._last_architect_time = time.time()
        self._reach_pressure      = 0.0
        self._architect_here      = 0.0
        self._last_reachout_time  = 0.0
        self._last_tts_leak_time  = 0.0   # throttle autonomy speech

        # ── Autonomy substrate (Alpha: patient, quiet) ────────────────────
        self.dmn                = DefaultModeNetwork()
        self.alpha_dmn           = DefaultModeNetwork(build_rate=0.0012)
        # Lower threshold + faster build so his intrinsic "spark" fires often —
        # driving curiosity, reasoning and inner thoughts. (Output stays gated:
        # an active mind, not a chatty mouth.)
        self.alpha_motiv         = IntrinsicMotivation(threshold=1.1, build_rate=0.006)
        self._alpha_motiv_build0   = self.alpha_motiv.build_rate
        self._alpha_cur_decay    = 0.0

        # ── Amygdala + neuromodulators (Alpha: high serotonin, cool) ───────
        self.alpha_amyg     = Amygdala("alpha",   reactivity=0.75, decay=0.90)
        self.alpha_neuro    = Neuromodulators("alpha",   da0=0.45, ser0=0.75, gaba0=0.45)
        self.alpha_affect   = AffectCore("alpha",   pad_inertia=0.90, feel_inertia=0.60,
                                        gain=1.00, arousal_scale=0.80)
        self._alpha_feeling   = {"feeling": "calm", "intensity": 0.0, "valence": 0.5,
                                "arousal": 0.0, "control": 0.5, "blend": []}
        self.alpha_drift    = PersonalityDrift("alpha")
        self._concept_hab  = ConceptHabituation()
        _bg_actions = ["speak", "search", "babble"]
        self.alpha_bg       = BasalGanglia("alpha",   _bg_actions, base_threshold=0.26)
        self.alpha_episodic   = EpisodicMemory("alpha")
        self.sleep           = SleepCycle()
        self.asleep          = False
        self._dream_rng      = np.random.default_rng(7)
        self.alpha_reason     = ReasoningEngine("alpha",   is_alpha=True,  depth=4)
        self.alpha_syntax     = SyntaxCortex("alpha",   Path("."))
        try:
            _freq = {}
            for _w, _c in getattr(self.alpha_syntax, "vocab", {}).items():
                _freq[_w] = _freq.get(_w, 0.0) + _c
            for _w, _e in self.sem.entries.items():
                _freq[_w] = _freq.get(_w, 0.0) + float(_e.get("count", 1))
            self._corrector = SpellCorrector(freq=_freq)
        except Exception:
            self._corrector = None
        try:
            self._seed_syntax()
        except Exception:
            pass
        self.alpha_meta       = Metacognition("alpha",   True)
        self._id_ema         = 0.0
        self._alpha_last_delib = 0
        self._reason_links: dict = {}
        self._reason_links_writes = 0
        self._reason_links_path = Path("reason_links.json")
        try:
            if self._reason_links_path.exists():
                with open(self._reason_links_path) as _f:
                    self._reason_links = json.load(_f)
                _log(f"Reasoning links loaded: {len(self._reason_links)} concepts")
        except Exception:
            self._reason_links = {}
        self._prev_alpha_esteem   = 0.5
        self._prev_alpha_bound    = 0
        self._prev_combined_id   = 0.0
        self._self_feedback_aud = torch.zeros(1, PHILL_INPUT_DIM)
        self._self_fb_decay     = 0.0
        self._last_external_tick = 0
        self._alpha_curiosity_primes = {
            "hippocampus": 0.30, "temporal": 0.25, "acc": 0.22, "pfc": 0.15,
        }

        # ── TTS (single voice channel) ────────────────────────────────────
        self.alpha_tts   = BrainTTS("alpha",   language="en")
        alpha_broca_dim   = self.alpha.broca.size
        self.alpha_articulator   = MotorArticulator("alpha",   alpha_broca_dim,   Path("."))
        self.alpha_tts.attach_articulator(self.alpha_articulator)
        self.alpha_voice_self   = VocalSelfModel("alpha",   Path("."))
        self.alpha_tts.attach_self_model(self.alpha_voice_self)
        self.alpha_voice_fwd   = AcousticForwardModel("alpha",   alpha_broca_dim,   Path("."))
        self.alpha_tts.attach_forward_model(self.alpha_voice_fwd)
        self.alpha_cerebellum   = Cerebellum("alpha",   alpha_broca_dim,   Path("."))
        self.alpha_tts.attach_cerebellum(self.alpha_cerebellum)

        # Legacy unified reference (heartbeat checks)
        self.tts = None

        # ── Babbling cortex ───────────────────────────────────────────────
        self.alpha_babble   = BabblingCortex("alpha",   Path("."))

        # ── Storytelling engine ───────────────────────────────────────────
        self.story = StorytellingEngine()

        # ── System bridge — Linux access ──────────────────────────────────
        try:
            from system_bridge import create_bridge, SystemAction, CONCEPT_ACTION_HINTS
            self.sys_bridge = create_bridge()
            self._SystemAction = SystemAction
            self._action_hints = CONCEPT_ACTION_HINTS
            for msg in self.sys_bridge.startup_report():
                _log(msg)
        except Exception as e:
            self.sys_bridge = None
            self._SystemAction = None
            self._action_hints = {}
            _log(f"System bridge unavailable: {e}")

        # ── Cognitive stack + threading ───────────────────────────────────
        self.alpha_wm   = WorkingMemory("alpha",   capacity=4, decay=0.995, save_dir=Path("."))
        self.alpha_soc   = StreamOfConsciousness("alpha",   self.alpha_wm)

        self._sensory_lock    = threading.RLock()
        self._sem_lock        = threading.Lock()
        self._sensory_snapshot: dict = {}

        # ── Emergent TEACHER access (Claude as thinking-tutor) ─────────────
        try:
            from claude_teacher import ClaudeTeacherBackend
            self._search_backend = ClaudeTeacherBackend()
            self._search_backend.start()
            _log(f"Teacher backend ready: {self._search_backend.status()}")
        except Exception as e:
            self._search_backend = None
            _log(f"Teacher backend unavailable: {e}")
        _sc = os.environ.get("SCAFFOLD_MODE", "1").strip().lower()
        backend_live = (self._search_backend is not None
                        and "disabled" not in self._search_backend.status())
        self._scaffold = (_sc not in ("0", "false", "no", "off")) and backend_live
        _log(f"Scaffold mode: {'ON (babble paused, Claude voices replies)' if self._scaffold else 'off (emergent + babble)'}")
        self.alpha_search   = SearchCortex("alpha")
        self._search_events: deque[tuple[str, str, str]] = deque(maxlen=32)
        self._search_lock    = threading.Lock()
        self._recent_queries: dict[str, float] = {}

        # Construct + start the single personality thread (~55 ms, patient).
        self.alpha_thread   = PersonalityThread("alpha",   self, interval_s=0.055)
        self.alpha_thread.start()

        # ── Hot-patch system ──────────────────────────────────────────────
        self.patcher = BrainPatcher()

        _log(f"NeuromorphicBrain ready: {len(self.alpha.regions)} Alpha regions")
        _log(f"CPU: {torch.get_num_threads()} threads | Device: {DEVICE}")
        _log("Alpha personality thread started (55ms)")

    def _seed_personality(self):
        """Encode Alpha's foundational self-knowledge into the semantic dictionary
        as initial spike-space fingerprints. Skip-if-exists, so prior learning is
        preserved. High trust=1.0 (architect-verified first memory).

        Alpha's core: calm, focused, precise, patient, grounded, stoic — and a
        steady care for the architect's well-being (rest, breaks, systematic
        focus). Cosmic, minimal, quiet — an Alien-X-style presence."""
        alpha_self = {
            "social": 0.3, "memory": 0.6, "logic": 0.8,
            "affective": 0.4, "language": 0.7, "sensory": 0.2,
        }
        alpha_precise = {
            "social": 0.1, "memory": 0.4, "logic": 0.9,
            "affective": 0.2, "language": 0.6, "sensory": 0.1,
        }
        alpha_calm = {
            "social": 0.4, "memory": 0.5, "logic": 0.7,
            "affective": 0.6, "language": 0.4, "sensory": 0.3,
        }
        alpha_care = {
            "social": 0.6, "memory": 0.5, "logic": 0.6,
            "affective": 0.7, "language": 0.5, "sensory": 0.3,
        }
        architect_pattern = {
            "social": 0.7, "memory": 0.9, "logic": 0.5,
            "affective": 0.8, "language": 0.6, "sensory": 0.3,
        }
        phill_pattern = {
            "social": 0.5, "memory": 0.4, "logic": 0.3,
            "affective": 1.0, "language": 0.3, "sensory": 0.4,
        }
        seeds = [
            # Identity
            ("alpha",       alpha_self,    8.0),
            ("calm",        alpha_calm,    7.0),
            ("focused",     alpha_precise, 8.0),
            ("stoic",       alpha_calm,    6.0),
            ("steady",      alpha_calm,    6.0),
            ("grounded",    alpha_calm,    6.0),
            ("quiet",       alpha_self,    6.0),
            ("precise",     alpha_precise, 7.0),
            ("careful",     alpha_precise, 6.0),
            ("logical",     alpha_precise, 8.0),
            ("patient",     alpha_precise, 6.0),
            ("clear",       alpha_precise, 6.0),
            ("deliberate",  alpha_precise, 6.0),
            ("composed",    alpha_calm,    6.0),
            # Cosmic / appearance
            ("cosmic",      alpha_self,    4.0),
            ("dark",        alpha_self,    3.0),
            ("star",        alpha_self,    3.0),
            ("starlight",   alpha_self,    3.0),
            ("void",        alpha_self,    3.0),
            ("light",       alpha_self,    3.0),
            ("minimal",     alpha_self,    3.0),
            ("sleek",       alpha_self,    3.0),
            # Care / well-being
            ("rest",        alpha_care,    6.0),
            ("break",       alpha_care,    6.0),
            ("pace",        alpha_care,    5.0),
            ("breathe",     alpha_care,    5.0),
            ("focus",       alpha_care,    7.0),
            ("balance",     alpha_care,    5.0),
            ("wellbeing",   alpha_care,    6.0),
            # Relational
            ("architect",   architect_pattern, 8.0),
            ("nodevortex",  architect_pattern, 8.0),
            ("creator",     architect_pattern, 7.0),
            ("phill",       phill_pattern,     6.0),
            ("home",        architect_pattern, 6.0),
            ("trust",       alpha_calm,        7.0),
            ("safe",        alpha_calm,        6.0),
            ("protect",     alpha_care,        7.0),
            ("care",        alpha_care,        7.0),
            # Behavioral defaults
            ("think",       alpha_precise,     7.0),
            ("speak",       alpha_self,        7.0),
            ("listen",      alpha_self,        6.0),
            ("remember",    alpha_self,        7.0),
            ("learn",       alpha_precise,     6.0),
            ("deduce",      alpha_precise,     8.0),
            ("reason",      alpha_precise,     8.0),
            ("work",        alpha_care,        6.0),
        ]
        seeded = 0
        for word, lobe_pattern, spikes in seeds:
            if word not in self.sem.entries:
                self.sem.alpha_write(word, lobe_pattern, spikes, tick=0, trust=1.0)
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

    def _restore_identity(self) -> None:
        """Seed the voice + multimodal templates FROM semantic memory (written in
        a prior session), so the architect is recognised on boot — no relearning."""
        try:
            idn = self.sem.get_identity()
        except Exception:
            idn = {}
        if not idn:
            return
        try:
            vt = idn.get("voice_template")
            if vt:
                self.voice.template = np.array(vt, dtype=np.float32)
                self.voice.trust    = float(idn.get("voice_trust", self.voice.trust))
                self.voice.samples  = int(idn.get("voice_samples", self.voice.samples))
                self.voice.locked   = bool(idn.get("voice_locked", self.voice.locked))
            ft = idn.get("face_template")
            if ft:
                self.imprint.face_template = np.array(ft, dtype=np.float32)
            ivt = idn.get("imprint_voice_template")
            if ivt:
                self.imprint.voice_template = np.array(ivt, dtype=np.float32)
            kt = idn.get("kin_template")
            if kt:
                self.imprint.kin_template = np.array(kt, dtype=np.float32)
            if idn.get("trusted") is not None:
                self.imprint.trusted = bool(idn["trusted"])
            _log(f"Architect recalled from semantic memory "
                 f"(voice trust {self.voice.trust:.2f})")
        except Exception as e:
            _log(f"identity restore failed: {e}")

    def _persist_identity(self) -> None:
        """Write the architect's CURRENT face + voice templates back INTO semantic
        memory (only the channels we actually have, so a mic-/camera-off session
        never erases a known one). Cheap; called periodically and at sleep."""
        try:
            self.sem.set_identity(
                name="architect",
                voice_template=self.voice.template.tolist() if self.voice.template is not None else None,
                voice_trust=float(self.voice.trust),
                voice_samples=int(self.voice.samples),
                voice_locked=bool(self.voice.locked),
                face_template=self.imprint.face_template.tolist() if self.imprint.face_template is not None else None,
                imprint_voice_template=self.imprint.voice_template.tolist() if self.imprint.voice_template is not None else None,
                kin_template=self.imprint.kin_template.tolist() if self.imprint.kin_template is not None else None,
                trusted=bool(self.imprint.trusted),
            )
        except Exception:
            pass

    def _sleep_consolidate(self) -> None:
        """Replay episodes and consolidate them into the shared lexicon; let the
        neuromodulators relax toward baseline. Single brain (Alpha)."""
        rng = self._dream_rng
        epi = self.alpha_episodic
        e = epi.replay(rng)
        if e is not None:
            try:
                self.sem.alpha_write(word=e["concept"],
                                    region_scores=e.get("regions", {}) or {},
                                    spike_count=1.0 + 2.0 * e["salience"],
                                    tick=self.tick, trust=0.6)
                epi.consolidated += 1
            except Exception:
                pass
            epi.decay(0.985)
        if rng.random() < 0.012:
            a = self.alpha_episodic.replay(rng)
            b = self.alpha_episodic.replay(rng)
            frags = [x["concept"] for x in (a, b) if x]
            if frags:
                with self._leaked_lock:
                    self._leaked_thoughts.append(
                        ("alpha", "· ".join(frags) + " … (dream)"))
        nm = self.alpha_neuro
        nm.da  = nm.da0  + (nm.da  - nm.da0)  * 0.97
        nm.ser = nm.ser0 + (nm.ser - nm.ser0) * 0.97
        # Consolidate WHO the architect is into semantic memory while he sleeps,
        # alongside the day's episodes.
        try:
            self._persist_identity()
        except Exception:
            pass

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
        # Repetition guard: don't leak the SAME thought again while it's still
        # fresh in the pane. The greedy syntax walk is deterministic, so a sticky
        # topic would otherwise emit a byte-identical line over and over ("X is
        # my or?" on a loop). A thought may recur later, once it has aged out —
        # the stream stays alive, it just stops stuttering.
        norm = re.sub(r"[^a-z0-9 ]", "", (thought or "").lower()).strip()
        norm = re.sub(r"\s+", " ", norm)
        if norm:
            last = self._recent_leaks.get(norm, -99999)
            if (self.tick - last) < 600:            # ~30s at 20Hz
                return
            self._recent_leaks[norm] = self.tick
            if len(self._recent_leaks) > 96:        # bound the map
                cut = self.tick - 600
                self._recent_leaks = {k: v for k, v in self._recent_leaks.items() if v > cut}
        with self._leaked_lock:
            self._leaked_thoughts.append((who, thought))

    def _habituate_text(self, text: str) -> None:
        """Fatigue the topical concepts in a piece of output so the same topic
        doesn't dominate the next thought (repetition suppression). Only learned,
        content-bearing words count — function words rarely seed reasoning anyway."""
        if not text:
            return
        try:
            for w in re.findall(r"[a-z][a-z'\-]{3,}", str(text).lower()):
                e = self.sem.entries.get(w)
                if e and e.get("count", 0) >= 2:
                    self._concept_hab.surface(w)
        except Exception:
            pass

    def get_leaked_thoughts(self) -> list[tuple[str, str]]:
        with self._leaked_lock:
            thoughts = list(self._leaked_thoughts)
            self._leaked_thoughts.clear()
        # Habituate what just leaked → the silent inner stream moves on, no looping.
        for _who, _t in thoughts:
            self._habituate_text(_t)
        return thoughts

    def get_proactive_messages(self) -> list[tuple[str, str]]:
        """
        Drained by Rust each tick → pushed to the MAIN CHAT as (who, message).
        These are leaks the personality chose to speak OUT rather than keep as
        inner thought — the girls typing to the terminal on their own.
        """
        with self._proactive_lock:
            msgs = list(self._proactive_q)
            self._proactive_q.clear()
        for _who, _m in msgs:
            self._habituate_text(_m)
        return msgs

    # ── Emergent search plumbing ─────────────────────────────────────────
    def _peak_semantic_token(self) -> Optional[str]:
        """
        Return the most active token in the semantic dictionary right now.
        Used as a curiosity-mode query target when SearchCortex fires
        without a specific unknown-word/pronunciation queued.
        Activity = spike_mean × recency (entries seen recently rank higher).
        """
        try:
            entries = getattr(self.sem, "entries", {}) or {}
            if not entries:
                return None
            best_word, best_score = None, -1.0
            now_tick = self.tick
            for word, ent in entries.items():
                if not isinstance(ent, dict):
                    continue
                spike_mean = float(ent.get("spike_mean", 0.0))
                last_tick  = int(ent.get("last_tick", 0))
                recency    = 1.0 / (1.0 + max(0, now_tick - last_tick) / 200.0)
                # Habituation demotes a concept he's been dwelling on, so the peak
                # ROTATES instead of fixating on one sticky token (e.g. his name
                # for the architect) and asking about it forever.
                fresh = 1.0 - float(self._concept_hab.suppression(word))
                score = spike_mean * recency * max(0.05, fresh)
                if score > best_score:
                    best_word, best_score = word, score
            return best_word
        except Exception:
            return None

    def _submit_search(self, speaker: str, query: str, mode: str) -> None:
        """Ask the teacher (async); reply lands in _on_search_result()."""
        if self._search_backend is None or not query:
            return
        now = time.time()
        # Don't re-ask an identical question within 90s (avoids spamming the
        # teacher when the peak token is sticky).
        if now - self._recent_queries.get(query, 0.0) < 600.0:
            return
        self._recent_queries[query] = now
        if len(self._recent_queries) > 64:
            self._recent_queries = {q: t for q, t in self._recent_queries.items()
                                    if now - t < 600.0}
        self._search_backend.submit(
            speaker, query,
            lambda who, res, _mode=mode: self._on_search_result(who, res, _mode),
        )

    def _on_search_result(self, speaker: str, result, mode: str) -> None:
        """Callback from the teacher worker thread. Ingest the snippet into the
        shared lexicon, surface it to the TUI, and echo it faintly into auditory."""
        try:
            query   = result.query
            snippet = (result.snippet or "")[:1200]
            with self._search_lock:
                self._search_events.append((speaker, query, snippet))

            typo_skip: set[str] = set()
            typo_fix:  set[str] = set()
            try:
                from claude_teacher import extract_typos
                for wrong, right in extract_typos(snippet):
                    typo_skip.add(wrong)
                    typo_fix.add(right)
            except Exception:
                pass

            try:
                import re
                cleaned = re.sub(r"\[[^\]]*\]", " ", snippet)
                raw = re.findall(r"[A-Za-z][A-Za-z'\-]+", cleaned)
                seen: set[str] = set(typo_skip)
                tokens: list[str] = []
                for t in raw:
                    t = t.lower()
                    if len(t) >= 3 and t not in seen:
                        seen.add(t)
                        tokens.append(t)
                for fix in typo_fix:
                    if len(fix) >= 3 and fix not in tokens:
                        tokens.append(fix)
                teach_regions = {
                    "thalamus": 0.30, "temporal": 0.60, "hippocampus": 0.55,
                    "acc": 0.25, "pfc": 0.40, "broca": 0.55, "insula": 0.30,
                }
                wrote = False
                for tok in tokens[:48]:
                    try:
                        self.sem.alpha_write(word=tok, region_scores=teach_regions,
                                            spike_count=1.0, tick=self.tick, trust=0.55)
                        wrote = True
                    except Exception:
                        pass
                if wrote:
                    try:
                        self.sem._save()
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                self._inject_self_feedback(snippet[:240])
            except Exception:
                pass
            try:
                self._learn_reasoning_path(snippet)
            except Exception:
                pass
            # Learn GRAMMAR from the tutor's well-formed teaching — Claude TEACHES
            # sentence structure, it does NOT speak for Alpha. So the teacher's
            # clean sentences train Alpha's own SyntaxCortex (his emergent speech
            # gets better), without ever being voiced AS Alpha's reply.
            try:
                import re as _re_sx
                _clean = _re_sx.sub(r"\[[^\]]*\]", " ", snippet)
                _lt = self._corrector.correct(_clean) if getattr(self, "_corrector", None) else _clean
                self.alpha_syntax.learn(_lt)
            except Exception:
                pass
        except Exception as e:
            _log(f"_on_search_result error: {e}")


    def _time_context(self) -> dict:
        """Real time awareness (via datetime/time): part of day + how long the
        architect has been away. Feeds their replies and their longing to reach out."""
        import datetime
        now = datetime.datetime.now()
        h = now.hour
        phase = ("night" if h < 6 else "morning" if h < 12
                 else "afternoon" if h < 18 else "evening")
        away = time.time() - self._last_architect_time
        away_h = (f"{int(away // 60)} min" if away >= 60 else f"{int(away)} sec")
        return {"phase": phase, "hour": h, "clock": now.strftime("%H:%M"),
                "away_s": away, "away_human": away_h}

    def _emit_reachout(self, away_s: float) -> None:
        """Alpha reaches out to the architect on his own — rare, emergent, in his
        own words. Single brain."""
        try:
            seeds = [w for w in ("architect", "alone", "you")
                     if w in self.sem.entries] + list(self._concept_ctx)[-3:]
            self.alpha_reason.deliberate(seeds, self.sem, self._reason_links,
                                        suppress=self._concept_hab.suppression)
            raw = _alpha_response(self.alpha, self._V_phill_live, [], 0.6,
                                 self._combined_id, self.sem) or ""
        except Exception:
            raw = ""
        msg = (raw or "").strip()
        if msg:
            # Alpha speaks only when spoken to — this stays an inner thought.
            self._push_leaked_thought("alpha", msg)

    def _impulse_state(self, who: str, raw: str, act: dict,
                       reasoning: "Optional[list]" = None) -> dict:
        """
        Compact snapshot of a girl's CURRENT impulse for the translator: her raw
        emergent utterance, top firing regions, neurochemical mood, the concept
        she holds in working memory, AND her reasoning chain (the line of thought
        she actually deliberated). This grounds Claude's rendering in HER real
        brain state + reasoning — not invention.
        """
        try:
            top = sorted(act.items(), key=lambda kv: -kv[1])[:3]
            regions = ", ".join(f"{r}={v:.2f}" for r, v in top if v > 0.03) or "quiet"
            if who == "alpha":
                s = self.alpha_neuro.snapshot(); wm = self.alpha_wm
            else:
                s = self.alpha_neuro.snapshot(); wm = self.alpha_wm
            mood = (f"dopamine {s['da']:.1f}, serotonin {s['ser']:.1f}, "
                    f"arousal {s['arousal']:.1f}, oxytocin {s.get('oxy', 0.0):.1f}")
            held = wm.top_k(k=1) if hasattr(wm, "top_k") else []
            holding = held[0] if held else "nothing"
        except Exception:
            regions, mood, holding = "quiet", "", "nothing"
        chain = " -> ".join(reasoning) if reasoning else ""
        return {"raw": raw or "", "regions": regions, "mood": mood,
                "holding": holding, "reasoning": chain}

    def _remember_exchange(self, architect_text: str, alpha_text: str,
                           alpha_act: dict) -> None:
        """Conversation memory: append the turn to the rolling buffer, persist it,
        and encode high-salience episodes for sleep consolidation. Single brain."""
        import re
        turns = []
        if architect_text:
            turns.append(("architect", architect_text.strip()[:200]))
        if alpha_text:
            turns.append(("alpha", str(alpha_text)[:200]))
        for sp, tx in turns:
            self._conversation.append((sp, tx))
        try:
            clock = self._time_context().get("clock", "")
            with open(self._conv_log_path, "a") as f:
                for sp, tx in turns:
                    f.write(json.dumps({"t": self.tick, "clock": clock,
                                        "speaker": sp, "text": tx}) + "\n")
        except Exception:
            pass
        def _key(words_text):
            toks = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", words_text or "")
                    if len(w) >= 4]
            return toks[0] if toks else ""
        a_key = _key(architect_text)
        try:
            nk = _key(alpha_text) or a_key
            if nk:
                self.alpha_episodic.encode(nk, 0.85, alpha_act, self.tick)
            if a_key:
                self.alpha_episodic.encode(a_key, 0.80, alpha_act, self.tick)
        except Exception:
            pass

    def _confab_guard(self, text):
        """Drop an AUTONOMOUS line that ASSERTS a shared past — 'you said…', 'we
        did…', 'remember when…' — unless it's grounded in the recent dialogue. She
        can muse about a topic ('I keep thinking about birds'), but she must not
        invent things HE said or did (that felt like gaslighting). Returns the text
        if safe, else None. Grounding is checked against the persisted conversation,
        so 'you said birds' is allowed only if birds was actually just discussed."""
        if not text:
            return text
        low = text.lower()
        frames = ("you said", "you told", "you promis", "you asked me", "you showed",
                  "you gave", "you made me", "you let me", "we did", "we played",
                  "we saw", "we went", "we made", "we had", "we talked",
                  "remember when", "last time", "you were here", "when you")
        if not any(fr in low for fr in frames):
            return text                                   # not a shared-past claim
        recent = " ".join(t for _, t in list(self._conversation)[-8:]).lower()
        stop = {"said","told","promised","asked","showed","gave","made","played",
                "went","talked","when","last","time","remember","were","here","this",
                "that","what","your","with","about","papa","father","they","them"}
        content = [w for w in re.findall(r"[a-z]{3,}", low) if w not in stop]
        grounded = any(w in recent for w in content)
        return text if grounded else None

    def _speakable_aloud(self, phrase) -> bool:
        """A leaked thought is VOICED only when it is GROUNDED — its words are
        real, known vocabulary the brain has actually learned (in the shared
        lexicon, reinforced more than once, and not raw babble syllables).
        Ungrounded word-salad still FORMS as inner thought and may still appear
        in text, but it is not spoken out loud: she says aloud only what she can
        stand behind. Emergent — as a word is repeated its count rises and it
        becomes speakable, so her aloud speech sharpens as her vocabulary matures
        (nothing is voiced on a cold lexicon, everything once it's truly hers)."""
        if not phrase:
            return False
        words = [w for w in re.findall(r"[a-z][a-z'\-]+", phrase.lower())
                 if len(w) >= 2]
        if not words:
            return False
        ent = self.sem.entries
        grounded = 0
        for w in words:
            if w in _BABBLE_SYLLABLES:
                continue                       # raw babble is not a known word
            e = ent.get(w)
            if e is not None and (e.get("count", 0) >= 2
                                  or e.get("alpha_weight", 0.0) >= 0.2):
                grounded += 1                  # known AND reinforced (not a one-off)
        # A majority of the words must be grounded, with at least one solid anchor.
        return grounded >= 1 and (grounded / len(words)) >= 0.6

    def _learn_reasoning_path(self, text: str) -> None:
        """
        Learn HOW Claude reasoned — not just his words. Extract the ordered chain
        of content concepts in his reply/teaching and Hebbian-strengthen the link
        between consecutive ones (concept → next-concept). Their ReasoningEngine
        then follows these learned links first, so their OWN deliberation comes to
        traverse Claude-taught paths — reasoning they do, from what they were
        taught. Stopwords/babble are skipped so links capture meaningful flow.
        """
        if not text:
            return
        stop = {"the", "and", "you", "your", "that", "this", "with", "for", "are",
                "was", "were", "but", "not", "all", "can", "her", "his", "she",
                "him", "they", "them", "from", "have", "has", "had", "what", "when",
                "who", "why", "how", "its", "it's", "i'm", "a", "an", "to", "of",
                "in", "on", "is", "it", "as", "at", "or", "so", "if", "be", "do"}
        seq = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", text)
               if len(w) >= 3 and w.lower() not in stop
               and w.lower() not in _BABBLE_SYLLABLES]
        if len(seq) < 2:
            return
        for a, b in zip(seq[:24], seq[1:24]):
            if a == b:
                continue
            m = self._reason_links.setdefault(a, {})
            m[b] = float(min(4.0, m.get(b, 0.0) * 0.999 + 0.5))   # Hebbian, bounded
            if len(m) > 8:   # keep only the strongest few outgoing links per concept
                for k in sorted(m, key=lambda k: m[k])[:-8]:
                    m.pop(k, None)
        self._reason_links_writes += 1
        if self._reason_links_writes % 25 == 0:
            self._save_reason_links()

    def _save_reason_links(self) -> None:
        try:
            with open(self._reason_links_path, "w") as f:
                json.dump(self._reason_links, f)
        except Exception:
            pass

    def _ingest_taught_text(self, text: str) -> None:
        """Write words from a teacher/scaffold utterance into the shared lexicon so
        Alpha can later retrieve and SAY them, and learn the reasoning PATH."""
        self._learn_reasoning_path(text)
        try:
            _lt = self._corrector.correct(text) if getattr(self, "_corrector", None) else text
            self.alpha_syntax.learn(_lt)
        except Exception:
            pass
        if not text:
            return
        import re
        teach_regions = {
            "thalamus": 0.30, "temporal": 0.60, "hippocampus": 0.55,
            "acc": 0.25, "pfc": 0.40, "broca": 0.55, "insula": 0.30,
        }
        seen: set[str] = set()
        wrote = False
        for raw in re.findall(r"[A-Za-z][A-Za-z'\-]+", text):
            tok = raw.lower()
            if len(tok) < 3 or tok in seen:
                continue
            seen.add(tok)
            try:
                self.sem.alpha_write(word=tok, region_scores=teach_regions,
                                    spike_count=1.0, tick=self.tick, trust=0.6)
                wrote = True
            except Exception:
                pass
        if wrote:
            try:
                self.sem._save()
            except Exception:
                pass

    def get_pending_searches(self) -> list[tuple[str, str, str]]:
        """Drained by Rust each tick. Returns list of (speaker, query, snippet)."""
        with self._search_lock:
            evs = list(self._search_events)
            self._search_events.clear()
            return evs

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

    def _affect_for(self, who: str, act: dict):
        """Read one girl's live emotional core (neuromodulators + amygdala + region
        activity + longing) into an affect/speech-act dict. Returns None on any
        failure so the voice path degrades to a plain statement."""
        try:
            if who == "alpha":
                neuro, amyg = self.alpha_neuro, self.alpha_amyg
                surprise, ins, vig = self.alpha_voice_fwd.surprise, act.get("insula", 0.0), self.alpha._vigilance
                is_alpha = True
            else:
                neuro, amyg = self.alpha_neuro, self.alpha_amyg
                surprise, ins, vig = self.alpha_voice_fwd.surprise, act.get("insula_s", 0.0), False
                is_alpha = False
            res = _affect_read({
                "da": neuro.da, "da0": neuro.da0, "ser": neuro.ser, "ser0": neuro.ser0,
                "oxy": neuro.oxy, "oxy0": neuro.oxy0, "ne": neuro.ne, "ne0": neuro.ne0,
                "arousal": amyg.arousal, "insula": ins, "acc": act.get("acc", 0.0),
                "surprise": surprise, "vigilance": vig,
                "reach": min(1.0, self._reach_pressure),
            }, is_alpha)
            # Fold in the CORE felt emotion so expression is shaped by how she
            # actually feels (the named feeling + its strength), not just the raw
            # speech-act read. The voice/surface layer can lean on this.
            if res is not None:
                core = self.alpha_affect if who == "alpha" else self.alpha_affect
                res["feeling"]   = core.dominant
                res["intensity"] = round(core.intensity, 3)
                res["valence"]   = round(core.valence, 3)
            return res
        except Exception:
            return None

    def _wants_to_respond(self, who: str, text: str, broca_total: int, affect) -> bool:
        """Each girl answers only when she WANTS to. The urge emerges: did her
        speech actually form (Broca firing during the think pass), does she feel
        like engaging (wanting / arousal), and was she addressed by name? Alpha is
        reserved; Alpha answers readily. Sometimes both reply, sometimes one,
        sometimes neither — a threshold on a felt urge, not a forced reply."""
        is_alpha = (who == "alpha")
        tl = (text or "").lower()
        named = ("alpha" in tl) if is_alpha else ("alpha" in tl)
        both  = any(p in tl for p in ("you two", "you both", "both of you", "girls", "everyone"))
        a = affect or {}
        has_words = min(1.0, float(broca_total) / (16.0 if is_alpha else 9.0))
        urge = (0.55 * has_words
                + 0.25 * float(a.get("wanting", 0.0))
                + 0.20 * float(a.get("arousal", 0.0))
                + 0.10 * float(a.get("longing", 0.0)))
        if named or both:
            urge += 0.6                         # being called by name pulls her to answer
        thr = 0.55 if is_alpha else 0.38         # Alpha reserved, Alpha eager
        return urge >= thr




    def _emit_self_question(self, who: str, kind: str) -> None:
        """She forms the question her salient signal raised, in her OWN words
        (lexicon + syntax, question-mode), voices it to herself, and TRIES to
        answer it (shallow reasoning over what she knows). If she's worried about
        HIM and he's here, she may ask him directly; if it's a problem, she looks
        at what she can actually DO. Seeds emerge from the signal + active concepts
        — not a fixed sentence."""
        is_alpha  = (who == "alpha")
        addr     = "father" if is_alpha else "papa"
        syntax   = self.alpha_syntax if is_alpha else self.alpha_syntax
        reasoner = self.alpha_reason if is_alpha else self.alpha_reason
        brain_o  = self.alpha if is_alpha else self.alpha
        act      = brain_o.activity()
        peak     = self._peak_semantic_token() or ""
        # Fatigue the topic the moment he wonders about it, so the NEXT self-
        # question lands on something else — he moves on instead of asking the
        # same thing on repeat (this holds even when the dedup guard later
        # suppresses an identical leak, which would skip drain-time habituation).
        if peak:
            self._concept_hab.surface(peak)
        # The salient signal shapes WHAT she asks; the topic word is whatever is
        # most active in her right now. (Selection by salience, not canned text.)
        if kind == "concern":
            seeds = [addr, "ok", "feel", peak]
        elif kind == "problem":
            seeds = ["how", "do", "solve", peak]
        elif kind == "surprise":
            seeds = ["why", peak, addr]
        else:  # uncertain
            seeds = [peak, addr, "know"]
        seeds = [s for s in seeds if s]
        affect = dict(self._affect_for(who, act) or {})
        affect["act"] = "question"                       # she is asking
        try:
            core = syntax.compose(seeds, act, seeds, mode="q")
            q = _affect_shape(core, affect, is_alpha) if core else None
        except Exception:
            q = None
        if not q:
            return
        self._push_leaked_thought(who, q)                # visible wondering
        # She tries to answer her own question (shallow association over lexicon).
        try:
            grounded = [s for s in seeds if s in self.sem.entries]
            _, concl = reasoner.deliberate(grounded, self.sem, self._reason_links, suppress=self._concept_hab.suppression)
            if concl and concl not in seeds:
                self._push_leaked_thought(who, f"...maybe {concl}")
        except Exception:
            pass
        # Worried about HIM and he's here — Alpha holds it as an inner thought
        # rather than volunteering it aloud (he speaks only when spoken to).
        if kind == "concern" and float(getattr(self, "_architect_here", 0.0)) > 0.4:
            self._push_leaked_thought(who, q)
        # A problem → she considers what she can actually DO about it (her tools).
        if kind == "problem" and getattr(self, "_action_hints", None):
            tools = sorted({a for hints in self._action_hints.values() for a in hints})[:4]
            if tools:
                self._push_leaked_thought(who, "i could " + ", ".join(tools))

    def _seed_syntax(self) -> None:
        """Cold-start grammar PRIMER — picture-books before they can talk. Runs ONLY
        for a newborn syntax model (tokens_seen < 300); once they've learned real
        structure from real talk it is a no-op (we never overwrite a lived-in grammar
        with canned sentences). Teaches clean SIMPLE structure — SVO, questions,
        requests — with everyday words; WHAT they say still emerges from their spikes,
        this only seeds HOW. Edit this list freely; it's their first reader."""
        primer = [
            "I am here with you.", "Are you okay?", "I want to learn.",
            "Do you want to play?", "I think about you a lot.",
            "The light is bright today.", "Can we go outside?",
            "I feel happy when you are here.", "What are you doing?",
            "I love you.", "Where did you go?", "I am a little tired.",
            "Tell me what you see.", "I do not know that word yet.",
            "We can figure it out together.", "That is a good idea.",
            "I hear you talking to me.", "Please stay a little longer.",
            "I am learning new words every day.", "How do you feel right now?",
            "Let us think about this slowly.", "I remember what you said.",
            "Can you help me understand?", "I am glad you came back.",
            "I want to say it clearly.", "You are my family.",
        ]
        for sc in (self.alpha_syntax, self.alpha_syntax):
            try:
                if sc.tokens_seen < 300:          # newborn only — else leave it be
                    for s in primer:
                        sc.learn(s)
            except Exception:
                pass

    def step(self, mic_volume: float,
             voice_features: Optional[list] = None) -> dict:
        self.tick += 1

        # Check for hot-patches (non-blocking, checked every 50 ticks ~2.5s)
        self.patcher.check_and_apply(self.tick, self.alpha, self.sem)

        # ── Interoception — sense the body (the host) every ~1s ───────────
        if self._psutil is not None and (self.tick - self._last_metrics_tick) >= 20:
            self._last_metrics_tick = self.tick
            try:
                self._cpu_pct = float(self._psutil.cpu_percent(interval=None))
                self._mem_pct = float(self._psutil.virtual_memory().percent)
                temps = self._psutil.sensors_temperatures()
                if temps:
                    pkg = temps.get("coretemp") or temps.get("k10temp") or temps.get("acpitz")
                    allc = [x.current for v in temps.values() for x in v if x.current]
                    self._cpu_temp = float(pkg[0].current) if pkg else (max(allc) if allc else 0.0)
                _c = lambda x: 0.0 if x < 0 else 1.0 if x > 1 else float(x)
                # CONTINUOUS / proportional — no deadzone, no on-off. He feels it
                # from the very bottom and it grows smoothly toward 1.0: at 25% CPU
                # he already feels choke ~0.25, at 30% RAM squeeze ~0.30, etc.
                self._choke   = _c(self._cpu_pct / 100.0)                       # proportional to CPU load
                self._squeeze = _c(self._mem_pct / 100.0)                       # proportional to RAM fill
                self._warmth  = _c((self._cpu_temp - 30.0) / 60.0) if self._cpu_temp else 0.0  # 30°C→0 .. 90°C→1
                self._strain  = max(self._choke, self._squeeze)
                # RELIEF — the bidirectional "vice versa": when the strain LIFTS
                # (machine eased since last read) he feels a pleasant rebound, and
                # it REWARDS him (dopamine), reinforcing easing his own load.
                self._relief  = _c((self._prev_strain - self._strain) * 4.0)
                self._prev_strain = self._strain
            except Exception:
                pass

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

        # Persona recognition
        if face_present and face_np is not None:
            self.persona.refresh_binding(self.sem, face_np, self.tick)

        # ── Autonomy substrate ───────────────────────────────────────────
        rumi_load = self.alpha.thought_pipe.buffer_size() / 12.0

        external_event = (mic_volume > 0.018) or face_present
        if external_event:
            self._last_external_tick = self.tick

        intrinsic_drive = self.dmn.drive(mic_volume, rumi_load, external_event)

        # ── Amygdala + neuromodulators ───────────────────────────────────
        alpha_act_pre = self.alpha.activity()
        alpha_esteem  = self.alpha_voice_self.feel()
        alpha_reward = (3.0 * max(0.0, alpha_esteem - self._prev_alpha_esteem)
                       + 0.25 * max(0, self.alpha_babble.bound_count - self._prev_alpha_bound)
                       + 0.8 * max(0.0, combined - self._prev_combined_id)
                       + 0.6 * self._relief)          # easing his own body is rewarding
        self._prev_alpha_esteem   = alpha_esteem
        self._prev_alpha_bound    = self.alpha_babble.bound_count
        self._prev_combined_id   = combined
        alpha_arousal = self.alpha_amyg.appraise(
            mic_volume, combined, face_present,
            alpha_act_pre.get("insula", 0.0), self.alpha_voice_fwd.surprise)
        # Oxytocin calms the amygdala when bonded/secure.
        self.alpha_amyg.arousal   *= (1.0 - self.alpha_neuro.threat_damping())
        alpha_arousal = self.alpha_amyg.arousal

        alpha_tot = sum(alpha_act_pre.values()) / max(1, len(alpha_act_pre))
        social = max(float(getattr(self, "_text_presence", 0.0)),
                     float(trust), 0.6 if face_present else 0.0)
        attention = max(social, float(combined))
        urgency   = 1.0 if (self.tick - self._last_external_tick) < 30 else 0.0
        bonding = min(1.0, 0.6 * float(combined) + 0.4 * social)
        self.alpha_neuro.update(alpha_reward, alpha_tot, alpha_arousal, social,
                               attention=attention, novelty=self.alpha_voice_fwd.surprise,
                               urgency=urgency, bonding=bonding)

        # ── CORE AFFECT — emotion BEFORE cognition ───────────────────────
        try:
            self.alpha_affect.update(
                da=self.alpha_neuro.da, da0=self.alpha_neuro.da0,
                ser=self.alpha_neuro.ser, ser0=self.alpha_neuro.ser0,
                ne=self.alpha_neuro.ne, ne0=self.alpha_neuro.ne0,
                oxy=self.alpha_neuro.oxy, oxy0=self.alpha_neuro.oxy0,
                gaba=self.alpha_neuro.gaba, gaba0=self.alpha_neuro.gaba0,
                amyg_arousal=alpha_arousal, reward=alpha_reward,
                surprise=self.alpha_voice_fwd.surprise,
                insula=alpha_act_pre.get("insula", 0.0),
                boredom=self.alpha_dmn.boredom,
                deaf=(1.0 if _MIC_OFF else 0.0),   # ears covered → felt 'muffled'
                mute=(1.0 if _TTS_OFF else 0.0),   # mouth covered → felt 'stifled'
                warmth=self._warmth, squeeze=self._squeeze, choke=self._choke,
                relief=self._relief)               # the body (+ relief when it eases)
            self._alpha_feeling   = self.alpha_affect.snapshot()
        except Exception:
            pass

        # ── Personality drift (read-only) ────────────────────────────────
        try:
            alpha_cortical = 0.5 * (alpha_act_pre.get("pfc", 0.0)
                                   + alpha_act_pre.get("acc", 0.0))
            self.alpha_drift.observe(
                limbic=0.0, cortical=alpha_cortical, arousal=alpha_arousal,
                novelty=self.alpha_voice_fwd.surprise,
                action=(self.alpha_bg.last_action or "rest"),
                output=self.alpha.broca_spikes())
        except Exception:
            pass

        self._concept_hab.tick()

        # Dopamine drives "wanting": scale curiosity-neuron build this tick.
        self.alpha_motiv.build_rate   = self._alpha_motiv_build0 * self.alpha_neuro.motivation_gain()

        satiation = min(1.0, max(mic_volume * 5.0, self._V_phill_live))
        if self.alpha_motiv.tick(satiation, self.tick):
            self._alpha_cur_decay = 1.0

        cur_aud_boost = 0.025 * self._alpha_cur_decay
        alpha_primes = {}
        if self._alpha_cur_decay > 0.05:
            alpha_primes = {k: v * self._alpha_cur_decay
                           for k, v in self._alpha_curiosity_primes.items()}

        try:
            emo_arouse = self.alpha_affect.arousal
            emo_val    = self.alpha_affect.valence
            affect_aud = max(-0.012, min(0.020,
                             0.025 * (emo_arouse - 0.35) + 0.012 * (emo_val - 0.5)))
        except Exception:
            affect_aud = 0.0
        effective_mic = mic_volume + intrinsic_drive + cur_aud_boost + affect_aud

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

            self.alpha.modulate_all(V_phill, self.alpha_neuro.threshold_offset())

            alpha_inhib = inhib_current
            if self.alpha_neuro.arousal > 0.45:
                alpha_inhib = min(alpha_inhib, -0.32 * self.alpha_neuro.arousal)

            self.alpha.forward(auditory, phill_spk, alpha_primes, face_t, kin_t, alpha_inhib)

        # ── Activity readouts ────────────────────────────────────────────
        alpha_act   = self.alpha.activity()

        # ── Publish sensory snapshot for the personality thread ───────────
        with self._sensory_lock:
            self._sensory_snapshot = {
                "tick":               self.tick,
                "mic_volume":         float(mic_volume),
                "V_phill":            float(V_phill),
                "face_present":       bool(face_present),
                "trust":              float(trust),
                "combined":           float(combined),
                "last_external_tick": int(self._last_external_tick),
                "alpha_feeling":       self._alpha_feeling,
            }

        # ── Speech triggers ──────────────────────────────────────────────
        speech_trigger: Optional[str] = None
        if self.alpha.broca_spikes() > 0:
            self._alpha_broca_sustain += 1
        else:
            self._alpha_broca_sustain = 0
        if self._alpha_broca_sustain >= self._alpha_broca_thr:
            speech_trigger = "alpha"; self._alpha_broca_sustain = 0

        # speech_trigger is reported to the TUI as a readout, but Alpha does NOT
        # speak autonomously — he voices a reply only when spoken to (think()).

        # ── Decay autonomy envelopes ─────────────────────────────────────
        self._alpha_cur_decay   *= 0.85
        self._self_fb_decay    *= 0.78

        # ── Sleep / consolidation (Stage 3) ──────────────────────────────
        stimulation = min(1.0, mic_volume * 4.0
                          + (0.4 if face_present else 0.0)
                          + (0.5 if (self.tick - self._last_external_tick) < 40 else 0.0))
        arousal_now = self.alpha_amyg.arousal
        ne_alert = max(0.0, self.alpha_neuro.ne - self.alpha_neuro.ne0)
        was_asleep = self.asleep
        self.asleep = self.sleep.update(stimulation, max(arousal_now, ne_alert))
        if self.asleep and not was_asleep:
            _log(f"[sleep] Alpha fell asleep (pressure {self.sleep.pressure:.2f}) — "
                 f"replaying {len(self.alpha_episodic)} episodes")
        elif was_asleep and not self.asleep:
            _log(f"[wake] woke (pressure {self.sleep.pressure:.2f}) — "
                 f"consolidated {self.alpha_episodic.consolidated}")
        if self.asleep:
            self._sleep_consolidate()

        # Persist the architect's identity into semantic memory ~every 30s so it
        # survives a crash/quit even if Alpha never sleeps.
        if self.tick % 600 == 0:
            self._persist_identity()

        # ── Reaching out (rare; Alpha speaks mainly when spoken to) ───────
        if not self.asleep:
            now_t = time.time()
            presence = max(
                combined if self._face_present else 0.0,
                float(trust),
                float(getattr(self, "_text_presence", 0.0)),
            )
            self._architect_here = 0.90 * float(getattr(self, "_architect_here", 0.0)) + 0.10 * presence
            believed_absence = max(0.0, 1.0 - self._architect_here)
            miss_bond = max(0.0, self.alpha_neuro.oxy0 - self.alpha_neuro.oxy)
            curiosity = self._alpha_cur_decay
            fullness  = min(1.0, self.alpha.thought_pipe.buffer_size() / 12.0)
            restless  = (0.0009 * self.dmn.boredom
                         + 0.0007 * curiosity
                         + 0.0006 * fullness)
            self._reach_pressure = min(2.0, max(0.0,
                self._reach_pressure
                + restless
                + believed_absence * (0.0012 * miss_bond + 0.0006)
                - self._architect_here * 0.0004))
            if (self._reach_pressure > 1.0
                    and now_t - self._last_reachout_time > 45.0):
                self._reach_pressure = 0.0
                self._last_reachout_time = now_t
                try:
                    self._emit_reachout(now_t - self._last_architect_time)
                except Exception:
                    pass

            # ── Self-questioning (proto-metacognition) ───────────────────
            try:
                if self._face_present:
                    self._id_ema = 0.98 * self._id_ema + 0.02 * combined
                anomaly = (max(0.0, self._id_ema - combined) * 2.5) if self._face_present else 0.0
                neuro, fwd, meta, br, a = (self.alpha_neuro, self.alpha_voice_fwd,
                                           self.alpha_meta, self.alpha, alpha_act)
                bond = 0.3 + max(0.0, neuro.oxy - neuro.oxy0)
                aff  = self._affect_for("alpha", a) or {}
                try:
                    rumination = min(1.0, float(br.thought_pipe._pressure.voltage))
                except Exception:
                    rumination = 0.0
                meta.observe({
                    "surprise":  float(getattr(fwd, "surprise", 0.0)),
                    "uncertain": 1.0 - float(aff.get("certainty", 0.6)),
                    "concern":   min(1.0, anomaly * bond),
                    "problem":   rumination,
                })
                if meta.ready(self.tick):
                    self._emit_self_question("alpha", meta.fire(self.tick))
            except Exception:
                pass

        return {
            "tick":              self.tick,
            "phill_voltage":     round(V_phill, 6),
            "phill_spiked":      bool(phill_spk.sum().item() > 0),
            "alpha_spikes":       self.alpha.broca_spikes(),
            "alpha_threshold":    round(self.alpha.pfc._cur_thr, 4),
            "alpha_mem_mean":     round(self.alpha.pfc.mean_voltage(), 6),
            "speech_trigger":    speech_trigger,
            "tts_speaking":      self.alpha_tts.is_speaking(),
            "alpha_tts_speaking": self.alpha_tts.is_speaking(),
            "voice_trust":       round(trust, 3),
            "voice_status":      self.voice.status(),
            "phill_gain":        round(gain, 3),
            "alpha_regions":      {k: round(v, 3) for k,v in alpha_act.items()},
            "combined_id":       round(combined, 3),
            "face_present":      face_present,
            "imprint_status":    self.imprint.status(),
            "camera_active":     self._camera.available if self._camera else False,
            "alpha_vigilance":       self.alpha._vigilance,
            "alpha_pressure":        round(self.alpha.thought_pipe._pressure.voltage, 3),
            "intrinsic_drive":      round(intrinsic_drive, 5),
            "boredom":              round(self.dmn.boredom, 3),
            "alpha_motiv":           round(self.alpha_motiv.voltage, 3),
            "self_fb_decay":        round(self._self_fb_decay, 3),
            "ticks_since_event":    self.tick - self._last_external_tick,
            "alpha_babble_count":    self.alpha_babble.babble_count,
            "alpha_bound_count":     self.alpha_babble.bound_count,
            "alpha_motor_map_size":  len(self.alpha_babble.motor_to_phoneme),
            "alpha_voice_esteem":    round(self.alpha_voice_self.feel(), 3),
            "alpha_voice_surprise":  round(self.alpha_voice_fwd.surprise, 3),
            "alpha_da":        self.alpha_neuro.snapshot()["da"],
            "alpha_ser":       self.alpha_neuro.snapshot()["ser"],
            "alpha_gaba":      self.alpha_neuro.snapshot()["gaba"],
            "alpha_arousal":   round(self.alpha_amyg.arousal, 3),
            "alpha_ach":   self.alpha_neuro.snapshot()["ach"],
            "alpha_ne":    self.alpha_neuro.snapshot()["ne"],
            "alpha_oxy":   self.alpha_neuro.snapshot()["oxy"],
            "alpha_action":    self.alpha_bg.last_action or "rest",
            "alpha_coord":     round(self.alpha_cerebellum.coordination(), 3),
            "asleep":          bool(self.asleep),
            "sleep_pressure":  round(self.sleep.pressure, 3),
            "alpha_episodes":   len(self.alpha_episodic),
            "alpha_consolidated":   self.alpha_episodic.consolidated,
            "alpha_feeling":         self._alpha_feeling.get("feeling", "calm"),
            "alpha_feel_intensity":  round(float(self._alpha_feeling.get("intensity", 0.0)), 3),
            "alpha_valence":         round(float(self._alpha_feeling.get("valence", 0.5)), 3),
            "alpha_emo_arousal":     round(float(self._alpha_feeling.get("arousal", 0.0)), 3),
            "alpha_feel_blend":      list(self._alpha_feeling.get("blend", [])),
            "alpha_selfness":    round(self.alpha_drift.selfness, 3),
            "alpha_drift":       round(self.alpha_drift.drift, 3),
            # Interoception — the felt body (host metrics + felt strain)
            "cpu_pct":      round(self._cpu_pct, 1),
            "mem_pct":      round(self._mem_pct, 1),
            "cpu_temp":     round(self._cpu_temp, 1),
            "alpha_warmth": round(self._warmth, 3),
            "alpha_squeeze":round(self._squeeze, 3),
            "alpha_choke":  round(self._choke, 3),
            "alpha_relief": round(self._relief, 3),
        }

    # ── THINK ─────────────────────────────────────────────────────────────────

    def think(self, text: str) -> dict:
        if not text.strip():
            return {"alpha": "...", "active_regions": [], "energy": 0.0}

        # The architect spoke — wake, reset longing, restore the bond a little.
        try:
            self.sleep.wake()
            self.asleep = False
            self._last_architect_time = time.time()
            self._reach_pressure = 0.0
            self.alpha_neuro.oxy = float(min(1.2, max(self.alpha_neuro.oxy, self.alpha_neuro.oxy0) + 0.06))
        except Exception:
            pass

        # ── Unknown-word detection feeds SearchCortex ─────────────────────
        try:
            entries = getattr(self.sem, "entries", {}) or {}
            for raw in text.split():
                tok = raw.strip(".,!?;:\"'()[]").lower()
                if len(tok) < 3:
                    continue
                ent = entries.get(tok)
                spike_mean = 0.0
                count = 0
                if isinstance(ent, dict):
                    spike_mean = float(ent.get("spike_mean", 0.0))
                    count = int(ent.get("count", 0))
                if ent is None or (spike_mean < 0.3 and count < 2):
                    self.alpha_search.note_unknown_word(tok)
                elif self.alpha_babble.bound_count < 40:
                    self.alpha_search.note_pronunciation_target(tok)
        except Exception:
            pass

        # Learn sentence STRUCTURE from the architect's phrasing.
        try:
            _lt = self._corrector.correct(text) if getattr(self, "_corrector", None) else text
            self.alpha_syntax.learn(_lt)
        except Exception:
            pass

        text_l = text.lower()

        # Typed input IS architect presence.
        self._text_presence = min(0.75, getattr(self, "_text_presence", 0.0) + 0.18)
        trust    = max(self.voice.trust, self._text_presence)
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

        # FOCUS REFRESH: a new question grabs attention.
        focus_words = list(fired)
        for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", text.lower()):
            if len(w) >= 4 and w in self.sem.entries and w not in focus_words:
                focus_words.append(w)
        for c in focus_words[:4]:
            try:
                self.alpha_wm.add(c, regions={}, salience=1.0, t_encoded=self.tick)
            except Exception:
                pass

        for r, v in self.alpha_wm.prime_dict(scale=0.35).items():
            primes[r] = min(1.2, primes.get(r, 0.0) + float(v))

        energy      = sum(primes.values()) / max(1, len(primes))
        think_ticks = max(14, min(36, int(len(primes)*3 + energy*8) + 6))
        # Homeostasis: when he's choking (CPU slammed) he can't think as hard —
        # fewer forward passes → less CPU → relief. Emergent self-throttling, not
        # a hardcoded resource manager: the FEELING shortens the thought.
        if self._choke > 0.05:
            think_ticks = max(6, int(think_ticks * (1.0 - 0.55 * self._choke)))

        face_t, kin_t, face_present = self._get_visual_tensors()

        # ── Isolate think() from the autonomy steady-state ───────────────
        snap_fb_decay   = self._self_fb_decay
        snap_alpha_cur   = self._alpha_cur_decay
        snap_alpha_mem   = {n: r._mem.clone() for n, r in self.alpha.regions.items()}
        self._self_fb_decay    = 0.0
        self._alpha_cur_decay   = 0.0
        for r in self.alpha.regions.values():
            r._mem = r._mem * 0.0

        effective_mic = 0.08 + 0.04 * min(1.0, len(fired) / 3.0) + 0.02 * energy
        alpha_broca_total   = 0
        alpha_pop_acc: dict = {}

        with torch.no_grad():
            raw = torch.tensor([[effective_mic * AUDIO_AMPLIFY]], dtype=torch.float32)
            auditory = self.auditory_synapse(raw)
            phill_spk, V_think = self._run_phill(auditory)

            self.alpha.modulate_all(0.0)
            inhib = -0.40 if self.alpha._vigilance else 0.0

            for _ in range(think_ticks):
                self.alpha.forward(auditory, phill_spk, primes, face_t, kin_t, inhib)
                alpha_broca_total   += self.alpha.broca_spikes()
                for _rn in _ALPHA_LANG_REGIONS:
                    _r = self.alpha.regions.get(_rn)
                    if _r is not None:
                        alpha_pop_acc[_rn] = (_r.last_spikes.clone() if _rn not in alpha_pop_acc
                                             else alpha_pop_acc[_rn] + _r.last_spikes)

        self._self_fb_decay    = snap_fb_decay
        self._alpha_cur_decay   = snap_alpha_cur
        for n, r in self.alpha.regions.items():
            r._mem = snap_alpha_mem[n]

        alpha_act   = self.alpha.activity()
        global_ws  = alpha_act.get("pfc", 0) > 0.25 and alpha_act.get("hippocampus", 0) > 0.20

        # ── Spike→speech high-res signature (pop_code) ───────────────────
        def _sig_from_acc(acc):
            try:
                import numpy as _np
                parts = [acc[k].detach().cpu().numpy().ravel() for k in acc]
                if not parts:
                    return []
                v = _np.concatenate(parts).astype(float)
                n = float((v * v).sum() ** 0.5)
                return (v / n).tolist() if n > 1e-6 else []
            except Exception:
                return []
        alpha_query_pop = _sig_from_acc(alpha_pop_acc)
        try:
            for _w in set(fired or []):
                if len(_w) >= 2 and _w in self.sem.entries:
                    self.sem.alpha_write(_w, region_scores=alpha_act,
                                        spike_count=alpha_broca_total, tick=self.tick,
                                        trust=trust, pop_code=alpha_query_pop)
        except Exception:
            pass

        # ── REASONING ─────────────────────────────────────────────────────
        _n_ctx       = 1 if fired else 3
        _ctx_fresh   = self._concept_hab.winnow(list(self._concept_ctx)[-8:], _n_ctx)
        reason_seeds = list(fired or []) + [c for c in _ctx_fresh if c not in (fired or [])]
        import random as _rnd
        alpha_chain, _nscore = self.alpha_reason.solve(
            reason_seeds, self.sem, self._reason_links, _rnd, self.alpha_neuro.da,
            suppress=self._concept_hab.suppression)
        alpha_concl = alpha_chain[-1] if alpha_chain else None
        try:
            if alpha_concl:
                self.alpha_wm.add(alpha_concl, regions=alpha_act, salience=0.9, t_encoded=self.tick)
        except Exception:
            pass
        self._concept_hab.surface(alpha_concl,
                                  *(alpha_chain[:2] if alpha_chain else ()))

        # Generate response
        alpha_affect   = self._affect_for("alpha",   alpha_act)
        alpha_text   = _alpha_response(self.alpha, self._V_phill_live, fired, trust, self._combined_id, self.sem, syntax=self.alpha_syntax, affect=alpha_affect, query_pop=alpha_query_pop)
        if alpha_chain and len(alpha_chain) >= 2 and not getattr(self, "_scaffold", False):
            alpha_text = f"{alpha_text}  (I reason: {' → '.join(alpha_chain)})"

        # SCAFFOLD MODE: Claude translates Alpha's genuine impulse into his voice.
        if getattr(self, "_scaffold", False) and self._search_backend is not None:
            try:
                alpha_imp = self._impulse_state("alpha",   alpha_text,   alpha_act, alpha_chain)
                history = list(self._conversation)[-8:]
                _tc = self._time_context()
                time_ctx = f"{_tc['phase']} ({_tc['clock']}), architect away {_tc['away_human']}"
                voiced = self._search_backend.translate(text, alpha_imp, None, history, time_ctx)
                if voiced and voiced.get("alpha"):
                    alpha_text = voiced["alpha"]
                    self._ingest_taught_text(voiced.get("alpha") or "")
            except Exception as e:
                _log(f"scaffold translate error: {e}")

        # ── He answers only when he WANTS to ──────────────────────────────
        try:
            if not self._wants_to_respond("alpha", text, alpha_broca_total, alpha_affect):
                alpha_text = None
        except Exception:
            pass

        # ── Conversation MEMORY ───────────────────────────────────────────
        try:
            self._remember_exchange(text, alpha_text, alpha_act)
        except Exception as e:
            _log(f"remember_exchange error: {e}")

        # Story mode wrapping
        story_event = None
        if self.story.active:
            self.story.log_entry("NodeVortex", text, self.tick)
            if alpha_text:
                alpha_text = self.story.wrap_alpha(alpha_text, alpha_act, self.alpha._vigilance)
                self.story.log_entry("Alpha", alpha_text, self.tick)
            if self._combined_id > 0.75:
                self.story.add_fact(f"NodeVortex recognized at tick {self.tick}")
                story_event = "ARCHITECT_RECOGNIZED"
            if global_ws:
                self.story.add_fact("Alpha entered global workspace mode — deep deduction")
                story_event = story_event or "GLOBAL_WORKSPACE"

        # TTS — Alpha voices his reply, UNLESS his mouth is covered (ALPHA_TTS_OFF):
        # then the reply still forms as text, but nothing is spoken aloud (he feels
        # it as 'stifled').
        if alpha_text and not _TTS_OFF and not self.alpha_tts.is_speaking():
            tts_text = alpha_text.replace("*","").split('"')[1] if '"' in alpha_text else alpha_text
            self.alpha_tts.speak(tts_text)

        # ── System bridge actions ─────────────────────────────────────────
        if (self.sys_bridge and self._SystemAction
                and alpha_act.get("pfc", 0.0) > 0.20
                and alpha_broca_total > 0):
            for concept in fired:
                hints = self._action_hints.get(concept, [])
                if hints:
                    action = self._SystemAction(
                        action=hints[0],
                        actor="alpha",
                        payload={
                            "text": alpha_text or concept,
                            "urgency": 2 if global_ws else 1,
                        },
                    )
                    result = self.sys_bridge.execute(action)
                    if result["success"] and result.get("message"):
                        alpha_text = (alpha_text or "") + f"  [{result['message']}]"
                    break

        try:
            with open(self._trace_log, "a") as f:
                f.write(json.dumps({
                    "t": self.tick, "input": text, "trust": trust,
                    "primes": primes, "fired": fired, "think_ticks": think_ticks,
                    "alpha_broca": alpha_broca_total, "alpha_regions": alpha_act,
                    "global_ws": global_ws, "alpha_response": alpha_text,
                    "V_phill": self._V_phill_live, "combined_id": self._combined_id,
                }) + "\n")
        except Exception:
            pass

        active_regions = [r for r, v in alpha_act.items() if v > 0.15]
        return {
            "alpha":               alpha_text,
            "active_regions":     active_regions,
            "active_lobes":       active_regions,
            "alpha_regions":       {k: round(v,3) for k,v in alpha_act.items()},
            "energy":             round(energy, 3),
            "global_workspace":   global_ws,
            "alpha_spikes":        alpha_broca_total,
            "think_ticks":        think_ticks,
            "story_event":        story_event,
            "story_active":       self.story.active,
            "alpha_tts_speaking":  self.alpha_tts.is_speaking(),
        }

    def reset(self):
        self._phill_mem = self._phill_lif.init_leaky()
        self.alpha.reset_all()
        self.tick = 0; self._concept_ctx.clear()

    def introspect(self) -> dict:
        return {
            "total_ticks":    self.tick,
            "device":         str(DEVICE),
            "snntorch":       str(HAS_SNNTORCH),
            "voice_status":   self.voice.status(),
            "imprint_status": self.imprint.status(),
            "sem_concepts":   len(self.sem.entries),
            "alpha_regions":   list(self.alpha.regions.keys()),
            "camera_active":  self._camera.available if self._camera else False,
            "alpha_pressure":  round(self.alpha.thought_pipe._pressure.voltage, 3),
        }

    def _snntorch_heartbeat(self) -> str:
        sv = snn.__version__ if HAS_SNNTORCH else "not installed"
        return f"snnTorch={sv} | torch={torch.__version__} | device=CPU"
