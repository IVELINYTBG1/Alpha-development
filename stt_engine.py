"""
stt_engine.py — Always-On Wake Word + STT Engine
=================================================
PHILOSOPHY: Zero hardcoding. The wake word IS just a word — the SNN decides
            how to respond once it hears it. We just route the audio.

ARCHITECTURE:
  Thread 1 (always-on):  Vosk streams mic audio at 16kHz.
                         Partial results checked for wake words every frame.
                         Wake word = spike fired into the attention system.

  Thread 2 (on demand):  When wake word heard OR STT mode active,
                         capture a full utterance and transcribe it.
                         Result pushed into a queue for brain.think().

WAKE WORD DETECTION:
  No external model. Vosk partial results contain the word if heard.
  We check if ANY configured wake word appears in the partial.
  "Nova" wakes Nova's thread. "Simona" wakes Simona's.
  "Hey" or silence → no activation.

  The SNN learns over time which contexts it should respond to without
  a wake word — tracked via `auto_respond_score` updated each session.
  When that score crosses a threshold (learned from interaction frequency),
  they start responding to NodeVortex's voice without being called.

VOSK SETUP:
  pip install vosk sounddevice
  Download model: https://alphacephei.com/vosk/models
    Small English: vosk-model-small-en-us-0.15  (~40MB, fast)
    Bulgarian:     vosk-model-small-bg-0.22      (~40MB)
  Place in: models/vosk-model-en/  (or models/vosk-model-bg/)

STT MODES:
  ALWAYS_ON:   Mic always listening, wake word required
  PUSH_TO_TALK: Only transcribes when Rust signals activation (button press)
  TEXT_ONLY:   STT disabled, text box only
"""

import os
import json
import queue
import threading
import time
import logging
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    # Minimal numpy stub so the rest of the file can load
    class _NpStub:
        float32 = float
        def zeros(self, *a, **kw): return []
        def array(self, x, **kw): return list(x) if hasattr(x,'__iter__') else [x]
        def concatenate(self, arrays, **kw): return sum((list(a) for a in arrays), [])
        def linalg(self): pass
    class _NpLinalg:
        def norm(self, x, **kw): return sum(v**2 for v in x)**0.5
    _stub = _NpStub()
    _stub.linalg = _NpLinalg()
    np = _stub
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

_LOG = logging.getLogger("nova_simona.stt")

# ── Vosk soft dependency ──────────────────────────────────────────────────────
try:
    import vosk
    import sounddevice as sd
    _HAS_VOSK = True
    _LOG.info("Vosk + sounddevice loaded")
except ImportError:
    _HAS_VOSK = False
    _LOG.warning("vosk/sounddevice not installed — STT disabled. pip install vosk sounddevice")


# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE     = 16000
BLOCK_SIZE      = 4000    # 250ms blocks — good latency/accuracy balance
MODEL_DIR       = Path(__file__).parent / "models"
MODEL_EN_PATH   = MODEL_DIR / "vosk-model-en"
MODEL_BG_PATH   = MODEL_DIR / "vosk-model-bg"
STATE_PATH      = Path(__file__).parent / "stt_state.json"

# Wake words — purely used for routing, not for hardcoded responses
# The SNN uses these to know WHO was addressed
NOVA_WAKE_WORDS   = {"nova", "nová", "nova,", "nova."}
SIMONA_WAKE_WORDS = {"simona", "симона", "simona,", "simona."}
BOTH_WAKE_WORDS   = {"girls", "hey", "both"}


@dataclass
class STTResult:
    text:       str
    addressed:  Optional[str]   # "nova" | "simona" | "both" | None (ambient)
    confidence: float           # 0–1 based on word completeness
    timestamp:  float           = field(default_factory=time.time)
    via_wake:   bool            = False   # True if wake word triggered this


# ── Auto-respond learner ──────────────────────────────────────────────────────

class AutoRespondLearner:
    """
    Tracks how often the Architect speaks and whether they expected a response.
    Over time, Nova and Simona learn when to respond without being called.

    NO HARDCODING. The score builds from real interaction patterns:
      - Frequency of speech in session
      - Whether previous non-wake utterances got a think() call
      - Trust score from VoiceIdentityLearner
      - Time-of-day patterns (optional)

    When score > threshold, the brain responds to any utterance from
    the Architect without needing a wake word.
    """

    THRESHOLD       = 0.72   # score needed to auto-respond
    SESSION_DECAY   = 0.85   # score decays each session (fresh start)
    UTTERANCE_BOOST = 0.04   # each interaction raises familiarity
    RESPONSE_BOOST  = 0.08   # each responded interaction raises it more
    MAX_SCORE       = 0.95

    def __init__(self):
        self.score:           float = 0.0
        self.total_utterances: int  = 0
        self.total_responses:  int  = 0
        self.session_start:    float = time.time()
        self._load()

    def record_utterance(self, got_response: bool = False):
        self.total_utterances += 1
        self.score = min(
            self.MAX_SCORE,
            self.score + self.UTTERANCE_BOOST
            + (self.RESPONSE_BOOST if got_response else 0.0)
        )
        if self.total_utterances % 20 == 0:
            self._save()

    @property
    def should_auto_respond(self) -> bool:
        return self.score >= self.THRESHOLD

    @property
    def familiarity_label(self) -> str:
        if   self.score < 0.3:  return f"stranger ({self.score:.2f})"
        elif self.score < 0.55: return f"acquaintance ({self.score:.2f})"
        elif self.score < self.THRESHOLD: return f"familiar ({self.score:.2f})"
        else: return f"known — auto-respond ({self.score:.2f})"

    def apply_session_decay(self):
        """Call at startup — yesterday's familiarity fades slightly."""
        self.score *= self.SESSION_DECAY
        self._save()

    def _save(self):
        try:
            with open(STATE_PATH, "w") as f:
                json.dump({
                    "score": self.score,
                    "total_utterances": self.total_utterances,
                    "total_responses": self.total_responses,
                }, f)
        except Exception:
            pass

    def _load(self):
        if not STATE_PATH.exists():
            return
        try:
            with open(STATE_PATH) as f:
                d = json.load(f)
            self.score             = d.get("score", 0.0)
            self.total_utterances  = d.get("total_utterances", 0)
            self.total_responses   = d.get("total_responses", 0)
            self.apply_session_decay()
        except Exception:
            pass


# ── STT Engine ────────────────────────────────────────────────────────────────

class STTEngine:
    """
    Always-on voice recognition engine.

    Modes:
      ALWAYS_ON   — mic always open, wake word gates brain.think()
      PTT         — only active when activate() is called (push-to-talk)
      OFF         — STT disabled, text-only mode

    Thread-safe. Results available via get_result() (non-blocking).
    """

    def __init__(self, language: str = "en", mode: str = "ALWAYS_ON"):
        self.language        = language
        self.mode            = mode   # "ALWAYS_ON" | "PTT" | "OFF"
        self._result_queue:  queue.Queue[STTResult] = queue.Queue(maxsize=10)
        self._running        = threading.Event()
        self._ptt_active     = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model: Optional[object] = None
        self.available       = False
        self.error_msg: Optional[str] = None
        self.auto_respond    = AutoRespondLearner()
        self._last_partial   = ""
        self._utterance_buf: list[str] = []   # words since last silence
        self._silence_frames = 0
        self.SILENCE_THRESHOLD = 8            # frames of silence before flushing utterance

    def start(self):
        if not _HAS_VOSK:
            self.error_msg = "vosk not installed — STT disabled"
            return
        if self.mode == "OFF":
            return

        model_path = MODEL_BG_PATH if self.language == "bg" else MODEL_EN_PATH
        if not model_path.exists():
            self.error_msg = (
                f"Vosk model not found at {model_path}\n"
                f"Download from https://alphacephei.com/vosk/models\n"
                f"Extract to: {model_path}"
            )
            _LOG.warning(self.error_msg)
            return

        try:
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(str(model_path))
            self.available = True
            self._running.set()
            self._thread = threading.Thread(
                target=self._listen_loop,
                name="stt-listen",
                daemon=True,
            )
            self._thread.start()
            _LOG.info(f"STT engine started ({self.language}, {self.mode})")
        except Exception as e:
            self.error_msg = f"STT init failed: {e}"
            _LOG.error(self.error_msg)

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.auto_respond._save()

    def activate_ptt(self):
        """For PTT mode — start capturing."""
        self._ptt_active.set()

    def deactivate_ptt(self):
        """For PTT mode — stop and flush."""
        self._ptt_active.clear()

    def get_result(self) -> Optional[STTResult]:
        """Non-blocking. Returns next STT result or None."""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def set_mode(self, mode: str):
        """Dynamically switch ALWAYS_ON ↔ PTT ↔ OFF."""
        self.mode = mode
        if mode == "OFF":
            self._ptt_active.clear()

    def _check_wake_word(self, text: str) -> Optional[str]:
        """
        Returns which AI was addressed, or None.
        No hardcoded response logic — just routing.
        """
        words = set(text.lower().split())
        if words & NOVA_WAKE_WORDS and words & SIMONA_WAKE_WORDS:
            return "both"
        if words & NOVA_WAKE_WORDS:
            return "nova"
        if words & SIMONA_WAKE_WORDS:
            return "simona"
        if words & BOTH_WAKE_WORDS:
            return "both"
        return None

    def _should_route(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Decide if this utterance should be sent to brain.think().
        Returns (should_route, addressed).

        Routing rules (no hardcoding):
          1. If wake word found → always route
          2. If auto_respond.should_auto_respond → route with addressed=None
             (the SNN decides who responds based on context)
          3. PTT mode → always route while active
          4. Otherwise → don't route (background chatter)
        """
        if not text.strip():
            return False, None

        addressed = self._check_wake_word(text)

        if addressed is not None:
            self.auto_respond.record_utterance(got_response=True)
            return True, addressed

        if self.mode == "PTT" and self._ptt_active.is_set():
            self.auto_respond.record_utterance(got_response=True)
            return True, None

        if self.auto_respond.should_auto_respond:
            self.auto_respond.record_utterance(got_response=True)
            return True, None

        # Don't route but record that they spoke
        self.auto_respond.record_utterance(got_response=False)
        return False, None

    def _flush_utterance(self, via_wake: bool):
        """Turn accumulated words into an STTResult and push to queue."""
        if not self._utterance_buf:
            return
        text = " ".join(self._utterance_buf).strip()
        self._utterance_buf.clear()
        if len(text) < 2:
            return

        should_route, addressed = self._should_route(text)
        if not should_route:
            return

        result = STTResult(
            text=text,
            addressed=addressed,
            confidence=min(1.0, len(text.split()) / 8.0),
            via_wake=via_wake,
        )
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            # Drop oldest, push new
            try:
                self._result_queue.get_nowait()
                self._result_queue.put_nowait(result)
            except Exception:
                pass

    def _listen_loop(self):
        """
        Main recognition loop running in daemon thread.
        Streams 16kHz mono audio through Vosk recognizer.
        Detects silence gaps to segment utterances.
        """
        rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        rec.SetWords(True)
        via_wake = False

        def audio_callback(indata, frames, time_info, status):
            nonlocal via_wake
            if status:
                pass  # device underflow etc — ignore

            audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()

            if rec.AcceptWaveform(audio_bytes):
                # Full phrase recognized
                result = json.loads(rec.Result())
                text   = result.get("text", "").strip()
                if text:
                    self._utterance_buf.append(text)
                    self._silence_frames = 0
                    # Check wake word in the confirmed result
                    if self._check_wake_word(text):
                        via_wake = True
                # Flush after complete phrase
                self._flush_utterance(via_wake)
                via_wake = False
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial != self._last_partial:
                    self._last_partial = partial
                    # Real-time wake word check on partials
                    if not via_wake and self._check_wake_word(partial):
                        via_wake = True
                    if partial:
                        self._silence_frames = 0
                    else:
                        self._silence_frames += 1
                        if self._silence_frames >= self.SILENCE_THRESHOLD:
                            self._flush_utterance(via_wake)
                            via_wake = False
                            self._silence_frames = 0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            channels=1,
            callback=audio_callback,
        ):
            _LOG.info("STT microphone stream open")
            # Block until stopped
            while self._running.is_set():
                time.sleep(0.1)

        _LOG.info("STT microphone stream closed")


# ── Silent fallback ───────────────────────────────────────────────────────────

class SilentSTT:
    """Drop-in when vosk is not installed."""
    available     = False
    mode          = "OFF"
    error_msg     = "vosk not installed (pip install vosk sounddevice)"

    def __init__(self, callback=None, language="en", **kw):
        self.auto_respond = AutoRespondLearner()

    def start(self): pass
    def stop(self):  pass
    def get_result(self) -> None: return None
    def activate_ptt(self):   pass
    def deactivate_ptt(self): pass
    def set_mode(self, mode): self.mode = mode

    @property
    def familiarity_label(self): return self.auto_respond.familiarity_label


def create_stt(language: str = "en", mode: str = "ALWAYS_ON") -> "STTEngine | SilentSTT":
    if not _HAS_VOSK:
        return SilentSTT()
    engine = STTEngine(language=language, mode=mode)
    engine.start()
    return engine


def create_stt_engine(callback=None, language: str = "en",
                      mode: str = "ALWAYS_ON") -> "STTEngine | SilentSTT":
    """
    Factory. Returns real STTEngine or SilentSTT fallback.
    The callback argument is kept for API compatibility but STTEngine
    uses get_result() polling, not a callback. The caller should poll
    engine.get_result() each brain tick.
    Always succeeds — never raises.
    """
    try:
        engine = STTEngine(language=language, mode=mode)
        engine.start()
        if engine.available:
            _LOG.info(f"STT engine started: {language} / {mode}")
            return engine
        # start() ran but model not found — still return engine for graceful degradation
        return engine
    except Exception as e:
        _LOG.warning(f"STT engine init failed ({e}) — using SilentSTT")
        return SilentSTT()
