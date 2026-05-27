"""
tts_engine.py — Voice Cloning TTS Engine
==========================================
Backend: Coqui XTTS v2 (local, offline, no API key)
  • Clones voices from a 6–30 second reference .wav file
  • Supports Bulgarian (bg) and English (en) natively
  • Runs entirely on CPU / Intel XPU — no CUDA required
  • Queue-based: brain loop pushes requests, TTS worker renders async

VOICE SETUP (do this once):
  1. Record 10–30 seconds of clean speech as a .wav file (16kHz+ mono/stereo)
  2. Place it at:
       voices/nova_reference.wav    ← Nova's voice
       voices/simona_reference.wav  ← Simona's voice
  3. Run once to trigger XTTS model download (~1.8 GB):
       python tts_engine.py --download

LANGUAGES:
  XTTS v2 supports 17 languages including Bulgarian (bg) and English (en).
  Set NOVA_LANG / SIMONA_LANG below to match the reference recording language.

FALLBACK:
  If TTS or torch is not installed, SilentTTS is used — logs to console,
  no audio, everything else still works.

Install:
  pip install TTS sounddevice numpy
  # On Windows you may need: pip install sounddevice --pre
"""

import os
import queue
import threading
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal

# ── Config ────────────────────────────────────────────────────────────────────

VOICES_DIR     = Path(__file__).parent / "voices"
NOVA_REF       = VOICES_DIR / "nova_reference.wav"
SIMONA_REF     = VOICES_DIR / "simona_reference.wav"

NOVA_LANG      = "en"   # change to "bg" for Bulgarian
SIMONA_LANG    = "en"   # change to "bg" for Bulgarian

MODEL_NAME     = "tts_models/multilingual/multi-dataset/xtts_v2"

# Output sample rate (XTTS v2 outputs 24kHz)
SAMPLE_RATE    = 24000

# Maximum queue depth — if brain is speaking faster than TTS renders, drop oldest
QUEUE_MAX      = 4

Speaker = Literal["nova", "simona"]


# ── Speech request ─────────────────────────────────────────────────────────────

@dataclass
class SpeechRequest:
    text:       str
    speaker:    Speaker
    language:   str = "en"
    priority:   int = 0     # higher = more urgent (not yet used)


# ── XTTS v2 backend ───────────────────────────────────────────────────────────

class XTTSBackend:
    """
    Wraps Coqui XTTS v2 for voice cloning.
    Lazy-loads the model on first use (~3s on Iris Xe CPU).
    """

    def __init__(self):
        self._model     = None
        self._lock      = threading.Lock()
        self._ready     = False
        self._error_msg: Optional[str] = None

        # Pre-validate reference files before loading the heavy model
        if not NOVA_REF.exists():
            self._error_msg = (
                f"Nova reference voice not found at {NOVA_REF}\n"
                f"Record 10–30s of clean speech and save as nova_reference.wav"
            )
        if not SIMONA_REF.exists():
            msg = (
                f"Simona reference voice not found at {SIMONA_REF}\n"
                f"Record 10–30s of clean speech and save as simona_reference.wav"
            )
            self._error_msg = (self._error_msg + "\n" + msg) if self._error_msg else msg

    def _load_model(self):
        """Lazy-load XTTS v2. Called in the TTS worker thread, not the brain loop."""
        if self._ready:
            return True
        with self._lock:
            if self._ready:
                return True
            try:
                from TTS.api import TTS
                print("[TTS] Loading XTTS v2 model (~3s)...")
                # gpu=False: forces CPU / lets IPEX handle XPU if available
                # For XPU acceleration: set gpu=True after confirming IPEX+XTTS compat
                self._model = TTS(MODEL_NAME, gpu=False)
                self._ready = True
                print("[TTS] XTTS v2 ready — voice cloning active")
                return True
            except Exception as e:
                self._error_msg = f"XTTS load failed: {e}"
                print(f"[TTS] {self._error_msg}")
                return False

    def synthesize(self, req: SpeechRequest) -> Optional[np.ndarray]:
        """
        Synthesize speech. Returns float32 numpy array at SAMPLE_RATE,
        or None on failure.

        XTTS v2 clones the voice from the reference wav every call.
        Caching the speaker embedding speeds this up — see _get_embedding().
        """
        if not self._load_model():
            return None
        if self._error_msg and "reference" in self._error_msg:
            print(f"[TTS] Skipping synthesis: {self._error_msg}")
            return None

        ref_wav = str(NOVA_REF if req.speaker == "nova" else SIMONA_REF)
        lang    = NOVA_LANG   if req.speaker == "nova" else SIMONA_LANG

        try:
            # tts_to_file writes a .wav; tts() returns raw waveform list
            audio_list = self._model.tts(
                text=req.text,
                speaker_wav=ref_wav,
                language=lang,
            )
            return np.array(audio_list, dtype=np.float32)
        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    @property
    def error(self) -> Optional[str]:
        return self._error_msg


# ── Playback ──────────────────────────────────────────────────────────────────

def _play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    Play synthesized audio through the default output device.
    Uses sounddevice (wraps PortAudio — works on Windows/macOS/Linux).
    Blocks until playback is complete (intentional: one utterance at a time).
    """
    try:
        import sounddevice as sd
        sd.play(audio, samplerate=sample_rate, blocking=True)
    except ImportError:
        print("[TTS] sounddevice not installed — audio output disabled")
        print(f"[TTS] Would have played {len(audio)/sample_rate:.2f}s of audio")
    except Exception as e:
        print(f"[TTS] Playback error: {e}")


# ── TTS Engine (public API) ───────────────────────────────────────────────────

class TTSEngine:
    """
    Queue-based TTS engine. Thread-safe.

    Usage (called from Rust via PyO3, or from brain.py):
        engine = TTSEngine()
        engine.start()

        engine.speak("Hello, I am Nova.", speaker="nova")
        engine.speak("Хей! Чуваш ли ме?", speaker="simona")

        engine.stop()
    """

    def __init__(self):
        self._backend    = XTTSBackend()
        self._queue: queue.Queue[Optional[SpeechRequest]] = queue.Queue(maxsize=QUEUE_MAX)
        self._worker     = None
        self._running    = False
        self._speaking   = False
        self._last_spoken: dict[str, str] = {}  # speaker → last text (dedup)

    def start(self):
        """Start the background TTS worker thread."""
        self._running = True
        self._worker  = threading.Thread(
            target=self._worker_loop,
            name="tts-worker",
            daemon=True,   # dies with the main process — no cleanup needed
        )
        self._worker.start()
        print("[TTS] Engine started")

    def stop(self):
        """Graceful shutdown — drain queue then stop."""
        self._running = False
        # Poison pill to unblock the worker
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker:
            self._worker.join(timeout=2.0)
        print("[TTS] Engine stopped")

    def speak(self, text: str, speaker: Speaker = "nova", language: str = "en") -> bool:
        """
        Enqueue a speech request. Non-blocking.

        Returns True if enqueued, False if queue is full (request dropped).
        Deduplicates consecutive identical utterances per speaker.
        """
        if not text or not text.strip():
            return False

        # Deduplicate: don't repeat the same phrase back-to-back
        if self._last_spoken.get(speaker) == text:
            return False

        req = SpeechRequest(text=text, speaker=speaker, language=language)
        try:
            self._queue.put_nowait(req)
            self._last_spoken[speaker] = text
            return True
        except queue.Full:
            # Drop oldest, push new
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(req)
                return True
            except queue.Full:
                return False

    def is_speaking(self) -> bool:
        """True while audio is playing — Rust can check this to avoid overlap."""
        return self._speaking

    @property
    def ready(self) -> bool:
        return self._backend._ready

    @property
    def error(self) -> Optional[str]:
        return self._backend.error

    def _worker_loop(self):
        """
        Background thread: dequeue requests → synthesize → play.
        Runs for the lifetime of the process.
        """
        while self._running:
            try:
                req = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if req is None:  # poison pill
                break

            print(f"[TTS] {req.speaker.upper()} → \"{req.text}\"")
            self._speaking = True

            audio = self._backend.synthesize(req)
            if audio is not None:
                _play_audio(audio)

            self._speaking = False
            self._queue.task_done()


# ── Silent fallback ───────────────────────────────────────────────────────────

class SilentTTS:
    """
    Drop-in replacement when TTS package is not installed.
    Logs speech to console instead of playing audio.
    Same API as TTSEngine.
    """

    def __init__(self):
        print("[TTS] TTS package not found — using SilentTTS (console only)")
        print("[TTS] Install: pip install TTS sounddevice")

    def start(self): pass
    def stop(self):  pass
    def is_speaking(self) -> bool: return False
    def speak(self, text: str, speaker: str = "nova", language: str = "en") -> bool:
        print(f"[SILENT TTS] {speaker.upper()}: {text}")
        return True
    ready = False
    error = "TTS package not installed"


# ── Factory ───────────────────────────────────────────────────────────────────

def create_engine() -> "TTSEngine | SilentTTS":
    """
    Returns a real TTSEngine if Coqui TTS is installed, SilentTTS otherwise.
    Called once at startup by brain.py or directly from Rust via PyO3.
    """
    try:
        import TTS  # noqa: F401
        import sounddevice  # noqa: F401
        engine = TTSEngine()
        engine.start()
        return engine
    except ImportError as e:
        print(f"[TTS] Import failed ({e}) — falling back to SilentTTS")
        return SilentTTS()


# ── CLI: trigger model download ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        print("Downloading XTTS v2 model (~1.8 GB)...")
        from TTS.api import TTS
        TTS(MODEL_NAME, gpu=False)
        print("Download complete. Place reference .wav files in voices/ and run the main app.")
    elif "--test" in sys.argv:
        engine = create_engine()
        engine.speak("Nova online. Auditory cortex active.", speaker="nova")
        engine.speak("Симона тук! Готова съм!", speaker="simona", language="bg")
        time.sleep(30)  # wait for playback
        engine.stop()
    else:
        print("Usage: python tts_engine.py --download | --test")
