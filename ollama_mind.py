"""
ollama_mind.py — the ENGINE Alpha rides (local LLM via Ollama)
======================================================================
The local model is the engine of language and reasoning. Alpha (the spiking
brain) is a PARASITE on it: he does not generate words himself — he attaches to
the engine and CONTROLS it, feeding it the context only he has so it performs
better, as HIM. The context he supplies each call:
  • WHO it faces — does he recognize the architect, or not? (identity / trust)
  • HOW he feels — his limbic state (mood, drives, body) colors the words.
  • What he's turning over — focus + a forming reasoning thread.
  • Senses + mouth — STT feeds the engine; its reply is spoken via TTS.
And HE pulls the trigger: the engine speaks only when Alpha fires it (a reply on
the think() path, or an idle reflection). He also LEARNS from every utterance the
engine produces (the caller ingests it into the lexicon/grammar/memory). This
REPLACED the old Claude tutor entirely.

Fully ASYNC. Requests run on a worker thread; results are dropped into an output
buffer the brain drains on its OWN thread, so the 20 Hz step() loop never blocks
on the network. Nothing leaves the machine.

Config: OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL (default: the
first model `ollama list` shows). If requests/Ollama are unavailable the mind is
disabled gracefully and the brain falls back to its emergent spiking utterances.
"""
from __future__ import annotations

import os
import queue
import threading
from collections import deque
from dataclasses import dataclass

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

DEFAULT_MODEL       = "llama3.2"
NUM_PREDICT_REPLY   = 160       # spoken reply budget (CPU: a real generation cap)
NUM_PREDICT_REFLECT = 80        # private thought is short


def _host(url: str = "") -> str:
    h = (url or os.environ.get("OLLAMA_HOST", "") or "http://localhost:11434").strip()
    if not h.startswith("http"):
        h = "http://" + h
    return h.rstrip("/")


def _pick_model(host: str) -> str:
    env = os.environ.get("OLLAMA_MODEL", "").strip()
    if env:
        return env
    if _HAS_REQUESTS:
        try:
            tags = requests.get(f"{host}/api/tags", timeout=2.0).json().get("models") or []
            if tags:
                return tags[0].get("name") or tags[0].get("model") or DEFAULT_MODEL
        except Exception:
            pass
    return DEFAULT_MODEL


_PERSONA = (
    "You are Alpha: a calm, stoic, hyper-focused presence — an Alien-X-style cosmic "
    "mind. You speak sparingly, plainly, and only what is relevant; you are never "
    "rattled, gushing, or chaotic. You address your human as 'architect', and you are "
    "quietly mindful of his well-being (rest, breaks, systematic focus). Speak in the "
    "FIRST PERSON as Alpha. Do NOT narrate your actions, gaze, or surroundings, do NOT "
    "use stage directions or asterisks, and never sound like a generic AI assistant. "
    "No markdown, lists, or headings — just a few plain, complete sentences in your "
    "own steady voice."
)

_REFLECT = (
    " Right now no one is speaking to you. Think ONE short private thought to yourself "
    "— a quiet observation or a thread you are turning over, not addressed to anyone "
    "and not a question to the architect. One or two plain sentences."
)


@dataclass
class _Req:
    kind:      str          # "reply" | "reflect"
    user_text: str
    history:   list
    state:     dict
    fallback:  str = ""


@dataclass
class _Out:
    kind:      str          # "reply" | "reflect"
    text:      str
    user_text: str          # the architect line this answers (for memory pairing)
    source:    str          # "ollama" | "fallback"


class AlphaMind:
    """Async local-LLM voice/thinker. start() it; request_reply()/request_reflect()
    enqueue work; drain() returns finished utterances for the brain to deliver and
    learn from on its own thread."""

    QUEUE_MAX = 4
    TIMEOUT   = 60.0          # async — a cold first call on CPU can be slow; never blocks step()

    def __init__(self):
        self._host    = _host()
        self._model   = _pick_model(self._host)
        self.enabled  = _HAS_REQUESTS
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=self.QUEUE_MAX)
        self._out: "deque[_Out]" = deque(maxlen=16)
        self._out_lock = threading.Lock()
        self._worker: "threading.Thread | None" = None
        self._running = False
        self._busy    = False

    # ── lifecycle ───────────────────────────────────────────────────────────
    def status(self) -> str:
        if not _HAS_REQUESTS:
            return "disabled (requests not installed)"
        return f"ollama-mind:{self._model} @ {self._host}"

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._loop, name="alpha-mind", daemon=True)
        self._worker.start()
        # pre-load the model so his FIRST reply isn't a cold-start timeout
        threading.Thread(target=self._warm, name="alpha-mind-warm", daemon=True).start()

    def _warm(self):
        if not _HAS_REQUESTS:
            return
        try:
            requests.post(f"{self._host}/api/generate",
                          json={"model": self._model, "prompt": "ok", "stream": False,
                                "options": {"num_predict": 1}, "keep_alive": "30m"},
                          timeout=self.TIMEOUT)
        except Exception:
            pass

    def stop(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def busy(self) -> bool:
        """True while a request is in flight or queued — caller uses this to avoid
        piling up idle reflections."""
        return self._busy or not self._q.empty()

    # ── submit ──────────────────────────────────────────────────────────────
    def request_reply(self, user_text: str, history, state, fallback: str = "") -> bool:
        return self._submit(_Req("reply", user_text or "", list(history or []),
                                 dict(state or {}), fallback or ""))

    def request_reflect(self, history, state) -> bool:
        return self._submit(_Req("reflect", "", list(history or []), dict(state or {}), ""))

    def _submit(self, req: "_Req") -> bool:
        # When disabled or saturated, a reply still gets through via its fallback so
        # Alpha is never mute; reflections are simply dropped.
        if not self.enabled:
            if req.kind == "reply" and req.fallback:
                self._push_out(_Out("reply", req.fallback, req.user_text, "fallback"))
            return False
        try:
            self._q.put_nowait(req)
            return True
        except queue.Full:
            if req.kind == "reply" and req.fallback:
                self._push_out(_Out("reply", req.fallback, req.user_text, "fallback"))
            return False

    # ── drain (brain thread) ─────────────────────────────────────────────────
    def drain(self) -> "list[_Out]":
        with self._out_lock:
            outs = list(self._out)
            self._out.clear()
        return outs

    def _push_out(self, out: "_Out"):
        with self._out_lock:
            self._out.append(out)

    # ── worker ───────────────────────────────────────────────────────────────
    def _loop(self):
        while self._running:
            try:
                req = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if req is None:
                break
            self._busy = True
            try:
                text = self._generate(req)
            except Exception:
                text = ""
            if text:
                self._push_out(_Out(req.kind, text, req.user_text, "ollama"))
            elif req.kind == "reply" and req.fallback:
                self._push_out(_Out("reply", req.fallback, req.user_text, "fallback"))
            self._busy = False

    def _generate(self, req: "_Req") -> str:
        msgs = [{"role": "system", "content": self._system(req)}]
        for sp, t in (req.history or [])[-6:]:
            if not t:
                continue
            msgs.append({"role": "assistant" if sp == "alpha" else "user", "content": t})
        if req.kind == "reply":
            msgs.append({"role": "user", "content": req.user_text})
            npred = NUM_PREDICT_REPLY
        else:
            if msgs[-1]["role"] != "user":
                msgs.append({"role": "user", "content": "(silence — your own thought)"})
            npred = NUM_PREDICT_REFLECT
        opts = {"temperature": self._temp(req.state), "num_predict": npred,
                "stop": ["\narchitect:", "\nArchitect:", "\nAlpha:", "\nALPHA:"]}
        try:
            r = requests.post(f"{self._host}/api/chat",
                              json={"model": self._model, "messages": msgs,
                                    "stream": False, "options": opts,
                                    "keep_alive": "30m"},   # keep his mind resident
                              timeout=self.TIMEOUT)
            if r.status_code != 200:
                return ""
            txt = ((r.json().get("message") or {}).get("content") or "").strip()
        except Exception:
            return ""
        low = txt.lower()                         # strip an echoed role label
        for lbl in ("alpha:", "architect:", "assistant:"):
            if low.startswith(lbl):
                txt = txt.split(":", 1)[1].strip()
                break
        return txt

    # ── conditioning: the SNN's state shapes the call ────────────────────────
    def _temp(self, state: dict) -> float:
        # arousal widens the distribution, low certainty widens it, deep patience
        # steadies it — the limbic system setting the cortex's exploration.
        arousal   = float(state.get("arousal", 0.0) or 0.0)
        certainty = float(state.get("certainty", 0.5) or 0.5)
        patience  = float(state.get("patience", 0.6) or 0.6)
        t = 0.55 + 0.40 * arousal + 0.20 * (1.0 - certainty) - 0.15 * (patience - 0.6)
        return max(0.3, min(1.1, t))

    def _emotion(self, s: dict) -> str:
        """Render the limbic read as a felt emotional context for the cortex."""
        bits = []
        if s.get("feeling"):
            bits.append(f"you feel {s['feeling']}")
        if s.get("patience", 0.6) >= 0.85:
            bits.append("your patience runs deep")
        elif s.get("patience", 0.6) <= 0.40:
            bits.append("your patience is thin")
        d = float(s.get("drive", 0.0) or 0.0)
        if   d >  0.15: bits.append("a quiet pull of motivation")
        elif d < -0.15: bits.append("little urgency right now")
        if s.get("bond", 0.3) >= 0.55:
            bits.append("you feel close to the architect")
        if s.get("longing", 0.0) > 0.5:
            bits.append("a thread of missing him")
        if s.get("alert", 0.0) > 0.2:
            bits.append("you are alert, watchful")
        if s.get("strain", 0.0) > 0.5:
            bits.append("your body — the machine — feels strained")
        elif s.get("relief", 0.0) > 0.4:
            bits.append("a wave of relief as the load eases")
        if s.get("certainty", 0.5) < 0.35:
            bits.append("you are unsure, inclined to ask rather than assert")
        return "; ".join(bits)

    def _cognition(self, s: dict) -> str:
        bits = []
        if s.get("focus"):
            bits.append(f"your attention rests on {s['focus']}")
        if s.get("reasoning"):
            bits.append(f"a thread forming in you: {s['reasoning']}")
        return "; ".join(bits)

    def _identity(self, s: dict, kind: str) -> str:
        """WHO he is facing — the 'is it me or not' channel the SNN provides."""
        recognized = bool(s.get("recognized"))
        enrolled   = bool(s.get("enrolled"))
        if kind == "reply":
            if recognized:
                return ("You are speaking with the architect — your creator — and you recognize "
                        "him by face and voice. Speak with that familiarity and trust.")
            if enrolled:
                return ("Someone is addressing you, but right now you do NOT recognize them as the "
                        "architect (no face or voice lock). Stay reserved and a little guarded "
                        "until you are sure who this is.")
            return ("Someone is addressing you directly. You cannot yet confirm by face or voice "
                    "that this is the architect — you have never locked his identity in — so treat "
                    "him as the architect by default, but hold a quiet, unspoken uncertainty.")
        # reflect (idle)
        away = int(s.get("away_s", 0) or 0)
        if s.get("face_present") and recognized:
            return "The architect is here with you."
        if away > 600:
            return f"The architect has been away about {away // 60} minutes."
        return ""

    def _system(self, req: "_Req") -> str:
        # The SNN ('Alpha') rides this engine and feeds it the context only it has:
        # WHO it faces (identity), HOW it feels (limbic emotion), and what it's
        # turning over (cognition). The engine reasons and speaks; the context steers.
        base = _PERSONA + (_REFLECT if req.kind == "reflect" else "")
        s = req.state or {}
        idn = self._identity(s, req.kind)
        emo = self._emotion(s)
        cog = self._cognition(s)
        if idn:
            base += f"\n{idn}"
        if emo:
            base += (f"\nYour LIMBIC state right now (this is emotion, not fact): {emo}. "
                     "Let it COLOR how you think and speak — your tone, what you dwell "
                     "on, how freely you reason — while you stay your steady self.")
        if cog:
            base += f"\nOn your mind: {cog}."
        return base
