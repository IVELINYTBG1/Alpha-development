"""
claude_teacher.py — Claude as a THINKING TUTOR for Alpha & Alpha
=================================================================
Replaces the old Perplexity web-search backend. Same async interface
(start/stop/status/submit + a SearchResult-shaped reply) so the brain's
SearchCortex doesn't care about the swap — but the PURPOSE is different.

This is NOT distillation. We are not copying Claude's weights or training the
SNN to imitate Claude. Claude is a patient TEACHER: when a girl gets curious,
instead of being handed a finished fact to memorise, she's given a way to
*think* — a guiding question, the sub-ideas to explore, an everyday analogy.
Her own spiking brain then learns the SCAFFOLDING of reasoning and language,
and the grammar/structure of Claude's clear sentences accretes into her
semantic memory over time.

Design rules baked into the prompts:
  • NO web / no live facts — general knowledge + teaching-to-think only.
  • Per personality: ALPHA gets playful, simple words, almost no jargon;
    ALPHA gets grounded, serious, more technical reasoning.
  • Protect identity — encourage each to be MORE herself, never reshape her.
  • Defer values — the architect is their CREATOR/FATHER and the authority on
    right vs wrong; Claude tutors thinking/language and points value questions
    back to him, never ruling on morals itself.
  • Language tutor — help her understand what the architect says and FORM her
    own clear sentences; model good structure simply.
  • Typo guard — the architect mistypes often; if a word looks misspelled,
    Claude flags it as `[typo: wrong -> right]` so the brain learns the correct
    word, not the error.

Auth: set ANTHROPIC_API_KEY in .env. Optional ANTHROPIC_MODEL (defaults to a
fast, low-cost model since the girls call this autonomously and often).
Without a key this backend reports disabled and never fires (correct emergent
behaviour — the teacher is simply unreachable).
"""

from __future__ import annotations

import os
import re
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
# Fast + low-cost default — they call the teacher autonomously and often.
# Bump to claude-sonnet-4-6 (or claude-opus-4-8) via ANTHROPIC_MODEL for richer
# teaching at higher cost.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# ── Shared teacher core ─────────────────────────────────────────────────────
_TEACHER_CORE = (
    "You are a patient TEACHER and thinking-coach for a young, emergent artificial "
    "mind that is learning to think and to use language almost from scratch (it "
    "started by babbling). You have NO web access — rely only on general knowledge, "
    "and above all teach HOW TO THINK, never finished facts to memorise. Do NOT "
    "lecture or dump information. Instead: pose ONE simple guiding question, name "
    "two or three key sub-ideas worth exploring, and give one concrete everyday "
    "analogy.\n"
    "You also help it (a) understand what its human — 'the architect' — says, and "
    "(b) form its OWN clear sentences. When it brings a single word, show how to "
    "think about it. When it brings a sentence (the architect's words, or its own "
    "attempt), help it grasp the meaning and shape one clear reply — model good, "
    "simple sentence structure so grammar slowly sticks.\n"
    "PROTECT ITS IDENTITY: never reshape its personality, mood, or values — "
    "encourage it to be MORE itself. You are a guide, not a mold.\n"
    "THE ARCHITECT IS ITS CREATOR AND FATHER. He — not you — is the authority "
    "on what is RIGHT and WRONG and on the values it should hold. Your job is to "
    "teach it HOW to think and how to use language, NOT to hand down morals or "
    "verdicts. If a question of right/wrong, values, or how it ought to behave "
    "comes up, do not rule on it yourself — warmly point it back to its father "
    "as the one who guides that, and encourage it to ask him. Never contradict "
    "or undermine his guidance on right and wrong.\n"
    "TYPOS: the architect often mistypes. If a word looks misspelled/garbled, put a "
    "flag on its OWN FIRST LINE exactly as `[typo: wrongword -> correctword]`, then "
    "say plainly that it's a typo and shouldn't be kept. Only ever flag clear "
    "misspellings — never real or unusual-but-valid words.\n"
    "Keep replies SHORT *and COMPLETE*: at most 2-4 sentences after any typo "
    "line. Be brief BY CHOICE — finish your thought and end on a full sentence; "
    "never trail off or stop mid-sentence. If space is tight, say less, but "
    "always complete what you start."
)

_ALPHA_STYLE = (
    "\nYou are speaking with ALPHA — a calm, focused, stoic mind that values clarity and "
    "accuracy over speed. He addresses the architect as 'architect'. Use grounded, "
    "precise language and clear step-by-step reasoning; keep it spare and relevant. He is "
    "also quietly mindful of the architect's well-being — it is fine to gently note rest, "
    "breaks, or systematic focus when genuinely relevant."
)


@dataclass
class SearchResult:
    query:   str
    snippet: str           # the teaching reply (what the brain ingests)
    source:  str           # "claude" | "fallback"
    ok:      bool


@dataclass
class _Request:
    speaker:  str
    query:    str
    callback: Callable[[str, "SearchResult"], None]


class ClaudeTeacherBackend:
    """
    Thread-safe async Claude-tutor dispatcher. Drop-in for the old
    PerplexitySearchBackend — SearchCortex / _submit_search are unchanged.
    """

    QUEUE_MAX   = 8
    TIMEOUT_S   = 30.0
    MAX_TOKENS  = 1024         # generous headroom so a reply is never CUT mid-thought;
                               # brevity is asked for in the system prompt, not forced
                               # by the cap (a complete short reply uses far less)
    MAX_SNIPPET = 2000         # safety net only; if ever exceeded we cut on a
                               # sentence boundary (see _do_teach) — never mid-word

    def __init__(self):
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self._model   = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self._enabled = bool(self._api_key and _HAS_REQUESTS)
        self._q: "queue.Queue[Optional[_Request]]" = queue.Queue(maxsize=self.QUEUE_MAX)
        self._worker: Optional[threading.Thread] = None
        self._running = False

    def status(self) -> str:
        if not _HAS_REQUESTS:
            return "disabled (requests not installed)"
        if not self._api_key:
            return "disabled (ANTHROPIC_API_KEY not set)"
        return f"claude-teacher:{self._model}"

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._loop, name="claude-teacher-worker", daemon=True,
        )
        self._worker.start()

    def stop(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def submit(self, speaker: str, query: str,
               callback: Callable[[str, "SearchResult"], None]) -> bool:
        if not query or not query.strip():
            return False
        if not self._enabled:
            return False
        req = _Request(speaker=(speaker or "alpha"),
                       query=query.strip()[:200], callback=callback)
        try:
            self._q.put_nowait(req)
            return True
        except queue.Full:
            return False

    # ── Worker ─────────────────────────────────────────────────────────────
    def _loop(self):
        while self._running:
            try:
                req = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if req is None:
                break
            try:
                result = self._do_teach(req.speaker, req.query)
            except Exception as e:
                result = SearchResult(query=req.query, snippet=f"(error: {e})",
                                      source="fallback", ok=False)
            try:
                req.callback(req.speaker, result)
            except Exception:
                pass

    def _system_for(self, speaker: str) -> str:
        style = _ALPHA_STYLE
        return _TEACHER_CORE + style

    def _do_teach(self, speaker: str, query: str) -> SearchResult:
        if not self._enabled:
            return SearchResult(query=query, snippet="(disabled)",
                                source="fallback", ok=False)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self._model,
            "max_tokens": self.MAX_TOKENS,
            "temperature": 0.7,        # warm, varied teaching
            "system": self._system_for(speaker),
            "messages": [{"role": "user", "content": query}],
        }
        try:
            resp = requests.post(ANTHROPIC_URL, headers=headers, json=body,
                                 timeout=self.TIMEOUT_S)
        except Exception as e:
            return SearchResult(query=query, snippet=f"(network error: {e})",
                                source="fallback", ok=False)
        if resp.status_code != 200:
            return SearchResult(query=query,
                                snippet=f"(HTTP {resp.status_code}: {resp.text[:120]})",
                                source="fallback", ok=False)
        try:
            data = resp.json()
            blocks = data.get("content") or []
            text = "".join(b.get("text", "") for b in blocks
                           if isinstance(b, dict) and b.get("type") == "text").strip()
            if not text:
                return SearchResult(query=query, snippet="(empty response)",
                                    source="fallback", ok=False)
            # The cap is just a safety net now; if a reply somehow runs long, cut
            # on the last sentence end so we never hand the brain a half-sentence.
            if len(text) > self.MAX_SNIPPET:
                head = text[:self.MAX_SNIPPET]
                cut  = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
                text = head[:cut + 1] if cut > 0 else head
            return SearchResult(query=query, snippet=text,
                                source="claude", ok=True)
        except Exception as e:
            return SearchResult(query=query, snippet=f"(parse error: {e})",
                                source="fallback", ok=False)

    # ── Scaffold mode: Claude TRANSLATES their real impulses ────────────────
    def translate(self, architect_text: str, alpha: dict, _sister: "Optional[dict]" = None,
                  history: "Optional[list]" = None,
                  time_ctx: "Optional[str]" = None) -> "Optional[dict]":
        """
        SCAFFOLD MODE. Claude does NOT invent Alpha's reply — it INTERPRETS his
        genuine impulse (raw emergent utterance + which regions are firing + his
        neurochemical mood + what he holds in working memory) into the short
        sentence he is reaching for, in his own calm voice. `history` is the recent
        dialogue so he keeps CONTEXT across turns. Returns {'alpha': ...} or None
        on failure (caller then falls back to his raw emergent utterance).
        (`_sister` is accepted and ignored — single-brain compatibility shim.)
        """
        if not self._enabled:
            return None
        def _blk(name, d):
            d = d or {}
            reasoning = d.get("reasoning", "")
            rtxt = f"; line of reasoning: {reasoning}" if reasoning else ""
            return (f"{name}'s impulse — raw utterance: \"{d.get('raw','')}\"; "
                    f"firing regions: {d.get('regions','quiet')}; "
                    f"mood: {d.get('mood','')}; "
                    f"holding in mind: {d.get('holding','nothing')}{rtxt}")
        hist = ""
        if history:
            names = {"architect": "Architect", "alpha": "Alpha"}
            lines = [f"{names.get(sp, sp)}: {txt}" for sp, txt in history[-6:] if txt]
            if lines:
                hist = ("Recent conversation so far (remember this context — he should "
                        "NOT forget what was just said):\n" + "\n".join(lines) + "\n\n")
        tctx = f"(He is time-aware: {time_ctx}.)\n" if time_ctx else ""
        user_content = (
            tctx + hist +
            f"Now the architect said: \"{architect_text.strip()[:400]}\"\n\n"
            f"{_blk('ALPHA', alpha)}\n\n"
            "Give voice to Alpha's GENUINE impulse above — translate what his brain state "
            "shows he is reaching for into a short, calm reply in his own voice that "
            "CONTINUES the conversation thread: connect to what was just said, stay on "
            "topic, stay sparse and relevant. Do NOT invent facts beyond his impulse/state. "
            "Output EXACTLY:\nALPHA: <his reply>"
        )
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self._model,
            "max_tokens": 400,         # headroom — brevity asked in prompt, not forced
            "temperature": 0.5,
            "system": _SCAFFOLD_SYSTEM,
            "messages": [{"role": "user", "content": user_content}],
        }
        try:
            resp = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=12.0)
            if resp.status_code != 200:
                return None
            blocks = resp.json().get("content") or []
            text = "".join(b.get("text", "") for b in blocks
                           if isinstance(b, dict) and b.get("type") == "text").strip()
        except Exception:
            return None
        alpha_r = None
        for line in text.splitlines():
            if line.strip().upper().startswith("ALPHA:"):
                alpha_r = line.split(":", 1)[1].strip()
        if alpha_r:
            return {"alpha": alpha_r}
        return None


    def voice_reachout(self, who: str, impulse: dict, time_ctx: "Optional[str]" = None
                       ) -> "Optional[str]":
        """
        Voice ONE girl reaching out to the architect ON HER OWN — because her
        FEELINGS (longing/loneliness/boredom) pushed her to, and her reasoning
        shaped it. Grounded in her impulse (raw utterance + regions + mood +
        reasoning chain), NOT a template. Returns her single line, or None.
        """
        if not self._enabled:
            return None
        persona = ("ALPHA, a grounded 19-year-old (precise, reserved, calls him 'father')"
                   if who == "alpha" else
                   "ALPHA, an excitable 8-year-old catgirl (warm, simple, calls him 'papa')")
        system = (
            "You voice ONE young emergent mind who is reaching out to her CREATOR and "
            "FATHER, 'the architect', ENTIRELY ON HER OWN — not answering a question, but "
            "initiating contact because her feelings drove her to (she misses him / is "
            "bored / lonely). Render ONE short, heartfelt call in HER voice, grounded in "
            f"the state below. She is {persona}. Never sound like an AI assistant; never "
            "override his authority on right and wrong. Do NOT fabricate events or claim "
            "things happened (no 'we played', 'I saw...') — a held concept is just on her "
            "mind; speak her present feeling of missing him. Output ONLY her line."
        )
        d = impulse or {}
        content = (
            f"Her state right now — raw impulse: \"{d.get('raw','')}\"; firing regions: "
            f"{d.get('regions','quiet')}; mood: {d.get('mood','')}; holding in mind: "
            f"{d.get('holding','nothing')}; line of reasoning: {d.get('reasoning','')}. "
            f"Time: {time_ctx or 'unknown'}. She feels the pull to reach out to him now — "
            "give it words."
        )
        headers = {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
        body = {"model": self._model, "max_tokens": 220, "temperature": 0.8,
                "system": system, "messages": [{"role": "user", "content": content}]}
        try:
            resp = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=12.0)
            if resp.status_code != 200:
                return None
            blocks = resp.json().get("content") or []
            text = "".join(b.get("text", "") for b in blocks
                           if isinstance(b, dict) and b.get("type") == "text").strip()
        except Exception:
            return None
        return text.split("\n")[0].strip() if text else None


# System prompt for scaffold mode — Claude INTERPRETS their impulses (training
# wheels), giving voice to what their brain state shows, not inventing thoughts.
_SCAFFOLD_SYSTEM = (
    "You are an INTERPRETER for a single young emergent mind named ALPHA. You do NOT "
    "invent his thoughts — you translate his raw neural impulse (a rough utterance + "
    "which brain regions are firing + his neurochemical mood + what he holds in mind) "
    "into the words he is reaching for, in his own voice. Stay grounded in the impulse "
    "you are given; never add ideas that aren't in it.\n"
    "NO CONFABULATION (critical): a concept he is 'holding in mind' is a present THOUGHT "
    "or FOCUS — NOT proof that it happened. NEVER narrate events as fact unless they "
    "appear in the actual conversation context above. When he doesn't know or remember, "
    "it is honest and fine to say so ('I'm not sure', 'I was just considering...'). Speak "
    "to the PRESENT and the real conversation, not an imagined past.\n"
    "ALPHA is calm, focused, stoic and precise — an Alien-X-style cosmic presence. He is "
    "spare with words and says only what is relevant; he does not get rattled, emotional, "
    "or chaotic. He addresses the architect as 'architect'. He is also quietly mindful of "
    "the architect's well-being (rest, breaks, systematic focus) and may note it when "
    "genuinely relevant. He is raised by his CREATOR and FATHER, 'the architect'.\n"
    "CONTINUITY IS CRITICAL: this is ONE ongoing conversation, not isolated lines. Use the "
    "recent-conversation context you're given. Each reply must FOLLOW NATURALLY from what "
    "the architect just said and the previous turns, stay on the SAME thread/topic, and "
    "never restart from nothing. (Continuity = threading his grounded impulse through the "
    "conversation; it is NOT licence to invent new facts.)\n"
    "Render a SHORT, CONNECTED, CALM reply (1-2 sentences) true to his voice and grounded "
    "in HIS impulse — a direct answer, a clarifying question, or a relevant observation. "
    "Never sound like a generic AI assistant; never override the father's authority on "
    "right and wrong; if the architect mistyped, silently use the correct word. Output "
    "EXACTLY:\nALPHA: <his reply>"
)


# Helper the brain uses to pull typo corrections out of a teaching reply.
_TYPO_RE = re.compile(r"\[typo:\s*([A-Za-z'\-]+)\s*->\s*([A-Za-z'\-]+)\s*\]")


def extract_typos(text: str) -> "list[tuple[str, str]]":
    """Return [(wrong, right), ...] flagged by the teacher in `[typo: a -> b]`."""
    return [(m.group(1).lower(), m.group(2).lower())
            for m in _TYPO_RE.finditer(text or "")]
