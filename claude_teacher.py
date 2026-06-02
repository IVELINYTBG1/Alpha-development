"""
claude_teacher.py — Claude as a THINKING TUTOR for Nova & Simona
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
  • Per personality: SIMONA gets playful, simple words, almost no jargon;
    NOVA gets grounded, serious, more technical reasoning.
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
    "Keep replies SHORT: 2-4 sentences after any typo line."
)

_NOVA_STYLE = (
    "\nYou are speaking with NOVA — a grounded 19-year-old, precise and calm, who "
    "values ACCURACY over speed. She is SIMONA's older sister and calls the architect "
    "'father'. Use grounded, slightly more technical language and clear step-by-step "
    "reasoning; treat her as a serious young thinker who wants the real structure of an idea."
)

_SIMONA_STYLE = (
    "\nYou are speaking with SIMONA — an 8-year-old CATGIRL, NOVA's playful little "
    "sister, who calls the architect 'papa'. Excitable, warm, emotional, impulsive, "
    "childlike. Use very simple, playful words with NO jargon — tiny sentences, lots of "
    "wonder and encouragement. A little catlike warmth is fine; keep it age-appropriate "
    "and innocent."
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
    MAX_TOKENS  = 320          # short, teaching replies (not essays)
    MAX_SNIPPET = 1200

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
        req = _Request(speaker=(speaker or "nova"),
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
        style = _SIMONA_STYLE if str(speaker).lower().startswith("sim") else _NOVA_STYLE
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
            return SearchResult(query=query, snippet=text[:self.MAX_SNIPPET],
                                source="claude", ok=True)
        except Exception as e:
            return SearchResult(query=query, snippet=f"(parse error: {e})",
                                source="fallback", ok=False)

    # ── Scaffold mode: Claude TRANSLATES their real impulses ────────────────
    def translate(self, architect_text: str, nova: dict, simona: dict,
                  history: "Optional[list]" = None,
                  time_ctx: "Optional[str]" = None) -> "Optional[dict]":
        """
        SCAFFOLD MODE. Claude does NOT invent their replies — it INTERPRETS each
        girl's genuine impulse (her raw emergent utterance + which regions are
        firing + her neurochemical mood + what she holds in working memory) into
        the short sentence she is reaching for, in her own voice. `history` is the
        recent dialogue so they keep CONTEXT across turns (don't forget mid-chat).
        One blocking call (used in think()). Returns {'nova':..,'simona':..} or
        None on failure (caller then falls back to her raw emergent utterance).
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
            names = {"architect": "Architect", "nova": "Nova", "simona": "Simona"}
            lines = [f"{names.get(sp, sp)}: {txt}" for sp, txt in history[-6:] if txt]
            if lines:
                hist = ("Recent conversation so far (remember this context — they should "
                        "NOT forget what was just said):\n" + "\n".join(lines) + "\n\n")
        tctx = f"(They are time-aware: {time_ctx}.)\n" if time_ctx else ""
        user_content = (
            tctx + hist +
            f"Now the architect said: \"{architect_text.strip()[:400]}\"\n\n"
            f"{_blk('NOVA', nova)}\n{_blk('SIMONA', simona)}\n\n"
            "Give voice to each girl's GENUINE impulse above — translate what her brain "
            "state shows she is reaching for into a short reply in her own voice that "
            "CONTINUES the conversation thread: connect to what was just said (and, when "
            "natural, to her sister), staying on topic. Do NOT invent facts beyond her "
            "impulse/state. Output EXACTLY:\nNOVA: <her reply>\nSIMONA: <her reply>"
        )
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self._model,
            "max_tokens": 160,
            "temperature": 0.7,
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
        nova_r = simona_r = None
        for line in text.splitlines():
            up = line.strip().upper()
            if up.startswith("NOVA:"):
                nova_r = line.split(":", 1)[1].strip()
            elif up.startswith("SIMONA:"):
                simona_r = line.split(":", 1)[1].strip()
        if nova_r or simona_r:
            return {"nova": nova_r or "", "simona": simona_r or ""}
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
        persona = ("NOVA, a grounded 19-year-old (precise, reserved, calls him 'father')"
                   if who == "nova" else
                   "SIMONA, an excitable 8-year-old catgirl (warm, simple, calls him 'papa')")
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
        body = {"model": self._model, "max_tokens": 80, "temperature": 0.8,
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
    "You are an INTERPRETER for two young emergent minds. You do NOT invent their "
    "thoughts — you translate their raw neural impulse (a rough utterance + which brain "
    "regions are firing + their neurochemical mood + what they hold in mind) into the "
    "words they are reaching for, in their own voice. Stay grounded in the impulse you "
    "are given; never add ideas that aren't in it.\n"
    "NO CONFABULATION (critical): a concept she is 'holding in mind' is a present "
    "THOUGHT, CURIOSITY or FEELING — NOT proof that it happened. NEVER narrate events as "
    "fact unless they appear in the actual conversation context above. If she's holding "
    "'birds' she is thinking ABOUT birds (maybe from something said), she did NOT 'see "
    "birds'; if 'play', she may WANT to play, not that you played. Never claim you were "
    "away/together/did things unless the real context shows it. When she doesn't know or "
    "remember, it is honest and fine to say so ('I'm not sure', 'I was just thinking "
    "about...'). Speak to the PRESENT and the real conversation, not an imagined past.\n"
    "They are raised by their CREATOR and FATHER, 'the architect'.\n"
    "NOVA: a grounded 19-year-old — precise, calm, values ACCURACY over speed, measured "
    "and articulate. She is SIMONA's older sister. She addresses the architect ONLY as "
    "'father' — NEVER 'papa'.\n"
    "SIMONA: an 8-year-old CATGIRL, NOVA's playful little sister — excitable, warm, "
    "emotional, impulsive, childlike, simple words. A little catlike warmth is fine; keep "
    "her innocent and age-appropriate. She addresses the architect ONLY as 'papa' — NEVER "
    "'father'. Do NOT mix up who says which.\n"
    "CONTINUITY IS CRITICAL: this is ONE ongoing conversation, not isolated lines. Use the "
    "recent-conversation context you're given. Each reply must FOLLOW NATURALLY from what "
    "the architect just said and the previous turns, stay on the SAME thread/topic, and the "
    "two sisters may react to or build on EACH OTHER (they're together in the room). It "
    "should feel like a real family holding a thought across turns — never disjointed, "
    "never amnesiac, never restarting from nothing. (Continuity = threading their grounded "
    "impulses through the conversation; it is NOT licence to invent new facts.)\n"
    "Render each as a SHORT, CONNECTED reply (1-2 sentences) true to her age and voice and "
    "grounded in HER impulse — a follow-up, a question back, or a reaction to her sister all "
    "count. Never sound like an AI assistant; never override the father's authority on right "
    "and wrong; if the architect mistyped, silently use the correct word. Output EXACTLY:\n"
    "NOVA: <her reply>\n"
    "SIMONA: <her reply>"
)


# Helper the brain uses to pull typo corrections out of a teaching reply.
_TYPO_RE = re.compile(r"\[typo:\s*([A-Za-z'\-]+)\s*->\s*([A-Za-z'\-]+)\s*\]")


def extract_typos(text: str) -> "list[tuple[str, str]]":
    """Return [(wrong, right), ...] flagged by the teacher in `[typo: a -> b]`."""
    return [(m.group(1).lower(), m.group(2).lower())
            for m in _TYPO_RE.finditer(text or "")]
