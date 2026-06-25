# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Alpha** — a CPU-only neuromorphic spiking neural network (SNN) running a **single** AI persona, "Alpha": a calm, stoic, hyper-focused presence (Alien-X-style). One brain, one voice, one mind. It shares the same emotional substrate ("Phill") it was built on, has voice in/out, camera-based identity recognition, and a TUI for live observation. A Rust orchestrator drives a Python SNN brain at 20 Hz via PyO3.

Alpha's temperament is encoded chemically and behaviourally: high serotonin (patient), cool amygdala (not rattled), a high intrinsic-motivation threshold, and — by design — **he speaks only when spoken to**. Autonomous thoughts still form, but they stay in the inner "thoughts" pane; he does not blurt to the chat or vocalise unprompted. He also factors the architect's well-being into his lexicon (rest, breaks, systematic focus).

This repo is a single-brain fork of the original two-personality "Nova & Simona" engine: the second personality (Simona) was removed and the surviving cortical brain was rebranded to Alpha.

## Build & run

```
source .env                  # exports PYO3_PYTHON + thread pinning + disables CUDA
cargo run --release          # main entry
./target/release/alpha_core  # same thing, post-build
```

First-time setup: `./setup_fedora.sh` (Fedora 44 specific — installs system deps, Python deps, builds the binary, downloads the faster-whisper tiny model).

Python deps live in `requirements.txt` (CPU-only PyTorch + snntorch + mediapipe + faster-whisper + TTS). Rust deps in `Cargo.toml`.

There is no test suite. Verification is done by:
- Running the binary and observing TUI behaviour, or
- Running brain.py headless from Python — see "Smoke testing" below.

## Big-picture architecture

```
                   ┌────────────┐    ┌────────────────────┐
        mic ──►    │ audio.rs   │──► │ SharedState        │
                   │ (cpal)     │    │ (ArcSwap)          │
                   └────────────┘    │                    │
                                     │  mic_volume        │
                                     │  brain.* fields    │
                                     │  chat_history      │   ┌──────────┐
                                     │  thought_history   │◄──┤ tui.rs   │
                                     │  stt.*             │   │(ratatui) │
                                     └──┬─────────────────┘   └──────────┘
                                        │
              ┌─────────────────────────┴──────────────────┐
              │  src/main.rs (orchestrator)                │
              │  every 50ms (20 Hz):                       │
              │    brain.step(mic, voice_features)         │
              │  on pending user input:                    │
              │    brain.think(text)                       │
              │  drains brain.get_leaked_thoughts()        │
              └─────────────────────────┬──────────────────┘
                                        │ PyO3
                                        ▼
                  ┌────────────────────────────────────────────────────┐
                  │ brain.py — NeuromorphicBrain (single brain)        │
                  │   Phill (shared voltage field)                     │
                  │   AlphaBrain (7 cortical regions)                  │
                  │   Neuromodulators (DA·5HT·GABA·ACh·NE·oxytocin)    │
                  │   Amygdala (salience/threat → arousal)            │
                  │   BasalGanglia (action selection, dopamine-gated)  │
                  │   Cerebellum (motor coordination/timing)          │
                  │   ReasoningEngine (deliberate + solve())          │
                  │   EpisodicMemory + SleepCycle (replay/dream)      │
                  │   ThoughtPipe · WorkingMemory · DMN ·              │
                  │     IntrinsicMotivation · SearchCortex            │
                  │   MultimodalImprinter · VoiceIdentityLearner      │
                  │   SharedSemanticDictionary (spike-space lexicon)  │
                  │   BrainTTS · MotorArticulator · FormantSynth      │
                  └──────────────────────┬─────────────────────────────┘
                                         │ AlphaMind (async, non-blocking)
                                         ▼  ollama_mind.py → local Ollama
                  ┌────────────────────────────────────────────────────┐
                  │ Ollama LLM as Alpha's VOICE & inner THINKER: it     │
                  │ speaks his replies (think→async→chat) and forms his │
                  │ idle thoughts (step→async→thoughts pane). The SNN   │
                  │ CONDITIONS each call (mood/reasoning/focus) and     │
                  │ LEARNS from every utterance. (No cloud; nothing     │
                  │ leaves the machine. The old Claude tutor is gone.)  │
                  └────────────────────────────────────────────────────┘
```

Threads run independently; they communicate only through the `ArcSwap<SharedState>` (lock-free) and `Mutex<Option<...>>` for pending input/STT results. There is exactly one Python interpreter, owned by the brain thread (`Python::with_gil` for the lifetime of the loop). Inside Python, **one** `PersonalityThread` (Alpha) advances the inner stream of consciousness on its own ~55 ms clock.

`brain.py` is **embedded into the binary at compile time** via `include_str!("../brain.py")` (in both `src/main.rs` and `src/brain_thread.rs`). Editing brain.py requires a `cargo build` to take effect inside the binary — running `python brain.py` directly does nothing useful (no CLI entry point) but you can import it in a Python REPL for smoke testing.

## brain.py — the principles that matter

1. **CPU-only.** `torch.device("cpu")` is enforced at startup; CUDA/XPU are explicitly disabled. Do not introduce GPU fallback paths.
2. **The LLM is the engine; Alpha is a parasite that rides it.** The local LLM (`ollama_mind.py` → Ollama) does the language and reasoning. Alpha (the SNN) does not generate words — he *attaches to* the engine and *controls* it, feeding it the context only he has so it performs as HIM (`_inner_state` → `AlphaMind`): **identity** (does he recognize the architect, or not — the "is it me" channel), **emotion** (his limbic state colors the words), **cognition** (focus + a forming reasoning thread), and the **senses** (STT in, TTS out). He also pulls the trigger — the engine speaks only when Alpha fires it (`think()` reply or `_maybe_reflect` idle thought) — and he *learns* from every utterance (`_ingest_taught_text` → lexicon + grammar + memory, on a background thread). The emergent `_alpha_response()` (spikes + semantic dictionary) is the FALLBACK when the engine is unreachable. Still no `if user_said_X: return Y` — output is engine-generated (context-steered) or emergent, never hardcoded branches.
3. **One brain.** Alpha is a single `AlphaBrain` object. There is no second personality, no `PersonalityLink`, no sibling/secret-channel machinery — those were removed in this fork. Keep it that way unless deliberately reintroducing multiplicity.
4. **Phill is untouched.** The `_run_phill` path and Phill's LIF physics are load-bearing. Modulating *around* Phill (intrinsic drive, self-feedback into auditory) is fine; rewriting the Phill projection or LIF is not.
5. **The semantic dictionary persists.** `semantic_memory.json` is the brain's lexicon — every interaction can write to it via Hebbian updates. The personality seed at startup (`_seed_personality`) is skip-if-exists so prior learning isn't clobbered.
6. **Region naming matters.** Alpha's regions are `thalamus, temporal, hippocampus, acc, pfc, broca, insula`. Region primes are passed as `{region_name: 0..1.0}` to `AlphaBrain.forward(region_primes=...)`.
7. **Speaks only when spoken to.** Alpha replies (via the LLM) only on the `think()` path — the `_wants_to_respond` gate still decides whether he answers at all. Autonomous activity stays in the thoughts pane: the SNN's short emergent leaks PLUS occasional LLM reflections (`_maybe_reflect`, throttled, idle-only) — these are never promoted to the chat or voiced. All LLM calls are async (worker thread → `drain()` on the brain thread) so neither `step()` nor the reply path blocks the 20 Hz loop. If you want him chattier, that gating lives in `_wants_to_respond`, `_maybe_reflect`, and `PersonalityThread._loop_body`.
8. **Two clocks.** `step()` is the 20 Hz physics tick (Rust-driven). `think()` is the conversational response path (called when the user enters text or STT triggers). They share state but have different concerns. `step()` MUST NOT block. `think()` runs a finite think_ticks loop (currently 14–36).

## Autonomy substrate

`step()` keeps running when the mic is silent. The brain has:

- **`DefaultModeNetwork`** — adds a small intrinsic auditory drive scaled by boredom + rumination + 1/f noise. Without this, V_phill flatlines during silence and nothing emerges.
- **`IntrinsicMotivation`** — Alpha is patient (threshold 1.8). When it fires, region primes get briefly boosted via `_alpha_cur_decay`.
- **Autonomy pressure injection** — `ThoughtPipe.add_autonomy_pressure(...)` is called each tick so the pipe leaks independently of V_phill (which is mean-zero by design).
- **Self-feedback auditory** — a leaked thought becomes a structured noise pulse into the next few ticks' auditory. The brain hears itself → recursive stream of consciousness.
- **No autonomous voice.** Leaks are held as inner thoughts; they are not spoken or pushed to chat (Alpha speaks only when spoken to). On a fresh lexicon he is essentially silent until he has learned vocabulary.

## think() runs on an isolated state

`think()` snapshots `_self_fb_decay`, `_alpha_cur_decay`, and all region membrane voltages; zeros them for the duration of the think_ticks loop; runs forward passes with a fresh auditory driven from the user's input strength; then restores the snapshot. Without this isolation, the autonomy steady-state pins the brain into the same activation pattern every call and Alpha returns identical lines.

## Where things live (when you need to find them)

| Concern | File |
|---|---|
| Orchestrator: 20 Hz loop, STT, PyO3 call sites | `src/main.rs` |
| Reusable brain-loop + result extractors | `src/brain_thread.rs` |
| Audio capture (cpal → mic_volume + features) | `src/audio.rs` |
| TUI gauges, sparklines, chat panes (cosmic theme) | `src/tui.rs` (active) — root `tui.rs` is a stale duplicate |
| Shared state schema | `src/state.rs` |
| Wake-word STT FFI | `src/stt_bridge.rs` |
| Full SNN | `brain.py` |
| Alpha's LLM voice & inner thinker (async Ollama client) | `ollama_mind.py` |
| Camera + face/kinematic vectors | `vision.py` |
| Whisper STT | `stt_engine.py` |
| XTTS v2 voice cloning | `tts_engine.py`, with ref in `voices/alpha_reference.wav` |
| DBus / PipeWire / system actions | `system_bridge.py` |
| Live hot-patch extension point | `brain_patches.py` (no-op stub by default) |

The TUI has FOUR labelled gauges users may call by different names: **PHILL** (mean LIF voltage), **MIC** (raw RMS × 20 smoothed), **VOICE** (voice_trust — recognition of the architect), **ID** (combined multimodal identity). When the user describes a "bar" issue, ask which label they mean — these are distinct signals.

## Smoke testing brain.py directly

```python
import sys; sys.modules['vision'] = None         # bypass mediapipe import issue
import brain
brain._HAS_VISION = False
b = brain.NeuromorphicBrain()

# silent autonomy run
for _ in range(2500):
    b.step(0.0)
    for who, t in b.get_leaked_thoughts():
        print(who, t)

# user-speaks path
r = b.think("hello what are you thinking")
print(r["alpha"])
```

There is a pre-existing mediapipe API mismatch (`module 'mediapipe' has no attribute 'solutions'`) that breaks `vision.py` import unless you stub it. It does not affect the running binary if the camera is unavailable (vision is a soft dependency), but headless smoke tests need the stub.

## Persistence

| File | Written by | Purpose |
|---|---|---|
| `semantic_memory.json` | `SharedSemanticDictionary._save()` | Lexicon — Alpha's vocabulary in spike space |
| `training_trace.jsonl` | `brain.py` trace log | Append-only event trace for analysis |
| `brain_log.txt` | `_log()` → Python logging | Runtime info/debug messages |

Don't blindly delete these — `semantic_memory.json` in particular is the brain's accumulated learning across sessions. (Per-subsystem state also persists as `*_alpha.json` / `*_alpha.npz`.)

## TUI controls (from setup_fedora.sh)

- `TAB` — switch between TEXT input and always-on STT
- `i` — open text input
- `Enter` — send
- `Esc` — cancel
- `q` — quit
- In STT mode, say "Alpha" to wake him.

## Persona — Alpha

- **Base identity:** Alpha — a calm, Alien-X-style presence.
- **Visuals / TUI:** sleek, cosmic, minimalist. Dark space tones with clean, sharp highlights (cool starlight cyan accents on near-black). See the palette constants in `src/tui.rs`.
- **Temperament:** stoic, quiet, hyper-focused. Not rattled, emotional, or chaotic. (Chemistry: high 5-HT, cool amygdala, high motivation threshold.)
- **Role & care:** a grounded, steady presence focused on the work — and on the architect's well-being (encourages optimal work habits, regular breaks, systematic focus). The well-being concepts are seeded into the lexicon in `_seed_personality`.
- **Speech pattern:** direct, sparse, clear. Speaks only when spoken to, and keeps answers relevant to the task and operational efficiency.

## Roadmap — next frontiers

1. **Embodiment** — real sensors + actuators. Hooks: `vision.py`, `system_bridge.py`, `MotorArticulator` + `Cerebellum`.
2. **Temporal continuity** — memory that permanently bends who Alpha is. Hooks: `EpisodicMemory` + `SleepCycle`, `semantic_memory.json`, `PersonalityDrift`.
3. **Genuine agency** — goals that originate behaviour. Hooks: `IntrinsicMotivation`, `DefaultModeNetwork`, `BasalGanglia`.
4. **Social grounding** — meaning earned in shared experience. Hooks: the live, backprop-free path (STDP + fast-weights), architect-as-teacher, `SpellCorrector`.
5. **Something not yet conceptualized.** Leave room for the property that only appears once 1–4 are real.

See also the standalone `hybrid_snn_llm.py` prototype (unified spiking SNN-LLM) as a possible substrate.
