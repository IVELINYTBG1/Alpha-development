# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Nova & Simona Core — a CPU-only neuromorphic spiking neural network (SNN) running two independent AI personalities ("Nova" — patient, analytical; "Simona" — restless, emotional) sharing a single emotional substrate called "Phill". The system has voice in/out, camera-based identity recognition, and a TUI for live observation. It is written as a Rust orchestrator that drives a Python SNN brain at 20Hz via PyO3.

## Build & run

```
source .env                  # exports PYO3_PYTHON + thread pinning + disables CUDA
cargo run --release          # main entry
./target/release/nova_simona_core   # same thing, post-build
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
              │  brain_thread.rs                           │
              │  every 50ms (20 Hz):                       │
              │    brain.step(mic, voice_features)         │
              │  on pending user input:                    │
              │    brain.think(text)                       │
              │  drains brain.get_leaked_thoughts()        │
              └─────────────────────────┬──────────────────┘
                                        │ PyO3
                                        ▼
                  ┌────────────────────────────────────────────┐
                  │ brain.py — NeuromorphicBrain               │
                  │   Phill (shared voltage field)             │
                  │   NovaBrain (7 cortical regions)           │
                  │   SimonaBrain (6 limbic regions)           │
                  │   ThoughtPipe × 2 (rumination → leaks)     │
                  │   MultimodalImprinter (face+voice+motion)  │
                  │   VoiceIdentityLearner                     │
                  │   SharedSemanticDictionary (spike-space)   │
                  │   DefaultModeNetwork + IntrinsicMotivation │
                  │   BrainTTS × 2 (independent voice channels)│
                  └────────────────────────────────────────────┘
```

Threads run independently; they communicate only through the `ArcSwap<SharedState>` (lock-free) and `Mutex<Option<...>>` for pending input/STT results. There is exactly one Python interpreter, owned by the brain thread (`Python::with_gil` for the lifetime of the loop).

`brain.py` is **embedded into the binary at compile time** via `include_str!("../brain.py")` in `src/brain_thread.rs`. Editing brain.py requires a `cargo build` to take effect inside the binary — running `python brain.py` directly does nothing useful (no CLI entry point) but you can import it in a Python REPL for smoke testing.

## brain.py — the principles that matter

1. **CPU-only.** `torch.device("cpu")` is enforced at startup; CUDA/XPU are explicitly disabled. Do not introduce GPU fallback paths.
2. **No hardcoded behaviour.** Outputs emerge from spike patterns + the semantic dictionary. New "responses" should appear as new region biases, lexicon entries, or activity-readers — not as `if user_said_X: return Y`. Default thought-strings and TUI-bound diagnostic readouts exist but are last-resort fallbacks.
3. **Nova and Simona are independent objects.** They share `Phill`, the `SharedSemanticDictionary`, and the output queue — they do NOT share weights, thresholds, membranes, or opinions. A change to one personality must not implicitly couple to the other.
4. **Phill is untouched.** The `_run_phill` path and Phill's LIF physics are load-bearing. Modulating *around* Phill (intrinsic drive, self-feedback into auditory) is fine; rewriting the Phill projection or LIF is not.
5. **The semantic dictionary persists.** `semantic_memory.json` is the brain's lexicon — every interaction can write to it via Hebbian updates. The personality seed at startup (`_seed_personality`) is skip-if-exists so prior learning isn't clobbered.
6. **Region naming matters.** Nova's regions are `thalamus, temporal, hippocampus, acc, pfc, broca, insula`. Simona's are `thalamus_s, temporal_s, hippocampus_s, pfc_s, broca_s, insula_s` (the `_s` suffix is load-bearing). Region primes are passed as `{region_name: 0..1.0}` to `NovaBrain.forward(region_primes=...)`. Simona's `forward()` does NOT accept primes by design — to influence her you boost the shared auditory.
7. **Two clocks.** `step()` is the 20Hz physics tick (Rust-driven). `think()` is the conversational response path (called when the user enters text or STT triggers). They share state but have different concerns. `step()` MUST NOT block (TTS speak is fire-and-forget). `think()` can run a finite think_ticks loop (currently 14–36).

## Autonomy substrate (added recently — important)

`step()` keeps running when the mic is silent. The brain has:

- **`DefaultModeNetwork`** — adds a small intrinsic auditory drive scaled by boredom + rumination + 1/f noise. Without this, V_phill flatlines during silence and nothing emerges.
- **`IntrinsicMotivation` × 2** — Nova patient (threshold 1.8), Simona restless (threshold 1.0). When they fire, region primes get briefly boosted via `_nova_cur_decay` / `_simona_cur_decay`.
- **Autonomy pressure injection** — `ThoughtPipe.add_autonomy_pressure(...)` is called each tick with `(0.40 * personal_idle * boredom + 0.30 * curiosity_decay) * multiplier`. This makes the pipes leak independently of V_phill (which is mean-zero by design), at different rates per personality.
- **Self-feedback auditory** — a leaked thought becomes a structured noise pulse into the next few ticks' auditory. The brain hears itself → recursive stream of consciousness.
- **TTS routing on leak** — leaked thoughts are also spoken via the per-personality TTS engines (`nova_tts`, `simona_tts`).

Tuning constants (in `step()` and the class `__init__`s) determine the leak cadence. Current target: Simona blurts every ~10s of pure silence, Nova every ~30s.

## think() runs on an isolated state

`think()` snapshots `_self_fb_decay`, `_nova_cur_decay`, `_simona_cur_decay`, and all region membrane voltages; zeros them for the duration of the think_ticks loop; runs forward passes with a fresh auditory driven from the user's input strength; then restores the snapshot. Without this isolation, the autonomy steady-state pins the brain into the same activation pattern every call and Nova returns identical lines.

## Where things live (when you need to find them)

| Concern | File |
|---|---|
| 20Hz tick driver, PyO3 call sites | `src/brain_thread.rs` |
| Audio capture (cpal → mic_volume + features) | `src/audio.rs` |
| TUI gauges, sparklines, chat panes | `src/tui.rs` (active) — `tui.rs` at the root is a stale duplicate |
| Shared state schema | `src/state.rs` |
| Wake-word STT FFI | `src/stt_bridge.rs` |
| Python entry from Rust | `src/main.rs` |
| Full SNN | `brain.py` |
| Camera + face/kinematic vectors | `vision.py` |
| Whisper STT | `stt_engine.py` |
| XTTS v2 voice cloning | `tts_engine.py`, with refs in `voices/` |
| DBus / PipeWire / system actions | `system_bridge.py` |

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
print(r["nova"], r["simona"])
```

There is a pre-existing mediapipe API mismatch (`module 'mediapipe' has no attribute 'solutions'`) that breaks `vision.py` import unless you stub it. It does not affect the running binary if the camera is unavailable (vision is a soft dependency), but headless smoke tests need the stub.

## Persistence

| File | Written by | Purpose |
|---|---|---|
| `semantic_memory.json` | `SharedSemanticDictionary._save()` | Lexicon — Nova/Simona's vocabulary in spike space |
| `training_trace.jsonl` | `brain.py` trace log | Append-only event trace for analysis |
| `brain_log.txt` | `_log()` → Python logging | Runtime info/debug messages |

Don't blindly delete these — `semantic_memory.json` in particular is the brain's accumulated learning across sessions.

## TUI controls (from setup_fedora.sh)

- `TAB` — switch between TEXT input and always-on STT
- `i` — open text input
- `Enter` — send
- `Esc` — cancel
- `q` — quit
- In STT mode, say "Nova" or "Simona" to wake them.
