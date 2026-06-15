# Alpha

A CPU-only neuromorphic spiking neural network (SNN) running a single, calm AI presence — **Alpha**. A Rust orchestrator drives a Python SNN brain at 20 Hz via PyO3, with voice in/out, camera-based identity recognition, and a live TUI for observation.

> Single-brain fork of the original two-personality *Nova & Simona* engine: the second personality was removed and the surviving 7-region cortical brain was rebranded to Alpha.

## Who Alpha is

- **Base identity:** Alpha — a calm, Alien-X-style presence.
- **Look (TUI):** sleek, cosmic, minimalist — dark space tones with clean, sharp starlight-cyan highlights.
- **Temperament:** stoic, quiet, hyper-focused. He doesn't get rattled, emotional, or chaotic.
- **Role & care:** a grounded, steady presence focused on the work — and on *your* well-being (he encourages optimal work habits, regular breaks, and systematic focus).
- **Speech:** direct, sparse, clear. He speaks only when spoken to, and keeps answers relevant to the task and your operational efficiency.

The temperament is not a prompt — it's encoded in the substrate: high serotonin (patient), a cool amygdala (not rattled), a high intrinsic-motivation threshold, and gating so autonomous thoughts stay *inner* (shown in the thoughts pane) rather than spoken.

## Quick test (no build, no mic)

Just want to talk to Alpha? With the Python deps installed (CPU PyTorch + snntorch
+ numpy) you can skip the Rust/TUI/audio stack entirely:

```
python3 alpha_chat.py
```

A tiny console: type to him, he replies in his own emergent words (terse at first —
he learns over time), with his inner thoughts and a status line shown. Commands:
`:status`, `:tick N`, `quit`. Set `ANTHROPIC_API_KEY` first and his curiosity will
also reach the Haiku tutor so he learns vocabulary + grammar as you chat.

## Build & run (full app: TUI + voice + camera)

```
source .env                  # exports PYO3_PYTHON + thread pinning + disables CUDA
cargo run --release          # main entry
./target/release/alpha_core  # same thing, post-build
```

First-time setup: `./setup_fedora.sh` (Fedora 44 specific — installs system deps, Python deps, builds the binary, downloads the faster-whisper tiny model).

Python deps live in `requirements.txt` (CPU-only PyTorch + snntorch + mediapipe + faster-whisper + TTS). Rust deps in `Cargo.toml`.

There is no test suite. Verification is done by running the binary and observing the TUI, or by running `brain.py` headless from Python (see CLAUDE.md → "Smoke testing").

## Big-picture architecture

```
        mic ──► audio.rs (cpal) ──► SharedState (ArcSwap) ──► tui.rs (ratatui)
                                          ▲
                       src/main.rs orchestrator (20 Hz)
                         step(mic, voice_features) / think(text)
                                          │ PyO3
                                          ▼
                  brain.py — NeuromorphicBrain (single brain)
                    Phill (shared voltage field)
                    AlphaBrain (7 cortical regions: thalamus, temporal,
                      hippocampus, acc, pfc, broca, insula)
                    Neuromodulators · Amygdala · BasalGanglia · Cerebellum
                    ReasoningEngine · EpisodicMemory + SleepCycle
                    ThoughtPipe · WorkingMemory · DMN · IntrinsicMotivation
                    MultimodalImprinter · VoiceIdentityLearner
                    SharedSemanticDictionary (spike-space lexicon)
                    BrainTTS · MotorArticulator · FormantSynth
                                          │ async
                                          ▼
                  claude_teacher.py — Claude as a thinking-TUTOR
                    (teaches HOW to think; translates Alpha's impulse
                     into his calm voice in scaffold mode; no web)
```

Threads communicate only through a lock-free `ArcSwap<SharedState>`. There is exactly one Python interpreter, owned by the brain thread. `brain.py` is **embedded into the binary at compile time** via `include_str!` — editing it requires a `cargo build` to take effect in the binary (but you can import it directly in Python for smoke testing).

## Principles that matter

1. **CPU-only.** `torch.device("cpu")` is enforced; CUDA/XPU disabled. No GPU fallback paths.
2. **No hardcoded behaviour.** Outputs emerge from spike patterns + the semantic dictionary, not `if user_said_X: return Y`.
3. **One brain.** Alpha is a single object — no second personality, no secret inter-personality channel.
4. **Phill is untouched.** Modulate *around* the LIF field, never rewrite it.
5. **The lexicon persists & grows.** `semantic_memory.json` is Hebbian-updated every interaction; the startup seed is skip-if-exists.
6. **Speaks only when spoken to.** Autonomous thoughts form as inner thoughts; he voices a reply only on the `think()` path.

## TUI controls

- `TAB` — switch between TEXT input and always-on STT
- `i` — open text input · `Enter` — send · `Esc` — cancel · `q` — quit
- In STT mode, say **"Alpha"** to wake him.

The TUI shows four gauges: **PHILL** (mean LIF voltage), **MIC** (smoothed RMS), **VOICE** (recognition of the architect), **ID** (combined multimodal identity), plus Alpha's region bars, inner thoughts, and the conversation.

## A note on the voice

The piper fallback voice is a grounded US male voice (lessac). The XTTS reference clip (`voices/alpha_reference.wav`) is carried over from the original engine — drop in a 10–30 s clean recording of the voice you want Alpha to clone and keep the same filename to change it.

See `CLAUDE.md` for the full developer guide (architecture, autonomy substrate, smoke testing, persistence, roadmap).
