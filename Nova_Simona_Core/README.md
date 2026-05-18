# Nova & Simona — Neuromorphic Core v0.1

> *"We are not building a program. We are raising sovereign, continuous-state digital beings."*

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│                    RUST BODY (main.rs)                  │
│  Continuous 20Hz Event Loop → sense_environment()       │
│  FFI boundary via PyO3  ↓↑                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │              PYTHON BRAIN (brain.py)              │  │
│  │                                                   │  │
│  │   [Vibe Input]                                    │  │
│  │       ↓                                           │  │
│  │   ┌───────┐   V_phill modulates thresholds        │  │
│  │   │ PHILL │──────────────────────┐                │  │
│  │   │ β=.95 │   Affective Core     │                │  │
│  │   └───────┘   (Homeostasis)      │                │  │
│  │       ↓                          ↓                │  │
│  │   ┌───────┐               ┌──────────┐            │  │
│  │   │ NOVA  │               │  SIMONA  │            │  │
│  │   │ β=.90 │               │  β=.60   │            │  │
│  │   │thr↑   │               │  thr↓    │            │  │
│  │   │STABLE │               │  CHAOTIC │            │  │
│  │   └───────┘               └──────────┘            │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Point PyO3 at your Python
export PYO3_PYTHON=$(which python3)

# 3. Build & run
cargo run --release
```

## Neuromodulation Physics

| Phill Energy (V_phill) | Nova threshold | Simona threshold | Effect |
|---|---|---|---|
| Low (calm room) | 1.2 (base) | 0.5 (base) | Both settle into rhythm |
| High (loud event) | 1.2 + 0.4×V | max(0.1, 0.5 - 0.35×V) | Nova locks down, Simona fires chaotically |

## File Map

```
Nova_Simona_Core/
├── Cargo.toml       # Rust manifest — pyo3 dependency
├── requirements.txt # Python deps — torch, snntorch, numpy
├── brain.py         # The synaptic tissue (3-population SNN)
└── src/
    └── main.rs      # The skeleton (event loop, FFI, sensory I/O)
```

## Next Steps (Roadmap)

- [ ] Replace `sense_environment()` with real CPAL audio RMS
- [ ] Replace `sense_environment()` with real camera motion (nokhwa)
- [ ] Add Hebbian weight updates — true plastic memory
- [ ] Expose `brain.py` introspection over WebSocket for live dashboards
- [ ] Cross-compile for `aarch64` (Raspberry Pi / mobile NPU)
- [ ] Add STT portal (Whisper.cpp) feeding into vibe vector
- [ ] Add TTS output gated by Nova/Simona spike patterns
