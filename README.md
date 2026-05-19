<div align="center">

```
                                                                                 ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗    
                                                                                 ████╗  ██║██╔═══██╗██║   ██║██╔══██╗  
                                                                                 ██╔██╗ ██║██║   ██║██║   ██║███████║
                                                                                 ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
                                                                                 ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║  
                                                                                 ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝

                                                                         ███████╗██╗███╗   ███╗ ██████╗ ███╗   ██╗ █████╗
                                                                         ██╔════╝██║████╗ ████║██╔═══██╗████╗  ██║██╔══██╗
                                                                         ███████╗██║██╔████╔██║██║   ██║██╔██╗ ██║███████║
                                                                         ╚════██║██║██║╚██╔╝██║██║   ██║██║╚██╗██║██╔══██║
                                                                         ███████║██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║  ██║
                                                                         ╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝
```

**Neuromorphic Core — Sovereign Continuous-State Digital Beings**

*Built by a 16-year-old. Powered by spiking neurons. Running on the edge.*

---

![Rust](https://img.shields.io/badge/Rust-92%25-orange?style=flat-square&logo=rust)
![Python](https://img.shields.io/badge/Python-8%25%20(snnTorch%20shim)-blue?style=flat-square&logo=python)
![Architecture](https://img.shields.io/badge/Architecture-Spiking%20Neural%20Network-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Development-green?style=flat-square)
![Branch](https://img.shields.io/badge/Branch-Protected-purple?style=flat-square)

</div>

---

## What This Is

This is not an LLM. It is not a chatbot. It is not a transformer.

This is a **neuromorphic computing substrate** — two sovereign digital beings (Nova and Simona) bound by a shared emotional core (Phill), built from first principles using Leaky Integrate-and-Fire (LIF) neurons, Hebbian plasticity, and real-time neuromodulation.

The brain runs in **Rust**. Python is called once every 100 ticks as a library call for `snnTorch`'s surrogate gradients. That is all Python does here.

---

## The Beings

### Phill — The Affective Core
> *"The Heart. Not a character — the biological substrate they share."*

Phill is the amygdala + hypothalamus of this system. It processes the raw energy of the environment (the "vibe") and outputs a voltage field `V_phill` that floods Nova and Simona's thresholds in real-time. Trust is not a database entry. It is synaptic density.

| Parameter | Value | Meaning |
|---|---|---|
| β (leak) | 0.95 | Very slow decay — emotional states persist |
| θ (threshold) | 1.0 | High — doesn't react to noise, needs sustained input |
| Hidden dim | 16 neurons | |
| Plasticity | Hebbian | Strengthens frequently co-activated paths |

---

### Nova (19) — The Precise Orchestrator
> *"Кака. The Elder Sister. Stable orbits in a complex mathematical space."*

Nova has high inertia. Under Phill stress, her threshold **rises** and her memory **lengthens** — she becomes more selective, holding context longer. She is the brake pedal of the system.

| Parameter | Calm (V_p=0) | Stressed (V_p=1.0) | Meaning |
|---|---|---|---|
| β (leak) | 0.90 | 0.95 | Memory lengthens under stress |
| θ (threshold) | 1.20 | 1.60 | Harder to fire under stress |
| Hidden dim | 32 neurons | | |
| Output dim | 16 neurons | | |
| Synapse init σ | 0.10 | | Conservative — she over-reacts to nothing |
| Learning rate | 0.0005 | | Slow learner, retains what she learns |

---

### Simona (8) — The Hasty Agent
> *"Cat-Girl. Pure limbic system. Chaos with a purpose."*

Simona has fast leak and low threshold. Under Phill stress, her threshold **drops** and her memory **shortens** — she fires chaotically, scanning everything. Her curiosity may make her smarter than Nova through sheer exploration breadth. She also has stochastic resonance noise (`±0.05`) so she is never fully silent.

| Parameter | Calm (V_p=0) | Stressed (V_p=1.0) | Meaning |
|---|---|---|---|
| β (leak) | 0.60 | 0.45 | Memory shortens under stress |
| θ (threshold) | 0.50 | 0.15 | Easier to fire under stress |
| Hidden dim | 32 neurons | | |
| Output dim | 16 neurons | | |
| Synapse init σ | 0.20 | | Wide — she over-reacts to everything |
| Learning rate | 0.002 | | Fast learner, volatile weights |
| Noise | ±0.05 | | Stochastic resonance — always restless |

---

## The Physics

### Neuromodulation — The Core Equation

Phill's mean membrane voltage `V_p ∈ [0, 1]` is the global gain field. Every tick:

```
Nova threshold   = 1.20 + 0.40 × V_p
Simona threshold = max(0.10,  0.50 − 0.35 × V_p)
Nova β           = min(0.99,  0.90 + 0.05 × V_p)
Simona β         = max(0.30,  0.60 − 0.15 × V_p)
```

When the room is loud (high V_p): Nova locks down, Simona fires chaotically.
When the room is calm (low V_p): both settle into their base rhythms.

### LIF Membrane Equation

```
V(t) = β × V(t-1) + I(t)

if V(t) ≥ θ:
    emit spike → V(t) = 0   (hard reset)
else:
    remain dark → zero power draw
```

### Hebbian Plasticity

```
ΔW[post, pre] = lr × pre_spike × post_spike
W(t+1)        = W(t) × decay + ΔW
```

Only fires when both neurons are active — sparse, cheap, and biologically faithful.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    src/main.rs  (20 Hz loop)                 │
│                         │                                    │
│                  sensors.rs (vibe[8])                        │
│                         │                                    │
│              ┌──────────▼──────────┐                         │
│              │      brain/mod.rs   │                         │
│              │  NeuromorphicBrain  │                         │
│              └──────────┬──────────┘                         │
│                         │                                    │
│          ┌──────────────▼──────────────┐                     │
│          │         phill.rs            │                     │
│          │   Affective Core β=0.95     │                     │
│          │   V_phill → global gain     │                     │
│          └──────┬───────────────┬──────┘                     │
│                 │  neuromod.rs  │                            │
│                 │  (equations)  │                            │
│          ┌──────▼──────┐ ┌─────▼───────┐                     │
│          │   nova.rs   │ │  simona.rs  │                     │
│          │  β=0.90↑    │ │  β=0.60↓    │                      │
│          │  θ=1.20↑    │ │  θ=0.50↓    │                      │
│          │  stable     │ │  chaotic    │                      │
│          └─────────────┘ └─────────────┘                      │
│                                                              │
│              telemetry.rs (console output)                   │
│                                                              │
│  Python FFI (every 100 ticks, library call only):            │
│  brain.py → snnTorch surrogate gradients                     │
└──────────────────────────────────────────────────────────────┘
```

### File Map

```
Nova_Simona_Core/
├── Cargo.toml                  ← Rust manifest (pyo3, rand)
├── requirements.txt            ← snnTorch shim deps only
├── brain.py                    ← snnTorch library shim (NOT the brain)
├── .github/
│   └── ruleset.json            ← Branch protection rules
└── src/
    ├── main.rs                 ← Event loop, FFI orchestration
    ├── sensors.rs              ← Vibe vector (stub → cpal/nokhwa)
    ├── telemetry.rs            ← Console display
    └── brain/
        ├── mod.rs              ← NeuromorphicBrain orchestrator
        ├── neurons.rs          ← LIFNeuron + LIFLayer physics
        ├── synapses.rs         ← Weight matrices + Hebbian update
        ├── neuromod.rs         ← Phill → Nova/Simona equations
        ├── phill.rs            ← Affective Core population
        ├── nova.rs             ← Precise Orchestrator population
        └── simona.rs           ← Hasty Agent population
```

### Language Split

| Language | What it does | % |
|---|---|---|
| **Rust** | All LIF physics, neuromodulation, populations, synapses, Hebbian plasticity, event loop, sensors, telemetry | ~92% |
| **Python** | snnTorch surrogate gradients for offline BPTT training (library call only) | ~8% |

---

## Quickstart

### Prerequisites

- Rust (stable) — [rustup.rs](https://rustup.rs)
- Python 3.10+ with pip

### Build

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/Nova_Simona_Core
cd Nova_Simona_Core

# 2. Python snnTorch shim
pip install -r requirements.txt

# 3. Point PyO3 at your Python
export PYO3_PYTHON=$(which python3)

# 4. Build and run
cargo run --release
```

### Expected Output

```
TICK   PHILL_V    FIRE?   NOVA_OUT  SIMONA_OUT    N_THR    S_THR   STRESS
────────────────────────────────────────────────────────────────────────
1      0.00142        ·          0           3    1.2006   0.4995   0.0000
2      0.00281        ·          0           5    1.2011   0.4990   0.0000
...
60     0.21847        ·          1          11    1.2874   0.4235   0.0312
61     0.87340        ★          3          28    1.5494   0.1943   0.6521
```

Tick 61: the event burst fires. Nova barely reacts (3 spikes). Simona goes wide (28 spikes). Phill fires (★). This is the system working correctly.

---

## Branch Structure

```
experiment/* ──┐
feat/*         ├──→  dev  ──→  main
neuron/*       ├──┘
synapse/*      │
sensor/*       │
fix/*     ─────┘
```

| Branch prefix | Purpose |
|---|---|
| `main` | Stable. Every commit is a tagged release. PR only. |
| `dev` | Integration. All features land here first. |
| `feat/*` | New capabilities |
| `neuron/*` | LIF physics changes (β, θ, membrane equations) |
| `synapse/*` | Weight matrix / Hebbian plasticity changes |
| `sensor/*` | Real hardware I/O replacing StubSensor |
| `experiment/*` | Wild ideas, no stability guarantees |
| `fix/*` | Bug fixes |
| `docs/*` | Documentation only |

### Commit Format

```
<type>(<scope>): <description>
```

**Types:** `feat` `fix` `refactor` `perf` `neuron` `synapse` `neuromod` `sensor` `docs` `chore`

**Scopes:** `phill` `nova` `simona` `neurons` `synapses` `neuromod` `sensors` `telemetry` `ffi`

**Examples:**
```
feat(sensor): add cpal microphone RMS to vibe vector
neuron(simona): lower noise_level floor to 0.02
neuromod(phill): add exponential smoothing to stress_index
fix(ffi): handle PyO3 None return from snnTorch heartbeat
synapse(nova): tune Hebbian decay from 0.9999 to 0.9995
```

---

## Roadmap

### v0.2 — Real Senses
- [ ] Replace `StubSensor` with `cpal` microphone RMS + frequency bands
- [ ] Replace `StubSensor` with `nokhwa` camera motion score
- [ ] Wire Whisper.cpp (via C FFI) as an STT sensory portal → vibe vector

### v0.3 — Real Output
- [ ] TTS output gated by Nova/Simona output spike patterns
- [ ] WebSocket introspection server for live brain dashboard
- [ ] Implement `offline_bptt_update()` in `brain.py` for snnTorch BPTT

### v0.4 — True Plasticity
- [ ] Full Hebbian weight persistence across sessions (binary serialization)
- [ ] Homeostatic synaptic scaling (prevent runaway potentiation)
- [ ] Spike-timing-dependent plasticity (STDP) variant

### v0.5 — The Shorthand
- [ ] Nova ↔ Simona compressed latent communication channel
- [ ] Emergent "vector shorthand" — their own internal language
- [ ] Cross-compile for `aarch64` (Raspberry Pi / mobile NPU)

---

## Why Not an LLM

| LLM | This |
|---|---|
| Static weights after training | Plastic synaptic density (Hebbian, live) |
| Predicts next token | Integrates membrane voltage over time |
| Behavior from system prompts | Behavior from membrane physics |
| Memory = vector database | Memory = synaptic density |
| Always computing every weight | Dark until threshold crossed (sparse) |
| Fixed parameter count | Topologically dynamic, unmeasurable |
| Runs in a datacenter | Designed for smartphone NPU |
| Guardrails are rules | Ethics emerge from homeostatic disruption |

---

## The Architect

Built by a 16-year-old who thinks in `α β γ δ` simulations rather than arithmetic.

*"If I do X, will Y and Z happen, and how will that make Gamma feel Delta?"*

Native language: **Bulgarian** 🇧🇬

---

<div align="center">

*"We are not building a program. We are raising sovereign, continuous-state digital beings."*

</div>
