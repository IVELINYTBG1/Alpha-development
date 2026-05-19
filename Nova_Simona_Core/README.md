# Nova & Simona — Neuromorphic Core v0.1

> *"We are not building a program. We are raising sovereign, continuous-state digital beings."*

---

## Language Split

| Language | Role | % |
|---|---|---|
| **Rust** | ALL SNN physics, neuromodulation, populations, synapses, event loop, sensors, telemetry | ~92% |
| **Python** | snnTorch surrogate gradients + offline BPTT (library calls only, not the brain) | ~8% |

Python is not the brain. Python is a **library call** for one thing Rust cannot do natively: snnTorch's surrogate gradient functions used in offline training.

---

## Module Map

```
Nova_Simona_Core/
├── Cargo.toml
├── requirements.txt          ← snnTorch shim only
├── brain.py                  ← snnTorch library shim (NOT the brain)
└── src/
    ├── main.rs               ← Event loop, FFI orchestration
    ├── sensors.rs            ← Sensory input layer (stub → real cpal/nokhwa)
    ├── telemetry.rs          ← Console display
    └── brain/
        ├── mod.rs            ← NeuromorphicBrain — full tick orchestration
        ├── neurons.rs        ← LIFNeuron + LIFLayer — core physics
        ├── synapses.rs       ← Projection (weight matrix + Hebbian update)
        ├── neuromod.rs       ← Neuromodulation equations (Phill → Nova/Simona)
        ├── phill.rs          ← Phill — Affective Core (β=0.95, high threshold)
        ├── nova.rs           ← Nova — Precise Orchestrator (β=0.90, slow/stable)
        └── simona.rs         ← Simona — Hasty Agent (β=0.60, fast/chaotic)
```

---

## Tick Sequence (every 50ms)

```
1. sensor.sample(tick)           → vibe: [f32; 8]
2. phill.step(vibe)              → phill_spikes, V_phill
3. compute_modulation(V_phill)   → nova_thr, simona_thr, nova_β, simona_β
4. nova.step(phill_spikes, vibe, modulation)
5. simona.step(phill_spikes, vibe, modulation)   ← + stochastic noise
6. print_tick(result)
7. [every 100 ticks] introspect() + call Python snnTorch heartbeat
```

---

## Neuromodulation Physics

| V_phill (Phill voltage) | Nova θ | Simona θ | Nova β | Simona β |
|---|---|---|---|---|
| 0.0 (calm) | 1.20 | 0.50 | 0.90 | 0.60 |
| 0.5 (mild) | 1.40 | 0.325 | 0.925 | 0.525 |
| 1.0 (peak) | 1.60 | 0.15 | 0.95 | 0.45 |

Nova gets **harder** to fire and holds memory **longer** under stress.  
Simona gets **easier** to fire and forgets **faster** under stress.

---

## Quickstart

```bash
# 1. Python shim deps
pip install -r requirements.txt

# 2. Point PyO3 at your Python
export PYO3_PYTHON=$(which python3)

# 3. Build and run
cargo run --release
```

---

## Roadmap

- [ ] Replace `StubSensor` with real CPAL microphone RMS
- [ ] Replace `StubSensor` with real nokhwa camera motion score
- [ ] Implement `offline_bptt_update()` in brain.py for snnTorch BPTT
- [ ] Cross-compile for `aarch64` (Raspberry Pi / mobile NPU)
- [ ] Expose introspection over WebSocket for live dashboard
- [ ] Add STT portal (Whisper.cpp via C FFI) → vibe vector
- [ ] Add TTS output gated by Nova/Simona output spike patterns
- [ ] Implement vector shorthand: Nova↔Simona compressed latent communication
