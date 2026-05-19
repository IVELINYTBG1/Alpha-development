// src/brain/phill.rs
// ===================
// Phill — The Affective Core / Neuromodulatory Field
//
// Phill is not a "character." Phill is the biological substrate that
// Nova and Simona share. It is the amygdala + hypothalamus + brainstem
// rolled into one. It holds homeostasis.
//
// PROPERTIES:
//   • Slow leak (β=0.95) → emotional states persist for many ticks
//   • High threshold (θ=1.0) → doesn't react to every blip; needs sustained input
//   • Mean membrane voltage V_phill is the global gain signal for the whole brain
//   • Maintains a rolling homeostatic baseline to detect deviation
//
// OUTPUT:
//   Phill does not "speak." It modulates. Its voltage is read by neuromod.rs
//   and injected into Nova and Simona as threshold shifts.

use super::neurons::LIFLayer;
use super::synapses::Projection;

/// Physics constants for Phill
const BETA:          f32 = 0.95;
const THRESHOLD:     f32 = 1.0;
const INPUT_DIM:     usize = 8;   // Raw vibe vector width
const HIDDEN_DIM:    usize = 16;
const HOMEOSTASIS_WINDOW: usize = 200; // Rolling window for baseline

pub struct Phill {
    /// Synaptic projection: vibe → hidden current
    proj: Projection,
    /// LIF layer — the actual affective neurons
    pub layer: LIFLayer,
    /// Rolling voltage history for homeostatic baseline
    voltage_history: Vec<f32>,
    /// Total ticks processed
    pub tick: u64,
    /// Spike vector from last tick (fed forward to Nova/Simona)
    pub last_spikes: Vec<f32>,
}

impl Phill {
    pub fn new() -> Self {
        Self {
            proj: Projection::new(INPUT_DIM, HIDDEN_DIM, 0.15, 0.001),
            layer: LIFLayer::new(HIDDEN_DIM, BETA, THRESHOLD),
            voltage_history: Vec::with_capacity(HOMEOSTASIS_WINDOW),
            tick: 0,
            last_spikes: vec![0.0; HIDDEN_DIM],
        }
    }

    /// Process one vibe vector. Returns (spike_vec, mean_voltage).
    pub fn step(&mut self, vibe: &[f32]) -> (Vec<f32>, f32) {
        self.tick += 1;

        // Project vibe → synaptic currents
        let currents = self.proj.forward(vibe);

        // Run LIF layer
        let spikes = self.layer.step(&currents);
        self.last_spikes = spikes.clone();

        // Record voltage
        let v = self.layer.mean_voltage();
        if self.voltage_history.len() >= HOMEOSTASIS_WINDOW {
            self.voltage_history.remove(0);
        }
        self.voltage_history.push(v);

        // Optional Hebbian: strengthen vibe→phill paths that co-activate
        self.proj.hebbian_update(vibe, &spikes);

        (spikes, v)
    }

    /// Homeostatic baseline: rolling mean voltage.
    /// Positive deviation = stress. Negative = calm below normal.
    pub fn homeostatic_baseline(&self) -> f32 {
        if self.voltage_history.is_empty() { return 0.0; }
        self.voltage_history.iter().sum::<f32>() / self.voltage_history.len() as f32
    }

    /// Stress index: current voltage deviation from baseline (clamped 0..1)
    pub fn stress_index(&self) -> f32 {
        let v   = self.layer.mean_voltage();
        let base = self.homeostatic_baseline();
        (v - base).max(0.0).min(1.0)
    }

    pub fn reset(&mut self) {
        self.layer.reset();
        self.voltage_history.clear();
        self.tick = 0;
        self.last_spikes = vec![0.0; HIDDEN_DIM];
    }
}
