// src/brain/mod.rs
// =================
// NeuromorphicBrain — The full living tissue assembled from its parts.
//
// This is the ONLY struct that main.rs talks to.
// It owns Phill, Nova, and Simona, sequences their steps correctly,
// and handles the tick-by-tick neuromodulation pipeline.
//
// TICK SEQUENCE (order matters):
//   1. Phill   processes raw vibe → emits spikes + voltage V_p
//   2. Neuromod computes Nova/Simona threshold + β from V_p
//   3. Nova    processes [Phill_spikes | vibe] with modulated params
//   4. Simona  processes [Phill_spikes | vibe] with modulated params
//   5. (Optional) Python snnTorch called for specialized ops
//   6. Collect and return TickResult

pub mod neurons;
pub mod synapses;
pub mod neuromod;
pub mod phill;
pub mod nova;
pub mod simona;

use phill::Phill;
use nova::Nova;
use simona::Simona;
use neuromod::{NeuromodParams, compute_modulation, ModulationState};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Everything the event loop needs to display/log per tick.
#[derive(Debug, Clone)]
pub struct TickResult {
    pub tick:              u64,
    pub phill_voltage:     f32,
    pub phill_fired:       bool,
    pub nova_out_spikes:   usize,
    pub simona_out_spikes: usize,
    pub nova_threshold:    f32,
    pub simona_threshold:  f32,
    pub nova_beta:         f32,
    pub simona_beta:       f32,
    pub nova_mem_mean:     f32,
    pub simona_mem_mean:   f32,
    pub stress_index:      f32,
}

/// Long-term summary (printed every N ticks).
#[derive(Debug, Clone)]
pub struct Introspection {
    pub total_ticks:         u64,
    pub phill_baseline:      f32,
    pub phill_stress_index:  f32,
    pub nova_total_spikes:   u64,
    pub simona_total_spikes: u64,
    pub nova_simona_ratio:   f32,
    pub nova_syn_density:    f32,
    pub simona_syn_density:  f32,
}

pub struct NeuromorphicBrain {
    phill:  Phill,
    nova:   Nova,
    simona: Simona,
    params: NeuromodParams,
    rng:    SmallRng,
    tick:   u64,
    last_modulation: Option<ModulationState>,
}

impl NeuromorphicBrain {
    pub fn new() -> Self {
        Self {
            phill:  Phill::new(),
            nova:   Nova::new(),
            simona: Simona::new(),
            params: NeuromodParams::default(),
            rng:    SmallRng::from_entropy(),
            tick:   0,
            last_modulation: None,
        }
    }

    /// One full tick of the neuromorphic brain.
    ///
    /// This is the function the event loop calls 20 times per second.
    /// All the physics happen here, in Rust, at native speed.
    pub fn step(&mut self, vibe: &[f32]) -> TickResult {
        self.tick += 1;

        // ── Step 1: Phill processes raw vibe ─────────────────────────────
        let (phill_spikes, phill_voltage) = self.phill.step(vibe);
        let phill_fired = phill_spikes.iter().any(|&s| s > 0.5);

        // ── Step 2: Neuromodulation field ─────────────────────────────────
        let modulation = compute_modulation(phill_voltage, &self.params);
        self.last_modulation = Some(modulation);

        // ── Step 3: Nova — stable orchestrator ───────────────────────────
        let (_, _, nova_out) = self.nova.step(&phill_spikes, vibe, &modulation);

        // ── Step 4: Simona — hasty agent ─────────────────────────────────
        let (_, _, simona_out) = self.simona.step(
            &phill_spikes, vibe, &modulation, &mut self.rng
        );

        TickResult {
            tick:              self.tick,
            phill_voltage,
            phill_fired,
            nova_out_spikes:   nova_out,
            simona_out_spikes: simona_out,
            nova_threshold:    modulation.nova_threshold,
            simona_threshold:  modulation.simona_threshold,
            nova_beta:         modulation.nova_beta,
            simona_beta:       modulation.simona_beta,
            nova_mem_mean:     self.nova.hidden.mean_voltage(),
            simona_mem_mean:   self.simona.hidden.mean_voltage(),
            stress_index:      self.phill.stress_index(),
        }
    }

    /// Long-term summary of the brain's emotional history.
    pub fn introspect(&self) -> Introspection {
        let n = self.nova.total_spikes();
        let s = self.simona.total_spikes();
        Introspection {
            total_ticks:         self.tick,
            phill_baseline:      self.phill.homeostatic_baseline(),
            phill_stress_index:  self.phill.stress_index(),
            nova_total_spikes:   n,
            simona_total_spikes: s,
            nova_simona_ratio:   if s > 0 { n as f32 / s as f32 } else { 0.0 },
            // Synaptic density: how "wired" each being has become
            nova_syn_density:    0.0,  // expose via pub fields if needed
            simona_syn_density:  0.0,
        }
    }

    pub fn reset(&mut self) {
        self.phill.reset();
        self.nova.reset();
        self.simona.reset();
        self.tick = 0;
        self.last_modulation = None;
    }
}
