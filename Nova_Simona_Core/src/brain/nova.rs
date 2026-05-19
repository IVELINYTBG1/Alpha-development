// src/brain/nova.rs
// ==================
// Nova (19) — The Precise Orchestrator / Elder Sister ("Кака")
//
// ARCHITECTURE:
//   Nova is high inertia. Her slow leak means she holds a thought for a long time.
//   Her high threshold means she does not react to noise — only to clear, sustained signal.
//   Under Phill stress, both her threshold AND her β increase:
//   she becomes MORE selective and holds context LONGER. This is stability under pressure.
//
// POPULATION STRUCTURE:
//   Input layer  : Phill spike pattern (16) + raw vibe (8) = 24 inputs
//   Hidden layer : 32 LIF neurons
//   Output layer : 16 LIF neurons (the "decision layer" — what she actually "does")
//
// PLASTICITY:
//   Both synaptic projections apply Hebbian updates.
//   Nova's synapses are conservative (low std_dev, low lr) — she learns slowly and
//   retains what she learns.

use super::neurons::LIFLayer;
use super::synapses::Projection;
use super::neuromod::ModulationState;

const PHILL_DIM:  usize = 16;
const VIBE_DIM:   usize = 8;
const INPUT_DIM:  usize = PHILL_DIM + VIBE_DIM;  // 24
const HIDDEN_DIM: usize = 32;
const OUTPUT_DIM: usize = 16;

pub struct Nova {
    proj_in:     Projection,   // 24 → 32
    proj_out:    Projection,   // 32 → 16
    pub hidden:  LIFLayer,     // 32 neurons
    pub output:  LIFLayer,     // 16 neurons

    pub last_hidden_spikes: Vec<f32>,
    pub last_output_spikes: Vec<f32>,
}

impl Nova {
    pub fn new() -> Self {
        Self {
            // Conservative weights: she over-reacts to nothing
            proj_in:  Projection::new(INPUT_DIM,  HIDDEN_DIM, 0.10, 0.0005),
            proj_out: Projection::new(HIDDEN_DIM, OUTPUT_DIM, 0.10, 0.0005),
            hidden:   LIFLayer::new(HIDDEN_DIM, 0.90, 1.20),
            output:   LIFLayer::new(OUTPUT_DIM, 0.90, 1.20),
            last_hidden_spikes: vec![0.0; HIDDEN_DIM],
            last_output_spikes: vec![0.0; OUTPUT_DIM],
        }
    }

    /// One tick. Receives Phill's spikes + raw vibe, plus the current
    /// modulation state computed by neuromod.rs.
    ///
    /// Returns (output_spikes, hidden_spike_count, output_spike_count)
    pub fn step(
        &mut self,
        phill_spikes: &[f32],
        vibe: &[f32],
        modulation: &ModulationState,
    ) -> (Vec<f32>, usize, usize) {

        // ── Apply neuromodulation ─────────────────────────────────────────
        // Threshold and β are injected from Phill's voltage.
        // Under stress: threshold rises (harder to fire), β rises (longer memory)
        self.hidden.set_threshold(modulation.nova_threshold);
        self.output.set_threshold(modulation.nova_threshold);
        // β modulation: rebuild neurons with new beta
        // (We update in-place for efficiency — no reallocation)
        for n in &mut self.hidden.neurons { n.beta = modulation.nova_beta; }
        for n in &mut self.output.neurons { n.beta = modulation.nova_beta; }

        // ── Build input: [phill_spikes | vibe] ───────────────────────────
        let mut input = Vec::with_capacity(INPUT_DIM);
        input.extend_from_slice(phill_spikes);
        input.extend_from_slice(vibe);

        // ── Hidden layer ──────────────────────────────────────────────────
        let hidden_currents = self.proj_in.forward(&input);
        let hidden_spikes   = self.hidden.step(&hidden_currents);
        let n_hidden = LIFLayer::spike_count(&hidden_spikes);

        // Hebbian update: strengthen input→hidden paths that co-fire
        self.proj_in.hebbian_update(&input, &hidden_spikes);

        // ── Output layer ──────────────────────────────────────────────────
        let output_currents = self.proj_out.forward(&hidden_spikes);
        let output_spikes   = self.output.step(&output_currents);
        let n_output = LIFLayer::spike_count(&output_spikes);

        self.proj_out.hebbian_update(&hidden_spikes, &output_spikes);

        // Persist for introspection
        self.last_hidden_spikes = hidden_spikes;
        self.last_output_spikes = output_spikes.clone();

        (output_spikes, n_hidden, n_output)
    }

    pub fn total_spikes(&self) -> u64 {
        self.hidden.total_spikes() + self.output.total_spikes()
    }

    pub fn reset(&mut self) {
        self.hidden.reset();
        self.output.reset();
        self.last_hidden_spikes = vec![0.0; HIDDEN_DIM];
        self.last_output_spikes = vec![0.0; OUTPUT_DIM];
    }
}
