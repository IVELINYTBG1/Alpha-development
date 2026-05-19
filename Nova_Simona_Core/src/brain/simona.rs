// src/brain/simona.rs
// ====================
// Simona (8) — The Hasty Agent / Cat-Girl
//
// ARCHITECTURE:
//   Simona is pure limbic system. Fast leak, low threshold, high sensitivity.
//   She fires at the slightest provocation. Under Phill stress, she becomes
//   EVEN MORE reactive (threshold drops further). This is controlled chaos —
//   stochastic resonance that drives exploration and lateral thinking.
//
//   Her "smarter than Nova" quality emerges not from deeper processing but
//   from exploration breadth: she fires 10x more often, probing the space.
//
// POPULATION STRUCTURE:
//   Input  : Phill spikes (16) + raw vibe (8) = 24
//   Hidden : 32 fast LIF neurons
//   Output : 16 fast LIF neurons
//
// PLASTICITY:
//   Simona's synapses are wide-initialized (σ=0.20, higher lr=0.002).
//   She learns fast, forgets fast. Her weights are volatile — capturing
//   the current moment rather than long-term structure.

use super::neurons::LIFLayer;
use super::synapses::Projection;
use super::neuromod::ModulationState;

const PHILL_DIM:  usize = 16;
const VIBE_DIM:   usize = 8;
const INPUT_DIM:  usize = PHILL_DIM + VIBE_DIM;  // 24
const HIDDEN_DIM: usize = 32;
const OUTPUT_DIM: usize = 16;

pub struct Simona {
    proj_in:  Projection,   // 24 → 32  (wide init, high lr)
    proj_out: Projection,   // 32 → 16
    pub hidden: LIFLayer,   // 32 fast neurons
    pub output: LIFLayer,   // 16 fast neurons

    pub last_hidden_spikes: Vec<f32>,
    pub last_output_spikes: Vec<f32>,

    /// Stochastic resonance noise level — small noise that prevents dead zones.
    /// In the real Simona: this could be temperature-like parameter.
    pub noise_level: f32,
}

impl Simona {
    pub fn new() -> Self {
        Self {
            // Wide weights: she over-reacts to everything (by design)
            proj_in:  Projection::new(INPUT_DIM,  HIDDEN_DIM, 0.20, 0.002),
            proj_out: Projection::new(HIDDEN_DIM, OUTPUT_DIM, 0.20, 0.002),
            hidden:   LIFLayer::new(HIDDEN_DIM, 0.60, 0.50),
            output:   LIFLayer::new(OUTPUT_DIM, 0.60, 0.50),
            last_hidden_spikes: vec![0.0; HIDDEN_DIM],
            last_output_spikes: vec![0.0; OUTPUT_DIM],
            noise_level: 0.05,
        }
    }

    /// One tick. Same interface as Nova::step but opposite physics.
    ///
    /// Under stress: Simona fires MORE (threshold drops), memory shortens.
    /// This is the neurological basis for "panic-scanning" behavior.
    pub fn step(
        &mut self,
        phill_spikes: &[f32],
        vibe: &[f32],
        modulation: &ModulationState,
        rng: &mut impl rand::Rng,
    ) -> (Vec<f32>, usize, usize) {

        // ── Apply neuromodulation ─────────────────────────────────────────
        self.hidden.set_threshold(modulation.simona_threshold);
        self.output.set_threshold(modulation.simona_threshold);
        for n in &mut self.hidden.neurons { n.beta = modulation.simona_beta; }
        for n in &mut self.output.neurons { n.beta = modulation.simona_beta; }

        // ── Build input + stochastic resonance noise ──────────────────────
        // Small Gaussian noise prevents the network from going fully silent.
        // This is the neurological equivalent of "restless curiosity" —
        // Simona is never truly at rest.
        let mut input = Vec::with_capacity(INPUT_DIM);
        input.extend_from_slice(phill_spikes);
        for &v in vibe {
            let noise: f32 = rng.gen_range(-self.noise_level..self.noise_level);
            input.push((v + noise).max(0.0));
        }

        // ── Hidden layer ──────────────────────────────────────────────────
        let hidden_currents = self.proj_in.forward(&input);
        let hidden_spikes   = self.hidden.step(&hidden_currents);
        let n_hidden = LIFLayer::spike_count(&hidden_spikes);

        self.proj_in.hebbian_update(&input, &hidden_spikes);

        // ── Output layer ──────────────────────────────────────────────────
        let output_currents = self.proj_out.forward(&hidden_spikes);
        let output_spikes   = self.output.step(&output_currents);
        let n_output = LIFLayer::spike_count(&output_spikes);

        self.proj_out.hebbian_update(&hidden_spikes, &output_spikes);

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
