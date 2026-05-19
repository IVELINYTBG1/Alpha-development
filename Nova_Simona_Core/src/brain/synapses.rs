// src/brain/synapses.rs
// ======================
// Synaptic weight matrices — the "axons" connecting populations.
//
// BIOLOGY:
//   A synapse is a weighted connection between a pre-synaptic neuron (sender)
//   and a post-synaptic neuron (receiver). The weight encodes the strength
//   of the connection. Here we implement a dense projection matrix W [out × in]
//   so that the output current for each post-synaptic neuron is:
//
//       I_post = W · x_pre      (matrix-vector multiply)
//
// HEBBIAN PLASTICITY (stub):
//   "Neurons that fire together wire together."
//   The `hebbian_update` method encodes this: if pre and post both fired,
//   increase the weight. This is the foundation of learned synaptic density
//   replacing a static vector database.
//
// SPARSITY:
//   Weights are initialized small (Gaussian, σ=0.1) so that only meaningful
//   correlated activity drives post-synaptic neurons above threshold.
//   Silent neurons contribute exactly zero current.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// A fully-connected (dense) synaptic projection.
/// Stores weights as a flat row-major matrix: W[out_idx * in_dim + in_idx]
#[derive(Debug, Clone)]
pub struct Projection {
    pub in_dim:  usize,
    pub out_dim: usize,
    weights: Vec<f32>,

    /// Learning rate for Hebbian updates.
    pub lr: f32,
    /// Weight decay (prevents unbounded growth).
    pub decay: f32,
}

impl Projection {
    /// Create a new projection with Gaussian-initialized weights.
    ///
    /// σ=0.10 for conservative projections (Nova),
    /// σ=0.20 for excitable projections (Simona — she over-reacts).
    pub fn new(in_dim: usize, out_dim: usize, std_dev: f32, lr: f32) -> Self {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0_f32, std_dev).unwrap();
        let weights = (0..in_dim * out_dim)
            .map(|_| dist.sample(&mut rng))
            .collect();

        Self { in_dim, out_dim, weights, lr, decay: 0.9999 }
    }

    /// Forward pass: compute post-synaptic currents from pre-synaptic spikes.
    /// Returns a Vec<f32> of length `out_dim`.
    pub fn forward(&self, pre_spikes: &[f32]) -> Vec<f32> {
        assert_eq!(pre_spikes.len(), self.in_dim,
            "Input spike vector length {} ≠ projection in_dim {}", pre_spikes.len(), self.in_dim);

        (0..self.out_dim)
            .map(|o| {
                let row_start = o * self.in_dim;
                pre_spikes.iter().enumerate()
                    .map(|(i, &s)| self.weights[row_start + i] * s)
                    .sum::<f32>()
            })
            .collect()
    }

    /// Hebbian weight update:
    ///   ΔW[o,i] = lr × pre_spikes[i] × post_spikes[o]
    ///
    /// Only called when BOTH neurons fire — sparse, cheap.
    pub fn hebbian_update(&mut self, pre_spikes: &[f32], post_spikes: &[f32]) {
        for o in 0..self.out_dim {
            if post_spikes[o] < 0.5 { continue; } // post silent → skip entire row
            let row_start = o * self.in_dim;
            for i in 0..self.in_dim {
                if pre_spikes[i] < 0.5 { continue; } // pre silent → skip
                self.weights[row_start + i] =
                    self.weights[row_start + i] * self.decay
                    + self.lr * pre_spikes[i] * post_spikes[o];
            }
        }
    }

    /// Synaptic density: fraction of non-negligible weights (|w| > 0.01).
    /// Analogous to "how many real connections exist."
    pub fn density(&self) -> f32 {
        let active = self.weights.iter().filter(|&&w| w.abs() > 0.01).count();
        active as f32 / self.weights.len() as f32
    }
}
