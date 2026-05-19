// src/brain/neurons.rs
// =====================
// The fundamental unit of the network: the Leaky Integrate-and-Fire (LIF) neuron.
//
// PHYSICS:
//   Every timestep, the membrane voltage V decays by factor β (the "leak"):
//
//       V(t) = β × V(t-1) + I(t)
//
//   When V crosses the threshold θ, the neuron emits a spike (binary 1)
//   and resets to 0 (hard reset). Between spikes it is completely dark —
//   zero compute, zero power draw. This is the source of sparsity.
//
//   The threshold θ is NOT fixed. It is injected each tick by the
//   Neuromodulator (phill.rs), making every neuron dynamically sensitive
//   to the emotional state of the shared affective core.

use std::fmt;

/// A single Leaky Integrate-and-Fire neuron.
/// Owns its membrane voltage state across ticks.
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    /// Membrane leak factor (0..1).  
    /// 0.95 = very slow decay (long memory).  
    /// 0.60 = fast decay (impulsive, short memory).
    pub beta: f32,

    /// Current firing threshold. Updated externally by the neuromodulator.
    pub threshold: f32,

    /// The membrane voltage — the neuron's "charge."
    /// Private: only the neuron integrates it, outsiders only see spikes.
    mem: f32,

    /// Total spike count since last reset (for introspection).
    pub total_spikes: u64,
}

impl LIFNeuron {
    pub fn new(beta: f32, threshold: f32) -> Self {
        Self { beta, threshold, mem: 0.0, total_spikes: 0 }
    }

    /// One timestep of physics.
    ///
    /// 1. Leak: decay the membrane by β
    /// 2. Integrate: add the incoming synaptic current
    /// 3. Fire & reset: if V ≥ θ emit spike, clamp V to 0
    ///
    /// Returns `true` if the neuron fired.
    pub fn step(&mut self, input_current: f32) -> bool {
        // Leak + integrate
        self.mem = self.beta * self.mem + input_current;

        // Fire and hard-reset
        if self.mem >= self.threshold {
            self.mem = 0.0;
            self.total_spikes += 1;
            true
        } else {
            false
        }
    }

    /// Read membrane voltage without stepping (for neuromodulation).
    #[inline]
    pub fn voltage(&self) -> f32 { self.mem }

    /// Hard reset — dreamless sleep.
    pub fn reset(&mut self) {
        self.mem = 0.0;
        self.total_spikes = 0;
    }
}

/// A layer of N homogeneous LIF neurons sharing the same β and base threshold.
/// Each neuron receives its own independent synaptic current.
#[derive(Debug, Clone)]
pub struct LIFLayer {
    pub neurons: Vec<LIFNeuron>,
    pub size: usize,
}

impl LIFLayer {
    pub fn new(size: usize, beta: f32, threshold: f32) -> Self {
        Self {
            neurons: (0..size).map(|_| LIFNeuron::new(beta, threshold)).collect(),
            size,
        }
    }

    /// Step all neurons. Returns spike vector (1.0 = fired, 0.0 = silent).
    pub fn step(&mut self, currents: &[f32]) -> Vec<f32> {
        assert_eq!(currents.len(), self.size, "Current vector size mismatch");
        self.neurons.iter_mut().zip(currents.iter())
            .map(|(n, &i)| if n.step(i) { 1.0 } else { 0.0 })
            .collect()
    }

    /// Apply a new threshold to every neuron (neuromodulation).
    pub fn set_threshold(&mut self, new_thr: f32) {
        for n in &mut self.neurons {
            n.threshold = new_thr;
        }
    }

    /// Mean membrane voltage across the layer.
    pub fn mean_voltage(&self) -> f32 {
        self.neurons.iter().map(|n| n.voltage()).sum::<f32>() / self.size as f32
    }

    /// Total spikes emitted this tick.
    pub fn spike_count(spikes: &[f32]) -> usize {
        spikes.iter().filter(|&&s| s > 0.5).count()
    }

    pub fn total_spikes(&self) -> u64 {
        self.neurons.iter().map(|n| n.total_spikes).sum()
    }

    pub fn reset(&mut self) {
        for n in &mut self.neurons { n.reset(); }
    }
}

impl fmt::Display for LIFLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LIFLayer(n={}, β={:.2}, thr={:.3}, V̄={:.4})",
               self.size,
               self.neurons[0].beta,
               self.neurons[0].threshold,
               self.mean_voltage())
    }
}
