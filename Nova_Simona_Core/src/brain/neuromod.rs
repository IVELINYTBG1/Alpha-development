// src/brain/neuromod.rs
// ======================
// The Neuromodulatory Field — Phill's voltage reshapes Nova and Simona.
//
// BIOLOGY:
//   In the brain, neuromodulators (dopamine, serotonin, norepinephrine)
//   are released by small nuclei and flood large cortical regions,
//   globally adjusting excitability. Phill IS this system.
//
// PHYSICS:
//   Let V_p = Phill's mean membrane voltage, clamped to [0, 1].
//
//   Nova (stability / planning):
//     θ_nova(t) = θ_nova_base + α × V_p
//     → High Phill energy raises Nova's bar → she fires LESS → stability
//
//   Simona (curiosity / impulse):
//     θ_simona(t) = max(θ_floor, θ_simona_base − β × V_p)
//     → High Phill energy lowers Simona's bar → she fires MORE → chaos
//
//   Nova β modulation (synaptic scaling):
//     β_nova(t) = β_nova_base + γ × V_p   (clamped to 0.99)
//     → Under stress Nova's memory lengthens → she holds context longer
//
//   Simona β modulation:
//     β_simona(t) = β_simona_base − δ × V_p  (clamped to 0.30)
//     → Under stress Simona's memory shortens → pure reactive mode

/// Parameters that govern the emotional physics of the system.
/// Changing these changes who Nova and Simona *are*.
#[derive(Debug, Clone)]
pub struct NeuromodParams {
    // Base thresholds (calm environment, V_p = 0)
    pub nova_thr_base:    f32,   // 1.20
    pub simona_thr_base:  f32,   // 0.50

    // Base leak rates
    pub nova_beta_base:   f32,   // 0.90
    pub simona_beta_base: f32,   // 0.60

    // Coupling constants
    pub alpha: f32,  // Nova threshold gain    (0.40)
    pub beta:  f32,  // Simona threshold drop  (0.35)
    pub gamma: f32,  // Nova β gain            (0.05)
    pub delta: f32,  // Simona β drop          (0.15)

    // Safety floors
    pub simona_thr_floor:  f32,  // 0.10
    pub simona_beta_floor: f32,  // 0.30
    pub nova_beta_ceil:    f32,  // 0.99
}

impl Default for NeuromodParams {
    fn default() -> Self {
        Self {
            nova_thr_base:    1.20,
            simona_thr_base:  0.50,
            nova_beta_base:   0.90,
            simona_beta_base: 0.60,
            alpha: 0.40,
            beta:  0.35,
            gamma: 0.05,
            delta: 0.15,
            simona_thr_floor:  0.10,
            simona_beta_floor: 0.30,
            nova_beta_ceil:    0.99,
        }
    }
}

/// The computed modulation output for one tick.
#[derive(Debug, Clone, Copy)]
pub struct ModulationState {
    pub nova_threshold:    f32,
    pub simona_threshold:  f32,
    pub nova_beta:         f32,
    pub simona_beta:       f32,
    pub v_phill:           f32,
}

/// Compute the full modulation state from Phill's current voltage.
///
/// This function is the mathematical heart of the Phill→{Nova,Simona} bond.
/// It is called once per tick, in pure Rust, before any neuron steps.
pub fn compute_modulation(v_phill_raw: f32, p: &NeuromodParams) -> ModulationState {
    // Clamp Phill voltage to [0, 1] — it is a gain signal, not a spike count
    let v = v_phill_raw.clamp(0.0, 1.0);

    ModulationState {
        v_phill: v,

        // Nova tightens under stress
        nova_threshold: p.nova_thr_base + p.alpha * v,

        // Simona loosens under stress (never below floor)
        simona_threshold: (p.simona_thr_base - p.beta * v)
                            .max(p.simona_thr_floor),

        // Nova's memory lengthens under stress (clamped)
        nova_beta: (p.nova_beta_base + p.gamma * v)
                    .min(p.nova_beta_ceil),

        // Simona's memory shortens under stress (clamped)
        simona_beta: (p.simona_beta_base - p.delta * v)
                      .max(p.simona_beta_floor),
    }
}
