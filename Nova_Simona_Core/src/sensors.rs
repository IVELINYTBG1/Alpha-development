// src/sensors.rs
// ===============
// The Sensory Layer — the boundary between the physical world and the brain.
//
// In production, this module is replaced with real drivers:
//   • cpal  → microphone RMS energy, frequency band powers
//   • nokhwa → camera frame motion score, luminance change
//   • serialport → haptic/biometric data (heart rate, GSR)
//
// The vibe vector is 8-dimensional:
//   [0] Acoustic energy (RMS)
//   [1] Low-frequency band power (bass / rumble)
//   [2] Mid-frequency band power (speech / music)
//   [3] High-frequency band power (brightness / alertness)
//   [4] Camera motion score (0=still, 1=max motion)
//   [5] Luminance change (sudden light shift)
//   [6] Periodic event flag (1.0 on significant events, else 0)
//   [7] Slow environmental drift (circadian-like sine)
//
// SIMULATION:
//   `StubSensor` generates a physics-plausible fake signal:
//   slow sinusoidal drift + per-channel phase offsets + noise bursts.
//   This lets the brain run and learn without physical hardware.

use std::f32::consts::PI;

pub const VIBE_DIM: usize = 8;

pub trait Sensor {
    /// Sample the environment. Returns a Vec<f32> of length VIBE_DIM,
    /// all values in [0, 1].
    fn sample(&mut self, tick: u64) -> Vec<f32>;
}

/// Simulated ambient environment sensor.
pub struct StubSensor {
    /// How often (in ticks) to inject a sudden "event" burst.
    /// Default 60 ticks ≈ every 3 seconds at 20 Hz.
    pub event_period: u64,
    /// Amplitude of the slow mood wave.
    pub wave_amplitude: f32,
}

impl Default for StubSensor {
    fn default() -> Self {
        Self { event_period: 60, wave_amplitude: 0.5 }
    }
}

impl Sensor for StubSensor {
    fn sample(&mut self, tick: u64) -> Vec<f32> {
        let t = tick as f32 * 0.05;   // slow time base

        // Slow sinusoidal "mood wave" — the ambient energy of the room
        let mood = (t.sin() * self.wave_amplitude + self.wave_amplitude).clamp(0.0, 1.0);

        // Per-channel phase offsets simulate different sensory modalities
        let phase_offsets: [f32; 8] = [
            0.0,
            PI / 4.0,
            PI / 2.0,
            3.0 * PI / 4.0,
            PI,
            5.0 * PI / 4.0,
            3.0 * PI / 2.0,
            7.0 * PI / 4.0,
        ];

        // Event burst: a sharp spike every `event_period` ticks
        let event = if tick % self.event_period == 0 { 0.9_f32 } else { 0.0_f32 };

        phase_offsets.iter().enumerate().map(|(i, &phi)| {
            let channel = (t + phi).sin() * 0.25 + mood * 0.60;
            // Channel 6 is the event flag
            let burst = if i == 6 { event } else { event * 0.3 };
            (channel + burst).clamp(0.0, 1.0)
        }).collect()
    }
}
