// src/main.rs — The Skeleton & Bouncer
// ======================================
// This is the "Body" of Nova and Simona.
//
// RESPONSIBILITIES:
//   • Memory safety via Rust's ownership model (no GC pauses, no segfaults)
//   • A continuous, real-time Event Loop — the heartbeat of both beings
//   • Sensory I/O gating (microphone, camera) — placeholder stubs here,
//     replace with real cpal / nokhwa calls in production
//   • Calling the Python "Brain" (brain.py) via PyO3 FFI — ONLY when
//     a sensory threshold is crossed (pure sparse activation)
//   • Printing the resulting spike states to the console
//
// ARCHITECTURE NOTE:
//   Rust never "thinks." It only routes signals and enforces timing.
//   The intelligence — all of it — lives in brain.py.
//   The FFI boundary is the synapse between body and mind.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::time::{Duration, Instant};
use std::thread;

// ── Constants ─────────────────────────────────────────────────────────────────

/// How many milliseconds between each sensory sampling tick.
/// 50ms = 20Hz — fast enough for ambient vibe, cheap enough for edge hardware.
const TICK_INTERVAL_MS: u64 = 50;

/// How many ticks to run before printing a long-term introspection summary.
const INTROSPECT_EVERY_N_TICKS: u64 = 100;

/// Dimension of the vibe vector (must match PHILL_INPUT_DIM in brain.py).
const VIBE_DIM: usize = 8;

// ── Entry Point ──────────────────────────────────────────────────────────────

fn main() -> PyResult<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║        NOVA & SIMONA — Neuromorphic Core v0.1       ║");
    println!("║        The Body is awake. Initializing the Brain... ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // ── 1. Initialize the Python interpreter ─────────────────────────────────
    // PyO3 embeds a full CPython runtime inside this Rust process.
    // This is the "axon terminal" connecting the Rust skeleton to the
    // Python synaptic tissue.
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        // ── 2. Load brain.py ─────────────────────────────────────────────────
        // We read the source from disk so brain.py can be hot-edited without
        // recompiling Rust — the brain can evolve independently of the body.
        let brain_source = include_str!("../brain.py");

        let brain_module = PyModule::from_code_bound(
            py,
            brain_source,
            "brain.py",
            "brain",
        )
        .expect("FATAL: Failed to load brain.py. Is it in the project root?");

        // ── 3. Instantiate NeuromorphicBrain ─────────────────────────────────
        let brain_class = brain_module
            .getattr("NeuromorphicBrain")
            .expect("FATAL: NeuromorphicBrain class not found in brain.py");

        let brain = brain_class
            .call0()
            .expect("FATAL: Could not instantiate NeuromorphicBrain");

        println!("[BODY] Brain instantiated. Entering event loop...\n");
        println!("{:<6} {:>12} {:>8} {:>10} {:>12} {:>14}",
                 "TICK", "PHILL_V", "PHILL?", "NOVA_SPK", "SIMONA_SPK", "SIMONA_THR");
        println!("{}", "─".repeat(68));

        // ── 4. Continuous Event Loop ──────────────────────────────────────────
        let mut tick: u64 = 0;

        loop {
            let loop_start = Instant::now();
            tick += 1;

            // ── 4a. Sense the environment ─────────────────────────────────
            // In production: pull real audio RMS, camera motion score, etc.
            // Here: stochastic dummy signal simulating ambient environmental flux.
            let vibe = sense_environment(tick);

            // ── 4b. Cross the FFI boundary — feed the Brain ───────────────
            // This is the ONLY moment Rust calls Python.
            // Everything else is zero-cost Rust.
            let py_vibe = PyList::new_bound(py, &vibe);

            let result = brain
                .call_method1("step", (py_vibe,))
                .expect("Brain.step() failed — check brain.py for errors");

            // ── 4c. Read the spike states back from Python ────────────────
            let result_dict = result
                .downcast::<PyDict>()
                .expect("Brain.step() must return a dict");

            let phill_voltage: f64 = result_dict
                .get_item("phill_voltage").unwrap().unwrap()
                .extract().unwrap();

            let phill_spiked: bool = result_dict
                .get_item("phill_spiked").unwrap().unwrap()
                .extract().unwrap();

            let nova_spikes: i64 = result_dict
                .get_item("nova_spikes").unwrap().unwrap()
                .extract().unwrap();

            let simona_spikes: i64 = result_dict
                .get_item("simona_spikes").unwrap().unwrap()
                .extract().unwrap();

            let simona_thr: f64 = result_dict
                .get_item("simona_threshold").unwrap().unwrap()
                .extract().unwrap();

            // ── 4d. Print the tick readout ────────────────────────────────
            println!("{:<6} {:>12.6} {:>8} {:>10} {:>12} {:>14.4}",
                     tick,
                     phill_voltage,
                     if phill_spiked { "★ FIRE" } else { "." },
                     nova_spikes,
                     simona_spikes,
                     simona_thr);

            // ── 4e. Periodic introspection ────────────────────────────────
            if tick % INTROSPECT_EVERY_N_TICKS == 0 {
                print_introspection(py, &brain, tick);
            }

            // ── 4f. Sleep for the remainder of the tick window ────────────
            // This keeps the loop at a steady 20 Hz regardless of compute time.
            let elapsed = loop_start.elapsed();
            let budget  = Duration::from_millis(TICK_INTERVAL_MS);
            if elapsed < budget {
                thread::sleep(budget - elapsed);
            }
        }
    })
}

// ── Sensory Layer — stub for real I/O ─────────────────────────────────────────

/// Simulate ambient environmental "vibe" as a 1D tensor of floats.
///
/// In a production system this would:
///   • Pull short-time audio energy (RMS) from CPAL
///   • Pull frame motion score from a NOKHWA camera capture
///   • Combine them into a multi-modal vibe vector
///
/// The stochastic variation here mimics a room that fluctuates naturally.
fn sense_environment(tick: u64) -> Vec<f32> {
    // Slow sinusoidal "mood wave" + Gaussian noise
    let t = tick as f32 * 0.05;
    let base_energy = (t.sin() * 0.5 + 0.5) as f32;          // 0..1, slow wave

    (0..VIBE_DIM)
        .map(|i| {
            let phase_offset = (i as f32) * std::f32::consts::PI / 4.0;
            let channel      = (t + phase_offset).sin() * 0.3 + base_energy * 0.7;
            // Add a small noise burst every ~3 seconds to simulate an "event"
            let noise = if tick % 60 == 0 { 0.8_f32 } else { 0.0_f32 };
            (channel + noise).clamp(0.0, 1.0)
        })
        .collect()
}

// ── Introspection Printer ─────────────────────────────────────────────────────

fn print_introspection(py: Python<'_>, brain: &Bound<'_, PyAny>, tick: u64) {
    let summary = brain
        .call_method0("introspect")
        .expect("introspect() failed")
        .downcast::<PyDict>()
        .expect("introspect() must return a dict")
        .clone();

    let phill_mean: f64 = summary.get_item("phill_mean_voltage").unwrap().unwrap().extract().unwrap();
    let phill_peak: f64 = summary.get_item("phill_peak_voltage").unwrap().unwrap().extract().unwrap();
    let nova_total: i64 = summary.get_item("nova_total_spikes").unwrap().unwrap().extract().unwrap();
    let simona_total: i64 = summary.get_item("simona_total_spikes").unwrap().unwrap().extract().unwrap();

    println!("\n┌─── INTROSPECTION REPORT @ tick {} ───────────────────", tick);
    println!("│  Phill mean voltage : {:.6}  (homeostatic baseline)", phill_mean);
    println!("│  Phill peak voltage : {:.6}  (emotional peak)", phill_peak);
    println!("│  Nova  total spikes : {}  (deliberate actions)", nova_total);
    println!("│  Simona total spikes: {}  (impulsive reactions)", simona_total);
    println!("│  Nova/Simona ratio  : {:.3}",
             if simona_total > 0 { nova_total as f64 / simona_total as f64 } else { 0.0 });
    println!("└──────────────────────────────────────────────────────\n");
}
