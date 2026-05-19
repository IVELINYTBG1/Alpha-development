// src/main.rs — The Skeleton & Bouncer
// ======================================
// Rust owns:
//   • The entire brain physics (brain/ module — neurons, synapses, neuromod, populations)
//   • The continuous event loop
//   • The sensory I/O layer (sensors.rs)
//   • All state, memory safety, no-crash guarantees
//
// Python is called ONLY for:
//   • snnTorch surrogate gradient computations (no Rust equivalent exists)
//   • Future: torch-based batch learning passes offline
//
// The FFI boundary is a narrow slit, not a flood.
// Rust does 95 %+ of the work; Python is a library call.

mod brain;
mod sensors;
mod telemetry;

use brain::NeuromorphicBrain;
use sensors::{Sensor, StubSensor};
use telemetry::{print_header, print_tick, print_introspection};

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::time::{Duration, Instant};
use std::thread;

// ── Loop parameters ───────────────────────────────────────────────────────────
const TICK_INTERVAL_MS:       u64 = 50;   // 20 Hz
const INTROSPECT_EVERY_N:     u64 = 100;  // every 5 seconds

fn main() -> PyResult<()> {
    print_header();

    // ── Initialize Python interpreter (narrow, one-time cost) ─────────────
    // We embed Python ONLY to access snnTorch's surrogate gradient library.
    // The brain physics themselves run in Rust.
    pyo3::prepare_freethreaded_python();

    // Load the thin Python shim — only snnTorch wrapper functions live here
    let snntorch_shim_src = include_str!("../brain.py");

    Python::with_gil(|py| {
        let shim = PyModule::from_code_bound(py, snntorch_shim_src, "brain.py", "brain")
            .expect("Failed to load brain.py snnTorch shim");

        // Verify the shim is healthy before entering the loop
        println!("[BODY] snnTorch shim loaded: {:?}", shim.name());
        println!("[BODY] Rust brain initializing...\n");

        // ── Rust brain and sensors ─────────────────────────────────────────
        let mut brain   = NeuromorphicBrain::new();
        let mut sensor  = StubSensor::default();

        // ── Continuous Event Loop ──────────────────────────────────────────
        let mut tick: u64 = 0;
        loop {
            let loop_start = Instant::now();
            tick += 1;

            // ── Sense ──────────────────────────────────────────────────────
            let vibe = sensor.sample(tick);

            // ── Think — pure Rust, no FFI ──────────────────────────────────
            // The entire LIF physics, neuromodulation, and Hebbian plasticity
            // execute here in compiled Rust at native speed.
            let result = brain.step(&vibe);

            // ── Display ────────────────────────────────────────────────────
            print_tick(&result);

            // ── Introspect ─────────────────────────────────────────────────
            if tick % INTROSPECT_EVERY_N == 0 {
                let summary = brain.introspect();
                print_introspection(&summary);

                // ── Python is called HERE — surrogate gradient / batch learn
                // In v0.1 this is a stub call to prove the FFI link is live.
                // In v0.2: pass spike history to snnTorch for BPTT weight update.
                let _py_result = shim
                    .call_method0("_snntorch_heartbeat")
                    .unwrap_or_else(|_| py.None().into_bound(py));
                // ^ If the shim doesn't have this method yet, we silently skip.
            }

            // ── Pace the loop to exactly TICK_INTERVAL_MS ──────────────────
            let elapsed = loop_start.elapsed();
            let budget  = Duration::from_millis(TICK_INTERVAL_MS);
            if elapsed < budget {
                thread::sleep(budget - elapsed);
            }
        }
    })
}
