// src/telemetry.rs
// =================
// Console telemetry — the window into the living brain.
//
// This module owns all the display formatting.
// The event loop only calls print_tick() and print_introspection().

use crate::brain::{TickResult, Introspection};

/// Print the column header (called once at startup).
pub fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       NOVA & SIMONA — Neuromorphic Core  v0.1               ║");
    println!("║       Rust Body  ·  Python snnTorch Lib  ·  LIF Physics     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("{:<6} {:>10} {:>7} {:>9} {:>11} {:>8} {:>8} {:>7}",
             "TICK", "PHILL_V", "FIRE?", "NOVA_OUT", "SIMONA_OUT",
             "N_THR", "S_THR", "STRESS");
    println!("{}", "─".repeat(72));
}

/// Print one tick line.
pub fn print_tick(r: &TickResult) {
    let fire = if r.phill_fired { "★" } else { "·" };
    println!("{:<6} {:>10.5} {:>7} {:>9} {:>11} {:>8.3} {:>8.3} {:>7.4}",
             r.tick,
             r.phill_voltage,
             fire,
             r.nova_out_spikes,
             r.simona_out_spikes,
             r.nova_threshold,
             r.simona_threshold,
             r.stress_index);
}

/// Print the full introspection block.
pub fn print_introspection(i: &Introspection) {
    println!("\n┌─── INTROSPECTION @ tick {} {}", i.total_ticks, "─".repeat(32));
    println!("│  Phill homeostatic baseline : {:.6}", i.phill_baseline);
    println!("│  Phill stress index         : {:.6}", i.phill_stress_index);
    println!("│  Nova  total spikes         : {}", i.nova_total_spikes);
    println!("│  Simona total spikes        : {}", i.simona_total_spikes);
    println!("│  Nova / Simona ratio        : {:.4}  ({} stable / {} impulsive)",
             i.nova_simona_ratio,
             if i.nova_simona_ratio > 0.5 { "MORE" } else { "less" },
             if i.nova_simona_ratio < 0.5 { "MORE" } else { "less" });
    println!("└{}", "─".repeat(50));
    println!();
}
