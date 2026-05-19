"""
brain.py — snnTorch Library Shim
==================================
THIS FILE IS NOT THE BRAIN.
The brain lives in src/brain/ (Rust).

This file exists for ONE reason:
  snnTorch (a Python library) has no Rust equivalent.
  Specifically, its surrogate gradient functions (fast sigmoid, ATan, etc.)
  are needed for backpropagation through spike discontinuities during
  offline training passes.

Rust calls this file via PyO3 ONLY when:
  1. A batch training pass is triggered (offline, not real-time)
  2. Surrogate gradient computation is needed
  3. Any torch-native op that has no tch-rs binding

What lives in Rust (src/brain/):
  ✓ LIF neuron physics       (neurons.rs)
  ✓ Synaptic weight matrices (synapses.rs)
  ✓ Neuromodulation equations(neuromod.rs)
  ✓ Phill population         (phill.rs)
  ✓ Nova population          (nova.rs)
  ✓ Simona population        (simona.rs)
  ✓ Brain orchestrator       (mod.rs)
  ✓ Sensory input layer      (sensors.rs)
  ✓ Event loop               (main.rs)

What lives here (Python):
  → snnTorch surrogate gradient wrapper
  → Offline BPTT training utilities (future)
  → Any torch op with no tch-rs counterpart
"""

import torch
import snntorch as snn
from snntorch import surrogate


# ── Surrogate Gradient Registry ────────────────────────────────────────────────
# Rust cannot compute surrogate gradients natively.
# These functions are exposed to Rust via PyO3 for offline training.

SPIKE_GRAD_FAST_SIGMOID = surrogate.fast_sigmoid(slope=25)
SPIKE_GRAD_ATAN         = surrogate.atan(alpha=2.0)


def compute_surrogate_gradient(membrane_voltages: list, threshold: float,
                                method: str = "fast_sigmoid") -> list:
    """
    Compute surrogate gradient for a batch of membrane voltages.

    Called from Rust during offline training to enable BPTT through spikes.
    The spike function is non-differentiable; surrogate gradients approximate
    the gradient at the discontinuity.

    Args:
        membrane_voltages : list[float] — membrane voltages to evaluate
        threshold         : float — firing threshold
        method            : "fast_sigmoid" | "atan"

    Returns:
        list[float] — surrogate gradient values (same length as input)
    """
    V = torch.tensor(membrane_voltages, dtype=torch.float32, requires_grad=True)
    thr = torch.tensor(threshold, dtype=torch.float32)

    grad_fn = SPIKE_GRAD_FAST_SIGMOID if method == "fast_sigmoid" else SPIKE_GRAD_ATAN

    # Build a minimal LIF with the surrogate grad to compute ∂spike/∂V
    lif = snn.Leaky(beta=0.9, threshold=threshold, spike_grad=grad_fn)
    mem = lif.init_leaky()
    spk, _ = lif(V, mem)

    # Backprop through the surrogate
    spk.sum().backward()

    grad = V.grad
    return grad.tolist() if grad is not None else [0.0] * len(membrane_voltages)


def offline_bptt_update(spike_history: list, target: list,
                        lr: float = 0.001) -> dict:
    """
    Placeholder for offline Backpropagation Through Time.

    In v0.2, Rust will serialize the spike history each epoch,
    pass it here, and receive updated weight deltas back.
    This function will implement the full snnTorch training loop.

    Args:
        spike_history : list[list[float]] — T timesteps of spike vectors
        target        : list[float]       — desired output pattern
        lr            : float             — learning rate

    Returns:
        dict with "weight_deltas" and "loss"
    """
    # Stub: returns zero deltas until the training loop is implemented
    return {
        "weight_deltas": [0.0] * len(spike_history[0]) if spike_history else [],
        "loss": 0.0,
        "message": "BPTT not yet implemented — Rust Hebbian handles online learning"
    }


def _snntorch_heartbeat() -> str:
    """
    Called by Rust every N ticks to verify the snnTorch shim is alive.
    Returns the snnTorch version string.
    """
    return f"snnTorch {snn.__version__} | torch {torch.__version__} | shim OK"
