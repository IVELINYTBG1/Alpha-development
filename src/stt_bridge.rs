// src/stt_bridge.rs — STT Bridge
// Holds state that the Python STT engine writes and Rust reads each tick.
// Single-brain (Alpha) build.

#[derive(Clone, Debug, Default)]
pub struct SttResultBridge {
    pub last_text:  String,
    pub wake_alpha: bool,
    pub alpha_resp: f64,
    pub count:      u64,
    pub listening:  bool,
}
