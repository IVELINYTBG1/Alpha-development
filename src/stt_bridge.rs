// src/stt_bridge.rs — STT Bridge
// Holds state that the Python STT engine writes and Rust reads each tick.

#[derive(Clone, Debug, Default)]
pub struct SttResultBridge {
    pub last_text:  String,
    pub wake_nova:  bool,
    pub wake_simona:bool,
    pub nova_resp:  f64,
    pub simona_resp:f64,
    pub count:      u64,
    pub listening:  bool,
}
