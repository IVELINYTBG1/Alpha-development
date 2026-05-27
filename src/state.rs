// src/state.rs — Shared Types
// All data structures shared between audio, brain, TUI, and STT threads.

use std::collections::VecDeque;

// ── Input mode ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum InputMode {
    /// TUI text box — user types, Enter sends
    Text,
    /// Always-on STT — mic icon shown, wake word active
    Stt,
}

impl Default for InputMode {
    fn default() -> Self { InputMode::Text }
}

impl InputMode {
    pub fn toggle(&self) -> Self {
        match self {
            InputMode::Text => InputMode::Stt,
            InputMode::Stt  => InputMode::Text,
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            InputMode::Text => "TEXT",
            InputMode::Stt  => "STT",
        }
    }
}

// ── Audio features ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct AudioFeatures {
    pub rms:       f32,
    pub zcr:       f32,
    pub band_low:  f32,
    pub band_mid:  f32,
    pub band_high: f32,
}

impl AudioFeatures {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.rms, self.zcr, self.band_low, self.band_mid, self.band_high]
    }
}

pub fn compute_features(w: &[f32]) -> AudioFeatures {
    let n = w.len();
    if n == 0 { return AudioFeatures::default(); }
    let rms = (w.iter().map(|&s| s*s).sum::<f32>() / n as f32).sqrt();
    let zcr = w.windows(2)
        .filter(|p| (p[0] >= 0.0) != (p[1] >= 0.0))
        .count() as f32 / n as f32;
    let t   = n / 3;
    let br  = |s: &[f32]| if s.is_empty() { 0.0f32 }
              else { (s.iter().map(|&x| x*x).sum::<f32>() / s.len() as f32).sqrt() };
    AudioFeatures {
        rms, zcr,
        band_low:  br(&w[..t]),
        band_mid:  br(&w[t..2*t]),
        band_high: br(&w[2*t..]),
    }
}

// ── Chat line ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ChatLine {
    pub speaker:    String,
    pub text:       String,
    pub regions:    Vec<String>,
    pub story_mode: bool,
    pub from_stt:   bool,   // true if this line came from voice recognition
}

impl ChatLine {
    pub fn system(text: impl Into<String>) -> Self {
        Self { speaker:"system".into(), text:text.into(),
               regions:vec![], story_mode:false, from_stt:false }
    }
}

// ── STT state ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct SttState {
    pub backend:          String,
    pub listening:        bool,
    pub last_transcript:  String,
    pub wake_nova:        bool,   // name heard this tick
    pub wake_simona:      bool,
    pub nova_resp:        f64,    // responsiveness score [0,1]
    pub simona_resp:      f64,
    pub total_transcripts:u64,
    pub error:            Option<String>,
}

// ── Brain result ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BrainResult {
    pub tick:                 u64,
    pub phill_voltage:        f64,
    pub phill_spiked:         bool,
    pub nova_broca_spikes:    u64,
    pub simona_broca_spikes:  u64,
    pub nova_pfc_threshold:   f64,
    pub simona_broca_thr:     f64,
    pub nova_pfc_voltage:     f64,
    pub simona_broca_voltage: f64,
    pub speech_trigger:       Option<String>,
    pub nova_tts_speaking:    bool,
    pub simona_tts_speaking:  bool,
    pub active_regions:       Vec<String>,
    pub energy:               f64,
    pub global_workspace:     bool,
    pub voice_trust:          f64,
    pub voice_status:         String,
    pub phill_gain:           f64,
    pub nova_regions:         Vec<(String, f64)>,
    pub simona_regions:       Vec<(String, f64)>,
    pub sem_concepts:         usize,
    pub combined_id:          f64,
    pub face_present:         bool,
    pub imprint_status:       String,
    pub camera_active:        bool,
    pub nova_vigilance:       bool,
    pub nova_pressure:        f64,
    pub simona_pressure:      f64,
    pub story_active:         bool,
    pub story_event:          Option<String>,
}

impl Default for BrainResult {
    fn default() -> Self {
        Self {
            tick:0, phill_voltage:0.0, phill_spiked:false,
            nova_broca_spikes:0, simona_broca_spikes:0,
            nova_pfc_threshold:1.4, simona_broca_thr:0.38,
            nova_pfc_voltage:0.0, simona_broca_voltage:0.0,
            speech_trigger:None, nova_tts_speaking:false, simona_tts_speaking:false,
            active_regions:vec![], energy:0.0, global_workspace:false,
            voice_trust:0.0, voice_status:"listening\u{2026}".into(), phill_gain:0.7,
            nova_regions:vec![], simona_regions:vec![], sem_concepts:0,
            combined_id:0.0, face_present:false, imprint_status:"learning".into(),
            camera_active:false, nova_vigilance:false, nova_pressure:0.0,
            simona_pressure:0.0, story_active:false, story_event:None,
        }
    }
}

// ── Shared state ──────────────────────────────────────────────────────────────

pub const SPARKLINE_LEN:    usize = 40;
pub const CHAT_HISTORY_MAX: usize = 300;
pub const THOUGHT_HISTORY:  usize = 8;

#[derive(Clone, Debug)]
pub struct SharedState {
    pub mic_volume:       f64,
    pub mic_active:       bool,
    pub audio_features:   AudioFeatures,
    pub brain:            BrainResult,
    pub stt:              SttState,
    pub input_mode:       InputMode,
    // Sparkline histories
    pub phill_history:    Vec<u64>,
    pub trust_history:    Vec<u64>,
    pub id_history:       Vec<u64>,
    pub nova_broca_hist:  Vec<u64>,
    pub sim_broca_hist:   Vec<u64>,
    pub total_ticks:      u64,
    pub error_msg:        Option<String>,
    // Chat panels
    pub chat_history:     Vec<ChatLine>,
    pub thought_history:  Vec<ChatLine>,
    // Input
    pub input_text:       String,
    pub typing_active:    bool,   // cursor shown in text box
    // Pending input from either text box or STT
    pub pending_from_stt: bool,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            mic_volume:0.0, mic_active:false,
            audio_features:AudioFeatures::default(),
            brain:BrainResult::default(),
            stt:SttState::default(),
            input_mode:InputMode::Text,
            phill_history:   vec![0u64; SPARKLINE_LEN],
            trust_history:   vec![0u64; SPARKLINE_LEN],
            id_history:      vec![0u64; SPARKLINE_LEN],
            nova_broca_hist: vec![0u64; SPARKLINE_LEN],
            sim_broca_hist:  vec![0u64; SPARKLINE_LEN],
            total_ticks:0, error_msg:None,
            chat_history: vec![
                ChatLine::system("Nova & Simona v0.5 -- CPU-native -- 13 anatomical regions"),
                ChatLine::system("TAB = switch TEXT/STT mode  |  i = type  |  q = quit"),
                ChatLine::system("In STT: say 'Nova' or 'Simona' to wake them. They learn over time."),
            ],
            thought_history: vec![],
            input_text:String::new(),
            typing_active:false,
            pending_from_stt:false,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn push_spark(v: &mut Vec<u64>, val: u64) {
    v.remove(0);
    v.push(val);
}

pub fn trim_chat(h: &mut Vec<ChatLine>) {
    if h.len() > CHAT_HISTORY_MAX {
        let excess = h.len() - CHAT_HISTORY_MAX;
        h.drain(..excess);
    }
}
