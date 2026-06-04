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
    // Babbling cortex counters
    pub nova_babble_count:    u64,
    pub nova_bound_count:     u64,
    pub nova_motor_map_size:  u64,
    pub simona_babble_count:  u64,
    pub simona_bound_count:   u64,
    pub simona_motor_map_size:u64,
    // Vocal self-esteem — each personality's "do I like how I sound?" (0..1)
    pub nova_voice_esteem:    f64,
    pub simona_voice_esteem:  f64,
    // Predictive self-monitoring — "surprise" = forward-model prediction error (0..1)
    pub nova_voice_surprise:  f64,
    pub simona_voice_surprise:f64,
    // Secure inter-personality link — pending message counts (content opaque to observer)
    pub link_nova_to_simona:  u64,
    pub link_simona_to_nova:  u64,
    // Neurochemistry — dopamine / serotonin / GABA / amygdala arousal, per personality
    pub nova_da:        f64,
    pub nova_ser:       f64,
    pub nova_gaba:      f64,
    pub nova_arousal:   f64,
    pub simona_da:      f64,
    pub simona_ser:     f64,
    pub simona_gaba:    f64,
    pub simona_arousal: f64,
    // Cerebellum — motor coordination/fluency (0..1, climbs as it learns)
    pub nova_coord:     f64,
    pub simona_coord:   f64,
    // Sleep / consolidation (Stage 3)
    pub asleep:         bool,
    pub sleep_pressure: f64,
    pub nova_episodes:  u64,
    pub simona_episodes:u64,
    // Stage 4 neuromodulators: acetylcholine / norepinephrine / oxytocin
    pub nova_ach:   f64,
    pub nova_ne:    f64,
    pub nova_oxy:   f64,
    pub simona_ach: f64,
    pub simona_ne:  f64,
    pub simona_oxy: f64,

    // Core felt emotion (AffectCore readout): the named feeling, its strength
    // [0,1], and valence [0,1] (0 = unpleasant, 0.5 = neutral, 1 = pleasant).
    pub nova_feeling:          String,
    pub nova_feel_intensity:   f64,
    pub nova_valence:          f64,
    pub simona_feeling:        String,
    pub simona_feel_intensity: f64,
    pub simona_valence:        f64,

    // Personality drift (mechanism #4): how 'in character' each is right now
    // (selfness 0..1) and how much that has grown this session (drift, +/-).
    pub nova_selfness:   f64,
    pub nova_drift:      f64,
    pub simona_selfness: f64,
    pub simona_drift:    f64,
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
            nova_babble_count:0, nova_bound_count:0, nova_motor_map_size:0,
            simona_babble_count:0, simona_bound_count:0, simona_motor_map_size:0,
            nova_voice_esteem:0.5, simona_voice_esteem:0.5,
            nova_voice_surprise:0.5, simona_voice_surprise:0.5,
            link_nova_to_simona:0, link_simona_to_nova:0,
            // Neurochemistry baselines (match brain.py Neuromodulators init)
            nova_da:0.45, nova_ser:0.75, nova_gaba:0.45, nova_arousal:0.0,
            simona_da:0.60, simona_ser:0.40, simona_gaba:0.35, simona_arousal:0.0,
            nova_coord:0.4, simona_coord:0.4,
            asleep:false, sleep_pressure:0.0, nova_episodes:0, simona_episodes:0,
            nova_ach:0.50, nova_ne:0.40, nova_oxy:0.30,
            simona_ach:0.50, simona_ne:0.40, simona_oxy:0.30,
            nova_feeling:"calm".into(),   nova_feel_intensity:0.0,   nova_valence:0.5,
            simona_feeling:"calm".into(), simona_feel_intensity:0.0, simona_valence:0.5,
            nova_selfness:0.5, nova_drift:0.0, simona_selfness:0.5, simona_drift:0.0,
        }
    }
}

// ── Web search event ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SearchEvent {
    pub speaker:   String,   // "nova" | "simona"
    pub query:     String,
    pub snippet:   String,
    pub timestamp: String,   // "HH:MM:SS" local time when ingested
}

// ── Shared state ──────────────────────────────────────────────────────────────

pub const SPARKLINE_LEN:    usize = 40;
pub const CHAT_HISTORY_MAX: usize = 300;
pub const THOUGHT_HISTORY:  usize = 8;
pub const SEARCH_HISTORY:   usize = 12;

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
    pub search_history:   Vec<SearchEvent>,
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
            search_history: vec![],
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
