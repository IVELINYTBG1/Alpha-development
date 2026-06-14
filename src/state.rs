// src/state.rs — Shared Types
// All data structures shared between audio, brain, TUI, and STT threads.
// Single-brain (Alpha) build.

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
    pub wake_alpha:       bool,   // name heard this tick
    pub alpha_resp:       f64,    // responsiveness score [0,1]
    pub total_transcripts:u64,
    pub error:            Option<String>,
}

// ── Brain result (single brain: Alpha) ──────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BrainResult {
    pub tick:                 u64,
    pub phill_voltage:        f64,
    pub phill_spiked:         bool,
    pub alpha_broca_spikes:   u64,
    pub alpha_pfc_threshold:  f64,
    pub alpha_pfc_voltage:    f64,
    pub speech_trigger:       Option<String>,
    pub alpha_tts_speaking:   bool,
    pub active_regions:       Vec<String>,
    pub energy:               f64,
    pub global_workspace:     bool,
    pub voice_trust:          f64,
    pub voice_status:         String,
    pub phill_gain:           f64,
    pub alpha_regions:        Vec<(String, f64)>,
    pub sem_concepts:         usize,
    pub combined_id:          f64,
    pub face_present:         bool,
    pub imprint_status:       String,
    pub camera_active:        bool,
    pub alpha_vigilance:      bool,
    pub alpha_pressure:       f64,
    pub story_active:         bool,
    pub story_event:          Option<String>,
    // Babbling cortex counters
    pub alpha_babble_count:   u64,
    pub alpha_bound_count:    u64,
    pub alpha_motor_map_size: u64,
    // Vocal self-esteem — "do I like how I sound?" (0..1)
    pub alpha_voice_esteem:   f64,
    // Predictive self-monitoring — "surprise" = forward-model prediction error (0..1)
    pub alpha_voice_surprise: f64,
    // Neurochemistry — dopamine / serotonin / GABA / amygdala arousal
    pub alpha_da:        f64,
    pub alpha_ser:       f64,
    pub alpha_gaba:      f64,
    pub alpha_arousal:   f64,
    // Cerebellum — motor coordination/fluency (0..1, climbs as it learns)
    pub alpha_coord:     f64,
    // Sleep / consolidation (Stage 3)
    pub asleep:         bool,
    pub sleep_pressure: f64,
    pub alpha_episodes: u64,
    // Stage 4 neuromodulators: acetylcholine / norepinephrine / oxytocin
    pub alpha_ach:   f64,
    pub alpha_ne:    f64,
    pub alpha_oxy:   f64,
    // Core felt emotion (AffectCore readout): named feeling, strength [0,1],
    // valence [0,1] (0 = unpleasant, 0.5 = neutral, 1 = pleasant).
    pub alpha_feeling:          String,
    pub alpha_feel_intensity:   f64,
    pub alpha_valence:          f64,
    // Personality drift: how 'in character' Alpha is now (selfness 0..1) and
    // how much that has grown this session (drift, +/-).
    pub alpha_selfness:   f64,
    pub alpha_drift:      f64,
}

impl Default for BrainResult {
    fn default() -> Self {
        Self {
            tick:0, phill_voltage:0.0, phill_spiked:false,
            alpha_broca_spikes:0,
            alpha_pfc_threshold:1.4,
            alpha_pfc_voltage:0.0,
            speech_trigger:None, alpha_tts_speaking:false,
            active_regions:vec![], energy:0.0, global_workspace:false,
            voice_trust:0.0, voice_status:"listening\u{2026}".into(), phill_gain:0.7,
            alpha_regions:vec![], sem_concepts:0,
            combined_id:0.0, face_present:false, imprint_status:"learning".into(),
            camera_active:false, alpha_vigilance:false, alpha_pressure:0.0,
            story_active:false, story_event:None,
            alpha_babble_count:0, alpha_bound_count:0, alpha_motor_map_size:0,
            alpha_voice_esteem:0.5, alpha_voice_surprise:0.5,
            // Neurochemistry baselines (match brain.py Neuromodulators init)
            alpha_da:0.45, alpha_ser:0.75, alpha_gaba:0.45, alpha_arousal:0.0,
            alpha_coord:0.4,
            asleep:false, sleep_pressure:0.0, alpha_episodes:0,
            alpha_ach:0.50, alpha_ne:0.40, alpha_oxy:0.30,
            alpha_feeling:"calm".into(), alpha_feel_intensity:0.0, alpha_valence:0.5,
            alpha_selfness:0.5, alpha_drift:0.0,
        }
    }
}

// ── Web search event ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SearchEvent {
    pub speaker:   String,   // "alpha"
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
    pub alpha_broca_hist: Vec<u64>,
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
            alpha_broca_hist:vec![0u64; SPARKLINE_LEN],
            total_ticks:0, error_msg:None,
            chat_history: vec![
                ChatLine::system("Alpha v0.5 -- CPU-native neuromorphic SNN -- 7 cortical regions"),
                ChatLine::system("TAB = switch TEXT/STT mode  |  i = type  |  q = quit"),
                ChatLine::system("In STT: say 'Alpha' to wake him. He learns over time."),
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
