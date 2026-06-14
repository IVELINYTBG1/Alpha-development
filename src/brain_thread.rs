// src/brain_thread.rs — Brain Loop Thread
// Calls brain.step() at 20Hz and brain.think() on pending input.
// Also polls the thought pipe and dispatches STT transcriptions.
// Single-brain (Alpha) build.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;

use arc_swap::ArcSwap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::state::{
    BrainResult, ChatLine, SearchEvent, SharedState,
    push_spark, trim_chat, SEARCH_HISTORY,
};

pub const BRAIN_INTERVAL_MS: u64 = 50; // 20 Hz

pub fn brain_thread(
    state:         Arc<ArcSwap<SharedState>>,
    running:       Arc<AtomicBool>,
    pending_input: Arc<Mutex<Option<(String, bool)>>>, // (text, from_stt)
    stt_results:   Arc<Mutex<Option<crate::stt_bridge::SttResultBridge>>>,
) {
    let brain_src = include_str!("../brain.py");

    Python::with_gil(|py| {
        let brain_mod = match PyModule::from_code_bound(py, brain_src, "brain.py", "brain") {
            Ok(m)  => m,
            Err(e) => {
                crate::update_state(&state, |s| {
                    s.error_msg = Some(format!("brain.py failed to load: {e}"));
                });
                return;
            }
        };

        let brain = brain_mod.getattr("NeuromorphicBrain").unwrap().call0().unwrap();

        // Show init messages
        if let Ok(msgs) = brain_mod.getattr("_INIT_MESSAGES")
            .and_then(|o| o.extract::<Vec<String>>()) {
            crate::update_state(&state, |s| {
                for msg in &msgs {
                    s.chat_history.push(ChatLine {
                        speaker: "system".into(),
                        text: format!("[init] {msg}"),
                        regions: vec![], story_mode: false, from_stt: false,
                    });
                }
            });
        }

        let mut tick: u64 = 0;

        while running.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            tick  += 1;

            // ── Read live state ───────────────────────────────────────────
            let cur   = state.load();
            let mic   = cur.mic_volume;
            let feats = cur.audio_features.clone();
            drop(cur);

            // ── step() — 20Hz physics tick ────────────────────────────────
            let py_feats = PyList::new_bound(py, &feats.to_vec());
            if let Ok(res) = brain.call_method1("step", (mic, py_feats)) {
                if let Ok(d) = res.downcast::<PyDict>() {
                    let br = extract_step_result(d, tick);
                    crate::update_state(&state, |s| {
                        push_spark(&mut s.phill_history,    (br.phill_voltage * 100.0) as u64);
                        push_spark(&mut s.trust_history,    (br.voice_trust   * 100.0) as u64);
                        push_spark(&mut s.id_history,       (br.combined_id   * 100.0) as u64);
                        push_spark(&mut s.alpha_broca_hist, br.alpha_broca_spikes.min(32) * 3);
                        s.brain       = br;
                        s.total_ticks = tick;
                    });
                }
            }

            // ── Poll thought pipe leaks ───────────────────────────────────
            if let Ok(leaked) = brain.call_method0("get_leaked_thoughts") {
                if let Ok(thoughts) = leaked.extract::<Vec<(String, String)>>() {
                    if !thoughts.is_empty() {
                        crate::update_state(&state, |s| {
                            for (who, thought) in &thoughts {
                                s.thought_history.push(ChatLine {
                                    speaker: format!("thought_{who}"),
                                    text: thought.clone(),
                                    regions: vec![], story_mode: false, from_stt: false,
                                });
                                if s.thought_history.len() > crate::state::THOUGHT_HISTORY {
                                    s.thought_history.remove(0);
                                }
                            }
                        });
                    }
                }
            }

            // ── Poll proactive chat messages (rare; Alpha is reserved) ────
            if let Ok(msgs) = brain.call_method0("get_proactive_messages") {
                if let Ok(items) = msgs.extract::<Vec<(String, String)>>() {
                    if !items.is_empty() {
                        crate::update_state(&state, |s| {
                            for (who, message) in &items {
                                s.chat_history.push(ChatLine {
                                    speaker: who.clone(),
                                    text: message.clone(),
                                    regions: vec![], story_mode: false, from_stt: false,
                                });
                            }
                            trim_chat(&mut s.chat_history);
                        });
                    }
                }
            }

            // ── Poll emergent web searches ────────────────────────────────
            if let Ok(searches) = brain.call_method0("get_pending_searches") {
                if let Ok(events) = searches.extract::<Vec<(String, String, String)>>() {
                    if !events.is_empty() {
                        let now = chrono::Local::now().format("%H:%M:%S").to_string();
                        crate::update_state(&state, |s| {
                            for (speaker, query, snippet) in &events {
                                s.search_history.push(SearchEvent {
                                    speaker:   speaker.clone(),
                                    query:     query.clone(),
                                    snippet:   snippet.clone(),
                                    timestamp: now.clone(),
                                });
                                if s.search_history.len() > SEARCH_HISTORY {
                                    s.search_history.remove(0);
                                }
                            }
                        });
                    }
                }
            }

            // ── STT status sync ───────────────────────────────────────────
            {
                let sr = stt_results.lock().unwrap();
                if let Some(ref bridge) = *sr {
                    crate::update_state(&state, |s| {
                        s.stt.last_transcript   = bridge.last_text.clone();
                        s.stt.wake_alpha        = bridge.wake_alpha;
                        s.stt.alpha_resp        = bridge.alpha_resp;
                        s.stt.total_transcripts = bridge.count;
                        s.stt.listening         = bridge.listening;
                    });
                }
            }

            // ── think() dispatch ──────────────────────────────────────────
            if let Some((text, from_stt)) = pending_input.lock().unwrap().take() {
                let story_active = state.load().brain.story_active;
                let speaker_name = if story_active { "NodeVortex" }
                                   else if from_stt { "Voice" }
                                   else { "You" };

                crate::update_state(&state, |s| {
                    s.chat_history.push(ChatLine {
                        speaker: "nodevortex".into(),
                        text: format!("[{speaker_name}] {text}"),
                        regions: vec![], story_mode: story_active, from_stt,
                    });
                });

                match brain.call_method1("think", (text.as_str(),)) {
                    Ok(res) => {
                        if let Ok(d) = res.downcast::<PyDict>() {
                            dispatch_think_result(d, &state, story_active, tick, &brain);
                        }
                    }
                    Err(e) => {
                        crate::update_state(&state, |s| {
                            s.chat_history.push(ChatLine {
                                speaker: "system".into(),
                                text: format!("think() error: {e}"),
                                regions: vec![], story_mode: false, from_stt: false,
                            });
                        });
                    }
                }
            }

            // ── Pace to 20 Hz ─────────────────────────────────────────────
            // Release the GIL during the inter-tick sleep so the Python-side
            // Alpha personality thread can advance its own loop. Without this,
            // Python threads spawned inside brain.py would never get CPU time
            // because Rust holds Python::with_gil for the entire brain loop.
            let el     = t0.elapsed();
            let budget = Duration::from_millis(BRAIN_INTERVAL_MS);
            if el < budget { py.allow_threads(|| thread::sleep(budget - el)); }
        }
    });
}

fn dispatch_think_result(
    d:            &pyo3::Bound<'_, PyDict>,
    state:        &Arc<ArcSwap<SharedState>>,
    _story_active: bool,
    tick:         u64,
    brain:        &pyo3::Bound<'_, PyAny>,
) {
    let alpha_t   = extract_str(d, "alpha");
    let energy    = extract_f64(d, "energy");
    let gw        = extract_bool(d, "global_workspace");
    let tticks    = extract_u64(d, "think_ticks");
    let story_ev  = extract_str(d, "story_event");
    let story_on  = extract_bool(d, "story_active");
    let act_reg: Vec<String> = d.get_item("active_regions").ok().flatten()
        .and_then(|v| v.extract().ok()).unwrap_or_default();
    let alpha_r   = extract_region_map(d, "alpha_regions");

    let sem_ct: usize = brain.call_method0("introspect").ok()
        .and_then(|r| r.downcast::<PyDict>().ok()
            .and_then(|d| d.get_item("sem_concepts").ok().flatten()
                .and_then(|v| v.extract().ok())))
        .unwrap_or(0);

    crate::update_state(state, |s| {
        s.brain.active_regions   = act_reg.clone();
        s.brain.energy           = energy;
        s.brain.global_workspace = gw;
        s.brain.alpha_regions    = alpha_r;
        s.brain.story_active     = story_on;
        s.brain.story_event      = story_ev.clone();
        s.brain.sem_concepts     = sem_ct;

        if let Some(t) = alpha_t {
            s.chat_history.push(ChatLine {
                speaker: "alpha".into(), text: t,
                regions: act_reg.clone(), story_mode: story_on, from_stt: false,
            });
        }
        if gw {
            s.chat_history.push(ChatLine {
                speaker: "system".into(),
                text: format!("[GW] {tticks} think-ticks -- {:.0}% energy", energy * 100.0),
                regions: vec![], story_mode: false, from_stt: false,
            });
        }
        if let Some(ev) = &story_ev {
            let ev_txt = match ev.as_str() {
                "STORY_MODE_START"     => Some("[STORY] Scene opens -- NodeVortex enters the lab"),
                "STORY_MODE_END"       => Some("[STORY] Scene closed"),
                "IMPRINTING_START"     => Some("[IMPRINT] 60s learning window active"),
                "ARCHITECT_RECOGNIZED" => Some("[ID] Architect confirmed by multimodal SNN"),
                "GLOBAL_WORKSPACE"     => Some("[GW] Deep deduction triggered"),
                "APPEARANCE"           => Some("[SELF] Appearance description"),
                _                      => None,
            };
            if let Some(txt) = ev_txt {
                s.chat_history.push(ChatLine {
                    speaker:"system".into(), text:txt.into(),
                    regions:vec![], story_mode:false, from_stt:false,
                });
            }
        }
        trim_chat(&mut s.chat_history);
    });
    let _ = tick;
}

// ── Result extractors ─────────────────────────────────────────────────────────

pub fn extract_str(d: &pyo3::Bound<'_, PyDict>, key: &str) -> Option<String> {
    d.get_item(key).ok().flatten().and_then(|v| v.extract::<String>().ok())
}
pub fn extract_f64(d: &pyo3::Bound<'_, PyDict>, key: &str) -> f64 {
    d.get_item(key).ok().flatten().and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.0)
}
pub fn extract_f64_or(d: &pyo3::Bound<'_, PyDict>, key: &str, default: f64) -> f64 {
    d.get_item(key).ok().flatten().and_then(|v| v.extract::<f64>().ok()).unwrap_or(default)
}
pub fn extract_u64(d: &pyo3::Bound<'_, PyDict>, key: &str) -> u64 {
    d.get_item(key).ok().flatten().and_then(|v| v.extract::<u64>().ok()).unwrap_or(0)
}
pub fn extract_bool(d: &pyo3::Bound<'_, PyDict>, key: &str) -> bool {
    d.get_item(key).ok().flatten().and_then(|v| v.extract::<bool>().ok()).unwrap_or(false)
}
pub fn extract_region_map(d: &pyo3::Bound<'_, PyDict>, key: &str) -> Vec<(String, f64)> {
    d.get_item(key).ok().flatten()
     .and_then(|v| v.downcast::<PyDict>().ok()
         .map(|dict| dict.iter().filter_map(|(k, v)|
             Some((k.extract::<String>().ok()?, v.extract::<f64>().ok()?))).collect()))
     .unwrap_or_default()
}

pub fn extract_step_result(d: &pyo3::Bound<'_, PyDict>, tick: u64) -> BrainResult {
    BrainResult {
        tick,
        phill_voltage:        extract_f64(d, "phill_voltage"),
        phill_spiked:         extract_bool(d, "phill_spiked"),
        alpha_broca_spikes:   extract_u64(d, "alpha_spikes"),
        alpha_pfc_threshold:  extract_f64(d, "alpha_threshold"),
        alpha_pfc_voltage:    extract_f64(d, "alpha_mem_mean"),
        speech_trigger:       extract_str(d, "speech_trigger"),
        alpha_tts_speaking:   extract_bool(d, "alpha_tts_speaking"),
        active_regions:       vec![],
        energy:               0.0,
        global_workspace:     false,
        voice_trust:          extract_f64(d, "voice_trust"),
        voice_status:         extract_str(d, "voice_status").unwrap_or_else(|| "...".into()),
        phill_gain:           extract_f64(d, "phill_gain"),
        alpha_regions:        extract_region_map(d, "alpha_regions"),
        sem_concepts:         0,
        combined_id:          extract_f64(d, "combined_id"),
        face_present:         extract_bool(d, "face_present"),
        imprint_status:       extract_str(d, "imprint_status").unwrap_or_else(|| "learning".into()),
        camera_active:        extract_bool(d, "camera_active"),
        alpha_vigilance:      extract_bool(d, "alpha_vigilance"),
        alpha_pressure:       extract_f64(d, "alpha_pressure"),
        story_active:         false,
        story_event:          None,
        alpha_babble_count:   extract_u64(d, "alpha_babble_count"),
        alpha_bound_count:    extract_u64(d, "alpha_bound_count"),
        alpha_motor_map_size: extract_u64(d, "alpha_motor_map_size"),
        alpha_voice_esteem:   extract_f64_or(d, "alpha_voice_esteem",   0.5),
        alpha_voice_surprise: extract_f64_or(d, "alpha_voice_surprise", 0.5),
        alpha_da:        extract_f64_or(d, "alpha_da",        0.45),
        alpha_ser:       extract_f64_or(d, "alpha_ser",       0.75),
        alpha_gaba:      extract_f64_or(d, "alpha_gaba",      0.45),
        alpha_arousal:   extract_f64_or(d, "alpha_arousal",   0.0),
        alpha_coord:     extract_f64_or(d, "alpha_coord",     0.4),
        asleep:          extract_bool(d, "asleep"),
        sleep_pressure:  extract_f64_or(d, "sleep_pressure", 0.0),
        alpha_episodes:  extract_u64(d, "alpha_episodes"),
        alpha_ach:   extract_f64_or(d, "alpha_ach",   0.50),
        alpha_ne:    extract_f64_or(d, "alpha_ne",    0.40),
        alpha_oxy:   extract_f64_or(d, "alpha_oxy",   0.30),
        alpha_feeling:          extract_str(d, "alpha_feeling").unwrap_or_else(|| "calm".into()),
        alpha_feel_intensity:   extract_f64_or(d, "alpha_feel_intensity", 0.0),
        alpha_valence:          extract_f64_or(d, "alpha_valence", 0.5),
        alpha_selfness:   extract_f64_or(d, "alpha_selfness",   0.5),
        alpha_drift:      extract_f64_or(d, "alpha_drift",      0.0),
    }
}

/// Public re-export so main.rs can call it directly.
pub fn dispatch_think_result_pub(
    d:            &pyo3::Bound<'_, pyo3::types::PyDict>,
    state:        &std::sync::Arc<arc_swap::ArcSwap<crate::state::SharedState>>,
    story_active: bool,
    tick:         u64,
    brain:        &pyo3::Bound<'_, pyo3::PyAny>,
) {
    dispatch_think_result(d, state, story_active, tick, brain);
}
