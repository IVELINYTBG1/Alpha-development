// src/brain_thread.rs — Brain Loop Thread
// Calls brain.step() at 20Hz and brain.think() on pending input.
// Also polls the thought pipe and dispatches STT transcriptions.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;

use arc_swap::ArcSwap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::state::{
    BrainResult, ChatLine, SharedState, SttState,
    push_spark, trim_chat, SPARKLINE_LEN,
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
                        push_spark(&mut s.phill_history,   (br.phill_voltage * 100.0) as u64);
                        push_spark(&mut s.trust_history,   (br.voice_trust   * 100.0) as u64);
                        push_spark(&mut s.id_history,      (br.combined_id   * 100.0) as u64);
                        push_spark(&mut s.nova_broca_hist, br.nova_broca_spikes.min(32) * 3);
                        push_spark(&mut s.sim_broca_hist,  br.simona_broca_spikes.min(32) * 3);
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

            // ── STT status sync ───────────────────────────────────────────
            {
                let sr = stt_results.lock().unwrap();
                if let Some(ref bridge) = *sr {
                    crate::update_state(&state, |s| {
                        s.stt.last_transcript   = bridge.last_text.clone();
                        s.stt.wake_nova         = bridge.wake_nova;
                        s.stt.wake_simona       = bridge.wake_simona;
                        s.stt.nova_resp         = bridge.nova_resp;
                        s.stt.simona_resp        = bridge.simona_resp;
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
            let el     = t0.elapsed();
            let budget = Duration::from_millis(BRAIN_INTERVAL_MS);
            if el < budget { thread::sleep(budget - el); }
        }
    });
}

fn dispatch_think_result(
    d:            &pyo3::Bound<'_, PyDict>,
    state:        &Arc<ArcSwap<SharedState>>,
    story_active: bool,
    tick:         u64,
    brain:        &pyo3::Bound<'_, PyAny>,
) {
    let nova_t    = extract_str(d, "nova");
    let simona_t  = extract_str(d, "simona");
    let energy    = extract_f64(d, "energy");
    let gw        = extract_bool(d, "global_workspace");
    let tticks    = extract_u64(d, "think_ticks");
    let story_ev  = extract_str(d, "story_event");
    let story_on  = extract_bool(d, "story_active");
    let act_reg: Vec<String> = d.get_item("active_regions").ok().flatten()
        .and_then(|v| v.extract().ok()).unwrap_or_default();
    let nova_r    = extract_region_map(d, "nova_regions");
    let simona_r  = extract_region_map(d, "simona_regions");

    let sem_ct: usize = brain.call_method0("introspect").ok()
        .and_then(|r| r.downcast::<PyDict>().ok()
            .and_then(|d| d.get_item("sem_concepts").ok().flatten()
                .and_then(|v| v.extract().ok())))
        .unwrap_or(0);

    crate::update_state(state, |s| {
        s.brain.active_regions   = act_reg.clone();
        s.brain.energy           = energy;
        s.brain.global_workspace = gw;
        s.brain.nova_regions     = nova_r;
        s.brain.simona_regions   = simona_r;
        s.brain.story_active     = story_on;
        s.brain.story_event      = story_ev.clone();
        s.brain.sem_concepts     = sem_ct;

        if let Some(t) = nova_t {
            s.chat_history.push(ChatLine {
                speaker: "nova".into(), text: t,
                regions: act_reg.clone(), story_mode: story_on, from_stt: false,
            });
        }
        if let Some(t) = simona_t {
            s.chat_history.push(ChatLine {
                speaker: "simona".into(), text: t,
                regions: vec![], story_mode: story_on, from_stt: false,
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
}

// ── Result extractors ─────────────────────────────────────────────────────────

pub fn extract_str(d: &pyo3::Bound<'_, PyDict>, key: &str) -> Option<String> {
    d.get_item(key).ok().flatten().and_then(|v| v.extract::<String>().ok())
}
pub fn extract_f64(d: &pyo3::Bound<'_, PyDict>, key: &str) -> f64 {
    d.get_item(key).ok().flatten().and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.0)
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
        nova_broca_spikes:    extract_u64(d, "nova_spikes"),
        simona_broca_spikes:  extract_u64(d, "simona_spikes"),
        nova_pfc_threshold:   extract_f64(d, "nova_threshold"),
        simona_broca_thr:     extract_f64(d, "simona_threshold"),
        nova_pfc_voltage:     extract_f64(d, "nova_mem_mean"),
        simona_broca_voltage: extract_f64(d, "simona_mem_mean"),
        speech_trigger:       extract_str(d, "speech_trigger"),
        nova_tts_speaking:    extract_bool(d, "nova_tts_speaking"),
        simona_tts_speaking:  extract_bool(d, "simona_tts_speaking"),
        active_regions:       vec![],
        energy:               0.0,
        global_workspace:     false,
        voice_trust:          extract_f64(d, "voice_trust"),
        voice_status:         extract_str(d, "voice_status").unwrap_or_else(|| "...".into()),
        phill_gain:           extract_f64(d, "phill_gain"),
        nova_regions:         extract_region_map(d, "nova_regions"),
        simona_regions:       extract_region_map(d, "simona_regions"),
        sem_concepts:         0,
        combined_id:          extract_f64(d, "combined_id"),
        face_present:         extract_bool(d, "face_present"),
        imprint_status:       extract_str(d, "imprint_status").unwrap_or_else(|| "learning".into()),
        camera_active:        extract_bool(d, "camera_active"),
        nova_vigilance:       extract_bool(d, "nova_vigilance"),
        nova_pressure:        extract_f64(d, "nova_pressure"),
        simona_pressure:      extract_f64(d, "simona_pressure"),
        story_active:         false,
        story_event:          None,
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
