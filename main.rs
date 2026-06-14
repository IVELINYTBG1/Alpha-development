// src/main.rs — Alpha v0.5 · Lean Orchestrator

mod state;
mod audio;
mod brain_thread;
mod stt_bridge;
mod tui;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use arc_swap::ArcSwap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use state::{InputMode, SharedState, ChatLine};
use stt_bridge::SttResultBridge;

pub fn update_state(state: &Arc<ArcSwap<SharedState>>, f: impl FnOnce(&mut SharedState)) {
    let cur  = state.load();
    let mut next = (**cur).clone();
    f(&mut next);
    state.store(Arc::new(next));
}

fn main() -> anyhow::Result<()> {
    let running       = Arc::new(AtomicBool::new(true));
    let state         = Arc::new(ArcSwap::from_pointee(SharedState::default()));
    let pending_input: Arc<Mutex<Option<(String, bool)>>> = Arc::new(Mutex::new(None));

    // Audio → STT sample buffer
    let stt_audio_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // Audio push fn for STT
    let stt_push_buf = Arc::clone(&stt_audio_buf);
    let stt_audio_push: audio::SttPushFn = Arc::new(move |samples: &[f32]| {
        if let Ok(mut b) = stt_push_buf.try_lock() {
            b.extend_from_slice(samples);
            let max = 16000 * 5;
            if b.len() > max { let d = b.len() - max; b.drain(..d); }
        }
    });

    // ── Audio thread ──────────────────────────────────────────────────────────
    {
        let s = Arc::clone(&state);
        let r = Arc::clone(&running);
        let push = stt_audio_push.clone();
        thread::Builder::new()
            .name("audio".into())
            .spawn(move || audio::audio_thread(s, r, Some(push)))?;
    }

    // ── Brain + STT combined thread ───────────────────────────────────────────
    {
        let s       = Arc::clone(&state);
        let r       = Arc::clone(&running);
        let p       = Arc::clone(&pending_input);
        let stt_buf = Arc::clone(&stt_audio_buf);

        thread::Builder::new()
            .name("brain-stt".into())
            .spawn(move || {
                Python::with_gil(|py| {
                    // ── Load brain.py ─────────────────────────────────────────
                    let brain_src = include_str!("../brain.py");
                    let brain_mod = match PyModule::from_code_bound(
                        py, brain_src, "brain.py", "brain") {
                        Ok(m)  => m,
                        Err(e) => {
                            update_state(&s, |st| {
                                st.error_msg = Some(format!("brain.py: {e}"));
                                st.chat_history.push(ChatLine::system(
                                    format!("[ERROR] brain.py: {e}")));
                            });
                            return;
                        }
                    };

                    let brain = match brain_mod
                        .getattr("NeuromorphicBrain")
                        .and_then(|c| c.call0()) {
                        Ok(b)  => b,
                        Err(e) => {
                            update_state(&s, |st| {
                                st.error_msg = Some(format!("Brain init: {e}"));
                                st.chat_history.push(ChatLine::system(
                                    format!("[ERROR] Brain init: {e}")));
                            });
                            return;
                        }
                    };

                    // Show init messages
                    if let Ok(msgs) = brain_mod.getattr("_INIT_MESSAGES")
                        .and_then(|o| o.extract::<Vec<String>>()) {
                        update_state(&s, |st| {
                            for msg in &msgs {
                                st.chat_history.push(ChatLine {
                                    speaker:"system".into(),
                                    text:format!("[init] {msg}"),
                                    regions:vec![], story_mode:false, from_stt:false,
                                });
                            }
                        });
                    }

                    // ── Load stt_engine.py — uses get_result() polling ────────
                    let stt_src = include_str!("../stt_engine.py");
                    let stt_engine: Option<Py<PyAny>> = (|| {
                        let stt_mod = PyModule::from_code_bound(
                            py, stt_src, "stt_engine.py", "stt_engine")
                            .map_err(|e| { eprintln!("[STT] load: {e}"); e })?;

                        let create_fn = stt_mod.getattr("create_stt_engine")
                            .map_err(|e| { eprintln!("[STT] create_stt_engine: {e}"); e })?;

                        // STTEngine takes language + mode, no callback
                        let engine = create_fn.call(
                            (),
                            Some(&pyo3::types::PyDict::new_bound(py)),
                        ).or_else(|_| create_fn.call0())
                        .map_err(|e| { eprintln!("[STT] init: {e}"); e })?;

                        // Report availability
                        let available: bool = engine.getattr("available")
                            .and_then(|v| v.extract()).unwrap_or(false);
                        let err_msg: Option<String> = engine.getattr("error_msg")
                            .and_then(|v| v.extract()).unwrap_or(None);
                        let familiarity: String = engine
                            .getattr("familiarity_label")
                            .and_then(|v| v.extract())
                            .unwrap_or_else(|_| "learning".into());

                        update_state(&s, |st| {
                            st.stt.backend = if available {
                                "vosk".into()
                            } else {
                                "silent".into()
                            };
                            if let Some(e) = &err_msg {
                                st.chat_history.push(ChatLine::system(
                                    format!("[STT] {e}")));
                            } else if available {
                                st.chat_history.push(ChatLine::system(
                                    format!("[STT] active  familiarity:{familiarity}")));
                            }
                        });

                        Ok::<_, PyErr>(engine.into())
                    })().ok();

                    if stt_engine.is_none() {
                        update_state(&s, |st| {
                            st.stt.backend = "disabled".into();
                            st.chat_history.push(ChatLine::system(
                                "[STT] disabled — pip install vosk sounddevice"));
                        });
                    }

                    // ── Main loop ─────────────────────────────────────────────
                    let mut tick: u64 = 0;

                    while r.load(Ordering::Relaxed) {
                        let t0 = std::time::Instant::now();
                        tick  += 1;

                        // ── STT: mode sync + get_result() poll ────────────────
                        if let Some(ref eng_py) = stt_engine {
                            let eng = eng_py.bind(py);

                            // Sync TEXT/STT mode
                            let cur_mode = s.load().input_mode.clone();
                            let mode_str = match cur_mode {
                                InputMode::Stt  => "ALWAYS_ON",
                                InputMode::Text => "OFF",
                            };
                            let _ = eng.call_method1("set_mode", (mode_str,));

                            // Poll result (non-blocking)
                            // STTEngine.get_result() returns STTResult or None
                            if let Ok(result_obj) = eng.call_method0("get_result") {
                                if !result_obj.is_none() {
                                    let text: String = result_obj.getattr("text")
                                        .and_then(|v| v.extract())
                                        .unwrap_or_default();
                                    let addressed: Option<String> = result_obj
                                        .getattr("addressed")
                                        .and_then(|v| v.extract())
                                        .unwrap_or(None);
                                    let via_wake: bool = result_obj.getattr("via_wake")
                                        .and_then(|v| v.extract())
                                        .unwrap_or(false);

                                    let wake_alpha   = addressed.as_deref()
                                        .map(|a| a=="alpha"||a=="both").unwrap_or(false);
                                    let wake_alpha = addressed.as_deref()
                                        .map(|a| a=="alpha"||a=="both").unwrap_or(false);

                                    // Get familiarity score for TUI
                                    let familiarity: String = eng
                                        .getattr("familiarity_label")
                                        .and_then(|v| v.extract())
                                        .unwrap_or_else(|_| "learning".into());
                                    let auto_resp: bool = eng
                                        .getattr("auto_respond")
                                        .and_then(|a| a.getattr("should_auto_respond"))
                                        .and_then(|v| v.extract())
                                        .unwrap_or(false);

                                    update_state(&s, |st| {
                                        st.stt.last_transcript   = text.clone();
                                        st.stt.wake_alpha         = wake_alpha;
                                        st.stt.wake_alpha       = wake_alpha;
                                        st.stt.total_transcripts += 1;
                                        st.stt.listening         = true;
                                        // Re-use alpha_resp field for familiarity
                                        st.stt.alpha_resp   = if wake_alpha   { 0.9 } else { 0.3 };
                                        st.stt.alpha_resp = if wake_alpha { 0.9 } else { 0.3 };
                                    });

                                    // Route to think() — STTEngine already
                                    // handled wake word gating via _should_route()
                                    if !text.is_empty() {
                                        let mut pi = p.lock().unwrap();
                                        if pi.is_none() {
                                            *pi = Some((text, true));
                                        }
                                    }
                                }
                            }

                            // Push audio to STT engine's internal mic stream
                            // (STTEngine opens its own sounddevice stream, but
                            //  we also push from CPAL for when sounddevice lacks access)
                            let samples: Vec<f32> = {
                                let mut b = stt_buf.lock().unwrap();
                                let out = b.clone(); b.clear(); out
                            };
                            if !samples.is_empty() {
                                // STTEngine doesn't have push_audio — it uses its
                                // own sd.InputStream. The stt_buf is a backup path
                                // that future versions can use. Drop for now.
                                let _ = samples;
                            }
                        }

                        // ── Brain step() ──────────────────────────────────────
                        let cur   = s.load();
                        let mic   = cur.mic_volume;
                        let feats = cur.audio_features.clone();
                        drop(cur);

                        let py_feats = PyList::new_bound(py, &feats.to_vec());
                        if let Ok(res) = brain.call_method1("step", (mic, py_feats)) {
                            if let Ok(d) = res.downcast::<PyDict>() {
                                let br = brain_thread::extract_step_result(d, tick);
                                update_state(&s, |st| {
                                    state::push_spark(&mut st.phill_history,
                                        (br.phill_voltage*100.0) as u64);
                                    state::push_spark(&mut st.trust_history,
                                        (br.voice_trust*100.0) as u64);
                                    state::push_spark(&mut st.id_history,
                                        (br.combined_id*100.0) as u64);
                                    state::push_spark(&mut st.alpha_broca_hist,
                                        br.alpha_broca_spikes.min(32)*3);
                                    state::push_spark(&mut st.sim_broca_hist,
                                        br.alpha_broca_spikes.min(32)*3);
                                    st.brain       = br;
                                    st.total_ticks = tick;
                                });
                            }
                        }

                        // ── Poll thought pipe leaks ───────────────────────────
                        if let Ok(leaked) = brain.call_method0("get_leaked_thoughts") {
                            if let Ok(thoughts) = leaked.extract::<Vec<(String, String)>>() {
                                if !thoughts.is_empty() {
                                    update_state(&s, |st| {
                                        for (who, thought) in &thoughts {
                                            st.thought_history.push(ChatLine {
                                                speaker: format!("thought_{who}"),
                                                text: thought.clone(),
                                                regions: vec![], story_mode: false,
                                                from_stt: false,
                                            });
                                            if st.thought_history.len()
                                                > state::THOUGHT_HISTORY {
                                                st.thought_history.remove(0);
                                            }
                                        }
                                    });
                                }
                            }
                        }

                        // ── think() dispatch ──────────────────────────────────
                        if let Some((text, from_stt)) = p.lock().unwrap().take() {
                            let story_active = s.load().brain.story_active;
                            let speaker = if story_active { "NodeVortex" }
                                          else if from_stt { "Voice" }
                                          else { "You" };
                            update_state(&s, |st| {
                                st.chat_history.push(ChatLine {
                                    speaker: "nodevortex".into(),
                                    text: format!("[{speaker}] {text}"),
                                    regions: vec![], story_mode: story_active,
                                    from_stt,
                                });
                            });
                            match brain.call_method1("think", (text.as_str(),)) {
                                Ok(res) => {
                                    if let Ok(d) = res.downcast::<PyDict>() {
                                        brain_thread::dispatch_think_result_pub(
                                            d, &s, story_active, tick, &brain);
                                    }
                                }
                                Err(e) => {
                                    update_state(&s, |st| {
                                        st.chat_history.push(ChatLine {
                                            speaker: "system".into(),
                                            text: format!("think() error: {e}"),
                                            regions: vec![], story_mode: false,
                                            from_stt: false,
                                        });
                                    });
                                }
                            }
                        }

                        // ── 20Hz pace ─────────────────────────────────────────
                        let el     = t0.elapsed();
                        let budget = Duration::from_millis(
                            brain_thread::BRAIN_INTERVAL_MS);
                        if el < budget { thread::sleep(budget - el); }
                    }
                });
            })?;
    }

    // ── TUI (main thread) ─────────────────────────────────────────────────────
    tui::run(
        Arc::clone(&state),
        Arc::clone(&running),
        Arc::clone(&pending_input),
    )?;

    running.store(false, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(300));
    Ok(())
}
