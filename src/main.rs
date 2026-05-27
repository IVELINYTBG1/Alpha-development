// src/main.rs — Nova & Simona v0.5 · Lean Orchestrator

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
    let stt_results:   Arc<Mutex<Option<SttResultBridge>>> = Arc::new(Mutex::new(None));

    // Audio → STT sample buffer (audio thread writes, brain-stt thread reads)
    let stt_audio_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // STT push fn for audio thread
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

    // ── Brain + STT thread ────────────────────────────────────────────────────
    {
        let s        = Arc::clone(&state);
        let r        = Arc::clone(&running);
        let p        = Arc::clone(&pending_input);
        let sr       = Arc::clone(&stt_results);
        let stt_buf  = Arc::clone(&stt_audio_buf);

        thread::Builder::new()
            .name("brain-stt".into())
            .spawn(move || {
                Python::with_gil(|py| {
                    // ── Load brain.py ─────────────────────────────────────────
                    let brain_src = include_str!("../brain.py");
                    let brain_mod = match PyModule::from_code_bound(py, brain_src, "brain.py", "brain") {
                        Ok(m)  => m,
                        Err(e) => {
                            update_state(&s, |st| {
                                st.error_msg = Some(format!("brain.py: {e}"));
                                st.chat_history.push(ChatLine::system(format!("[ERROR] brain.py failed: {e}")));
                            });
                            return;
                        }
                    };

                    let brain = match brain_mod.getattr("NeuromorphicBrain")
                        .and_then(|c| c.call0()) {
                        Ok(b)  => b,
                        Err(e) => {
                            update_state(&s, |st| {
                                st.error_msg = Some(format!("NeuromorphicBrain init: {e}"));
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

                    // ── Load stt_engine.py (optional — degrades gracefully) ────
                    let stt_src = include_str!("../stt_engine.py");
                    let stt_engine_opt: Option<(Py<PyAny>, Py<PyAny>)> = (|| {
                        let stt_mod = PyModule::from_code_bound(py, stt_src, "stt_engine.py", "stt_engine")
                            .map_err(|e| { eprintln!("[STT] module load failed: {e}"); e })?;

                        let queue_mod = py.import_bound("queue")
                            .map_err(|e| { eprintln!("[STT] queue import failed: {e}"); e })?;
                        let py_queue  = queue_mod.call_method0("Queue")?;
                        let cb        = py_queue.getattr("put")?;

                        let create_fn = stt_mod.getattr("create_stt_engine")
                            .map_err(|e| { eprintln!("[STT] create_stt_engine not found: {e}"); e })?;

                        let engine = create_fn.call1((cb,))
                            .map_err(|e| { eprintln!("[STT] engine init failed: {e}"); e })?;

                        // Get backend name for display
                        let backend: String = engine.getattr("backend_name")
                            .and_then(|v| v.extract()).unwrap_or_else(|_| "unknown".into());

                        update_state(&s, |st| {
                            st.stt.backend  = backend.clone();
                            st.stt.listening = false;
                            st.chat_history.push(ChatLine::system(
                                format!("[STT] backend: {backend}  |  TAB to switch TEXT/STT")
                            ));
                        });

                        Ok::<_, PyErr>((engine.into(), py_queue.into()))
                    })().ok();

                    if stt_engine_opt.is_none() {
                        update_state(&s, |st| {
                            st.stt.backend = "disabled".into();
                            st.chat_history.push(ChatLine::system(
                                "[STT] disabled — install: pip install faster-whisper soundfile"
                            ));
                        });
                    }

                    // ── Main loop ─────────────────────────────────────────────
                    let mut tick: u64 = 0;

                    while r.load(Ordering::Relaxed) {
                        let t0 = std::time::Instant::now();
                        tick  += 1;

                        // ── STT mode sync + audio push ────────────────────────
                        if let Some((ref engine, ref py_queue)) = stt_engine_opt {
                            let engine = engine.bind(py);
                            let queue  = py_queue.bind(py);

                            // Sync mode (TEXT/STT) from shared state
                            let cur_mode = s.load().input_mode.clone();
                            let is_stt = matches!(cur_mode, InputMode::Stt);
                            if let Ok(stt_mod) = PyModule::from_code_bound(
                                py, stt_src, "stt_engine.py", "stt_engine") {
                                if let Ok(mode_cls) = stt_mod.getattr("STTMode") {
                                    let attr = if is_stt { "STT" } else { "TEXT" };
                                    if let Ok(mode_val) = mode_cls.getattr(attr) {
                                        let _ = engine.call_method1("set_mode", (mode_val,));
                                    }
                                }
                            }

                            // Push audio samples
                            let samples: Vec<f32> = {
                                let mut b = stt_buf.lock().unwrap();
                                let out = b.clone(); b.clear(); out
                            };
                            if !samples.is_empty() {
                                let py_samples = PyList::new_bound(py, &samples);
                                let _ = engine.call_method1("push_audio", (py_samples,));
                            }

                            // Poll STT results (non-blocking)
                            if let Ok(item) = queue.call_method1("get_nowait", ()) {
                                let text: String = item.getattr("text")
                                    .and_then(|v| v.extract()).unwrap_or_default();
                                let wake_nova:   bool = item.getattr("wake_nova")
                                    .and_then(|v| v.extract()).unwrap_or(false);
                                let wake_simona: bool = item.getattr("wake_simona")
                                    .and_then(|v| v.extract()).unwrap_or(false);
                                let is_ambient:  bool = item.getattr("is_ambient")
                                    .and_then(|v| v.extract()).unwrap_or(true);

                                // Update STT bridge
                                {
                                    let mut b = sr.lock().unwrap();
                                    let bridge = b.get_or_insert_with(SttResultBridge::default);
                                    bridge.last_text   = text.clone();
                                    bridge.wake_nova   = wake_nova;
                                    bridge.wake_simona = wake_simona;
                                    bridge.count      += 1;
                                    bridge.listening   = is_stt;

                                    if let Ok(status_d) = engine.call_method0("status")
                                        .and_then(|r| r.extract::<std::collections::HashMap<String,PyObject>>()) {
                                        // best-effort extract responsiveness
                                    }
                                }

                                // Sync wake state to TUI
                                update_state(&s, |st| {
                                    st.stt.last_transcript   = text.clone();
                                    st.stt.wake_nova         = wake_nova;
                                    st.stt.wake_simona       = wake_simona;
                                    st.stt.total_transcripts += 1;
                                    st.stt.listening         = is_stt;
                                });

                                // Send to think() if meaningful voice input
                                if !text.is_empty() && !is_ambient && is_stt {
                                    let mut pi = p.lock().unwrap();
                                    if pi.is_none() {
                                        *pi = Some((text, true));
                                    }
                                }
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
                                    state::push_spark(&mut st.phill_history, (br.phill_voltage*100.0) as u64);
                                    state::push_spark(&mut st.trust_history, (br.voice_trust*100.0) as u64);
                                    state::push_spark(&mut st.id_history,    (br.combined_id*100.0) as u64);
                                    state::push_spark(&mut st.nova_broca_hist, br.nova_broca_spikes.min(32)*3);
                                    state::push_spark(&mut st.sim_broca_hist,  br.simona_broca_spikes.min(32)*3);
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
                                                regions: vec![], story_mode: false, from_stt: false,
                                            });
                                            if st.thought_history.len() > state::THOUGHT_HISTORY {
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
                            let speaker_name = if story_active { "NodeVortex" }
                                               else if from_stt { "Voice" }
                                               else { "You" };
                            update_state(&s, |st| {
                                st.chat_history.push(ChatLine {
                                    speaker: "nodevortex".into(),
                                    text: format!("[{speaker_name}] {text}"),
                                    regions: vec![], story_mode: story_active, from_stt,
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
                                            speaker:"system".into(),
                                            text: format!("think() error: {e}"),
                                            regions:vec![], story_mode:false, from_stt:false,
                                        });
                                    });
                                }
                            }
                        }

                        // ── 20Hz pace ─────────────────────────────────────────
                        let el     = t0.elapsed();
                        let budget = Duration::from_millis(brain_thread::BRAIN_INTERVAL_MS);
                        if el < budget { thread::sleep(budget - el); }
                    }
                });
            })?;
    }

    // ── TUI (main thread — owns the terminal) ─────────────────────────────────
    tui::run(Arc::clone(&state), Arc::clone(&running), Arc::clone(&pending_input))?;

    running.store(false, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(300));
    Ok(())
}
