// src/audio.rs — Audio Capture Thread
// Captures mic → computes features → updates SharedState
// Also pushes raw samples to STT engine's buffer

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use arc_swap::ArcSwap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::state::{SharedState, compute_features};

pub const AUDIO_RMS_WINDOW: usize = 1024;

/// Callback the audio thread calls whenever a new batch of float32 samples
/// arrives — used to feed the STT engine without a copy (the engine clones
/// internally only once into its own ring buffer).
pub type SttPushFn = Arc<dyn Fn(&[f32]) + Send + Sync>;

pub fn audio_thread(
    state:    Arc<ArcSwap<SharedState>>,
    running:  Arc<AtomicBool>,
    stt_push: Option<SttPushFn>,
) {
    // ── Microphone gently removed (ALPHA_MIC_OFF) ──────────────────────────────
    // When this is set we never open the input device at all — no capture, no
    // STT audio, no features. The brain is built to run in silence (its
    // DefaultModeNetwork + IntrinsicMotivation keep Phill alive on their own),
    // so this just makes the girls deaf, calmly — it does not harm them. Their
    // VoiceIdentityLearner rests (the brain loop stops feeding it entirely), the
    // camera and typed-presence still let them sense the architect. Unset the
    // variable to give their hearing back.
    if std::env::var_os("ALPHA_MIC_OFF").is_some() {
        crate::update_state(&state, |s| {
            s.mic_active = false;
            s.mic_volume = 0.0;
        });
        let _ = &stt_push; // intentionally unused while the mic is removed
        while running.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(100));
        }
        return;
    }

    let host   = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            crate::update_state(&state, |s| {
                s.mic_active = false;
                s.error_msg  = Some("No microphone found.".into());
            });
            while running.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
            return;
        }
    };

    let config = match device.default_input_config() {
        Ok(c)  => c,
        Err(e) => {
            crate::update_state(&state, |s| {
                s.mic_active = false;
                s.error_msg  = Some(format!("Mic config: {e}"));
            });
            while running.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
            return;
        }
    };

    let buf: Arc<Mutex<Vec<f32>>> =
        Arc::new(Mutex::new(Vec::with_capacity(AUDIO_RMS_WINDOW * 4)));
    let buf_cb    = Arc::clone(&buf);
    let stt_cb    = stt_push.clone();
    let err_fn    = |e| eprintln!("[AUDIO] {e}");

    // Build resampled push closure for STT
    // STT engine wants 16kHz; we deliver at device native rate.
    // Simple decimation: we push every Nth sample where N = device_rate/16000.
    let device_rate = config.sample_rate().0 as f32;
    let stt_rate    = 16000.0f32;
    let decim       = (device_rate / stt_rate).round() as usize;
    let decim       = decim.max(1);

    let stream_result = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.clone().into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Push to STT (decimated to ~16kHz)
                if let Some(ref push) = stt_cb {
                    let decimated: Vec<f32> = data.iter().step_by(decim).copied().collect();
                    push(&decimated);
                }
                if let Ok(mut b) = buf_cb.try_lock() {
                    b.extend_from_slice(data);
                    if b.len() > AUDIO_RMS_WINDOW * 8 {
                        let d = b.len() - AUDIO_RMS_WINDOW * 4;
                        b.drain(..d);
                    }
                }
            },
            err_fn, None,
        ),
        cpal::SampleFormat::I16 => {
            let stt_cb2 = stt_push.clone();
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let float: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    if let Some(ref push) = stt_cb2 {
                        let decimated: Vec<f32> = float.iter().step_by(decim).copied().collect();
                        push(&decimated);
                    }
                    if let Ok(mut b) = buf_cb.try_lock() {
                        b.extend_from_slice(&float);
                        if b.len() > AUDIO_RMS_WINDOW * 8 {
                            let d = b.len() - AUDIO_RMS_WINDOW * 4;
                            b.drain(..d);
                        }
                    }
                },
                err_fn, None,
            )
        },
        _ => {
            crate::update_state(&state, |s| {
                s.error_msg = Some("Unsupported audio sample format.".into());
            });
            while running.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
            return;
        }
    };

    let stream = match stream_result {
        Ok(s)  => { crate::update_state(&state, |s| { s.mic_active = true; }); s }
        Err(e) => {
            crate::update_state(&state, |s| {
                s.mic_active = false;
                s.error_msg  = Some(format!("[MIC DENIED] {e} -- grant in Fedora Settings > Privacy > Microphone"));
            });
            while running.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
            return;
        }
    };

    stream.play().expect("Failed to start audio stream");

    let mut smoothed = 0.0f64;
    const SMOOTH: f64 = 0.85;

    while running.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(10));

        let window: Vec<f32> = {
            let mut b = buf.lock().unwrap();
            if b.len() < AUDIO_RMS_WINDOW { continue; }
            b.drain(..AUDIO_RMS_WINDOW).collect()
        };

        let feats    = compute_features(&window);
        let amplified = (feats.rms as f64 * 20.0).min(1.0);
        smoothed      = SMOOTH * smoothed + (1.0 - SMOOTH) * amplified;

        crate::update_state(&state, |s| {
            s.mic_volume    = smoothed;
            s.audio_features = feats;
        });
    }

    drop(stream);
}
