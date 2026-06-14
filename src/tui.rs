// src/tui.rs — Full TUI Module
// ==============================
// Layout (top → bottom):
//
//  ┌─ TITLE: voice · identity · mode · camera · story · tick ───────────────┐
//  ├─ PHILL ──────────┬─ ALPHA ──────────────────────────────────────────┤
//  │ 4 gauges         │ 7 anatomical region bars │ 6 region bars            │
//  │ 3 sparklines     │ Broca sparkline          │ Broca sparkline          │
//  │                  │ Vigilance status         │ Insula bar               │
//  │                  │ Thought pressure         │ Thought pressure         │
//  ├─ INNER THOUGHTS (leaked from thought pipes) ─────────────────────────  ┤
//  ├─ CONVERSATION ─────────────────────────────────────────────────────────┤
//  ├─ INPUT (TEXT mode: text box | STT mode: mic status + wake indicator) ──┤
//  ├─ STATUS ───────────────────────────────────────────────────────────────┤
//  └────────────────────────────────────────────────────────────────────────┘
//
// TAB toggles TEXT ↔ STT at any time.
// In TEXT: 'i' opens input box, Enter sends, Esc cancels.
// In STT:  mic is always on, wake words prime the SNN, no explicit send.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;

use arc_swap::ArcSwap;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Gauge, List, ListItem, Paragraph, Scrollbar,
        ScrollbarOrientation, ScrollbarState, Sparkline,
    },
    Frame, Terminal,
};

use crate::state::{InputMode, SharedState};

pub const TUI_FPS:          u64 = 30;
pub const TUI_INTERVAL_MS:  u64 = 1000 / TUI_FPS;

// ── TUI-local view state (scrollback) ──────────────────────────────────────────
// Lives ONLY in the render thread — never in SharedState (which the brain/audio
// threads overwrite every tick). Tracks the conversation scrollback position.
struct UiState {
    chat_scroll:   usize, // lines scrolled UP from the bottom (0 = newest pinned)
    chat_follow:   bool,  // true = auto-stick to the newest line
    last_chat_len: usize, // previous chat length, for append-anchoring while scrolled
    chat_visible:  usize, // last rendered visible row count (used as the page step)
}

impl Default for UiState {
    fn default() -> Self {
        Self { chat_scroll: 0, chat_follow: true, last_chat_len: 0, chat_visible: 1 }
    }
}

// ── Pending input channel ─────────────────────────────────────────────────────
// (text, from_stt) — from_stt=true when it came from voice recognition

pub fn run(
    state:         Arc<ArcSwap<SharedState>>,
    running:       Arc<AtomicBool>,
    pending_input: Arc<Mutex<Option<(String, bool)>>>,
) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(stdout))?;
    let result   = event_loop(&mut term, &state, &running, &pending_input);
    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen)?;
    term.show_cursor()?;
    result
}

fn event_loop(
    term:          &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    state:         &Arc<ArcSwap<SharedState>>,
    running:       &Arc<AtomicBool>,
    pending_input: &Arc<Mutex<Option<(String, bool)>>>,
) -> anyhow::Result<()> {
    let frame_dur = Duration::from_millis(TUI_INTERVAL_MS);
    let mut ui = UiState::default();

    while running.load(Ordering::Relaxed) {
        let t0 = Instant::now();
        let s  = state.load();
        term.draw(|f| draw(f, &s, &mut ui))?;

        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press { continue; }

                // Global: Ctrl+C always exits
                if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
                    running.store(false, Ordering::SeqCst);
                    break;
                }

                // TAB: toggle input mode at any time
                if k.code == KeyCode::Tab {
                    crate::update_state(state, |s| {
                        s.input_mode   = s.input_mode.toggle();
                        s.typing_active = false;
                        s.input_text.clear();
                    });
                    // Tell STT engine about mode change via shared flag
                    // (main.rs reads input_mode and calls stt.set_mode)
                    continue;
                }

                // ── Conversation scrollback ───────────────────────────────────
                // Works in any mode (harmless while typing — these keys aren't
                // used for text entry). PgUp/PgDn page, Home/End jump.
                match k.code {
                    KeyCode::PageUp => {
                        ui.chat_follow = false;
                        ui.chat_scroll = ui.chat_scroll.saturating_add(ui.chat_visible.max(1));
                        continue;
                    }
                    KeyCode::PageDown => {
                        let step = ui.chat_visible.max(1);
                        if ui.chat_scroll <= step {
                            ui.chat_scroll = 0;
                            ui.chat_follow = true;   // back at the bottom → resume following
                        } else {
                            ui.chat_scroll -= step;
                        }
                        continue;
                    }
                    KeyCode::Home => {
                        ui.chat_follow = false;
                        ui.chat_scroll = ui.chat_scroll.saturating_add(100_000); // clamped to top in draw
                        continue;
                    }
                    KeyCode::End => {
                        ui.chat_follow = true;
                        ui.chat_scroll = 0;
                        continue;
                    }
                    _ => {}
                }

                let mode        = state.load().input_mode.clone();
                let typing      = state.load().typing_active;

                match mode {
                    // ── TEXT MODE ─────────────────────────────────────────
                    InputMode::Text => {
                        if typing {
                            match k.code {
                                KeyCode::Enter => {
                                    let txt = state.load().input_text.trim().to_string();
                                    if !txt.is_empty() {
                                        *pending_input.lock().unwrap() = Some((txt, false));
                                    }
                                    crate::update_state(state, |s| {
                                        s.input_text.clear();
                                        s.typing_active = false;
                                    });
                                }
                                KeyCode::Esc => {
                                    crate::update_state(state, |s| {
                                        s.input_text.clear();
                                        s.typing_active = false;
                                    });
                                }
                                KeyCode::Backspace => {
                                    crate::update_state(state, |s| { s.input_text.pop(); });
                                }
                                KeyCode::Char(c) => {
                                    crate::update_state(state, |s| { s.input_text.push(c); });
                                }
                                _ => {}
                            }
                        } else {
                            match k.code {
                                KeyCode::Char('i') => {
                                    crate::update_state(state, |s| { s.typing_active = true; });
                                }
                                KeyCode::Char('q') | KeyCode::Esc => {
                                    running.store(false, Ordering::SeqCst);
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }

                    // ── STT MODE ──────────────────────────────────────────
                    InputMode::Stt => {
                        match k.code {
                            // In STT mode 'q' still quits when not typing
                            KeyCode::Char('q') | KeyCode::Esc if !typing => {
                                running.store(false, Ordering::SeqCst);
                                break;
                            }
                            // Allow manual override: 'i' opens text box even in STT mode
                            KeyCode::Char('i') if !typing => {
                                crate::update_state(state, |s| { s.typing_active = true; });
                            }
                            KeyCode::Enter if typing => {
                                let txt = state.load().input_text.trim().to_string();
                                if !txt.is_empty() {
                                    *pending_input.lock().unwrap() = Some((txt, false));
                                }
                                crate::update_state(state, |s| {
                                    s.input_text.clear();
                                    s.typing_active = false;
                                });
                            }
                            KeyCode::Esc if typing => {
                                crate::update_state(state, |s| {
                                    s.input_text.clear();
                                    s.typing_active = false;
                                });
                            }
                            KeyCode::Backspace if typing => {
                                crate::update_state(state, |s| { s.input_text.pop(); });
                            }
                            KeyCode::Char(c) if typing => {
                                crate::update_state(state, |s| { s.input_text.push(c); });
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let el = t0.elapsed();
        if el < frame_dur { thread::sleep(frame_dur - el); }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// DRAW
// ─────────────────────────────────────────────────────────────────────────────

fn draw(f: &mut Frame, s: &SharedState, ui: &mut UiState) {
    let area = f.size();
    let has_thoughts = !s.thought_history.is_empty();
    let thought_h    = if has_thoughts { 4u16 } else { 0 };
    let has_search   = !s.search_history.is_empty();
    let search_h     = if has_search { 6u16 } else { 0 };

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),             // title
            Constraint::Length(3),             // speaker banner
            Constraint::Min(16),               // brain panels
            Constraint::Length(thought_h),     // inner thoughts
            Constraint::Length(search_h),      // teacher Q&A (emergent)
            Constraint::Length(8),             // conversation
            Constraint::Length(5),             // input panel (taller for STT)
            Constraint::Length(3),             // status
        ])
        .split(area);

    draw_title(f, root[0], s);
    draw_speaker_banner(f, root[1], s);
    draw_brains(f, root[2], s);
    if has_thoughts { draw_thoughts(f, root[3], s); }
    if has_search   { draw_searches(f, root[4], s); }
    draw_chat(f, root[5], s, ui);
    draw_input(f, root[6], s);
    draw_status(f, root[7], s);
}

// ── SPEAKER BANNER ────────────────────────────────────────────────────────────
// Big, can't-miss indicator of which personality is currently vocalizing.
// Uses reverse-video (filled bar) in the persona's color so audio events
// have an unmistakable visual analog while the babble is being heard.
fn draw_speaker_banner(f: &mut Frame, area: Rect, s: &SharedState) {
    let speaking = s.brain.alpha_tts_speaking;

    let (text, color) = if speaking {
        (" >>>  ALPHA SPEAKING  <<< ".to_string(), ALPHA_ACCENT)
    } else {
        (" \u{00B7} silent \u{00B7} ".to_string(), ALPHA_DIM)
    };

    let style = if speaking {
        Style::default()
            .fg(color)
            .add_modifier(Modifier::BOLD | Modifier::REVERSED)
    } else {
        Style::default().fg(ALPHA_DIM)
    };

    let width    = area.width.saturating_sub(2) as usize; // borders
    let pad_each = width.saturating_sub(text.len()) / 2;
    let padded   = format!("{:pad$}{}{:pad$}", "", text, "", pad = pad_each);

    f.render_widget(
        Paragraph::new(Line::from(vec![Span::styled(padded, style)]))
            .block(Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(if speaking { color } else { ALPHA_DIM }))),
        area,
    );
}

// ── TITLE ─────────────────────────────────────────────────────────────────────

fn draw_title(f: &mut Frame, area: Rect, s: &SharedState) {
    let (vc, vl) = voice_style(s.brain.voice_trust, &s.brain.voice_status);
    let (ic, il) = id_style(s.brain.combined_id, &s.brain.imprint_status);

    let mic_str  = if s.mic_active { "MIC:ON " } else { "MIC:OFF" };
    let cam_str  = if s.brain.camera_active { "CAM:ON " } else { "CAM:OFF" };
    let mode_str = match s.input_mode {
        InputMode::Text => "TEXT",
        InputMode::Stt  => "STT ",
    };
    let mode_color = match s.input_mode {
        InputMode::Text => Color::Cyan,
        InputMode::Stt  => Color::Green,
    };
    let gw_str   = if s.brain.global_workspace { " [GW]"      } else { "" };
    let vig_str  = if s.brain.alpha_vigilance  { " [VIGILANCE]" } else { "" };
    let sto_str  = if s.brain.story_active     { " [STORY]"   } else { "" };
    let a_tts    = if s.brain.alpha_tts_speaking  { " TTS" } else { "" };

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  \u{2726} ALPHA v0.5  ",
                Style::default().fg(ALPHA_ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled(mode_str, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
            Span::styled("]", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("  {mic_str}"), Style::default().fg(if s.mic_active { Color::Green } else { Color::Red })),
            Span::styled(format!("  {cam_str}"), Style::default().fg(if s.brain.camera_active { Color::Green } else { Color::DarkGray })),
            Span::styled("  V:", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{vl}"), Style::default().fg(vc).add_modifier(Modifier::BOLD)),
            Span::styled("  ID:", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{il}"), Style::default().fg(ic).add_modifier(Modifier::BOLD)),
            Span::styled(a_tts, Style::default().fg(ALPHA_ACCENT)),
            Span::styled(gw_str, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(vig_str, Style::default().fg(ALPHA_ALERT).add_modifier(Modifier::BOLD)),
            Span::styled(sto_str, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  #{}", s.total_ticks), Style::default().fg(Color::DarkGray)),
        ]))
        .block(Block::default().borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))),
        area,
    );
}

// ── BRAIN PANELS ─────────────────────────────────────────────────────────────

fn draw_brains(f: &mut Frame, area: Rect, s: &SharedState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(26),
            Constraint::Percentage(74),
        ])
        .split(area);
    draw_phill_panel(f, cols[0], s);
    draw_alpha_panel(f, cols[1], s);
}

fn draw_phill_panel(f: &mut Frame, area: Rect, s: &SharedState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(4),
        ])
        .split(area);

    // Phill voltage
    let pv = s.brain.phill_voltage;
    let pc = voltage_color(pv);
    f.render_widget(
        Gauge::default()
            .block(Block::default()
                .title(Span::styled(
                    format!(" PHILL {}  {}",
                        if s.brain.phill_spiked { "*" } else { "" },
                        if s.brain.asleep {
                            format!("[SLEEP zzz {:.0}%]", s.brain.sleep_pressure * 100.0)
                        } else {
                            format!("[awake {:.0}%]", s.brain.sleep_pressure * 100.0)
                        }),
                    Style::default().fg(pc).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)))
            .gauge_style(Style::default().fg(pc).bg(Color::DarkGray))
            .percent((pv * 100.0).clamp(0.0, 100.0) as u16)
            .label(format!("V={:.4}", pv)),
        rows[0],
    );

    // Mic RMS
    let mv = s.mic_volume;
    f.render_widget(
        Gauge::default()
            .block(Block::default().title(" MIC")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)))
            .gauge_style(Style::default()
                .fg(if s.mic_active { Color::Green } else { Color::Red })
                .bg(Color::DarkGray))
            .percent((mv * 100.0).clamp(0.0, 100.0) as u16)
            .label(format!("{:.4}", mv)),
        rows[1],
    );

    // Voice trust
    let tv = s.brain.voice_trust;
    let (tc, _) = voice_style(tv, &s.brain.voice_status);
    f.render_widget(
        Gauge::default()
            .block(Block::default().title(" VOICE")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)))
            .gauge_style(Style::default().fg(tc).bg(Color::DarkGray))
            .percent((tv * 100.0).clamp(0.0, 100.0) as u16)
            .label(format!("{:.0}%", tv * 100.0)),
        rows[2],
    );

    // Identity
    let iv = s.brain.combined_id;
    let (ic, _) = id_style(iv, &s.brain.imprint_status);
    f.render_widget(
        Gauge::default()
            .block(Block::default()
                .title(Span::styled(
                    format!(" ID {}", if s.brain.face_present { "[face]" } else { "" }),
                    Style::default().fg(ic)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)))
            .gauge_style(Style::default().fg(ic).bg(Color::DarkGray))
            .percent((iv * 100.0).clamp(0.0, 100.0) as u16)
            .label(format!("{:.0}%", iv * 100.0)),
        rows[3],
    );

    // Sparklines
    let sp = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(rows[4]);

    for (i, (data, color, label)) in [
        (&s.phill_history, pc, " Phill"),
        (&s.trust_history, tc, " Trust"),
        (&s.id_history,    ic, " ID   "),
    ].iter().enumerate() {
        f.render_widget(
            Sparkline::default()
                .block(Block::default()
                    .title(Span::styled(*label, Style::default().fg(*color)))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)))
                .data(data)
                .style(Style::default().fg(*color))
                .max(100),
            sp[i],
        );
    }
}

// ── Region bar orders ─────────────────────────────────────────────────────────

// ── Cosmic palette ──────────────────────────────────────────────────────────
// Sleek, minimalist, dark space tones with clean, sharp highlights.
const ALPHA_ACCENT: Color = Color::Rgb(120, 200, 255);  // cool starlight cyan
const ALPHA_BRIGHT: Color = Color::Rgb(235, 245, 255);  // near-white highlight
const ALPHA_DIM:    Color = Color::Rgb(78,  92,  128);  // dim indigo-gray
const ALPHA_ALERT:  Color = Color::Rgb(224, 96,  96);   // sparing red (vigilance)

const ALPHA_ORDER: &[(&str, Color)] = &[
    ("thalamus",    Color::Rgb(78,  92,  128)),
    ("temporal",    Color::Rgb(96,  158, 214)),
    ("hippocampus", Color::Rgb(112, 190, 245)),
    ("acc",         Color::Rgb(150, 172, 232)),
    ("insula",      Color::Rgb(176, 152, 236)),
    ("pfc",         Color::Rgb(150, 220, 255)),
    ("broca",       Color::Rgb(235, 245, 255)),
];

fn draw_alpha_panel(f: &mut Frame, area: Rect, s: &SharedState) {
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(9),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(area);

    let accent = if s.brain.alpha_vigilance { ALPHA_ALERT } else { ALPHA_ACCENT };
    let block   = Block::default()
        .title(Span::styled(
            format!(" ALPHA  \u{2767} feels {} {:.0}%  self{:.0}%/{:+.0}%  PFC-thr={:.2}  pressure={:.2}  babble:{}/{} map:{}  voice\u{2665}{:.0}%  pred{:.0}%  DA{:.2} 5HT{:.2} GA{:.2} AR{:.2}  coord{:.0}%  ACh{:.1} NE{:.1} OXY{:.2}",
                    s.brain.alpha_feeling, s.brain.alpha_feel_intensity * 100.0,
                    s.brain.alpha_selfness * 100.0, s.brain.alpha_drift * 100.0,
                    s.brain.alpha_pfc_threshold, s.brain.alpha_pressure,
                    s.brain.alpha_babble_count, s.brain.alpha_bound_count,
                    s.brain.alpha_motor_map_size, s.brain.alpha_voice_esteem * 100.0,
                    (1.0 - s.brain.alpha_voice_surprise) * 100.0,
                    s.brain.alpha_da, s.brain.alpha_ser, s.brain.alpha_gaba, s.brain.alpha_arousal,
                    s.brain.alpha_coord * 100.0,
                    s.brain.alpha_ach, s.brain.alpha_ne, s.brain.alpha_oxy),
            Style::default().fg(accent).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent));
    let region_area = block.inner(inner[0]);
    f.render_widget(block, inner[0]);
    draw_region_bars(f, region_area, &s.brain.alpha_regions, ALPHA_ORDER);

    // Broca sparkline
    f.render_widget(
        Sparkline::default()
            .block(Block::default()
                .title(Span::styled(
                    format!(" Broca  V={:.4}{}", s.brain.alpha_pfc_voltage,
                            if s.brain.alpha_tts_speaking { "  [TTS]" } else { "" }),
                    Style::default().fg(ALPHA_BRIGHT)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(ALPHA_ACCENT)))
            .data(&s.alpha_broca_hist)
            .style(Style::default().fg(ALPHA_BRIGHT))
            .max(96),
        inner[1],
    );

    // Vigilance indicator
    let vig_txt = if s.brain.alpha_vigilance {
        "VIGILANCE  ACC inhibited PFC  skepticism active"
    } else {
        "composed"
    };
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("  {vig_txt}"),
            Style::default().fg(if s.brain.alpha_vigilance { ALPHA_ALERT } else { ALPHA_DIM })))
            .block(Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(ALPHA_DIM))),
        inner[2],
    );

    // Pressure mini-bar
    let np   = s.brain.alpha_pressure.clamp(0.0, 1.0);
    let fill = (np * 16.0) as usize;
    let bar: String = "#".repeat(fill) + &".".repeat(16_usize.saturating_sub(fill));
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("  thought pressure [{bar}] {:.2}", np),
            Style::default().fg(if np > 0.6 { ALPHA_ACCENT } else { ALPHA_DIM }))),
        inner[3],
    );
}

// ── Region bars (Unicode fill) ────────────────────────────────────────────────

fn draw_region_bars(
    f: &mut Frame,
    area: Rect,
    regions: &[(String, f64)],
    order: &[(&str, Color)],
) {
    let map: std::collections::HashMap<&str, f64> =
        regions.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let n = order.len().min(area.height as usize);
    if n == 0 { return; }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints((0..n).map(|_| Constraint::Length(1)).collect::<Vec<_>>())
        .split(area);

    for (i, (name, color)) in order.iter().enumerate().take(n) {
        let act = map.get(*name).copied().unwrap_or(0.0);
        f.render_widget(
            Paragraph::new(region_bar_line(name, act, area.width as usize, *color)),
            rows[i],
        );
    }
}

fn region_bar_line(name: &str, act: f64, width: usize, color: Color) -> Line<'static> {
    let label  = name.to_string();
    let bar_w  = width.saturating_sub(18);
    let filled = (act.clamp(0.0, 1.0) * bar_w as f64) as usize;
    let empty  = bar_w.saturating_sub(filled);
    // Block-fill characters
    let fc = if act > 0.80 { '\u{2588}' }      // █
             else if act > 0.55 { '\u{2593}' }  // ▓
             else if act > 0.28 { '\u{2592}' }  // ▒
             else if act > 0.06 { '\u{2591}' }  // ░
             else { ' ' };
    let bar: String = std::iter::repeat(fc).take(filled)
        .chain(std::iter::repeat('\u{00B7}').take(empty))  // ·
        .collect();
    let active_style = if act > 0.5 {
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    } else if act > 0.1 {
        Style::default().fg(color)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    Line::from(vec![
        Span::styled(format!(" {:<10}", label),
            Style::default().fg(if act > 0.05 { color } else { Color::DarkGray })),
        Span::styled("[", Style::default().fg(Color::DarkGray)),
        Span::styled(bar, active_style),
        Span::styled("]", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.2}", act),
            Style::default().fg(if act > 0.3 { color } else { Color::DarkGray })),
    ])
}

// ── INNER THOUGHTS ────────────────────────────────────────────────────────────

fn draw_thoughts(f: &mut Frame, area: Rect, s: &SharedState) {
    if area.height < 2 { return; }
    let visible = area.height.saturating_sub(2).max(1) as usize;
    let total   = s.thought_history.len();
    let start   = total.saturating_sub(visible);   // tail — show the newest leaks
    let items: Vec<ListItem> = s.thought_history[start..].iter().map(|t| {
        let (label, color) = ("Alpha think | ", ALPHA_ACCENT);
        let _ = &t.speaker;
        ListItem::new(Line::from(vec![
            Span::styled(label, Style::default().fg(color)),
            Span::styled(t.text.clone(),
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
        ]))
    }).collect();

    f.render_widget(
        List::new(items)
            .block(Block::default()
                .title(Span::styled(
                    " INNER THOUGHTS  (thought pipe leaks -- emergent, not scheduled)",
                    Style::default().fg(Color::DarkGray)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))),
        area,
    );

    if total > visible {
        let mut sb = ScrollbarState::new(total).position(start);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight).begin_symbol(None).end_symbol(None),
            area.inner(&Margin { vertical: 1, horizontal: 0 }),
            &mut sb,
        );
    }
}

// ── TEACHER Q&A ───────────────────────────────────────────────────────────────
// Emergent only — fired when SearchCortex pressure crosses threshold. This is
// NOT a web search: the question goes to claude_teacher.py (Claude as a tutor).
// No internet/browsing — just the Anthropic API. Shows newest at the bottom:
//   [HH:MM:SS] WHO -> question  | teaching reply (truncated)
fn draw_searches(f: &mut Frame, area: Rect, s: &SharedState) {
    if area.height < 2 { return; }
    let inner_w = area.width.saturating_sub(4) as usize;
    let visible = area.height.saturating_sub(2).max(1) as usize;
    let total   = s.search_history.len();
    let start   = total.saturating_sub(visible);   // tail — newest questions
    let items: Vec<ListItem> = s.search_history[start..].iter().map(|ev| {
        let (label, color) = ("Alpha ", ALPHA_ACCENT);
        let _ = &ev.speaker;
        // Truncate snippet to fit visible width after the prefix.
        let prefix_len = ev.timestamp.len() + label.len() + ev.query.len() + 10;
        let snippet_max = inner_w.saturating_sub(prefix_len).max(8);
        // Truncate by CHARACTER, never by byte — teacher replies contain
        // multi-byte UTF-8 (em-dash, emoji); a byte-index slice that lands inside
        // one panics the main thread and takes the whole TUI (and brain loop) down.
        let snip = if ev.snippet.chars().count() > snippet_max {
            let cut: String = ev.snippet.chars().take(snippet_max).collect();
            format!("{cut}\u{2026}")
        } else {
            ev.snippet.clone()
        };
        ListItem::new(Line::from(vec![
            Span::styled(format!("[{}] ", ev.timestamp),
                Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{label} "),
                Style::default().fg(color).add_modifier(Modifier::BOLD)),
            Span::styled("\u{2192} ", Style::default().fg(Color::Yellow)),
            Span::styled(ev.query.clone(),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" | ", Style::default().fg(Color::DarkGray)),
            Span::styled(snip, Style::default().fg(Color::Gray)),
        ]))
    }).collect();

    f.render_widget(
        List::new(items)
            .block(Block::default()
                .title(Span::styled(
                    " TEACHER  (emergent -- Alpha asks Claude-as-tutor; NO web -- fires on curiosity / unknown-word / pronunciation pressure)",
                    Style::default().fg(Color::Yellow)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))),
        area,
    );

    if total > visible {
        let mut sb = ScrollbarState::new(total).position(start);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight).begin_symbol(None).end_symbol(None),
            area.inner(&Margin { vertical: 1, horizontal: 0 }),
            &mut sb,
        );
    }
}

// ── CONVERSATION ──────────────────────────────────────────────────────────────

fn draw_chat(f: &mut Frame, area: Rect, s: &SharedState, ui: &mut UiState) {
    let inner_h = area.height.saturating_sub(2) as usize;
    let visible = inner_h.max(1);
    ui.chat_visible = visible;

    let total = s.chat_history.len();

    // Anchor the viewport while scrolled up: if new lines arrived at the bottom
    // since last frame, push the scroll offset up by the same amount so the
    // lines the user is reading don't drift. (No-op while following.)
    if !ui.chat_follow && total > ui.last_chat_len {
        ui.chat_scroll = ui.chat_scroll.saturating_add(total - ui.last_chat_len);
    }
    ui.last_chat_len = total;

    let max_scroll = total.saturating_sub(visible);
    if ui.chat_follow { ui.chat_scroll = 0; }
    ui.chat_scroll = ui.chat_scroll.min(max_scroll);

    let end   = total.saturating_sub(ui.chat_scroll);
    let start = end.saturating_sub(visible);

    let items: Vec<ListItem> = s.chat_history[start..end].iter().map(|line| {
        let (label, color, bold) = match line.speaker.as_str() {
            "alpha"      => ("Alpha       |", ALPHA_ACCENT,  true),
            "nodevortex" => ("NodeVortex  |", Color::Green,   true),
            _            => ("System      |", ALPHA_DIM,      false),
        };
        let reg_tag = if line.regions.is_empty() { String::new() } else {
            format!("  [{}]",
                line.regions.iter()
                    .map(|r| r.chars().take(4).collect::<String>())
                    .collect::<Vec<_>>()
                    .join(">"))
        };
        let stt_badge  = if line.from_stt { " [v]" } else { "" };
        let story_pfx  = if line.story_mode { "[S] " } else { "" };
        let label_style = if bold {
            Style::default().fg(color).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(color)
        };

        ListItem::new(Line::from(vec![
            Span::styled(format!(" {label} "), label_style),
            Span::styled(format!("{story_pfx}{}{stt_badge}", line.text),
                Style::default().fg(Color::White)),
            Span::styled(reg_tag, Style::default().fg(Color::DarkGray)),
        ]))
    }).collect();

    let scrolled = ui.chat_scroll > 0;
    let title_color = if scrolled { Color::Cyan }
                      else if s.brain.story_active { Color::Yellow }
                      else { Color::DarkGray };
    let title_style = if scrolled || s.brain.story_active {
        Style::default().fg(title_color).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(title_color)
    };
    let base = if s.brain.story_active {
        " CONVERSATION  [STORY: NodeVortex / Alpha]"
    } else {
        " CONVERSATION"
    };
    let title_txt = if scrolled {
        format!("{base}  [scrolled +{}  PgDn/End=resume] ", ui.chat_scroll)
    } else {
        format!("{base}  (PgUp/Home=scroll back) ")
    };

    f.render_widget(
        List::new(items)
            .block(Block::default()
                .title(Span::styled(title_txt, title_style))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(title_color))),
        area,
    );

    // Scrollbar on the right border (only when there's overflow to scroll).
    if total > visible {
        let mut sb = ScrollbarState::new(total).position(start);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight).begin_symbol(None).end_symbol(None),
            area.inner(&Margin { vertical: 1, horizontal: 0 }),
            &mut sb,
        );
    }
}

// ── INPUT PANEL ───────────────────────────────────────────────────────────────

fn draw_input(f: &mut Frame, area: Rect, s: &SharedState) {
    match s.input_mode {
        InputMode::Text => draw_text_input(f, area, s),
        InputMode::Stt  => draw_stt_input(f, area, s),
    }
}

fn draw_text_input(f: &mut Frame, area: Rect, s: &SharedState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(2)])
        .split(area);

    let (border_color, title, display) = if s.typing_active {
        let who = if s.brain.story_active { "NodeVortex" } else { "You" };
        (Color::Cyan,
         format!(" [{who}] Enter=send  Esc=cancel  TAB=switch to STT "),
         format!("> {}|", s.input_text))
    } else {
        (Color::DarkGray,
         " TEXT INPUT  (i=type  TAB=switch to STT) ".into(),
         "  Press 'i' to type...".into())
    };

    f.render_widget(
        Paragraph::new(display)
            .style(Style::default().fg(if s.typing_active { Color::White } else { Color::DarkGray }))
            .block(Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(border_color))
                .title(Span::styled(title,
                    Style::default().fg(border_color).add_modifier(Modifier::BOLD)))),
        rows[0],
    );

    // Hint bar
    f.render_widget(
        Paragraph::new(Span::styled(
            "  No commands -- just talk to them; everything they do emerges from their own state.",
            Style::default().fg(Color::DarkGray))),
        rows[1],
    );
}

fn draw_stt_input(f: &mut Frame, area: Rect, s: &SharedState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(2)])
        .split(area);

    // Wake word responsiveness bar
    let ar    = s.stt.alpha_resp.clamp(0.0, 1.0);
    let afill = (ar * 10.0) as usize;
    let a_bar: String = "#".repeat(afill) + &".".repeat(10usize.saturating_sub(afill));

    let mic_icon   = if s.mic_active { "* MIC LIVE *" } else { "[ MIC OFF ]" };
    let last_txt   = if s.stt.last_transcript.is_empty() {
        "...listening...".to_string()
    } else {
        format!("\"{}\"", s.stt.last_transcript)
    };

    let border_color = if s.mic_active { Color::Green } else { Color::Red };

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!("  {mic_icon}  "), Style::default().fg(border_color).add_modifier(Modifier::BOLD)),
            Span::styled(last_txt, Style::default().fg(Color::White)),
            Span::styled(format!("   Alpha:[{a_bar}]"), Style::default().fg(ALPHA_ACCENT)),
            Span::styled(format!("  {}x transcribed", s.stt.total_transcripts), Style::default().fg(Color::DarkGray)),
        ]))
        .block(Block::default()
            .title(Span::styled(
                format!(" STT MODE  ({})  TAB=switch to TEXT  i=manual override ",
                        s.stt.backend),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))),
        rows[0],
    );

    // Wake word hint + manual override hint
    let override_txt = if s.typing_active {
        format!("> {}|  (Enter=send  Esc=cancel)", s.input_text)
    } else {
        "  Say 'Alpha' to wake him.  Press 'i' to type manually.".into()
    };
    f.render_widget(
        Paragraph::new(Span::styled(
            override_txt,
            Style::default().fg(if s.typing_active { Color::White } else { Color::DarkGray }))),
        rows[1],
    );
}

// ── STATUS ────────────────────────────────────────────────────────────────────

fn draw_status(f: &mut Frame, area: Rect, s: &SharedState) {
    let (msg, style) = if let Some(err) = &s.error_msg {
        (err.as_str(), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
    } else {
        ("q=quit  TAB=TEXT/STT  i=type  Ctrl+C=force exit",
         Style::default().fg(Color::DarkGray))
    };

    let sem_str = format!("  brain:{} concepts", s.brain.sem_concepts);
    let imp_str = format!("  imprint:{}", s.brain.imprint_status);
    let sto_str = if s.brain.story_active { "  [STORY]" } else { "" };

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!("  {msg}"), style),
            Span::styled(imp_str, Style::default().fg(Color::DarkGray)),
            Span::styled(sem_str, Style::default().fg(Color::DarkGray)),
            Span::styled(sto_str, Style::default().fg(Color::Yellow)),
        ]))
        .block(Block::default().borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))),
        area,
    );
}

// ── Color helpers ─────────────────────────────────────────────────────────────

fn voltage_color(v: f64) -> Color {
    if v < 0.2 { Color::Blue } else if v < 0.5 { Color::Green }
    else if v < 0.8 { Color::Yellow } else { Color::Red }
}

fn voice_style(trust: f64, status: &str) -> (Color, String) {
    if status.contains("ARCHITECT") {
        (Color::Green, "ARCHITECT-OK".into())
    } else if status.contains("learning") {
        (Color::Yellow, status.into())
    } else if status.contains("uncertain") {
        (Color::Yellow, format!("~{:.0}%", trust * 100.0))
    } else {
        (Color::Red, "stranger".into())
    }
}

fn id_style(id: f64, status: &str) -> (Color, String) {
    if id > 0.80 { (Color::Green,  format!("confirmed{:.0}%", id * 100.0)) }
    else if id > 0.55 { (Color::Cyan,   format!("likely{:.0}%",    id * 100.0)) }
    else if id > 0.30 { (Color::Yellow, format!("~{:.0}%",          id * 100.0)) }
    else if status.contains("learning") { (Color::DarkGray, status.into()) }
    else { (Color::DarkGray, format!("{:.0}%", id * 100.0)) }
}
