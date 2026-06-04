// src/tui.rs — Full TUI Module
// ==============================
// Layout (top → bottom):
//
//  ┌─ TITLE: voice · identity · mode · camera · story · tick ───────────────┐
//  ├─ PHILL ──────────┬─ NOVA (19) ─────────────┬─ SIMONA (8) ─────────────┤
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
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline},
    Frame, Terminal,
};

use crate::state::{InputMode, SharedState};

pub const TUI_FPS:          u64 = 30;
pub const TUI_INTERVAL_MS:  u64 = 1000 / TUI_FPS;

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

    while running.load(Ordering::Relaxed) {
        let t0 = Instant::now();
        let s  = state.load();
        term.draw(|f| draw(f, &s))?;

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

fn draw(f: &mut Frame, s: &SharedState) {
    let area = f.size();
    let has_thoughts = !s.thought_history.is_empty();
    let thought_h    = if has_thoughts { 4u16 } else { 0 };

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),             // title
            Constraint::Min(16),               // brain panels
            Constraint::Length(thought_h),     // inner thoughts
            Constraint::Length(8),             // conversation
            Constraint::Length(5),             // input panel (taller for STT)
            Constraint::Length(3),             // status
        ])
        .split(area);

    draw_title(f, root[0], s);
    draw_brains(f, root[1], s);
    if has_thoughts { draw_thoughts(f, root[2], s); }
    draw_chat(f, root[3], s);
    draw_input(f, root[4], s);
    draw_status(f, root[5], s);
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
    let vig_str  = if s.brain.nova_vigilance   { " [VIGILANCE]" } else { "" };
    let sto_str  = if s.brain.story_active     { " [STORY]"   } else { "" };
    let n_tts    = if s.brain.nova_tts_speaking   { " N-TTS" } else { "" };
    let s_tts    = if s.brain.simona_tts_speaking { " S-TTS" } else { "" };

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  NOVA & SIMONA v0.5  ",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("[", Style::default().fg(Color::DarkGray)),
            Span::styled(mode_str, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
            Span::styled("]", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("  {mic_str}"), Style::default().fg(if s.mic_active { Color::Green } else { Color::Red })),
            Span::styled(format!("  {cam_str}"), Style::default().fg(if s.brain.camera_active { Color::Green } else { Color::DarkGray })),
            Span::styled("  V:", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{vl}"), Style::default().fg(vc).add_modifier(Modifier::BOLD)),
            Span::styled("  ID:", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{il}"), Style::default().fg(ic).add_modifier(Modifier::BOLD)),
            Span::styled(n_tts, Style::default().fg(Color::Blue)),
            Span::styled(s_tts, Style::default().fg(Color::Magenta)),
            Span::styled(gw_str, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(vig_str, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
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
            Constraint::Percentage(21),
            Constraint::Percentage(39),
            Constraint::Percentage(40),
        ])
        .split(area);
    draw_phill_panel(f, cols[0], s);
    draw_nova_panel(f, cols[1], s);
    draw_simona_panel(f, cols[2], s);
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
                    format!(" PHILL {}", if s.brain.phill_spiked { "*" } else { "" }),
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
        // Dynamic max: at least 5 (so tiny values are still visible),
        // otherwise 20% above the actual peak to use full bar height
        let peak = data.iter().copied().max().unwrap_or(0);
        let dyn_max = (peak + peak / 4).max(5).min(100);
        f.render_widget(
            Sparkline::default()
                .block(Block::default()
                    .title(Span::styled(*label, Style::default().fg(*color)))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)))
                .data(data)
                .style(Style::default().fg(*color))
                .max(dyn_max),
            sp[i],
        );
    }
}

// ── Region bar orders ─────────────────────────────────────────────────────────

const NOVA_ORDER: &[(&str, Color)] = &[
    ("thalamus",    Color::DarkGray),
    ("temporal",    Color::Cyan),
    ("hippocampus", Color::Blue),
    ("acc",         Color::Yellow),
    ("insula",      Color::Magenta),
    ("pfc",         Color::Green),
    ("broca",       Color::White),
];

const SIMONA_ORDER: &[(&str, Color)] = &[
    ("thalamus_s",    Color::DarkGray),
    ("temporal_s",    Color::Cyan),
    ("insula_s",      Color::Magenta),
    ("hippocampus_s", Color::Blue),
    ("pfc_s",         Color::Green),
    ("broca_s",       Color::White),
];

fn draw_nova_panel(f: &mut Frame, area: Rect, s: &SharedState) {
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(9),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(area);

    let vig_col = if s.brain.nova_vigilance { Color::Red } else { Color::Blue };
    let block   = Block::default()
        .title(Span::styled(
            format!(" NOVA (19)  7 regions  PFC-thr={:.2}  pressure={:.2}",
                    s.brain.nova_pfc_threshold, s.brain.nova_pressure),
            Style::default().fg(vig_col).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(vig_col));
    let region_area = block.inner(inner[0]);
    f.render_widget(block, inner[0]);
    draw_region_bars(f, region_area, &s.brain.nova_regions, NOVA_ORDER, false);

    // Broca sparkline
    f.render_widget(
        Sparkline::default()
            .block(Block::default()
                .title(Span::styled(
                    format!(" Broca  V={:.4}{}", s.brain.nova_pfc_voltage,
                            if s.brain.nova_tts_speaking { "  [TTS]" } else { "" }),
                    Style::default().fg(Color::White)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)))
            .data(&s.nova_broca_hist)
            .style(Style::default().fg(Color::White))
            .max(s.nova_broca_hist.iter().copied().max().unwrap_or(0).max(3).saturating_add(2)),
        inner[1],
    );

    // Vigilance indicator
    let vig_txt = if s.brain.nova_vigilance {
        "VIGILANCE  ACC inhibited PFC  skepticism active"
    } else {
        "Normal operation"
    };
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("  {vig_txt}"),
            Style::default().fg(if s.brain.nova_vigilance { Color::Red } else { Color::DarkGray })))
            .block(Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))),
        inner[2],
    );

    // Pressure mini-bar
    let np   = s.brain.nova_pressure.clamp(0.0, 1.0);
    let fill = (np * 16.0) as usize;
    let bar: String = "#".repeat(fill) + &".".repeat(16_usize.saturating_sub(fill));
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("  thought pressure [{bar}] {:.2}", np),
            Style::default().fg(if np > 0.6 { Color::Blue } else { Color::DarkGray }))),
        inner[3],
    );
}

fn draw_simona_panel(f: &mut Frame, area: Rect, s: &SharedState) {
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(area);

    let block = Block::default()
        .title(Span::styled(
            format!(" SIMONA (8)  6 regions  Broca-thr={:.2}  pressure={:.2}",
                    s.brain.simona_broca_thr, s.brain.simona_pressure),
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));
    let region_area = block.inner(inner[0]);
    f.render_widget(block, inner[0]);
    draw_region_bars(f, region_area, &s.brain.simona_regions, SIMONA_ORDER, true);

    // Broca_S sparkline
    f.render_widget(
        Sparkline::default()
            .block(Block::default()
                .title(Span::styled(
                    format!(" Broca_S  V={:.4}{}", s.brain.simona_broca_voltage,
                            if s.brain.simona_tts_speaking { "  [TTS]" } else { "" }),
                    Style::default().fg(Color::White)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta)))
            .data(&s.sim_broca_hist)
            .style(Style::default().fg(Color::Magenta))
            .max(s.sim_broca_hist.iter().copied().max().unwrap_or(0).max(3).saturating_add(2)),
        inner[1],
    );

    // Insula activity bar
    let ins = s.brain.simona_regions.iter()
        .find(|(n, _)| n == "insula_s").map(|(_, v)| *v).unwrap_or(0.0);
    let fill  = (ins.clamp(0.0, 1.0) * 16.0) as usize;
    let bar: String = "#".repeat(fill) + &".".repeat(16_usize.saturating_sub(fill));
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("  Insula [{bar}] {:.2}", ins),
            Style::default().fg(if ins > 0.4 { Color::Magenta } else { Color::DarkGray })
                .add_modifier(if ins > 0.6 { Modifier::BOLD } else { Modifier::empty() })))
            .block(Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))),
        inner[2],
    );

    // Simona pressure
    let sp   = s.brain.simona_pressure.clamp(0.0, 1.0);
    let fill2 = (sp * 16.0) as usize;
    let bar2: String = "#".repeat(fill2) + &".".repeat(16_usize.saturating_sub(fill2));
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("  thought pressure [{bar2}] {:.2}", sp),
            Style::default().fg(if sp > 0.4 { Color::Magenta } else { Color::DarkGray }))),
        inner[3],
    );
}

// ── Region bars (Unicode fill) ────────────────────────────────────────────────

fn draw_region_bars(
    f: &mut Frame,
    area: Rect,
    regions: &[(String, f64)],
    order: &[(&str, Color)],
    is_simona: bool,
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
            Paragraph::new(region_bar_line(name, act, area.width as usize, *color, is_simona)),
            rows[i],
        );
    }
}

fn region_bar_line(name: &str, act: f64, width: usize, color: Color, is_simona: bool) -> Line<'static> {
    let label  = if is_simona { name.replace("_s", "") } else { name.to_string() };
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
    let items: Vec<ListItem> = s.thought_history.iter().map(|t| {
        let (label, color) = if t.speaker == "thought_nova" {
            ("Nova  think | ", Color::Blue)
        } else {
            ("Simona think | ", Color::Magenta)
        };
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
}

// ── CONVERSATION ──────────────────────────────────────────────────────────────

fn draw_chat(f: &mut Frame, area: Rect, s: &SharedState) {
    let inner_h = area.height.saturating_sub(2) as usize;
    let start   = s.chat_history.len().saturating_sub(inner_h);

    let items: Vec<ListItem> = s.chat_history[start..].iter().map(|line| {
        let (label, color, bold) = match line.speaker.as_str() {
            "nova"       => ("Nova        |", Color::Blue,    true),
            "simona"     => ("Simona      |", Color::Magenta, true),
            "nodevortex" => ("NodeVortex  |", Color::Green,   true),
            _            => ("System      |", Color::DarkGray, false),
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

    let title_style = if s.brain.story_active {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let title_txt = if s.brain.story_active {
        " CONVERSATION  [STORY: NodeVortex / Nova / Simona] "
    } else {
        " CONVERSATION "
    };

    f.render_widget(
        List::new(items)
            .block(Block::default()
                .title(Span::styled(title_txt, title_style))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(
                    if s.brain.story_active { Color::Yellow } else { Color::DarkGray }
                ))),
        area,
    );
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

    // Wake word responsiveness bars
    let nr   = s.stt.nova_resp.clamp(0.0, 1.0);
    let sr   = s.stt.simona_resp.clamp(0.0, 1.0);
    let nfill = (nr * 10.0) as usize;
    let sfill = (sr * 10.0) as usize;
    let n_bar: String = "#".repeat(nfill) + &".".repeat(10usize.saturating_sub(nfill));
    let s_bar: String = "#".repeat(sfill) + &".".repeat(10usize.saturating_sub(sfill));

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
            Span::styled(format!("   Nova:[{n_bar}]"), Style::default().fg(Color::Blue)),
            Span::styled(format!("  Simona:[{s_bar}]"), Style::default().fg(Color::Magenta)),
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
        "  Say 'Nova' or 'Simona' to wake them.  Press 'i' to type manually.".into()
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
        (Color::Green, format!("ARCHITECT {:.0}%", trust * 100.0))
    } else if status.contains("learning") {
        // parse progress from "learning (N/40)"
        let frames: u32 = status.split('(').nth(1)
            .and_then(|s| s.split('/').next())
            .and_then(|n| n.trim().parse().ok())
            .unwrap_or(0);
        let color = if frames > 20 { Color::Yellow } else { Color::DarkGray };
        (color, status.into())
    } else if status.contains("uncertain") {
        (Color::Yellow, format!("uncertain {:.0}%", trust * 100.0))
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
