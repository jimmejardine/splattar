//! Frame-stepping sanity player: shows frames from any [`FrameSource`] in a
//! window with keyboard stepping. Built for eyeballing decoder output
//! (H.264/H.265 paths, tone mapping, crop) — a debug tool, not a video
//! player: no audio, no realtime playback, forward-only stepping (hardware
//! decode has no random access; "restart" reopens the source).
//!
//! Keys: → +1 frame · Shift+→ +10 frames · ↑ +1 s · Shift+↑ +10 s ·
//! R restart · Esc/Q quit.
//!
//! CPU blit via softbuffer (nearest-neighbor letterbox) — a sanity tool has
//! no business owning a GPU pipeline.

use std::num::NonZeroU32;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes};

/// One displayable frame, tightly packed RGBA8.
pub struct PlayerFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Forward-stepping source of frames (the CLI wraps the video reader).
pub trait FrameSource {
    /// Advance `n` frames; the last decoded one. None at end of stream.
    fn step_frames(&mut self, n: u32) -> Option<PlayerFrame>;
    /// Advance until presentation time moves >= `secs` forward.
    fn step_secs(&mut self, secs: f64) -> Option<PlayerFrame>;
    /// Reopen from the beginning; the first frame.
    fn restart(&mut self) -> Option<PlayerFrame>;
    /// Title-bar status ("frame 42 · 1.40s · 1920x1080 h265").
    fn status(&self) -> String;
}

pub fn run(mut source: Box<dyn FrameSource>) -> Result<(), String> {
    let first = source
        .step_frames(1)
        .ok_or_else(|| "no frames in source".to_string())?;
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    let mut app = App {
        source,
        frame: first,
        window: None,
        surface: None,
        shift: false,
        at_end: false,
    };
    event_loop.run_app(&mut app).map_err(|e| e.to_string())
}

struct App {
    source: Box<dyn FrameSource>,
    frame: PlayerFrame,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    shift: bool,
    at_end: bool,
}

impl App {
    fn title(&self) -> String {
        let end = if self.at_end { " · END" } else { "" };
        format!(
            "splattar play — {}{end}   [→ +1f  ⇧→ +10f  ↑ +1s  ⇧↑ +10s  R restart  Esc quit]",
            self.source.status()
        )
    }

    fn advance(&mut self, f: impl FnOnce(&mut dyn FrameSource) -> Option<PlayerFrame>) {
        match f(self.source.as_mut()) {
            Some(frame) => {
                self.frame = frame;
                self.at_end = false;
            }
            None => self.at_end = true,
        }
        if let Some(w) = &self.window {
            w.set_title(&self.title());
            w.request_redraw();
        }
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::ArrowRight => {
                let n = if self.shift { 10 } else { 1 };
                self.advance(|s| s.step_frames(n));
            }
            KeyCode::ArrowUp => {
                let secs = if self.shift { 10.0 } else { 1.0 };
                self.advance(|s| s.step_secs(secs));
            }
            KeyCode::KeyR => self.advance(|s| s.restart()),
            KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
            _ => {}
        }
    }

    fn redraw(&mut self) {
        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else {
            return;
        };
        let size = window.inner_size();
        let (ww, wh) = (size.width.max(1), size.height.max(1));
        let (Some(nw), Some(nh)) = (NonZeroU32::new(ww), NonZeroU32::new(wh)) else {
            return;
        };
        if surface.resize(nw, nh).is_err() {
            return;
        }
        let Ok(mut buffer) = surface.buffer_mut() else { return };

        // Nearest-neighbor letterbox into the window buffer (0RGB u32).
        let (fw, fh) = (self.frame.width as usize, self.frame.height as usize);
        let scale = ((ww as f64 / fw as f64).min(wh as f64 / fh as f64)).max(1e-6);
        let (dw, dh) = (
            ((fw as f64 * scale) as usize).clamp(1, ww as usize),
            ((fh as f64 * scale) as usize).clamp(1, wh as usize),
        );
        let (x_off, y_off) = ((ww as usize - dw) / 2, (wh as usize - dh) / 2);
        buffer.fill(0);
        for dy in 0..dh {
            let sy = (dy * fh / dh).min(fh - 1);
            let dst_row = (y_off + dy) * ww as usize + x_off;
            for dx in 0..dw {
                let sx = (dx * fw / dw).min(fw - 1);
                let s = (sy * fw + sx) * 4;
                let (r, g, b) = (
                    self.frame.rgba[s] as u32,
                    self.frame.rgba[s + 1] as u32,
                    self.frame.rgba[s + 2] as u32,
                );
                buffer[dst_row + dx] = (r << 16) | (g << 8) | b;
            }
        }
        let _ = buffer.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(self.title())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.frame.width.min(1600),
                self.frame.height.min(1000),
            ));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create player window"),
        );
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer context");
        let surface =
            softbuffer::Surface::new(&context, window.clone()).expect("softbuffer surface");
        window.request_redraw();
        self.window = Some(window);
        self.surface = Some(surface);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ModifiersChanged(m) => {
                self.shift = m.state().shift_key();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => self.handle_key(code, event_loop),
            WindowEvent::Resized(_) => {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}
