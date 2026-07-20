//! Frame input state, deliberately free of winit types so app-web and app-vr
//! can drive the same camera. The windowing layer translates into this.

#[derive(Debug, Default, Clone)]
pub struct InputState {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub sprint: bool,
    /// Accumulated mouse deltas since the last frame (pixels).
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    /// Accumulated scroll ticks since the last frame.
    pub scroll: f32,
}

impl InputState {
    /// Clear per-frame accumulators (mouse/scroll), keep held keys.
    pub fn end_frame(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.scroll = 0.0;
    }
}
