//! The per-frame diagnostic record.
//!
//! This type is the project's steering signal. Per CLAUDE.md, dB is not: it is
//! a scalar that hides where and why. What goes wrong in direct SLAM is
//! *spatial* (which part of the frame the model fails to explain) and
//! *temporal* (when tracking started to slip), and neither survives being
//! averaged into a single number.
//!
//! One record per processed frame, consumed by two things that must stay in
//! step: the live window, and a trace on disk that can be replayed and
//! compared against another run.

use glam::{Quat, Vec3};

/// An 8-bit image panel. Kept as bytes rather than GPU handles so a record can
/// be written to disk and replayed without a GPU present.
#[derive(Clone)]
pub struct Panel {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row 0 at top.
    pub rgba: Vec<u8>,
}

impl Panel {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        debug_assert_eq!(rgba.len(), (width * height * 4) as usize);
        Self {
            width,
            height,
            rgba,
        }
    }

    /// Grayscale error magnitude rendered as a heatmap: black → red → yellow →
    /// white with rising error. A signed difference would hide the magnitude at
    /// a glance, which is the one thing this panel exists to show.
    pub fn heatmap(width: u32, height: u32, err: &[f32], scale: f32) -> Self {
        let mut rgba = Vec::with_capacity(err.len() * 4);
        for &e in err {
            let t = (e * scale).clamp(0.0, 1.0);
            // Three ramps stacked: red rises first, then green, then blue.
            let r = (t * 3.0).clamp(0.0, 1.0);
            let g = ((t - 0.333) * 3.0).clamp(0.0, 1.0);
            let b = ((t - 0.667) * 3.0).clamp(0.0, 1.0);
            rgba.extend_from_slice(&[
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
                255,
            ]);
        }
        Self::new(width, height, rgba)
    }
}

/// Why a frame was not tracked, when it was not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrackState {
    /// Locked on; the pose is trusted.
    Tracking,
    /// Residual too high or diverging — the pose is a guess.
    Lost,
    /// Searching for a lock after loss (relocalization).
    Relocalizing,
    /// Deliberately skipped: too little motion since the last frame.
    Skipped,
}

/// One frame's worth of diagnostics.
///
/// Panels are `Option` because not every stage produces every panel — during
/// M0 there is no model, so there is nothing to render or difference against.
/// The record is meant to be filled in progressively as milestones land, not
/// redesigned at each one.
#[derive(Clone)]
pub struct FrameRecord {
    pub index: usize,
    /// Presentation time, seconds. VFR-safe — never index/fps (CLAUDE.md).
    pub pts: f64,
    /// What the camera captured.
    pub frame: Panel,
    /// What the model predicts from the estimated pose.
    pub render: Option<Panel>,
    /// Where the two disagree.
    pub error: Option<Panel>,
    pub state: TrackState,
    /// Estimated camera pose, if tracked.
    pub pose: Option<(Vec3, Quat)>,
    /// Photometric residual per pyramid level, coarse first. The shape of this
    /// over time is how tracking degradation becomes visible before it fails.
    pub residual: Vec<f32>,
    /// Fraction of the frame the model explains at all. Distinct from
    /// residual: a model can fit what it covers perfectly and still cover a
    /// third of the image.
    pub coverage: f32,
    pub surfels: u32,
    pub island: u32,
}

impl FrameRecord {
    /// A record with only the captured frame — what M0 can produce, and what
    /// every later stage starts from before it fills the rest in.
    pub fn captured(index: usize, pts: f64, frame: Panel) -> Self {
        Self {
            index,
            pts,
            frame,
            render: None,
            error: None,
            state: TrackState::Tracking,
            pose: None,
            residual: Vec::new(),
            coverage: 0.0,
            surfels: 0,
            island: 0,
        }
    }
}
