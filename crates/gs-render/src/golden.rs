//! Golden-image infrastructure: deterministic bbox-derived camera poses and
//! PSNR comparison. Shared by the golden test and gs-cli's --render-golden
//! regeneration path so both always render through the identical pipeline.

use glam::Vec3;
use gs_core::Camera;
use gs_wgpu::GpuContext;

use crate::pipeline::{RenderSettings, SplatRenderer};
use crate::{offscreen, scene::GpuScene};

pub const GOLDEN_SIZE: (u32, u32) = (800, 600);
/// PSNR tolerance in dB — catches real regressions while absorbing
/// driver/backend rounding differences (never compare bytes).
pub const GOLDEN_PSNR_DB: f64 = 45.0;

/// Three deterministic poses derived purely from the scene bbox:
/// front, three-quarter high orbit, and a closer low angle.
pub fn golden_cameras(bbox: (Vec3, Vec3)) -> [Camera; 3] {
    let center = (bbox.0 + bbox.1) * 0.5;
    let radius = 0.5 * (bbox.1 - bbox.0).length();
    let at = |dir: Vec3, dist: f32| {
        Camera::look_at(center + dir.normalize() * dist * radius, center, Vec3::Y)
    };
    [
        at(Vec3::new(0.0, 0.0, 1.0), 2.2),
        at(Vec3::new(1.0, 0.7, 1.0), 2.0),
        at(Vec3::new(-1.0, 0.25, 0.8), 1.3),
    ]
}

/// Render the three golden poses; returns tightly-packed RGBA images.
pub fn render_goldens(ctx: &GpuContext, scene: &GpuScene, renderer: &SplatRenderer) -> [Vec<u8>; 3] {
    let settings = RenderSettings::default();
    let (w, h) = GOLDEN_SIZE;
    golden_cameras(scene.bbox)
        .map(|camera| offscreen::render_to_rgba(ctx, renderer, &camera, w, h, &settings))
}

/// PSNR between two same-sized RGBA images, alpha ignored.
pub fn psnr_rgba(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "image size mismatch");
    let mut se = 0f64;
    let mut count = 0usize;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            let d = pa[c] as f64 - pb[c] as f64;
            se += d * d;
        }
        count += 3;
    }
    let mse = se / count as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}
