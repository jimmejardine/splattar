//! The surfel map — the scene the tracker aligns against and the mapper
//! updates.
//!
//! One map, one representation, one loss. In the first attempt the poses lived
//! in a solver and the scene lived in a trainer, and the two drifted apart;
//! here the map is the only thing that holds geometry, and camera poses are
//! recovered by aligning frames against it.

use anyhow::Context as _;
use glam::{Quat, Vec3};
use gs_diag::Panel;
use gs_kernels::{RasterCamera, Rasterizer, SceneInput};
use gs_wgpu::GpuContext;

pub mod synthetic;

/// A 2D gaussian surfel: a flat disc with an orientation, so it has a real
/// normal and a well-defined depth from any view (unlike a 3D gaussian, whose
/// projected extent and "depth" both depend on where you look from).
#[derive(Clone, Copy, Debug)]
pub struct Surfel {
    pub position: Vec3,
    /// Local frame; +z is the surface normal.
    pub orientation: Quat,
    /// In-plane radii.
    pub scale: [f32; 2],
    pub opacity: f32,
    /// Degree-0 spherical harmonic, i.e. base colour.
    pub color: [f32; 3],
}

/// Structure-of-arrays surfel map, sized to a fixed budget.
///
/// Fixed because the GPU buffers are allocated once and never reallocated
/// (CLAUDE.md): growth consumes dead slots. `len` is how many are live, and
/// the tail beyond it is spare capacity for spawning.
pub struct SurfelMap {
    pub(crate) positions: Vec<Vec3>,
    pub(crate) scales: Vec<[f32; 2]>,
    pub(crate) quats: Vec<[f32; 4]>,
    pub(crate) opacities: Vec<f32>,
    /// Coefficient-major rgb-interleaved. Degree 0 only for now; the map grows
    /// higher bands when there is enough view diversity to fit them.
    pub(crate) sh: Vec<f32>,
}

const SH_C0: f32 = 0.282_094_79;

impl Default for SurfelMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SurfelMap {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            scales: Vec::new(),
            quats: Vec::new(),
            opacities: Vec::new(),
            sh: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn push(&mut self, s: Surfel) {
        self.positions.push(s.position);
        self.scales.push(s.scale);
        self.quats.push(s.orientation.normalize().to_array());
        self.opacities.push(s.opacity);
        // Colour is stored as the degree-0 SH coefficient, which is the
        // rendered colour scaled and biased — not the colour itself.
        for c in 0..3 {
            self.sh.push((s.color[c] - 0.5) / SH_C0);
        }
    }

    pub fn extend(&mut self, surfels: impl IntoIterator<Item = Surfel>) {
        for s in surfels {
            self.push(s);
        }
    }

    fn scene_input(&self) -> SceneInput<'_> {
        SceneInput {
            positions: &self.positions,
            scales: &self.scales,
            quats: &self.quats,
            opacities: &self.opacities,
            sh: &self.sh,
            sh_coeffs: 1,
        }
    }
}

/// A map resident on the GPU, ready to render.
///
/// Separate from [`SurfelMap`] because the CPU-side map is what gets built and
/// edited, while this owns the rasterizer and its fixed-size buffers. Keeping
/// them apart means the map can be constructed and tested with no GPU at all.
pub struct GpuMap {
    pub raster: Rasterizer,
    width: u32,
    height: u32,
    live: u32,
}

impl GpuMap {
    /// `capacity` is the surfel budget: buffers are sized once for it and never
    /// grow.
    pub fn new(ctx: &GpuContext, map: &SurfelMap, capacity: u32, width: u32, height: u32) -> Self {
        let cap = capacity.max(map.len() as u32).max(1);
        // Tile-binning entries: an upper bound on (surfel, tile) pairs. The
        // first attempt settled on ~48-64 per surfel and capped the per-surfel
        // pixel radius so one splat cannot flood the binner.
        let raster = Rasterizer::new(ctx, cap, width, height, cap * 64);
        raster.upload_scene(ctx, &map.scene_input());
        Self {
            raster,
            width,
            height,
            live: map.len() as u32,
        }
    }

    pub fn upload(&mut self, ctx: &GpuContext, map: &SurfelMap) {
        self.raster.upload_scene(ctx, &map.scene_input());
        self.live = map.len() as u32;
    }

    pub fn live(&self) -> u32 {
        self.live
    }

    /// Render to linear RGBA f32 — what the tracker differences against.
    pub fn render_f32(&self, ctx: &GpuContext, camera: &RasterCamera) -> Vec<[f32; 4]> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("map-render"),
            });
        self.raster
            .forward(ctx, &mut encoder, camera, self.live);
        ctx.queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            &self.raster.out_color,
        ))
        .to_vec()
    }

    /// Render to a diagnostic panel — what you look at.
    pub fn render_panel(&self, ctx: &GpuContext, camera: &RasterCamera) -> Panel {
        to_panel(self.width, self.height, &self.render_f32(ctx, camera))
    }
}

/// Linear RGBA f32 → an 8-bit panel.
pub fn to_panel(width: u32, height: u32, rgba: &[[f32; 4]]) -> Panel {
    let bytes = rgba
        .iter()
        .flat_map(|p| {
            [
                (p[0] * 255.0).clamp(0.0, 255.0) as u8,
                (p[1] * 255.0).clamp(0.0, 255.0) as u8,
                (p[2] * 255.0).clamp(0.0, 255.0) as u8,
                255,
            ]
        })
        .collect();
    Panel::new(width, height, bytes)
}

/// Per-pixel absolute colour difference, averaged over rgb.
///
/// This is the quantity the error panel shows and the tracker minimizes, so it
/// lives here rather than being reinvented per call site — a diagnostic that
/// shows something different from what is being optimized is worse than none.
pub fn abs_error(a: &[[f32; 4]], b: &[[f32; 4]]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            ((x[0] - y[0]).abs() + (x[1] - y[1]).abs() + (x[2] - y[2]).abs()) / 3.0
        })
        .collect()
}

/// Open a GPU context, or explain why not.
pub fn gpu() -> anyhow::Result<GpuContext> {
    pollster::block_on(gs_wgpu::GpuContext::new(wgpu::Backends::all()))
        .context("opening a GPU device")
}
