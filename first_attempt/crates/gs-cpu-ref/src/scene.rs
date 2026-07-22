//! Micro-scene types for the oracle. Everything is f64: the reference must be
//! numerically boring so finite differences against it are trustworthy.
//!
//! Conventions (must match the GPU kernels exactly):
//! - Camera: right-handed, y-up, looks down its local −z. Pose is a
//!   camera-to-world rotation quaternion (unnormalized storage; normalized on
//!   use) plus the camera center C in world space: x_cam = R(q̂)ᵀ (p − C).
//! - Pixel ray through (px, py): d = ((px+0.5−cx)/f, −(py+0.5−cy)/f, −1),
//!   cx = w/2, cy = h/2 (screen y down, camera y up).
//! - Surfel: center p, tangent axes t_u = R(q̂)·e0·s_u, t_v = R(q̂)·e1·s_v
//!   (activated scales — activation chains live in the trainer), opacity in
//!   (0,1), SH coefficients coefficient-major (rgb per coefficient).
//! - Low-pass: ĝ = max(G_ray, G_screen) with screen σ² = 0.5 (2DGS filter).

use glam::DVec3;

pub const LOWPASS_SIGMA2: f64 = 0.5;
pub const ALPHA_SKIP: f64 = 1.0 / 255.0;
pub const ALPHA_CLAMP: f64 = 0.995;
pub const T_TERMINATE: f64 = 1e-4;
pub const NEAR_DEPTH: f64 = 0.05;

#[derive(Debug, Clone)]
pub struct Surfel {
    pub pos: DVec3,
    /// Activated scales (s_u, s_v), world units.
    pub scales: [f64; 2],
    /// Unnormalized quaternion, xyzw.
    pub quat: [f64; 4],
    /// Activated opacity in (0, 1).
    pub opacity: f64,
    /// Coefficient-major SH, one DVec3 (rgb) per coefficient.
    pub sh: Vec<DVec3>,
}

#[derive(Debug, Clone)]
pub struct RefCamera {
    /// Camera center in world space.
    pub center: DVec3,
    /// Camera-to-world rotation, unnormalized quaternion xyzw.
    pub quat: [f64; 4],
    pub focal: f64,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone)]
pub struct MicroScene {
    pub surfels: Vec<Surfel>,
    pub camera: RefCamera,
    pub sh_degree: u8,
}

#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Per-pixel composited color (premultiplied over black).
    pub color: Vec<DVec3>,
    /// Per-pixel accumulated alpha (1 − final transmittance).
    pub alpha: Vec<f64>,
    /// Alpha-weighted intersection depth (aux target; no gradients in M2).
    pub depth: Vec<f64>,
    /// Alpha-weighted camera-space normal (aux target; no gradients in M2).
    pub normal: Vec<DVec3>,
}

/// Gradients w.r.t. every differentiated parameter class.
#[derive(Debug, Clone)]
pub struct Gradients {
    pub pos: Vec<DVec3>,
    pub scales: Vec<[f64; 2]>,
    pub quat: Vec<[f64; 4]>,
    pub opacity: Vec<f64>,
    /// Same layout as Surfel::sh.
    pub sh: Vec<Vec<DVec3>>,
    pub cam_center: DVec3,
    pub cam_quat: [f64; 4],
    pub focal: f64,
}

impl Gradients {
    pub fn zeros(scene: &MicroScene) -> Self {
        Self {
            pos: vec![DVec3::ZERO; scene.surfels.len()],
            scales: vec![[0.0; 2]; scene.surfels.len()],
            quat: vec![[0.0; 4]; scene.surfels.len()],
            opacity: vec![0.0; scene.surfels.len()],
            sh: scene
                .surfels
                .iter()
                .map(|s| vec![DVec3::ZERO; s.sh.len()])
                .collect(),
            cam_center: DVec3::ZERO,
            cam_quat: [0.0; 4],
            focal: 0.0,
        }
    }
}
