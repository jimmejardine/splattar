//! Reference forward renderer: explicit ray–splat intersection in camera
//! space. For pixel ray d and surfel plane {c + u·τu + v·τv}, solve
//! [τu τv −d]·(u,v,t)ᵀ = −c by Cramer's rule on scalar triple products —
//! mathematically equivalent to the 2DGS homography formulation but directly
//! differentiable with the cross-product gradient rules in `math`.

use glam::DVec3;

use crate::math::{self, quat_to_mat, triple};
use crate::scene::*;

/// Per-surfel camera-space quantities shared by forward and backward.
#[derive(Debug, Clone)]
pub struct SurfelCam {
    pub idx: usize,
    pub c: DVec3,
    pub tu: DVec3,
    pub tv: DVec3,
    pub color: DVec3,
    /// Which color channels were clamped at zero (grad mask).
    pub clamped: [bool; 3],
    pub opacity: f64,
    pub normal: DVec3,
    /// Center depth (sort key source).
    pub depth: f64,
    /// Projected center in pixel coordinates (low-pass branch).
    pub px: f64,
    pub py: f64,
}

/// Ray through a pixel center, camera space (unnormalized, z = −1).
pub fn pixel_ray(cam: &RefCamera, x: usize, y: usize) -> DVec3 {
    let cx = cam.width as f64 * 0.5;
    let cy = cam.height as f64 * 0.5;
    DVec3::new(
        (x as f64 + 0.5 - cx) / cam.focal,
        -(y as f64 + 0.5 - cy) / cam.focal,
        -1.0,
    )
}

/// Camera-space setup for every surfel, culled and sorted exactly like the
/// GPU pipeline: by the f32 bit pattern of center depth, ties by index.
pub fn prepare(scene: &MicroScene) -> Vec<SurfelCam> {
    let cam = &scene.camera;
    let r_cam = quat_to_mat(cam.quat); // camera-to-world
    let cx = cam.width as f64 * 0.5;
    let cy = cam.height as f64 * 0.5;

    let mut out = Vec::new();
    for (idx, s) in scene.surfels.iter().enumerate() {
        let x = s.pos - cam.center;
        let c = r_cam.transpose() * x;
        let depth = -c.z;
        if depth < NEAR_DEPTH {
            continue;
        }
        let rs = quat_to_mat(s.quat);
        let tu = r_cam.transpose() * (rs.col(0) * s.scales[0]);
        let tv = r_cam.transpose() * (rs.col(1) * s.scales[1]);

        let dir = (s.pos - cam.center).normalize();
        let basis = math::sh_basis(scene.sh_degree, dir);
        let mut raw = DVec3::splat(0.5);
        for (k, b) in basis.iter().enumerate() {
            raw += *b * s.sh[k];
        }
        let clamped = [raw.x < 0.0, raw.y < 0.0, raw.z < 0.0];
        let color = raw.max(DVec3::ZERO);

        let mut normal = tu.cross(tv);
        if normal.length_squared() > 0.0 {
            normal = normal.normalize();
            if normal.dot(c) > 0.0 {
                normal = -normal; // face the camera
            }
        }

        let w = -c.z;
        out.push(SurfelCam {
            idx,
            c,
            tu,
            tv,
            color,
            clamped,
            opacity: s.opacity,
            normal,
            depth,
            px: cx + cam.focal * c.x / w,
            py: cy - cam.focal * c.y / w,
        });
    }
    // Mirror the GPU: ascending f32 depth-key bits, stable by original index.
    out.sort_by_key(|s| ((s.depth as f32).to_bits(), s.idx));
    out
}

/// One pixel–surfel evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub ghat: f64,
    /// True if the ray-intersection gaussian won the low-pass max.
    pub ray_branch: bool,
    pub u: f64,
    pub v: f64,
    pub t: f64,
}

pub fn evaluate(sc: &SurfelCam, d: DVec3, pix_x: f64, pix_y: f64) -> Hit {
    let s = -d;
    let det = triple(sc.tu, sc.tv, s);
    let (mut g_ray, mut u, mut v, mut t) = (0.0, 0.0, 0.0, 0.0);
    if det.abs() > 1e-12 {
        u = triple(-sc.c, sc.tv, s) / det;
        v = triple(sc.tu, -sc.c, s) / det;
        t = triple(sc.tu, sc.tv, -sc.c) / det;
        if t > NEAR_DEPTH {
            g_ray = (-(u * u + v * v) * 0.5).exp();
        }
    }
    let dx = pix_x - sc.px;
    let dy = pix_y - sc.py;
    let g_scr = (-(dx * dx + dy * dy) / (2.0 * LOWPASS_SIGMA2)).exp();

    if g_ray >= g_scr {
        Hit {
            ghat: g_ray,
            ray_branch: true,
            u,
            v,
            t,
        }
    } else {
        Hit {
            ghat: g_scr,
            ray_branch: false,
            u,
            v,
            t: sc.depth, // degenerate view: fall back to center depth
        }
    }
}

/// Total per-ray pairwise depth-distortion loss over the image:
/// Σ_rays Σᵢ Σⱼ wᵢ wⱼ |tᵢ − tⱼ|, evaluated as 2·Σᵢ wᵢ(tᵢ·Aᵢ − Bᵢ) with the
/// composite order treated as depth order (prefix sums A, B).
pub fn distortion_loss(scene: &MicroScene) -> f64 {
    let cam = &scene.camera;
    let surfels = prepare(scene);
    let mut total = 0.0;
    for y in 0..cam.height {
        for x in 0..cam.width {
            let d = pixel_ray(cam, x, y);
            let (pix_x, pix_y) = (x as f64 + 0.5, y as f64 + 0.5);
            let mut transmittance = 1.0;
            let (mut a_pre, mut b_pre) = (0.0, 0.0);
            for sc in &surfels {
                let hit = evaluate(sc, d, pix_x, pix_y);
                let alpha = (sc.opacity * hit.ghat).min(ALPHA_CLAMP);
                if alpha < ALPHA_SKIP {
                    continue;
                }
                let w = transmittance * alpha;
                total += 2.0 * w * (hit.t * a_pre - b_pre);
                a_pre += w;
                b_pre += w * hit.t;
                transmittance *= 1.0 - alpha;
                if transmittance < T_TERMINATE {
                    break;
                }
            }
        }
    }
    total
}

pub fn render(scene: &MicroScene) -> RenderOutput {
    let cam = &scene.camera;
    let surfels = prepare(scene);
    let n_px = cam.width * cam.height;
    let mut out = RenderOutput {
        color: vec![DVec3::ZERO; n_px],
        alpha: vec![0.0; n_px],
        depth: vec![0.0; n_px],
        normal: vec![DVec3::ZERO; n_px],
    };

    for y in 0..cam.height {
        for x in 0..cam.width {
            let d = pixel_ray(cam, x, y);
            let (pix_x, pix_y) = (x as f64 + 0.5, y as f64 + 0.5);
            let i = y * cam.width + x;
            let mut transmittance = 1.0;
            for sc in &surfels {
                let hit = evaluate(sc, d, pix_x, pix_y);
                let alpha = (sc.opacity * hit.ghat).min(ALPHA_CLAMP);
                if alpha < ALPHA_SKIP {
                    continue;
                }
                let w = transmittance * alpha;
                out.color[i] += sc.color * w;
                out.alpha[i] += w;
                out.depth[i] += hit.t * w;
                out.normal[i] += sc.normal * w;
                transmittance *= 1.0 - alpha;
                if transmittance < T_TERMINATE {
                    break;
                }
            }
        }
    }
    out
}
