//! Iterative PnP: Gauss-Newton on reprojection error with Huber weights and
//! a MAD-based rejection round. VO always has a motion-predicted prior pose,
//! so no minimal solver is needed — the prior puts GN in the convergence
//! basin, and the robust kernel + rejection handle stray bad tracks.

use nalgebra::{Matrix2x6, Matrix6, Vector2, Vector3, Vector6};

use crate::se3::{Se3, skew};

pub struct PnpConfig {
    pub max_iters: usize,
    /// Huber threshold in normalized coords (≈ pixels / focal).
    pub huber: f64,
    /// Hard-reject observations beyond k·MAD after the first converge.
    pub reject_mad_k: f64,
    pub min_points: usize,
}

impl Default for PnpConfig {
    fn default() -> Self {
        Self {
            max_iters: 12,
            huber: 3e-3,
            reject_mad_k: 6.0,
            min_points: 8,
        }
    }
}

pub struct PnpResult {
    pub pose: Se3,
    pub inliers: Vec<usize>,
    /// RMS reprojection error over inliers (normalized coords).
    pub rms: f64,
}

/// Residual and Jacobian (left-perturbation) of one observation.
#[inline]
fn residual_jac(
    pose: &Se3,
    p_world: &Vector3<f64>,
    obs: (f64, f64),
) -> Option<(Vector2<f64>, Matrix2x6<f64>)> {
    let pc = pose.act(p_world);
    let z = pc[2];
    if z <= 1e-6 {
        return None;
    }
    let (x, y) = (pc[0], pc[1]);
    let r = Vector2::new(x / z - obs.0, y / z - obs.1);
    // d(proj)/d(pc)
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    #[rustfmt::skip]
    let dproj = nalgebra::Matrix2x3::new(
        iz, 0.0, -x * iz2,
        0.0, iz, -y * iz2,
    );
    // Left perturbation: pc' = exp(δ)·(R p + t) ⇒ d pc/dδ = [I | −[pc]×].
    let mut dpc = nalgebra::Matrix3x6::zeros();
    dpc.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&nalgebra::Matrix3::identity());
    dpc.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(&pc)));
    Some((r, dproj * dpc))
}

/// Solve for the world→camera pose given 3D points and their observations.
pub fn solve_pnp(
    points: &[Vector3<f64>],
    obs: &[(f64, f64)],
    prior: &Se3,
    cfg: &PnpConfig,
) -> Option<PnpResult> {
    assert_eq!(points.len(), obs.len());
    if points.len() < cfg.min_points {
        return None;
    }
    let mut pose = *prior;
    let mut active: Vec<usize> = (0..points.len()).collect();

    for round in 0..2 {
        for _ in 0..cfg.max_iters {
            let mut h = Matrix6::<f64>::zeros();
            let mut g = Vector6::<f64>::zeros();
            let mut used = 0;
            for &i in &active {
                let Some((r, j)) = residual_jac(&pose, &points[i], obs[i]) else {
                    continue;
                };
                let e = r.norm();
                let w = if e <= cfg.huber { 1.0 } else { cfg.huber / e };
                h += j.transpose() * j * w;
                g += j.transpose() * r * w;
                used += 1;
            }
            if used < cfg.min_points {
                return None;
            }
            // Small LM damping keeps H invertible in low-parallax cases.
            for d in 0..6 {
                h[(d, d)] *= 1.0 + 1e-6;
                h[(d, d)] += 1e-12;
            }
            let delta = h.lu().solve(&(-g))?;
            pose = pose.boxplus(&delta);
            if delta.norm() < 1e-10 {
                break;
            }
        }
        if round == 0 {
            // MAD rejection: drop gross outliers, then re-converge.
            let mut errs: Vec<(usize, f64)> = active
                .iter()
                .filter_map(|&i| {
                    residual_jac(&pose, &points[i], obs[i]).map(|(r, _)| (i, r.norm()))
                })
                .collect();
            if errs.len() < cfg.min_points {
                return None;
            }
            let mut mags: Vec<f64> = errs.iter().map(|e| e.1).collect();
            mags.sort_by(|a, b| a.total_cmp(b));
            let med = mags[mags.len() / 2];
            let mut dev: Vec<f64> = mags.iter().map(|m| (m - med).abs()).collect();
            dev.sort_by(|a, b| a.total_cmp(b));
            let mad = dev[dev.len() / 2].max(1e-6);
            let cut = med + cfg.reject_mad_k * 1.4826 * mad;
            errs.retain(|(_, e)| *e <= cut);
            active = errs.into_iter().map(|(i, _)| i).collect();
            if active.len() < cfg.min_points {
                return None;
            }
        }
    }

    let mut ss = 0.0;
    let mut n = 0;
    for &i in &active {
        if let Some((r, _)) = residual_jac(&pose, &points[i], obs[i]) {
            ss += r.norm_squared();
            n += 1;
        }
    }
    if n < cfg.min_points {
        return None;
    }
    Some(PnpResult {
        pose,
        inliers: active,
        rms: (ss / n as f64).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twoview::Rng64;
    use nalgebra::Vector6;

    fn scene(n: usize, seed: u64) -> (Vec<Vector3<f64>>, Se3) {
        let mut rng = Rng64::new(seed);
        let mut r = |s: f64| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) * s;
        let pts = (0..n)
            .map(|_| Vector3::new(r(2.0), r(1.5), 4.0 + r(1.5)))
            .collect();
        let pose = Se3::exp(&Vector6::new(0.3, -0.1, 0.15, 0.05, -0.04, 0.08));
        (pts, pose)
    }

    #[test]
    fn converges_from_perturbed_prior() {
        let (pts, pose_gt) = scene(60, 5);
        let obs: Vec<(f64, f64)> = pts
            .iter()
            .map(|p| {
                let c = pose_gt.act(p);
                (c[0] / c[2], c[1] / c[2])
            })
            .collect();
        let prior = pose_gt.boxplus(&Vector6::new(0.05, -0.03, 0.02, 0.02, 0.03, -0.02));
        let res = solve_pnp(&pts, &obs, &prior, &PnpConfig::default()).expect("pnp");
        let dr = res.pose.r.inverse() * pose_gt.r;
        assert!(dr.angle() < 1e-8, "rot err {}", dr.angle());
        assert!((res.pose.t - pose_gt.t).norm() < 1e-8);
        assert!(res.rms < 1e-10);
    }

    #[test]
    fn rejects_outlier_observations() {
        let (pts, pose_gt) = scene(80, 11);
        let mut obs: Vec<(f64, f64)> = pts
            .iter()
            .map(|p| {
                let c = pose_gt.act(p);
                (c[0] / c[2], c[1] / c[2])
            })
            .collect();
        for k in 0..12 {
            obs[k * 6].0 += 0.08 + 0.01 * k as f64;
        }
        let prior = pose_gt.boxplus(&Vector6::new(0.03, 0.02, -0.02, -0.015, 0.02, 0.01));
        let res = solve_pnp(&pts, &obs, &prior, &PnpConfig::default()).expect("pnp");
        let dr = res.pose.r.inverse() * pose_gt.r;
        assert!(dr.angle() < 1e-6, "rot err {}", dr.angle());
        assert!((res.pose.t - pose_gt.t).norm() < 1e-6);
        assert!(res.inliers.len() >= 66 && res.inliers.len() <= 68);
    }
}
