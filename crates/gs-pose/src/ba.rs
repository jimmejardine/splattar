//! Sliding-window bundle adjustment: Levenberg–Marquardt over keyframe poses
//! and landmarks with the standard per-landmark Schur complement (landmarks
//! are conditionally independent given poses, so the reduced camera system is
//! only 6·n_poses wide). Huber-robust; gauge fixed by holding the first pose
//! (and, scale, by optionally holding the second pose's translation norm via
//! a prior residual).

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x6, Matrix3, Vector2, Vector3};
use rayon::prelude::*;

use crate::se3::{Se3, skew};

type Matrix6 = nalgebra::Matrix6<f64>;
type Matrix6x3 = nalgebra::Matrix6x3<f64>;
type Vector6 = nalgebra::Vector6<f64>;

/// One observation: landmark `lm` seen from pose `cam` at normalized (x, y).
#[derive(Debug, Clone, Copy)]
pub struct Obs {
    pub cam: usize,
    pub lm: usize,
    pub xy: (f64, f64),
}

pub struct BaConfig {
    pub max_iters: usize,
    pub huber: f64,
    /// Number of leading poses held constant (≥1 fixes the gauge; 2 also
    /// fixes scale in a pure two-frame problem).
    pub fixed_poses: usize,
}

impl Default for BaConfig {
    fn default() -> Self {
        Self {
            max_iters: 20,
            huber: 3e-3,
            fixed_poses: 1,
        }
    }
}

pub struct BaProblem {
    pub poses: Vec<Se3>,
    pub landmarks: Vec<Vector3<f64>>,
    pub obs: Vec<Obs>,
}

struct Residual {
    r: Vector2<f64>,
    jp: Matrix2x6<f64>,
    jl: Matrix2x3<f64>,
    w: f64,
}

fn eval_one(pose: &Se3, lm: &Vector3<f64>, xy: (f64, f64), huber: f64) -> Option<Residual> {
    let pc = pose.act(lm);
    let z = pc[2];
    if z <= 1e-6 {
        return None;
    }
    let (x, y) = (pc[0], pc[1]);
    let r = Vector2::new(x / z - xy.0, y / z - xy.1);
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    #[rustfmt::skip]
    let dproj = Matrix2x3::new(
        iz, 0.0, -x * iz2,
        0.0, iz, -y * iz2,
    );
    let mut dpc = nalgebra::Matrix3x6::zeros();
    dpc.fixed_view_mut::<3, 3>(0, 0).copy_from(&Matrix3::identity());
    dpc.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(&pc)));
    let jp = dproj * dpc;
    let jl = dproj * pose.r.to_rotation_matrix().matrix();
    let e = r.norm();
    let w = if e <= huber { 1.0 } else { huber / e };
    Some(Residual { r, jp, jl, w })
}

fn total_cost(poses: &[Se3], landmarks: &[Vector3<f64>], obs: &[Obs], huber: f64) -> f64 {
    // Fixed-size chunks, serial within a chunk, chunk sums combined in
    // order: deterministic under any thread count.
    let chunk_sums: Vec<f64> = obs
        .par_chunks(4096)
        .map(|chunk| {
            chunk
                .iter()
                .filter_map(|o| eval_one(&poses[o.cam], &landmarks[o.lm], o.xy, huber))
                .map(|res| {
                    let e = res.r.norm();
                    if e <= huber {
                        0.5 * e * e
                    } else {
                        huber * (e - 0.5 * huber)
                    }
                })
                .sum::<f64>()
        })
        .collect();
    chunk_sums.into_iter().sum()
}

/// Dense SPD solve via faer's Cholesky (runtime-dispatched SIMD + rayon —
/// nalgebra's unblocked serial factorization was the global-BA bottleneck).
/// The reduced camera system is SPD after LM damping; returns None if the
/// factorization rejects it (caller falls back to LU).
fn spd_solve(h: &DMatrix<f64>, neg_g: &DVector<f64>) -> Option<DVector<f64>> {
    use faer::linalg::solvers::{Llt, Solve};
    let n = h.nrows();
    let a = faer::Mat::<f64>::from_fn(n, n, |i, j| h[(i, j)]);
    let llt = Llt::new(a.as_ref(), faer::Side::Lower).ok()?;
    let mut rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| neg_g[i]);
    llt.solve_in_place(rhs.as_mut());
    Some(DVector::from_fn(n, |i, _| rhs[(i, 0)]))
}

/// Per-landmark normal-equation contribution, built in parallel and merged
/// in landmark order (deterministic).
struct LmAcc {
    h_ll: Matrix3<f64>,
    g_l: Vector3<f64>,
    /// Per free camera seeing this landmark:
    /// (ci, Σ JpᵀJp·w, Σ Jpᵀr·w, Σ JpᵀJl·w).
    cams: Vec<(usize, Matrix6, Vector6, Matrix6x3)>,
}

/// Run LM; mutates poses and landmarks in place. Returns final cost.
///
/// Parallelism: assembly, Schur, cost, and landmark back-substitution fan
/// out per landmark / per chunk with all floating-point merges done in
/// landmark (or chunk) order — results are deterministic under any thread
/// count. The dense reduced-camera solve stays serial (Cholesky, LU
/// fallback) and is the remaining scaling wall for very large windows.
pub fn optimize(p: &mut BaProblem, cfg: &BaConfig) -> f64 {
    let n_pose = p.poses.len();
    let n_free = n_pose.saturating_sub(cfg.fixed_poses);
    if n_free == 0 || p.landmarks.is_empty() {
        return total_cost(&p.poses, &p.landmarks, &p.obs, cfg.huber);
    }
    let dim = 6 * n_free;
    let n_lm = p.landmarks.len();
    // Observation indices per landmark (in obs order) — obs never change
    // across LM iterations, so group once.
    let mut obs_of_lm: Vec<Vec<u32>> = vec![Vec::new(); n_lm];
    for (i, o) in p.obs.iter().enumerate() {
        obs_of_lm[o.lm].push(i as u32);
    }
    // Only very large (global-BA-sized) problems log progress.
    let chatty = p.obs.len() > 20_000;
    let mut lambda = 1e-4;
    let mut cost = total_cost(&p.poses, &p.landmarks, &p.obs, cfg.huber);

    for iter in 0..cfg.max_iters {
        // Assembly: per-landmark contributions in parallel, merged into the
        // reduced camera system in landmark order.
        let accs: Vec<LmAcc> = obs_of_lm
            .par_iter()
            .enumerate()
            .map(|(lm, idxs)| {
                let mut acc = LmAcc {
                    h_ll: Matrix3::zeros(),
                    g_l: Vector3::zeros(),
                    cams: Vec::new(),
                };
                for &i in idxs {
                    let o = &p.obs[i as usize];
                    let Some(res) =
                        eval_one(&p.poses[o.cam], &p.landmarks[lm], o.xy, cfg.huber)
                    else {
                        continue;
                    };
                    let w = res.w;
                    acc.h_ll += res.jl.transpose() * res.jl * w;
                    acc.g_l += res.jl.transpose() * res.r * w;
                    if o.cam >= cfg.fixed_poses {
                        let ci = o.cam - cfg.fixed_poses;
                        let jtj: Matrix6 = res.jp.transpose() * res.jp * w;
                        let gc: Vector6 = res.jp.transpose() * res.r * w;
                        let cross: Matrix6x3 = res.jp.transpose() * res.jl * w;
                        match acc.cams.iter_mut().find(|(c, ..)| *c == ci) {
                            Some((_, mj, mg, mc)) => {
                                *mj += jtj;
                                *mg += gc;
                                *mc += cross;
                            }
                            None => acc.cams.push((ci, jtj, gc, cross)),
                        }
                    }
                }
                acc
            })
            .collect();

        let mut h_cc = DMatrix::<f64>::zeros(dim, dim);
        let mut g_c = DVector::<f64>::zeros(dim);
        for acc in &accs {
            for &(ci, ref jtj, ref gc, _) in &acc.cams {
                let r0 = 6 * ci;
                for a in 0..6 {
                    for b in 0..6 {
                        h_cc[(r0 + a, r0 + b)] += jtj[(a, b)];
                    }
                    g_c[r0 + a] += gc[a];
                }
            }
        }

        // Schur complement: H_cc − Σ_l H_cl · (H_ll + λ·diag)⁻¹ · H_lc,
        // g_c − Σ_l H_cl · H_ll⁻¹ · g_l. Landmarks are independent; chunked
        // so the transient per-landmark block lists stay bounded, merged in
        // order per chunk.
        let mut h_ll_inv = vec![Matrix3::<f64>::zeros(); n_lm];
        const SCHUR_CHUNK: usize = 256;
        for (cb, chunk) in accs.chunks(SCHUR_CHUNK).enumerate() {
            type SchurOut = (
                Matrix3<f64>,
                Vec<(usize, Vector6)>,
                Vec<(usize, usize, Matrix6)>,
            );
            let outs: Vec<SchurOut> = chunk
                .par_iter()
                .map(|acc| {
                    let mut m = acc.h_ll;
                    for d in 0..3 {
                        m[(d, d)] *= 1.0 + lambda;
                        m[(d, d)] += 1e-12;
                    }
                    let Some(inv) = m.try_inverse() else {
                        // Unconstrained landmark: zero inverse, no update.
                        return (Matrix3::zeros(), Vec::new(), Vec::new());
                    };
                    let tmp = inv * acc.g_l;
                    let mut gcs = Vec::with_capacity(acc.cams.len());
                    let mut blocks = Vec::with_capacity(acc.cams.len() * acc.cams.len());
                    for &(ci, .., ref cl) in &acc.cams {
                        gcs.push((6 * ci, cl * tmp));
                        for &(cj, .., ref cl2) in &acc.cams {
                            blocks.push((6 * ci, 6 * cj, cl * inv * cl2.transpose()));
                        }
                    }
                    (inv, gcs, blocks)
                })
                .collect();
            for (li, (inv, gcs, blocks)) in outs.into_iter().enumerate() {
                h_ll_inv[cb * SCHUR_CHUNK + li] = inv;
                for (r0, v) in gcs {
                    for a in 0..6 {
                        g_c[r0 + a] -= v[a];
                    }
                }
                for (r0, c0, blk) in blocks {
                    for a in 0..6 {
                        for b in 0..6 {
                            h_cc[(r0 + a, c0 + b)] -= blk[(a, b)];
                        }
                    }
                }
            }
        }
        for d in 0..dim {
            h_cc[(d, d)] *= 1.0 + lambda;
            h_cc[(d, d)] += 1e-12;
        }

        // The damped Schur complement is symmetric positive definite —
        // faer Cholesky (SIMD + threaded); serial LU as numerical fallback.
        let neg_g = -&g_c;
        let delta_c = match spd_solve(&h_cc, &neg_g) {
            Some(d) => d,
            None => match h_cc.clone().lu().solve(&neg_g) {
                Some(d) => d,
                None => {
                    lambda *= 10.0;
                    continue;
                }
            },
        };

        // Back-substitute landmarks: δl = −H_ll⁻¹ (g_l + H_lc δc).
        let mut new_poses = p.poses.clone();
        for ci in 0..n_free {
            let mut d6 = Vector6::zeros();
            for a in 0..6 {
                d6[a] = delta_c[6 * ci + a];
            }
            new_poses[cfg.fixed_poses + ci] = p.poses[cfg.fixed_poses + ci].boxplus(&d6);
        }
        let mut new_lms = p.landmarks.clone();
        new_lms
            .par_iter_mut()
            .enumerate()
            .for_each(|(l, lm)| {
                let mut rhs = accs[l].g_l;
                for &(ci, .., ref cl) in &accs[l].cams {
                    let mut d6 = Vector6::zeros();
                    for a in 0..6 {
                        d6[a] = delta_c[6 * ci + a];
                    }
                    rhs += cl.transpose() * d6;
                }
                *lm -= h_ll_inv[l] * rhs;
            });

        let new_cost = total_cost(&new_poses, &new_lms, &p.obs, cfg.huber);
        if chatty {
            log::info!(
                "  BA iter {iter}: cost {cost:.3e} -> {new_cost:.3e} (lambda {lambda:.1e})"
            );
        }
        if new_cost < cost {
            p.poses = new_poses;
            p.landmarks = new_lms;
            let improved = (cost - new_cost) / cost.max(1e-18);
            cost = new_cost;
            lambda = (lambda * 0.5).max(1e-9);
            if improved < 1e-9 {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e6 {
                break;
            }
        }
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twoview::Rng64;
    use nalgebra::Vector6;

    fn build_problem(
        n_pose: usize,
        n_lm: usize,
        noise: f64,
        seed: u64,
    ) -> (BaProblem, Vec<Se3>, Vec<Vector3<f64>>) {
        let mut rng = Rng64::new(seed);
        let mut r = |s: f64| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) * s;
        let gt_lms: Vec<Vector3<f64>> = (0..n_lm)
            .map(|_| Vector3::new(r(2.5), r(1.8), 5.0 + r(2.0)))
            .collect();
        let gt_poses: Vec<Se3> = (0..n_pose)
            .map(|k| {
                Se3::exp(&Vector6::new(
                    0.35 * k as f64,
                    0.02 * k as f64,
                    0.05 * k as f64,
                    0.01 * k as f64,
                    -0.02 * k as f64,
                    0.005 * k as f64,
                ))
            })
            .collect();
        let mut obs = Vec::new();
        for (ci, pose) in gt_poses.iter().enumerate() {
            for (li, lm) in gt_lms.iter().enumerate() {
                let c = pose.act(lm);
                if c[2] > 0.5 {
                    obs.push(Obs {
                        cam: ci,
                        lm: li,
                        xy: (c[0] / c[2] + r(noise), c[1] / c[2] + r(noise)),
                    });
                }
            }
        }
        // Perturb everything except the fixed gauge poses.
        let mut poses = gt_poses.clone();
        for pose in poses.iter_mut().skip(2) {
            *pose = pose.boxplus(&Vector6::new(r(0.03), r(0.03), r(0.03), r(0.01), r(0.01), r(0.01)));
        }
        let landmarks: Vec<Vector3<f64>> = gt_lms
            .iter()
            .map(|l| l + Vector3::new(r(0.05), r(0.05), r(0.08)))
            .collect();
        (
            BaProblem {
                poses,
                landmarks,
                obs,
            },
            gt_poses,
            gt_lms,
        )
    }

    #[test]
    fn recovers_noiseless_geometry() {
        let (mut p, gt_poses, gt_lms) = build_problem(5, 80, 0.0, 17);
        let cfg = BaConfig {
            fixed_poses: 2,
            ..Default::default()
        };
        let cost = optimize(&mut p, &cfg);
        assert!(cost < 1e-14, "final cost {cost}");
        for (est, gt) in p.poses.iter().zip(&gt_poses) {
            assert!((est.r.inverse() * gt.r).angle() < 1e-6);
            assert!((est.t - gt.t).norm() < 1e-6);
        }
        for (est, gt) in p.landmarks.iter().zip(&gt_lms) {
            assert!((est - gt).norm() < 1e-5);
        }
    }

    #[test]
    fn improves_under_noise() {
        let (mut p, gt_poses, _) = build_problem(6, 120, 5e-4, 23);
        let before: f64 = p
            .poses
            .iter()
            .zip(&gt_poses)
            .map(|(e, g)| (e.t - g.t).norm())
            .sum();
        let cfg = BaConfig {
            fixed_poses: 2,
            ..Default::default()
        };
        optimize(&mut p, &cfg);
        let after: f64 = p
            .poses
            .iter()
            .zip(&gt_poses)
            .map(|(e, g)| (e.t - g.t).norm())
            .sum();
        assert!(
            after < before * 0.35,
            "BA did not improve enough: {before} -> {after}"
        );
    }
}
