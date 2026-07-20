//! Sliding-window bundle adjustment: Levenberg–Marquardt over keyframe poses
//! and landmarks with the standard per-landmark Schur complement (landmarks
//! are conditionally independent given poses, so the reduced camera system is
//! only 6·n_poses wide). Huber-robust; gauge fixed by holding the first pose
//! (and, scale, by optionally holding the second pose's translation norm via
//! a prior residual).

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x6, Matrix3, Vector2, Vector3};

use crate::se3::{Se3, skew};

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

fn total_cost(p: &BaProblem, huber: f64) -> f64 {
    p.obs
        .iter()
        .filter_map(|o| eval_one(&p.poses[o.cam], &p.landmarks[o.lm], o.xy, huber))
        .map(|res| {
            let e = res.r.norm();
            if e <= huber {
                0.5 * e * e
            } else {
                huber * (e - 0.5 * huber)
            }
        })
        .sum()
}

/// Run LM; mutates poses and landmarks in place. Returns final cost.
pub fn optimize(p: &mut BaProblem, cfg: &BaConfig) -> f64 {
    let n_pose = p.poses.len();
    let n_free = n_pose.saturating_sub(cfg.fixed_poses);
    if n_free == 0 || p.landmarks.is_empty() {
        return total_cost(p, cfg.huber);
    }
    let dim = 6 * n_free;
    let mut lambda = 1e-4;
    let mut cost = total_cost(p, cfg.huber);

    for _ in 0..cfg.max_iters {
        // Accumulators: reduced camera system + per-landmark blocks.
        let mut h_cc = DMatrix::<f64>::zeros(dim, dim);
        let mut g_c = DVector::<f64>::zeros(dim);
        let n_lm = p.landmarks.len();
        let mut h_ll = vec![Matrix3::<f64>::zeros(); n_lm];
        let mut g_l = vec![Vector3::<f64>::zeros(); n_lm];
        // Off-diagonal blocks per (landmark, cam) pair, sparse by landmark.
        let mut h_cl: Vec<Vec<(usize, Matrix6x3)>> = vec![Vec::new(); n_lm];

        type Matrix6x3 = nalgebra::Matrix6x3<f64>;

        for o in &p.obs {
            let Some(res) = eval_one(&p.poses[o.cam], &p.landmarks[o.lm], o.xy, cfg.huber)
            else {
                continue;
            };
            let w = res.w;
            // Landmark block always accumulates.
            h_ll[o.lm] += res.jl.transpose() * res.jl * w;
            g_l[o.lm] += res.jl.transpose() * res.r * w;
            if o.cam >= cfg.fixed_poses {
                let ci = o.cam - cfg.fixed_poses;
                let jp_t_jp = res.jp.transpose() * res.jp * w;
                let r0 = 6 * ci;
                for a in 0..6 {
                    for b in 0..6 {
                        h_cc[(r0 + a, r0 + b)] += jp_t_jp[(a, b)];
                    }
                    g_c[r0 + a] += (res.jp.transpose() * res.r * w)[a];
                }
                let cross: Matrix6x3 = res.jp.transpose() * res.jl * w;
                // Merge into the landmark's cam list.
                match h_cl[o.lm].iter_mut().find(|(c, _)| *c == ci) {
                    Some((_, m)) => *m += cross,
                    None => h_cl[o.lm].push((ci, cross)),
                }
            }
        }

        // Schur complement: H_cc − Σ_l H_cl · (H_ll + λ·diag)⁻¹ · H_lc,
        // g_c − Σ_l H_cl · H_ll⁻¹ · g_l.
        let mut h_ll_inv = vec![Matrix3::<f64>::zeros(); n_lm];
        for l in 0..n_lm {
            let mut m = h_ll[l];
            for d in 0..3 {
                m[(d, d)] *= 1.0 + lambda;
                m[(d, d)] += 1e-12;
            }
            let Some(inv) = m.try_inverse() else {
                continue; // unconstrained landmark, leave zero (no update)
            };
            h_ll_inv[l] = inv;
            let tmp = inv * g_l[l];
            for &(ci, ref cl) in &h_cl[l] {
                let r0 = 6 * ci;
                let contrib = cl * tmp;
                for a in 0..6 {
                    g_c[r0 + a] -= contrib[a];
                }
                for &(cj, ref cl2) in &h_cl[l] {
                    let c0 = 6 * cj;
                    let block = cl * inv * cl2.transpose();
                    for a in 0..6 {
                        for b in 0..6 {
                            h_cc[(r0 + a, c0 + b)] -= block[(a, b)];
                        }
                    }
                }
            }
        }
        for d in 0..dim {
            h_cc[(d, d)] *= 1.0 + lambda;
            h_cc[(d, d)] += 1e-12;
        }

        let Some(delta_c) = h_cc.clone().lu().solve(&(-&g_c)) else {
            lambda *= 10.0;
            continue;
        };

        // Back-substitute landmarks: δl = −H_ll⁻¹ (g_l + H_lc δc).
        let mut new_poses = p.poses.clone();
        for ci in 0..n_free {
            let mut d6 = nalgebra::Vector6::<f64>::zeros();
            for a in 0..6 {
                d6[a] = delta_c[6 * ci + a];
            }
            new_poses[cfg.fixed_poses + ci] = p.poses[cfg.fixed_poses + ci].boxplus(&d6);
        }
        let mut new_lms = p.landmarks.clone();
        for l in 0..n_lm {
            let mut rhs = g_l[l];
            for &(ci, ref cl) in &h_cl[l] {
                let mut d6 = nalgebra::Vector6::<f64>::zeros();
                for a in 0..6 {
                    d6[a] = delta_c[6 * ci + a];
                }
                rhs += cl.transpose() * d6;
            }
            new_lms[l] -= h_ll_inv[l] * rhs;
        }

        let trial = BaProblem {
            poses: new_poses,
            landmarks: new_lms,
            obs: p.obs.clone(),
        };
        let new_cost = total_cost(&trial, cfg.huber);
        if new_cost < cost {
            p.poses = trial.poses;
            p.landmarks = trial.landmarks;
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
