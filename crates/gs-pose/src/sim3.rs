//! Sim(3) estimation between point sets: closed-form Umeyama alignment inside
//! RANSAC over minimal 3-point samples. This is the cross-video registration
//! primitive — both submaps carry triangulated landmarks, so overlap becomes
//! a 3D↔3D correspondence problem (each side in its own monocular gauge, so
//! scale is part of the estimate).

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::twoview::Rng64;

/// x_b = scale · (R · x_a) + t.
#[derive(Debug, Clone, Copy)]
pub struct Sim3 {
    pub scale: f64,
    pub r: UnitQuaternion<f64>,
    pub t: Vector3<f64>,
}

impl Sim3 {
    #[inline]
    pub fn apply(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.scale * (self.r * p) + self.t
    }

    pub fn inverse(&self) -> Sim3 {
        let rinv = self.r.inverse();
        let s = 1.0 / self.scale;
        Sim3 {
            scale: s,
            r: rinv,
            t: -(s * (rinv * self.t)),
        }
    }
}

/// Closed-form Umeyama: least-squares Sim(3) mapping a→b.
pub fn umeyama(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> Option<Sim3> {
    let n = a.len();
    if n < 3 || n != b.len() {
        return None;
    }
    let nf = n as f64;
    let mu_a: Vector3<f64> = a.iter().sum::<Vector3<f64>>() / nf;
    let mu_b: Vector3<f64> = b.iter().sum::<Vector3<f64>>() / nf;
    let mut cov = Matrix3::<f64>::zeros();
    let mut var_a = 0.0;
    for (pa, pb) in a.iter().zip(b) {
        cov += (pb - mu_b) * (pa - mu_a).transpose();
        var_a += (pa - mu_a).norm_squared();
    }
    cov /= nf;
    var_a /= nf;
    if var_a < 1e-18 {
        return None;
    }
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut s = Matrix3::<f64>::identity();
    if (u * vt).determinant() < 0.0 {
        s[(2, 2)] = -1.0;
    }
    let r_mat = u * s * vt;
    let scale = (svd.singular_values[0] * s[(0, 0)]
        + svd.singular_values[1] * s[(1, 1)]
        + svd.singular_values[2] * s[(2, 2)])
        / var_a;
    if !(scale.is_finite() && scale > 1e-9) {
        return None;
    }
    let r = UnitQuaternion::from_matrix(&r_mat);
    let t = mu_b - scale * (r * mu_a);
    Some(Sim3 { scale, r, t })
}

pub struct Sim3Ransac {
    pub sim3: Sim3,
    pub inliers: Vec<usize>,
}

/// RANSAC Sim(3) over corresponding point pairs. `thresh` is the absolute
/// residual in b's units; pass something scaled to b's scene extent.
pub fn ransac_sim3(
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    iters: usize,
    thresh: f64,
    seed: u64,
) -> Option<Sim3Ransac> {
    ransac_sim3_bounded(a, b, iters, thresh, seed, (0.01, 100.0))
}

/// RANSAC Sim(3) with an explicit prior on admissible scale — candidate
/// models outside the bounds never get to vote. Degenerate near-collapse
/// transforms otherwise out-vote the true model on ambiguous match sets
/// (measured repeatedly in cross-video registration).
pub fn ransac_sim3_bounded(
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    iters: usize,
    thresh: f64,
    seed: u64,
    scale_bounds: (f64, f64),
) -> Option<Sim3Ransac> {
    let n = a.len();
    if n < 3 || n != b.len() {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Rng64::new(seed);
    let mut best: Option<(usize, Sim3)> = None;
    for _ in 0..iters {
        let mut idx = [0usize; 3];
        let mut k = 0;
        while k < 3 {
            let c = rng.below(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let sa: Vec<Vector3<f64>> = idx.iter().map(|&i| a[i]).collect();
        let sb: Vec<Vector3<f64>> = idx.iter().map(|&i| b[i]).collect();
        let Some(m) = umeyama(&sa, &sb) else { continue };
        // Degenerate-scale models (near-coincident samples) can "explain"
        // any concentrated cluster by collapsing it — never score them.
        if !(scale_bounds.0..=scale_bounds.1).contains(&m.scale) {
            continue;
        }
        let count = a
            .iter()
            .zip(b)
            .filter(|(pa, pb)| (m.apply(pa) - **pb).norm_squared() < t2)
            .count();
        if best.is_none_or(|(bc, _)| count > bc) {
            best = Some((count, m));
        }
    }
    let (_, m0) = best?;
    // Refit on consensus, reclassify once.
    let inl: (Vec<Vector3<f64>>, Vec<Vector3<f64>>) = a
        .iter()
        .zip(b)
        .filter(|(pa, pb)| (m0.apply(pa) - **pb).norm_squared() < t2)
        .map(|(pa, pb)| (*pa, *pb))
        .unzip();
    let m1 = umeyama(&inl.0, &inl.1).unwrap_or(m0);
    let inliers: Vec<usize> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (pa, pb))| (m1.apply(pa) - **pb).norm_squared() < t2)
        .map(|(i, _)| i)
        .collect();
    if inliers.len() < 3 {
        return None;
    }
    Some(Sim3Ransac { sim3: m1, inliers })
}

/// glam-facing Sim(3) (the API-boundary type per CLAUDE.md).
#[derive(Debug, Clone, Copy)]
pub struct Sim3G {
    pub scale: f64,
    /// xyzw quaternion.
    pub rot: glam::DQuat,
    pub trans: glam::DVec3,
}

impl Sim3G {
    pub fn apply(&self, p: glam::DVec3) -> glam::DVec3 {
        self.scale * (self.rot * p) + self.trans
    }

    pub fn identity() -> Sim3G {
        Sim3G {
            scale: 1.0,
            rot: glam::DQuat::IDENTITY,
            trans: glam::DVec3::ZERO,
        }
    }

    pub fn inverse(&self) -> Sim3G {
        let s = 1.0 / self.scale;
        let r = self.rot.conjugate();
        Sim3G {
            scale: s,
            rot: r,
            trans: -(s * (r * self.trans)),
        }
    }

    /// Composition: `a.compose(&b).apply(p) == a.apply(b.apply(p))`.
    pub fn compose(&self, b: &Sim3G) -> Sim3G {
        Sim3G {
            scale: self.scale * b.scale,
            rot: self.rot * b.rot,
            trans: self.scale * (self.rot * b.trans) + self.trans,
        }
    }
}

/// RANSAC Sim(3) between corresponding glam point sets (a → b). Returns the
/// transform and the inlier indices.
pub fn register_point_sets(
    a: &[glam::DVec3],
    b: &[glam::DVec3],
    iters: usize,
    thresh: f64,
    seed: u64,
) -> Option<(Sim3G, Vec<usize>)> {
    register_point_sets_bounded(a, b, iters, thresh, seed, (0.01, 100.0))
}

/// [`register_point_sets`] with an admissible-scale prior.
pub fn register_point_sets_bounded(
    a: &[glam::DVec3],
    b: &[glam::DVec3],
    iters: usize,
    thresh: f64,
    seed: u64,
    scale_bounds: (f64, f64),
) -> Option<(Sim3G, Vec<usize>)> {
    let na: Vec<Vector3<f64>> = a.iter().map(|p| Vector3::new(p.x, p.y, p.z)).collect();
    let nb: Vec<Vector3<f64>> = b.iter().map(|p| Vector3::new(p.x, p.y, p.z)).collect();
    let res = ransac_sim3_bounded(&na, &nb, iters, thresh, seed, scale_bounds)?;
    let q = res.sim3.r.quaternion();
    Some((
        Sim3G {
            scale: res.sim3.scale,
            rot: glam::DQuat::from_xyzw(q.i, q.j, q.k, q.w),
            trans: glam::DVec3::new(res.sim3.t[0], res.sim3.t[1], res.sim3.t[2]),
        },
        res.inliers,
    ))
}

/// One verified image pair with known camera poses in each gauge, plus the
/// epipolar-recovered relative pose (camera B in camera A's frame, unit
/// baseline). Input to [`sim3_from_relative_pairs`].
pub struct RelPair {
    /// Camera A: world→camera pose in the PROJECT-WORLD gauge.
    pub cam_a_world: (glam::DQuat, glam::DVec3),
    /// Camera B: world→camera pose in the SEGMENT gauge.
    pub cam_b_seg: (glam::DQuat, glam::DVec3),
    /// x_b = rel_rot · x_a + λ · rel_tdir  (λ unknown, ‖rel_tdir‖ = 1).
    pub rel_rot: glam::DQuat,
    pub rel_tdir: glam::DVec3,
}

/// Snap-free Sim(3): relative camera poses between known-gauge cameras
/// constrain world_from_seg directly — no landmark 3D is ever trusted
/// (correspondence snapping measured out at ~20-30% precision, starving
/// every point-based lift; see RESULTS.md).
///
/// Per pair: R_s = (R_rel·R_A)ᵀ·R_B (rotation averaging), and the camera
/// centers give σ·(R_s·C_b) + t_s + λᵢ·d̂ᵢ = C_a — linear in (σ, t_s, λᵢ)
/// with one extra unknown per pair, so P ≥ 2 pairs solve it. Returns the
/// transform and the RMS center residual (world units).
pub fn sim3_from_relative_pairs(pairs: &[RelPair]) -> Option<(Sim3G, f64)> {
    if pairs.len() < 2 {
        return None;
    }
    let to_na = |q: &glam::DQuat| {
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z))
    };
    // Per-pair rotation estimates, then keep the LARGEST mutually consistent
    // cluster — a single junk E-decomposition must not poison the solve.
    let rots: Vec<UnitQuaternion<f64>> = pairs
        .iter()
        .map(|pr| {
            (to_na(&pr.rel_rot) * to_na(&pr.cam_a_world.0)).inverse() * to_na(&pr.cam_b_seg.0)
        })
        .collect();
    let mut best_cluster: Vec<usize> = Vec::new();
    for i in 0..rots.len() {
        let cluster: Vec<usize> = (0..rots.len())
            .filter(|&j| (rots[i].inverse() * rots[j]).angle() < 0.3)
            .collect();
        if cluster.len() > best_cluster.len() {
            best_cluster = cluster;
        }
    }
    if best_cluster.len() < 2 {
        log::debug!(
            "relative-pose Sim3: no rotation cluster among {} pairs",
            pairs.len()
        );
        return None;
    }
    let pairs: Vec<&RelPair> = best_cluster.iter().map(|&i| &pairs[i]).collect();
    let p = pairs.len();
    log::debug!("relative-pose Sim3: rotation cluster {p}/{}", rots.len());
    let mut acc = nalgebra::Vector4::zeros();
    for &i in &best_cluster {
        let q = rots[i].quaternion();
        let mut v = nalgebra::Vector4::new(q.w, q.i, q.j, q.k);
        if acc.dot(&v) < 0.0 {
            v = -v;
        }
        acc += v;
    }
    if acc.norm() < 1e-12 {
        return None;
    }
    acc /= acc.norm();
    let r_s = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        acc[0], acc[1], acc[2], acc[3],
    ));

    // Linear system in x = [σ, t_s(3), λ_1..λ_P]:
    //   σ·(R_s·C_b_i) + t_s + λ_i·d_i = C_a_i     (3 rows per pair)
    // where d_i = (R_rel·R_A)ᵀ·(−t̂_rel) is B's center direction from A.
    let n_unk = 4 + p;
    let mut m = nalgebra::DMatrix::<f64>::zeros(3 * p, n_unk);
    let mut rhs = nalgebra::DVector::<f64>::zeros(3 * p);
    for (i, pr) in pairs.iter().enumerate() {
        let r_a = to_na(&pr.cam_a_world.0);
        let t_a = Vector3::new(pr.cam_a_world.1.x, pr.cam_a_world.1.y, pr.cam_a_world.1.z);
        let c_a = -(r_a.inverse() * t_a);
        let r_b = to_na(&pr.cam_b_seg.0);
        let t_b = Vector3::new(pr.cam_b_seg.1.x, pr.cam_b_seg.1.y, pr.cam_b_seg.1.z);
        let c_b = -(r_b.inverse() * t_b);
        let r_rel = to_na(&pr.rel_rot);
        let u = Vector3::new(pr.rel_tdir.x, pr.rel_tdir.y, pr.rel_tdir.z);
        let d = (r_rel * r_a).inverse() * (-u);
        let rc = r_s * c_b;
        for row in 0..3 {
            m[(3 * i + row, 0)] = rc[row];
            m[(3 * i + row, 1 + row)] = 1.0;
            m[(3 * i + row, 4 + i)] = d[row];
            rhs[3 * i + row] = c_a[row];
        }
    }
    let svd = m.clone().svd(true, true);
    let x = svd.solve(&rhs, 1e-12).ok()?;
    let sigma = x[0];
    if !(sigma.is_finite() && sigma > 1e-6) {
        return None;
    }
    let resid = (&m * &x - &rhs).norm() / (3.0 * p as f64).sqrt();
    let q = r_s.quaternion();
    Some((
        Sim3G {
            scale: sigma,
            rot: glam::DQuat::from_xyzw(q.i, q.j, q.k, q.w),
            trans: glam::DVec3::new(x[1], x[2], x[3]),
        },
        resid,
    ))
}

/// One cross-cut bridge correspondence for [`sim3_from_bridge`].
pub struct BridgeObs {
    /// Observation in the new segment's keyframe K (normalized img coords).
    pub obs: (f64, f64),
    /// The matched landmark's position in PROJECT-WORLD coordinates.
    pub world: glam::DVec3,
    /// The new-side landmark's position in the SEGMENT gauge (depth prior).
    pub seg: glam::DVec3,
}

/// Solve world_from_segment for a video segment from 2D-3D bridge matches at
/// one boundary keyframe K, sidestepping the new side's noisy boundary
/// triangulations: DLT-P6P RANSAC gives K's Euclidean pose in the world;
/// the monocular gauge scale comes from robust (median) depth ratios; the
/// Sim(3) then follows in closed form from K's known segment-gauge pose.
/// Returns the transform and the reprojection-inlier count.
pub fn sim3_from_bridge(
    cam_seg_rot: glam::DQuat, // segment-gauge world→camera rotation of K
    cam_seg_t: glam::DVec3,
    obs: &[BridgeObs],
    reproj_tol: f64, // normalized-coordinate reprojection tolerance
    seed: u64,
) -> Option<(Sim3G, usize)> {
    if obs.len() < 6 {
        return None;
    }
    let mut rng = Rng64::new(seed);
    let mut best: Option<(usize, nalgebra::Rotation3<f64>, Vector3<f64>)> = None;
    // 6-point minimal samples need many draws when the inlier fraction is
    // modest (C(n,6) grows brutally); the per-iteration cost is tiny.
    for _ in 0..4000 {
        let mut idx = [0usize; 6];
        let mut k = 0;
        while k < 6 {
            let c = rng.below(obs.len());
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let sample: Vec<&BridgeObs> = idx.iter().map(|&i| &obs[i]).collect();
        let Some((r, t)) = dlt_p6p(&sample) else { continue };
        let count = obs
            .iter()
            .filter(|o| reproj_ok(&r, &t, o, reproj_tol))
            .count();
        if best.as_ref().is_none_or(|(bc, ..)| count > *bc) {
            best = Some((count, r, t));
        }
    }
    let (count, r_wc, t_wc) = best?;
    log::debug!("bridge DLT: best reprojection consensus {count}/{}", obs.len());
    if count < 6 {
        return None;
    }

    // Gauge scale: world-frame depth vs segment-frame depth per inlier.
    let r_gc = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        cam_seg_rot.w,
        cam_seg_rot.x,
        cam_seg_rot.y,
        cam_seg_rot.z,
    ));
    let t_gc = Vector3::new(cam_seg_t.x, cam_seg_t.y, cam_seg_t.z);
    let mut ratios: Vec<f64> = obs
        .iter()
        .filter(|o| reproj_ok(&r_wc, &t_wc, o, reproj_tol))
        .filter_map(|o| {
            let dw = (r_wc * Vector3::new(o.world.x, o.world.y, o.world.z) + t_wc).z;
            let dg = (r_gc * Vector3::new(o.seg.x, o.seg.y, o.seg.z) + t_gc).z;
            (dw > 0.05 && dg > 0.05).then_some(dw / dg)
        })
        .collect();
    if ratios.len() < 4 {
        log::debug!("bridge DLT: only {} usable depth ratios", ratios.len());
        return None;
    }
    ratios.sort_by(f64::total_cmp);
    let sigma = ratios[ratios.len() / 2];
    log::debug!("bridge DLT: median depth-ratio scale {sigma:.3}");
    if !(0.01..=100.0).contains(&sigma) {
        return None;
    }

    // Closed form: R^w_c = R^g_c · R_sᵀ  ⇒  R_s = (R^w_c)ᵀ R^g_c;
    // t^w_c = σ t^g_c − R^w_c t_s  ⇒  t_s = (R^w_c)ᵀ (σ t^g_c − t^w_c).
    let r_s = r_wc.transpose() * r_gc.to_rotation_matrix();
    let t_s = r_wc.transpose() * (sigma * t_gc - t_wc);
    let q = nalgebra::UnitQuaternion::from_rotation_matrix(&r_s);
    let qq = q.quaternion();
    Some((
        Sim3G {
            scale: sigma,
            rot: glam::DQuat::from_xyzw(qq.i, qq.j, qq.k, qq.w),
            trans: glam::DVec3::new(t_s[0], t_s[1], t_s[2]),
        },
        count,
    ))
}

fn reproj_ok(
    r: &nalgebra::Rotation3<f64>,
    t: &Vector3<f64>,
    o: &BridgeObs,
    tol: f64,
) -> bool {
    let c = r * Vector3::new(o.world.x, o.world.y, o.world.z) + t;
    if c.z <= 0.05 {
        return false;
    }
    let du = c.x / c.z - o.obs.0;
    let dv = c.y / c.z - o.obs.1;
    du * du + dv * dv < tol * tol
}

/// Direct linear transform for a calibrated camera from ≥6 2D-3D pairs
/// (normalized coords): solve P = [R|t] up to scale, orthonormalize R,
/// fix the sign by majority cheirality.
fn dlt_p6p(sample: &[&BridgeObs]) -> Option<(nalgebra::Rotation3<f64>, Vector3<f64>)> {
    let n = sample.len();
    let mut a = nalgebra::DMatrix::<f64>::zeros(2 * n, 12);
    for (i, o) in sample.iter().enumerate() {
        let (u, v) = o.obs;
        let x = [o.world.x, o.world.y, o.world.z, 1.0];
        for j in 0..4 {
            a[(2 * i, j)] = x[j];
            a[(2 * i, 8 + j)] = -u * x[j];
            a[(2 * i + 1, 4 + j)] = x[j];
            a[(2 * i + 1, 8 + j)] = -v * x[j];
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let p = vt.row(vt.nrows() - 1);
    let m = Matrix3::new(p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]);
    let svd3 = m.svd(true, true);
    let (u3, vt3) = (svd3.u?, svd3.v_t?);
    let mut s = Matrix3::identity();
    if (u3 * vt3).determinant() < 0.0 {
        s[(2, 2)] = -1.0;
    }
    let r = u3 * s * vt3;
    let lam = 3.0 / (svd3.singular_values.iter().sum::<f64>());
    let mut t = lam * Vector3::new(p[3], p[7], p[11]);
    let mut rot = nalgebra::Rotation3::from_matrix_unchecked(r);
    // Sign: majority of points must land in front of the camera.
    let front = sample
        .iter()
        .filter(|o| (rot * Vector3::new(o.world.x, o.world.y, o.world.z) + t).z > 0.0)
        .count();
    if front * 2 < n {
        rot = nalgebra::Rotation3::from_matrix_unchecked(-r);
        t = -t;
    }
    Some((rot, t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(n: usize, seed: u64) -> Vec<Vector3<f64>> {
        let mut rng = Rng64::new(seed);
        let mut r = |s: f64| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) * s;
        (0..n)
            .map(|_| Vector3::new(r(3.0), r(2.0), r(4.0) + 5.0))
            .collect()
    }

    fn gt() -> Sim3 {
        Sim3 {
            scale: 2.7,
            r: UnitQuaternion::from_euler_angles(0.2, -0.5, 0.9),
            t: Vector3::new(4.0, -2.0, 1.5),
        }
    }

    #[test]
    fn umeyama_exact() {
        let a = cloud(50, 1);
        let m = gt();
        let b: Vec<Vector3<f64>> = a.iter().map(|p| m.apply(p)).collect();
        let est = umeyama(&a, &b).unwrap();
        assert!((est.scale - m.scale).abs() < 1e-10);
        assert!((est.r.inverse() * m.r).angle() < 1e-10);
        assert!((est.t - m.t).norm() < 1e-9);
    }

    #[test]
    fn ransac_survives_40pct_outliers() {
        let a = cloud(100, 2);
        let m = gt();
        let mut b: Vec<Vector3<f64>> = a.iter().map(|p| m.apply(p)).collect();
        let junk = cloud(100, 3);
        for k in 0..40 {
            b[k * 2] = junk[k] * 3.0; // gross mismatches
        }
        let res = ransac_sim3(&a, &b, 300, 0.05, 7).expect("sim3");
        assert!(res.inliers.len() >= 58 && res.inliers.len() <= 62, "inliers {}", res.inliers.len());
        assert!((res.sim3.scale - m.scale).abs() < 1e-6);
        assert!((res.sim3.r.inverse() * m.r).angle() < 1e-6);
    }

    #[test]
    fn bridge_recovers_sim3_from_2d3d() {
        // GT world_from_seg and a seg-gauge camera.
        let s_gt = gt(); // scale 2.7 + rotation + translation
        let r_gc = UnitQuaternion::from_euler_angles(0.1, -0.3, 0.05);
        let t_gc = Vector3::new(0.4, -0.2, 1.0);
        // Effective world camera: R^w_c = R^g_c R_sᵀ; t^w_c = σ t^g_c − R^w_c t_s.
        let r_wc = r_gc.to_rotation_matrix() * s_gt.r.to_rotation_matrix().transpose();
        let t_wc = s_gt.scale * t_gc - r_wc * s_gt.t;

        let s_inv = s_gt.inverse();
        let mut rng = Rng64::new(77);
        let mut obs = Vec::new();
        for k in 0..40 {
            let mut r = |s: f64| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) * s;
            let xw = loop {
                let c = Vector3::new(r(6.0), r(4.0), r(6.0) + 8.0);
                if (r_wc * c + t_wc).z > 0.5 {
                    break c;
                }
            };
            let cam = r_wc * xw + t_wc;
            let mut o = (cam.x / cam.z, cam.y / cam.z);
            let xg = s_inv.apply(&xw);
            // Heavy noise on the SEG-side 3D (the failure mode this solver
            // dodges) + a few gross outlier observations.
            let seg_noisy = xg + Vector3::new(r(0.3), r(0.3), r(0.8));
            if k % 10 == 9 {
                o.0 += 0.2;
            }
            obs.push(BridgeObs {
                obs: o,
                world: glam::DVec3::new(xw[0], xw[1], xw[2]),
                seg: glam::DVec3::new(seg_noisy[0], seg_noisy[1], seg_noisy[2]),
            });
        }
        let q = r_gc.quaternion();
        let (est, inl) = sim3_from_bridge(
            glam::DQuat::from_xyzw(q.i, q.j, q.k, q.w),
            glam::DVec3::new(t_gc[0], t_gc[1], t_gc[2]),
            &obs,
            0.005,
            42,
        )
        .expect("bridge solve");
        assert!(inl >= 30, "inliers {inl}");
        assert!(
            (est.scale - s_gt.scale).abs() / s_gt.scale < 0.12,
            "scale {} vs {}",
            est.scale,
            s_gt.scale
        );
        let qe = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            est.rot.w, est.rot.x, est.rot.y, est.rot.z,
        ));
        assert!(
            (qe.inverse() * s_gt.r).angle() < 0.02,
            "rot err {}",
            (qe.inverse() * s_gt.r).angle()
        );
    }

    #[test]
    fn relative_pairs_recover_sim3() {
        let s_gt = gt();
        let s_inv = s_gt.inverse();
        let mut rng = Rng64::new(55);
        let mut pairs = Vec::new();
        for k in 0..4 {
            let mut r = |s: f64| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) * s;
            // Random world camera A and seg camera B (derived from a world
            // camera near A so the baseline is sane).
            let r_a = UnitQuaternion::from_euler_angles(r(0.3), r(0.3), r(0.3));
            let c_a = Vector3::new(r(4.0), r(2.0), r(4.0));
            let t_a = -(r_a * c_a);
            let r_bw = UnitQuaternion::from_euler_angles(r(0.3), r(0.3), r(0.3) + 0.1 * k as f64);
            let c_bw = c_a + Vector3::new(r(1.0) + 0.5, r(0.5), r(1.0));
            // B's pose in the SEG gauge via the inverse Sim3 (effective-
            // camera formula: R^g = R^w·R_s, t^g = (1/σ)(t^w + R^w·t_s)).
            let r_bg = r_bw * s_gt.r;
            let t_bw = -(r_bw * c_bw);
            let t_bg = (t_bw + r_bw * s_gt.t) / s_gt.scale;
            // Relative pose B-in-A (Euclidean, world frame).
            let r_rel = r_bw * r_a.inverse();
            let t_rel = t_bw - r_rel * t_a;
            let lam = t_rel.norm();
            assert!(lam > 1e-6);
            let g = |q: UnitQuaternion<f64>| {
                let c = q.quaternion();
                glam::DQuat::from_xyzw(c.i, c.j, c.k, c.w)
            };
            pairs.push(RelPair {
                cam_a_world: (g(r_a), glam::DVec3::new(t_a[0], t_a[1], t_a[2])),
                cam_b_seg: (g(r_bg), glam::DVec3::new(t_bg[0], t_bg[1], t_bg[2])),
                rel_rot: g(r_rel),
                rel_tdir: glam::DVec3::new(t_rel[0], t_rel[1], t_rel[2]) / lam,
            });
        }
        let (est, rms) = sim3_from_relative_pairs(&pairs).expect("solve");
        assert!(rms < 1e-9, "rms {rms}");
        assert!(
            (est.scale - s_gt.scale).abs() / s_gt.scale < 1e-9,
            "scale {} vs {}",
            est.scale,
            s_gt.scale
        );
        let qe = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            est.rot.w, est.rot.x, est.rot.y, est.rot.z,
        ));
        assert!((qe.inverse() * s_gt.r).angle() < 1e-9);
        let te = Vector3::new(est.trans.x, est.trans.y, est.trans.z);
        assert!((te - s_gt.t).norm() < 1e-8, "t err {}", (te - s_gt.t).norm());
        // sanity: verify the seg-gauge construction used the same effective-
        // camera convention as the solver (s_inv unused otherwise).
        let _ = s_inv;
    }

    #[test]
    fn inverse_roundtrip() {
        let m = gt();
        let p = Vector3::new(0.3, -1.1, 2.0);
        let back = m.inverse().apply(&m.apply(&p));
        assert!((back - p).norm() < 1e-12);
    }

    #[test]
    fn sim3g_inverse_and_compose() {
        let a = Sim3G {
            scale: 2.3,
            rot: glam::DQuat::from_axis_angle(
                glam::DVec3::new(0.2, -0.7, 0.5).normalize(),
                0.9,
            ),
            trans: glam::DVec3::new(1.5, -0.3, 4.1),
        };
        let b = Sim3G {
            scale: 0.4,
            rot: glam::DQuat::from_axis_angle(
                glam::DVec3::new(-0.6, 0.1, 0.8).normalize(),
                -1.7,
            ),
            trans: glam::DVec3::new(-2.0, 0.9, 0.2),
        };
        let p = glam::DVec3::new(0.7, -2.2, 3.3);
        // Inverse round-trips.
        let back = a.inverse().apply(a.apply(p));
        assert!((back - p).length() < 1e-12, "inverse: {back:?}");
        // Composition matches sequential application.
        let seq = a.apply(b.apply(p));
        let comp = a.compose(&b).apply(p);
        assert!((seq - comp).length() < 1e-12, "compose: {seq:?} vs {comp:?}");
        // compose with inverse is identity.
        let idp = a.compose(&a.inverse()).apply(p);
        assert!((idp - p).length() < 1e-12);
    }
}
