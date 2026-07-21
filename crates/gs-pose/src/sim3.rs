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
        if !(0.01..=100.0).contains(&m.scale) {
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
    let na: Vec<Vector3<f64>> = a.iter().map(|p| Vector3::new(p.x, p.y, p.z)).collect();
    let nb: Vec<Vector3<f64>> = b.iter().map(|p| Vector3::new(p.x, p.y, p.z)).collect();
    let res = ransac_sim3(&na, &nb, iters, thresh, seed)?;
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
    fn inverse_roundtrip() {
        let m = gt();
        let p = Vector3::new(0.3, -1.1, 2.0);
        let back = m.inverse().apply(&m.apply(&p));
        assert!((back - p).norm() < 1e-12);
    }
}
