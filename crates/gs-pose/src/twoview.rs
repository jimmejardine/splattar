//! Two-view bootstrap: normalized eight-point essential matrix inside RANSAC,
//! Sampson-distance scoring, essential-manifold projection, and (R, t)
//! selection by cheirality. Coordinates are **normalized image coordinates**
//! (pixel offsets divided by focal length — intrinsics live with the caller).
//!
//! Known limitation (documented, guarded): pure-plane scenes are degenerate
//! for the eight-point solve. The bootstrap policy compensates upstream by
//! requiring parallax and a healthy inlier ratio before accepting.

use nalgebra::{DMatrix, Matrix3, UnitQuaternion, Vector3};

use crate::se3::Se3;
use crate::triangulate::triangulate_two;

/// A feature seen in two views (normalized image coords, z = 1 implied).
#[derive(Debug, Clone, Copy)]
pub struct Match2 {
    pub a: (f64, f64),
    pub b: (f64, f64),
}

/// Deterministic small RNG (xorshift64*) — geometry must be reproducible.
pub struct Rng64(u64);

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Eight-point (or more) least-squares essential matrix with Hartley
/// normalization, projected onto the essential manifold (σ = (1, 1, 0)).
pub fn eight_point(matches: &[Match2]) -> Option<Matrix3<f64>> {
    let n = matches.len();
    if n < 8 {
        return None;
    }
    // Hartley normalization per image: zero centroid, mean distance √2.
    let norm = |pts: &mut dyn Iterator<Item = (f64, f64)>| -> (f64, f64, f64) {
        let v: Vec<(f64, f64)> = pts.collect();
        let cx = v.iter().map(|p| p.0).sum::<f64>() / v.len() as f64;
        let cy = v.iter().map(|p| p.1).sum::<f64>() / v.len() as f64;
        let d = v
            .iter()
            .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
            .sum::<f64>()
            / v.len() as f64;
        let s = if d > 1e-12 { std::f64::consts::SQRT_2 / d } else { 1.0 };
        (cx, cy, s)
    };
    let (cax, cay, sa) = norm(&mut matches.iter().map(|m| m.a));
    let (cbx, cby, sb) = norm(&mut matches.iter().map(|m| m.b));

    let mut amat = DMatrix::<f64>::zeros(n, 9);
    for (i, m) in matches.iter().enumerate() {
        let x1 = (m.a.0 - cax) * sa;
        let y1 = (m.a.1 - cay) * sa;
        let x2 = (m.b.0 - cbx) * sb;
        let y2 = (m.b.1 - cby) * sb;
        // x2ᵀ E x1 = 0, row = [x2x1, x2y1, x2, y2x1, y2y1, y2, x1, y1, 1]
        amat[(i, 0)] = x2 * x1;
        amat[(i, 1)] = x2 * y1;
        amat[(i, 2)] = x2;
        amat[(i, 3)] = y2 * x1;
        amat[(i, 4)] = y2 * y1;
        amat[(i, 5)] = y2;
        amat[(i, 6)] = x1;
        amat[(i, 7)] = y1;
        amat[(i, 8)] = 1.0;
    }
    let svd = amat.svd(false, true);
    let vt = svd.v_t?;
    let e = vt.row(vt.nrows() - 1);
    let en = Matrix3::new(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);

    // Denormalize: E = T2ᵀ · En · T1.
    let t1 = Matrix3::new(sa, 0.0, -sa * cax, 0.0, sa, -sa * cay, 0.0, 0.0, 1.0);
    let t2 = Matrix3::new(sb, 0.0, -sb * cbx, 0.0, sb, -sb * cby, 0.0, 0.0, 1.0);
    let e_raw = t2.transpose() * en * t1;

    // Project to the essential manifold.
    let svd3 = e_raw.svd(true, true);
    let u = svd3.u?;
    let vt3 = svd3.v_t?;
    let e_proj = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)) * vt3;
    Some(e_proj)
}

/// First-order (Sampson) squared distance to the epipolar constraint.
#[inline]
pub fn sampson_sq(e: &Matrix3<f64>, m: &Match2) -> f64 {
    let x1 = Vector3::new(m.a.0, m.a.1, 1.0);
    let x2 = Vector3::new(m.b.0, m.b.1, 1.0);
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let num = x2.dot(&ex1);
    let den = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
    if den < 1e-18 {
        f64::INFINITY
    } else {
        num * num / den
    }
}

pub struct RansacResult {
    pub e: Matrix3<f64>,
    pub inliers: Vec<usize>,
}

/// RANSAC over minimal 8-point samples, refit on the consensus set.
/// `thresh` is the Sampson distance in normalized coords (e.g. 1.5 px / focal).
pub fn ransac_essential(
    matches: &[Match2],
    iters: usize,
    thresh: f64,
    seed: u64,
) -> Option<RansacResult> {
    if matches.len() < 8 {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Rng64::new(seed);
    let mut best: Option<(usize, Matrix3<f64>)> = None;

    for _ in 0..iters {
        // Sample 8 distinct indices.
        let mut idx = [0usize; 8];
        let mut k = 0;
        while k < 8 {
            let cand = rng.below(matches.len());
            if !idx[..k].contains(&cand) {
                idx[k] = cand;
                k += 1;
            }
        }
        let sample: Vec<Match2> = idx.iter().map(|&i| matches[i]).collect();
        let Some(e) = eight_point(&sample) else {
            continue;
        };
        let count = matches.iter().filter(|m| sampson_sq(&e, m) < t2).count();
        if best.is_none_or(|(bc, _)| count > bc) {
            best = Some((count, e));
        }
    }

    let (_, e0) = best?;
    // Refit on inliers of the best model, then reclassify (one IRLS-ish round).
    let inl: Vec<Match2> = matches
        .iter()
        .filter(|m| sampson_sq(&e0, m) < t2)
        .copied()
        .collect();
    let e1 = eight_point(&inl).unwrap_or(e0);
    let inliers: Vec<usize> = matches
        .iter()
        .enumerate()
        .filter(|(_, m)| sampson_sq(&e1, m) < t2)
        .map(|(i, _)| i)
        .collect();
    if inliers.len() < 8 {
        return None;
    }
    Some(RansacResult { e: e1, inliers })
}

pub struct Bootstrap {
    /// Pose of view B in A's frame: x_b = R x_a + t, ‖t‖ = 1 (scale-free).
    pub pose_ba: Se3,
    /// Triangulated points in A's frame, paired with match indices.
    pub points: Vec<(usize, Vector3<f64>)>,
}

/// Decompose E into the four (R, t) candidates and pick the one with the most
/// points in front of both cameras; triangulate the survivors.
pub fn recover_pose(e: &Matrix3<f64>, matches: &[Match2], inliers: &[usize]) -> Option<Bootstrap> {
    recover_pose_with(e, matches, inliers, 0.75)
}

/// [`recover_pose`] with a caller-chosen cheirality majority. VO bootstrap
/// wants the strict 75%; cross-take registration pairs are noisier and the
/// downstream rotation-clustering rejects bad decompositions anyway.
pub fn recover_pose_with(
    e: &Matrix3<f64>,
    matches: &[Match2],
    inliers: &[usize],
    min_cheirality: f64,
) -> Option<Bootstrap> {
    let svd = e.svd(true, true);
    let mut u = svd.u?;
    let mut vt = svd.v_t?;
    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = u * w * vt;
    let r2 = u * w.transpose() * vt;
    let t = u.column(2).into_owned();

    let mut best: Option<(usize, Bootstrap)> = None;
    for (r, tv) in [(r1, t), (r1, -t), (r2, t), (r2, -t)] {
        let pose = Se3::new(
            UnitQuaternion::from_matrix(&r),
            Vector3::new(tv[0], tv[1], tv[2]),
        );
        let ident = Se3::identity();
        let mut pts = Vec::new();
        let mut good = 0usize;
        for &i in inliers {
            let m = &matches[i];
            if let Some(p) = triangulate_two(&ident, m.a, &pose, m.b) {
                let za = p[2];
                let zb = pose.act(&p)[2];
                if za > 0.0 && zb > 0.0 {
                    good += 1;
                    pts.push((i, p));
                }
            }
        }
        if best.as_ref().is_none_or(|(bg, _)| good > *bg) {
            best = Some((
                good,
                Bootstrap {
                    pose_ba: pose,
                    points: pts,
                },
            ));
        }
    }
    let (good, boot) = best?;
    // Require a decisive cheirality vote — ambiguity means bad geometry.
    (good as f64 >= inliers.len() as f64 * min_cheirality).then_some(boot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector6;

    /// Random-ish deterministic scene: points in a box in front of camera A.
    fn scene(n: usize) -> (Vec<Vector3<f64>>, Se3) {
        let mut rng = Rng64::new(42);
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            let r = |rng: &mut Rng64| (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            pts.push(Vector3::new(
                r(&mut rng) * 2.0,
                r(&mut rng) * 1.5,
                3.0 + r(&mut rng) * 1.2,
            ));
        }
        // Camera B: translated + slightly rotated (world→camera convention;
        // A is the identity/world frame).
        let pose_ba = Se3::exp(&Vector6::new(0.4, 0.05, 0.1, 0.02, -0.06, 0.01));
        (pts, pose_ba)
    }

    fn project(pose: &Se3, p: &Vector3<f64>) -> (f64, f64) {
        let c = pose.act(p);
        (c[0] / c[2], c[1] / c[2])
    }

    #[test]
    fn bootstrap_recovers_relative_pose() {
        let (pts, pose_ba) = scene(120);
        let ident = Se3::identity();
        let matches: Vec<Match2> = pts
            .iter()
            .map(|p| Match2 {
                a: project(&ident, p),
                b: project(&pose_ba, p),
            })
            .collect();

        let res = ransac_essential(&matches, 200, 1e-3, 7).expect("ransac");
        assert!(res.inliers.len() >= 110, "inliers {}", res.inliers.len());
        let boot = recover_pose(&res.e, &matches, &res.inliers).expect("pose");

        // Rotation must match; translation up to scale.
        let dr = boot.pose_ba.r.inverse() * pose_ba.r;
        assert!(dr.angle() < 1e-3, "rot err {}", dr.angle());
        let t_est = boot.pose_ba.t.normalize();
        let t_gt = pose_ba.t.normalize();
        assert!((t_est - t_gt).norm() < 1e-2, "t dir err {}", (t_est - t_gt).norm());

        // Triangulated points match GT up to the global scale factor.
        let scale = pose_ba.t.norm() / boot.pose_ba.t.norm();
        for (i, p) in &boot.points {
            let gt = &pts[*i];
            assert!((p * scale - gt).norm() < 1e-2 * gt.norm());
        }
    }

    #[test]
    fn ransac_survives_outliers() {
        let (pts, pose_ba) = scene(100);
        let ident = Se3::identity();
        let mut matches: Vec<Match2> = pts
            .iter()
            .map(|p| Match2 {
                a: project(&ident, p),
                b: project(&pose_ba, p),
            })
            .collect();
        // 25% gross outliers.
        let mut rng = Rng64::new(99);
        for k in 0..25 {
            let i = k * 4;
            matches[i].b.0 += (rng.next_u64() as f64 / u64::MAX as f64) * 0.4 + 0.05;
            matches[i].b.1 -= (rng.next_u64() as f64 / u64::MAX as f64) * 0.3 + 0.05;
        }
        let res = ransac_essential(&matches, 500, 1e-3, 3).expect("ransac");
        assert!(res.inliers.len() >= 70 && res.inliers.len() <= 80,
            "inliers {}", res.inliers.len());
        let boot = recover_pose(&res.e, &matches, &res.inliers).expect("pose");
        let dr = boot.pose_ba.r.inverse() * pose_ba.r;
        assert!(dr.angle() < 5e-3, "rot err {}", dr.angle());
    }
}
