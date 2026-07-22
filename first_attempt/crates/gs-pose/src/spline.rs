//! PTS-parameterized SE(3) trajectory: C¹ Hermite interpolation on the
//! manifold (geodesic cubic via De Casteljau on se(3) increments, tangents
//! from central differences). Interpolates keyframe poses exactly — the
//! property per-frame pose lookup actually needs; a smoothing cumulative
//! B-spline can replace it later if continuous-time BA arrives.

use nalgebra::Vector6;

use crate::se3::Se3;

pub struct PoseSpline {
    /// Strictly increasing PTS knots.
    pts: Vec<f64>,
    poses: Vec<Se3>,
    /// Per-knot tangents in the local frame (se3, per unit time).
    tangents: Vec<Vector6<f64>>,
}

impl PoseSpline {
    /// Build from (pts, world→camera pose) samples; pts must be strictly
    /// increasing and there must be at least one sample.
    pub fn fit(samples: &[(f64, Se3)]) -> Option<PoseSpline> {
        if samples.is_empty() {
            return None;
        }
        let n = samples.len();
        let pts: Vec<f64> = samples.iter().map(|s| s.0).collect();
        if pts.windows(2).any(|w| w[1] <= w[0]) {
            return None;
        }
        let poses: Vec<Se3> = samples.iter().map(|s| s.1).collect();

        // Local increment between consecutive poses: ξ_k = log(T_k⁻¹ ∘ T_{k+1})
        // (right-increment convention; evaluation matches below).
        let mut tangents = vec![Vector6::zeros(); n];
        if n >= 2 {
            for k in 0..n {
                let (a, b, dt) = if k == 0 {
                    (0, 1, pts[1] - pts[0])
                } else if k == n - 1 {
                    (n - 2, n - 1, pts[n - 1] - pts[n - 2])
                } else {
                    (k - 1, k + 1, pts[k + 1] - pts[k - 1])
                };
                let inc = poses[a].inverse().compose(&poses[b]).log();
                tangents[k] = inc / dt;
            }
        }
        Some(PoseSpline {
            pts,
            poses,
            tangents,
        })
    }

    pub fn t_min(&self) -> f64 {
        self.pts[0]
    }

    pub fn t_max(&self) -> f64 {
        *self.pts.last().unwrap()
    }

    /// Evaluate at time t (clamped to the knot range).
    pub fn sample(&self, t: f64) -> Se3 {
        let n = self.pts.len();
        if n == 1 || t <= self.pts[0] {
            return self.poses[0];
        }
        if t >= self.pts[n - 1] {
            return self.poses[n - 1];
        }
        let k = match self
            .pts
            .binary_search_by(|p| p.partial_cmp(&t).unwrap())
        {
            Ok(i) => return self.poses[i],
            Err(i) => i - 1,
        };
        let dt = self.pts[k + 1] - self.pts[k];
        let u = (t - self.pts[k]) / dt;

        // Hermite in the local chart at pose k: x(u) over se(3) with
        // endpoints 0 and ξ01 = log(T_k⁻¹ T_{k+1}), derivatives scaled to dt.
        let xi01 = self.poses[k].inverse().compose(&self.poses[k + 1]).log();
        let m0 = self.tangents[k] * dt;
        let m1 = self.tangents[k + 1] * dt;
        let u2 = u * u;
        let u3 = u2 * u;
        // h00 multiplies the zero start point and drops out.
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        // h00·0 + h10·m0 + h01·ξ01 + h11·m1
        let x = m0 * h10 + xi01 * h01 + m1 * h11;
        self.poses[k].compose(&Se3::exp(&x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(t: f64) -> Se3 {
        Se3::exp(&Vector6::new(
            0.8 * t,
            0.1 * (t * 1.3).sin(),
            0.05 * t,
            0.2 * (t * 0.7).sin(),
            0.15 * t,
            0.1 * (t * 0.9).cos() - 0.1,
        ))
    }

    #[test]
    fn interpolates_knots_exactly() {
        let samples: Vec<(f64, Se3)> = (0..8).map(|k| (k as f64 * 0.4, traj(k as f64 * 0.4))).collect();
        let sp = PoseSpline::fit(&samples).unwrap();
        for (t, pose) in &samples {
            let s = sp.sample(*t);
            assert!((s.t - pose.t).norm() < 1e-12);
            assert!((s.r.inverse() * pose.r).angle() < 1e-12);
        }
    }

    #[test]
    fn tracks_smooth_trajectory_between_knots() {
        // Knots every 0.4 s on a smooth path; mid-knot error should be small.
        let samples: Vec<(f64, Se3)> = (0..10).map(|k| (k as f64 * 0.4, traj(k as f64 * 0.4))).collect();
        let sp = PoseSpline::fit(&samples).unwrap();
        for k in 0..40 {
            let t = 0.05 + k as f64 * 0.09;
            if t >= 3.6 {
                break;
            }
            let gt = traj(t);
            let est = sp.sample(t);
            assert!(
                (est.t - gt.t).norm() < 0.01,
                "t={t}: trans err {}",
                (est.t - gt.t).norm()
            );
            assert!(
                (est.r.inverse() * gt.r).angle() < 0.01,
                "t={t}: rot err {}",
                (est.r.inverse() * gt.r).angle()
            );
        }
    }

    #[test]
    fn clamps_out_of_range() {
        let samples = vec![(1.0, traj(1.0)), (2.0, traj(2.0))];
        let sp = PoseSpline::fit(&samples).unwrap();
        assert_eq!(sp.sample(0.0), samples[0].1);
        assert_eq!(sp.sample(9.0), samples[1].1);
    }

    #[test]
    fn rejects_non_monotonic() {
        let samples = vec![(1.0, traj(1.0)), (0.5, traj(0.5))];
        assert!(PoseSpline::fit(&samples).is_none());
    }
}
