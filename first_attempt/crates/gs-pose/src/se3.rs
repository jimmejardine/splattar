//! SE(3) with exp/log maps (f64, nalgebra). Convention throughout gs-pose:
//! poses are **world→camera** (T_cw): `x_cam = R · x_world + t`. Convert to
//! the renderer's camera→world at the API boundary.

use nalgebra::{Matrix3, UnitQuaternion, Vector3, Vector6};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    pub r: UnitQuaternion<f64>,
    pub t: Vector3<f64>,
}

impl Default for Se3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Se3 {
    pub fn identity() -> Self {
        Self {
            r: UnitQuaternion::identity(),
            t: Vector3::zeros(),
        }
    }

    pub fn new(r: UnitQuaternion<f64>, t: Vector3<f64>) -> Self {
        Self { r, t }
    }

    /// Apply: world point → camera point.
    #[inline]
    pub fn act(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.r * p + self.t
    }

    /// Composition: (self ∘ other)(x) = self(other(x)).
    #[inline]
    pub fn compose(&self, other: &Se3) -> Se3 {
        Se3 {
            r: self.r * other.r,
            t: self.r * other.t + self.t,
        }
    }

    #[inline]
    pub fn inverse(&self) -> Se3 {
        let rinv = self.r.inverse();
        Se3 {
            r: rinv,
            t: -(rinv * self.t),
        }
    }

    /// Camera center in world coordinates: C = −Rᵀ t.
    #[inline]
    pub fn center(&self) -> Vector3<f64> {
        -(self.r.inverse() * self.t)
    }

    /// Exponential map. Twist layout: [ρ (translation), φ (rotation)].
    pub fn exp(xi: &Vector6<f64>) -> Se3 {
        let rho = Vector3::new(xi[0], xi[1], xi[2]);
        let phi = Vector3::new(xi[3], xi[4], xi[5]);
        let theta = phi.norm();
        let r = UnitQuaternion::from_scaled_axis(phi);
        let v = if theta < 1e-9 {
            Matrix3::identity() + 0.5 * skew(&phi)
        } else {
            let k = skew(&phi);
            Matrix3::identity()
                + ((1.0 - theta.cos()) / (theta * theta)) * k
                + ((theta - theta.sin()) / (theta * theta * theta)) * (k * k)
        };
        Se3 { r, t: v * rho }
    }

    /// Logarithm map, inverse of `exp`.
    pub fn log(&self) -> Vector6<f64> {
        let phi = self.r.scaled_axis();
        let theta = phi.norm();
        let v_inv = if theta < 1e-9 {
            Matrix3::identity() - 0.5 * skew(&phi)
        } else {
            let k = skew(&phi);
            let half = 0.5 * theta;
            let cot_half = half.cos() / half.sin();
            Matrix3::identity() - 0.5 * k
                + ((1.0 - half * cot_half) / (theta * theta)) * (k * k)
        };
        let rho = v_inv * self.t;
        Vector6::new(rho[0], rho[1], rho[2], phi[0], phi[1], phi[2])
    }

    /// Left-multiplicative update: exp(δ) ∘ self (the BA/PnP parameterization).
    pub fn boxplus(&self, delta: &Vector6<f64>) -> Se3 {
        Se3::exp(delta).compose(self)
    }
}

#[inline]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_roundtrip() {
        for k in 0..50 {
            let s = k as f64 * 0.13;
            let xi = Vector6::new(
                s.sin(),
                (s * 1.7).cos() * 0.5,
                s * 0.02,
                (s * 0.9).sin() * 2.0,
                (s * 1.3).cos(),
                (s * 0.4).sin() * 0.7,
            );
            let g = Se3::exp(&xi);
            let back = g.log();
            // log can differ by 2π in the axis-angle only near θ=π; keep θ<π here.
            assert!((back - xi).norm() < 1e-9, "k={k}: {:?}", (back - xi).norm());
        }
    }

    #[test]
    fn compose_inverse_is_identity() {
        let g = Se3::exp(&Vector6::new(0.3, -0.2, 0.9, 0.4, -0.6, 0.1));
        let e = g.compose(&g.inverse());
        assert!(e.t.norm() < 1e-12);
        assert!(e.r.angle() < 1e-12);
    }

    #[test]
    fn act_matches_compose() {
        let a = Se3::exp(&Vector6::new(0.1, 0.2, 0.3, -0.2, 0.15, 0.4));
        let b = Se3::exp(&Vector6::new(-0.4, 0.05, 0.2, 0.3, -0.1, 0.2));
        let p = Vector3::new(0.7, -1.2, 2.5);
        let via_compose = a.compose(&b).act(&p);
        let via_seq = a.act(&b.act(&p));
        assert!((via_compose - via_seq).norm() < 1e-12);
    }
}
