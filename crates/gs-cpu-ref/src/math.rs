//! Differentiable building blocks: quaternion→rotation with analytic
//! derivatives (through normalization), scalar triple products with
//! cross-product gradients, and SH basis values + direction derivatives.
//!
//! Everything here is verified indirectly by the finite-difference gate; keep
//! the implementations boring and explicit.

use glam::{DMat3, DVec3};

// ---------------------------------------------------------------- rotation

/// Rotation matrix from an unnormalized quaternion (xyzw), columns = images
/// of e0/e1/e2.
pub fn quat_to_mat(q: [f64; 4]) -> DMat3 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (x, y, z, w) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    DMat3::from_cols(
        DVec3::new(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        DVec3::new(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        DVec3::new(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )
}

/// Given dL/dR (as a matrix of per-entry gradients), accumulate dL/dq for the
/// unnormalized quaternion. Chains through the normalization Jacobian.
pub fn quat_grad(q: [f64; 4], dl_dr: &DMat3) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let qh = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    let (x, y, z, w) = (qh[0], qh[1], qh[2], qh[3]);

    // dR/dq̂_k, from differentiating the entries of quat_to_mat.
    let dr = |k: usize| -> DMat3 {
        match k {
            // ∂/∂x
            0 => DMat3::from_cols(
                DVec3::new(0.0, 2.0 * y, 2.0 * z),
                DVec3::new(2.0 * y, -4.0 * x, 2.0 * w),
                DVec3::new(2.0 * z, -2.0 * w, -4.0 * x),
            ),
            // ∂/∂y
            1 => DMat3::from_cols(
                DVec3::new(-4.0 * y, 2.0 * x, -2.0 * w),
                DVec3::new(2.0 * x, 0.0, 2.0 * z),
                DVec3::new(2.0 * w, 2.0 * z, -4.0 * y),
            ),
            // ∂/∂z
            2 => DMat3::from_cols(
                DVec3::new(-4.0 * z, 2.0 * w, 2.0 * x),
                DVec3::new(-2.0 * w, -4.0 * z, 2.0 * y),
                DVec3::new(2.0 * x, 2.0 * y, 0.0),
            ),
            // ∂/∂w
            _ => DMat3::from_cols(
                DVec3::new(0.0, 2.0 * z, -2.0 * y),
                DVec3::new(-2.0 * z, 0.0, 2.0 * x),
                DVec3::new(2.0 * y, -2.0 * x, 0.0),
            ),
        }
    };

    // dL/dq̂_k = <dL/dR, dR/dq̂_k> (Frobenius inner product).
    let mut dqh = [0.0; 4];
    for (k, slot) in dqh.iter_mut().enumerate() {
        let d = dr(k);
        for col in 0..3 {
            *slot += dl_dr.col(col).dot(d.col(col));
        }
    }
    // Through normalization: dq = (I − q̂ q̂ᵀ)/n · dq̂.
    let dot: f64 = (0..4).map(|i| dqh[i] * qh[i]).sum();
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = (dqh[i] - qh[i] * dot) / n;
    }
    out
}

// ---------------------------------------------------------- triple products

/// det[a b c] = a · (b × c). Gradients: ∇a = b×c, ∇b = c×a, ∇c = a×b.
pub fn triple(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    a.dot(b.cross(c))
}

pub fn triple_grads(a: DVec3, b: DVec3, c: DVec3) -> (DVec3, DVec3, DVec3) {
    (b.cross(c), c.cross(a), a.cross(b))
}

// ------------------------------------------------------------- SH (deg 0–3)

const C0: f64 = 0.282_094_791_773_878_14;
const C1: f64 = 0.488_602_511_902_919_9;
const C2: [f64; 5] = [
    1.092_548_430_592_079_2,
    -1.092_548_430_592_079_2,
    0.315_391_565_252_520_05,
    -1.092_548_430_592_079_2,
    0.546_274_215_296_039_6,
];
const C3: [f64; 7] = [
    -0.590_043_589_926_643_5,
    2.890_611_442_640_554,
    -0.457_045_799_464_465_8,
    0.373_176_332_590_115_4,
    -0.457_045_799_464_465_8,
    1.445_305_721_320_277,
    -0.590_043_589_926_643_5,
];

pub fn num_coeffs(degree: u8) -> usize {
    (degree as usize + 1) * (degree as usize + 1)
}

/// Basis values b_k(dir) for a normalized direction.
pub fn sh_basis(degree: u8, d: DVec3) -> Vec<f64> {
    let (x, y, z) = (d.x, d.y, d.z);
    let mut b = vec![C0];
    if degree >= 1 {
        b.extend([-C1 * y, C1 * z, -C1 * x]);
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        b.extend([
            C2[0] * x * y,
            C2[1] * y * z,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * x * z,
            C2[4] * (xx - yy),
        ]);
    }
    if degree >= 3 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        b.extend([
            C3[0] * y * (3.0 * xx - yy),
            C3[1] * x * y * z,
            C3[2] * y * (4.0 * zz - xx - yy),
            C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            C3[4] * x * (4.0 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3.0 * yy),
        ]);
    }
    b
}

/// ∂b_k/∂dir for a normalized direction (before the normalization Jacobian).
pub fn sh_basis_grad(degree: u8, d: DVec3) -> Vec<DVec3> {
    let (x, y, z) = (d.x, d.y, d.z);
    let mut g = vec![DVec3::ZERO];
    if degree >= 1 {
        g.extend([
            DVec3::new(0.0, -C1, 0.0),
            DVec3::new(0.0, 0.0, C1),
            DVec3::new(-C1, 0.0, 0.0),
        ]);
    }
    if degree >= 2 {
        g.extend([
            C2[0] * DVec3::new(y, x, 0.0),
            C2[1] * DVec3::new(0.0, z, y),
            C2[2] * DVec3::new(-2.0 * x, -2.0 * y, 4.0 * z),
            C2[3] * DVec3::new(z, 0.0, x),
            C2[4] * DVec3::new(2.0 * x, -2.0 * y, 0.0),
        ]);
    }
    if degree >= 3 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        g.extend([
            C3[0] * DVec3::new(6.0 * x * y, 3.0 * xx - 3.0 * yy, 0.0),
            C3[1] * DVec3::new(y * z, x * z, x * y),
            C3[2] * DVec3::new(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z),
            C3[3] * DVec3::new(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy),
            C3[4] * DVec3::new(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z),
            C3[5] * DVec3::new(2.0 * x * z, -2.0 * y * z, xx - yy),
            C3[6] * DVec3::new(3.0 * xx - 3.0 * yy, -6.0 * x * y, 0.0),
        ]);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FD sanity for the local pieces (the full pipeline gate lives in
    /// tests/gradcheck.rs).
    #[test]
    fn quat_grad_matches_fd() {
        let q = [0.3, -0.5, 0.2, 0.8];
        // L = sum of all entries of R weighted by fixed weights.
        let w = DMat3::from_cols(
            DVec3::new(0.3, -1.2, 0.7),
            DVec3::new(0.9, 0.1, -0.4),
            DVec3::new(-0.6, 0.8, 0.5),
        );
        let loss = |q: [f64; 4]| -> f64 {
            let r = quat_to_mat(q);
            (0..3).map(|c| r.col(c).dot(w.col(c))).sum()
        };
        let analytic = quat_grad(q, &w);
        let h = 1e-6;
        for k in 0..4 {
            let mut qp = q;
            let mut qm = q;
            qp[k] += h;
            qm[k] -= h;
            let fd = (loss(qp) - loss(qm)) / (2.0 * h);
            assert!(
                (fd - analytic[k]).abs() < 1e-7,
                "quat grad {k}: fd {fd}, analytic {}",
                analytic[k]
            );
        }
    }

    #[test]
    fn triple_grads_match_fd() {
        let a = DVec3::new(0.3, -0.7, 1.1);
        let b = DVec3::new(-0.2, 0.9, 0.4);
        let c = DVec3::new(0.8, 0.1, -0.5);
        let (ga, gb, gc) = triple_grads(a, b, c);
        let h = 1e-6;
        for i in 0..3 {
            let mut e = DVec3::ZERO;
            e[i] = h;
            assert!(((triple(a + e, b, c) - triple(a - e, b, c)) / (2.0 * h) - ga[i]).abs() < 1e-8);
            assert!(((triple(a, b + e, c) - triple(a, b - e, c)) / (2.0 * h) - gb[i]).abs() < 1e-8);
            assert!(((triple(a, b, c + e) - triple(a, b, c - e)) / (2.0 * h) - gc[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn sh_basis_grad_matches_fd() {
        // Use an unnormalized-direction-free check: perturb dir directly
        // (normalization is chained separately in the renderer backward).
        let d = DVec3::new(0.48, -0.6, 0.64); // unit-ish; basis is polynomial, no need for exact unit
        for degree in 0..=3u8 {
            let g = sh_basis_grad(degree, d);
            let h = 1e-6;
            for i in 0..3 {
                let mut e = DVec3::ZERO;
                e[i] = h;
                let bp = sh_basis(degree, d + e);
                let bm = sh_basis(degree, d - e);
                for k in 0..bp.len() {
                    let fd = (bp[k] - bm[k]) / (2.0 * h);
                    assert!(
                        (fd - g[k][i]).abs() < 1e-6,
                        "deg {degree} basis {k} dim {i}: fd {fd}, analytic {}",
                        g[k][i]
                    );
                }
            }
        }
    }
}
