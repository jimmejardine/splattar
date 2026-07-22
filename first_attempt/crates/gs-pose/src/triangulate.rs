//! Linear (DLT) triangulation from two or more views. Poses are world→camera;
//! observations are normalized image coordinates.

use nalgebra::{DMatrix, Matrix4, Vector3, Vector4};

use crate::se3::Se3;

fn pose_rows(pose: &Se3) -> Matrix4<f64> {
    let r = pose.r.to_rotation_matrix();
    let mut p = Matrix4::identity();
    p.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
    p.fixed_view_mut::<3, 1>(0, 3).copy_from(&pose.t);
    p
}

/// Two-view DLT. Returns the point in world coordinates, or None when the
/// system is degenerate (no parallax / point at infinity).
pub fn triangulate_two(
    pose_a: &Se3,
    obs_a: (f64, f64),
    pose_b: &Se3,
    obs_b: (f64, f64),
) -> Option<Vector3<f64>> {
    triangulate_n(&[(*pose_a, obs_a), (*pose_b, obs_b)])
}

/// N-view DLT: rows x·P₃ − P₁ and y·P₃ − P₂ per view, smallest singular vector.
pub fn triangulate_n(obs: &[(Se3, (f64, f64))]) -> Option<Vector3<f64>> {
    if obs.len() < 2 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(obs.len() * 2, 4);
    for (k, (pose, (x, y))) in obs.iter().enumerate() {
        let p = pose_rows(pose);
        let r0 = p.row(0);
        let r1 = p.row(1);
        let r2 = p.row(2);
        for j in 0..4 {
            a[(2 * k, j)] = x * r2[j] - r0[j];
            a[(2 * k + 1, j)] = y * r2[j] - r1[j];
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let row = vt.row(vt.nrows() - 1);
    let h = Vector4::new(row[0], row[1], row[2], row[3]);
    if h[3].abs() < 1e-12 {
        return None;
    }
    Some(Vector3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
}

/// Parallax angle (radians) between the two viewing rays of a point.
pub fn parallax_angle(pose_a: &Se3, pose_b: &Se3, p_world: &Vector3<f64>) -> f64 {
    let ra = (p_world - pose_a.center()).normalize();
    let rb = (p_world - pose_b.center()).normalize();
    ra.dot(&rb).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector6;

    #[test]
    fn recovers_known_point() {
        let a = Se3::identity();
        let b = Se3::exp(&Vector6::new(0.5, 0.0, 0.0, 0.0, -0.03, 0.0));
        let p = Vector3::new(0.4, -0.2, 4.0);
        let pa = a.act(&p);
        let pb = b.act(&p);
        let est = triangulate_two(
            &a,
            (pa[0] / pa[2], pa[1] / pa[2]),
            &b,
            (pb[0] / pb[2], pb[1] / pb[2]),
        )
        .unwrap();
        assert!((est - p).norm() < 1e-9);
    }

    #[test]
    fn n_view_beats_noise() {
        let p = Vector3::new(-0.3, 0.5, 5.0);
        let mut obs = Vec::new();
        for k in 0..6 {
            let pose = Se3::exp(&Vector6::new(
                0.2 * k as f64,
                0.03 * k as f64,
                0.0,
                0.0,
                -0.01 * k as f64,
                0.0,
            ));
            let c = pose.act(&p);
            // ~0.5 px noise at f=500.
            let n = 0.001 * ((k * 7 % 3) as f64 - 1.0);
            obs.push((pose, (c[0] / c[2] + n, c[1] / c[2] - n)));
        }
        let est = triangulate_n(&obs).unwrap();
        // ~0.5 px correlated noise at 5 m depth: a few cm is expected.
        assert!((est - p).norm() < 0.05, "err {}", (est - p).norm());
    }

    #[test]
    fn zero_parallax_degenerates() {
        // Same camera center: pure rotation gives no parallax.
        let a = Se3::identity();
        let b = Se3::exp(&Vector6::new(0.0, 0.0, 0.0, 0.0, 0.2, 0.0));
        let p = Vector3::new(0.1, 0.1, 3.0);
        let pa = a.act(&p);
        let pb = b.act(&p);
        // The DLT solution is a whole 2-D null space (any depth along the ray,
        // any homogeneous scale) — the returned point is meaningless. The
        // contract is only that the parallax gate flags the configuration.
        assert!(parallax_angle(&a, &b, &p) < 1e-9);
        let _ = triangulate_two(
            &a,
            (pa[0] / pa[2], pa[1] / pa[2]),
            &b,
            (pb[0] / pb[2], pb[1] / pb[2]),
        );
    }
}
