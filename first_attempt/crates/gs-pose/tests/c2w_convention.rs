//! The untested seam: `KeyframePose::c2w()` hands VO geometry to the
//! renderer/trainer. A landmark VO observed at pixel (u, v) must project back
//! to (u, v) through the renderer's camera math, or every training view sees
//! the reconstructed cloud through a wrong camera and the model fits mush.
//!
//! VO is world→camera, y down, +z forward (`se3.rs`); the renderer is y up,
//! looks −z (`surfel_prep_fwd.wgsl`). This test pins the conversion between
//! them end to end rather than asserting a matrix shape.

use gs_pose::se3::Se3;
use gs_pose::vo::KeyframePose;
use nalgebra::{UnitQuaternion, Vector3};

const FOCAL: f64 = 500.0;
const W: f64 = 640.0;
const H: f64 = 480.0;

/// VO's own projection: `x_cam = R·p + t`, y down, +z forward.
fn project_vo(pose: &Se3, p: Vector3<f64>) -> (f64, f64) {
    let c = pose.act(&p);
    assert!(c.z > 0.0, "test point must be in front of the camera");
    (W * 0.5 + FOCAL * c.x / c.z, H * 0.5 + FOCAL * c.y / c.z)
}

/// The renderer's projection, transcribed from `surfel_prep_fwd.wgsl`:
/// `x = p − center; c = R_c2wᵀ·x; depth = −c.z; px = cx + f·c.x/depth;
/// py = cy − f·c.y/depth`.
fn project_renderer(kp: &KeyframePose, p: Vector3<f64>) -> (f64, f64) {
    let c2w = kp.c2w();
    let center = c2w.w_axis.truncate();
    let r_c2w = glam::DMat3::from_cols(
        c2w.x_axis.truncate(),
        c2w.y_axis.truncate(),
        c2w.z_axis.truncate(),
    );
    let c = r_c2w.transpose() * (glam::DVec3::new(p.x, p.y, p.z) - center);
    let depth = -c.z;
    assert!(
        depth > 0.0,
        "renderer places the point BEHIND the camera (depth {depth:.3}) — \
         the camera-to-world conversion is inverted"
    );
    (W * 0.5 + FOCAL * c.x / depth, H * 0.5 - FOCAL * c.y / depth)
}

#[test]
fn c2w_projects_landmarks_where_vo_saw_them() {
    // Rotated about all three axes and off the origin: near-identity poses
    // hide transpose bugs.
    let r = UnitQuaternion::from_euler_angles(0.21, -0.35, 0.12);
    let center = Vector3::new(1.3, -0.7, 2.1);
    let pose = Se3::new(r, -(r * center));
    let kp = KeyframePose { pts: 0.0, pose };

    // Points defined in CAMERA coords (z > 0 = in front), lifted to world, so
    // cheirality holds by construction.
    for p_cam in [
        Vector3::new(0.0, 0.0, 3.0),
        Vector3::new(0.9, 0.4, 2.5),
        Vector3::new(-1.2, 0.8, 4.0),
        Vector3::new(0.5, -1.1, 1.8),
    ] {
        let p_world = r.inverse() * (p_cam - pose.t);
        let (u, v) = project_vo(&pose, p_world);
        let (px, py) = project_renderer(&kp, p_world);
        assert!(
            (px - u).abs() < 1e-6 && (py - v).abs() < 1e-6,
            "landmark at camera-space {p_cam:?}: VO saw it at ({u:.2}, {v:.2}), \
             renderer puts it at ({px:.2}, {py:.2})"
        );
    }
}

/// The same invariant stated on the rotation alone: `c2w()`'s rotation must
/// equal the COLMAP loader's (`gs-io::colmap`) `R_w2cᵀ · diag(1,−1,−1)`, since
/// both feed the identical `RasterCamera::quat` field.
#[test]
fn c2w_matches_the_colmap_loader_convention() {
    let r = UnitQuaternion::from_euler_angles(0.4, 0.15, -0.6);
    let pose = Se3::new(r, Vector3::new(0.3, 1.1, -0.4));
    let kp = KeyframePose { pts: 0.0, pose };

    let m = r.to_rotation_matrix();
    let m = m.matrix();
    let r_w2c = glam::DMat3::from_cols(
        glam::DVec3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]),
        glam::DVec3::new(m[(0, 1)], m[(1, 1)], m[(2, 1)]),
        glam::DVec3::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]),
    );
    let flip = glam::DMat3::from_diagonal(glam::DVec3::new(1.0, -1.0, -1.0));
    let expect = r_w2c.transpose() * flip;

    let c2w = kp.c2w();
    for (i, want) in [expect.x_axis, expect.y_axis, expect.z_axis].iter().enumerate() {
        let got = [c2w.x_axis, c2w.y_axis, c2w.z_axis][i].truncate();
        assert!(
            (got - *want).length() < 1e-12,
            "c2w column {i}: got {got:?}, expected {want:?}"
        );
    }
}
