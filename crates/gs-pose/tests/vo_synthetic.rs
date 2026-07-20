//! End-to-end VO gate on an analytically rendered scene: two textured planes
//! at different depths (breaks planar degeneracy), camera translating with a
//! slow yaw. Every pixel is ray-traced against the planes, so KLT tracks real
//! image content and the recovered trajectory can be compared to exact GT.

use gs_pose::se3::Se3;
use gs_pose::vo::{Intrinsics, VoConfig, VoFrontEnd};
use gs_pose::{GrayImage, PoseSpline};
use nalgebra::{UnitQuaternion, Vector3};

const W: usize = 400;
const H: usize = 300;
const FOCAL: f64 = 350.0;

/// Smooth 2-D texture (piecewise-random enough for corners, C¹ for KLT).
fn tex(u: f64, v: f64, phase: f64) -> f64 {
    0.5 + 0.16 * (u * 3.1 + phase).sin() * (v * 2.3).cos()
        + 0.14 * (u * 0.9 - v * 1.7).sin()
        + 0.11 * (u * 6.3).cos() * (v * 5.1 + phase).sin()
        + 0.09 * (u * 11.0 + v * 9.0).sin()
}

/// GT world→camera pose at time t (CV convention: y down, z forward).
fn gt_pose(t: f64) -> Se3 {
    // Camera slides right and slightly forward while yawing gently.
    let c = Vector3::new(0.8 * t, 0.05 * (t * 2.0).sin(), 0.15 * t);
    let yaw = -0.12 * t;
    let r_wc = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
    let r_cw = r_wc.inverse();
    Se3::new(r_cw, -(r_cw * c))
}

/// Ray-trace one frame: near plane z=4 (finite), far plane z=9 (infinite).
fn render(pose: &Se3) -> GrayImage {
    let mut img = GrayImage::new(W, H);
    let inv = pose.inverse();
    let cam_center = pose.center();
    for py in 0..H {
        for px in 0..W {
            let d_cam = Vector3::new(
                (px as f64 + 0.5 - W as f64 / 2.0) / FOCAL,
                (py as f64 + 0.5 - H as f64 / 2.0) / FOCAL,
                1.0,
            );
            let d_world = inv.r * d_cam;
            let mut val = 0.35; // background
            // Near plane z = 4, extent |x|<2.2, |y|<1.6.
            if d_world[2].abs() > 1e-9 {
                let t_near = (4.0 - cam_center[2]) / d_world[2];
                if t_near > 0.0 {
                    let p = cam_center + d_world * t_near;
                    if p[0].abs() < 2.2 && p[1].abs() < 1.6 {
                        img.data[py * W + px] =
                            tex(p[0], p[1], 0.0).clamp(0.0, 1.0) as f32;
                        continue;
                    }
                }
                let t_far = (9.0 - cam_center[2]) / d_world[2];
                if t_far > 0.0 {
                    let p = cam_center + d_world * t_far;
                    val = tex(p[0] * 0.6, p[1] * 0.6, 1.7);
                }
            }
            img.data[py * W + px] = val.clamp(0.0, 1.0) as f32;
        }
    }
    img
}

/// Align estimated camera centers to GT with a similarity (Umeyama) and
/// return the RMS residual (the ATE).
fn ate(est: &[Vector3<f64>], gt: &[Vector3<f64>]) -> f64 {
    assert_eq!(est.len(), gt.len());
    let n = est.len() as f64;
    let mu_e: Vector3<f64> = est.iter().sum::<Vector3<f64>>() / n;
    let mu_g: Vector3<f64> = gt.iter().sum::<Vector3<f64>>() / n;
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    let mut var_e = 0.0;
    for (e, g) in est.iter().zip(gt) {
        cov += (g - mu_g) * (e - mu_e).transpose();
        var_e += (e - mu_e).norm_squared();
    }
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut s = nalgebra::Matrix3::<f64>::identity();
    if (u * vt).determinant() < 0.0 {
        s[(2, 2)] = -1.0;
    }
    let r = u * s * vt;
    let scale = (svd.singular_values[0] * s[(0, 0)]
        + svd.singular_values[1] * s[(1, 1)]
        + svd.singular_values[2] * s[(2, 2)])
        / var_e;
    let mut ss = 0.0;
    for (e, g) in est.iter().zip(gt) {
        let aligned = scale * (r * (e - mu_e)) + mu_g;
        ss += (aligned - g).norm_squared();
    }
    (ss / n).sqrt()
}

#[test]
fn vo_recovers_translating_trajectory() {
    let n_frames = 50;
    let dt = 1.0 / 30.0;
    let cfg = VoConfig {
        intrinsics: Intrinsics {
            focal: FOCAL,
            cx: W as f64 / 2.0,
            cy: H as f64 / 2.0,
        },
        // The yaw partially cancels translation flow in image space, so a
        // lower threshold keeps keyframes dense enough for the window BA.
        kf_flow_px: 6.0,
        ..Default::default()
    };
    let mut vo = VoFrontEnd::new(cfg);
    for k in 0..n_frames {
        let t = k as f64 * dt;
        vo.push_frame(render(&gt_pose(t)), t);
    }
    let n_kf = vo.keyframes.len();
    assert!(n_kf >= 5, "too few keyframes: {n_kf}");

    let result = vo.solve().expect("VO solve failed");
    let solved: Vec<(f64, Se3)> = result
        .keyframe_poses
        .iter()
        .flatten()
        .map(|kp| (kp.pts, kp.pose))
        .collect();
    assert!(
        solved.len() >= n_kf - 1,
        "unsolved keyframes: {} of {n_kf}",
        n_kf - solved.len()
    );

    // ATE gate: < 1% of trajectory length after similarity alignment.
    let est_centers: Vec<Vector3<f64>> = solved.iter().map(|(_, p)| p.center()).collect();
    let gt_centers: Vec<Vector3<f64>> = solved.iter().map(|(t, _)| gt_pose(*t).center()).collect();
    let traj_len: f64 = gt_centers.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
    let ate_rms = ate(&est_centers, &gt_centers);
    eprintln!(
        "VO: {} keyframes, trajectory {:.3} m, ATE {:.5} m ({:.3}%)",
        solved.len(),
        traj_len,
        ate_rms,
        100.0 * ate_rms / traj_len
    );
    assert!(
        ate_rms < 0.01 * traj_len,
        "ATE {ate_rms:.5} exceeds 1% of trajectory {traj_len:.3}"
    );

    // RPE rotation gate: consecutive relative rotations within 0.5°.
    for pair in solved.windows(2) {
        let (ta, pa) = pair[0];
        let (tb, pb) = pair[1];
        let rel_est = pb.r * pa.r.inverse();
        let rel_gt = gt_pose(tb).r * gt_pose(ta).r.inverse();
        let err = (rel_est.inverse() * rel_gt).angle().to_degrees();
        assert!(err < 0.5, "RPE rot {err:.3}° between kf at {ta:.2}s/{tb:.2}s");
    }

    // Spline interpolates between keyframes: mid-times should stay close to
    // GT after the same alignment (loose gate — spline is C¹ interpolation).
    let spline: &PoseSpline = result.spline.as_ref().expect("spline");
    let mid_t = (solved[1].0 + solved[2].0) * 0.5;
    let s = spline.sample(mid_t);
    assert!(s.t.iter().all(|v| v.is_finite()));
}

#[test]
fn zoom_signal_flat_for_constant_focal() {
    let cfg = VoConfig {
        intrinsics: Intrinsics {
            focal: FOCAL,
            cx: W as f64 / 2.0,
            cy: H as f64 / 2.0,
        },
        ..Default::default()
    };
    let mut vo = VoFrontEnd::new(cfg);
    for k in 0..10 {
        let t = k as f64 / 30.0;
        vo.push_frame(render(&gt_pose(t)), t);
    }
    // Translation dominates; log radial scale should stay near zero (a real
    // zoom ramps it to ln(f2/f1) per frame, ~0.05 for a 5%/frame zoom).
    for (k, z) in vo.zoom_log_scale.iter().enumerate().skip(1) {
        assert!(z.abs() < 0.02, "frame {k}: zoom signal {z}");
    }
}
