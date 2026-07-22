//! Gates for intra-video segment merging (gs_pose::merge): a hard cut in the
//! middle of a clip splits VO into two segments; if both halves film the SAME
//! scene the merge pass must unify them into one gauge (fusing revisit
//! observations), and if they film DIFFERENT content it must refuse — a false
//! merge poisons everything downstream.
//!
//! Fixture mirrors vo_synthetic.rs: two textured planes, analytic GT.

use gs_pose::GrayImage;
use gs_pose::merge::{MergeConfig, merge_segments};
use gs_pose::se3::Se3;
use gs_pose::vo::{Intrinsics, VoConfig, VoFrontEnd};
use nalgebra::{UnitQuaternion, Vector3};

const W: usize = 400;
const H: usize = 300;
const FOCAL: f64 = 350.0;

fn tex(u: f64, v: f64, phase: f64) -> f64 {
    0.5 + 0.16 * (u * 3.1 + phase).sin() * (v * 2.3).cos()
        + 0.14 * (u * 0.9 - v * 1.7).sin()
        + 0.11 * (u * 6.3).cos() * (v * 5.1 + phase).sin()
        + 0.09 * (u * 11.0 + v * 9.0).sin()
}

fn gt_pose(t: f64) -> Se3 {
    let c = Vector3::new(0.8 * t, 0.05 * (t * 2.0).sin(), 0.15 * t);
    let yaw = -0.12 * t;
    let r_cw = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0).inverse();
    Se3::new(r_cw, -(r_cw * c))
}

fn render_with(pose: &Se3, phase: f64) -> GrayImage {
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
            let mut val = 0.35;
            if d_world[2].abs() > 1e-9 {
                let t_near = (4.0 - cam_center[2]) / d_world[2];
                if t_near > 0.0 {
                    let p = cam_center + d_world * t_near;
                    if p[0].abs() < 2.2 && p[1].abs() < 1.6 {
                        img.data[py * W + px] =
                            tex(p[0] + 3.0 * phase, p[1] - 2.0 * phase, phase)
                                .clamp(0.0, 1.0) as f32;
                        continue;
                    }
                }
                let t_far = (9.0 - cam_center[2]) / d_world[2];
                if t_far > 0.0 {
                    let p = cam_center + d_world * t_far;
                    val = tex(
                        p[0] * 0.6 + 3.0 * phase,
                        p[1] * 0.6 - 2.0 * phase,
                        1.7 + phase,
                    );
                }
            }
            img.data[py * W + px] = val.clamp(0.0, 1.0) as f32;
        }
    }
    img
}

fn front_end() -> VoFrontEnd {
    VoFrontEnd::new(VoConfig {
        intrinsics: Intrinsics {
            focal: FOCAL,
            cx: W as f64 / 2.0,
            cy: H as f64 / 2.0,
        },
        kf_flow_frac: 0.012, // 6 px at 400x300
        ..Default::default()
    })
}

fn intr() -> Intrinsics {
    Intrinsics {
        focal: FOCAL,
        cx: W as f64 / 2.0,
        cy: H as f64 / 2.0,
    }
}

fn solved(seg: &gs_pose::VoResult) -> usize {
    seg.keyframe_poses.iter().flatten().count()
}

/// Same scene on both sides of a blur cut, second half REVISITS the first
/// half's trajectory: the two segments must merge into one gauge, and merged
/// landmarks must carry observations from both halves (that fusion is the
/// entire point — revisit information constraining one reconstruction).
#[test]
fn revisit_across_a_cut_merges_into_one_gauge() {
    let n_half = 60;
    let dt = 1.0 / 30.0;
    let mut vo = front_end();
    for k in 0..2 * n_half + 3 {
        let img = if (n_half..n_half + 3).contains(&k) {
            let mut flat = GrayImage::new(W, H);
            flat.data.fill(0.35);
            flat
        } else if k < n_half {
            render_with(&gt_pose(k as f64 * dt), 0.0)
        } else {
            // Replay the SAME camera path — a pure revisit (PTS keeps
            // increasing; only the content repeats).
            render_with(&gt_pose((k - n_half - 3) as f64 * dt), 0.0)
        };
        vo.push_frame(img, k as f64 * dt);
    }
    let segs = vo.solve_segments();
    assert!(
        segs.len() >= 2,
        "fixture must produce >=2 segments, got {}",
        segs.len()
    );

    // Record each original segment's solved keyframe set before the merge.
    let solved_sets: Vec<std::collections::HashSet<usize>> = segs
        .iter()
        .map(|s| {
            s.keyframe_poses
                .iter()
                .enumerate()
                .filter_map(|(k, p)| p.as_ref().map(|_| k))
                .collect()
        })
        .collect();

    let (merged, n_merges) = merge_segments(segs, &intr(), &MergeConfig::default());
    assert!(n_merges >= 1, "no merge happened on a pure revisit");
    assert_eq!(
        merged.len(),
        1,
        "revisit of the same scene must unify into one segment, got {}",
        merged.len()
    );
    let m = &merged[0];
    assert_eq!(
        solved(m),
        solved_sets.iter().map(|s| s.len()).sum::<usize>(),
        "merged gauge must keep every solved keyframe"
    );
    // Revisit fusion: some landmark must be observed from BOTH halves.
    let fused = m.landmark_obs_all.iter().any(|list| {
        solved_sets.iter().all(|set| {
            list.iter().any(|(kf, _)| set.contains(kf))
        })
    });
    assert!(fused, "no landmark carries observations from both segments");
}

/// Different content on each side of the cut (texture phase + coordinate
/// shift): the merge pass must refuse — islands are honest, false merges
/// are catastrophic.
#[test]
fn different_scenes_across_a_cut_do_not_merge() {
    let n_half = 60;
    let dt = 1.0 / 30.0;
    let mut vo = front_end();
    for k in 0..2 * n_half {
        let img = if (n_half..n_half + 3).contains(&k) {
            let mut flat = GrayImage::new(W, H);
            flat.data.fill(0.35);
            flat
        } else {
            let phase = if k < n_half { 0.0 } else { 4.2 };
            render_with(&gt_pose(k as f64 * dt), phase)
        };
        vo.push_frame(img, k as f64 * dt);
    }
    let segs = vo.solve_segments();
    assert!(segs.len() >= 2, "fixture must produce >=2 segments");
    let n_before = segs.len();
    let (merged, n_merges) = merge_segments(segs, &intr(), &MergeConfig::default());
    assert_eq!(
        n_merges, 0,
        "merged {} unrelated segments — false-positive merge",
        n_merges
    );
    assert_eq!(merged.len(), n_before);
}
