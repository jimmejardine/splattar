//! Pairwise image registration primitive: given TWO keyframe images (from
//! different videos or segments), detect fresh corners, match descriptors
//! image-to-image, and verify with essential-matrix RANSAC. The verified
//! correspondences carry far higher precision than landmark-DB retrieval —
//! repeated indoor texture defeats one-shot global matching, but epipolar
//! geometry between a specific image pair prunes it decisively (measured:
//! landmark retrieval never exceeded ~5/25 region-consistent matches; see
//! RESULTS.md M8 addenda).

use crate::descriptor::{describe_multi, match_descriptors, MultiDescriptor};
use crate::detect::{DetectConfig, detect};
use crate::fivepoint::ransac_essential_5pt;
use crate::image::{GrayImage, Pyramid};
use crate::twoview::Match2;

/// A verified correspondence, endpoints in FULL-RESOLUTION pixel coords,
/// with each endpoint's descriptor (lets callers snap to landmark databases
/// by appearance rather than distance alone).
#[derive(Debug, Clone, Copy)]
pub struct PairMatch {
    pub a_px: (f32, f32),
    pub b_px: (f32, f32),
    pub a_desc: MultiDescriptor,
    pub b_desc: MultiDescriptor,
}

pub struct PairwiseConfig {
    /// Scale of the input images relative to full resolution (thumbnails
    /// are pyramid level 1 → 0.5).
    pub image_scale: f32,
    /// Full-resolution intrinsics.
    pub focal: f64,
    pub cx: f64,
    pub cy: f64,
    /// Descriptor gates (looser than DB retrieval — RANSAC cleans up).
    pub max_dist: u32,
    pub ratio: f32,
    /// Epipolar Sampson threshold in FULL-RES pixels.
    pub epi_px: f64,
    /// Minimum epipolar inliers to report success.
    pub min_inliers: usize,
    /// Essential RANSAC iterations. Eight-point samples need MANY draws at
    /// realistic inlier fractions (0.4⁸ ≈ 7e-4 per draw) — go big, the
    /// per-iteration cost at ≤100 matches is trivial.
    pub ransac_iters: usize,
}

impl Default for PairwiseConfig {
    fn default() -> Self {
        Self {
            image_scale: 0.5,
            focal: 722.0,
            cx: 239.0,
            cy: 425.0,
            // Loose matching keeps depth DIVERSITY (tight ratio gates skew
            // toward dominant-plane texture and degenerate the eight-point
            // solve); the iteration count carries the inlier-fraction cost.
            max_dist: 64,
            ratio: 0.85,
            // Half-res corner localization + rolling shutter between takes.
            epi_px: 4.0,
            min_inliers: 10,
            ransac_iters: 5000,
        }
    }
}

/// Detect + describe corners of one image (shared by both sides).
fn features(img: &GrayImage) -> (Vec<(f32, f32)>, Vec<MultiDescriptor>) {
    // Denser than tracking detection: pairwise matching lives on having
    // hundreds of candidates per image (thumbnails are small).
    let det = DetectConfig {
        cell: 10,
        min_score: 5e-5,
        ..Default::default()
    };
    let corners = detect(img, &det, &[]);
    let pyr = Pyramid::build(
        GrayImage {
            width: img.width,
            height: img.height,
            data: img.data.clone(),
        },
        3,
    );
    let pts: Vec<(f32, f32)> = corners.iter().map(|c| (c.x, c.y)).collect();
    use rayon::prelude::*;
    let descs: Vec<MultiDescriptor> = pts
        .par_iter()
        .map(|&(x, y)| describe_multi(&pyr, x, y))
        .collect();
    (pts, descs)
}

/// Relative pose of camera B in camera A's frame recovered from the
/// verified essential matrix: x_b = R·x_a + λ·t̂ with ‖t̂‖ = 1, λ unknown
/// (monocular). glam at the boundary per CLAUDE.md.
#[derive(Debug, Clone, Copy)]
pub struct RelativePose {
    pub rot: glam::DQuat,
    pub tdir: glam::DVec3,
}

/// Match two keyframe images and epipolar-verify. Returns the verified
/// correspondences (full-res pixel coords), or an empty vec when the pair
/// doesn't support a fundamental relation (no real covisibility).
pub fn match_image_pair(a: &GrayImage, b: &GrayImage, cfg: &PairwiseConfig) -> Vec<PairMatch> {
    match_image_pair_with_pose(a, b, cfg).0
}

/// [`match_image_pair`] that also decomposes the verified essential matrix
/// into the relative camera pose — the snap-free route to Sim(3): relative
/// poses between known-gauge cameras constrain the transform directly,
/// without ever trusting landmark 3D.
pub fn match_image_pair_with_pose(
    a: &GrayImage,
    b: &GrayImage,
    cfg: &PairwiseConfig,
) -> (Vec<PairMatch>, Option<RelativePose>) {
    let (pa, da) = features(a);
    let (pb, db) = features(b);
    if pa.len() < 20 || pb.len() < 20 {
        log::debug!("pairwise: too few corners ({} / {})", pa.len(), pb.len());
        return (Vec::new(), None);
    }
    let pairs = match_descriptors(&da, &db, cfg.max_dist, cfg.ratio);
    log::debug!(
        "pairwise: {} / {} corners, {} raw matches",
        pa.len(),
        pb.len(),
        pairs.len()
    );
    if pairs.len() < cfg.min_inliers {
        return (Vec::new(), None);
    }
    // Normalized image coordinates from thumbnail pixels: full-res px =
    // thumb px / image_scale.
    let s = cfg.image_scale as f64;
    let norm = |p: (f32, f32)| {
        (
            (p.0 as f64 / s - cfg.cx) / cfg.focal,
            (p.1 as f64 / s - cfg.cy) / cfg.focal,
        )
    };
    let matches: Vec<Match2> = pairs
        .iter()
        .map(|&(i, j)| Match2 {
            a: norm(pa[i]),
            b: norm(pb[j]),
        })
        .collect();
    // Five-point minimal samples: the whole reason this stage works at
    // cross-take precision (~20% inliers ⇒ 0.2⁵ per draw, vs 0.2⁸ for the
    // eight-point solver — 1000× fewer usable draws).
    let Some(rr) =
        ransac_essential_5pt(&matches, cfg.ransac_iters, cfg.epi_px / cfg.focal, 0xEB1B01A)
    else {
        log::debug!("pairwise: essential RANSAC found no model");
        return (Vec::new(), None);
    };
    log::debug!("pairwise: {} epipolar inliers", rr.inliers.len());
    if rr.inliers.len() < cfg.min_inliers {
        return (Vec::new(), None);
    }
    let verified: Vec<PairMatch> = rr
        .inliers
        .iter()
        .map(|&m| {
            let (i, j) = pairs[m];
            PairMatch {
                a_px: (pa[i].0 / cfg.image_scale, pa[i].1 / cfg.image_scale),
                b_px: (pb[j].0 / cfg.image_scale, pb[j].1 / cfg.image_scale),
                a_desc: da[i],
                b_desc: db[j],
            }
        })
        .collect();
    // Decompose E into the cheirality-consistent relative pose.
    let rel = crate::twoview::recover_pose_with(&rr.e, &matches, &rr.inliers, 0.6).map(|boot| {
        let q = boot.pose_ba.r.quaternion();
        RelativePose {
            rot: glam::DQuat::from_xyzw(q.i, q.j, q.k, q.w),
            tdir: glam::DVec3::new(
                boot.pose_ba.t[0],
                boot.pose_ba.t[1],
                boot.pose_ba.t[2],
            )
            .normalize_or_zero(),
        }
    });
    (verified, rel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::se3::Se3;
    use nalgebra::{UnitQuaternion, Vector3};

    const W: usize = 200; // half-res thumbnail size
    const H: usize = 150;
    const FOCAL: f64 = 350.0; // full-res focal (image_scale 0.5)

    fn tex_p(u: f64, v: f64, phase: f64) -> f64 {
        0.5 + 0.16 * (u * 3.1 + phase).sin() * (v * 2.3 - phase).cos()
            + 0.14 * (u * 0.9 - v * 1.7 + 2.0 * phase).sin()
            + 0.11 * (u * 6.3 - phase).cos() * (v * 5.1).sin()
            + 0.09 * (u * 11.0 + v * 9.0 + phase).sin()
    }

    /// Ray-trace a two-plane scene at thumbnail resolution.
    fn render(pose: &Se3) -> GrayImage {
        render_phase(pose, 0.0)
    }

    fn render_phase(pose: &Se3, phase: f64) -> GrayImage {
        let mut img = GrayImage::new(W, H);
        let inv = pose.inverse();
        let cam = pose.center();
        let scale = 0.5f64;
        for py in 0..H {
            for px in 0..W {
                let d_cam = Vector3::new(
                    (px as f64 / scale + 0.5 - 2.0 * W as f64 / 2.0) / FOCAL,
                    (py as f64 / scale + 0.5 - 2.0 * H as f64 / 2.0) / FOCAL,
                    1.0,
                );
                let d = inv.r * d_cam;
                let mut val = 0.35;
                if d[2].abs() > 1e-9 {
                    let t_near = (4.0 - cam[2]) / d[2];
                    if t_near > 0.0 {
                        let p = cam + d * t_near;
                        // Near plane covers ~half the view — an all-planar
                        // image is degenerate for the eight-point step by
                        // design (documented in twoview.rs).
                        if p[0].abs() < 1.3 && p[1].abs() < 0.9 {
                            img.data[py * W + px] =
                                tex_p(p[0], p[1], phase).clamp(0.0, 1.0) as f32;
                            continue;
                        }
                    }
                    let t_far = (9.0 - cam[2]) / d[2];
                    if t_far > 0.0 {
                        let p = cam + d * t_far;
                        val = tex_p(p[0] * 0.6 + 1.3, p[1] * 0.6 - 0.7, phase);
                    }
                }
                img.data[py * W + px] = val.clamp(0.0, 1.0) as f32;
            }
        }
        img
    }

    #[test]
    fn verifies_true_pair_and_rejects_unrelated() {
        let a = render(&Se3::identity());
        // A genuinely different viewpoint: sideways + a bit of yaw.
        let b_pose = Se3::new(
            UnitQuaternion::from_euler_angles(0.0, -0.08, 0.02).inverse(),
            Vector3::new(-0.6, 0.05, 0.1),
        );
        let b = render(&b_pose);
        let cfg = PairwiseConfig {
            image_scale: 0.5,
            focal: FOCAL,
            cx: W as f64,
            cy: H as f64,
            ..Default::default()
        };
        let (pa, da) = features(&a);
        let (pb, db) = features(&b);
        let raw = match_descriptors(&da, &db, cfg.max_dist, cfg.ratio);
        eprintln!(
            "diag: {} vs {} corners, {} raw matches",
            pa.len(),
            pb.len(),
            raw.len()
        );
        let verified = match_image_pair(&a, &b, &cfg);
        eprintln!("diag: {} verified", verified.len());
        assert!(
            verified.len() >= 20,
            "true pair: only {} verified matches",
            verified.len()
        );

        // Genuinely unrelated content (decorrelated texture) must not verify.
        let c = render_phase(&Se3::identity(), 4.7);
        let bogus = match_image_pair(&a, &c, &cfg);
        assert!(
            bogus.len() < 15,
            "unrelated pair produced {} 'verified' matches",
            bogus.len()
        );
    }
}
