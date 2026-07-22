//! Intra-video segment merging: unify VO segments that observe the same
//! scene back into ONE gauge before any submap is built.
//!
//! Segmentation is a *tracking* reality (whip pans, blur, featureless walls
//! kill KLT), but treating every segment as an independent reconstruction
//! unit discards the most valuable data in a walkthrough — revisits, where
//! the camera pans back and films the same surfaces from another angle.
//! Submaps exist for multi-VIDEO ingestion; within one video we can do much
//! better than islands: same session, same lens, same lighting, and every
//! segment's full landmark/descriptor/observation state is in RAM.
//!
//! Ladder per candidate pair (largest segment absorbs, iterated to fixpoint):
//! descriptor match (cross-checked, scale-searching) → covisibility-vote
//! filter → 3D-3D Sim(3) RANSAC with WIDE scale bounds (each segment's
//! monocular scale is arbitrary) + spread gate → merge + joint bundle
//! adjustment over the union, with a reprojection-RMS acceptance gate that
//! rejects the merge outright if the fused geometry doesn't agree. A false
//! merge poisons everything downstream, so every gate errs toward "island".

use glam::DVec3;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

use crate::descriptor::match_descriptors;
use crate::se3::Se3;
use crate::sim3::{Sim3G, register_point_sets_bounded};
use crate::vo::{Intrinsics, KeyframePose, VoResult};

pub struct MergeConfig {
    /// Descriptor gates (same as the cross-video global matcher).
    pub max_desc_dist: u32,
    pub ratio: f32,
    /// Keep only matches whose (kf_b/8, kf_a/8) covisibility bucket collects
    /// at least this many votes (repetitive-texture stray matches don't
    /// cluster; real co-observation does).
    pub min_votes: usize,
    /// Minimum vote-filtered matches to attempt geometry at all.
    pub min_matches: usize,
    /// Sim(3) RANSAC: inlier threshold as a fraction of the target cloud's
    /// median spread, iterations, minimum inliers, scale bounds. Bounds are
    /// WIDE: independent monocular bootstraps give arbitrary relative scale.
    pub thresh_frac: f64,
    pub ransac_iters: usize,
    pub min_inliers: usize,
    pub scale_bounds: (f64, f64),
    /// Joint-BA acceptance: mean reprojection error of the merged problem
    /// must come in under this many pixels or the merge is rejected.
    pub max_rms_px: f64,
    pub ba_iters: usize,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_desc_dist: 55,
            ratio: 0.85,
            min_votes: 3,
            min_matches: 20,
            thresh_frac: 0.02,
            ransac_iters: 4000,
            min_inliers: 15,
            scale_bounds: (0.02, 50.0),
            max_rms_px: 4.0,
            ba_iters: 12,
        }
    }
}

/// A Sim(3) hypothesis with its fusion pairs and origin label.
type MergeCandidate = (Sim3G, Vec<(usize, usize)>, &'static str);

fn solved(seg: &VoResult) -> usize {
    seg.keyframe_poses.iter().flatten().count()
}

/// Indices of voxel-dedup survivors (0.5% of the median centroid distance —
/// same discipline as the cross-video register path). KLT respawns
/// re-triangulate the same physical corner ~20×; coincident duplicates let a
/// degenerate scale→0 Sim(3) out-vote the truth (measured: every real-clip
/// merge attempt collapsed to scale 0.006–0.05 before this dedup existed).
fn dedup_indices(pts: &[DVec3]) -> Vec<usize> {
    if pts.is_empty() {
        return Vec::new();
    }
    let centroid = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
    let mut d: Vec<f64> = pts.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(f64::total_cmp);
    let voxel = (0.005 * d[d.len() / 2]).max(1e-6);
    let mut seen = std::collections::HashSet::new();
    (0..pts.len())
        .filter(|&i| {
            let p = pts[i];
            seen.insert((
                (p.x / voxel).floor() as i64,
                (p.y / voxel).floor() as i64,
                (p.z / voxel).floor() as i64,
            ))
        })
        .collect()
}

/// Merge co-observing segments of one video. Returns the surviving segments
/// (merged ones absorbed into the largest) and the number of merges applied.
pub fn merge_segments(
    mut segs: Vec<VoResult>,
    intr: &Intrinsics,
    cfg: &MergeConfig,
) -> (Vec<VoResult>, usize) {
    let mut merges = 0;
    loop {
        segs.sort_by_key(|s| std::cmp::Reverse(solved(s)));
        let mut absorbed: Option<(usize, usize, VoResult)> = None;
        'search: for base in 0..segs.len() {
            for other in base + 1..segs.len() {
                if let Some(merged) = try_merge(&segs[base], &segs[other], intr, cfg) {
                    absorbed = Some((base, other, merged));
                    break 'search;
                }
            }
        }
        match absorbed {
            Some((base, other, merged)) => {
                segs[base] = merged;
                segs.remove(other);
                merges += 1;
            }
            None => break,
        }
    }
    (segs, merges)
}

/// Attempt to merge segment `b` into segment `a`'s gauge. Returns the fused,
/// jointly-refined result only if every gate passes.
fn try_merge(
    a: &VoResult,
    b: &VoResult,
    intr: &Intrinsics,
    cfg: &MergeConfig,
) -> Option<VoResult> {
    if b.landmarks.len() < cfg.min_inliers || a.landmarks.len() < cfg.min_inliers {
        return None;
    }
    // Descriptor match B→A over voxel-DEDUPED landmark sets (duplicates make
    // the Sim(3) collapse degenerate), cross-checked, then vote-filtered.
    let keep_b = dedup_indices(&b.landmarks);
    let keep_a = dedup_indices(&a.landmarks);
    let desc_b: Vec<_> = keep_b.iter().map(|&i| b.landmark_desc[i]).collect();
    let desc_a: Vec<_> = keep_a.iter().map(|&i| a.landmark_desc[i]).collect();
    let raw: Vec<(usize, usize)> =
        match_descriptors(&desc_b, &desc_a, cfg.max_desc_dist, cfg.ratio)
            .into_iter()
            .map(|(bi, aj)| (keep_b[bi], keep_a[aj]))
            .collect();
    const BUCKET: usize = 8;
    let mut votes: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for &(bi, aj) in &raw {
        let key = (b.landmark_obs[bi].0 / BUCKET, a.landmark_obs[aj].0 / BUCKET);
        *votes.entry(key).or_default() += 1;
    }
    let matches: Vec<(usize, usize)> = raw
        .iter()
        .copied()
        .filter(|&(bi, aj)| {
            votes[&(b.landmark_obs[bi].0 / BUCKET, a.landmark_obs[aj].0 / BUCKET)]
                >= cfg.min_votes
        })
        .collect();
    log::debug!(
        "segment merge probe: {} raw / {} vote-filtered matches",
        raw.len(),
        matches.len()
    );
    if matches.len() < cfg.min_matches {
        return None;
    }

    // Rung a — Sim(3) B→A over matched 3D pairs (A-side spread threshold).
    let b_pts: Vec<DVec3> = matches.iter().map(|&(bi, _)| b.landmarks[bi]).collect();
    let a_pts: Vec<DVec3> = matches.iter().map(|&(_, aj)| a.landmarks[aj]).collect();
    let centroid = a_pts.iter().copied().sum::<DVec3>() / a_pts.len() as f64;
    let mut d: Vec<f64> = a_pts.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(f64::total_cmp);
    let thresh = (cfg.thresh_frac * d[d.len() / 2]).max(1e-9);

    let mut candidates: Vec<MergeCandidate> = Vec::new();
    if let Some((s, inliers)) = register_point_sets_bounded(
        &b_pts,
        &a_pts,
        cfg.ransac_iters,
        thresh,
        0x536567,
        cfg.scale_bounds,
    ) {
        // Degeneracy gate: inliers must span real structure, not one corner.
        let inl_a: Vec<DVec3> = inliers.iter().map(|&i| a_pts[i]).collect();
        let cen = inl_a.iter().copied().sum::<DVec3>() / inl_a.len() as f64;
        let spread = (inl_a
            .iter()
            .map(|p| (*p - cen).length_squared())
            .sum::<f64>()
            / inl_a.len() as f64)
            .sqrt();
        log::info!(
            "segment merge 3D-3D: {}/{} inliers, scale {:.3}, spread {:.3} vs thresh {:.3}",
            inliers.len(),
            matches.len(),
            s.scale,
            spread,
            thresh
        );
        if inliers.len() >= cfg.min_inliers && spread > 4.0 * thresh {
            let pairs: Vec<(usize, usize)> = inliers.iter().map(|&i| matches[i]).collect();
            candidates.push((s, pairs, "3D-3D"));
        }
    }

    // Rung b — 2D bridge: B landmark depths are the noisiest part of a
    // boundary reconstruction; solving from B's IMAGE observations against
    // A's 3D sidesteps them. Try the B keyframes with the most matches.
    if candidates.is_empty() {
        let mut by_kf: std::collections::HashMap<usize, Vec<(usize, usize)>> =
            std::collections::HashMap::new();
        for &(bi, aj) in &matches {
            by_kf.entry(b.landmark_obs[bi].0).or_default().push((bi, aj));
        }
        let mut kfs: Vec<_> = by_kf.into_iter().filter(|(_, v)| v.len() >= 8).collect();
        kfs.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
        for (kf, pairs_at_kf) in kfs.into_iter().take(3) {
            let Some(kp) = b.keyframe_poses.get(kf).and_then(|p| p.as_ref()) else {
                continue;
            };
            let obs: Vec<crate::sim3::BridgeObs> = pairs_at_kf
                .iter()
                .map(|&(bi, aj)| crate::sim3::BridgeObs {
                    obs: (
                        (b.landmark_obs[bi].1.0 as f64 - intr.cx) / intr.focal,
                        (b.landmark_obs[bi].1.1 as f64 - intr.cy) / intr.focal,
                    ),
                    world: a.landmarks[aj],
                    seg: b.landmarks[bi],
                })
                .collect();
            let q = kp.pose.r;
            let rot = glam::DQuat::from_xyzw(q.i, q.j, q.k, q.w);
            let t = glam::DVec3::new(kp.pose.t[0], kp.pose.t[1], kp.pose.t[2]);
            if let Some((s, consensus)) =
                crate::sim3::sim3_from_bridge(rot, t, &obs, 3.0 / intr.focal, 0xB41D6E)
            {
                // Fusion pairs from 3D agreement under the bridge transform
                // (loose tol — B depths are noisy; joint BA is the arbiter).
                let fusion: Vec<(usize, usize)> = matches
                    .iter()
                    .copied()
                    .filter(|&(bi, aj)| {
                        (s.apply(b.landmarks[bi]) - a.landmarks[aj]).length() < 10.0 * thresh
                    })
                    .collect();
                log::info!(
                    "segment merge bridge @kf{kf}: consensus {consensus}/{}, scale {:.3}, {} fusion pairs",
                    obs.len(),
                    s.scale,
                    fusion.len()
                );
                if consensus >= 8 && fusion.len() >= 10 {
                    candidates.push((s, fusion, "bridge"));
                    break;
                }
            }
        }
    }

    for (s, pairs, label) in candidates {
        // Without fused duplicate landmarks the joint BA decomposes into two
        // independent problems and the RMS gate is vacuous — never accept
        // a merge with too few shared observations.
        if pairs.len() < 10 {
            continue;
        }
        // Fuse into A's gauge; duplicates union their observation lists so
        // revisit constraints actually meet in one problem.
        let mut merged = fuse(a, b, &s, &pairs);
        // Joint BA over the union — the final arbiter. A wrong Sim(3) cannot
        // reprojection-converge across both segments' observations.
        let rms_px = joint_refine(&mut merged, intr, cfg.ba_iters);
        if !rms_px.is_finite() || rms_px > cfg.max_rms_px {
            log::info!(
                "segment merge [{label}] rejected: joint BA rms {rms_px:.2} px > {} px",
                cfg.max_rms_px
            );
            continue;
        }
        log::info!(
            "segment merge ACCEPTED [{label}]: {} + {} kf -> one gauge, joint rms {:.2} px",
            solved(a),
            solved(b),
            rms_px
        );
        return Some(merged);
    }
    None
}

/// Build the fused VoResult: B's poses and landmarks mapped through `s`
/// (B-gauge → A-gauge), landmark arrays concatenated with index remap,
/// matched duplicates unioned into A's entries.
fn fuse(a: &VoResult, b: &VoResult, s: &Sim3G, inlier_pairs: &[(usize, usize)]) -> VoResult {
    // World change x_a = scale·R_s·x_b + t_s. For a world→camera pose
    // (R_b, t_b): R' = R_b·R_sᵀ and center' = s(center_b); the camera frame
    // then sees every point scaled uniformly by `scale`, so normalized image
    // observations are unchanged — reprojection-invariant by construction.
    let r_s = UnitQuaternion::from_quaternion(Quaternion::new(
        s.rot.w, s.rot.x, s.rot.y, s.rot.z,
    ));
    let n_kf = a.keyframe_poses.len().max(b.keyframe_poses.len());
    let mut keyframe_poses: Vec<Option<KeyframePose>> = Vec::with_capacity(n_kf);
    for k in 0..n_kf {
        let from_a = a.keyframe_poses.get(k).and_then(|p| p.as_ref());
        let from_b = b.keyframe_poses.get(k).and_then(|p| p.as_ref());
        keyframe_poses.push(match (from_a, from_b) {
            // Segments partition the solved keyframes; prefer A on overlap.
            (Some(kp), _) => Some(KeyframePose {
                pts: kp.pts,
                pose: kp.pose,
            }),
            (None, Some(kp)) => {
                let c_b = kp.pose.center();
                let c_a = s.apply(DVec3::new(c_b[0], c_b[1], c_b[2]));
                let r = kp.pose.r * r_s.inverse();
                let t = -(r * Vector3::new(c_a.x, c_a.y, c_a.z));
                Some(KeyframePose {
                    pts: kp.pts,
                    pose: Se3::new(r, t),
                })
            }
            (None, None) => None,
        });
    }

    let mut landmarks = a.landmarks.clone();
    let mut landmark_obs = a.landmark_obs.clone();
    let mut landmark_obs_all = a.landmark_obs_all.clone();
    let mut landmark_desc = a.landmark_desc.clone();

    let dup_of: std::collections::HashMap<usize, usize> =
        inlier_pairs.iter().copied().collect();
    for (bi, lm) in b.landmarks.iter().enumerate() {
        if let Some(&aj) = dup_of.get(&bi) {
            // Same physical point: union the observation lists so the joint
            // BA links both segments' cameras through one landmark.
            landmark_obs_all[aj].extend(b.landmark_obs_all[bi].iter().copied());
        } else {
            landmarks.push(s.apply(*lm));
            landmark_obs.push(b.landmark_obs[bi]);
            landmark_obs_all.push(b.landmark_obs_all[bi].clone());
            landmark_desc.push(b.landmark_desc[bi]);
        }
    }

    VoResult {
        keyframe_poses,
        landmarks,
        landmark_obs,
        landmark_obs_all,
        landmark_desc,
        spline: None, // per-segment splines don't survive a gauge merge
        anchor: a.anchor,
    }
}

/// Joint bundle adjustment over the fused problem; writes refined poses and
/// landmarks back in place and returns the mean reprojection error (px).
fn joint_refine(merged: &mut VoResult, intr: &Intrinsics, ba_iters: usize) -> f64 {
    let kfs: Vec<usize> = merged
        .keyframe_poses
        .iter()
        .enumerate()
        .filter_map(|(k, p)| p.as_ref().map(|_| k))
        .collect();
    let cam_of_kf: std::collections::HashMap<usize, usize> =
        kfs.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut poses: Vec<(glam::DQuat, glam::DVec3)> = kfs
        .iter()
        .map(|&k| {
            let p = &merged.keyframe_poses[k].as_ref().unwrap().pose;
            (
                glam::DQuat::from_xyzw(p.r.i, p.r.j, p.r.k, p.r.w),
                glam::DVec3::new(p.t[0], p.t[1], p.t[2]),
            )
        })
        .collect();
    let mut lms = merged.landmarks.clone();
    let mut obs: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (li, list) in merged.landmark_obs_all.iter().enumerate() {
        for &(kf, (px, py)) in list {
            let Some(&cam) = cam_of_kf.get(&kf) else { continue };
            obs.push((
                cam,
                li,
                (px as f64 - intr.cx) / intr.focal,
                (py as f64 - intr.cy) / intr.focal,
            ));
        }
    }
    if obs.is_empty() {
        return f64::INFINITY;
    }
    crate::ba::refine_submap_glam(&mut poses, &mut lms, &obs, ba_iters);

    // Mean reprojection error in pixels over every observation.
    let mut se = 0.0f64;
    let mut n = 0usize;
    for &(cam, li, xn, yn) in &obs {
        let (q, t) = poses[cam];
        let c = q * lms[li] + t;
        if c.z <= 1e-9 {
            se += 100.0; // behind the camera counts as a gross error
            n += 1;
            continue;
        }
        let dx = (c.x / c.z - xn) * intr.focal;
        let dy = (c.y / c.z - yn) * intr.focal;
        se += dx * dx + dy * dy;
        n += 1;
    }
    let rms = (se / n as f64).sqrt();

    // Write the refined state back.
    for (i, &k) in kfs.iter().enumerate() {
        let (q, t) = poses[i];
        let kp = merged.keyframe_poses[k].as_mut().unwrap();
        kp.pose = Se3::new(
            UnitQuaternion::from_quaternion(Quaternion::new(q.w, q.x, q.y, q.z)),
            Vector3::new(t.x, t.y, t.z),
        );
    }
    merged.landmarks = lms;
    rms
}
