//! Visual-odometry front-end orchestration.
//!
//! Two phases, per the PLAN architecture:
//! - **Causal pass** (`push_frame`): decode-order KLT tracking with
//!   constant-velocity motion prediction, per-frame sharpness + radial-flow
//!   zoom signal, keyframe promotion by flow/survival.
//! - **Anchor-out solve** (`solve`): bootstrap at the best-conditioned
//!   keyframe pair near the segment interior (never at a boundary), then grow
//!   the reconstruction in both temporal directions with PnP + triangulation
//!   and sliding-window BA. Output poses are world→camera in the anchor gauge
//!   (monocular: scale is arbitrary per segment, per CLAUDE.md).
//!
//! Image convention: x right, y **down** (raster order), camera looks +z —
//! plain CV pinhole. Conversion to the renderer's y-up/−z convention happens
//! at the glam boundary (`KeyframePose::c2w`).

use glam::{DMat3, DMat4, DVec3};
use nalgebra::Vector3;

use crate::ba::{BaConfig, BaProblem, Obs, optimize};
use crate::detect::{DetectConfig, detect};
use crate::image::{GrayImage, Pyramid};
use crate::klt::{KltConfig, track_point_fb};
use crate::pnp::{PnpConfig, solve_pnp};
use crate::se3::Se3;
use crate::spline::PoseSpline;
use crate::triangulate::{parallax_angle, triangulate_n};
use crate::twoview::{Match2, ransac_essential, recover_pose};

#[derive(Debug, Clone, Copy)]
pub struct Intrinsics {
    pub focal: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Intrinsics {
    #[inline]
    fn norm(&self, p: (f32, f32)) -> (f64, f64) {
        (
            (p.0 as f64 - self.cx) / self.focal,
            (p.1 as f64 - self.cy) / self.focal,
        )
    }
}

pub struct VoConfig {
    pub intrinsics: Intrinsics,
    pub pyramid_levels: usize,
    pub klt: KltConfig,
    pub detect: DetectConfig,
    /// Median flow (px) since the last keyframe that forces a new keyframe.
    pub kf_flow_px: f32,
    /// Track-survival ratio (vs the last keyframe) that forces a keyframe.
    pub kf_survival: f32,
    /// Respawn detection when live tracks drop below this.
    pub min_tracks: usize,
    /// RANSAC Sampson threshold in pixels (divided by focal internally).
    pub ransac_px: f64,
    pub ransac_iters: usize,
    /// Bootstrap acceptance: minimum median triangulation parallax (radians).
    pub boot_min_parallax: f64,
    /// Bootstrap acceptance: minimum inlier count.
    pub boot_min_inliers: usize,
    /// Bootstrap pair selection: extend the pair (ka, kb) forward until the
    /// RMS residual of a global affine fit reaches this many pixels. Raw flow
    /// is useless here — panning creates huge flow with zero baseline; the
    /// affine residual only responds to parallax (depth structure).
    pub boot_parallax_px: f64,
    /// Sliding-window size for local BA during expansion.
    pub ba_window: usize,
    /// Keyframes this close to the segment ends are not anchor candidates
    /// (fraction of the keyframe count).
    pub anchor_margin: f64,
}

impl Default for VoConfig {
    fn default() -> Self {
        Self {
            intrinsics: Intrinsics {
                focal: 500.0,
                cx: 320.0,
                cy: 240.0,
            },
            pyramid_levels: 4,
            klt: KltConfig::default(),
            detect: DetectConfig::default(),
            kf_flow_px: 15.0,
            // Per-frame attrition compounds between keyframes (~0.9^k on
            // handheld footage); 0.5 keeps promotion driven by flow, not churn.
            kf_survival: 0.5,
            min_tracks: 150,
            ransac_px: 1.5,
            ransac_iters: 300,
            boot_min_parallax: 1.0f64.to_radians(),
            boot_min_inliers: 30,
            boot_parallax_px: 4.0,
            ba_window: 6,
            anchor_margin: 0.1,
        }
    }
}

struct Track {
    /// Current position (finest level px), None once lost.
    cur: Option<(f32, f32)>,
    /// Position at the most recent keyframe (for flow statistics).
    at_last_kf: Option<(f32, f32)>,
    /// Observations at keyframes: (kf index, position).
    obs: Vec<(usize, (f32, f32))>,
    /// Binary patch descriptor per observation (for cross-video matching).
    obs_desc: Vec<crate::descriptor::Descriptor>,
    /// Landmark index once triangulated.
    landmark: Option<usize>,
}

pub struct Keyframe {
    pub frame_idx: usize,
    pub pts: f64,
    pub sharpness: f32,
}

pub struct KeyframePose {
    pub pts: f64,
    /// world→camera (anchor gauge).
    pub pose: Se3,
}

impl KeyframePose {
    /// Camera-to-world matrix in the renderer convention (y up, looks −z):
    /// columns are the camera axes with the CV y-down/z-forward flipped.
    pub fn c2w(&self) -> DMat4 {
        let r = self.pose.r.to_rotation_matrix();
        let m = r.matrix();
        // R_c2w(cv) = Rᵀ; renderer basis = cv basis · diag(1,−1,−1).
        let rc2w = DMat3::from_cols(
            DVec3::new(m[(0, 0)], m[(0, 1)], m[(0, 2)]),
            DVec3::new(-m[(1, 0)], -m[(1, 1)], -m[(1, 2)]),
            DVec3::new(-m[(2, 0)], -m[(2, 1)], -m[(2, 2)]),
        );
        let c = self.pose.center();
        let mut out = DMat4::from_mat3(rc2w.transpose());
        out.w_axis = glam::DVec4::new(c[0], c[1], c[2], 1.0);
        out
    }
}

pub struct VoResult {
    pub keyframe_poses: Vec<Option<KeyframePose>>,
    pub landmarks: Vec<DVec3>,
    /// One reference observation per landmark: (keyframe index, pixel
    /// position). Lets callers sample appearance for initialization.
    pub landmark_obs: Vec<(usize, (f32, f32))>,
    /// Binary descriptor at the reference observation (cross-video matching).
    pub landmark_desc: Vec<crate::descriptor::Descriptor>,
    pub spline: Option<PoseSpline>,
    /// Index of the anchor keyframe (gauge origin).
    pub anchor: usize,
}

pub struct VoFrontEnd {
    cfg: VoConfig,
    prev: Option<Pyramid>,
    prev_median_flow: (f32, f32),
    tracks: Vec<Track>,
    pub keyframes: Vec<Keyframe>,
    n_frames: usize,
    frame_pts: Vec<f64>,
    /// Per-frame log radial-flow scale (zoom signal; ≈0 when focal constant).
    pub zoom_log_scale: Vec<f64>,
}

impl VoFrontEnd {
    pub fn new(cfg: VoConfig) -> Self {
        Self {
            cfg,
            prev: None,
            prev_median_flow: (0.0, 0.0),
            tracks: Vec::new(),
            keyframes: Vec::new(),
            n_frames: 0,
            frame_pts: Vec::new(),
            zoom_log_scale: Vec::new(),
        }
    }

    /// Causal pass: feed frames in decode order with their PTS.
    pub fn push_frame(&mut self, gray: GrayImage, pts: f64) {
        let sharpness = gradient_energy(&gray);
        let pyr = Pyramid::build(gray, self.cfg.pyramid_levels);
        let frame_idx = self.n_frames;
        self.n_frames += 1;
        self.frame_pts.push(pts);

        let mut zoom = 0.0f64;
        if let Some(prev) = &self.prev {
            // Track with constant-velocity prediction (median of last flow).
            let (mfx, mfy) = self.prev_median_flow;
            let mut flows: Vec<(f32, f32)> = Vec::new();
            let mut radial_num = 0.0f64;
            let mut radial_den = 0.0f64;
            let (cx, cy) = (self.cfg.intrinsics.cx as f32, self.cfg.intrinsics.cy as f32);
            for tr in &mut self.tracks {
                let Some(p) = tr.cur else { continue };
                let guess = (p.0 + mfx, p.1 + mfy);
                match track_point_fb(prev, &pyr, p, guess, &self.cfg.klt) {
                    Some(q) => {
                        flows.push((q.0 - p.0, q.1 - p.1));
                        // Radial components about the principal point.
                        let r0 = ((p.0 - cx) as f64, (p.1 - cy) as f64);
                        let r1 = ((q.0 - cx) as f64, (q.1 - cy) as f64);
                        radial_num += r1.0 * r0.0 + r1.1 * r0.1;
                        radial_den += r0.0 * r0.0 + r0.1 * r0.1;
                        tr.cur = Some(q);
                    }
                    None => tr.cur = None,
                }
            }
            if radial_den > 1.0 {
                zoom = (radial_num / radial_den).max(1e-6).ln();
            }
            if !flows.is_empty() {
                let mut xs: Vec<f32> = flows.iter().map(|f| f.0).collect();
                let mut ys: Vec<f32> = flows.iter().map(|f| f.1).collect();
                xs.sort_by(f32::total_cmp);
                ys.sort_by(f32::total_cmp);
                self.prev_median_flow = (xs[xs.len() / 2], ys[ys.len() / 2]);
            }
        }
        self.zoom_log_scale.push(zoom);

        // Keyframe decision.
        let is_first = self.keyframes.is_empty();
        let (flow_med, survival) = self.kf_statistics();
        let promote = is_first
            || flow_med >= self.cfg.kf_flow_px
            || survival < self.cfg.kf_survival
            || self.live_count() < self.cfg.min_tracks;
        if promote {
            let kf_idx = self.keyframes.len();
            self.keyframes.push(Keyframe {
                frame_idx,
                pts,
                sharpness,
            });
            let smooth = &pyr.levels[1.min(pyr.levels.len() - 1)];
            let lvl_scale = if pyr.levels.len() > 1 { 0.5 } else { 1.0 };
            for tr in &mut self.tracks {
                if let Some(p) = tr.cur {
                    tr.obs.push((kf_idx, p));
                    tr.obs_desc.push(crate::descriptor::describe(
                        smooth,
                        p.0 * lvl_scale,
                        p.1 * lvl_scale,
                    ));
                }
                // Dead tracks must drop out of the survival statistics, or
                // the ratio decays monotonically and every frame becomes a
                // keyframe (seen on real footage).
                tr.at_last_kf = tr.cur;
            }
            // Drop dead tracks that never got a second observation — they
            // can't contribute geometry and bloat the causal pass.
            self.tracks
                .retain(|t| t.cur.is_some() || t.obs.len() >= 2);
            // Spawn fresh corners away from live tracks.
            let existing: Vec<(f32, f32)> =
                self.tracks.iter().filter_map(|t| t.cur).collect();
            let corners = detect(&pyr.levels[0], &self.cfg.detect, &existing);
            for c in corners {
                self.tracks.push(Track {
                    cur: Some((c.x, c.y)),
                    at_last_kf: Some((c.x, c.y)),
                    obs: vec![(kf_idx, (c.x, c.y))],
                    obs_desc: vec![crate::descriptor::describe(
                        smooth,
                        c.x * lvl_scale,
                        c.y * lvl_scale,
                    )],
                    landmark: None,
                });
            }
        }
        self.prev = Some(pyr);
    }

    fn live_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.cur.is_some()).count()
    }

    /// (median flow since last KF, survival ratio vs last KF).
    fn kf_statistics(&self) -> (f32, f32) {
        let mut disp: Vec<f32> = Vec::new();
        let mut alive_since_kf = 0usize;
        let mut seen_at_kf = 0usize;
        for tr in &self.tracks {
            if let Some(kf_pos) = tr.at_last_kf {
                seen_at_kf += 1;
                if let Some(cur) = tr.cur {
                    alive_since_kf += 1;
                    disp.push(((cur.0 - kf_pos.0).powi(2) + (cur.1 - kf_pos.1).powi(2)).sqrt());
                }
            }
        }
        if seen_at_kf == 0 {
            return (0.0, 1.0);
        }
        disp.sort_by(f32::total_cmp);
        let med = if disp.is_empty() { 0.0 } else { disp[disp.len() / 2] };
        (med, alive_since_kf as f32 / seen_at_kf as f32)
    }

    /// Shared-track matches between two keyframes (normalized coords), with
    /// the track indices that produced them.
    fn kf_matches(&self, ka: usize, kb: usize) -> (Vec<Match2>, Vec<usize>) {
        let mut matches = Vec::new();
        let mut track_ids = Vec::new();
        for (ti, tr) in self.tracks.iter().enumerate() {
            let pa = tr.obs.iter().find(|o| o.0 == ka).map(|o| o.1);
            let pb = tr.obs.iter().find(|o| o.0 == kb).map(|o| o.1);
            if let (Some(pa), Some(pb)) = (pa, pb) {
                matches.push(Match2 {
                    a: self.cfg.intrinsics.norm(pa),
                    b: self.cfg.intrinsics.norm(pb),
                });
                track_ids.push(ti);
            }
        }
        (matches, track_ids)
    }

    /// Anchor-out solve over the collected segment.
    pub fn solve(&mut self) -> Option<VoResult> {
        let n_kf = self.keyframes.len();
        if n_kf < 2 {
            return None;
        }
        let thresh = self.cfg.ransac_px / self.cfg.intrinsics.focal;

        // Candidate anchor pairs: from each start keyframe, extend forward
        // until the shared tracks carry enough flow for a well-conditioned
        // two-view solve (adjacent keyframes rarely have the baseline).
        let margin = ((n_kf as f64 * self.cfg.anchor_margin) as usize).min(n_kf / 4);
        let lo = margin;
        let hi = (n_kf - 1 - margin).max(lo + 1).min(n_kf - 1);
        // Sample candidate starts — scanning every keyframe is O(n²) in
        // track lookups and adds nothing once the pool is a few dozen.
        let step = ((hi - lo) / 48).max(1);
        let mut candidates: Vec<(f64, usize, usize)> = (lo..hi)
            .step_by(step)
            .filter_map(|ka| {
                let mut chosen: Option<(usize, usize, f64)> = None;
                for kb in ka + 1..n_kf {
                    let (m, _) = self.kf_matches(ka, kb);
                    if m.len() < self.cfg.boot_min_inliers {
                        break;
                    }
                    let res = affine_residual_px(&m, self.cfg.intrinsics.focal);
                    if chosen.as_ref().is_none_or(|c| res > c.2) {
                        chosen = Some((kb, m.len(), res));
                    }
                    if res >= self.cfg.boot_parallax_px {
                        break;
                    }
                }
                chosen.map(|(kb, n_shared, res)| {
                    let sharp = self.keyframes[ka]
                        .sharpness
                        .min(self.keyframes[kb].sharpness) as f64;
                    (n_shared as f64 * res * sharp, ka, kb)
                })
            })
            .collect();
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Bootstrap at the best-conditioned pair that passes the gates.
        let mut poses: Vec<Option<Se3>> = vec![None; n_kf];
        let mut landmarks: Vec<Vector3<f64>> = Vec::new();
        let mut anchor = usize::MAX;
        for &(score, ka, kb) in candidates.iter().take(12) {
            let (matches, track_ids) = self.kf_matches(ka, kb);
            if matches.len() < self.cfg.boot_min_inliers {
                log::debug!("boot ({ka},{kb}) score {score:.0}: too few matches {}", matches.len());
                continue;
            }
            let Some(rr) =
                ransac_essential(&matches, self.cfg.ransac_iters, thresh, 0xC0FFEE)
            else {
                log::debug!("boot ({ka},{kb}): RANSAC found no model");
                continue;
            };
            if rr.inliers.len() < self.cfg.boot_min_inliers
                || rr.inliers.len() * 2 < matches.len()
            {
                log::debug!(
                    "boot ({ka},{kb}): weak consensus {}/{}",
                    rr.inliers.len(),
                    matches.len()
                );
                continue;
            }
            let Some(boot) = recover_pose(&rr.e, &matches, &rr.inliers) else {
                log::debug!("boot ({ka},{kb}): cheirality ambiguous");
                continue;
            };
            // Parallax gate: median angle between rays of triangulated points.
            let ident = Se3::identity();
            let mut angs: Vec<f64> = boot
                .points
                .iter()
                .map(|(_, p)| parallax_angle(&ident, &boot.pose_ba, p))
                .collect();
            angs.sort_by(f64::total_cmp);
            if angs.is_empty() || angs[angs.len() / 2] < self.cfg.boot_min_parallax {
                log::debug!(
                    "boot ({ka},{kb}): median parallax {:.2}° below gate",
                    angs.get(angs.len() / 2).copied().unwrap_or(0.0).to_degrees()
                );
                continue;
            }
            log::info!(
                "bootstrap at keyframes ({ka},{kb}): {} inliers, median parallax {:.2}°",
                rr.inliers.len(),
                angs[angs.len() / 2].to_degrees()
            );
            poses[ka] = Some(Se3::identity());
            poses[kb] = Some(boot.pose_ba);
            for (mi, p) in &boot.points {
                let ti = track_ids[*mi];
                self.tracks[ti].landmark = Some(landmarks.len());
                landmarks.push(*p);
            }
            anchor = ka;
            break;
        }
        if anchor == usize::MAX {
            return None;
        }

        // Anchor-out expansion: everything outside the (solved) anchor pair,
        // nearest-to-anchor first, alternating temporal directions.
        let mut order: Vec<usize> = (0..n_kf).filter(|&k| poses[k].is_none()).collect();
        order.sort_by_key(|&k| k.abs_diff(anchor));

        for k in order {
            // PnP from tracks with landmarks observed at keyframe k.
            let mut pts3 = Vec::new();
            let mut obs2 = Vec::new();
            for tr in &self.tracks {
                let Some(l) = tr.landmark else { continue };
                if let Some(&(_, p)) = tr.obs.iter().find(|o| o.0 == k) {
                    pts3.push(landmarks[l]);
                    obs2.push(self.cfg.intrinsics.norm(p));
                }
            }
            // Prior: nearest solved pose.
            let prior = nearest_pose(&poses, k)?;
            let Some(res) = solve_pnp(&pts3, &obs2, &prior, &PnpConfig::default()) else {
                // Track loss / occlusion gap: leave unsolved (spline covers it).
                continue;
            };
            poses[k] = Some(res.pose);

            // Triangulate tracks that now have ≥2 solved observations.
            for tr in &mut self.tracks {
                if tr.landmark.is_some() {
                    continue;
                }
                let solved: Vec<(Se3, (f64, f64))> = tr
                    .obs
                    .iter()
                    .filter_map(|(kf, p)| {
                        poses[*kf].map(|pose| (pose, self.cfg.intrinsics.norm(*p)))
                    })
                    .collect();
                if solved.len() < 2 {
                    continue;
                }
                if let Some(p) = triangulate_n(&solved) {
                    // Cheirality + parallax over the solved views.
                    let ok_z = solved.iter().all(|(pose, _)| pose.act(&p)[2] > 1e-3);
                    let ang = parallax_angle(&solved[0].0, &solved[solved.len() - 1].0, &p);
                    if ok_z && ang > self.cfg.boot_min_parallax * 0.5 {
                        tr.landmark = Some(landmarks.len());
                        landmarks.push(p);
                    }
                }
            }

            // Sliding-window BA around k (every other keyframe — the window
            // overlaps heavily and the final global pass polishes the rest).
            if k % 2 == 0 {
                self.local_ba(&mut poses, &mut landmarks, k);
            }
        }

        // Final polish: full BA with the anchor pair fixed (cheap at VO scale).
        self.global_ba(&mut poses, &mut landmarks, anchor);

        let keyframe_poses: Vec<Option<KeyframePose>> = poses
            .iter()
            .enumerate()
            .map(|(k, p)| {
                p.map(|pose| KeyframePose {
                    pts: self.keyframes[k].pts,
                    pose,
                })
            })
            .collect();
        let spline_samples: Vec<(f64, Se3)> = keyframe_poses
            .iter()
            .flatten()
            .map(|kp| (kp.pts, kp.pose))
            .collect();
        let spline = PoseSpline::fit(&spline_samples);
        // Reference observation per landmark: the middle keyframe obs of the
        // owning track (median viewpoint → least grazing appearance sample).
        let mut landmark_obs = vec![(usize::MAX, (0.0f32, 0.0f32)); landmarks.len()];
        let mut landmark_desc =
            vec![[0u8; crate::descriptor::DESC_BYTES]; landmarks.len()];
        for tr in &self.tracks {
            let Some(l) = tr.landmark else { continue };
            if !tr.obs.is_empty() {
                let mid = tr.obs.len() / 2;
                let (kf, p) = tr.obs[mid];
                landmark_obs[l] = (kf, p);
                landmark_desc[l] = tr.obs_desc[mid];
            }
        }
        Some(VoResult {
            keyframe_poses,
            landmarks: landmarks
                .iter()
                .map(|l| DVec3::new(l[0], l[1], l[2]))
                .collect(),
            landmark_obs,
            landmark_desc,
            spline,
            anchor,
        })
    }

    /// BA over the window of solved keyframes nearest to `center`.
    fn local_ba(
        &self,
        poses: &mut [Option<Se3>],
        landmarks: &mut [Vector3<f64>],
        center: usize,
    ) {
        let solved: Vec<usize> = (0..poses.len()).filter(|&k| poses[k].is_some()).collect();
        if solved.len() < 3 {
            return;
        }
        let mut window: Vec<usize> = solved.clone();
        window.sort_by_key(|&k| k.abs_diff(center));
        window.truncate(self.cfg.ba_window);
        window.sort_unstable();
        self.run_ba(poses, landmarks, &window, 2, 8);
    }

    fn global_ba(
        &self,
        poses: &mut [Option<Se3>],
        landmarks: &mut [Vector3<f64>],
        anchor: usize,
    ) {
        let mut window: Vec<usize> = (0..poses.len()).filter(|&k| poses[k].is_some()).collect();
        // Fix only the anchor: scale is a free monocular gauge (absorbed by
        // downstream Sim(3) alignment), and pinning the bootstrap pair would
        // freeze its two-view error into the solution.
        window.sort_by_key(|&k| (k != anchor, k));
        self.run_ba(poses, landmarks, &window, 1, 40);
    }

    fn run_ba(
        &self,
        poses: &mut [Option<Se3>],
        landmarks: &mut [Vector3<f64>],
        window: &[usize],
        fixed: usize,
        iters: usize,
    ) {
        // Compact indices: BA cam i ↔ keyframe window[i].
        let cam_of_kf: std::collections::HashMap<usize, usize> =
            window.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let mut lm_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut prob = BaProblem {
            poses: window.iter().map(|&k| poses[k].unwrap()).collect(),
            landmarks: Vec::new(),
            obs: Vec::new(),
        };
        for tr in &self.tracks {
            let Some(l) = tr.landmark else { continue };
            for (kf, p) in &tr.obs {
                let Some(&cam) = cam_of_kf.get(kf) else { continue };
                let lm = *lm_map.entry(l).or_insert_with(|| {
                    prob.landmarks.push(landmarks[l]);
                    prob.landmarks.len() - 1
                });
                prob.obs.push(Obs {
                    cam,
                    lm,
                    xy: self.cfg.intrinsics.norm(*p),
                });
            }
        }
        if prob.landmarks.is_empty() {
            return;
        }
        let cfg = BaConfig {
            fixed_poses: fixed.min(window.len().saturating_sub(1)).max(1),
            max_iters: iters,
            ..Default::default()
        };
        optimize(&mut prob, &cfg);
        for (i, &k) in window.iter().enumerate() {
            poses[k] = Some(prob.poses[i]);
        }
        for (&l, &pl) in &lm_map {
            landmarks[l] = prob.landmarks[pl];
        }
    }
}

/// RMS residual (px) of the best global affine warp a→b over the matches.
/// Rotation and zoom are (nearly) affine at video field-of-view; what's left
/// is parallax, i.e. usable baseline.
fn affine_residual_px(matches: &[Match2], focal: f64) -> f64 {
    let n = matches.len();
    if n < 6 {
        return 0.0;
    }
    let mut m = nalgebra::Matrix3::<f64>::zeros();
    let mut bx = Vector3::zeros();
    let mut by = Vector3::zeros();
    for mm in matches {
        let v = Vector3::new(mm.a.0, mm.a.1, 1.0);
        m += v * v.transpose();
        bx += v * mm.b.0;
        by += v * mm.b.1;
    }
    let Some(minv) = m.try_inverse() else {
        return 0.0;
    };
    let px = minv * bx;
    let py = minv * by;
    let mut ss = 0.0;
    for mm in matches {
        let v = Vector3::new(mm.a.0, mm.a.1, 1.0);
        let rx = px.dot(&v) - mm.b.0;
        let ry = py.dot(&v) - mm.b.1;
        ss += rx * rx + ry * ry;
    }
    (ss / n as f64).sqrt() * focal
}

fn nearest_pose(poses: &[Option<Se3>], k: usize) -> Option<Se3> {
    (0..poses.len())
        .filter(|&i| poses[i].is_some())
        .min_by_key(|&i| i.abs_diff(k))
        .and_then(|i| poses[i])
}

/// Cheap sharpness proxy: mean squared central-difference gradient.
fn gradient_energy(img: &GrayImage) -> f32 {
    let (w, h) = (img.width, img.height);
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for y in (1..h - 1).step_by(2) {
        for x in (1..w - 1).step_by(2) {
            let gx = img.get(x + 1, y) - img.get(x - 1, y);
            let gy = img.get(x, y + 1) - img.get(x, y - 1);
            sum += (gx * gx + gy * gy) as f64;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (sum / n as f64) as f32 }
}
