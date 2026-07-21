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
use rayon::prelude::*;

use crate::ba::{BaConfig, BaProblem, Obs, optimize};
use crate::detect::{DetectConfig, detect};
use crate::image::{GrayImage, Pyramid};
use crate::klt::{KltConfig, track_point_fb};
use crate::pnp::{PnpConfig, solve_pnp};
use crate::se3::Se3;
use crate::spline::PoseSpline;
use crate::triangulate::{parallax_angle, triangulate_n};
use crate::twoview::{Match2, recover_pose};

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
    /// Median flow since the last keyframe that forces a new keyframe, as a
    /// FRACTION of the image diagonal (resolution-independent; an absolute
    /// pixel threshold promoted ~90% of frames on high-res phone footage).
    /// Resolved to pixels when the first frame arrives.
    pub kf_flow_frac: f32,
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
            // 0.015 ≈ the field-proven 15 px at 478×850 (diag ~975).
            kf_flow_frac: 0.015,
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
    /// Oriented multi-scale descriptor per observation (cross-video matching).
    obs_desc: Vec<crate::descriptor::MultiDescriptor>,
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
    /// ALL keyframe observations per landmark — (keyframe index, pixel).
    /// Registration bridges need to assemble many matched landmarks observed
    /// in ONE keyframe (single-camera PnP), which the median obs alone can't.
    pub landmark_obs_all: Vec<Vec<(usize, (f32, f32))>>,
    /// Binary descriptor at the reference observation (cross-video matching).
    pub landmark_desc: Vec<crate::descriptor::MultiDescriptor>,
    pub spline: Option<PoseSpline>,
    /// Index of the anchor keyframe (gauge origin).
    pub anchor: usize,
}

/// Half-resolution grayscale snapshot of a keyframe (pyramid level 1,
/// quantized to u8) — the raw material for pairwise image registration.
pub struct Thumb {
    pub kf: usize,
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Why keyframes were promoted (diagnostic; logged with the causal summary).
#[derive(Default, Debug, Clone, Copy)]
pub struct PromotionStats {
    pub flow: u32,
    pub survival: u32,
    pub tracks: u32,
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
    /// Every 4th keyframe's half-res image, kept for registration.
    pub thumbs: Vec<Thumb>,
    /// kf_flow_frac resolved to pixels from the first frame's diagonal.
    kf_flow_px: f32,
    pub promotions: PromotionStats,
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
            thumbs: Vec::new(),
            kf_flow_px: 0.0,
            promotions: PromotionStats::default(),
        }
    }

    /// Causal pass: feed frames in decode order with their PTS.
    pub fn push_frame(&mut self, gray: GrayImage, pts: f64) {
        if self.n_frames == 0 {
            let diag =
                ((gray.width * gray.width + gray.height * gray.height) as f32).sqrt();
            self.kf_flow_px = self.cfg.kf_flow_frac * diag;
        }
        let pyr = Pyramid::build(gray, self.cfg.pyramid_levels);
        // Sharpness from the half-res pyramid level: scores are only compared
        // relatively across frames, and this skips a full-res pass per frame.
        let sharpness = gradient_energy(&pyr.levels[1.min(pyr.levels.len() - 1)]);
        let frame_idx = self.n_frames;
        self.n_frames += 1;
        self.frame_pts.push(pts);

        let mut zoom = 0.0f64;
        if let Some(prev) = &self.prev {
            // Track with constant-velocity prediction (median of last flow).
            let (mfx, mfy) = self.prev_median_flow;
            let (cx, cy) = (self.cfg.intrinsics.cx as f32, self.cfg.intrinsics.cy as f32);
            let klt = &self.cfg.klt;
            // Per-feature pyramidal LK is the causal-pass hot loop and is
            // embarrassingly parallel (the pyramids are read-only, each track
            // owns its state). Per-track results are reduced in track order
            // below, so the flow median and the zoom sums are identical to
            // the serial loop regardless of thread count.
            /// (flow, radial numerator, radial denominator) per surviving track.
            type Tracked = Option<((f32, f32), f64, f64)>;
            let tracked: Vec<Tracked> = self
                .tracks
                .par_iter_mut()
                .map(|tr| {
                    let p = tr.cur?;
                    let guess = (p.0 + mfx, p.1 + mfy);
                    match track_point_fb(prev, &pyr, p, guess, klt) {
                        Some(q) => {
                            tr.cur = Some(q);
                            // Radial components about the principal point.
                            let r0 = ((p.0 - cx) as f64, (p.1 - cy) as f64);
                            let r1 = ((q.0 - cx) as f64, (q.1 - cy) as f64);
                            Some((
                                (q.0 - p.0, q.1 - p.1),
                                r1.0 * r0.0 + r1.1 * r0.1,
                                r0.0 * r0.0 + r0.1 * r0.1,
                            ))
                        }
                        None => {
                            tr.cur = None;
                            None
                        }
                    }
                })
                .collect();
            let mut flows: Vec<(f32, f32)> = Vec::with_capacity(tracked.len());
            let mut radial_num = 0.0f64;
            let mut radial_den = 0.0f64;
            for (flow, rn, rd) in tracked.into_iter().flatten() {
                flows.push(flow);
                radial_num += rn;
                radial_den += rd;
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

        // Keyframe decision (reasons counted for the causal summary log).
        let is_first = self.keyframes.is_empty();
        let (flow_med, survival) = self.kf_statistics();
        let promote = is_first
            || flow_med >= self.kf_flow_px
            || survival < self.cfg.kf_survival
            || self.live_count() < self.cfg.min_tracks;
        if promote && !is_first {
            if flow_med >= self.kf_flow_px {
                self.promotions.flow += 1;
            } else if survival < self.cfg.kf_survival {
                self.promotions.survival += 1;
            } else {
                self.promotions.tracks += 1;
            }
        }
        if promote {
            let kf_idx = self.keyframes.len();
            self.keyframes.push(Keyframe {
                frame_idx,
                pts,
                sharpness,
            });
            // Keep every 4th keyframe's half-res image for pairwise
            // registration (vote buckets are 8 keyframes wide, so this
            // guarantees ≥2 snapshots per bucket).
            if kf_idx.is_multiple_of(4) {
                let lvl = &pyr.levels[1.min(pyr.levels.len() - 1)];
                self.thumbs.push(Thumb {
                    kf: kf_idx,
                    width: lvl.width,
                    height: lvl.height,
                    data: lvl
                        .data
                        .iter()
                        .map(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
                        .collect(),
                });
            }
            // Descriptor extraction fans out over tracks (indexed map applied
            // in track order — deterministic under any thread count); it was
            // the dominant serial per-keyframe cost.
            let descs: Vec<Option<crate::descriptor::MultiDescriptor>> = self
                .tracks
                .par_iter()
                .map(|tr| {
                    tr.cur
                        .map(|p| crate::descriptor::describe_multi(&pyr, p.0, p.1))
                })
                .collect();
            for (tr, desc) in self.tracks.iter_mut().zip(descs) {
                if let (Some(p), Some(d)) = (tr.cur, desc) {
                    tr.obs.push((kf_idx, p));
                    tr.obs_desc.push(d);
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
            let corner_descs: Vec<crate::descriptor::MultiDescriptor> = corners
                .par_iter()
                .map(|c| crate::descriptor::describe_multi(&pyr, c.x, c.y))
                .collect();
            for (c, d) in corners.into_iter().zip(corner_descs) {
                self.tracks.push(Track {
                    cur: Some((c.x, c.y)),
                    at_last_kf: Some((c.x, c.y)),
                    obs: vec![(kf_idx, (c.x, c.y))],
                    obs_desc: vec![d],
                    landmark: None,
                });
            }
        }
        self.prev = Some(pyr);
    }

    fn live_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.cur.is_some()).count()
    }

    /// Live-track count (diagnostics/tests).
    pub fn live_tracks(&self) -> usize {
        self.live_count()
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

    /// Anchor-out solve over the whole clip, returning the segment with the
    /// most solved keyframes. Kept for callers (and validation gates) that
    /// want a single gauge; well-connected footage yields exactly one
    /// segment, so this matches the pre-segmentation behavior there.
    pub fn solve(&mut self) -> Option<VoResult> {
        self.solve_segments()
            .into_iter()
            .max_by_key(|r| r.keyframe_poses.iter().flatten().count())
    }

    /// Segment-recursive anchor-out solve: solve the best-conditioned
    /// segment of a keyframe range, then recurse into the unsolved flanks —
    /// each segment gets its own monocular gauge (scale is per-submap until
    /// Sim(3) alignment, per CLAUDE.md). Track continuity breaks (fast pans,
    /// doorways) end a segment exactly where PnP stops connecting, so
    /// boundaries land on low-quality footage by construction. Returned in
    /// temporal order.
    pub fn solve_segments(&mut self) -> Vec<VoResult> {
        /// Ranges shorter than this can't bootstrap + margin meaningfully.
        /// (Deliberately permissive — downstream training applies its own
        /// minimum view/landmark counts before a sliver becomes a submap.)
        const MIN_SEGMENT_KF: usize = 8;
        /// Runaway backstop, far above any real walkthrough.
        const MAX_SEGMENTS: usize = 12;
        let n_kf = self.keyframes.len();
        let mut out: Vec<VoResult> = Vec::new();
        let mut pending: Vec<(usize, usize)> = vec![(0, n_kf)];
        while let Some((lo, hi)) = pending.pop() {
            if hi.saturating_sub(lo) < MIN_SEGMENT_KF || out.len() >= MAX_SEGMENTS {
                continue;
            }
            let Some(res) = self.solve_range(lo, hi) else {
                log::info!("segment [{lo},{hi}): no viable bootstrap — dropped");
                continue;
            };
            let solved: Vec<usize> = res
                .keyframe_poses
                .iter()
                .enumerate()
                .filter_map(|(k, p)| p.is_some().then_some(k))
                .collect();
            let (first, last) = (solved[0], *solved.last().unwrap());
            log::info!(
                "segment solved: keyframes [{first}..{last}] ({} of {} in range [{lo},{hi}))",
                solved.len(),
                hi - lo
            );
            pending.push((lo, first));
            pending.push((last + 1, hi));
            out.push(res);
        }
        out.sort_by_key(|r| {
            r.keyframe_poses
                .iter()
                .position(|p| p.is_some())
                .unwrap_or(usize::MAX)
        });
        out
    }

    /// Anchor-out solve restricted to keyframes [lo, hi).
    fn solve_range(&mut self, range_lo: usize, range_hi: usize) -> Option<VoResult> {
        let n_kf = self.keyframes.len();
        let range_n = range_hi.saturating_sub(range_lo);
        if range_n < 2 {
            return None;
        }
        // Landmark assignments are per-segment (each segment is its own
        // gauge) — clear anything a previous segment's solve left behind.
        for tr in &mut self.tracks {
            tr.landmark = None;
        }
        let thresh = self.cfg.ransac_px / self.cfg.intrinsics.focal;

        // Candidate anchor pairs: from each start keyframe, extend forward
        // until the shared tracks carry enough flow for a well-conditioned
        // two-view solve (adjacent keyframes rarely have the baseline).
        // Margins keep the anchor away from the range ends — boundaries sit
        // at low-quality footage by construction.
        let margin = ((range_n as f64 * self.cfg.anchor_margin) as usize).min(range_n / 4);
        let lo = range_lo + margin;
        let hi = (range_hi - 1 - margin).max(lo + 1).min(range_hi - 1);
        // Sample candidate starts — scanning every keyframe is O(n²) in
        // track lookups and adds nothing once the pool is a few dozen.
        let step = ((hi - lo) / 48).max(1);
        // Each candidate start is scored independently (read-only track
        // scans, the O(n²) cost noted above) — parallel, order preserved.
        let starts: Vec<usize> = (lo..hi).step_by(step).collect();
        let mut candidates: Vec<(f64, usize, usize)> = starts
            .par_iter()
            .filter_map(|&ka| {
                let mut chosen: Option<(usize, usize, f64)> = None;
                for kb in ka + 1..range_hi {
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
                crate::fivepoint::ransac_essential_5pt(&matches, self.cfg.ransac_iters, thresh, 0xC0FFEE)
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

        // Anchor-out expansion: everything in range outside the (solved)
        // anchor pair, nearest-to-anchor first, alternating temporal
        // directions.
        let mut order: Vec<usize> =
            (range_lo..range_hi).filter(|&k| poses[k].is_none()).collect();
        order.sort_by_key(|&k| k.abs_diff(anchor));

        let total_pending = order.len();
        let mut solved_count = 2usize; // the bootstrap pair
        for (processed, k) in order.into_iter().enumerate() {
            if processed > 0 && processed % 100 == 0 {
                log::info!(
                    "anchor-out mapping: {processed}/{total_pending} keyframes ({solved_count} solved, {} landmarks)",
                    landmarks.len()
                );
            }
            // PnP from tracks with landmarks observed at keyframe k. The scan
            // over all tracks is per-track independent; order-preserving
            // parallel collect keeps PnP inputs identical to the serial scan.
            let gathered: Vec<(Vector3<f64>, (f64, f64))> = self
                .tracks
                .par_iter()
                .filter_map(|tr| {
                    let l = tr.landmark?;
                    tr.obs
                        .iter()
                        .find(|o| o.0 == k)
                        .map(|&(_, p)| (landmarks[l], self.cfg.intrinsics.norm(p)))
                })
                .collect();
            let (pts3, obs2): (Vec<_>, Vec<_>) = gathered.into_iter().unzip();
            // PnP needs real support: a pose glued on from a handful of
            // (possibly spurious) surviving tracks poisons the gauge across
            // genuine continuity breaks — leave it unsolved instead (the
            // segment recursion picks the far side up in its own gauge).
            const MIN_PNP_LANDMARKS: usize = 12;
            if pts3.len() < MIN_PNP_LANDMARKS {
                continue;
            }
            // Prior: nearest solved pose.
            let prior = nearest_pose(&poses, k)?;
            let Some(res) = solve_pnp(&pts3, &obs2, &prior, &PnpConfig::default()) else {
                // Track loss / occlusion gap: leave unsolved (spline covers it).
                continue;
            };
            poses[k] = Some(res.pose);
            solved_count += 1;

            // Triangulate tracks that now have ≥2 solved observations —
            // candidates found in parallel, committed serially in track order
            // so landmark numbering stays deterministic.
            let new_points: Vec<(usize, Vector3<f64>)> = self
                .tracks
                .par_iter()
                .enumerate()
                .filter_map(|(ti, tr)| {
                    if tr.landmark.is_some() {
                        return None;
                    }
                    let solved: Vec<(Se3, (f64, f64))> = tr
                        .obs
                        .iter()
                        .filter_map(|(kf, p)| {
                            poses[*kf].map(|pose| (pose, self.cfg.intrinsics.norm(*p)))
                        })
                        .collect();
                    if solved.len() < 2 {
                        return None;
                    }
                    let p = triangulate_n(&solved)?;
                    // Cheirality + parallax over the solved views.
                    let ok_z = solved.iter().all(|(pose, _)| pose.act(&p)[2] > 1e-3);
                    let ang = parallax_angle(&solved[0].0, &solved[solved.len() - 1].0, &p);
                    (ok_z && ang > self.cfg.boot_min_parallax * 0.5).then_some((ti, p))
                })
                .collect();
            for (ti, p) in new_points {
                self.tracks[ti].landmark = Some(landmarks.len());
                landmarks.push(p);
            }

            // Sliding-window BA around k (every other keyframe — the window
            // overlaps heavily and the final global pass polishes the rest).
            if k % 2 == 0 {
                self.local_ba(&mut poses, &mut landmarks, k);
            }
        }

        // Final polish: full BA with the anchor pair fixed (cheap at VO scale).
        log::info!(
            "anchor-out mapping done: {solved_count}/{range_n} solved in [{range_lo},{range_hi}), {} landmarks; global BA...",
            landmarks.len()
        );
        let t_ba = std::time::Instant::now();
        self.global_ba(&mut poses, &mut landmarks, anchor);
        log::info!("global BA done in {:.1}s", t_ba.elapsed().as_secs_f64());

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
        let mut landmark_obs_all: Vec<Vec<(usize, (f32, f32))>> =
            vec![Vec::new(); landmarks.len()];
        let mut landmark_desc =
            vec![[[0u8; crate::descriptor::DESC_BYTES]; crate::descriptor::DESC_LEVELS];
                landmarks.len()];
        for tr in &self.tracks {
            let Some(l) = tr.landmark else { continue };
            if !tr.obs.is_empty() {
                let mid = tr.obs.len() / 2;
                let (kf, p) = tr.obs[mid];
                landmark_obs[l] = (kf, p);
                landmark_obs_all[l] = tr.obs.clone();
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
            landmark_obs_all,
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
