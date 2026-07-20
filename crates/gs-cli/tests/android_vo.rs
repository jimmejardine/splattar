//! VO gate on real Android walkthrough footage (skips when the sample is
//! absent). Also serves as the tracking diagnostic harness: run with
//! `--nocapture` to see per-frame survival.

use std::path::PathBuf;

use gs_pose::vo::{Intrinsics, VoConfig, VoFrontEnd};
use gs_pose::{DetectConfig, GrayImage, KltConfig, Pyramid, detect, track_point, track_point_fb};

fn sample() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/video/prinsengracht-494-android/1.mp4");
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIPPING android VO test: {} not present", p.display());
        None
    }
}

fn gray(frame: &gs_video::DecodedFrame) -> GrayImage {
    GrayImage::from_luma8(&frame.y, frame.width as usize, frame.height as usize)
}

/// Diagnostic: frame-to-frame KLT survival on raw consecutive frames.
/// Slow (decodes + tracks 40 frames): `cargo test -- --ignored`.
#[test]
#[ignore = "slow: real-video decode + tracking (~30 s)"]
fn tracking_survival_diagnostic() {
    let Some(path) = sample() else { return };
    let mut reader = gs_video::Mp4H264Reader::open(&path).expect("open");
    let klt = KltConfig::default();
    let det = DetectConfig::default();

    let mut prev: Option<Pyramid> = None;
    let mut worst_fb = 1.0f64;
    let mut worst_raw = 1.0f64;
    for k in 0..40 {
        let Some(frame) = reader.next_frame().expect("decode") else {
            break;
        };
        let pyr = Pyramid::build(gray(&frame), 4);
        if let Some(p) = &prev {
            let corners = detect(&p.levels[0], &det, &[]);
            let mut raw_ok = 0;
            let mut fb_ok = 0;
            let mut flows: Vec<f32> = Vec::new();
            for c in &corners {
                if track_point(p, &pyr, (c.x, c.y), (c.x, c.y), &klt).is_some() {
                    raw_ok += 1;
                }
                if let Some(q) = track_point_fb(p, &pyr, (c.x, c.y), (c.x, c.y), &klt) {
                    fb_ok += 1;
                    flows.push(((q.0 - c.x).powi(2) + (q.1 - c.y).powi(2)).sqrt());
                }
            }
            flows.sort_by(f32::total_cmp);
            let med = flows.get(flows.len() / 2).copied().unwrap_or(-1.0);
            let n = corners.len().max(1);
            eprintln!(
                "frame {k}: {} corners, raw {raw_ok} ({:.0}%), fb {fb_ok} ({:.0}%), med flow {med:.1}px",
                corners.len(),
                100.0 * raw_ok as f64 / n as f64,
                100.0 * fb_ok as f64 / n as f64,
            );
            worst_raw = worst_raw.min(raw_ok as f64 / n as f64);
            worst_fb = worst_fb.min(fb_ok as f64 / n as f64);
        }
        prev = Some(pyr);
    }
    eprintln!("worst raw survival {worst_raw:.2}, worst fb survival {worst_fb:.2}");
    assert!(
        worst_fb > 0.5,
        "frame-to-frame FB tracking collapses: worst {worst_fb:.2}"
    );
}

/// The M6 real-footage gate: causal pass + anchor-out solve succeed on the
/// first 20 s of the walkthrough with a healthy solved ratio.
#[test]
#[ignore = "slow: full VO on 600 real frames (minutes)"]
fn vo_solves_android_walkthrough() {
    let Some(path) = sample() else { return };
    let mut reader = gs_video::Mp4H264Reader::open(&path).expect("open");
    let mut vo: Option<VoFrontEnd> = None;
    let mut n = 0;
    while let Some(frame) = reader.next_frame().expect("decode") {
        let fe = vo.get_or_insert_with(|| {
            VoFrontEnd::new(VoConfig {
                intrinsics: Intrinsics {
                    focal: 0.85 * frame.width.max(frame.height) as f64,
                    cx: frame.width as f64 / 2.0,
                    cy: frame.height as f64 / 2.0,
                },
                ..Default::default()
            })
        });
        fe.push_frame(gray(&frame), frame.pts);
        n += 1;
        if n >= 600 {
            break;
        }
    }
    let mut fe = vo.expect("frames");
    let n_kf = fe.keyframes.len();
    eprintln!("{n} frames -> {n_kf} keyframes");
    // Dense keyframes are legitimate during fast pans; only a KF for
    // literally every frame would indicate broken statistics.
    assert!(
        n_kf < n * 9 / 10,
        "keyframe promotion runaway: {n_kf} keyframes from {n} frames"
    );
    let result = fe.solve().expect("VO solve");
    let solved = result.keyframe_poses.iter().flatten().count();
    eprintln!(
        "solved {solved}/{n_kf} keyframes, {} landmarks, anchor {}",
        result.landmarks.len(),
        result.anchor
    );
    assert!(
        solved as f64 >= 0.5 * n_kf as f64,
        "only {solved}/{n_kf} keyframes solved"
    );
    assert!(result.landmarks.len() > 300, "too few landmarks");
}
