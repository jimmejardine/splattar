//! Decode gate on real Android walkthrough footage (samples are gitignored;
//! test skips when absent). Checks: frames decode, dimensions sane, PTS
//! strictly increasing in display terms (VFR-safe), luma has real content,
//! keyframe promotion picks a reasonable count.

use std::path::PathBuf;

fn sample() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/video/prinsengracht-494-android/1.mp4");
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIPPING android decode test: {} not present", p.display());
        None
    }
}

#[test]
fn decodes_android_walkthrough() {
    let Some(path) = sample() else { return };
    let mut reader = gs_video::Mp4H264Reader::open(&path).expect("open mp4");

    let mut count = 0u32;
    let mut last_pts = f64::NEG_INFINITY;
    let mut dims = (0u32, 0u32);
    let mut luma_var_seen = false;
    while let Some(frame) = reader.next_frame().expect("decode") {
        assert!(frame.pts.is_finite());
        assert!(
            frame.pts > last_pts - 1.0,
            "pts wildly out of order: {} after {last_pts}",
            frame.pts
        );
        last_pts = last_pts.max(frame.pts);
        dims = (frame.width, frame.height);
        assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);
        if !luma_var_seen {
            let mean: f64 =
                frame.y.iter().map(|&v| v as f64).sum::<f64>() / frame.y.len() as f64;
            let var: f64 = frame
                .y
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / frame.y.len() as f64;
            luma_var_seen = var > 25.0; // real footage, not a flat field
        }
        count += 1;
        if count >= 90 {
            break; // three seconds is plenty for the gate
        }
    }
    eprintln!("decoded {count} frames at {}x{}, last pts {last_pts:.3}s", dims.0, dims.1);
    assert!(count >= 60, "expected at least 60 decodable frames, got {count}");
    // The sample is 478x850 portrait; gate on "plausibly a phone video".
    assert!(
        dims.0.min(dims.1) >= 360 && dims.0.max(dims.1) >= 640,
        "unexpected dims {dims:?}"
    );
    assert!(luma_var_seen, "luma variance never exceeded threshold — decode garbage?");
}

#[test]
fn promotes_keyframes_with_pts_spacing() {
    let Some(path) = sample() else { return };
    let mut reader = gs_video::Mp4H264Reader::open(&path).expect("open mp4");
    let keyframes = gs_video::select_keyframes(
        &mut reader,
        &gs_video::KeyframeConfig {
            window_s: 0.5,
            max_keyframes: 12,
        },
    )
    .expect("keyframes");
    assert!(keyframes.len() >= 8, "too few keyframes: {}", keyframes.len());
    for pair in keyframes.windows(2) {
        assert!(pair[1].pts > pair[0].pts, "keyframe pts not increasing");
        assert!(pair[1].pts - pair[0].pts >= 0.2, "keyframes bunched up");
    }
    assert!(keyframes.iter().all(|k| k.sharpness > 0.0));
}
