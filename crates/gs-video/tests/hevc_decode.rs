//! HEVC decode gate on a real H.265 file (skips when the sample is absent).
//! `samples/video/hevc-test/jellyfish.mp4` is a public 1080p hev1 test clip
//! — it validates demux (hvcC), header parsing, and the NVDEC H.265 session.
//! The 10-bit Main-10 + wide-gamut path additionally needs a native iPhone
//! clip in `samples/video/*-iphone/` when one is available.

use std::path::PathBuf;

fn sample() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/video/hevc-test/jellyfish.mp4");
    p.exists().then_some(p)
}

#[test]
fn decodes_hevc_clip() {
    let Some(path) = sample() else {
        eprintln!("SKIPPING: no HEVC sample at samples/video/hevc-test/jellyfish.mp4");
        return;
    };
    let mut reader = gs_video::VideoReader::open(&path).expect("open");
    assert!(
        matches!(reader, gs_video::VideoReader::H265(_)),
        "expected the HEVC track to route to the H.265 decoder"
    );

    let mut count = 0u32;
    let mut last_pts = f64::NEG_INFINITY;
    let mut max_var = 0.0f64;
    while let Some(frame) = reader.next_frame().expect("decode") {
        assert!(frame.pts.is_finite());
        // Reorder queue must deliver presentation order (strictly increasing
        // for constant-frame-rate test footage).
        assert!(
            frame.pts > last_pts,
            "pts went backwards: {} after {last_pts}",
            frame.pts
        );
        last_pts = frame.pts;
        assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);
        assert_eq!(
            frame.u.len(),
            (frame.width / 2 * frame.height / 2) as usize
        );
        // Luma variance: decoded content, not a flat field.
        let n = frame.y.len() as f64;
        let mean = frame.y.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = frame.y.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        max_var = max_var.max(var);
        count += 1;
        if count >= 90 {
            break;
        }
    }
    assert!(count >= 60, "decoded only {count} frames");
    assert!(max_var > 25.0, "luma variance {max_var:.1} — flat output?");
    eprintln!("decoded {count} HEVC frames, max luma variance {max_var:.1}");
}
