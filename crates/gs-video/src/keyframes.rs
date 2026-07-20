//! Sharpest-in-window keyframe promotion. M5 windows by PTS spacing; flow-
//! displacement-driven promotion joins in M6 when tracking exists. All timing
//! is PTS-based (VFR-safe).

use crate::mp4_reader::{DecodedFrame, Mp4H264Reader};
use crate::{VideoError, sharpness};

pub struct KeyframeConfig {
    /// Target spacing between keyframes, seconds of PTS.
    pub window_s: f64,
    /// Optional hard cap on emitted keyframes (0 = uncapped).
    pub max_keyframes: usize,
}

impl Default for KeyframeConfig {
    fn default() -> Self {
        Self {
            window_s: 0.4,
            max_keyframes: 0,
        }
    }
}

pub struct Keyframe {
    pub pts: f64,
    pub width: u32,
    pub height: u32,
    pub sharpness: f64,
    /// Luma plane (tracking input).
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    /// Index of the source frame in decode order.
    pub frame_index: u32,
}

/// Stream the video once (decode order), promoting the sharpest frame in each
/// PTS window. Returns keyframes in PTS order.
pub fn select_keyframes(
    reader: &mut Mp4H264Reader,
    config: &KeyframeConfig,
) -> Result<Vec<Keyframe>, VideoError> {
    let mut keyframes: Vec<Keyframe> = Vec::new();
    let mut window_start: Option<f64> = None;
    let mut best: Option<Keyframe> = None;
    let mut index = 0u32;

    let flush = |best: &mut Option<Keyframe>, keyframes: &mut Vec<Keyframe>| {
        if let Some(kf) = best.take() {
            keyframes.push(kf);
        }
    };

    while let Some(frame) = reader.next_frame()? {
        let DecodedFrame {
            pts,
            width,
            height,
            y,
            u,
            v,
        } = frame;
        let sharp = sharpness::laplacian_variance(&y, width as usize, height as usize);
        let start = *window_start.get_or_insert(pts);
        if pts - start >= config.window_s {
            flush(&mut best, &mut keyframes);
            window_start = Some(pts);
            if config.max_keyframes > 0 && keyframes.len() >= config.max_keyframes {
                break;
            }
        }
        if best.as_ref().is_none_or(|b| sharp > b.sharpness) {
            best = Some(Keyframe {
                pts,
                width,
                height,
                sharpness: sharp,
                y,
                u,
                v,
                frame_index: index,
            });
        }
        index += 1;
    }
    flush(&mut best, &mut keyframes);
    keyframes.sort_by(|a, b| a.pts.partial_cmp(&b.pts).unwrap());
    log::info!(
        "keyframes: {} promoted from {} decoded frames",
        keyframes.len(),
        index
    );
    Ok(keyframes)
}
