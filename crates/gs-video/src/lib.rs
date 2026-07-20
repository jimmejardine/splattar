//! Video ingest: MP4 demux + pure-Rust H.264 decode with per-frame PTS
//! (iPhone/Android video is VFR — never frame_index/fps), YUV→RGB, sharpness
//! scoring, and sharpest-in-window keyframe promotion. Video is the only
//! product input; this crate is the front door.

pub mod color;
pub mod h264;
pub mod keyframes;
pub mod mp4_reader;
// Raw Vulkan Video FFI: every call is unsafe by nature; per-call unsafe
// blocks would be pure noise here.
#[allow(unsafe_op_in_unsafe_fn)]
pub mod nvdec;
pub mod sharpness;

pub use keyframes::{Keyframe, KeyframeConfig, select_keyframes};
pub use mp4_reader::{DecodedFrame, Mp4H264Reader};

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("mp4 demux error: {0}")]
    Mp4(#[from] mp4::Error),
    #[error("no H.264 video track in file (iPhone HEVC is not yet supported — record H.264 / 'Most Compatible')")]
    NoH264Track,
    #[error("h264 decode error at sample {sample}: {message}")]
    Decode { sample: u32, message: String },
    #[error("nvdec: {0}")]
    NvDec(#[from] nvdec::NvDecError),
}
