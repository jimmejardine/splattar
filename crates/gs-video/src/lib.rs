//! Video ingest: MP4/ISO-BMFF demux + hardware H.264/H.265 decode (Vulkan
//! Video/NVDEC) with per-frame PTS (iPhone/Android video is VFR — never
//! frame_index/fps), YUV→RGB, sharpness scoring, and sharpest-in-window
//! keyframe promotion. Video is the only product input; this crate is the
//! front door. iPhone HEVC (hvc1, 8/10-bit, wide gamut) decodes through the
//! same NVDEC machinery and is folded to SDR BT.709 planes at the reader
//! boundary.

pub mod color;
pub mod h264;
pub mod h265;
pub mod hevc_demux;
pub mod keyframes;
pub mod mp4_reader;
// Raw Vulkan Video FFI: every call is unsafe by nature; per-call unsafe
// blocks would be pure noise here.
#[allow(unsafe_op_in_unsafe_fn)]
pub mod nvdec;
#[allow(unsafe_op_in_unsafe_fn)]
pub mod nvdec_h265;
pub mod sharpness;

pub use keyframes::{Keyframe, KeyframeConfig, select_keyframes};
pub use mp4_reader::{DecodedFrame, Mp4H264Reader, VideoReader};

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("mp4 demux error: {0}")]
    Mp4(#[from] mp4::Error),
    #[error("no decodable video track in file (H.264 and H.265/HEVC are supported)")]
    NoH264Track,
    #[error("decode error at sample {sample}: {message}")]
    Decode { sample: u32, message: String },
    #[error("nvdec: {0}")]
    NvDec(#[from] nvdec::NvDecError),
}
