//! `gs-cli play` — frame-stepping sanity player over the real decode path
//! (H.264 and H.265 via NVDEC). Steps forward only; restart reopens the
//! reader. For eyeballing decoder output, tone mapping, and crop.

use std::path::PathBuf;

use gs_viewer::frame_player::{FrameSource, PlayerFrame};

pub fn run_play(video: PathBuf) -> anyhow::Result<()> {
    let source = VideoSource::open(video)?;
    gs_viewer::frame_player::run(Box::new(source)).map_err(|e| anyhow::anyhow!(e))
}

struct VideoSource {
    path: PathBuf,
    reader: gs_video::VideoReader,
    codec: &'static str,
    frame_idx: u64,
    pts: f64,
    size: (u32, u32),
}

impl VideoSource {
    fn open(path: PathBuf) -> anyhow::Result<Self> {
        let reader = gs_video::VideoReader::open(&path)?;
        let codec = match &reader {
            gs_video::VideoReader::H264(_) => "h264",
            gs_video::VideoReader::H265(_) => "h265",
        };
        Ok(Self {
            path,
            reader,
            codec,
            frame_idx: 0,
            pts: 0.0,
            size: (0, 0),
        })
    }

    fn next(&mut self) -> Option<PlayerFrame> {
        let frame = match self.reader.next_frame() {
            Ok(f) => f?,
            Err(e) => {
                log::warn!("decode error at frame {}: {e}", self.frame_idx + 1);
                return None;
            }
        };
        self.frame_idx += 1;
        self.pts = frame.pts;
        self.size = (frame.width, frame.height);
        let rgba_f = gs_video::color::yuv420_to_rgba_f32(
            &frame.y,
            &frame.u,
            &frame.v,
            frame.width as usize,
            frame.height as usize,
        );
        let mut rgba = Vec::with_capacity(rgba_f.len() * 4);
        for p in rgba_f {
            for c in p {
                rgba.push((c * 255.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        Some(PlayerFrame {
            width: frame.width,
            height: frame.height,
            rgba,
        })
    }
}

impl FrameSource for VideoSource {
    fn step_frames(&mut self, n: u32) -> Option<PlayerFrame> {
        let mut last = None;
        for _ in 0..n {
            match self.next() {
                Some(f) => last = Some(f),
                None => break,
            }
        }
        last
    }

    fn step_secs(&mut self, secs: f64) -> Option<PlayerFrame> {
        let target = self.pts + secs;
        let mut last = None;
        while self.pts < target {
            match self.next() {
                Some(f) => last = Some(f),
                None => break,
            }
        }
        last
    }

    fn restart(&mut self) -> Option<PlayerFrame> {
        match gs_video::VideoReader::open(&self.path) {
            Ok(r) => {
                self.reader = r;
                self.frame_idx = 0;
                self.pts = 0.0;
                self.next()
            }
            Err(e) => {
                log::warn!("restart failed: {e}");
                None
            }
        }
    }

    fn status(&self) -> String {
        format!(
            "frame {} · {:.2}s · {}x{} {}",
            self.frame_idx, self.pts, self.size.0, self.size.1, self.codec
        )
    }
}
