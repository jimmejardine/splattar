//! MP4 (ISO BMFF) demux + NVDEC (Vulkan Video) decode, H.264 and H.265.
//! Samples are access units in decode order; PTS comes from sample start
//! time + composition offset over the track timescale (VFR-safe). H.264
//! phone footage has no B-frames, so decode order = presentation order;
//! HEVC may use B-frames, so [`Mp4H265Reader`] holds a small reorder queue
//! and emits frames in PTS order. Length-prefixed NALs are split and VCL
//! NALs are handed to the decoder. 10-bit wide-gamut HEVC is folded to
//! 8-bit BT.709 planes here, so every consumer sees ordinary SDR frames.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::VideoError;
use crate::h264::{Pps, Sps, parse_pps, parse_slice_header, parse_sps};
use crate::nvdec::NvDecoder;

pub struct DecodedFrame {
    /// Presentation time in seconds (VFR-safe: from the sample table).
    pub pts: f64,
    pub width: u32,
    pub height: u32,
    /// Planar 4:2:0, strides equal plane widths.
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Codec-dispatching reader — the crate's entry point. The historical name
/// [`Mp4H264Reader`] is kept as an alias so callers never changed.
pub enum VideoReader {
    H264(Box<H264Reader>),
    H265(Box<Mp4H265Reader>),
}

pub type Mp4H264Reader = VideoReader;

impl VideoReader {
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        // HEVC first: iPhone's `hvc1` tracks are invisible to the mp4
        // crate's media_type(), so probe the container bytes directly.
        if let Some(info) = crate::hevc_demux::find_hevc_track(path)? {
            return Ok(Self::H265(Box::new(Mp4H265Reader::open(path, info)?)));
        }
        Ok(Self::H264(Box::new(H264Reader::open(path)?)))
    }

    pub fn sample_count(&self) -> u32 {
        match self {
            Self::H264(r) => r.sample_count(),
            Self::H265(r) => r.sample_count(),
        }
    }

    pub fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        match self {
            Self::H264(r) => r.next_frame(),
            Self::H265(r) => r.next_frame(),
        }
    }

    /// Decode-stage timing (container read, submit, fence wait,
    /// post-process) — HEVC path only.
    pub fn decode_timing(&self) -> Option<(f64, f64, f64, f64)> {
        match self {
            Self::H264(_) => None,
            Self::H265(r) => Some(r.decode_timing()),
        }
    }
}

pub struct H264Reader {
    mp4: mp4::Mp4Reader<BufReader<File>>,
    track_id: u32,
    timescale: f64,
    sample_count: u32,
    next_sample: u32,
    sps: Sps,
    pps: Pps,
    decoder: NvDecoder,
    /// Bytes of the AVCC length prefix (usually 4).
    length_size: usize,
    /// B-frame reorder queue (Main/High profiles): decoded frames held
    /// until `reorder_depth` newer ones exist, emitted in PTS order.
    /// Baseline streams get depth 0 — behavior identical to before.
    pending: Vec<DecodedFrame>,
    reorder_depth: usize,
    /// Display rotation (degrees CW) from the tkhd matrix, baked into the
    /// emitted pixels.
    rotation: u32,
}

impl H264Reader {
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mp4 = mp4::Mp4Reader::read_header(BufReader::new(file), size)?;

        let (track_id, track) = mp4
            .tracks()
            .iter()
            .find(|(_, t)| matches!(t.media_type(), Ok(mp4::MediaType::H264)))
            .map(|(id, t)| (*id, t))
            .ok_or(VideoError::NoH264Track)?;

        // SPS/PPS + NAL length size from the avcC configuration record.
        let avcc = &track
            .trak
            .mdia
            .minf
            .stbl
            .stsd
            .avc1
            .as_ref()
            .ok_or(VideoError::NoH264Track)?
            .avcc;
        let sps_nal = avcc
            .sequence_parameter_sets
            .first()
            .ok_or(VideoError::NoH264Track)?;
        let pps_nal = avcc
            .picture_parameter_sets
            .first()
            .ok_or(VideoError::NoH264Track)?;
        let sps = parse_sps(&sps_nal.bytes).map_err(|e| VideoError::Decode {
            sample: 0,
            message: format!("SPS parse: {e}"),
        })?;
        let pps = parse_pps(&pps_nal.bytes).map_err(|e| VideoError::Decode {
            sample: 0,
            message: format!("PPS parse: {e}"),
        })?;
        let length_size = avcc.length_size_minus_one as usize + 1;

        let timescale = track.timescale() as f64;
        let sample_count = track.sample_count();
        let m = &track.trak.tkhd.matrix;
        let rotation = rotation_from_matrix(m.a, m.b, m.c, m.d);
        log::info!(
            "h264 track {track_id}: {}x{}, {} samples, timescale {}, rotation {rotation}°",
            track.width(),
            track.height(),
            sample_count,
            timescale
        );

        let decoder = NvDecoder::new(&sps, &pps)?;
        // Baseline (66) can't contain B slices → decode order is already
        // presentation order and the queue stays empty. Anything else may
        // reorder; buffer a few frames and emit by PTS.
        let reorder_depth = if sps.profile_idc > 66 {
            (sps.max_num_ref_frames as usize).max(2)
        } else {
            0
        };

        Ok(Self {
            mp4,
            track_id,
            timescale,
            sample_count,
            next_sample: 0,
            sps,
            pps,
            decoder,
            length_size,
            pending: Vec::new(),
            reorder_depth,
            rotation,
        })
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Pop the earliest pending frame (presentation order).
    fn pop_earliest(&mut self) -> Option<DecodedFrame> {
        let idx = self
            .pending
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.pts.total_cmp(&b.pts))
            .map(|(i, _)| i)?;
        Some(self.pending.swap_remove(idx))
    }

    /// Decode the next access unit (presentation order). None at end.
    pub fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        loop {
            if self.next_sample >= self.sample_count {
                return Ok(self.pop_earliest()); // drain the reorder queue
            }
            self.next_sample += 1;
            let sample_id = self.next_sample; // 1-based
            let Some(sample) = self.mp4.read_sample(self.track_id, sample_id)? else {
                return Ok(None);
            };
            let pts =
                (sample.start_time as f64 + sample.rendering_offset as f64) / self.timescale;

            // Split AVCC length-prefixed NALs; keep only slice NALs (in-band
            // SPS/PPS on phone streams repeat the avcC ones, SEI is ignored).
            let data = &sample.bytes[..];
            let mut slices: Vec<&[u8]> = Vec::new();
            let mut off = 0usize;
            while off + self.length_size <= data.len() {
                let mut len = 0usize;
                for k in 0..self.length_size {
                    len = (len << 8) | data[off + k] as usize;
                }
                off += self.length_size;
                if off + len > data.len() || len == 0 {
                    break;
                }
                let nal = &data[off..off + len];
                off += len;
                if matches!(nal[0] & 0x1f, 1 | 5) {
                    slices.push(nal);
                }
            }
            let Some(first) = slices.first() else {
                continue; // no coded picture in this sample (SEI-only etc.)
            };

            let header =
                parse_slice_header(first, &self.sps, &self.pps).map_err(|e| {
                    VideoError::Decode {
                        sample: sample_id,
                        message: format!("slice header: {e}"),
                    }
                })?;
            let nv12 = self.decoder.decode(&slices, &header).map_err(|e| {
                VideoError::Decode {
                    sample: sample_id,
                    message: e.to_string(),
                }
            })?;

            // NV12 (coded size) → planar I420 at the cropped display size.
            let (w, h) = self.decoder.cropped_size();
            let (x0, y0) = self.decoder.crop_offsets();
            let stride = nv12.stride as usize;
            let (wu, hu) = (w as usize, h as usize);
            let (x0u, y0u) = (x0 as usize, y0 as usize);
            let mut y = vec![0u8; wu * hu];
            for row in 0..hu {
                let src = (y0u + row) * stride + x0u;
                y[row * wu..(row + 1) * wu].copy_from_slice(&nv12.y[src..src + wu]);
            }
            let (cw, ch) = (wu / 2, hu / 2);
            let mut u = vec![0u8; cw * ch];
            let mut v = vec![0u8; cw * ch];
            for row in 0..ch {
                let src = (y0u / 2 + row) * stride + (x0u & !1);
                for col in 0..cw {
                    u[row * cw + col] = nv12.uv[src + 2 * col];
                    v[row * cw + col] = nv12.uv[src + 2 * col + 1];
                }
            }

            self.pending.push(apply_rotation(
                DecodedFrame {
                    pts,
                    width: w,
                    height: h,
                    y,
                    u,
                    v,
                },
                self.rotation,
            ));
            if self.pending.len() > self.reorder_depth {
                return Ok(self.pop_earliest());
            }
        }
    }
}

pub struct Mp4H265Reader {
    mp4: mp4::Mp4Reader<BufReader<File>>,
    track_id: u32,
    timescale: f64,
    sample_count: u32,
    next_sample: u32,
    sps: crate::h265::H265Sps,
    pps: crate::h265::H265Pps,
    decoder: crate::nvdec_h265::H265Decoder,
    length_size: usize,
    /// BT.2020 → BT.709 gamut fold for 10-bit wide-gamut streams.
    wide_gamut: bool,
    /// B-frame reorder queue: decoded frames held until `reorder_depth`
    /// newer ones exist, then emitted in PTS order.
    pending: Vec<DecodedFrame>,
    reorder_depth: usize,
    /// PTS of submitted-but-not-yet-retired decodes, in submission order —
    /// the pipelined decoder returns frames a few submissions late, and its
    /// outputs arrive strictly in submission order.
    pts_queue: std::collections::VecDeque<f64>,
    /// Display rotation (degrees CW) from the tkhd matrix.
    rotation: u32,
    /// Accumulated wall time per stage (diagnostic): container read, decode
    /// submit, post-processing (crop/fold/rotate). Fence-wait time lives on
    /// the decoder (`wait_s`).
    t_read: f64,
    t_submit: f64,
    t_post: f64,
}

impl Mp4H265Reader {
    fn open(path: &Path, info: crate::hevc_demux::HevcTrackInfo) -> Result<Self, VideoError> {
        use crate::h265;

        let bad = |message: String| VideoError::Decode { sample: 0, message };
        let vps_nal = info.vps.first().ok_or_else(|| bad("hvcC without VPS".into()))?;
        let sps_nal = info.sps.first().ok_or_else(|| bad("hvcC without SPS".into()))?;
        let pps_nal = info.pps.first().ok_or_else(|| bad("hvcC without PPS".into()))?;
        let vps = h265::parse_vps(vps_nal).map_err(|e| bad(format!("VPS parse: {e}")))?;
        let sps = h265::parse_sps(sps_nal).map_err(|e| bad(format!("SPS parse: {e}")))?;
        let pps = h265::parse_pps(pps_nal).map_err(|e| bad(format!("PPS parse: {e}")))?;

        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mp4 = mp4::Mp4Reader::read_header(BufReader::new(file), size)?;
        let track = mp4
            .tracks()
            .get(&info.track_id)
            .ok_or(VideoError::NoH264Track)?;
        let timescale = track.timescale() as f64;
        let sample_count = track.sample_count();

        // Wide gamut when the container says BT.2020/HLG-family, or when the
        // stream is 10-bit and says nothing (the iPhone default).
        let wide_gamut = match info.colr {
            Some((primaries, _transfer, _matrix)) => primaries == 9,
            None => sps.bit_depth_luma == 10,
        };
        log::info!(
            "h265 track {}: {}x{} coded, {}-bit{}, {} samples, timescale {}, fourcc {}, rotation {}°",
            info.track_id,
            sps.width,
            sps.height,
            sps.bit_depth_luma,
            if wide_gamut { " wide-gamut→709" } else { "" },
            sample_count,
            timescale,
            String::from_utf8_lossy(&info.fourcc),
            info.rotation,
        );

        let decoder = crate::nvdec_h265::H265Decoder::new(&vps, &sps, &pps)?;
        let reorder_depth = sps.max_num_reorder_pics as usize;
        Ok(Self {
            mp4,
            track_id: info.track_id,
            timescale,
            sample_count,
            next_sample: 0,
            sps,
            pps,
            decoder,
            length_size: info.length_size,
            wide_gamut,
            pending: Vec::new(),
            reorder_depth,
            pts_queue: std::collections::VecDeque::new(),
            rotation: info.rotation,
            t_read: 0.0,
            t_submit: 0.0,
            t_post: 0.0,
        })
    }

    /// (container read, submit, fence wait, post-process) seconds so far.
    pub fn decode_timing(&self) -> (f64, f64, f64, f64) {
        (self.t_read, self.t_submit, self.decoder.wait_s, self.t_post)
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Pop the earliest pending frame (presentation order).
    fn pop_earliest(&mut self) -> Option<DecodedFrame> {
        let idx = self
            .pending
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.pts.total_cmp(&b.pts))
            .map(|(i, _)| i)?;
        Some(self.pending.swap_remove(idx))
    }

    /// Retire one completed decode from the pipelined decoder into the
    /// reorder queue. Outputs arrive in submission order, so the front of
    /// `pts_queue` is always the retired frame's PTS.
    fn retire(&mut self, frame: crate::nvdec_h265::H265Frame) {
        let t = std::time::Instant::now();
        let pts = self.pts_queue.pop_front().expect("pts for retired frame");
        let decoded = apply_rotation(self.to_display(frame, pts), self.rotation);
        self.pending.push(decoded);
        self.t_post += t.elapsed().as_secs_f64();
    }

    pub fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        use crate::h265;
        loop {
            if self.next_sample >= self.sample_count {
                // Drain the decoder's in-flight ring, then the reorder queue.
                while let Some((frame, _poc)) =
                    self.decoder.drain().map_err(|e| VideoError::Decode {
                        sample: self.next_sample,
                        message: e.to_string(),
                    })?
                {
                    self.retire(frame);
                }
                return Ok(self.pop_earliest());
            }
            self.next_sample += 1;
            let sample_id = self.next_sample;
            let t = std::time::Instant::now();
            let sample = self.mp4.read_sample(self.track_id, sample_id)?;
            self.t_read += t.elapsed().as_secs_f64();
            let Some(sample) = sample else {
                continue; // fall through to EOS drain on the next iteration
            };
            let pts =
                (sample.start_time as f64 + sample.rendering_offset as f64) / self.timescale;

            // Split length-prefixed NALs; keep VCL only (hev1 may repeat
            // parameter sets in-band — the session already has them).
            let data = &sample.bytes[..];
            let mut slices: Vec<&[u8]> = Vec::new();
            let mut off = 0usize;
            while off + self.length_size <= data.len() {
                let mut len = 0usize;
                for k in 0..self.length_size {
                    len = (len << 8) | data[off + k] as usize;
                }
                off += self.length_size;
                if off + len > data.len() || len == 0 {
                    break;
                }
                let nal = &data[off..off + len];
                off += len;
                if h265::is_vcl(h265::nal_type(nal)) {
                    slices.push(nal);
                }
            }
            let Some(first) = slices.first() else {
                continue;
            };

            let header = h265::parse_slice_header(first, &self.sps, &self.pps).map_err(|e| {
                VideoError::Decode {
                    sample: sample_id,
                    message: format!("slice header: {e}"),
                }
            })?;
            // Pipelined submit: the returned frame (if any) is an OLDER
            // submission retiring; this sample's own frame surfaces a few
            // calls later.
            self.pts_queue.push_back(pts);
            let t = std::time::Instant::now();
            let retired =
                self.decoder.submit(&slices, &header).map_err(|e| VideoError::Decode {
                    sample: sample_id,
                    message: e.to_string(),
                })?;
            self.t_submit += t.elapsed().as_secs_f64();
            if let Some((frame, _poc)) = retired {
                self.retire(frame);
            }
            if self.pending.len() > self.reorder_depth {
                return Ok(self.pop_earliest());
            }
        }
    }

    /// Coded NV12/P010 → cropped 8-bit I420 at display size, with the
    /// wide-gamut fold applied for 10-bit streams.
    fn to_display(&self, frame: crate::nvdec_h265::H265Frame, pts: f64) -> DecodedFrame {
        let (w, h) = self.decoder.cropped_size();
        let (x0, y0) = self.decoder.crop_offsets();
        let stride = frame.stride as usize;
        let (wu, hu) = (w as usize, h as usize);
        let (x0u, y0u) = (x0 as usize, y0 as usize);
        let (cw, ch) = (wu / 2, hu / 2);
        let mut y = vec![0u8; wu * hu];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];

        // Row-parallel: this conversion runs on the decode thread and was
        // measured as (part of) the causal-pass wall on HEVC clips.
        use rayon::prelude::*;
        if frame.bit_depth == 8 {
            y.par_chunks_mut(wu).enumerate().for_each(|(row, dst)| {
                let src = (y0u + row) * stride + x0u;
                dst.copy_from_slice(&frame.y[src..src + wu]);
            });
            u.par_chunks_mut(cw)
                .zip(v.par_chunks_mut(cw))
                .enumerate()
                .for_each(|(row, (ur, vr))| {
                    let src = (y0u / 2 + row) * stride + (x0u & !1);
                    for col in 0..cw {
                        ur[col] = frame.uv[src + 2 * col];
                        vr[col] = frame.uv[src + 2 * col + 1];
                    }
                });
        } else {
            // 10-bit X6-packed (value in the top 10 bits of each u16 LE).
            let y16 = le_u16(&frame.y);
            let uv16 = le_u16(&frame.uv);
            let (y16, uv16) = (&y16[..], &uv16[..]);
            let sample = |plane: &[u16], idx: usize| -> u16 { plane[idx] >> 6 };
            let wide = self.wide_gamut;
            y.par_chunks_mut(wu).enumerate().for_each(|(row, dst)| {
                let ysrc = (y0u + row) * stride + x0u;
                let csrc = (y0u / 2 + row / 2) * stride + (x0u & !1);
                for (col, d) in dst.iter_mut().enumerate() {
                    let y10 = sample(y16, ysrc + col);
                    if wide {
                        let cb10 = sample(uv16, csrc + (col & !1));
                        let cr10 = sample(uv16, csrc + (col & !1) + 1);
                        let (y8, _, _) = crate::color::px2020_10_to_709_8(y10, cb10, cr10);
                        *d = y8;
                    } else {
                        *d = (y10 >> 2) as u8;
                    }
                }
            });
            u.par_chunks_mut(cw)
                .zip(v.par_chunks_mut(cw))
                .enumerate()
                .for_each(|(row, (ur, vr))| {
                    let csrc = (y0u / 2 + row) * stride + (x0u & !1);
                    // Luma sample co-sited with this chroma pair, for the fold.
                    let ysrc = (y0u + row * 2) * stride + x0u;
                    for col in 0..cw {
                        let cb10 = sample(uv16, csrc + 2 * col);
                        let cr10 = sample(uv16, csrc + 2 * col + 1);
                        if wide {
                            let y10 = sample(y16, ysrc + 2 * col);
                            let (_, cb8, cr8) =
                                crate::color::px2020_10_to_709_8(y10, cb10, cr10);
                            ur[col] = cb8;
                            vr[col] = cr8;
                        } else {
                            ur[col] = (cb10 >> 2) as u8;
                            vr[col] = (cr10 >> 2) as u8;
                        }
                    }
                });
        }
        DecodedFrame {
            pts,
            width: w,
            height: h,
            y,
            u,
            v,
        }
    }
}

/// Little-endian byte buffer → u16 samples (P010 readback). A copy — safe
/// against Vec<u8>'s alignment-1 guarantee and cheap next to the decode.
fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Display rotation in degrees clockwise from the tkhd matrix (16.16 fixed
/// point; phones store sensor-landscape pixels + this matrix). Anything that
/// isn't one of the four axis-aligned rotations maps to 0.
pub(crate) fn rotation_from_matrix(a: i32, b: i32, c: i32, d: i32) -> u32 {
    const ONE: i32 = 0x0001_0000;
    match (a, b, c, d) {
        (0, bb, cc, 0) if bb == ONE && cc == -ONE => 90,
        (aa, 0, 0, dd) if aa == -ONE && dd == -ONE => 180,
        (0, bb, cc, 0) if bb == -ONE && cc == ONE => 270,
        _ => 0,
    }
}

fn rotate_plane(src: &[u8], w: usize, h: usize, rot: u32) -> Vec<u8> {
    use rayon::prelude::*;
    let mut dst = vec![0u8; w * h];
    match rot {
        90 => {
            // Clockwise: (x, y) → (h−1−y, x) in an h×w image. Parallel over
            // destination rows (each reads one strided source column).
            dst.par_chunks_mut(h).enumerate().for_each(|(x, drow)| {
                for y in 0..h {
                    drow[h - 1 - y] = src[y * w + x];
                }
            });
        }
        180 => {
            for (d, s) in dst.iter_mut().zip(src.iter().rev()) {
                *d = *s;
            }
        }
        270 => {
            dst.par_chunks_mut(h).enumerate().for_each(|(i, drow)| {
                let x = w - 1 - i;
                for (y, d) in drow.iter_mut().enumerate() {
                    *d = src[y * w + x];
                }
            });
        }
        _ => dst.copy_from_slice(src),
    }
    dst
}

/// Bake the container's display rotation into the pixels so every consumer
/// (player, VO, training) sees upright frames — otherwise the whole splat
/// model reconstructs sideways (floor as a wall).
/// `SPLATTAR_NO_ROTATION=1` disables the bake (debug isolation).
fn apply_rotation(f: DecodedFrame, rot: u32) -> DecodedFrame {
    if rot == 0 || std::env::var_os("SPLATTAR_NO_ROTATION").is_some_and(|v| v != "0") {
        return f;
    }
    let (w, h) = (f.width as usize, f.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let y = rotate_plane(&f.y, w, h, rot);
    let u = rotate_plane(&f.u, cw, ch, rot);
    let v = rotate_plane(&f.v, cw, ch, rot);
    let (nw, nh) = if rot == 180 {
        (f.width, f.height)
    } else {
        (f.height, f.width)
    };
    DecodedFrame {
        pts: f.pts,
        width: nw,
        height: nh,
        y,
        u,
        v,
    }
}
