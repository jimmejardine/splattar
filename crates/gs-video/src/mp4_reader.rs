//! MP4 (ISO BMFF) demux + NVDEC (Vulkan Video) H.264 decode. Samples are
//! access units in decode order; PTS comes from sample start time +
//! composition offset over the track timescale, so decoded frame k pairs with
//! sample k's PTS without POC bookkeeping (phone footage: no B-frames). AVCC
//! length-prefixed NALs are split and slice NALs are handed to the decoder.

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

pub struct Mp4H264Reader {
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
}

impl Mp4H264Reader {
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
        log::info!(
            "h264 track {track_id}: {}x{}, {} samples, timescale {}",
            track.width(),
            track.height(),
            sample_count,
            timescale
        );

        let decoder = NvDecoder::new(&sps, &pps)?;

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
        })
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Decode the next access unit. Returns None at end of stream.
    pub fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        loop {
            if self.next_sample >= self.sample_count {
                return Ok(None);
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

            return Ok(Some(DecodedFrame {
                pts,
                width: w,
                height: h,
                y,
                u,
                v,
            }));
        }
    }
}
