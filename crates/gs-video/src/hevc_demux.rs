//! Minimal ISO-BMFF box walker for HEVC track configuration. The `mp4`
//! crate demuxes samples fine for any codec, but it neither recognizes the
//! `hvc1` sample entry (iPhone's default) nor parses `hvcC` beyond a stub —
//! so the HEVCDecoderConfigurationRecord (VPS/SPS/PPS + NAL length size) is
//! extracted here straight from the container bytes. Sample iteration stays
//! with the `mp4` crate.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::VideoError;

#[derive(Debug, Clone)]
pub struct HevcTrackInfo {
    pub track_id: u32,
    /// `hvc1` (parameter sets out-of-band) or `hev1` (may repeat in-band).
    pub fourcc: [u8; 4],
    /// Bytes of the NAL length prefix in samples (1, 2 or 4).
    pub length_size: usize,
    pub vps: Vec<Vec<u8>>,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
    /// nclx colour info when present: (primaries, transfer, matrix).
    pub colr: Option<(u16, u16, u16)>,
    /// Display rotation in degrees clockwise (tkhd matrix).
    pub rotation: u32,
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}
fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

/// Iterate child boxes of `data`, calling `f(fourcc, payload)`.
fn walk(data: &[u8], mut f: impl FnMut(&[u8; 4], &[u8])) {
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let size32 = be32(&data[off..]) as usize;
        let fourcc: [u8; 4] = data[off + 4..off + 8].try_into().unwrap();
        let (hdr, size) = if size32 == 1 {
            if off + 16 > data.len() {
                return;
            }
            let s = u64::from_be_bytes(data[off + 8..off + 16].try_into().unwrap()) as usize;
            (16, s)
        } else if size32 == 0 {
            (8, data.len() - off) // box extends to end
        } else {
            (8, size32)
        };
        if size < hdr || off + size > data.len() {
            return; // malformed; stop rather than misparse
        }
        f(&fourcc, &data[off + hdr..off + size]);
        off += size;
    }
}

fn find_child<'a>(data: &'a [u8], name: &[u8; 4]) -> Option<&'a [u8]> {
    let mut found = None;
    walk(data, |fourcc, payload| {
        if fourcc == name && found.is_none() {
            // Payload borrows from `data`; recover the slice via pointers.
            let start = payload.as_ptr() as usize - data.as_ptr() as usize;
            found = Some(start..start + payload.len());
        }
    });
    found.map(|r| &data[r])
}

/// (length_size, vps, sps, pps) from an HEVCDecoderConfigurationRecord.
type HvccRecord = (usize, Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Parse an HEVCDecoderConfigurationRecord (the `hvcC` payload).
fn parse_hvcc(rec: &[u8]) -> Option<HvccRecord> {
    if rec.len() < 23 {
        return None;
    }
    let length_size = (rec[21] & 3) as usize + 1;
    let num_arrays = rec[22] as usize;
    let (mut vps, mut sps, mut pps) = (Vec::new(), Vec::new(), Vec::new());
    let mut off = 23usize;
    for _ in 0..num_arrays {
        if off + 3 > rec.len() {
            return None;
        }
        let nal_type = rec[off] & 0x3f;
        let count = be16(&rec[off + 1..]) as usize;
        off += 3;
        for _ in 0..count {
            if off + 2 > rec.len() {
                return None;
            }
            let len = be16(&rec[off..]) as usize;
            off += 2;
            if off + len > rec.len() {
                return None;
            }
            let nal = rec[off..off + len].to_vec();
            off += len;
            match nal_type {
                32 => vps.push(nal),
                33 => sps.push(nal),
                34 => pps.push(nal),
                _ => {}
            }
        }
    }
    Some((length_size, vps, sps, pps))
}

/// Scan the container for an HEVC video track; None when there is none.
pub fn find_hevc_track(path: &Path) -> Result<Option<HevcTrackInfo>, VideoError> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(0))?;

    // Top level: find moov without reading mdat.
    let mut moov: Option<Vec<u8>> = None;
    let mut pos = 0u64;
    let mut hdr = [0u8; 16];
    while pos + 8 <= file_len {
        file.seek(SeekFrom::Start(pos))?;
        file.read_exact(&mut hdr[..8])?;
        let size32 = be32(&hdr) as u64;
        let fourcc: [u8; 4] = hdr[4..8].try_into().unwrap();
        let (hdr_len, size) = if size32 == 1 {
            file.read_exact(&mut hdr[8..16])?;
            (16u64, u64::from_be_bytes(hdr[8..16].try_into().unwrap()))
        } else if size32 == 0 {
            (8, file_len - pos)
        } else {
            (8, size32)
        };
        if size < hdr_len || pos + size > file_len {
            break;
        }
        if &fourcc == b"moov" {
            let payload = size - hdr_len;
            if payload > 512 << 20 {
                break; // sanity: moov over 512 MiB is not a phone video
            }
            let mut buf = vec![0u8; payload as usize];
            file.read_exact(&mut buf)?;
            moov = Some(buf);
            break;
        }
        pos += size;
    }
    let Some(moov) = moov else { return Ok(None) };

    let mut result = None;
    walk(&moov, |fourcc, trak| {
        if fourcc != b"trak" || result.is_some() {
            return;
        }
        let Some(tkhd) = find_child(trak, b"tkhd") else { return };
        // tkhd: version(1) flags(3), then v0: ctime(4) mtime(4) id(4);
        // v1: 8+8+4. The display matrix sits after duration + reserved.
        let (track_id, matrix_off) = match tkhd.first() {
            Some(0) if tkhd.len() >= 76 => (be32(&tkhd[12..]), 40),
            Some(1) if tkhd.len() >= 88 => (be32(&tkhd[20..]), 52),
            _ => return,
        };
        let m = |i: usize| be32(&tkhd[matrix_off + 4 * i..]) as i32;
        let rotation =
            crate::mp4_reader::rotation_from_matrix(m(0), m(1), m(3), m(4));
        let stsd = find_child(trak, b"mdia")
            .and_then(|m| find_child(m, b"minf"))
            .and_then(|m| find_child(m, b"stbl"))
            .and_then(|s| find_child(s, b"stsd"));
        let Some(stsd) = stsd else { return };
        if stsd.len() < 8 {
            return;
        }
        // stsd: version/flags(4) + entry_count(4), then sample entries.
        walk(&stsd[8..], |entry_fourcc, entry| {
            if result.is_some() || !(entry_fourcc == b"hvc1" || entry_fourcc == b"hev1") {
                return;
            }
            // VisualSampleEntry: 6 reserved + 2 data_ref_index + 70 bytes of
            // video fields, then child boxes (hvcC, colr, ...).
            if entry.len() < 78 {
                return;
            }
            let children = &entry[78..];
            let Some(hvcc) = find_child(children, b"hvcC") else { return };
            let Some((length_size, vps, sps, pps)) = parse_hvcc(hvcc) else { return };
            let colr = find_child(children, b"colr").and_then(|c| {
                (c.len() >= 10 && &c[0..4] == b"nclx")
                    .then(|| (be16(&c[4..]), be16(&c[6..]), be16(&c[8..])))
            });
            result = Some(HevcTrackInfo {
                track_id,
                fourcc: *entry_fourcc,
                length_size,
                vps,
                sps,
                pps,
                colr,
                rotation,
            });
        });
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hvcc_record_parses() {
        // Minimal record: 22 bytes of config (lengthSizeMinusOne=3 at byte
        // 21), one array of one SPS NAL [0x42, 0x01].
        let mut rec = vec![0u8; 22];
        rec[0] = 1; // configurationVersion
        rec[21] = 0xfc | 3; // reserved | lengthSizeMinusOne
        rec.push(1); // numOfArrays
        rec.push(0x80 | 33); // array_completeness | NAL type SPS
        rec.extend_from_slice(&1u16.to_be_bytes()); // numNalus
        rec.extend_from_slice(&2u16.to_be_bytes()); // nalUnitLength
        rec.extend_from_slice(&[0x42, 0x01]);
        let (len_size, vps, sps, pps) = parse_hvcc(&rec).unwrap();
        assert_eq!(len_size, 4);
        assert!(vps.is_empty() && pps.is_empty());
        assert_eq!(sps, vec![vec![0x42, 0x01]]);
    }

    #[test]
    fn box_walker_finds_nested() {
        // [outer [inner "payload"]]
        let inner_payload = b"payload";
        let mut inner = Vec::new();
        inner.extend_from_slice(&(8 + inner_payload.len() as u32).to_be_bytes());
        inner.extend_from_slice(b"innr");
        inner.extend_from_slice(inner_payload);
        let mut outer = Vec::new();
        outer.extend_from_slice(&(8 + inner.len() as u32).to_be_bytes());
        outer.extend_from_slice(b"outr");
        outer.extend_from_slice(&inner);
        let out = find_child(&outer, b"outr").unwrap();
        let inn = find_child(out, b"innr").unwrap();
        assert_eq!(inn, inner_payload);
    }
}
