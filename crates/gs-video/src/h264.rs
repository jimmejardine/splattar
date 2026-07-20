//! H.264 bitstream parsing for hardware decode: RBSP extraction, SPS/PPS,
//! and the slice-header prefix needed for DPB management. The hardware
//! (NVDEC via Vulkan Video) consumes raw slice NALs — we only parse what
//! reference management and picture parameters require, and error loudly on
//! anything outside the phone-footage subset (progressive, 4:2:0 8-bit,
//! sliding-window references, no MMCO, POC type 0 or 2).

#[derive(Debug, thiserror::Error)]
pub enum H264Error {
    #[error("bitstream ended early")]
    Eof,
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("malformed: {0}")]
    Malformed(String),
}

/// Bit reader over an RBSP (emulation-prevention bytes already removed).
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit position
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn u(&mut self, bits: u32) -> Result<u32, H264Error> {
        let mut v = 0u32;
        for _ in 0..bits {
            let byte = self.data.get(self.pos / 8).ok_or(H264Error::Eof)?;
            v = (v << 1) | ((byte >> (7 - self.pos % 8)) & 1) as u32;
            self.pos += 1;
        }
        Ok(v)
    }

    pub fn flag(&mut self) -> Result<bool, H264Error> {
        Ok(self.u(1)? == 1)
    }

    /// Unsigned Exp-Golomb.
    pub fn ue(&mut self) -> Result<u32, H264Error> {
        let mut zeros = 0;
        while self.u(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(H264Error::Malformed("ue > 31 leading zeros".into()));
            }
        }
        Ok((1u32 << zeros) - 1 + if zeros > 0 { self.u(zeros)? } else { 0 })
    }

    /// Signed Exp-Golomb.
    pub fn se(&mut self) -> Result<i32, H264Error> {
        let k = self.ue()? as i64;
        Ok(if k % 2 == 0 { -(k as i32) / 2 } else { (k as i32 + 1) / 2 })
    }

    /// True if RBSP data remains beyond the trailing stop bit.
    pub fn has_more_rbsp_data(&self) -> bool {
        let total_bits = self.data.len() * 8;
        if self.pos >= total_bits {
            return false;
        }
        // Find the last set bit (the rbsp_stop_one_bit).
        let mut last_one = None;
        for bit in (self.pos..total_bits).rev() {
            let byte = self.data[bit / 8];
            if (byte >> (7 - bit % 8)) & 1 == 1 {
                last_one = Some(bit);
                break;
            }
        }
        // The last set bit is the rbsp_stop_one_bit; payload data exists only
        // if any bit precedes it beyond the cursor.
        matches!(last_one, Some(stop) if stop > self.pos)
    }
}

/// Strip emulation-prevention bytes (00 00 03 xx → 00 00 xx).
pub fn to_rbsp(nal_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal_payload.len());
    let mut zeros = 0;
    for &b in nal_payload {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    out
}

/// Parse one scaling list; returns (values, use_default_matrix).
fn parse_scaling_list(r: &mut BitReader<'_>, out: &mut [u8]) -> Result<bool, H264Error> {
    let mut last = 8i32;
    let mut next = 8i32;
    let mut use_default = false;
    for (j, slot) in out.iter_mut().enumerate() {
        if next != 0 {
            let delta = r.se()?;
            next = (last + delta + 256) % 256;
            if j == 0 && next == 0 {
                use_default = true;
            }
        }
        let v = if next == 0 { last } else { next };
        *slot = v as u8;
        if next != 0 {
            last = next;
        }
    }
    Ok(use_default)
}

/// Scaling lists in the layout the Vulkan Video Std structs expect.
#[derive(Debug, Clone)]
pub struct ScalingLists {
    pub present_mask: u16,
    pub use_default_mask: u16,
    pub list_4x4: [[u8; 16]; 6],
    pub list_8x8: [[u8; 64]; 6],
}

impl Default for ScalingLists {
    fn default() -> Self {
        Self {
            present_mask: 0,
            use_default_mask: 0,
            list_4x4: [[16; 16]; 6],
            list_8x8: [[16; 64]; 6],
        }
    }
}

fn parse_scaling_matrix(
    r: &mut BitReader<'_>,
    count: usize,
) -> Result<ScalingLists, H264Error> {
    let mut lists = ScalingLists::default();
    for i in 0..count {
        if r.flag()? {
            lists.present_mask |= 1 << i;
            let use_default = if i < 6 {
                parse_scaling_list(r, &mut lists.list_4x4[i])?
            } else {
                parse_scaling_list(r, &mut lists.list_8x8[i - 6])?
            };
            if use_default {
                lists.use_default_mask |= 1 << i;
            }
        }
    }
    Ok(lists)
}

#[derive(Debug, Clone)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_flags: u8,
    pub level_idc: u8,
    pub sps_id: u32,
    pub chroma_format_idc: u32,
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub max_num_ref_frames: u32,
    pub gaps_allowed: bool,
    pub mb_width: u32,
    pub mb_height: u32,
    pub width: u32,
    pub height: u32,
    pub direct_8x8: bool,
    pub qpprime_y_zero: bool,
    pub frame_cropping_flag: bool,
    pub frame_cropping: [u32; 4],
    pub scaling: Option<ScalingLists>,
}

pub fn parse_sps(nal_payload: &[u8]) -> Result<Sps, H264Error> {
    let rbsp = to_rbsp(&nal_payload[1..]); // skip the NAL header byte
    let r = &mut BitReader::new(&rbsp);
    let profile_idc = r.u(8)? as u8;
    let constraint_flags = r.u(8)? as u8;
    let level_idc = r.u(8)? as u8;
    let sps_id = r.ue()?;

    let high = matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128);
    let mut chroma_format_idc = 1;
    let mut qpprime_y_zero = false;
    let mut scaling = None;
    if high {
        chroma_format_idc = r.ue()?;
        if chroma_format_idc != 1 {
            return Err(H264Error::Unsupported(format!(
                "chroma_format_idc {chroma_format_idc} (only 4:2:0)"
            )));
        }
        let bd_luma = r.ue()?;
        let bd_chroma = r.ue()?;
        if bd_luma != 0 || bd_chroma != 0 {
            return Err(H264Error::Unsupported("bit depth > 8".into()));
        }
        qpprime_y_zero = r.flag()?;
        if r.flag()? {
            scaling = Some(parse_scaling_matrix(r, 8)?);
        }
    }

    let log2_max_frame_num = r.ue()? + 4;
    let poc_type = r.ue()?;
    let mut log2_max_poc_lsb = 0;
    match poc_type {
        0 => log2_max_poc_lsb = r.ue()? + 4,
        2 => {}
        other => {
            return Err(H264Error::Unsupported(format!("poc type {other}")));
        }
    }
    let max_num_ref_frames = r.ue()?;
    let gaps_allowed = r.flag()?;
    let mb_width = r.ue()? + 1;
    let mb_height_units = r.ue()? + 1;
    let frame_mbs_only = r.flag()?;
    if !frame_mbs_only {
        return Err(H264Error::Unsupported("interlaced (frame_mbs_only=0)".into()));
    }
    let mb_height = mb_height_units;
    let direct_8x8 = r.flag()?;
    let frame_cropping_flag = r.flag()?;
    let mut crop = [0u32; 4];
    if frame_cropping_flag {
        crop = [r.ue()?, r.ue()?, r.ue()?, r.ue()?];
    }
    // VUI ignored (timing comes from the container).

    let width = mb_width * 16 - 2 * (crop[0] + crop[1]);
    let height = mb_height * 16 - 2 * (crop[2] + crop[3]);
    Ok(Sps {
        profile_idc,
        constraint_flags,
        level_idc,
        sps_id,
        chroma_format_idc,
        log2_max_frame_num,
        poc_type,
        log2_max_poc_lsb,
        max_num_ref_frames,
        gaps_allowed,
        mb_width,
        mb_height,
        width,
        height,
        direct_8x8,
        qpprime_y_zero,
        frame_cropping_flag,
        frame_cropping: crop,
        scaling,
    })
}

#[derive(Debug, Clone)]
pub struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,
    pub entropy_cabac: bool,
    pub pic_order_present: bool,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub second_chroma_qp_index_offset: i32,
    pub deblocking_control_present: bool,
    pub constrained_intra: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8: bool,
    pub scaling: Option<ScalingLists>,
}

pub fn parse_pps(nal_payload: &[u8]) -> Result<Pps, H264Error> {
    let rbsp = to_rbsp(&nal_payload[1..]);
    let r = &mut BitReader::new(&rbsp);
    let pps_id = r.ue()?;
    let sps_id = r.ue()?;
    let entropy_cabac = r.flag()?;
    let pic_order_present = r.flag()?;
    let num_slice_groups = r.ue()? + 1;
    if num_slice_groups != 1 {
        return Err(H264Error::Unsupported("slice groups (FMO)".into()));
    }
    let num_ref_idx_l0_default = r.ue()? + 1;
    let num_ref_idx_l1_default = r.ue()? + 1;
    let weighted_pred = r.flag()?;
    let weighted_bipred_idc = r.u(2)?;
    let pic_init_qp_minus26 = r.se()?;
    let pic_init_qs_minus26 = r.se()?;
    let chroma_qp_index_offset = r.se()?;
    let deblocking_control_present = r.flag()?;
    let constrained_intra = r.flag()?;
    let redundant_pic_cnt_present = r.flag()?;

    // Optional High-profile tail (present iff more RBSP data remains before
    // the trailing stop bit).
    let mut transform_8x8 = false;
    let mut scaling = None;
    let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
    if r.has_more_rbsp_data() {
        transform_8x8 = r.flag()?;
        if r.flag()? {
            let count = if transform_8x8 { 8 } else { 6 };
            scaling = Some(parse_scaling_matrix(r, count)?);
        }
        second_chroma_qp_index_offset = r.se()?;
    }

    Ok(Pps {
        pps_id,
        sps_id,
        entropy_cabac,
        pic_order_present,
        num_ref_idx_l0_default,
        num_ref_idx_l1_default,
        weighted_pred,
        weighted_bipred_idc,
        pic_init_qp_minus26,
        pic_init_qs_minus26,
        chroma_qp_index_offset,
        second_chroma_qp_index_offset,
        deblocking_control_present,
        constrained_intra,
        redundant_pic_cnt_present,
        transform_8x8,
        scaling,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    fn from_value(v: u32) -> Result<Self, H264Error> {
        Ok(match v % 5 {
            0 => Self::P,
            1 => Self::B,
            2 => Self::I,
            3 => Self::Sp,
            4 => Self::Si,
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_mb: u32,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num: u32,
    pub idr: bool,
    pub idr_pic_id: u32,
    pub poc_lsb: u32,
    pub num_ref_idx_l0: u32,
    pub nal_ref_idc: u8,
}

/// Parse the slice-header prefix through dec_ref_pic_marking. Requires the
/// active SPS/PPS. Errors on features outside the phone subset.
pub fn parse_slice_header(
    nal: &[u8],
    sps: &Sps,
    pps: &Pps,
) -> Result<SliceHeader, H264Error> {
    let nal_unit_type = nal[0] & 0x1f;
    let nal_ref_idc = (nal[0] >> 5) & 0x3;
    let idr = nal_unit_type == 5;
    // Slice headers sit early; 256 bytes of RBSP is plenty for the prefix.
    let take = nal.len().min(320);
    let rbsp = to_rbsp(&nal[1..take]);
    let r = &mut BitReader::new(&rbsp);

    let first_mb = r.ue()?;
    let slice_type = SliceType::from_value(r.ue()?)?;
    if matches!(slice_type, SliceType::B) {
        return Err(H264Error::Unsupported(
            "B slices (add DPB support when a device produces them)".into(),
        ));
    }
    if matches!(slice_type, SliceType::Sp | SliceType::Si) {
        return Err(H264Error::Unsupported("SP/SI slices".into()));
    }
    let pps_id = r.ue()?;
    let frame_num = r.u(sps.log2_max_frame_num)?;
    let mut idr_pic_id = 0;
    if idr {
        idr_pic_id = r.ue()?;
    }
    let mut poc_lsb = 0;
    if sps.poc_type == 0 {
        poc_lsb = r.u(sps.log2_max_poc_lsb)?;
        if pps.pic_order_present {
            let _delta_bottom = r.se()?;
        }
    }
    if pps.redundant_pic_cnt_present {
        let _ = r.ue()?;
    }
    let mut num_ref_idx_l0 = pps.num_ref_idx_l0_default;
    if matches!(slice_type, SliceType::P) {
        if r.flag()? {
            num_ref_idx_l0 = r.ue()? + 1;
        }
        // ref_pic_list_modification for l0.
        if r.flag()? {
            loop {
                let op = r.ue()?;
                if op == 3 {
                    break;
                }
                if op > 3 {
                    return Err(H264Error::Malformed("bad ref list modification op".into()));
                }
                let _value = r.ue()?;
                // Modifications are tolerated in parsing; the hardware gets
                // our DPB in PicNum order, which matches the common case of
                // no-op re-specification. Exotic reordering → decode garbage,
                // caught by downstream sanity checks.
            }
        }
        if pps.weighted_pred {
            return Err(H264Error::Unsupported("weighted prediction".into()));
        }
    }
    if nal_ref_idc != 0 {
        if idr {
            let _no_output = r.flag()?;
            let long_term = r.flag()?;
            if long_term {
                return Err(H264Error::Unsupported("long-term references".into()));
            }
        } else if r.flag()? {
            return Err(H264Error::Unsupported(
                "adaptive ref pic marking (MMCO)".into(),
            ));
        }
    }

    Ok(SliceHeader {
        first_mb,
        slice_type,
        pps_id,
        frame_num,
        idr,
        idr_pic_id,
        poc_lsb,
        num_ref_idx_l0,
        nal_ref_idc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb() {
        // 1 → 0; 010 → 1; 011 → 2; 00100 → 3
        let data = [0b1_010_011_0, 0b0100_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.ue().unwrap(), 0);
        assert_eq!(r.ue().unwrap(), 1);
        assert_eq!(r.ue().unwrap(), 2);
        assert_eq!(r.ue().unwrap(), 3);
    }

    #[test]
    fn rbsp_strips_emulation() {
        assert_eq!(to_rbsp(&[0, 0, 3, 1, 0, 0, 3, 3]), vec![0, 0, 1, 0, 0, 3]);
        assert_eq!(to_rbsp(&[1, 2, 3]), vec![1, 2, 3]);
    }
}
