//! H.265/HEVC bitstream header parsing — just enough to drive the Vulkan
//! Video (NVDEC) decode session: VPS/SPS/PPS from the container's hvcC
//! record and the first slice-segment header per access unit. The driver
//! parses everything below the slice header itself.
//!
//! Scope (iPhone + transcode footage): 4:2:0, 8/10-bit, I/P/B slices,
//! short-term RPS reference management. Rejected explicitly: separate colour
//! planes, scaling-list data, long-term reference pictures, dependent slice
//! segments (none appear in phone footage; failing loudly beats decoding
//! garbage).
//!
//! Reuses the H.264 `BitReader`/`to_rbsp` (identical RBSP + Exp-Golomb
//! coding); errors share the same shape via [`H265Error`].

use crate::h264::{BitReader, H264Error, to_rbsp};

pub type H265Error = H264Error;

// NAL unit types (Table 7-1). VCL types are 0..=31.
pub const NAL_BLA_W_LP: u8 = 16;
pub const NAL_IDR_W_RADL: u8 = 19;
pub const NAL_IDR_N_LP: u8 = 20;
pub const NAL_CRA: u8 = 21;
pub const NAL_RSV_IRAP_VCL23: u8 = 23;
pub const NAL_VPS: u8 = 32;
pub const NAL_SPS: u8 = 33;
pub const NAL_PPS: u8 = 34;

/// NAL type from the 2-byte HEVC NAL header.
pub fn nal_type(nal: &[u8]) -> u8 {
    (nal.first().copied().unwrap_or(0) >> 1) & 0x3f
}

pub fn is_vcl(t: u8) -> bool {
    t <= 31
}

pub fn is_irap(t: u8) -> bool {
    (NAL_BLA_W_LP..=NAL_RSV_IRAP_VCL23).contains(&t)
}

pub fn is_idr(t: u8) -> bool {
    t == NAL_IDR_W_RADL || t == NAL_IDR_N_LP
}

/// Sub-layer non-reference pictures have even NAL types below 16.
pub fn is_reference(t: u8) -> bool {
    t >= 16 || t % 2 == 1
}

/// profile_tier_level() — general layer only; sub-layer PTL parsed and
/// discarded (needed to keep the bit cursor honest).
#[derive(Debug, Clone, Default)]
pub struct Ptl {
    pub tier_flag: bool,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub progressive_source: bool,
    pub interlaced_source: bool,
    pub non_packed: bool,
    pub frame_only: bool,
    pub compat_flags: u32,
}

fn parse_ptl(r: &mut BitReader, max_sub_layers_minus1: u8) -> Result<Ptl, H265Error> {
    let _profile_space = r.u(2)?;
    let tier_flag = r.flag()?;
    let profile_idc = r.u(5)? as u8;
    let compat_flags = r.u(32)?;
    let progressive_source = r.flag()?;
    let interlaced_source = r.flag()?;
    let non_packed = r.flag()?;
    let frame_only = r.flag()?;
    // 43 reserved/constraint bits + 1 inbld/reserved bit.
    r.u(22)?;
    r.u(22)?;
    let level_idc = r.u(8)? as u8;
    let mut profile_present = [false; 8];
    let mut level_present = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        profile_present[i] = r.flag()?;
        level_present[i] = r.flag()?;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.u(2)?; // reserved_zero_2bits
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if profile_present[i] {
            r.u(32)?;
            r.u(32)?; // 2+1+5+32+4+... = first 64 of the 88-bit sub-layer profile
            r.u(24)?; // remaining 88−64 = 24
        }
        if level_present[i] {
            r.u(8)?;
        }
    }
    Ok(Ptl {
        tier_flag,
        profile_idc,
        level_idc,
        progressive_source,
        interlaced_source,
        non_packed,
        frame_only,
        compat_flags,
    })
}

/// A short-term reference picture set, RESOLVED to explicit deltas
/// (inter-set prediction is decompressed at parse time — semantically
/// identical, and lets the Std structs be emitted in explicit form).
#[derive(Debug, Clone, Default)]
pub struct StRps {
    /// (cumulative delta POC < 0, used_by_curr_pic), nearest first.
    pub neg: Vec<(i32, bool)>,
    /// (cumulative delta POC > 0, used_by_curr_pic), nearest first.
    pub pos: Vec<(i32, bool)>,
}

impl StRps {
    pub fn num_delta_pocs(&self) -> usize {
        self.neg.len() + self.pos.len()
    }
}

/// st_ref_pic_set(stRpsIdx). `sets` are the already-parsed sets (for
/// inter-set prediction); `slice_context` selects the delta_idx behavior
/// (present only when parsed from a slice header). Returns the resolved set
/// plus the index of the reference set when inter-predicted.
fn parse_st_rps(
    r: &mut BitReader,
    idx: usize,
    sets: &[StRps],
    slice_context: bool,
) -> Result<(StRps, Option<usize>), H265Error> {
    let inter = if idx != 0 { r.flag()? } else { false };
    if inter {
        let delta_idx_minus1 = if slice_context { r.ue()? as usize } else { 0 };
        let ref_idx = idx
            .checked_sub(delta_idx_minus1 + 1)
            .ok_or_else(|| H265Error::Malformed("RPS delta_idx out of range".into()))?;
        let rps_ref = sets
            .get(ref_idx)
            .ok_or_else(|| H265Error::Malformed("RPS reference index out of range".into()))?;
        let sign = r.flag()?;
        let abs_minus1 = r.ue()? as i32;
        let delta_rps = if sign { -(abs_minus1 + 1) } else { abs_minus1 + 1 };
        let n = rps_ref.num_delta_pocs();
        let mut used = vec![false; n + 1];
        let mut use_delta = vec![true; n + 1];
        for j in 0..=n {
            used[j] = r.flag()?;
            if !used[j] {
                use_delta[j] = r.flag()?;
            }
        }
        // Derivation per 7.4.8: build the new set from the reference set's
        // deltas shifted by delta_rps, keeping sorted nearest-first order.
        let mut out = StRps::default();
        let nn = rps_ref.neg.len();
        // Negative half: positive refs (descending), the delta itself, then
        // negative refs (ascending) — yields deltas in decreasing order
        // (i.e. nearest-to-current first).
        for j in (0..rps_ref.pos.len()).rev() {
            let d = rps_ref.pos[j].0 + delta_rps;
            if d < 0 && use_delta[nn + j] {
                out.neg.push((d, used[nn + j]));
            }
        }
        if delta_rps < 0 && use_delta[n] {
            out.neg.push((delta_rps, used[n]));
        }
        for j in 0..nn {
            let d = rps_ref.neg[j].0 + delta_rps;
            if d < 0 && use_delta[j] {
                out.neg.push((d, used[j]));
            }
        }
        // Positive half, mirrored.
        for j in (0..nn).rev() {
            let d = rps_ref.neg[j].0 + delta_rps;
            if d > 0 && use_delta[j] {
                out.pos.push((d, used[j]));
            }
        }
        if delta_rps > 0 && use_delta[n] {
            out.pos.push((delta_rps, used[n]));
        }
        for j in 0..rps_ref.pos.len() {
            let d = rps_ref.pos[j].0 + delta_rps;
            if d > 0 && use_delta[nn + j] {
                out.pos.push((d, used[nn + j]));
            }
        }
        Ok((out, Some(ref_idx)))
    } else {
        let num_neg = r.ue()? as usize;
        let num_pos = r.ue()? as usize;
        if num_neg > 16 || num_pos > 16 {
            return Err(H265Error::Malformed("RPS with >16 pics".into()));
        }
        let mut out = StRps::default();
        let mut d = 0i32;
        for _ in 0..num_neg {
            d -= r.ue()? as i32 + 1;
            out.neg.push((d, r.flag()?));
        }
        let mut d = 0i32;
        for _ in 0..num_pos {
            d += r.ue()? as i32 + 1;
            out.pos.push((d, r.flag()?));
        }
        Ok((out, None))
    }
}

#[derive(Debug, Clone)]
pub struct H265Sps {
    pub vps_id: u8,
    pub sps_id: u8,
    pub max_sub_layers_minus1: u8,
    pub temporal_id_nesting: bool,
    pub ptl: Ptl,
    pub chroma_format_idc: u32,
    /// Coded luma size (already MinCbSize-aligned by the encoder).
    pub width: u32,
    pub height: u32,
    /// Conformance window in LUMA pixels: left, right, top, bottom.
    pub crop: [u32; 4],
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub log2_max_poc_lsb: u32,
    pub max_dec_pic_buffering: u8,
    pub max_num_reorder_pics: u8,
    pub log2_min_cb_minus3: u8,
    pub log2_diff_cb: u8,
    pub log2_min_tb_minus2: u8,
    pub log2_diff_tb: u8,
    pub max_th_depth_inter: u8,
    pub max_th_depth_intra: u8,
    pub scaling_list_enabled: bool,
    pub amp_enabled: bool,
    pub sao_enabled: bool,
    pub pcm_enabled: bool,
    pub st_rps: Vec<StRps>,
    pub temporal_mvp: bool,
    pub strong_intra_smoothing: bool,
}

pub fn parse_sps(nal: &[u8]) -> Result<H265Sps, H265Error> {
    if nal.len() < 4 {
        return Err(H265Error::Malformed("SPS too short".into()));
    }
    let rbsp = to_rbsp(&nal[2..]); // 2-byte NAL header
    let mut r = BitReader::new(&rbsp);
    let vps_id = r.u(4)? as u8;
    let max_sub_layers_minus1 = r.u(3)? as u8;
    let temporal_id_nesting = r.flag()?;
    let ptl = parse_ptl(&mut r, max_sub_layers_minus1)?;
    let sps_id = r.ue()? as u8;
    let chroma_format_idc = r.ue()?;
    if chroma_format_idc != 1 {
        return Err(H265Error::Unsupported(format!(
            "chroma_format_idc {chroma_format_idc} (only 4:2:0)"
        )));
    }
    let width = r.ue()?;
    let height = r.ue()?;
    let mut crop = [0u32; 4];
    if r.flag()? {
        // Conformance window is in chroma units for 4:2:0 → ×2 for luma px.
        for c in &mut crop {
            *c = r.ue()? * 2;
        }
    }
    let bit_depth_luma = r.ue()? as u8 + 8;
    let bit_depth_chroma = r.ue()? as u8 + 8;
    if !matches!(bit_depth_luma, 8 | 10) || bit_depth_chroma != bit_depth_luma {
        return Err(H265Error::Unsupported(format!(
            "bit depth {bit_depth_luma}/{bit_depth_chroma} (8- or 10-bit only)"
        )));
    }
    let log2_max_poc_lsb = r.ue()? + 4;
    let sub_layer_ordering = r.flag()?;
    let start = if sub_layer_ordering {
        0
    } else {
        max_sub_layers_minus1
    };
    let mut max_dec_pic_buffering = 1u8;
    let mut max_num_reorder_pics = 0u8;
    for _ in start..=max_sub_layers_minus1 {
        max_dec_pic_buffering = r.ue()? as u8 + 1; // highest sub-layer wins
        max_num_reorder_pics = r.ue()? as u8;
        let _latency = r.ue()?;
    }
    let log2_min_cb_minus3 = r.ue()? as u8;
    let log2_diff_cb = r.ue()? as u8;
    let log2_min_tb_minus2 = r.ue()? as u8;
    let log2_diff_tb = r.ue()? as u8;
    let max_th_depth_inter = r.ue()? as u8;
    let max_th_depth_intra = r.ue()? as u8;
    let scaling_list_enabled = r.flag()?;
    if scaling_list_enabled && r.flag()? {
        return Err(H265Error::Unsupported("SPS scaling list data".into()));
    }
    let amp_enabled = r.flag()?;
    let sao_enabled = r.flag()?;
    let pcm_enabled = r.flag()?;
    if pcm_enabled {
        r.u(4)?; // pcm_sample_bit_depth_luma_minus1
        r.u(4)?; // pcm_sample_bit_depth_chroma_minus1
        r.ue()?; // log2_min_pcm_luma_coding_block_size_minus3
        r.ue()?; // log2_diff_max_min_pcm_luma_coding_block_size
        r.flag()?; // pcm_loop_filter_disabled_flag
    }
    let num_st_rps = r.ue()? as usize;
    if num_st_rps > 64 {
        return Err(H265Error::Malformed("num_short_term_ref_pic_sets > 64".into()));
    }
    let mut st_rps: Vec<StRps> = Vec::with_capacity(num_st_rps);
    for i in 0..num_st_rps {
        let (set, _) = parse_st_rps(&mut r, i, &st_rps, false)?;
        st_rps.push(set);
    }
    if r.flag()? {
        // long_term_ref_pics_present_flag
        let n = r.ue()?;
        if n > 0 {
            return Err(H265Error::Unsupported("long-term reference pictures".into()));
        }
    }
    let temporal_mvp = r.flag()?;
    let strong_intra_smoothing = r.flag()?;
    // vui_parameters / extensions: not needed.

    Ok(H265Sps {
        vps_id,
        sps_id,
        max_sub_layers_minus1,
        temporal_id_nesting,
        ptl,
        chroma_format_idc,
        width,
        height,
        crop,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_poc_lsb,
        max_dec_pic_buffering,
        max_num_reorder_pics,
        log2_min_cb_minus3,
        log2_diff_cb,
        log2_min_tb_minus2,
        log2_diff_tb,
        max_th_depth_inter,
        max_th_depth_intra,
        scaling_list_enabled,
        amp_enabled,
        sao_enabled,
        pcm_enabled,
        st_rps,
        temporal_mvp,
        strong_intra_smoothing,
    })
}

#[derive(Debug, Clone)]
pub struct H265Pps {
    pub pps_id: u8,
    pub sps_id: u8,
    pub dependent_slice_segments: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub init_qp_minus26: i32,
    pub constrained_intra: bool,
    pub transform_skip: bool,
    pub cu_qp_delta_enabled: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass: bool,
    pub tiles_enabled: bool,
    pub entropy_coding_sync: bool,
    pub loop_filter_across_tiles: bool,
    pub loop_filter_across_slices: bool,
    pub deblocking_control_present: bool,
    pub deblocking_override_enabled: bool,
    pub deblocking_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level_minus2: u32,
    pub slice_header_extension: bool,
}

pub fn parse_pps(nal: &[u8]) -> Result<H265Pps, H265Error> {
    if nal.len() < 3 {
        return Err(H265Error::Malformed("PPS too short".into()));
    }
    let rbsp = to_rbsp(&nal[2..]);
    let mut r = BitReader::new(&rbsp);
    let pps_id = r.ue()? as u8;
    let sps_id = r.ue()? as u8;
    let dependent_slice_segments = r.flag()?;
    let output_flag_present = r.flag()?;
    let num_extra_slice_header_bits = r.u(3)? as u8;
    let sign_data_hiding = r.flag()?;
    let cabac_init_present = r.flag()?;
    let num_ref_idx_l0_default = r.ue()? + 1;
    let num_ref_idx_l1_default = r.ue()? + 1;
    let init_qp_minus26 = r.se()?;
    let constrained_intra = r.flag()?;
    let transform_skip = r.flag()?;
    let cu_qp_delta_enabled = r.flag()?;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled { r.ue()? } else { 0 };
    let cb_qp_offset = r.se()?;
    let cr_qp_offset = r.se()?;
    let slice_chroma_qp_offsets_present = r.flag()?;
    let weighted_pred = r.flag()?;
    let weighted_bipred = r.flag()?;
    let transquant_bypass = r.flag()?;
    let tiles_enabled = r.flag()?;
    let entropy_coding_sync = r.flag()?;
    let mut loop_filter_across_tiles = true;
    if tiles_enabled {
        // Tile geometry is parsed by the driver; walk past it here.
        let cols = r.ue()?;
        let rows = r.ue()?;
        if cols > 0 || rows > 0 {
            return Err(H265Error::Unsupported("tiled streams".into()));
        }
        let uniform = r.flag()?;
        if !uniform {
            return Err(H265Error::Unsupported("non-uniform tiles".into()));
        }
        loop_filter_across_tiles = r.flag()?;
    }
    let loop_filter_across_slices = r.flag()?;
    let deblocking_control_present = r.flag()?;
    let mut deblocking_override_enabled = false;
    let mut deblocking_disabled = false;
    let mut beta_offset_div2 = 0;
    let mut tc_offset_div2 = 0;
    if deblocking_control_present {
        deblocking_override_enabled = r.flag()?;
        deblocking_disabled = r.flag()?;
        if !deblocking_disabled {
            beta_offset_div2 = r.se()?;
            tc_offset_div2 = r.se()?;
        }
    }
    if r.flag()? {
        return Err(H265Error::Unsupported("PPS scaling list data".into()));
    }
    let lists_modification_present = r.flag()?;
    let log2_parallel_merge_level_minus2 = r.ue()?;
    let slice_header_extension = r.flag()?;

    Ok(H265Pps {
        pps_id,
        sps_id,
        dependent_slice_segments,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding,
        cabac_init_present,
        num_ref_idx_l0_default,
        num_ref_idx_l1_default,
        init_qp_minus26,
        constrained_intra,
        transform_skip,
        cu_qp_delta_enabled,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets_present,
        weighted_pred,
        weighted_bipred,
        transquant_bypass,
        tiles_enabled,
        entropy_coding_sync,
        loop_filter_across_tiles,
        loop_filter_across_slices,
        deblocking_control_present,
        deblocking_override_enabled,
        deblocking_disabled,
        beta_offset_div2,
        tc_offset_div2,
        lists_modification_present,
        log2_parallel_merge_level_minus2,
        slice_header_extension,
    })
}

/// Minimal VPS: the decode session only needs its id and sub-layer count
/// (timing info is optional and absent from the Std struct requirements).
#[derive(Debug, Clone)]
pub struct H265Vps {
    pub vps_id: u8,
    pub max_sub_layers_minus1: u8,
    pub temporal_id_nesting: bool,
    pub ptl: Ptl,
}

pub fn parse_vps(nal: &[u8]) -> Result<H265Vps, H265Error> {
    if nal.len() < 4 {
        return Err(H265Error::Malformed("VPS too short".into()));
    }
    let rbsp = to_rbsp(&nal[2..]);
    let mut r = BitReader::new(&rbsp);
    let vps_id = r.u(4)? as u8;
    r.u(2)?; // vps_base_layer_internal/available (reserved in v1 spec)
    r.u(6)?; // vps_max_layers_minus1
    let max_sub_layers_minus1 = r.u(3)? as u8;
    let temporal_id_nesting = r.flag()?;
    r.u(16)?; // vps_reserved_0xffff_16bits
    let ptl = parse_ptl(&mut r, max_sub_layers_minus1)?;
    Ok(H265Vps {
        vps_id,
        max_sub_layers_minus1,
        temporal_id_nesting,
        ptl,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H265SliceType {
    B,
    P,
    I,
}

#[derive(Debug, Clone)]
pub struct H265SliceHeader {
    pub nal_type: u8,
    pub first_slice: bool,
    pub pps_id: u8,
    pub slice_type: H265SliceType,
    pub poc_lsb: u32,
    /// The RPS in effect for this picture (resolved), empty for IDR.
    pub rps: StRps,
    /// Bits consumed by an inline st_ref_pic_set (0 when taken from the SPS)
    /// — the driver re-parses it from the bitstream.
    pub num_bits_st_rps: u16,
    /// NumDeltaPocs of the reference set when the inline RPS was
    /// inter-predicted from an SPS set (0 otherwise).
    pub num_delta_pocs_of_ref: u8,
}

pub fn parse_slice_header(
    nal: &[u8],
    sps: &H265Sps,
    pps: &H265Pps,
) -> Result<H265SliceHeader, H265Error> {
    if nal.len() < 4 {
        return Err(H265Error::Malformed("slice NAL too short".into()));
    }
    let t = nal_type(nal);
    // De-emulate a bounded prefix — slice headers are small.
    let take = nal.len().min(320);
    let rbsp = to_rbsp(&nal[2..take]);
    let mut r = BitReader::new(&rbsp);

    let first_slice = r.flag()?;
    if is_irap(t) {
        r.flag()?; // no_output_of_prior_pics_flag
    }
    let pps_id = r.ue()? as u8;
    if !first_slice {
        if pps.dependent_slice_segments {
            return Err(H265Error::Unsupported("dependent slice segments".into()));
        }
        // slice_segment_address: ceil(log2(PicSizeInCtbsY)) bits.
        let log2_ctb =
            (sps.log2_min_cb_minus3 + 3 + sps.log2_diff_cb) as u32;
        let ctbs_x = sps.width.div_ceil(1 << log2_ctb);
        let ctbs_y = sps.height.div_ceil(1 << log2_ctb);
        let pic_size = (ctbs_x * ctbs_y).max(1);
        let bits = 32 - (pic_size - 1).leading_zeros().min(31);
        r.u(bits.max(1))?;
    }
    for _ in 0..pps.num_extra_slice_header_bits {
        r.flag()?;
    }
    let slice_type = match r.ue()? {
        0 => H265SliceType::B,
        1 => H265SliceType::P,
        2 => H265SliceType::I,
        v => return Err(H265Error::Malformed(format!("slice_type {v}"))),
    };
    if pps.output_flag_present {
        r.flag()?;
    }
    let mut poc_lsb = 0;
    let mut rps = StRps::default();
    let mut num_bits_st_rps = 0u16;
    let mut num_delta_pocs_of_ref = 0u8;
    if !is_idr(t) {
        poc_lsb = r.u(sps.log2_max_poc_lsb)?;
        let from_sps = r.flag()?;
        if !from_sps {
            let start = r.bit_pos();
            let (set, ref_idx) =
                parse_st_rps(&mut r, sps.st_rps.len(), &sps.st_rps, true)?;
            num_bits_st_rps = (r.bit_pos() - start) as u16;
            num_delta_pocs_of_ref = ref_idx
                .map(|i| sps.st_rps[i].num_delta_pocs() as u8)
                .unwrap_or(0);
            rps = set;
        } else if !sps.st_rps.is_empty() {
            let n = sps.st_rps.len() as u32;
            let idx = if n > 1 {
                let bits = 32 - (n - 1).leading_zeros();
                r.u(bits)? as usize
            } else {
                0
            };
            rps = sps
                .st_rps
                .get(idx)
                .cloned()
                .ok_or_else(|| H265Error::Malformed("st_rps idx out of range".into()))?;
        }
        // Long-term refs rejected at SPS parse; nothing further needed —
        // the driver parses the remainder of the header itself.
    }

    Ok(H265SliceHeader {
        nal_type: t,
        first_slice,
        pps_id,
        slice_type,
        poc_lsb,
        rps,
        num_bits_st_rps,
        num_delta_pocs_of_ref,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-writer for hand-building test bitstreams.
    struct Bw {
        bits: Vec<bool>,
    }
    impl Bw {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn u(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                self.bits.push((v >> i) & 1 == 1);
            }
        }
        fn ue(&mut self, v: u32) {
            let vp = v + 1;
            let n = 32 - vp.leading_zeros();
            self.u(0, n - 1);
            self.u(vp, n);
        }
        fn bytes(mut self) -> Vec<u8> {
            self.bits.push(true); // rbsp_stop_one_bit
            while self.bits.len() % 8 != 0 {
                self.bits.push(false);
            }
            self.bits
                .chunks(8)
                .map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | b as u8))
                .collect()
        }
    }

    #[test]
    fn explicit_rps_roundtrip() {
        // 2 negative (deltas −1, −3; used, unused), 1 positive (+2, used).
        let mut w = Bw::new();
        w.ue(2); // num_negative_pics
        w.ue(1); // num_positive_pics
        w.ue(0); // delta_poc_s0_minus1 → −1
        w.u(1, 1); // used
        w.ue(1); // delta_poc_s0_minus1 → −3
        w.u(0, 1); // not used
        w.ue(1); // delta_poc_s1_minus1 → +2
        w.u(1, 1); // used
        let bytes = w.bytes();
        let mut r = BitReader::new(&bytes);
        let (rps, ref_idx) = parse_st_rps(&mut r, 0, &[], false).unwrap();
        assert!(ref_idx.is_none());
        assert_eq!(rps.neg, vec![(-1, true), (-3, false)]);
        assert_eq!(rps.pos, vec![(2, true)]);
    }

    #[test]
    fn inter_predicted_rps_resolves() {
        // Reference set: neg = [(-1, used)], pos = [].
        let base = StRps {
            neg: vec![(-1, true)],
            pos: vec![],
        };
        // Inter-predict with deltaRps = −1: candidates are ref deltas −1
        // shifted → −2, plus the delta itself −1.
        let mut w = Bw::new();
        w.u(1, 1); // inter_ref_pic_set_prediction_flag
        w.u(1, 1); // delta_rps_sign (negative)
        w.ue(0); // abs_delta_rps_minus1 → |deltaRps| = 1
        // j = 0 (ref neg[0] → −2): used
        w.u(1, 1);
        // j = 1 (the deltaRps itself → −1): used
        w.u(1, 1);
        let bytes = w.bytes();
        let mut r = BitReader::new(&bytes);
        let (rps, ref_idx) = parse_st_rps(&mut r, 1, &[base], false).unwrap();
        assert_eq!(ref_idx, Some(0));
        // Nearest-first: −1 (deltaRps), then −2.
        assert_eq!(rps.neg, vec![(-1, true), (-2, true)]);
        assert!(rps.pos.is_empty());
    }

    #[test]
    fn nal_helpers() {
        // NAL header byte 0: type in bits 6..1.
        let idr = [(NAL_IDR_W_RADL << 1), 1u8];
        assert_eq!(nal_type(&idr), NAL_IDR_W_RADL);
        assert!(is_idr(NAL_IDR_W_RADL));
        assert!(is_irap(NAL_CRA));
        assert!(is_vcl(0));
        assert!(!is_vcl(NAL_SPS));
        assert!(is_reference(1));
        assert!(!is_reference(0));
        assert!(is_reference(NAL_IDR_N_LP));
    }
}
