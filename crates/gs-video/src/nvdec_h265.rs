//! Hardware H.265/HEVC decode via Vulkan Video (`video_decode_h265`) — the
//! same NVDEC machinery as `nvdec.rs`, with the codec-specific parts swapped:
//! Main / Main 10 profiles (8- and 10-bit 4:2:0), RPS-driven reference
//! management (HEVC has no sliding window), and P010-style output for
//! 10-bit streams. The session boilerplate intentionally mirrors the H.264
//! decoder rather than sharing it — that path is golden-tested and stays
//! untouched; factoring a common core is a follow-up once both are proven.
//!
//! Same ash conventions (CLAUDE.md): raw `fns.fp().xxx_khr` calls, zeroed
//! `StdVideo*` structs with bitfield setters.

use ash::vk;
use ash::vk::native as stdvk;

use crate::h265::{H265Pps, H265SliceHeader, H265Sps, H265Vps, StRps, is_idr, is_irap, is_reference};
use crate::nvdec::NvDecError;

/// Decoded coded-size frame. 8-bit: 1 byte/sample. 10-bit: 2 bytes/sample,
/// little-endian, value in the TOP 10 bits (Vulkan X6-packed convention).
pub struct H265Frame {
    pub width: u32,
    pub height: u32,
    /// Stride in SAMPLES (multiply by bytes/sample for byte offsets).
    pub stride: u32,
    pub bit_depth: u8,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
}

struct DpbSlot {
    image: vk::Image,
    view: vk::ImageView,
    active: bool,
    poc: i32,
    age: u64,
}

pub struct H265Decoder {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    video_queue_fns: ash::khr::video_queue::Device,
    decode_queue_fns: ash::khr::video_decode_queue::Device,
    queue: vk::Queue,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,
    bitstream_align: u64,
    slots: Vec<DpbSlot>,
    coded_w: u32,
    coded_h: u32,
    crop: [u32; 4],
    bit_depth: u8,
    bitstream: vk::Buffer,
    bitstream_mem: vk::DeviceMemory,
    bitstream_ptr: *mut u8,
    bitstream_cap: usize,
    readback: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    reset_done: bool,
    age_counter: u64,
    prev_poc_lsb: i32,
    prev_poc_msb: i32,
    sps: H265Sps,
    pps: H265Pps,
}

fn profile_to_std(idc: u8) -> Result<u32, NvDecError> {
    Ok(match idc {
        1 => stdvk::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        2 => stdvk::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10,
        other => {
            return Err(NvDecError::Unsupported(format!(
                "HEVC profile_idc {other} (Main/Main10 only)"
            )));
        }
    })
}

fn level_to_std(level_idc: u8) -> u32 {
    match level_idc {
        0..=30 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
        31..=60 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0,
        61..=63 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1,
        64..=90 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
        91..=93 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
        94..=120 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
        121..=123 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
        124..=150 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0,
        151..=153 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        154..=156 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
        157..=180 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0,
        181..=183 => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1,
        _ => stdvk::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
    }
}

/// Emit a resolved RPS in explicit (non-inter-predicted) Std form.
fn std_st_rps(s: &StRps) -> stdvk::StdVideoH265ShortTermRefPicSet {
    let mut out: stdvk::StdVideoH265ShortTermRefPicSet = unsafe { std::mem::zeroed() };
    out.num_negative_pics = s.neg.len() as u8;
    out.num_positive_pics = s.pos.len() as u8;
    let mut prev = 0i32;
    for (j, &(d, used)) in s.neg.iter().enumerate() {
        out.delta_poc_s0_minus1[j] = (prev - d - 1) as u16;
        out.used_by_curr_pic_s0_flag |= (used as u16) << j;
        prev = d;
    }
    let mut prev = 0i32;
    for (j, &(d, used)) in s.pos.iter().enumerate() {
        out.delta_poc_s1_minus1[j] = (d - prev - 1) as u16;
        out.used_by_curr_pic_s1_flag |= (used as u16) << j;
        prev = d;
    }
    out
}

impl H265Decoder {
    pub fn new(vps: &H265Vps, sps: &H265Sps, pps: &H265Pps) -> Result<Self, NvDecError> {
        unsafe { Self::new_impl(vps, sps, pps) }
    }

    unsafe fn new_impl(
        vps: &H265Vps,
        sps: &H265Sps,
        pps: &H265Pps,
    ) -> Result<Self, NvDecError> {
        let entry = ash::Entry::load().map_err(|e| NvDecError::Loading(e.to_string()))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?;

        // Physical device + queue family with H.265 decode.
        let mut chosen = None;
        for pd in instance.enumerate_physical_devices()? {
            let count = instance.get_physical_device_queue_family_properties(pd).len();
            let mut video_props = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
            let mut props2: Vec<vk::QueueFamilyProperties2> = video_props
                .iter_mut()
                .map(|vp| vk::QueueFamilyProperties2::default().push_next(vp))
                .collect();
            instance.get_physical_device_queue_family_properties2(pd, &mut props2);
            let flags: Vec<vk::QueueFlags> = props2
                .iter()
                .map(|p| p.queue_family_properties.queue_flags)
                .collect();
            drop(props2);
            for (idx, (&f, vp)) in flags.iter().zip(&video_props).enumerate() {
                if f.contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                    && vp
                        .video_codec_operations
                        .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
                {
                    chosen = Some((pd, idx as u32, f));
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        let (pd, queue_family, qflags) = chosen.ok_or(NvDecError::NoDecodeQueue)?;
        if !qflags.contains(vk::QueueFlags::TRANSFER) {
            return Err(NvDecError::Missing("transfer ops on the decode queue".into()));
        }

        let exts = [
            ash::khr::video_queue::NAME.as_ptr(),
            ash::khr::video_decode_queue::NAME.as_ptr(),
            ash::khr::video_decode_h265::NAME.as_ptr(),
        ];
        let mut sync2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let prio = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&prio)];
        let device = instance.create_device(
            pd,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&exts)
                .push_next(&mut sync2),
            None,
        )?;
        let queue = device.get_device_queue(queue_family, 0);
        let video_instance_fns = ash::khr::video_queue::Instance::new(&entry, &instance);
        let video_queue_fns = ash::khr::video_queue::Device::new(&instance, &device);
        let decode_queue_fns = ash::khr::video_decode_queue::Device::new(&instance, &device);

        let bit_depth = sps.bit_depth_luma;
        let depth_flag = if bit_depth == 10 {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10
        } else {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8
        };
        let mut h265_profile = vk::VideoDecodeH265ProfileInfoKHR::default()
            .std_profile_idc(profile_to_std(sps.ptl.profile_idc)?);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(depth_flag)
            .chroma_bit_depth(depth_flag)
            .push_next(&mut h265_profile);

        let mut h265_caps = vk::VideoDecodeH265CapabilitiesKHR::default();
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let (bitstream_align, caps_max_dpb_slots) = {
            let mut caps = vk::VideoCapabilitiesKHR::default()
                .push_next(&mut decode_caps)
                .push_next(&mut h265_caps);
            (video_instance_fns.fp().get_physical_device_video_capabilities_khr)(
                pd, &profile, &mut caps,
            )
            .result()?;
            (caps.min_bitstream_buffer_size_alignment, caps.max_dpb_slots)
        };
        if !decode_caps
            .flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE)
        {
            return Err(NvDecError::Unsupported(
                "driver requires distinct DPB/output images (not yet implemented)".into(),
            ));
        }

        let coded_w = sps.width;
        let coded_h = sps.height;
        let max_refs = (sps.max_dec_pic_buffering as u32).clamp(1, 8);
        let max_slots = (max_refs + 1).min(caps_max_dpb_slots);

        let format = if bit_depth == 10 {
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
        } else {
            vk::Format::G8_B8R8_2PLANE_420_UNORM
        };
        let usage = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
            | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
            | vk::ImageUsageFlags::TRANSFER_SRC;
        {
            let profiles = [profile];
            let mut plist = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
            let fmt_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
                .image_usage(usage)
                .push_next(&mut plist);
            let get = video_instance_fns
                .fp()
                .get_physical_device_video_format_properties_khr;
            let mut count = 0u32;
            (get)(pd, &fmt_info, &mut count, std::ptr::null_mut()).result()?;
            let mut fmts = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
            (get)(pd, &fmt_info, &mut count, fmts.as_mut_ptr()).result()?;
            if !fmts.iter().any(|f| f.format == format) {
                return Err(NvDecError::Unsupported(format!(
                    "{format:?} not offered for HEVC decode (got {:?})",
                    fmts.iter().map(|f| f.format).collect::<Vec<_>>()
                )));
            }
        }

        // Session (Std header VK_STD_vulkan_video_codec_h265_decode 1.0.0).
        let std_version = {
            let mut v = vk::ExtensionProperties::default()
                .spec_version(vk::make_api_version(0, 1, 0, 0));
            let name = c"VK_STD_vulkan_video_codec_h265_decode";
            for (dst, src) in v.extension_name.iter_mut().zip(name.to_bytes_with_nul()) {
                *dst = *src as i8;
            }
            v
        };
        let session_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(queue_family)
            .video_profile(&profile)
            .picture_format(format)
            .max_coded_extent(vk::Extent2D {
                width: coded_w,
                height: coded_h,
            })
            .reference_picture_format(format)
            .max_dpb_slots(max_slots)
            .max_active_reference_pictures(max_refs)
            .std_header_version(&std_version);
        let session = {
            let mut s = vk::VideoSessionKHR::null();
            (video_queue_fns.fp().create_video_session_khr)(
                device.handle(),
                &session_info,
                std::ptr::null(),
                &mut s,
            )
            .result()?;
            s
        };

        // Bind session memory.
        let mem_reqs = {
            let get = video_queue_fns.fp().get_video_session_memory_requirements_khr;
            let mut count = 0u32;
            (get)(device.handle(), session, &mut count, std::ptr::null_mut()).result()?;
            let mut reqs =
                vec![vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
            (get)(device.handle(), session, &mut count, reqs.as_mut_ptr()).result()?;
            reqs
        };
        let mem_props = instance.get_physical_device_memory_properties(pd);
        let mut session_memory = Vec::new();
        let mut binds = Vec::new();
        for req in &mem_reqs {
            let type_idx = find_memory_type(
                &mem_props,
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type(
                    &mem_props,
                    req.memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
            .ok_or_else(|| NvDecError::Missing("session memory type".into()))?;
            let mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.memory_requirements.size)
                    .memory_type_index(type_idx),
                None,
            )?;
            binds.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(req.memory_bind_index)
                    .memory(mem)
                    .memory_size(req.memory_requirements.size),
            );
            session_memory.push(mem);
        }
        (video_queue_fns.fp().bind_video_session_memory_khr)(
            device.handle(),
            session,
            binds.len() as u32,
            binds.as_ptr(),
        )
        .result()?;

        // Session parameters: Std VPS + SPS + PPS. Sub-structs must stay
        // alive through the create call.
        let mut ptl: stdvk::StdVideoH265ProfileTierLevel = std::mem::zeroed();
        ptl.flags.set_general_tier_flag(u32::from(sps.ptl.tier_flag));
        ptl.flags
            .set_general_progressive_source_flag(u32::from(sps.ptl.progressive_source));
        ptl.flags
            .set_general_interlaced_source_flag(u32::from(sps.ptl.interlaced_source));
        ptl.flags
            .set_general_non_packed_constraint_flag(u32::from(sps.ptl.non_packed));
        ptl.flags
            .set_general_frame_only_constraint_flag(u32::from(sps.ptl.frame_only));
        ptl.general_profile_idc = profile_to_std(sps.ptl.profile_idc)?;
        ptl.general_level_idc = level_to_std(sps.ptl.level_idc);

        let mut dpbm: stdvk::StdVideoH265DecPicBufMgr = std::mem::zeroed();
        for i in 0..7 {
            dpbm.max_dec_pic_buffering_minus1[i] = sps.max_dec_pic_buffering - 1;
            dpbm.max_num_reorder_pics[i] = sps.max_num_reorder_pics;
        }

        let std_rps: Vec<stdvk::StdVideoH265ShortTermRefPicSet> =
            sps.st_rps.iter().map(std_st_rps).collect();

        let mut std_vps: stdvk::StdVideoH265VideoParameterSet = std::mem::zeroed();
        std_vps
            .flags
            .set_vps_temporal_id_nesting_flag(u32::from(vps.temporal_id_nesting));
        std_vps.vps_video_parameter_set_id = vps.vps_id;
        std_vps.vps_max_sub_layers_minus1 = vps.max_sub_layers_minus1;
        std_vps.pDecPicBufMgr = &dpbm;
        std_vps.pProfileTierLevel = &ptl;

        let mut std_sps: stdvk::StdVideoH265SequenceParameterSet = std::mem::zeroed();
        std_sps
            .flags
            .set_sps_temporal_id_nesting_flag(u32::from(sps.temporal_id_nesting));
        std_sps
            .flags
            .set_conformance_window_flag(u32::from(sps.crop.iter().any(|&c| c > 0)));
        std_sps.flags.set_sps_sub_layer_ordering_info_present_flag(1);
        std_sps
            .flags
            .set_scaling_list_enabled_flag(u32::from(sps.scaling_list_enabled));
        std_sps.flags.set_amp_enabled_flag(u32::from(sps.amp_enabled));
        std_sps
            .flags
            .set_sample_adaptive_offset_enabled_flag(u32::from(sps.sao_enabled));
        std_sps.flags.set_pcm_enabled_flag(u32::from(sps.pcm_enabled));
        std_sps
            .flags
            .set_sps_temporal_mvp_enabled_flag(u32::from(sps.temporal_mvp));
        std_sps
            .flags
            .set_strong_intra_smoothing_enabled_flag(u32::from(sps.strong_intra_smoothing));
        std_sps.chroma_format_idc = sps.chroma_format_idc;
        std_sps.pic_width_in_luma_samples = sps.width;
        std_sps.pic_height_in_luma_samples = sps.height;
        std_sps.sps_video_parameter_set_id = sps.vps_id;
        std_sps.sps_max_sub_layers_minus1 = sps.max_sub_layers_minus1;
        std_sps.sps_seq_parameter_set_id = sps.sps_id;
        std_sps.bit_depth_luma_minus8 = sps.bit_depth_luma - 8;
        std_sps.bit_depth_chroma_minus8 = sps.bit_depth_chroma - 8;
        std_sps.log2_max_pic_order_cnt_lsb_minus4 = (sps.log2_max_poc_lsb - 4) as u8;
        std_sps.log2_min_luma_coding_block_size_minus3 = sps.log2_min_cb_minus3;
        std_sps.log2_diff_max_min_luma_coding_block_size = sps.log2_diff_cb;
        std_sps.log2_min_luma_transform_block_size_minus2 = sps.log2_min_tb_minus2;
        std_sps.log2_diff_max_min_luma_transform_block_size = sps.log2_diff_tb;
        std_sps.max_transform_hierarchy_depth_inter = sps.max_th_depth_inter;
        std_sps.max_transform_hierarchy_depth_intra = sps.max_th_depth_intra;
        std_sps.num_short_term_ref_pic_sets = std_rps.len() as u8;
        // Conformance window in the Std struct is in chroma units (as coded).
        std_sps.conf_win_left_offset = sps.crop[0] / 2;
        std_sps.conf_win_right_offset = sps.crop[1] / 2;
        std_sps.conf_win_top_offset = sps.crop[2] / 2;
        std_sps.conf_win_bottom_offset = sps.crop[3] / 2;
        std_sps.pProfileTierLevel = &ptl;
        std_sps.pDecPicBufMgr = &dpbm;
        if !std_rps.is_empty() {
            std_sps.pShortTermRefPicSet = std_rps.as_ptr();
        }

        let mut std_pps: stdvk::StdVideoH265PictureParameterSet = std::mem::zeroed();
        let f = &mut std_pps.flags;
        f.set_dependent_slice_segments_enabled_flag(u32::from(pps.dependent_slice_segments));
        f.set_output_flag_present_flag(u32::from(pps.output_flag_present));
        f.set_sign_data_hiding_enabled_flag(u32::from(pps.sign_data_hiding));
        f.set_cabac_init_present_flag(u32::from(pps.cabac_init_present));
        f.set_constrained_intra_pred_flag(u32::from(pps.constrained_intra));
        f.set_transform_skip_enabled_flag(u32::from(pps.transform_skip));
        f.set_cu_qp_delta_enabled_flag(u32::from(pps.cu_qp_delta_enabled));
        f.set_pps_slice_chroma_qp_offsets_present_flag(
            u32::from(pps.slice_chroma_qp_offsets_present),
        );
        f.set_weighted_pred_flag(u32::from(pps.weighted_pred));
        f.set_weighted_bipred_flag(u32::from(pps.weighted_bipred));
        f.set_transquant_bypass_enabled_flag(u32::from(pps.transquant_bypass));
        f.set_tiles_enabled_flag(u32::from(pps.tiles_enabled));
        f.set_entropy_coding_sync_enabled_flag(u32::from(pps.entropy_coding_sync));
        f.set_uniform_spacing_flag(u32::from(pps.uniform_spacing));
        f.set_loop_filter_across_tiles_enabled_flag(u32::from(pps.loop_filter_across_tiles));
        f.set_pps_loop_filter_across_slices_enabled_flag(
            u32::from(pps.loop_filter_across_slices),
        );
        f.set_deblocking_filter_control_present_flag(u32::from(pps.deblocking_control_present));
        f.set_deblocking_filter_override_enabled_flag(
            u32::from(pps.deblocking_override_enabled),
        );
        f.set_pps_deblocking_filter_disabled_flag(u32::from(pps.deblocking_disabled));
        f.set_lists_modification_present_flag(u32::from(pps.lists_modification_present));
        f.set_slice_segment_header_extension_present_flag(
            u32::from(pps.slice_header_extension),
        );
        std_pps.pps_pic_parameter_set_id = pps.pps_id;
        std_pps.pps_seq_parameter_set_id = pps.sps_id;
        std_pps.sps_video_parameter_set_id = sps.vps_id;
        std_pps.num_extra_slice_header_bits = pps.num_extra_slice_header_bits;
        std_pps.num_ref_idx_l0_default_active_minus1 = (pps.num_ref_idx_l0_default - 1) as u8;
        std_pps.num_ref_idx_l1_default_active_minus1 = (pps.num_ref_idx_l1_default - 1) as u8;
        std_pps.init_qp_minus26 = pps.init_qp_minus26 as i8;
        std_pps.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth as u8;
        std_pps.pps_cb_qp_offset = pps.cb_qp_offset as i8;
        std_pps.pps_cr_qp_offset = pps.cr_qp_offset as i8;
        std_pps.pps_beta_offset_div2 = pps.beta_offset_div2 as i8;
        std_pps.pps_tc_offset_div2 = pps.tc_offset_div2 as i8;
        std_pps.log2_parallel_merge_level_minus2 =
            pps.log2_parallel_merge_level_minus2 as u8;
        std_pps.num_tile_columns_minus1 = pps.num_tile_columns_minus1 as u8;
        std_pps.num_tile_rows_minus1 = pps.num_tile_rows_minus1 as u8;
        for (i, &w) in pps.tile_col_widths_minus1.iter().enumerate().take(19) {
            std_pps.column_width_minus1[i] = w as u16;
        }
        for (i, &h) in pps.tile_row_heights_minus1.iter().enumerate().take(21) {
            std_pps.row_height_minus1[i] = h as u16;
        }

        let std_vps_arr = [std_vps];
        let std_sps_arr = [std_sps];
        let std_pps_arr = [std_pps];
        let mut h265_params_add = vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&std_vps_arr)
            .std_sp_ss(&std_sps_arr)
            .std_pp_ss(&std_pps_arr);
        let mut h265_params = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
            .max_std_vps_count(1)
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&h265_params_add);
        let params_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut h265_params);
        let session_params = {
            let mut p = vk::VideoSessionParametersKHR::null();
            (video_queue_fns.fp().create_video_session_parameters_khr)(
                device.handle(),
                &params_info,
                std::ptr::null(),
                &mut p,
            )
            .result()?;
            p
        };
        let _ = &mut h265_params_add;

        // DPB slot images (coincide mode).
        let profiles = [profile];
        let mut plist = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let mut slots = Vec::new();
        for _ in 0..max_slots {
            let image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: coded_w,
                        height: coded_h,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .push_next(&mut plist),
                None,
            )?;
            let req = device.get_image_memory_requirements(image);
            let type_idx = find_memory_type(
                &mem_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| NvDecError::Missing("image memory type".into()))?;
            let mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(type_idx),
                None,
            )?;
            device.bind_image_memory(image, mem, 0)?;
            session_memory.push(mem);
            let view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(full_range()),
                None,
            )?;
            slots.push(DpbSlot {
                image,
                view,
                active: false,
                poc: 0,
                age: 0,
            });
        }

        // Bitstream + readback buffers.
        let bytes_per_sample = if bit_depth == 10 { 2 } else { 1 } as usize;
        let bitstream_cap = 8 << 20;
        let (bitstream, bitstream_mem, bitstream_ptr) = {
            let mut plist = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bitstream_cap as u64)
                    .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut plist),
                None,
            )?;
            let req = device.get_buffer_memory_requirements(buffer);
            let type_idx = find_memory_type(
                &mem_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| NvDecError::Missing("host-visible bitstream memory".into()))?;
            let mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(type_idx),
                None,
            )?;
            device.bind_buffer_memory(buffer, mem, 0)?;
            let ptr =
                device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? as *mut u8;
            (buffer, mem, ptr)
        };
        let readback_size = (coded_w as usize * coded_h as usize * 3 / 2) * bytes_per_sample;
        let (readback, readback_mem) = {
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(readback_size as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = device.get_buffer_memory_requirements(buffer);
            let type_idx = find_memory_type(
                &mem_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| NvDecError::Missing("host-visible readback memory".into()))?;
            let mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(type_idx),
                None,
            )?;
            device.bind_buffer_memory(buffer, mem, 0)?;
            (buffer, mem)
        };

        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let cmd = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

        log::info!(
            "nvdec-h265: session ready — {}x{} coded, {}-bit, {} DPB slots, coincide mode",
            coded_w,
            coded_h,
            bit_depth,
            max_slots
        );

        Ok(Self {
            _entry: entry,
            instance,
            device,
            video_queue_fns,
            decode_queue_fns,
            queue,
            session,
            session_params,
            session_memory,
            bitstream_align,
            slots,
            coded_w,
            coded_h,
            crop: sps.crop,
            bit_depth,
            bitstream,
            bitstream_mem,
            bitstream_ptr,
            bitstream_cap,
            readback,
            readback_mem,
            cmd_pool,
            cmd,
            fence,
            reset_done: false,
            age_counter: 0,
            prev_poc_lsb: 0,
            prev_poc_msb: 0,
            sps: sps.clone(),
            pps: pps.clone(),
        })
    }

    /// PicOrderCntVal (8.3.1): msb wrap over poc_lsb; IDR resets to 0.
    /// prev state updates on reference pictures that are not RADL/RASL.
    fn compute_poc(&mut self, header: &H265SliceHeader) -> i32 {
        if is_idr(header.nal_type) {
            self.prev_poc_lsb = 0;
            self.prev_poc_msb = 0;
            return 0;
        }
        let max_lsb = 1i32 << self.sps.log2_max_poc_lsb;
        let lsb = header.poc_lsb as i32;
        let msb = if lsb < self.prev_poc_lsb && self.prev_poc_lsb - lsb >= max_lsb / 2 {
            self.prev_poc_msb + max_lsb
        } else if lsb > self.prev_poc_lsb && lsb - self.prev_poc_lsb > max_lsb / 2 {
            self.prev_poc_msb - max_lsb
        } else {
            self.prev_poc_msb
        };
        let is_radl_rasl = (6..=9).contains(&header.nal_type);
        if is_reference(header.nal_type) && !is_radl_rasl {
            self.prev_poc_lsb = lsb;
            self.prev_poc_msb = msb;
        }
        msb + lsb
    }

    /// Decode one access unit's VCL NALs (raw, no start codes). Returns the
    /// coded-size frame plus its PicOrderCntVal (for reorder bookkeeping).
    pub fn decode(
        &mut self,
        slices: &[&[u8]],
        header: &H265SliceHeader,
    ) -> Result<(H265Frame, i32), NvDecError> {
        unsafe { self.decode_impl(slices, header) }
    }

    unsafe fn decode_impl(
        &mut self,
        slices: &[&[u8]],
        header: &H265SliceHeader,
    ) -> Result<(H265Frame, i32), NvDecError> {
        let poc = self.compute_poc(header);
        if is_idr(header.nal_type) {
            for s in &mut self.slots {
                s.active = false;
            }
        }

        // RPS → the POCs this frame reads (curr) and keeps (curr + foll).
        let before: Vec<i32> = header
            .rps
            .neg
            .iter()
            .filter(|(_, used)| *used)
            .map(|(d, _)| poc + d)
            .collect();
        let after: Vec<i32> = header
            .rps
            .pos
            .iter()
            .filter(|(_, used)| *used)
            .map(|(d, _)| poc + d)
            .collect();
        let keep: Vec<i32> = header
            .rps
            .neg
            .iter()
            .chain(header.rps.pos.iter())
            .map(|(d, _)| poc + d)
            .collect();
        // HEVC DPB is RPS-driven: anything not in the current RPS is done.
        if !is_idr(header.nal_type) {
            for s in &mut self.slots {
                if s.active && !keep.contains(&s.poc) {
                    s.active = false;
                }
            }
        }
        let slot_of_poc = |slots: &[DpbSlot], p: i32| -> Option<usize> {
            slots.iter().position(|s| s.active && s.poc == p)
        };
        let mut ref_slot_indices: Vec<usize> = Vec::new();
        let mut before_slots = [0xffu8; 8];
        let mut after_slots = [0xffu8; 8];
        for (i, p) in before.iter().enumerate().take(8) {
            let idx = slot_of_poc(&self.slots, *p).ok_or_else(|| {
                NvDecError::Unsupported(format!("reference POC {p} not in DPB"))
            })?;
            before_slots[i] = idx as u8;
            if !ref_slot_indices.contains(&idx) {
                ref_slot_indices.push(idx);
            }
        }
        for (i, p) in after.iter().enumerate().take(8) {
            let idx = slot_of_poc(&self.slots, *p).ok_or_else(|| {
                NvDecError::Unsupported(format!("reference POC {p} not in DPB"))
            })?;
            after_slots[i] = idx as u8;
            if !ref_slot_indices.contains(&idx) {
                ref_slot_indices.push(idx);
            }
        }

        // Bitstream: Annex-B start code before each NAL.
        let mut offsets = Vec::with_capacity(slices.len());
        let mut cursor = 0usize;
        for nal in slices {
            offsets.push(cursor as u32);
            let total = 3 + nal.len();
            if cursor + total > self.bitstream_cap {
                return Err(NvDecError::Unsupported("bitstream buffer overflow".into()));
            }
            std::ptr::copy_nonoverlapping(
                [0u8, 0, 1].as_ptr(),
                self.bitstream_ptr.add(cursor),
                3,
            );
            std::ptr::copy_nonoverlapping(
                nal.as_ptr(),
                self.bitstream_ptr.add(cursor + 3),
                nal.len(),
            );
            cursor += total;
        }
        let range = (cursor as u64).next_multiple_of(self.bitstream_align.max(1));

        // Setup slot: inactive first, else evict oldest.
        let setup_idx = self
            .slots
            .iter()
            .enumerate()
            .position(|(i, s)| !s.active && !ref_slot_indices.contains(&i))
            .unwrap_or_else(|| {
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !ref_slot_indices.contains(i))
                    .min_by_key(|(_, s)| s.age)
                    .map(|(i, _)| i)
                    .unwrap()
            });

        let device = &self.device;
        device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            self.cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .dst_access_mask(
                vk::AccessFlags2::VIDEO_DECODE_READ_KHR | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
            .image(self.slots[setup_idx].image)
            .subresource_range(full_range());
        device.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier)),
        );

        // Stable per-slot arrays for the pNext chains.
        let pic_resources: Vec<vk::VideoPictureResourceInfoKHR> = self
            .slots
            .iter()
            .map(|s| {
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_extent(vk::Extent2D {
                        width: self.coded_w,
                        height: self.coded_h,
                    })
                    .image_view_binding(s.view)
            })
            .collect();
        let std_ref_infos: Vec<stdvk::StdVideoDecodeH265ReferenceInfo> = self
            .slots
            .iter()
            .map(|s| {
                let mut info: stdvk::StdVideoDecodeH265ReferenceInfo = std::mem::zeroed();
                info.PicOrderCntVal = s.poc;
                info
            })
            .collect();
        let h265_dpb_infos: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> = (0..self.slots.len())
            .map(|idx| {
                vk::VideoDecodeH265DpbSlotInfoKHR::default()
                    .std_reference_info(&std_ref_infos[idx])
            })
            .collect();

        // Begin-info: every active slot + the setup slot at −1.
        let active_indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| *i != setup_idx && s.active)
            .map(|(i, _)| i)
            .collect();
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for &idx in &active_indices {
            let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(idx as i32)
                .picture_resource(&pic_resources[idx]);
            slot.p_next = &h265_dpb_infos[idx] as *const _ as *const std::ffi::c_void;
            begin_slots.push(slot);
        }
        let setup_begin_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&pic_resources[setup_idx]);
        begin_slots.push(setup_begin_slot);

        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.session_params)
            .reference_slots(&begin_slots);
        (self.video_queue_fns.fp().cmd_begin_video_coding_khr)(self.cmd, &begin_info);

        if !self.reset_done {
            (self.video_queue_fns.fp().cmd_control_video_coding_khr)(
                self.cmd,
                &vk::VideoCodingControlInfoKHR::default()
                    .flags(vk::VideoCodingControlFlagsKHR::RESET),
            );
            self.reset_done = true;
        }

        let is_ref = is_reference(header.nal_type);
        let std_pic = {
            let mut pic: stdvk::StdVideoDecodeH265PictureInfo = std::mem::zeroed();
            pic.flags.set_IrapPicFlag(u32::from(is_irap(header.nal_type)));
            pic.flags.set_IdrPicFlag(u32::from(is_idr(header.nal_type)));
            pic.flags.set_IsReference(u32::from(is_ref));
            pic.flags.set_short_term_ref_pic_set_sps_flag(u32::from(
                header.num_bits_st_rps == 0,
            ));
            pic.sps_video_parameter_set_id = self.sps.vps_id;
            pic.pps_seq_parameter_set_id = self.sps.sps_id;
            pic.pps_pic_parameter_set_id = self.pps.pps_id;
            pic.NumDeltaPocsOfRefRpsIdx = header.num_delta_pocs_of_ref;
            pic.PicOrderCntVal = poc;
            pic.NumBitsForSTRefPicSetInSlice = header.num_bits_st_rps;
            pic.RefPicSetStCurrBefore = before_slots;
            pic.RefPicSetStCurrAfter = after_slots;
            pic.RefPicSetLtCurr = [0xff; 8];
            pic
        };
        let mut h265_pic = vk::VideoDecodeH265PictureInfoKHR::default()
            .std_picture_info(&std_pic)
            .slice_segment_offsets(&offsets);

        let setup_std_ref = {
            let mut info: stdvk::StdVideoDecodeH265ReferenceInfo = std::mem::zeroed();
            info.PicOrderCntVal = poc;
            info
        };
        let setup_h265_dpb =
            vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&setup_std_ref);
        let mut setup_slot_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup_idx as i32)
            .picture_resource(&pic_resources[setup_idx]);
        setup_slot_info.p_next = &setup_h265_dpb as *const _ as *const std::ffi::c_void;

        let mut decode_ref_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for &idx in &ref_slot_indices {
            let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(idx as i32)
                .picture_resource(&pic_resources[idx]);
            slot.p_next = &h265_dpb_infos[idx] as *const _ as *const std::ffi::c_void;
            decode_ref_slots.push(slot);
        }

        let mut decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(pic_resources[setup_idx])
            .setup_reference_slot(&setup_slot_info)
            .push_next(&mut h265_pic);
        if !decode_ref_slots.is_empty() {
            decode_info = decode_info.reference_slots(&decode_ref_slots);
        }
        (self.decode_queue_fns.fp().cmd_decode_video_khr)(self.cmd, &decode_info);

        (self.video_queue_fns.fp().cmd_end_video_coding_khr)(
            self.cmd,
            &vk::VideoEndCodingInfoKHR::default(),
        );

        // Copy out.
        let bytes_per_sample = if self.bit_depth == 10 { 2u32 } else { 1 };
        let to_src = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(self.slots[setup_idx].image)
            .subresource_range(full_range());
        device.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_src)),
        );
        let regions = [
            vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: self.coded_w,
                    height: self.coded_h,
                    depth: 1,
                },
            },
            vk::BufferImageCopy {
                buffer_offset: (self.coded_w * self.coded_h * bytes_per_sample) as u64,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: self.coded_w / 2,
                    height: self.coded_h / 2,
                    depth: 1,
                },
            },
        ];
        device.cmd_copy_image_to_buffer(
            self.cmd,
            self.slots[setup_idx].image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.readback,
            &regions,
        );
        let back_to_dpb = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .dst_access_mask(
                vk::AccessFlags2::VIDEO_DECODE_READ_KHR | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            )
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
            .image(self.slots[setup_idx].image)
            .subresource_range(full_range());
        device.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&back_to_dpb)),
        );

        device.end_command_buffer(self.cmd)?;
        let cmds = [self.cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        device.queue_submit(self.queue, &[submit], self.fence)?;
        device.wait_for_fences(&[self.fence], true, 10_000_000_000)?;
        device.reset_fences(&[self.fence])?;

        self.age_counter += 1;
        {
            let slot = &mut self.slots[setup_idx];
            slot.active = is_ref;
            slot.poc = poc;
            slot.age = self.age_counter;
        }

        let mapped = device.map_memory(
            self.readback_mem,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::empty(),
        )? as *const u8;
        let bps = bytes_per_sample as usize;
        let y_size = self.coded_w as usize * self.coded_h as usize * bps;
        let uv_size = y_size / 2;
        let mut y = vec![0u8; y_size];
        let mut uv = vec![0u8; uv_size];
        std::ptr::copy_nonoverlapping(mapped, y.as_mut_ptr(), y_size);
        std::ptr::copy_nonoverlapping(mapped.add(y_size), uv.as_mut_ptr(), uv_size);
        device.unmap_memory(self.readback_mem);

        Ok((
            H265Frame {
                width: self.coded_w,
                height: self.coded_h,
                stride: self.coded_w,
                bit_depth: self.bit_depth,
                y,
                uv,
            },
            poc,
        ))
    }

    /// Display size after the conformance window (luma pixels).
    pub fn cropped_size(&self) -> (u32, u32) {
        (
            self.coded_w - self.crop[0] - self.crop[1],
            self.coded_h - self.crop[2] - self.crop[3],
        )
    }

    pub fn crop_offsets(&self) -> (u32, u32) {
        (self.crop[0], self.crop[2])
    }
}

impl Drop for H265Decoder {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_buffer(self.readback, None);
            self.device.free_memory(self.readback_mem, None);
            self.device.unmap_memory(self.bitstream_mem);
            self.device.destroy_buffer(self.bitstream, None);
            self.device.free_memory(self.bitstream_mem, None);
            for slot in &self.slots {
                self.device.destroy_image_view(slot.view, None);
                self.device.destroy_image(slot.image, None);
            }
            (self.video_queue_fns.fp().destroy_video_session_parameters_khr)(
                self.device.handle(),
                self.session_params,
                std::ptr::null(),
            );
            (self.video_queue_fns.fp().destroy_video_session_khr)(
                self.device.handle(),
                self.session,
                std::ptr::null(),
            );
            for mem in &self.session_memory {
                self.device.free_memory(*mem, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn full_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(required)
    })
}
