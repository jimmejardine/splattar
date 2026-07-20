//! Hardware H.264 decode via Vulkan Video (NVDEC on the dev machine).
//! Pure-Rust bindings (ash); the driver is a system runtime — no third-party
//! codec code, no licensing entanglement (see the M5 bake-off notes).
//!
//! Scope: the phone-footage subset — progressive 4:2:0 8-bit, I/P slices,
//! sliding-window references, POC type 0/2. The decoder owns its own Vulkan
//! device (decode output goes to the CPU for tracking anyway).
//!
//! DPB strategy: "coincide" mode when available (decode straight into the DPB
//! slot image, copy that image out), else distinct output image.

use std::ffi::CStr;

use ash::vk;
use ash::vk::native as stdvk;

use crate::h264::{Pps, ScalingLists, SliceHeader, SliceType, Sps};

pub struct Nv12Frame {
    pub width: u32,
    pub height: u32,
    /// Luma plane, stride = padded width.
    pub y: Vec<u8>,
    /// Interleaved UV plane at half resolution, stride = padded width.
    pub uv: Vec<u8>,
    pub stride: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum NvDecError {
    #[error("vulkan: {0}")]
    Vk(#[from] vk::Result),
    #[error("vulkan loading failed: {0}")]
    Loading(String),
    #[error("no video-decode-capable device/queue found")]
    NoDecodeQueue,
    #[error("driver lacks {0}")]
    Missing(String),
    #[error("unsupported stream: {0}")]
    Unsupported(String),
}

struct DpbSlot {
    image: vk::Image,
    view: vk::ImageView,
    /// Currently holds a reference picture?
    active: bool,
    frame_num: u32,
    poc: i32,
    /// Monotonic age for sliding-window eviction.
    age: u64,
}

pub struct NvDecoder {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    video_queue_fns: ash::khr::video_queue::Device,
    decode_queue_fns: ash::khr::video_decode_queue::Device,
    queue: vk::Queue,
    queue_family: u32,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,
    profile_align: ProfileAlign,
    slots: Vec<DpbSlot>,
    coded_w: u32,
    coded_h: u32,
    crop: [u32; 4],
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
    // POC type 0 state.
    prev_poc_lsb: i32,
    prev_poc_msb: i32,
    sps: Sps,
    max_refs: u32,
}

struct ProfileAlign {
    /// Alignment for the bitstream buffer range (offset is always 0 here,
    /// which trivially satisfies the offset alignment).
    bitstream_size: u64,
}

/// Fill a Std scaling-list struct from parsed lists (or flat defaults).
fn std_scaling(lists: Option<&ScalingLists>) -> stdvk::StdVideoH264ScalingLists {
    let mut out = stdvk::StdVideoH264ScalingLists {
        scaling_list_present_mask: 0,
        use_default_scaling_matrix_mask: 0,
        ScalingList4x4: [[16; 16]; 6],
        ScalingList8x8: [[16; 64]; 6],
    };
    if let Some(l) = lists {
        out.scaling_list_present_mask = l.present_mask;
        out.use_default_scaling_matrix_mask = l.use_default_mask;
        out.ScalingList4x4 = l.list_4x4;
        out.ScalingList8x8 = l.list_8x8;
    }
    out
}

impl NvDecoder {
    pub fn new(sps: &Sps, pps: &Pps) -> Result<Self, NvDecError> {
        unsafe { Self::new_impl(sps, pps) }
    }

    unsafe fn new_impl(sps: &Sps, pps: &Pps) -> Result<Self, NvDecError> {
        let entry = ash::Entry::load().map_err(|e| NvDecError::Loading(e.to_string()))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?;

        // Find a physical device + queue family with H.264 decode.
        let mut chosen = None;
        for pd in instance.enumerate_physical_devices()? {
            let count = instance.get_physical_device_queue_family_properties(pd).len();
            let mut video_props =
                vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
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
                        .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
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
            ash::khr::video_decode_h264::NAME.as_ptr(),
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

        // Video profile: H.264 decode, progressive, 4:2:0 8-bit.
        let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(sps.profile_idc as u32)
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);

        let mut h264_caps = vk::VideoDecodeH264CapabilitiesKHR::default();
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let (profile_align, caps_max_dpb_slots) = {
            let mut caps = vk::VideoCapabilitiesKHR::default()
                .push_next(&mut decode_caps)
                .push_next(&mut h264_caps);
            (video_instance_fns.fp().get_physical_device_video_capabilities_khr)(
                pd, &profile, &mut caps,
            )
            .result()?;
            (
                ProfileAlign {
                    bitstream_size: caps.min_bitstream_buffer_size_alignment,
                },
                caps.max_dpb_slots,
            )
        };
        let coincide = decode_caps
            .flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE);
        if !coincide {
            return Err(NvDecError::Unsupported(
                "driver requires distinct DPB/output images (not yet implemented)".into(),
            ));
        }

        let coded_w = sps.mb_width * 16;
        let coded_h = sps.mb_height * 16;
        let max_refs = sps.max_num_ref_frames.max(1);
        let max_slots = (max_refs + 1).min(caps_max_dpb_slots);

        // Decode format: NV12.
        let format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
        let usage = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
            | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
            | vk::ImageUsageFlags::TRANSFER_SRC;
        {
            let profiles = [profile];
            let mut plist = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
            let fmt_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
                .image_usage(usage)
                .push_next(&mut plist);
            let get = video_instance_fns.fp().get_physical_device_video_format_properties_khr;
            let mut count = 0u32;
            (get)(pd, &fmt_info, &mut count, std::ptr::null_mut()).result()?;
            let mut fmts = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
            (get)(pd, &fmt_info, &mut count, fmts.as_mut_ptr()).result()?;
            if !fmts.iter().any(|f| f.format == format) {
                return Err(NvDecError::Unsupported(format!(
                    "NV12 not offered for decode (got {:?})",
                    fmts.iter().map(|f| f.format).collect::<Vec<_>>()
                )));
            }
        }

        // Session. Std header version 1.0.0 (VK_STD_vulkan_video_codec_h264_decode).
        let std_version = {
            let mut v = vk::ExtensionProperties::default()
                .spec_version(vk::make_api_version(0, 1, 0, 0));
            let name = c"VK_STD_vulkan_video_codec_h264_decode";
            let bytes = name.to_bytes_with_nul();
            for (dst, src) in v.extension_name.iter_mut().zip(bytes) {
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
            let mut reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
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
            .or_else(|| find_memory_type(&mem_props, req.memory_requirements.memory_type_bits, vk::MemoryPropertyFlags::empty()))
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

        // Session parameters: Std SPS/PPS.
        let sps_scaling = std_scaling(sps.scaling.as_ref());
        let std_sps = build_std_sps(sps, &sps_scaling);
        let pps_scaling = std_scaling(pps.scaling.as_ref());
        let std_pps = build_std_pps(sps, pps, &pps_scaling);
        let std_sps_arr = [std_sps];
        let std_pps_arr = [std_pps];
        let mut h264_params_add = vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&std_sps_arr)
            .std_pp_ss(&std_pps_arr);
        let mut h264_params = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&h264_params_add);
        let params_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut h264_params);
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
        let _ = &mut h264_params_add;

        // DPB slot images (coincide: decode output IS the slot image).
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
            session_memory.push(mem); // freed with the rest
            let view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?;
            slots.push(DpbSlot {
                image,
                view,
                active: false,
                frame_num: 0,
                poc: 0,
                age: 0,
            });
        }

        // Bitstream buffer (host visible) + readback buffer.
        let bitstream_cap = 4 << 20;
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
            let ptr = device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                as *mut u8;
            (buffer, mem, ptr)
        };
        let readback_size = (coded_w * coded_h * 3 / 2) as usize;
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
            "nvdec: session ready — {}x{} coded, {} DPB slots, coincide mode",
            coded_w,
            coded_h,
            max_slots
        );

        Ok(Self {
            _entry: entry,
            instance,
            device,
            video_queue_fns,
            decode_queue_fns,
            queue,
            queue_family,
            session,
            session_params,
            session_memory,
            profile_align,
            slots,
            coded_w,
            coded_h,
            crop: sps.frame_cropping,
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
            max_refs,
        })
    }

    fn compute_poc(&mut self, header: &SliceHeader) -> i32 {
        match self.sps.poc_type {
            0 => {
                if header.idr {
                    self.prev_poc_lsb = 0;
                    self.prev_poc_msb = 0;
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
                if header.nal_ref_idc != 0 {
                    self.prev_poc_lsb = lsb;
                    self.prev_poc_msb = msb;
                }
                msb + lsb
            }
            _ => 2 * header.frame_num as i32, // type 2, P-only
        }
    }

    /// Decode one access unit's slice NALs (raw, no start codes) and return
    /// the NV12 frame.
    pub fn decode(
        &mut self,
        slices: &[&[u8]],
        header: &SliceHeader,
    ) -> Result<Nv12Frame, NvDecError> {
        unsafe { self.decode_impl(slices, header) }
    }

    unsafe fn decode_impl(
        &mut self,
        slices: &[&[u8]],
        header: &SliceHeader,
    ) -> Result<Nv12Frame, NvDecError> {
        let poc = self.compute_poc(header);
        if header.idr {
            for s in &mut self.slots {
                s.active = false;
            }
        }

        // Write bitstream: 00 00 01 start code before each slice NAL.
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
            std::ptr::copy_nonoverlapping(nal.as_ptr(), self.bitstream_ptr.add(cursor + 3), nal.len());
            cursor += total;
        }
        let range = (cursor as u64).next_multiple_of(self.profile_align.bitstream_size.max(1));

        // Choose a DPB slot: reuse an inactive one, else evict the oldest.
        let setup_idx = self
            .slots
            .iter()
            .position(|s| !s.active)
            .unwrap_or_else(|| {
                self.slots
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, s)| s.age)
                    .map(|(i, _)| i)
                    .unwrap()
            });

        // Reference list: active slots, most recent frame_num first.
        let mut ref_indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| *i != setup_idx && s.active)
            .map(|(i, _)| i)
            .collect();
        ref_indices.sort_by_key(|&i| std::cmp::Reverse(self.slots[i].age));
        if matches!(header.slice_type, SliceType::P) && ref_indices.is_empty() {
            return Err(NvDecError::Unsupported(
                "P slice without any reference in the DPB".into(),
            ));
        }

        let device = &self.device;
        device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            self.cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        // Transition the setup image to the DPB layout.
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

        // Per-slot picture resources + reference infos (stable storage).
        let mut pic_resources: Vec<vk::VideoPictureResourceInfoKHR> = Vec::new();
        for idx in 0..self.slots.len() {
            pic_resources.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_extent(vk::Extent2D {
                        width: self.coded_w,
                        height: self.coded_h,
                    })
                    .image_view_binding(self.slots[idx].view),
            );
        }
        let std_ref_infos: Vec<stdvk::StdVideoDecodeH264ReferenceInfo> = self
            .slots
            .iter()
            .map(|s| {
                let mut info: stdvk::StdVideoDecodeH264ReferenceInfo =
                    std::mem::zeroed();
                info.FrameNum = s.frame_num as u16;
                info.PicOrderCnt = [s.poc, s.poc];
                info
            })
            .collect();

        // Begin-info reference slots: active refs with real indices + the
        // setup slot with index −1 (to be activated by this decode). The
        // pNext chains point into stable arrays — raw pointers because the
        // ash builder pattern can't express arrays of chained structs.
        let h264_dpb_infos: Vec<vk::VideoDecodeH264DpbSlotInfoKHR> = (0..self.slots.len())
            .map(|idx| {
                vk::VideoDecodeH264DpbSlotInfoKHR::default()
                    .std_reference_info(&std_ref_infos[idx])
            })
            .collect();
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for &idx in &ref_indices {
            let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(idx as i32)
                .picture_resource(&pic_resources[idx]);
            slot.p_next = &h264_dpb_infos[idx] as *const _ as *const std::ffi::c_void;
            begin_slots.push(slot);
        }
        // Setup slot enters the session scope deactivated.
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

        // Picture info for this frame.
        let is_ref = header.nal_ref_idc != 0;
        let std_pic = {
            let mut pic: stdvk::StdVideoDecodeH264PictureInfo = std::mem::zeroed();
            pic.flags
                .set_is_intra(u32::from(matches!(header.slice_type, SliceType::I)));
            pic.flags.set_IdrPicFlag(u32::from(header.idr));
            pic.flags.set_is_reference(u32::from(is_ref));
            pic.seq_parameter_set_id = self.sps.sps_id as u8;
            pic.pic_parameter_set_id = header.pps_id as u8;
            pic.frame_num = header.frame_num as u16;
            pic.idr_pic_id = header.idr_pic_id as u16;
            pic.PicOrderCnt = [poc, poc];
            pic
        };
        let mut h264_pic = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&std_pic)
            .slice_offsets(&offsets);

        let setup_std_ref = {
            let mut info: stdvk::StdVideoDecodeH264ReferenceInfo = std::mem::zeroed();
            info.FrameNum = header.frame_num as u16;
            info.PicOrderCnt = [poc, poc];
            info
        };
        let setup_h264_dpb =
            vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_std_ref);
        let mut setup_slot_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup_idx as i32)
            .picture_resource(&pic_resources[setup_idx]);
        setup_slot_info.p_next = &setup_h264_dpb as *const _ as *const std::ffi::c_void;

        let mut decode_ref_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for &idx in &ref_indices {
            let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(idx as i32)
                .picture_resource(&pic_resources[idx]);
            slot.p_next = &h264_dpb_infos[idx] as *const _ as *const std::ffi::c_void;
            decode_ref_slots.push(slot);
        }

        let mut decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(pic_resources[setup_idx])
            .setup_reference_slot(&setup_slot_info)
            .push_next(&mut h264_pic);
        if !decode_ref_slots.is_empty() {
            decode_info = decode_info.reference_slots(&decode_ref_slots);
        }
        (self.decode_queue_fns.fp().cmd_decode_video_khr)(self.cmd, &decode_info);

        (self.video_queue_fns.fp().cmd_end_video_coding_khr)(
            self.cmd,
            &vk::VideoEndCodingInfoKHR::default(),
        );

        // Copy the decoded image out (transition → transfer → back to DPB).
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
                buffer_offset: (self.coded_w * self.coded_h) as u64,
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

        // Bookkeeping: sliding window.
        self.age_counter += 1;
        {
            let slot = &mut self.slots[setup_idx];
            slot.active = is_ref;
            slot.frame_num = header.frame_num;
            slot.poc = poc;
            slot.age = self.age_counter;
        }
        let active = self.slots.iter().filter(|s| s.active).count() as u32;
        if active > self.max_refs
            && let Some(oldest) = self
                .slots
                .iter_mut()
                .filter(|s| s.active)
                .min_by_key(|s| s.age)
        {
            oldest.active = false;
        }

        // Read NV12 out.
        let mapped = device.map_memory(
            self.readback_mem,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::empty(),
        )? as *const u8;
        let y_size = (self.coded_w * self.coded_h) as usize;
        let uv_size = y_size / 2;
        let mut y = vec![0u8; y_size];
        let mut uv = vec![0u8; uv_size];
        std::ptr::copy_nonoverlapping(mapped, y.as_mut_ptr(), y_size);
        std::ptr::copy_nonoverlapping(mapped.add(y_size), uv.as_mut_ptr(), uv_size);
        device.unmap_memory(self.readback_mem);

        Ok(Nv12Frame {
            width: self.coded_w,
            height: self.coded_h,
            y,
            uv,
            stride: self.coded_w,
        })
    }

    pub fn cropped_size(&self) -> (u32, u32) {
        (
            self.coded_w - 2 * (self.crop[0] + self.crop[1]),
            self.coded_h - 2 * (self.crop[2] + self.crop[3]),
        )
    }

    pub fn crop_offsets(&self) -> (u32, u32) {
        (2 * self.crop[0], 2 * self.crop[2])
    }
}

impl Drop for NvDecoder {
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

/// Safety: constructed from zeroed memory + bindgen setters; the raw pointers
/// must outlive the session-parameters creation call.
unsafe fn build_std_sps(
    sps: &Sps,
    scaling: &stdvk::StdVideoH264ScalingLists,
) -> stdvk::StdVideoH264SequenceParameterSet {
    let mut s: stdvk::StdVideoH264SequenceParameterSet = std::mem::zeroed();
    s.flags.set_constraint_set0_flag((sps.constraint_flags >> 7) as u32 & 1);
    s.flags.set_constraint_set1_flag((sps.constraint_flags >> 6) as u32 & 1);
    s.flags.set_constraint_set2_flag((sps.constraint_flags >> 5) as u32 & 1);
    s.flags.set_constraint_set3_flag((sps.constraint_flags >> 4) as u32 & 1);
    s.flags.set_constraint_set4_flag((sps.constraint_flags >> 3) as u32 & 1);
    s.flags.set_constraint_set5_flag((sps.constraint_flags >> 2) as u32 & 1);
    s.flags.set_direct_8x8_inference_flag(u32::from(sps.direct_8x8));
    s.flags.set_frame_mbs_only_flag(1);
    s.flags.set_gaps_in_frame_num_value_allowed_flag(u32::from(sps.gaps_allowed));
    s.flags.set_qpprime_y_zero_transform_bypass_flag(u32::from(sps.qpprime_y_zero));
    s.flags.set_frame_cropping_flag(u32::from(sps.frame_cropping_flag));
    s.flags.set_seq_scaling_matrix_present_flag(u32::from(sps.scaling.is_some()));
    s.profile_idc = sps.profile_idc as u32;
    s.level_idc = level_to_std(sps.level_idc);
    s.chroma_format_idc = sps.chroma_format_idc;
    s.seq_parameter_set_id = sps.sps_id as u8;
    s.log2_max_frame_num_minus4 = (sps.log2_max_frame_num - 4) as u8;
    s.pic_order_cnt_type = sps.poc_type;
    s.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_poc_lsb.saturating_sub(4) as u8;
    s.max_num_ref_frames = sps.max_num_ref_frames as u8;
    s.pic_width_in_mbs_minus1 = sps.mb_width - 1;
    s.pic_height_in_map_units_minus1 = sps.mb_height - 1;
    s.frame_crop_left_offset = sps.frame_cropping[0];
    s.frame_crop_right_offset = sps.frame_cropping[1];
    s.frame_crop_top_offset = sps.frame_cropping[2];
    s.frame_crop_bottom_offset = sps.frame_cropping[3];
    s.pScalingLists = scaling;
    s
}

/// Safety: see [`build_std_sps`].
unsafe fn build_std_pps(
    sps: &Sps,
    pps: &Pps,
    scaling: &stdvk::StdVideoH264ScalingLists,
) -> stdvk::StdVideoH264PictureParameterSet {
    let mut p: stdvk::StdVideoH264PictureParameterSet = std::mem::zeroed();
    p.flags.set_transform_8x8_mode_flag(u32::from(pps.transform_8x8));
    p.flags.set_redundant_pic_cnt_present_flag(u32::from(pps.redundant_pic_cnt_present));
    p.flags.set_constrained_intra_pred_flag(u32::from(pps.constrained_intra));
    p.flags
        .set_deblocking_filter_control_present_flag(u32::from(pps.deblocking_control_present));
    p.flags.set_weighted_pred_flag(u32::from(pps.weighted_pred));
    p.flags
        .set_bottom_field_pic_order_in_frame_present_flag(u32::from(pps.pic_order_present));
    p.flags.set_entropy_coding_mode_flag(u32::from(pps.entropy_cabac));
    p.flags.set_pic_scaling_matrix_present_flag(u32::from(pps.scaling.is_some()));
    p.seq_parameter_set_id = sps.sps_id as u8;
    p.pic_parameter_set_id = pps.pps_id as u8;
    p.num_ref_idx_l0_default_active_minus1 = (pps.num_ref_idx_l0_default - 1) as u8;
    p.num_ref_idx_l1_default_active_minus1 = (pps.num_ref_idx_l1_default - 1) as u8;
    p.weighted_bipred_idc = pps.weighted_bipred_idc;
    p.pic_init_qp_minus26 = pps.pic_init_qp_minus26 as i8;
    p.pic_init_qs_minus26 = pps.pic_init_qs_minus26 as i8;
    p.chroma_qp_index_offset = pps.chroma_qp_index_offset as i8;
    p.second_chroma_qp_index_offset = pps.second_chroma_qp_index_offset as i8;
    p.pScalingLists = scaling;
    p
}

fn level_to_std(level_idc: u8) -> u32 {
    // StdVideoH264LevelIdc is an enum 0..N in level order; map the common ones.
    match level_idc {
        10 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0,
        11 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1,
        12 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_2,
        13 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_3,
        20 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_0,
        21 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_1,
        22 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_2,
        30 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_0,
        31 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1,
        32 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_2,
        40 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0,
        41 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        42 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2,
        50 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0,
        51 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1,
        52 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2,
        60 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_0,
        61 => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_1,
        _ => stdvk::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2,
    }
}

// Silence unused-field warnings for identifiers kept for Drop/debug clarity.
#[allow(dead_code)]
fn _unused(d: &NvDecoder) -> (&u32, &CStr) {
    (&d.queue_family, c"")
}
