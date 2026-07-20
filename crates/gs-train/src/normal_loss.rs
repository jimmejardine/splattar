//! Host orchestration for the normal-consistency loss. Reads the rasterizer's
//! forward outputs, writes dl_dnormal and the depth-loss gradient channel
//! (dl_dcolor.w). Must be encoded AFTER the color loss (which owns rgb) and
//! BEFORE the rasterizer backward.

use gs_wgpu::{GpuContext, buffers};

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormalUniform {
    width: u32,
    height: u32,
    focal: f32,
    lambda: f32,
    alpha_min: f32,
    _pad: [f32; 3],
}

pub struct NormalLoss {
    width: u32,
    height: u32,
    uniform: wgpu::Buffer,
    #[allow(dead_code)] // kept alive for the bind groups
    cpx: wgpu::Buffer,
    #[allow(dead_code)]
    cpy: wgpu::Buffer,
    pub loss_map: wgpu::Buffer,
    pass1_pipeline: wgpu::ComputePipeline,
    pass1_bg: wgpu::BindGroup,
    pass2_pipeline: wgpu::ComputePipeline,
    pass2_bg: wgpu::BindGroup,
}

impl NormalLoss {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        out_color: &wgpu::Buffer,
        out_aux: &wgpu::Buffer,
        out_normal: &wgpu::Buffer,
        dl_dnormal: &wgpu::Buffer,
        dl_dcolor: &wgpu::Buffer,
    ) -> Self {
        let device = &ctx.device;
        let px_bytes = (width * height) as u64 * 16;
        let cpx = buffers::storage_empty(device, "nl-cpx", px_bytes);
        let cpy = buffers::storage_empty(device, "nl-cpy", px_bytes);
        let loss_map = buffers::storage_empty(device, "nl-loss", px_bytes);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nl-uniform"),
            size: std::mem::size_of::<NormalUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normal-loss"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/normal_loss.wgsl").into()),
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pass1_pipeline = make("pass1");
        let pass2_pipeline = make("pass2");

        let pass1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nl-pass1"),
            layout: &pass1_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &uniform),
                bind(1, out_color),
                bind(2, out_aux),
                bind(3, out_normal),
                bind(4, &cpx),
                bind(5, &cpy),
                bind(6, dl_dnormal),
                bind(7, &loss_map),
            ],
        });
        let pass2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nl-pass2"),
            layout: &pass2_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &uniform),
                bind(1, out_color),
                bind(4, &cpx),
                bind(5, &cpy),
                bind(8, dl_dcolor),
            ],
        });

        Self {
            width,
            height,
            uniform,
            cpx,
            cpy,
            loss_map,
            pass1_pipeline,
            pass1_bg,
            pass2_pipeline,
            pass2_bg,
        }
    }

    /// `lambda` is the raw loss weight; normalized by pixel count here.
    pub fn set_lambda(&self, ctx: &GpuContext, lambda: f32, focal: f32) {
        let u = NormalUniform {
            width: self.width,
            height: self.height,
            focal,
            lambda: lambda / (self.width * self.height) as f32,
            alpha_min: 0.2,
            _pad: [0.0; 3],
        };
        ctx.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let groups = (self.width * self.height).div_ceil(256);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normal-loss"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pass1_pipeline);
        pass.set_bind_group(0, &self.pass1_bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        pass.set_pipeline(&self.pass2_pipeline);
        pass.set_bind_group(0, &self.pass2_bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}
