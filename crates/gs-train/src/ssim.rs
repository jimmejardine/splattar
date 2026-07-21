//! Host orchestration for the fused L1 + D-SSIM loss. Produces dL/d(color)
//! directly into the rasterizer's `dl_dcolor` buffer.
//!
//! Loss = (1−λ)·mean|x−y| + λ·mean((1−SSIM)/2), means over pixels×channels.

use gs_wgpu::{GpuContext, buffers};

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsimUniform {
    width: u32,
    height: u32,
    inv_n3: f32,
    lambda: f32,
}

#[allow(dead_code)] // several buffers are held only to keep bind groups valid
pub struct SsimLoss {
    width: u32,
    height: u32,
    pub lambda: f32,
    uniform: wgpu::Buffer,
    // Intermediates (all vec4-per-pixel).
    pub target: wgpu::Buffer,
    x2: wgpu::Buffer,
    y2: wgpu::Buffer,
    xy: wgpu::Buffer,
    tmp: wgpu::Buffer,
    mu_x: wgpu::Buffer,
    mu_y: wgpu::Buffer,
    bx2: wgpu::Buffer,
    /// After `partials` this holds Cxy; before, blurred y².
    by2: wgpu::Buffer,
    /// After `partials` this holds the SSIM map; before, blurred xy.
    pub ssim_map: wgpu::Buffer,
    c_mu: wgpu::Buffer,
    c_x2: wgpu::Buffer,
    bc_mu: wgpu::Buffer,
    bc_x2: wgpu::Buffer,
    bc_xy: wgpu::Buffer,
    pub l1_map: wgpu::Buffer,
    products_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    partials_pipeline: wgpu::ComputePipeline,
    combine_pipeline: wgpu::ComputePipeline,
    products_bg: wgpu::BindGroup,
    partials_bg: wgpu::BindGroup,
    combine_bg: wgpu::BindGroup,
    /// (h, v) bind-group pairs for the 8 blur jobs, in encode order.
    blur_bgs: Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
}

impl SsimLoss {
    /// `img` is the rasterizer's out_color; `dl_out` its dl_dcolor.
    pub fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        lambda: f32,
        img: &wgpu::Buffer,
        dl_out: &wgpu::Buffer,
    ) -> Self {
        let device = &ctx.device;
        let px_bytes = (width * height) as u64 * 16;
        let buf = |name: &str| buffers::storage_empty(device, name, px_bytes);

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssim-uniform"),
            size: std::mem::size_of::<SsimUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let u = SsimUniform {
            width,
            height,
            inv_n3: 1.0 / (width as f32 * height as f32 * 3.0),
            lambda,
        };
        ctx.queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&u));

        let target = buf("ssim-target");
        let x2 = buf("ssim-x2");
        let y2 = buf("ssim-y2");
        let xy = buf("ssim-xy");
        let tmp = buf("ssim-tmp");
        let mu_x = buf("ssim-mu-x");
        let mu_y = buf("ssim-mu-y");
        let bx2 = buf("ssim-bx2");
        let by2 = buf("ssim-by2");
        let ssim_map = buf("ssim-bxy-map");
        let c_mu = buf("ssim-c-mu");
        let c_x2 = buf("ssim-c-x2");
        let bc_mu = buf("ssim-bc-mu");
        let bc_x2 = buf("ssim-bc-x2");
        let bc_xy = buf("ssim-bc-xy");
        let l1_map = buf("ssim-l1-map");

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssim"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssim.wgsl").into()),
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
        let products_pipeline = make("products");
        let blur_h_pipeline = make("blur_h");
        let blur_v_pipeline = make("blur_v");
        let partials_pipeline = make("partials");
        let combine_pipeline = make("combine");

        let products_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssim-products"),
            layout: &products_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &uniform),
                bind(1, img),
                bind(2, &target),
                bind(6, &x2),
                bind(7, &y2),
                bind(8, &xy),
            ],
        });
        let partials_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssim-partials"),
            layout: &partials_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &uniform),
                bind(3, &mu_x),
                bind(4, &mu_y),
                bind(5, &bx2),
                bind(6, &c_mu),
                bind(7, &c_x2),
                bind(8, &by2),
                bind(9, &ssim_map),
            ],
        });
        let combine_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssim-combine"),
            layout: &combine_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &uniform),
                bind(1, img),
                bind(2, &target),
                bind(3, &bc_mu),
                bind(4, &bc_x2),
                bind(5, &bc_xy),
                bind(6, dl_out),
                bind(7, &l1_map),
            ],
        });

        // Blur jobs in encode order: 5 forward moments, 3 backward coefficient
        // maps (Cxy lives in the by2 buffer after `partials`).
        let jobs: [(&wgpu::Buffer, &wgpu::Buffer); 8] = [
            (img, &mu_x),
            (&target, &mu_y),
            (&x2, &bx2),
            (&y2, &by2),
            (&xy, &ssim_map),
            (&c_mu, &bc_mu),
            (&c_x2, &bc_x2),
            (&by2, &bc_xy),
        ];
        let blur_bgs = jobs
            .iter()
            .map(|(src, dst)| {
                let h = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssim-blur-h"),
                    layout: &blur_h_pipeline.get_bind_group_layout(0),
                    entries: &[bind(0, &uniform), bind(3, src), bind(6, &tmp)],
                });
                let v = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssim-blur-v"),
                    layout: &blur_v_pipeline.get_bind_group_layout(0),
                    entries: &[bind(0, &uniform), bind(3, &tmp), bind(6, dst)],
                });
                (h, v)
            })
            .collect();

        Self {
            width,
            height,
            lambda,
            uniform,
            target,
            x2,
            y2,
            xy,
            tmp,
            mu_x,
            mu_y,
            bx2,
            by2,
            ssim_map,
            c_mu,
            c_x2,
            bc_mu,
            bc_x2,
            bc_xy,
            l1_map,
            products_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            partials_pipeline,
            combine_pipeline,
            products_bg,
            partials_bg,
            combine_bg,
            blur_bgs,
        }
    }

    /// Record the full loss pipeline: assumes the forward image is current and
    /// the target uploaded. Writes dL/d(color) into the rasterizer's buffer.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_timed(encoder, None);
    }

    /// [`encode`] wrapped in a GpuTimer scope.
    pub fn encode_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mut timer: Option<&mut gs_wgpu::GpuTimer>,
    ) {
        let npix = self.width * self.height;
        let groups_1d = npix.div_ceil(256);
        let gx = self.width.div_ceil(16);
        let gy = self.height.div_ceil(16);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssim-loss"),
            timestamp_writes: gs_wgpu::profile::scope(&mut timer, "ssim-loss"),
        });
        pass.set_pipeline(&self.products_pipeline);
        pass.set_bind_group(0, &self.products_bg, &[]);
        pass.dispatch_workgroups(groups_1d, 1, 1);

        for (h, v) in &self.blur_bgs[..5] {
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.set_bind_group(0, h, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.set_bind_group(0, v, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        pass.set_pipeline(&self.partials_pipeline);
        pass.set_bind_group(0, &self.partials_bg, &[]);
        pass.dispatch_workgroups(groups_1d, 1, 1);

        for (h, v) in &self.blur_bgs[5..] {
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.set_bind_group(0, h, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.set_bind_group(0, v, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        pass.set_pipeline(&self.combine_pipeline);
        pass.set_bind_group(0, &self.combine_bg, &[]);
        pass.dispatch_workgroups(groups_1d, 1, 1);
    }

    /// Blocking loss readback for logging: (l1_mean, dssim_mean).
    pub fn read_losses(&self, ctx: &GpuContext) -> (f64, f64) {
        let l1: Vec<[f32; 4]> =
            bytemuck::cast_slice(&buffers::readback(&ctx.device, &ctx.queue, &self.l1_map))
                .to_vec();
        let ssim: Vec<[f32; 4]> =
            bytemuck::cast_slice(&buffers::readback(&ctx.device, &ctx.queue, &self.ssim_map))
                .to_vec();
        let n3 = (self.width * self.height * 3) as f64;
        let l1_mean = l1.iter().flat_map(|p| &p[..3]).map(|&v| v as f64).sum::<f64>() / n3;
        let dssim_mean = ssim
            .iter()
            .flat_map(|p| &p[..3])
            .map(|&v| (1.0 - v as f64) * 0.5)
            .sum::<f64>()
            / n3;
        (l1_mean, dssim_mean)
    }
}
