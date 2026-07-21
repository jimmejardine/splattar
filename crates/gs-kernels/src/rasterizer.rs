//! Host orchestration for the training rasterizer: surfel preprocess →
//! tile binning → per-tile forward compositing (backward lands next to it).
//! Image size and capacities are fixed per instance; the trainer allocates
//! once at the MCMC budget (no mid-training reallocation — see CLAUDE.md).

use glam::{Mat3, Quat, Vec3};
use gs_wgpu::{GpuContext, GpuTimer, buffers};

use crate::binning::TileBinner;

pub(crate) use gs_wgpu::profile::scope as ts;

pub const TILE: u32 = 16;
/// GPU SH storage is always padded to degree 3 (48 floats per surfel).
pub const SH_FLOATS: usize = 48;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    rot0: [f32; 4],
    rot1: [f32; 4],
    rot2: [f32; 4],
    center: [f32; 4],
    focal: f32,
    width: u32,
    height: u32,
    sh_degree: u32,
    num_surfels: u32,
    tiles_x: u32,
    tiles_y: u32,
    lambda_dist: f32,
}

#[derive(Debug, Clone)]
pub struct RasterCamera {
    pub center: Vec3,
    /// Camera-to-world rotation (normalized on upload).
    pub quat: Quat,
    pub focal: f32,
    pub sh_degree: u32,
}

/// CPU-side scene arrays, uploaded as one block. Lengths must agree.
pub struct SceneInput<'a> {
    pub positions: &'a [Vec3],
    pub scales: &'a [[f32; 2]],
    /// xyzw, normalized on the GPU.
    pub quats: &'a [[f32; 4]],
    pub opacities: &'a [f32],
    /// Coefficient-major rgb-interleaved, `coeffs*3` floats per surfel
    /// (unpadded; padded to 48 on upload).
    pub sh: &'a [f32],
    pub sh_coeffs: usize,
}

pub struct Rasterizer {
    capacity: u32,
    /// Depth-distortion loss weight, written into the camera uniform each
    /// forward (0 disables the distortion gradient path).
    pub lambda_dist: f32,
    pub width: u32,
    pub height: u32,
    tiles_x: u32,
    tiles_y: u32,
    camera_buf: wgpu::Buffer,
    pub positions: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub quats: wgpu::Buffer,
    pub opacities: wgpu::Buffer,
    pub sh: wgpu::Buffer,
    /// Kept alive by the bind groups; exposed for future debug tooling.
    pub surf_cam: wgpu::Buffer,
    pub out_color: wgpu::Buffer,
    pub out_aux: wgpu::Buffer,
    pub out_normal: wgpu::Buffer,
    pub out_ncontrib: wgpu::Buffer,
    binner: TileBinner,
    prep_pipeline: wgpu::ComputePipeline,
    prep_bg: wgpu::BindGroup,
    fwd_pipeline: wgpu::ComputePipeline,
    fwd_bg: wgpu::BindGroup,
    // Backward: dL/d(color image) in (w channel = depth-loss grad),
    // dL/d(normal image), parameter gradients out.
    pub dl_dcolor: wgpu::Buffer,
    pub dl_dnormal: wgpu::Buffer,
    grad_geom: wgpu::Buffer,
    pub grad_pos: wgpu::Buffer,
    pub grad_scales: wgpu::Buffer,
    pub grad_quat: wgpu::Buffer,
    pub grad_opacity: wgpu::Buffer,
    pub grad_sh: wgpu::Buffer,
    /// [dl_dR_cam 9, dcam_center 3, dfocal 1, pad 3] as f32 bits.
    pub grad_cam: wgpu::Buffer,
    bwd_pipeline: wgpu::ComputePipeline,
    bwd_bg: wgpu::BindGroup,
    geom_bwd_pipeline: wgpu::ComputePipeline,
    geom_bwd_bg: wgpu::BindGroup,
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl Rasterizer {
    pub fn new(ctx: &GpuContext, capacity: u32, width: u32, height: u32, max_entries: u32) -> Self {
        let device = &ctx.device;
        let tiles_x = width.div_ceil(TILE);
        let tiles_y = height.div_ceil(TILE);
        let binner = TileBinner::new(ctx, capacity, max_entries, tiles_x * tiles_y);

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster-camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let n = capacity as u64;
        let positions = buffers::storage_empty(device, "raster-positions", n * 16);
        let scales = buffers::storage_empty(device, "raster-scales", n * 8);
        let quats = buffers::storage_empty(device, "raster-quats", n * 16);
        let opacities = buffers::storage_empty(device, "raster-opacities", n * 4);
        let sh = buffers::storage_empty(device, "raster-sh", n * SH_FLOATS as u64 * 4);
        let surf_cam = buffers::storage_empty(device, "raster-surfcam", n * 80);

        let px = (width * height) as u64;
        let out_color = buffers::storage_empty(device, "raster-out-color", px * 16);
        let out_aux = buffers::storage_empty(device, "raster-out-aux", px * 16);
        let out_normal = buffers::storage_empty(device, "raster-out-normal", px * 16);
        let out_ncontrib = buffers::storage_empty(device, "raster-out-ncontrib", px * 4);

        let common = include_str!("shaders/raster_common.wgsl");
        let make = |label: &str, body: &str, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(format!("{common}\n{body}").into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let prep_pipeline = make(
            "surfel-prep",
            include_str!("shaders/surfel_prep_fwd.wgsl"),
            "surfel_prep",
        );
        let fwd_pipeline = make(
            "rasterize-fwd",
            include_str!("shaders/rasterize_fwd.wgsl"),
            "rasterize_fwd",
        );
        let bwd_pipeline = make(
            "rasterize-bwd",
            include_str!("shaders/rasterize_bwd.wgsl"),
            "rasterize_bwd",
        );
        let geom_bwd_pipeline = make(
            "surfel-prep-bwd",
            include_str!("shaders/surfel_prep_bwd.wgsl"),
            "surfel_prep_bwd",
        );

        let dl_dcolor = buffers::storage_empty(device, "raster-dl-dcolor", px * 16);
        let dl_dnormal = buffers::storage_empty(device, "raster-dl-dnormal", px * 16);
        let grad_geom = buffers::storage_empty(device, "raster-grad-geom", n * 16 * 4);
        let grad_pos = buffers::storage_empty(device, "raster-grad-pos", n * 16);
        let grad_scales = buffers::storage_empty(device, "raster-grad-scales", n * 8);
        let grad_quat = buffers::storage_empty(device, "raster-grad-quat", n * 16);
        let grad_opacity = buffers::storage_empty(device, "raster-grad-opacity", n * 4);
        let grad_sh = buffers::storage_empty(device, "raster-grad-sh", n * SH_FLOATS as u64 * 4);
        let grad_cam = buffers::storage_empty(device, "raster-grad-cam", 16 * 4);

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surfel-prep-bg"),
            layout: &prep_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &camera_buf),
                bind(1, &positions),
                bind(2, &scales),
                bind(3, &quats),
                bind(4, &opacities),
                bind(5, &sh),
                bind(6, &surf_cam),
                bind(7, &binner.rects),
                bind(8, &binner.depths),
            ],
        });
        let fwd_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rasterize-fwd-bg"),
            layout: &fwd_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &camera_buf),
                bind(1, &surf_cam),
                bind(2, binner.sorted_entries()),
                bind(3, &binner.entry_items),
                bind(4, &binner.ranges),
                bind(5, &out_color),
                bind(6, &out_aux),
                bind(7, &out_normal),
                bind(8, &out_ncontrib),
            ],
        });

        let bwd_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rasterize-bwd-bg"),
            layout: &bwd_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &camera_buf),
                bind(1, &surf_cam),
                bind(2, binner.sorted_entries()),
                bind(3, &binner.entry_items),
                bind(4, &binner.ranges),
                bind(5, &dl_dcolor),
                bind(6, &out_aux),
                bind(7, &out_ncontrib),
                bind(8, &grad_geom),
                bind(9, &grad_cam),
                bind(10, &dl_dnormal),
                bind(11, &out_color),
            ],
        });
        let geom_bwd_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surfel-prep-bwd-bg"),
            layout: &geom_bwd_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &camera_buf),
                bind(1, &positions),
                bind(2, &scales),
                bind(3, &quats),
                bind(4, &sh),
                bind(5, &surf_cam),
                bind(6, &grad_geom),
                bind(7, &grad_pos),
                bind(8, &grad_scales),
                bind(9, &grad_quat),
                bind(10, &grad_opacity),
                bind(11, &grad_sh),
                bind(12, &grad_cam),
            ],
        });

        Self {
            capacity,
            lambda_dist: 0.0,
            width,
            height,
            tiles_x,
            tiles_y,
            camera_buf,
            positions,
            scales,
            quats,
            opacities,
            sh,
            surf_cam,
            out_color,
            out_aux,
            out_normal,
            out_ncontrib,
            binner,
            prep_pipeline,
            prep_bg,
            fwd_pipeline,
            fwd_bg,
            dl_dcolor,
            dl_dnormal,
            grad_geom,
            grad_pos,
            grad_scales,
            grad_quat,
            grad_opacity,
            grad_sh,
            grad_cam,
            bwd_pipeline,
            bwd_bg,
            geom_bwd_pipeline,
            geom_bwd_bg,
        }
    }

    /// Record the backward pass. Caller has uploaded dL/d(color) into
    /// [`dl_dcolor`] and already recorded a matching [`forward`] this frame.
    pub fn backward(&self, encoder: &mut wgpu::CommandEncoder, num_surfels: u32) {
        self.backward_timed(encoder, num_surfels, None);
    }

    /// [`backward`] with each pass wrapped in a GpuTimer scope.
    pub fn backward_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        num_surfels: u32,
        mut timer: Option<&mut GpuTimer>,
    ) {
        encoder.clear_buffer(&self.grad_geom, 0, None);
        encoder.clear_buffer(&self.grad_cam, 0, None);
        encoder.clear_buffer(&self.grad_pos, 0, None);
        encoder.clear_buffer(&self.grad_scales, 0, None);
        encoder.clear_buffer(&self.grad_quat, 0, None);
        encoder.clear_buffer(&self.grad_opacity, 0, None);
        encoder.clear_buffer(&self.grad_sh, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize-bwd"),
                timestamp_writes: ts(&mut timer, "rasterize-bwd"),
            });
            pass.set_pipeline(&self.bwd_pipeline);
            pass.set_bind_group(0, &self.bwd_bg, &[]);
            pass.dispatch_workgroups(self.tiles_x, self.tiles_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surfel-prep-bwd"),
                timestamp_writes: ts(&mut timer, "surfel-prep-bwd"),
            });
            pass.set_pipeline(&self.geom_bwd_pipeline);
            pass.set_bind_group(0, &self.geom_bwd_bg, &[]);
            pass.dispatch_workgroups(num_surfels.div_ceil(256), 1, 1);
        }
    }

    /// Upload scene arrays (padding SH to 48 floats per surfel).
    pub fn upload_scene(&self, ctx: &GpuContext, scene: &SceneInput<'_>) {
        let n = scene.positions.len();
        assert!(n as u32 <= self.capacity);
        assert_eq!(scene.scales.len(), n);
        assert_eq!(scene.quats.len(), n);
        assert_eq!(scene.opacities.len(), n);
        assert_eq!(scene.sh.len(), n * scene.sh_coeffs * 3);

        let mut positions = Vec::with_capacity(n * 4);
        for p in scene.positions {
            positions.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
        }
        let mut sh = vec![0f32; n * SH_FLOATS];
        let per = scene.sh_coeffs * 3;
        for i in 0..n {
            sh[i * SH_FLOATS..i * SH_FLOATS + per]
                .copy_from_slice(&scene.sh[i * per..(i + 1) * per]);
        }
        ctx.queue
            .write_buffer(&self.positions, 0, bytemuck::cast_slice(&positions));
        ctx.queue
            .write_buffer(&self.scales, 0, bytemuck::cast_slice(scene.scales));
        ctx.queue
            .write_buffer(&self.quats, 0, bytemuck::cast_slice(scene.quats));
        ctx.queue
            .write_buffer(&self.opacities, 0, bytemuck::cast_slice(scene.opacities));
        ctx.queue.write_buffer(&self.sh, 0, bytemuck::cast_slice(&sh));
    }

    /// Record the full forward pass for `num_surfels`.
    pub fn forward(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        camera: &RasterCamera,
        num_surfels: u32,
    ) {
        self.forward_timed(ctx, encoder, camera, num_surfels, None);
    }

    /// [`forward`] with each pass (including binning + both sorts) wrapped in
    /// a GpuTimer scope.
    pub fn forward_timed(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        camera: &RasterCamera,
        num_surfels: u32,
        mut timer: Option<&mut GpuTimer>,
    ) {
        assert!(num_surfels <= self.capacity);
        let r = Mat3::from_quat(camera.quat.normalize());
        let uniform = CameraUniform {
            rot0: r.x_axis.extend(0.0).into(),
            rot1: r.y_axis.extend(0.0).into(),
            rot2: r.z_axis.extend(0.0).into(),
            center: camera.center.extend(1.0).into(),
            focal: camera.focal,
            width: self.width,
            height: self.height,
            sh_degree: camera.sh_degree,
            num_surfels,
            tiles_x: self.tiles_x,
            tiles_y: self.tiles_y,
            lambda_dist: self.lambda_dist,
        };
        ctx.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surfel-prep"),
                timestamp_writes: ts(&mut timer, "surfel-prep"),
            });
            pass.set_pipeline(&self.prep_pipeline);
            pass.set_bind_group(0, &self.prep_bg, &[]);
            pass.dispatch_workgroups(num_surfels.div_ceil(256), 1, 1);
        }
        self.binner
            .encode_timed(ctx, encoder, num_surfels, self.tiles_x, timer.as_deref_mut());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize-fwd"),
                timestamp_writes: ts(&mut timer, "rasterize-fwd"),
            });
            pass.set_pipeline(&self.fwd_pipeline);
            pass.set_bind_group(0, &self.fwd_bg, &[]);
            pass.dispatch_workgroups(self.tiles_x, self.tiles_y, 1);
        }
    }
}
