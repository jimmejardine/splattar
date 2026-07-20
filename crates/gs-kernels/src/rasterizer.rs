//! Host orchestration for the training rasterizer: surfel preprocess →
//! tile binning → per-tile forward compositing (backward lands next to it).
//! Image size and capacities are fixed per instance; the trainer allocates
//! once at the MCMC budget (no mid-training reallocation — see CLAUDE.md).

use glam::{Mat3, Quat, Vec3};
use gs_wgpu::{GpuContext, buffers};

use crate::binning::TileBinner;

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
    _pad: u32,
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
    surf_cam: wgpu::Buffer,
    pub out_color: wgpu::Buffer,
    pub out_aux: wgpu::Buffer,
    pub out_normal: wgpu::Buffer,
    pub out_ncontrib: wgpu::Buffer,
    binner: TileBinner,
    prep_pipeline: wgpu::ComputePipeline,
    prep_bg: wgpu::BindGroup,
    fwd_pipeline: wgpu::ComputePipeline,
    fwd_bg: wgpu::BindGroup,
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

        Self {
            capacity,
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
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surfel-prep"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prep_pipeline);
            pass.set_bind_group(0, &self.prep_bg, &[]);
            pass.dispatch_workgroups(num_surfels.div_ceil(256), 1, 1);
        }
        self.binner.encode(ctx, encoder, num_surfels, self.tiles_x);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize-fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fwd_pipeline);
            pass.set_bind_group(0, &self.fwd_bg, &[]);
            pass.dispatch_workgroups(self.tiles_x, self.tiles_y, 1);
        }
    }
}
