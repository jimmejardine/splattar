//! SplatRenderer: preprocess → depth sort → indirect instanced draw.
//! Shared verbatim by the interactive window and the offscreen/golden path.

use glam::Vec2;
use gs_core::Camera;
use gs_wgpu::{GpuContext, RadixSorter};

use crate::scene::GpuScene;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    viewport: [f32; 2],
    focal: [f32; 2],
    num_splats: u32,
    sh_degree: u32,
    splat_scale: f32,
    _pad: u32,
}

pub struct RenderSettings {
    /// Active SH degree (clamped to the scene's own degree).
    pub sh_degree: u8,
    /// Global multiplier on splat extents (debug lever).
    pub splat_scale: f32,
    pub background: wgpu::Color,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            sh_degree: 3,
            splat_scale: 1.0,
            background: wgpu::Color::BLACK,
        }
    }
}

pub struct SplatRenderer {
    scene_degree: u8,
    num_splats: u32,
    sorter: RadixSorter,
    camera_buf: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    preprocess_pipeline: wgpu::ComputePipeline,
    preprocess_bg: wgpu::BindGroup,
    draw_prep_pipeline: wgpu::ComputePipeline,
    draw_prep_bg: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    draw_bg: wgpu::BindGroup,
}

impl SplatRenderer {
    pub fn new(ctx: &GpuContext, scene: &GpuScene, target_format: wgpu::TextureFormat) -> Self {
        let device = &ctx.device;
        let n = scene.num_splats;
        let sorter = RadixSorter::new(ctx, n);

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let splats_2d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splats-2d"),
            size: n as u64 * 48, // 3 × vec4<f32>
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let draw_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw-args"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        // ---- preprocess ----
        let pre_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/preprocess.wgsl").into()),
        });
        let pre_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preprocess-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                compute_storage(1, true),
                compute_storage(2, true),
                compute_storage(3, true),
                compute_storage(4, true),
                compute_storage(5, true),
                compute_storage(6, false),
                compute_storage(7, false),
                compute_storage(8, false),
                compute_storage(9, false),
            ],
        });
        let preprocess_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preprocess-bg"),
            layout: &pre_bgl,
            entries: &[
                entry(0, &camera_buf),
                entry(1, &scene.positions),
                entry(2, &scene.sh),
                entry(3, &scene.opacity),
                entry(4, &scene.scales),
                entry(5, &scene.rotations),
                entry(6, &splats_2d),
                entry(7, sorter.keys()),
                entry(8, sorter.payloads()),
                entry(9, sorter.counts()),
            ],
        });
        let pre_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess-pl"),
            bind_group_layouts: &[Some(&pre_bgl)],
            immediate_size: 0,
        });
        let preprocess_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("preprocess"),
                layout: Some(&pre_pl),
                module: &pre_module,
                entry_point: Some("preprocess"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ---- draw-args prep ----
        let prep_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("draw-prep"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/draw_prep.wgsl").into()),
        });
        let prep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw-prep-bgl"),
            entries: &[compute_storage(0, true), compute_storage(1, false)],
        });
        let draw_prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw-prep-bg"),
            layout: &prep_bgl,
            entries: &[entry(0, sorter.counts()), entry(1, &draw_args)],
        });
        let prep_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("draw-prep-pl"),
            bind_group_layouts: &[Some(&prep_bgl)],
            immediate_size: 0,
        });
        let draw_prep_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("draw-prep"),
                layout: Some(&prep_pl),
                module: &prep_module,
                entry_point: Some("draw_prep"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ---- draw ----
        let draw_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat-draw"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/splat_draw.wgsl").into()),
        });
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw-bgl"),
            entries: &[vertex_storage(0), vertex_storage(1)],
        });
        let draw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw-bg"),
            layout: &draw_bgl,
            entries: &[entry(0, &splats_2d), entry(1, sorter.payloads())],
        });
        let draw_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("draw-pl"),
            bind_group_layouts: &[Some(&draw_bgl)],
            immediate_size: 0,
        });
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat-draw"),
            layout: Some(&draw_pl),
            vertex: wgpu::VertexState {
                module: &draw_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &draw_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            scene_degree: scene.sh_degree,
            num_splats: n,
            sorter,
            camera_buf,
            draw_args,
            preprocess_pipeline,
            preprocess_bg,
            draw_prep_pipeline,
            draw_prep_bg,
            render_pipeline,
            draw_bg,
        }
    }

    /// Record one frame: preprocess → sort → draw into `target`.
    pub fn render(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        viewport: Vec2,
        settings: &RenderSettings,
    ) {
        self.write_camera(ctx, camera, viewport, settings);
        encoder.clear_buffer(self.sorter.counts(), 0, None);
        self.encode_preprocess(encoder, None);
        self.sorter.encode(encoder);
        self.encode_draw(encoder, target, settings.background, None);
    }

    /// [`render`] with per-stage GpuTimer scopes: preprocess / sort / draw.
    #[cfg(feature = "profile")]
    #[allow(clippy::too_many_arguments)] // mirrors render() + the timer
    pub fn render_profiled(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        viewport: Vec2,
        settings: &RenderSettings,
        timer: &mut gs_wgpu::GpuTimer,
    ) {
        self.write_camera(ctx, camera, viewport, settings);
        encoder.clear_buffer(self.sorter.counts(), 0, None);
        let ts = timer.compute_scope("preprocess");
        self.encode_preprocess(encoder, ts);
        self.sorter.encode_profiled(encoder, timer);
        let ts = timer.render_scope("draw");
        self.encode_draw(encoder, target, settings.background, ts);
    }

    fn write_camera(
        &self,
        ctx: &GpuContext,
        camera: &Camera,
        viewport: Vec2,
        settings: &RenderSettings,
    ) {
        let uniform = CameraUniform {
            view: camera.view_matrix().to_cols_array_2d(),
            proj: camera.proj_matrix(viewport.x / viewport.y).to_cols_array_2d(),
            cam_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            viewport: viewport.into(),
            focal: camera.focal_px(viewport).into(),
            num_splats: self.num_splats,
            sh_degree: settings.sh_degree.min(self.scene_degree) as u32,
            splat_scale: settings.splat_scale,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));
    }

    fn encode_preprocess(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("preprocess"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.preprocess_pipeline);
        pass.set_bind_group(0, &self.preprocess_bg, &[]);
        pass.dispatch_workgroups(self.num_splats.div_ceil(256), 1, 1);
        pass.set_pipeline(&self.draw_prep_pipeline);
        pass.set_bind_group(0, &self.draw_prep_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn encode_draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        background: wgpu::Color,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat-draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.draw_bg, &[]);
        pass.draw_indirect(&self.draw_args, 0);
    }
}

fn compute_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn vertex_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
