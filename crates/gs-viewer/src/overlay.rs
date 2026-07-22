//! Reference-thumbnail overlay: while walking the recorded camera path, show
//! the frame the phone actually captured at the nearest keyframe — as a
//! picture-in-picture bottom-right, or as a semi-transparent blend over the
//! render for direct ground-truth comparison. Toggled with `T`.

use gs_wgpu::GpuContext;

/// Grayscale keyframe thumbnail (row 0 = top).
pub struct ThumbImage {
    pub width: u32,
    pub height: u32,
    pub gray: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OverlayMode {
    Off,
    /// Bottom-right picture-in-picture.
    Pip,
    /// Semi-transparent, fit to the window height (the tracking FOV is close
    /// to the fly camera's 60°, so a snapped pose lines up approximately).
    Blend,
}

impl OverlayMode {
    pub fn next(self) -> Self {
        match self {
            OverlayMode::Off => OverlayMode::Pip,
            OverlayMode::Pip => OverlayMode::Blend,
            OverlayMode::Blend => OverlayMode::Off,
        }
    }
}

const SHADER: &str = r#"
struct Uni {
    rect: vec4<f32>,   // x0, y0, x1, y1 in clip space
    params: vec4<f32>, // x = alpha
}
@group(0) @binding(0) var<uniform> uni: Uni;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    // Two triangles over the uniform rect; uv v=0 at the top.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vi];
    var out: VsOut;
    let x = mix(uni.rect.x, uni.rect.z, corner.x);
    let y = mix(uni.rect.w, uni.rect.y, corner.y); // rect.y = top, rect.w = bottom
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let g = textureSample(tex, samp, in.uv).r;
    return vec4<f32>(g, g, g, uni.params.x);
}
"#;

pub struct ThumbOverlay {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    bind: Option<wgpu::BindGroup>,
    tex_size: (u32, u32),
    aspect: f32,
    pub mode: OverlayMode,
    /// Whether the current image was captured at exactly the pose being
    /// rendered (set by the caller; controls blend strength).
    pub exact: bool,
}

impl ThumbOverlay {
    pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thumb-overlay"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thumb-overlay"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thumb-overlay"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("thumb-overlay"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("thumb-overlay"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thumb-overlay-uni"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            sampler,
            uniform,
            bind: None,
            tex_size: (0, 0),
            aspect: 1.0,
            mode: OverlayMode::Pip,
            exact: false,
        }
    }

    /// Upload the current thumbnail (bytes_per_row padded to wgpu's 256).
    pub fn set_image(&mut self, ctx: &GpuContext, img: &ThumbImage) {
        let device = &ctx.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("thumb-overlay-tex"),
            size: wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let padded = (img.width as usize).next_multiple_of(256);
        let mut rows = vec![0u8; padded * img.height as usize];
        for y in 0..img.height as usize {
            let src = &img.gray[y * img.width as usize..(y + 1) * img.width as usize];
            rows[y * padded..y * padded + img.width as usize].copy_from_slice(src);
        }
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rows,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(img.height),
            },
            wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.bind = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thumb-overlay"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
        self.tex_size = (img.width, img.height);
        self.aspect = img.width as f32 / img.height.max(1) as f32;
    }

    pub fn draw(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        window: (f32, f32),
        aligned: bool,
    ) {
        let Some(bind) = &self.bind else { return };
        if self.mode == OverlayMode::Off {
            return;
        }
        let (ww, wh) = window;
        // Rect in pixels → clip space. Pip: 38% of window height, 12 px
        // margin bottom-right. Blend: fit by height, centered.
        let (x0, y0, x1, y1, alpha) = match self.mode {
            OverlayMode::Pip => {
                let h = 0.38 * wh;
                let w = h * self.aspect;
                (ww - w - 12.0, wh - h - 12.0, ww - 12.0, wh - 12.0, 1.0)
            }
            OverlayMode::Blend => {
                let h = wh;
                let w = h * self.aspect;
                let x = 0.5 * (ww - w);
                // Faint when the view only approximates the thumbnail's
                // frame; full strength when it IS that frame.
                (x, 0.0, x + w, h, if aligned { 0.45 } else { 0.18 })
            }
            OverlayMode::Off => unreachable!(),
        };
        let to_clip_x = |p: f32| p / ww * 2.0 - 1.0;
        let to_clip_y = |p: f32| 1.0 - p / wh * 2.0;
        let uni: [f32; 8] = [
            to_clip_x(x0),
            to_clip_y(y0), // top
            to_clip_x(x1),
            to_clip_y(y1), // bottom
            alpha,
            0.0,
            0.0,
            0.0,
        ];
        ctx.queue
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&uni));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("thumb-overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..6, 0..1);
    }
}
