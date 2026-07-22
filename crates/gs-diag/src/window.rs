//! Live diagnostic window: shows [`FrameRecord`]s as they arrive.
//!
//! Deliberately a passive CONSUMER of records. The pipeline pushes records and
//! never learns whether anything is displaying them, so every stage stays
//! runnable headless (CLAUDE.md) and the same records can be replayed from a
//! trace with no pipeline running at all.
//!
//! Panels are blitted as textures rather than drawn through the splat
//! renderer: this window has to work at M0, before any renderer exists, and
//! must keep working when the renderer is the thing being debugged.

use std::sync::{Arc, Mutex};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::record::FrameRecord;

/// Records shared between the producing pipeline and the window.
///
/// Bounded: a fast producer must not grow this without limit, and old records
/// are the ones you least want to keep if something has to be dropped — the
/// live window is for watching the present. The disk trace is what preserves
/// history.
pub struct DiagStream {
    inner: Mutex<StreamInner>,
    capacity: usize,
}

struct StreamInner {
    records: Vec<FrameRecord>,
    /// Which record the window is showing. `None` = follow the newest.
    cursor: Option<usize>,
    paused: bool,
    /// Set by the window when the user closes it, so a headless pipeline can
    /// notice and stop producing.
    closed: bool,
}

impl DiagStream {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(StreamInner {
                records: Vec::new(),
                cursor: None,
                paused: false,
                closed: false,
            }),
            capacity: capacity.max(1),
        })
    }

    /// Push a record. Never blocks the pipeline, even with no window attached.
    pub fn push(&self, record: FrameRecord) {
        let mut s = self.inner.lock().expect("diag stream poisoned");
        if s.records.len() == self.capacity {
            s.records.remove(0);
            // Keep the cursor on the same record rather than letting it slide
            // onto a different frame while the user is looking at it.
            if let Some(c) = s.cursor.as_mut() {
                *c = c.saturating_sub(1);
            }
        }
        s.records.push(record);
    }

    pub fn is_paused(&self) -> bool {
        self.inner.lock().expect("diag stream poisoned").paused
    }

    pub fn is_closed(&self) -> bool {
        self.inner.lock().expect("diag stream poisoned").closed
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("diag stream poisoned").records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The record currently under the cursor (or the newest when following).
    pub fn current(&self) -> Option<FrameRecord> {
        let s = self.inner.lock().expect("diag stream poisoned");
        let i = s.cursor.unwrap_or(s.records.len().saturating_sub(1));
        s.records.get(i).cloned()
    }

    fn step(&self, delta: isize) {
        let mut s = self.inner.lock().expect("diag stream poisoned");
        if s.records.is_empty() {
            return;
        }
        let last = s.records.len() - 1;
        let cur = s.cursor.unwrap_or(last) as isize;
        s.cursor = Some((cur + delta).clamp(0, last as isize) as usize);
        // Stepping implies you want to look at something: stop following.
        s.paused = true;
    }

    fn toggle_pause(&self) {
        let mut s = self.inner.lock().expect("diag stream poisoned");
        s.paused = !s.paused;
        if !s.paused {
            s.cursor = None; // resume following the newest
        }
    }

    fn close(&self) {
        self.inner.lock().expect("diag stream poisoned").closed = true;
    }
}

/// Run the window on the calling thread until it is closed.
///
/// winit requires the event loop on the main thread on some platforms, so the
/// pipeline is expected to run on a worker and call this from `main`.
pub fn run(stream: Arc<DiagStream>) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = DiagApp {
        stream,
        state: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct DiagApp {
    stream: Arc<DiagStream>,
    state: Option<Gpu>,
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    blit: Blit,
    /// Index of the record currently uploaded, so identical frames are not
    /// re-uploaded while paused.
    shown: Option<usize>,
}

impl ApplicationHandler for DiagApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match pollster::block_on(Gpu::new(event_loop)) {
            Ok(g) => self.state = Some(g),
            Err(e) => {
                log::error!("diagnostic window failed to start: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => {
                self.stream.close();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.config.width = size.width.max(1);
                gpu.config.height = size.height.max(1);
                gpu.surface.configure(&gpu.device, &gpu.config);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match logical_key {
                Key::Named(NamedKey::Space) => self.stream.toggle_pause(),
                Key::Named(NamedKey::ArrowLeft) => self.stream.step(-1),
                Key::Named(NamedKey::ArrowRight) => self.stream.step(1),
                Key::Named(NamedKey::Home) => self.stream.step(isize::MIN / 2),
                Key::Named(NamedKey::End) => self.stream.step(isize::MAX / 2),
                Key::Named(NamedKey::Escape) => {
                    self.stream.close();
                    event_loop.exit();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                if let Err(e) = gpu.draw(&self.stream) {
                    log::warn!("diagnostic draw failed: {e:#}");
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(g) = self.state.as_ref() {
            g.window.request_redraw();
        }
    }
}

impl Gpu {
    async fn new(event_loop: &ActiveEventLoop) -> anyhow::Result<Self> {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("splattar — diagnostics  [space] pause  [←/→] step")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
        )?);
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("diag"),
                ..Default::default()
            })
            .await?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let blit = Blit::new(&device, format);
        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            blit,
            shown: None,
        })
    }

    fn draw(&mut self, stream: &DiagStream) -> anyhow::Result<()> {
        let Some(rec) = stream.current() else {
            return Ok(());
        };
        if self.shown != Some(rec.index) {
            // Panels present in this record, left to right. Absent ones are
            // simply not shown rather than drawn as placeholders — an empty
            // slot would read as "the model rendered nothing".
            let panels: Vec<&crate::record::Panel> = [
                Some(&rec.frame),
                rec.render.as_ref(),
                rec.error.as_ref(),
            ]
            .into_iter()
            .flatten()
            .collect();
            self.blit.upload(&self.device, &self.queue, &panels);
            self.shown = Some(rec.index);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            _ => return Ok(()),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.blit.draw(&self.queue, &mut encoder, &view);
        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(frame);
        Ok(())
    }
}

const SHADER: &str = r#"
struct Uni { rect: vec4<f32> }
@group(0) @binding(0) var<uniform> uni: Uni;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var c = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let k = c[vi];
    var o: VsOut;
    o.pos = vec4<f32>(mix(uni.rect.x, uni.rect.z, k.x), mix(uni.rect.w, uni.rect.y, k.y), 0.0, 1.0);
    o.uv = vec2<f32>(k.x, 1.0 - k.y);
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

/// Blits N panels side by side, each letterboxed into an equal column.
struct Blit {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// One (uniform, bind group, aspect) per panel currently uploaded.
    panels: Vec<(wgpu::Buffer, wgpu::BindGroup, f32)>,
}

impl Blit {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diag-blit"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("diag-blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
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
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("diag-blit"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("diag-blit"),
            layout: Some(&pl),
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
                targets: &[Some(format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("diag-blit"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
            panels: Vec::new(),
        }
    }

    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, panels: &[&crate::record::Panel]) {
        self.panels.clear();
        for p in panels {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("diag-panel"),
                size: wgpu::Extent3d {
                    width: p.width,
                    height: p.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &p.rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(p.width * 4),
                    rows_per_image: Some(p.height),
                },
                wgpu::Extent3d {
                    width: p.width,
                    height: p.height,
                    depth_or_array_layers: 1,
                },
            );
            let tv = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("diag-panel-uni"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("diag-panel"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&tv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let aspect = p.width as f32 / p.height.max(1) as f32;
            self.panels.push((uniform, bg, aspect));
        }
    }

    fn draw(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        if self.panels.is_empty() {
            return;
        }
        let n = self.panels.len() as f32;
        let col_w = 2.0 / n;
        for (i, (uniform, _, aspect)) in self.panels.iter().enumerate() {
            // Letterbox inside the column: panels have the video's aspect and
            // the window rarely matches it.
            let x0 = -1.0 + i as f32 * col_w;
            let col_aspect = col_w / 2.0; // column w/h in clip units
            let (w, h) = if *aspect > col_aspect {
                (col_w, col_w / *aspect)
            } else {
                (2.0 * *aspect, 2.0)
            };
            let cx = x0 + col_w * 0.5;
            let u: [f32; 4] = [cx - w * 0.5, h * 0.5, cx + w * 0.5, -h * 0.5];
            queue.write_buffer(uniform, 0, bytemuck::cast_slice(&u));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("diag-blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (_, bg, _) in &self.panels {
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}
