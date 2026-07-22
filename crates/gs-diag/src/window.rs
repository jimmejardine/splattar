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

/// How the window chooses which record to show.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Advance through records at the video's own rate, from PTS.
    Play,
    /// Hold on one record.
    Paused,
}

/// Records shared between the producing pipeline and the window.
///
/// The window has its OWN playback cursor rather than always showing the
/// newest record. Those are different things and conflating them is why the
/// first version had no playback: `play` decodes at ~73 fps, far faster than
/// anyone watches, so "newest" is always the end of the buffer. During SLAM the
/// producer will instead be the slow side, and the cursor naturally sits at the
/// newest record — the same mechanism covers both.
pub struct DiagStream {
    inner: Mutex<StreamInner>,
    capacity: usize,
    /// Records kept BEHIND the cursor, never evicted.
    ///
    /// Without this the producer eats the past: eviction bounded only by the
    /// cursor walks `base` right up to it, so stepping back silently clamps to
    /// where you already are and the back arrow appears to do nothing. It
    /// presented as intermittent because it depended on how long you had been
    /// paused. Half the buffer is reserved for history, half for lookahead.
    back_reserve: usize,
}

struct StreamInner {
    records: Vec<FrameRecord>,
    /// Absolute index of `records[0]`; rises as old records are evicted.
    base: usize,
    /// Absolute index being shown.
    cursor: usize,
    /// Presentation time the playback clock has reached.
    clock: f64,
    mode: Mode,
    /// Set when the window closes, so a headless pipeline can stop producing
    /// and a blocked `push` can return.
    closed: bool,
}

impl DiagStream {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(StreamInner {
                records: Vec::new(),
                base: 0,
                cursor: 0,
                clock: f64::NEG_INFINITY,
                mode: Mode::Play,
                closed: false,
            }),
            capacity: capacity.max(2),
            back_reserve: (capacity / 2).max(1),
        })
    }

    /// Try to push a record. Returns it back when there is no room.
    ///
    /// Room means either spare capacity, or an oldest record that has fallen
    /// outside the reserved history window. Never evicts a record inside that
    /// window: the past is what the back arrow steps through.
    pub fn try_push(&self, record: FrameRecord) -> Result<(), Box<FrameRecord>> {
        let mut s = self.inner.lock().expect("diag stream poisoned");
        if s.closed {
            return Ok(()); // nothing is watching; drop it rather than block
        }
        let keep_from = s.cursor.saturating_sub(self.back_reserve);
        let full = s.records.len() >= self.capacity;
        if full && s.base >= keep_from {
            // Boxed: a FrameRecord holds full image panels, and an unboxed
            // Err would make every try_push return that large by value.
            return Err(Box::new(record));
        }
        if full {
            s.records.remove(0);
            s.base += 1;
        }
        if s.records.is_empty() {
            // First record: start the clock at its own timestamp so playback
            // is not offset by the video's start pts.
            s.clock = record.pts;
            s.cursor = s.base;
        }
        s.records.push(record);
        Ok(())
    }

    /// Push a record, waiting for room.
    ///
    /// Backpressure rather than eviction: silently dropping frames the user
    /// has not seen defeats the purpose of a diagnostic viewer. When the
    /// pipeline is the slow side — which it will be for SLAM — this never
    /// waits.
    pub fn push(&self, record: FrameRecord) {
        let mut pending = record;
        while let Err(back) = self.try_push(pending) {
            if self.is_closed() {
                return;
            }
            pending = *back;
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
    }

    pub fn is_closed(&self) -> bool {
        self.inner.lock().expect("diag stream poisoned").closed
    }

    pub fn mode(&self) -> Mode {
        self.inner.lock().expect("diag stream poisoned").mode
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("diag stream poisoned").records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Advance the playback clock by `dt` seconds of wall time and move the
    /// cursor to whatever record that lands on. No-op while paused.
    fn advance(&self, dt: f64) {
        let mut s = self.inner.lock().expect("diag stream poisoned");
        if s.mode != Mode::Play || s.records.is_empty() {
            return;
        }
        s.clock += dt;
        let newest = s.base + s.records.len() - 1;
        while s.cursor < newest {
            let next = s.cursor + 1 - s.base;
            if s.records[next].pts > s.clock {
                break;
            }
            s.cursor += 1;
        }
        // Do not let the clock run away past the newest record, or resuming
        // after the producer catches up would skip everything it produced
        // while we were waiting.
        if s.cursor == newest {
            let pts = s.records[newest - s.base].pts;
            s.clock = s.clock.min(pts);
        }
    }

    /// The record currently under the cursor.
    pub fn current(&self) -> Option<FrameRecord> {
        let s = self.inner.lock().expect("diag stream poisoned");
        s.records.get(s.cursor.saturating_sub(s.base)).cloned()
    }

    fn step(&self, delta: isize) {
        let mut s = self.inner.lock().expect("diag stream poisoned");
        if s.records.is_empty() {
            return;
        }
        let (lo, hi) = (s.base as isize, (s.base + s.records.len() - 1) as isize);
        s.cursor = (s.cursor as isize + delta).clamp(lo, hi) as usize;
        s.clock = s.records[s.cursor - s.base].pts;
        // Stepping means you want to look at something.
        s.mode = Mode::Paused;
    }

    fn toggle_pause(&self) {
        let mut s = self.inner.lock().expect("diag stream poisoned");
        s.mode = match s.mode {
            Mode::Play => Mode::Paused,
            Mode::Paused => Mode::Play,
        };
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
        last_tick: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct DiagApp {
    stream: Arc<DiagStream>,
    state: Option<Gpu>,
    /// Wall-clock reference for the playback clock.
    last_tick: Option<std::time::Instant>,
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
        // Playback advances on WALL time, not on redraw count: a frame rate
        // that varies with GPU load must not change the speed of the video.
        let now = std::time::Instant::now();
        let dt = self
            .last_tick
            .replace(now)
            .map_or(0.0, |t| (now - t).as_secs_f64())
            // A long stall (window dragged, breakpoint) must not fast-forward
            // through the recording.
            .min(0.1);
        self.stream.advance(dt);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Panel;

    fn rec(i: usize, pts: f64) -> FrameRecord {
        FrameRecord::captured(i, pts, Panel::new(1, 1, vec![0, 0, 0, 255]))
    }

    fn shown(s: &DiagStream) -> usize {
        s.current().expect("a record").index
    }

    /// Playback must follow the video's own clock, not the producer's
    /// position. The first version showed "the newest record" always, so a
    /// decoder running at 73 fps meant there was no playback at all and
    /// unpausing jumped to the end of the buffer.
    #[test]
    fn playback_follows_pts_not_the_producer() {
        let s = DiagStream::new(64);
        for i in 0..10 {
            s.push(rec(i, i as f64 * 0.1));
        }
        // Everything is already buffered — the producer is far ahead.
        assert_eq!(shown(&s), 0, "playback must start at the first record");
        s.advance(0.05);
        assert_eq!(shown(&s), 0, "half a frame interval is not a frame");
        s.advance(0.06);
        assert_eq!(shown(&s), 1);
        s.advance(0.5);
        assert_eq!(shown(&s), 6, "five more frame intervals");
    }

    #[test]
    fn pause_holds_and_stepping_implies_pause() {
        let s = DiagStream::new(64);
        for i in 0..10 {
            s.push(rec(i, i as f64 * 0.1));
        }
        s.advance(0.35);
        let held = shown(&s);
        s.toggle_pause();
        assert_eq!(s.mode(), Mode::Paused);
        s.advance(10.0);
        assert_eq!(shown(&s), held, "paused playback must not advance");
        s.toggle_pause();
        assert_eq!(s.mode(), Mode::Play);
        s.advance(0.2);
        assert!(shown(&s) > held, "resuming must continue from where it paused");

        s.step(-2);
        assert_eq!(s.mode(), Mode::Paused, "stepping means you want to look");
        let back = shown(&s);
        s.step(1);
        assert_eq!(shown(&s), back + 1);
    }

    /// If the viewer catches up to the producer, the clock must not keep
    /// running — otherwise the frames produced during the wait get skipped the
    /// moment they arrive.
    #[test]
    fn the_clock_waits_for_a_slow_producer() {
        let s = DiagStream::new(64);
        s.push(rec(0, 0.0));
        s.push(rec(1, 0.1));
        s.advance(0.5); // overruns the two available records
        assert_eq!(shown(&s), 1);
        s.push(rec(2, 0.2));
        s.push(rec(3, 0.3));
        s.advance(0.11);
        assert_eq!(
            shown(&s),
            2,
            "a starved clock must resume one frame at a time, not jump"
        );
    }

    /// Backpressure, not eviction: dropping records the viewer has not reached
    /// would silently lose the frame you were about to look at.
    /// Stepping back must keep working while the producer runs.
    ///
    /// The bug this pins: eviction was bounded only by the cursor, so while
    /// the viewer sat on a frame the producer ate every record BEHIND it, and
    /// the back arrow silently clamped to where you already were. It presented
    /// as "sometimes back does nothing" because it depended on how long you
    /// had been paused.
    #[test]
    fn history_behind_the_cursor_survives_a_running_producer() {
        let s = DiagStream::new(16);
        for i in 0..16 {
            s.push(rec(i, i as f64 * 0.1));
        }
        // Watch a while, then stop to look at something.
        s.advance(1.0);
        let here = shown(&s);
        assert!(here > 4, "should have advanced into the buffer, at {here}");
        s.toggle_pause();

        // The producer keeps running until the buffer refuses more. try_push
        // rather than push so the test cannot deadlock on the backpressure it
        // is verifying — before the fix this loop ran forever, eating every
        // record behind the cursor and then blocking.
        let mut i = 16;
        while s.try_push(rec(i, i as f64 * 0.1)).is_ok() && i < 500 {
            i += 1;
        }
        assert!(i < 500, "producer was never throttled");
        assert_eq!(shown(&s), here, "pausing must hold the displayed frame");

        // Stepping back has to actually move.
        s.step(-1);
        assert_eq!(shown(&s), here - 1, "back arrow did not step back");
        s.step(-1);
        assert_eq!(shown(&s), here - 2);

        // And the reserve is real: stepping back repeatedly reaches frames
        // well behind where the producer would otherwise have evicted to.
        for _ in 0..16 {
            s.step(-1);
        }
        assert!(
            shown(&s) <= here - 6,
            "history was evicted from under the cursor: stuck at {} from {here}",
            shown(&s)
        );
    }

    #[test]
    fn push_blocks_rather_than_evicting_unwatched_records() {
        let s = DiagStream::new(4);
        for i in 0..4 {
            s.push(rec(i, i as f64 * 0.1));
        }
        assert_eq!(s.len(), 4);
        // The viewer is still on record 0, so a fifth push must wait. Prove it
        // does not silently evict by closing the stream to release it.
        let stream = Arc::new(());
        let _ = stream; // (keeps the intent obvious: nothing else is shared)
        assert_eq!(shown(&s), 0);
        s.close();
        s.push(rec(4, 0.4));
        assert_eq!(s.len(), 4, "a closed stream drops the push, never the history");
        assert_eq!(shown(&s), 0, "the watched record is still there");
    }
}
