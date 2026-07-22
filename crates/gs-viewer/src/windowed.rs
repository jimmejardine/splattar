//! winit window + event loop around the shared render pipeline. Kept behind
//! the default `winit` feature; app-web and app-vr bring their own surfaces.

use std::sync::Arc;
use std::time::Instant;

use glam::{Quat, Vec2};
use gs_core::SplatCloud;
use gs_render::{GpuScene, RenderSettings, SplatRenderer};
use gs_wgpu::GpuContext;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera::FlyCamera;
use crate::input::InputState;
use crate::overlay::{ThumbImage, ThumbOverlay};

pub struct ViewOptions {
    pub backends: wgpu::Backends,
    pub vsync: bool,
    pub width: u32,
    pub height: u32,
    pub sh_degree: u8,
    pub splat_scale: f32,
    /// Rotate the scene 180° about z (COLMAP-convention data renders upside
    /// down in a y-up viewer). On by default.
    pub flip_scene: bool,
    pub title: String,
    /// Recorded camera path in SCENE space (renderer convention), e.g. the
    /// training keyframe poses of a project. When non-empty the viewer spawns
    /// at the first pose instead of framing the bbox (which floaters skew),
    /// and `[` / `]` snap along the path — walk exactly what the video saw.
    /// Each pose carries the capture camera's vertical FOV (per submap, from
    /// its refined focal) so a snapped view reprojects pixel-exactly.
    pub spawn_cameras: Vec<(glam::Vec3, Quat, f32)>,
    /// Per spawn pose: index into `thumbs` of the nearest keyframe thumbnail
    /// (the frame the phone actually captured there) and whether that thumb
    /// was captured at EXACTLY this pose's keyframe (thumbs land every 4th).
    pub spawn_thumbs: Vec<Option<(usize, bool)>>,
    /// Deduplicated keyframe thumbnails referenced by `spawn_thumbs`. `T`
    /// cycles PIP → semi-transparent blend → off while walking the path.
    pub thumbs: Vec<ThumbImage>,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL,
            vsync: false,
            width: 1600,
            height: 900,
            sh_degree: 3,
            splat_scale: 1.0,
            flip_scene: true,
            title: "splattar".to_string(),
            spawn_cameras: Vec::new(),
            spawn_thumbs: Vec::new(),
            thumbs: Vec::new(),
        }
    }
}

pub fn run(cloud: SplatCloud, options: ViewOptions) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    let mut app = App {
        cloud: Some(cloud),
        options,
        state: None,
        input: InputState::default(),
        locked: false,
        spawn_idx: 0,
    };
    event_loop.run_app(&mut app).map_err(|e| e.to_string())
}

struct GfxState {
    window: Arc<Window>,
    ctx: GpuContext,
    overlay: Option<ThumbOverlay>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: SplatRenderer,
    camera: FlyCamera,
    /// While Some, render EXACTLY this recorded pose (scene space) with the
    /// capture FOV — the ground-truth blend then matches the render
    /// pixel-for-pixel (roll included, which the yaw/pitch fly camera cannot
    /// represent). Any movement input releases the lock back to free flight.
    exact_pose: Option<(glam::Vec3, Quat, f32)>,
    /// Upright-space anchor points for zoom-proportional movement: the
    /// recorded camera path when there is one (it traces the actual content;
    /// the bbox is floater-skewed), else the bbox center. WASD speed scales
    /// with the distance to the nearest anchor.
    nav_anchors: Vec<glam::Vec3>,
    nav_floor: f32,
    scene_rot: Quat,
    settings: RenderSettings,
    num_splats: u32,
    last_frame: Instant,
    fps_window: Vec<f32>,
    last_title: Instant,
}

struct App {
    cloud: Option<SplatCloud>,
    options: ViewOptions,
    state: Option<GfxState>,
    input: InputState,
    locked: bool,
    /// Current index into `options.spawn_cameras` for `[` / `]` stepping.
    spawn_idx: usize,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) {
        let Some(cloud) = self.cloud.take() else {
            return; // already initialized
        };
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(&self.options.title)
                        .with_inner_size(winit::dpi::PhysicalSize::new(
                            self.options.width,
                            self.options.height,
                        )),
                )
                .expect("create window"),
        );

        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle().with_env();
        desc.backends = self.options.backends;
        let instance = wgpu::Instance::new(desc);
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let ctx = pollster::block_on(GpuContext::with_instance(instance, Some(&surface)))
            .expect("gpu context");

        let caps = surface.get_capabilities(&ctx.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let present_mode = if self.options.vsync {
            wgpu::PresentMode::Fifo
        } else if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        log::info!("surface: {format:?}, present mode: {present_mode:?}");

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
        };
        surface.configure(&ctx.device, &config);

        let scene = GpuScene::upload(&ctx, &cloud);
        let renderer = SplatRenderer::new(&ctx, &scene, format);
        // Ground-truth thumbnail overlay for the recorded path (T toggles).
        let mut overlay = (!self.options.thumbs.is_empty())
            .then(|| ThumbOverlay::new(&ctx, format));
        if let (Some(ov), Some(Some((ti, exact)))) =
            (overlay.as_mut(), self.options.spawn_thumbs.first())
        {
            ov.set_image(&ctx, &self.options.thumbs[*ti]);
            ov.exact = *exact;
        }
        let scene_rot = if self.options.flip_scene {
            Quat::from_rotation_z(std::f32::consts::PI)
        } else {
            Quat::IDENTITY
        };
        // Frame the bbox in upright space: rotate the scene bbox corners.
        let (lo, hi) = scene.bbox;
        let corners = [scene_rot * lo, scene_rot * hi];
        let bbox = (corners[0].min(corners[1]), corners[0].max(corners[1]));
        let mut camera = FlyCamera::framing(bbox);
        // A recorded camera path beats bbox framing: floaters skew the bbox
        // far off the content, while the path starts exactly where the video
        // did. Movement speed drops to a walkable fraction of the scene.
        let mut exact_pose = None;
        if let Some(&(pos, rot, fov)) = self.options.spawn_cameras.first() {
            camera.snap_to(pos, rot, scene_rot);
            exact_pose = Some((pos, rot, fov));
            log::info!(
                "spawned on the recorded camera path ({} poses; [ / ] to step along it, T for ground truth)",
                self.options.spawn_cameras.len()
            );
        }

        // Zoom-proportional movement anchors (subsampled path, upright space).
        let nav_anchors: Vec<glam::Vec3> = if self.options.spawn_cameras.is_empty() {
            vec![(bbox.0 + bbox.1) * 0.5]
        } else {
            self.options
                .spawn_cameras
                .iter()
                .step_by(4)
                .map(|&(p, _, _)| scene_rot * p)
                .collect()
        };
        let nav_floor = (0.01 * (bbox.1 - bbox.0).length()).clamp(0.02, 2.0);

        self.state = Some(GfxState {
            window,
            ctx,
            overlay,
            surface,
            config,
            renderer,
            camera,
            exact_pose,
            nav_anchors,
            nav_floor,
            scene_rot,
            settings: RenderSettings {
                sh_degree: self.options.sh_degree,
                splat_scale: self.options.splat_scale,
                ..Default::default()
            },
            num_splats: scene.num_splats,
            last_frame: Instant::now(),
            fps_window: Vec::with_capacity(120),
            last_title: Instant::now(),
        });
    }

    fn set_locked(&mut self, locked: bool) {
        let Some(state) = &self.state else { return };
        if locked {
            let ok = state
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            if ok {
                state.window.set_cursor_visible(false);
                self.locked = true;
            }
        } else {
            let _ = state.window.set_cursor_grab(CursorGrabMode::None);
            state.window.set_cursor_visible(true);
            self.locked = false;
        }
    }

    fn frame(&mut self) {
        let Some(state) = &mut self.state else { return };
        let dt = state.last_frame.elapsed().as_secs_f32().min(0.1);
        state.last_frame = Instant::now();

        let moving = self.input.forward
            || self.input.back
            || self.input.left
            || self.input.right
            || self.input.up
            || self.input.down
            || self.input.mouse_dx != 0.0
            || self.input.mouse_dy != 0.0;
        if self.locked && moving {
            state.exact_pose = None; // hand control back to free flight
        }
        if self.locked {
            // Zoom-proportional speed: a WASD tap far from the content
            // ("zoomed out") covers much more ground than one up close.
            let pos = state.camera.position;
            let d = state
                .nav_anchors
                .iter()
                .map(|a| (pos - *a).length())
                .fold(f32::INFINITY, f32::min);
            state.camera.speed = (d.max(state.nav_floor)).min(1e4);
            state.camera.update(dt, &self.input);
        }
        self.input.end_frame();

        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match state.surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            Cst::Outdated | Cst::Lost => {
                state.surface.configure(&state.ctx.device, &state.config);
                return;
            }
            Cst::Timeout | Cst::Occluded => return,
            other => {
                log::error!("surface acquire failed: {other:?}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = state
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // A locked recorded pose bypasses the fly camera entirely: exact
        // position, exact rotation (roll included), capture FOV.
        let camera = match state.exact_pose {
            Some((pos, rot, fov)) => gs_core::Camera {
                position: pos,
                rotation: rot,
                fov_y: fov,
                ..Default::default()
            },
            None => state.camera.to_camera(state.scene_rot),
        };
        state.renderer.render(
            &state.ctx,
            &mut encoder,
            &view,
            &camera,
            Vec2::new(state.config.width as f32, state.config.height as f32),
            &state.settings,
        );
        if let Some(ov) = &state.overlay {
            // Full blend strength only when the render provably shows the
            // thumbnail's own frame: pose lock held AND the thumb was
            // captured at exactly this keyframe. Otherwise fade it.
            let aligned = state.exact_pose.is_some() && ov.exact;
            ov.draw(
                &state.ctx,
                &mut encoder,
                &view,
                (state.config.width as f32, state.config.height as f32),
                aligned,
            );
        }
        state.ctx.queue.submit([encoder.finish()]);
        state.window.pre_present_notify();
        state.ctx.queue.present(frame);

        state.fps_window.push(dt);
        if state.last_title.elapsed().as_secs_f32() > 0.25 && !state.fps_window.is_empty() {
            let avg = state.fps_window.iter().sum::<f32>() / state.fps_window.len() as f32;
            state.fps_window.clear();
            state.last_title = Instant::now();
            state.window.set_title(&format!(
                "{} — {} splats — {:.1} ms / {:.0} FPS — SH deg {} — {}",
                self.options.title,
                state.num_splats,
                avg * 1e3,
                1.0 / avg.max(1e-6),
                state.settings.sh_degree,
                if self.locked { "esc to release" } else { "click to fly" },
            ));
        }
    }

    fn handle_key(&mut self, code: KeyCode, pressed: bool, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => self.input.forward = pressed,
            KeyCode::KeyS | KeyCode::ArrowDown => self.input.back = pressed,
            KeyCode::KeyA | KeyCode::ArrowLeft => self.input.left = pressed,
            KeyCode::KeyD | KeyCode::ArrowRight => self.input.right = pressed,
            KeyCode::KeyE | KeyCode::Space => self.input.up = pressed,
            KeyCode::KeyQ | KeyCode::ControlLeft => self.input.down = pressed,
            KeyCode::ShiftLeft => self.input.sprint = pressed,
            KeyCode::Escape if pressed => {
                if self.locked {
                    self.set_locked(false);
                } else {
                    event_loop.exit();
                }
            }
            KeyCode::Digit0 | KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3
                if pressed =>
            {
                if let Some(state) = &mut self.state {
                    state.settings.sh_degree = match code {
                        KeyCode::Digit0 => 0,
                        KeyCode::Digit1 => 1,
                        KeyCode::Digit2 => 2,
                        _ => 3,
                    };
                }
            }
            // Toggle the ground-truth thumbnail: PIP -> blend -> off.
            KeyCode::KeyT if pressed => {
                if let Some(ov) = self.state.as_mut().and_then(|s| s.overlay.as_mut()) {
                    ov.mode = ov.mode.next();
                    log::info!("thumbnail overlay: {:?}", ov.mode);
                }
            }
            // Step along the recorded camera path (replay the walkthrough).
            KeyCode::BracketLeft | KeyCode::BracketRight
                if pressed && !self.options.spawn_cameras.is_empty() =>
            {
                let n = self.options.spawn_cameras.len();
                self.spawn_idx = match code {
                    KeyCode::BracketRight => (self.spawn_idx + 1) % n,
                    _ => (self.spawn_idx + n - 1) % n,
                };
                if let Some(state) = &mut self.state {
                    let (pos, rot, fov) = self.options.spawn_cameras[self.spawn_idx];
                    state.camera.snap_to(pos, rot, state.scene_rot);
                    state.exact_pose = Some((pos, rot, fov));
                    if let (Some(ov), Some(Some((ti, exact)))) = (
                        state.overlay.as_mut(),
                        self.options.spawn_thumbs.get(self.spawn_idx),
                    ) {
                        ov.set_image(&state.ctx, &self.options.thumbs[*ti]);
                        ov.exact = *exact;
                    }
                    log::info!("camera path pose {}/{n}", self.spawn_idx + 1);
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.init(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.config.width = size.width.max(1);
                    state.config.height = size.height.max(1);
                    state.surface.configure(&state.ctx.device, &state.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.handle_key(code, event.state == ElementState::Pressed, event_loop);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } if !self.locked => self.set_locked(true),
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                };
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.locked
        {
            self.input.mouse_dx += delta.0 as f32;
            self.input.mouse_dy += delta.1 as f32;
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
