//! Headless frame-cost benchmark (the M0 perf gate without a window):
//! renders N frames at a given resolution on an orbiting camera, blocking on
//! the GPU each frame, and reports average frame cost after warmup.
//!
//! `cargo run -p gs-render --release --example bench_offscreen -- <file.ply> [width] [height]`

use glam::Vec3;
use gs_core::Camera;
use gs_render::{GpuScene, RenderSettings, SplatRenderer, offscreen};
use gs_wgpu::GpuContext;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args().skip(1);
    let ply = args.next().expect("usage: bench_offscreen <file.ply> [w] [h]");
    let w: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(2560);
    let h: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(1440);

    let gs_io::PlyContents::Splats(cloud) = gs_io::load_ply(&ply).expect("load ply") else {
        panic!("not a splat ply");
    };
    let ctx = pollster::block_on(GpuContext::new(wgpu::Backends::all())).expect("gpu");
    let scene = GpuScene::upload(&ctx, &cloud);
    let renderer = SplatRenderer::new(&ctx, &scene, offscreen::OFFSCREEN_FORMAT);

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench-target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: offscreen::OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());

    let (lo, hi) = scene.bbox;
    let center = (lo + hi) * 0.5;
    let radius = 0.5 * (hi - lo).length();
    let settings = RenderSettings::default();

    const WARMUP: usize = 30;
    const FRAMES: usize = 200;
    let mut total = std::time::Duration::ZERO;
    for i in 0..WARMUP + FRAMES {
        let angle = i as f32 * 0.01;
        let eye = center + Vec3::new(angle.sin(), 0.3, angle.cos()) * 2.0 * radius;
        let camera = Camera::look_at(eye, center, Vec3::Y);

        let start = std::time::Instant::now();
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.render(
            &ctx,
            &mut encoder,
            &view,
            &camera,
            glam::Vec2::new(w as f32, h as f32),
            &settings,
        );
        ctx.queue.submit([encoder.finish()]);
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        if i >= WARMUP {
            total += start.elapsed();
        }
    }
    let avg = total / FRAMES as u32;
    println!(
        "{} splats @ {w}x{h}: avg frame {:.2} ms  ->  {:.0} FPS (orbit, GPU-blocking, no present)",
        scene.num_splats,
        avg.as_secs_f64() * 1e3,
        1.0 / avg.as_secs_f64()
    );
}
