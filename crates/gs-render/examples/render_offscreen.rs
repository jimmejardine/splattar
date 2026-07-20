//! Offscreen smoke test:
//! `cargo run -p gs-render --release --example render_offscreen -- <file.ply> <out.png>`

use glam::Vec3;
use gs_core::Camera;
use gs_render::{GpuScene, RenderSettings, SplatRenderer, offscreen};
use gs_wgpu::GpuContext;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args().skip(1);
    let ply = args.next().expect("usage: render_offscreen <file.ply> <out.png>");
    let out = args.next().expect("usage: render_offscreen <file.ply> <out.png>");

    let gs_io::PlyContents::Splats(cloud) = gs_io::load_ply(&ply).expect("load ply") else {
        panic!("not a splat ply");
    };

    let ctx = pollster::block_on(GpuContext::new(wgpu::Backends::all())).expect("gpu");
    let scene = GpuScene::upload(&ctx, &cloud);
    let renderer = SplatRenderer::new(&ctx, &scene, offscreen::OFFSCREEN_FORMAT);

    let (lo, hi) = scene.bbox;
    let center = (lo + hi) * 0.5;
    let radius = 0.5 * (hi - lo).length();
    let camera = Camera {
        position: center + Vec3::new(0.0, 0.0, 2.2 * radius),
        ..Default::default()
    };

    let (w, h) = (800u32, 600u32);
    let rgba = offscreen::render_to_rgba(&ctx, &renderer, &camera, w, h, &RenderSettings::default());
    image::RgbaImage::from_raw(w, h, rgba)
        .expect("image size")
        .save(&out)
        .expect("save png");
    println!("wrote {out}");
}
