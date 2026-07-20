//! Offscreen rendering through the exact same SplatRenderer as the window —
//! mandatory for golden-test validity. Blocking readback; never on a frame loop.

use glam::Vec2;
use gs_core::Camera;
use gs_wgpu::GpuContext;

use crate::pipeline::{RenderSettings, SplatRenderer};

pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Render one frame at `width`×`height` and return tightly-packed RGBA8 bytes.
pub fn render_to_rgba(
    ctx: &GpuContext,
    renderer: &SplatRenderer,
    camera: &Camera,
    width: u32,
    height: u32,
    settings: &RenderSettings,
) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // wgpu requires 256-byte row alignment for texture→buffer copies.
    let unpadded = width as usize * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen-readback"),
        size: (padded * height as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    renderer.render(
        ctx,
        &mut encoder,
        &view,
        camera,
        Vec2::new(width as f32, height as f32),
        settings,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("map failed");
    let data = slice.get_mapped_range().expect("mapped range");

    let mut rgba = Vec::with_capacity(unpadded * height as usize);
    for row in 0..height as usize {
        rgba.extend_from_slice(&data[row * padded..row * padded + unpadded]);
    }
    drop(data);
    readback.unmap();
    rgba
}
