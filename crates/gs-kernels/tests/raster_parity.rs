//! GPU forward rasterizer vs the f64 CPU oracle: per-pixel color agreement on
//! randomized micro-scenes. Tolerances absorb f32 arithmetic, not structure —
//! a wrong branch or sort order fails immediately.

mod common;

use gs_kernels::{Rasterizer, SceneInput};
use gs_wgpu::GpuContext;

fn context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(wgpu::Backends::all())) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPING GPU raster tests: {e}");
            None
        }
    }
}

fn run_parity(ctx: &GpuContext, seed: u64, n: usize, deg: u8, size: usize) {
    let scene = common::make_scene(seed, n, deg, size);
    let cpu = gs_cpu_ref::render(&scene);
    let gpu_data = common::to_gpu(&scene);

    let raster = Rasterizer::new(ctx, n as u32, size as u32, size as u32, (n * 64) as u32);
    raster.upload_scene(
        ctx,
        &SceneInput {
            positions: &gpu_data.positions,
            scales: &gpu_data.scales,
            quats: &gpu_data.quats,
            opacities: &gpu_data.opacities,
            sh: &gpu_data.sh,
            sh_coeffs: gpu_data.sh_coeffs,
        },
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    raster.forward(ctx, &mut encoder, &gpu_data.camera, n as u32);
    ctx.queue.submit([encoder.finish()]);

    let color: Vec<[f32; 4]> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
        &ctx.device,
        &ctx.queue,
        &raster.out_color,
    ))
    .to_vec();

    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;
    for (i, (gpu_px, cpu_px)) in color.iter().zip(cpu.color.iter()).enumerate() {
        for ch in 0..3 {
            let diff = (gpu_px[ch] as f64 - cpu_px[ch]).abs();
            sum_diff += diff;
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 3e-3,
                "seed {seed}: pixel {i} ch {ch}: gpu {} vs cpu {} (diff {diff:.2e})",
                gpu_px[ch],
                cpu_px[ch]
            );
        }
    }
    let mean = sum_diff / (color.len() * 3) as f64;
    eprintln!("parity seed={seed} n={n} deg={deg} {size}x{size}: max {max_diff:.2e}, mean {mean:.2e}");
    assert!(mean < 3e-4, "mean diff too high: {mean:.2e}");
}

#[test]
fn forward_matches_cpu_oracle() {
    let Some(ctx) = context() else { return };
    run_parity(&ctx, 101, 20, 0, 64);
    run_parity(&ctx, 202, 20, 1, 64);
    run_parity(&ctx, 303, 12, 3, 48);
    run_parity(&ctx, 404, 50, 1, 128); // overfit-demo shape
}
