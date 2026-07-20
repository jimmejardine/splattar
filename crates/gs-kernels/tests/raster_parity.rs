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

// ---------------------------------------------------------------- backward

struct GradCheck {
    worst: f64,
    worst_label: String,
}

impl GradCheck {
    fn new() -> Self {
        Self {
            worst: 0.0,
            worst_label: String::new(),
        }
    }
    fn check(&mut self, label: &str, gpu: f64, cpu: f64) {
        let denom = gpu.abs().max(cpu.abs());
        let err = (gpu - cpu).abs();
        let rel = if denom > 1e-6 { err / denom } else { 0.0 };
        if err > 1e-6 && rel > self.worst {
            self.worst = rel;
            self.worst_label = label.to_string();
        }
        assert!(
            err <= 1e-5 + 2e-3 * denom,
            "{label}: gpu {gpu:.6e} vs cpu {cpu:.6e} (rel {rel:.2e})"
        );
    }
}

fn run_grad_parity(ctx: &GpuContext, seed: u64, n: usize, deg: u8, size: usize) {
    let scene = common::make_scene(seed, n, deg, size);
    // Fixed random per-pixel loss weights, shared with the CPU analytic pass.
    let mut rng = seed ^ 0xfeed;
    let weights: Vec<glam::DVec3> = (0..size * size)
        .map(|_| {
            glam::DVec3::new(
                common::uniform(&mut rng, -1.0, 1.0),
                common::uniform(&mut rng, -1.0, 1.0),
                common::uniform(&mut rng, -1.0, 1.0),
            )
        })
        .collect();
    let cpu = gs_cpu_ref::gradients(&scene, &weights);

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
    let dl: Vec<[f32; 4]> = weights
        .iter()
        .map(|w| [w.x as f32, w.y as f32, w.z as f32, 0.0])
        .collect();
    ctx.queue
        .write_buffer(&raster.dl_dcolor, 0, bytemuck::cast_slice(&dl));

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    raster.forward(ctx, &mut encoder, &gpu_data.camera, n as u32);
    raster.backward(&mut encoder, n as u32);
    ctx.queue.submit([encoder.finish()]);

    let read_f32 = |buf: &wgpu::Buffer| -> Vec<f32> {
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, buf)).to_vec()
    };
    let g_pos = read_f32(&raster.grad_pos);
    let g_scales = read_f32(&raster.grad_scales);
    let g_quat = read_f32(&raster.grad_quat);
    let g_op = read_f32(&raster.grad_opacity);
    let g_sh = read_f32(&raster.grad_sh);
    let g_cam = read_f32(&raster.grad_cam);

    let mut ck = GradCheck::new();
    let n_coeffs = gs_cpu_ref::math::num_coeffs(deg);
    for i in 0..n {
        for dim in 0..3 {
            ck.check(&format!("pos[{i}][{dim}]"), g_pos[i * 4 + dim] as f64, cpu.pos[i][dim]);
        }
        for dim in 0..2 {
            ck.check(
                &format!("scale[{i}][{dim}]"),
                g_scales[i * 2 + dim] as f64,
                cpu.scales[i][dim],
            );
        }
        for dim in 0..4 {
            ck.check(&format!("quat[{i}][{dim}]"), g_quat[i * 4 + dim] as f64, cpu.quat[i][dim]);
        }
        ck.check(&format!("opacity[{i}]"), g_op[i] as f64, cpu.opacity[i]);
        for k in 0..n_coeffs {
            for ch in 0..3 {
                ck.check(
                    &format!("sh[{i}][{k}][{ch}]"),
                    g_sh[i * 48 + k * 3 + ch] as f64,
                    cpu.sh[i][k][ch],
                );
            }
        }
    }
    for dim in 0..3 {
        ck.check(
            &format!("cam_center[{dim}]"),
            g_cam[9 + dim] as f64,
            cpu.cam_center[dim],
        );
    }
    ck.check("focal", g_cam[12] as f64, cpu.focal);
    // Camera quaternion: chain the GPU's dl/dR_cam matrix on the host with the
    // same f64 math the oracle uses.
    let dl_dr = glam::DMat3::from_cols(
        glam::DVec3::new(g_cam[0] as f64, g_cam[1] as f64, g_cam[2] as f64),
        glam::DVec3::new(g_cam[3] as f64, g_cam[4] as f64, g_cam[5] as f64),
        glam::DVec3::new(g_cam[6] as f64, g_cam[7] as f64, g_cam[8] as f64),
    );
    let gpu_cam_quat = gs_cpu_ref::math::quat_grad(scene.camera.quat, &dl_dr);
    for dim in 0..4 {
        ck.check(
            &format!("cam_quat[{dim}]"),
            gpu_cam_quat[dim],
            cpu.cam_quat[dim],
        );
    }
    eprintln!(
        "grad parity seed={seed} n={n} deg={deg}: worst rel {:.2e} at {}",
        ck.worst, ck.worst_label
    );
}

#[test]
fn backward_matches_cpu_oracle() {
    let Some(ctx) = context() else { return };
    run_grad_parity(&ctx, 111, 15, 0, 48);
    run_grad_parity(&ctx, 222, 15, 1, 48);
    run_grad_parity(&ctx, 333, 10, 3, 40);
}
