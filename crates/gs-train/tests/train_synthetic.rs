//! M3 end-to-end gate on a synthetic posed sequence: render ground-truth
//! views of a random surfel scene, train a different random init from scratch,
//! and demand held-out PSNR. Isolates the trainer from pose errors exactly as
//! PLAN.md prescribes (poses here are known perfectly).

use glam::Vec3;
use gs_kernels::{RasterCamera, Rasterizer, SceneInput};
use gs_train::{InitialSurfels, TrainConfig, TrainView, Trainer};
use gs_wgpu::GpuContext;

const SIZE: u32 = 128;
const N_GT: usize = 300;
const N_TRAIN: usize = 300;

fn context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(wgpu::Backends::all())) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPING GPU training test: {e}");
            None
        }
    }
}

fn xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 11) as f64 / (1u64 << 53) as f64) as f32
}

fn uni(rng: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * xorshift(rng)
}

fn random_surfels(seed: u64, n: usize, radius: f32, colorful: bool) -> InitialSurfels {
    let mut rng = seed;
    let mut init = InitialSurfels {
        positions: Vec::new(),
        scales: Vec::new(),
        quats: Vec::new(),
        opacities: Vec::new(),
        sh: Vec::new(),
        sh_coeffs: 1,
    };
    for _ in 0..n {
        init.positions.push(Vec3::new(
            uni(&mut rng, -radius, radius),
            uni(&mut rng, -radius, radius),
            uni(&mut rng, -radius, radius),
        ));
        init.scales.push([uni(&mut rng, 0.06, 0.25), uni(&mut rng, 0.06, 0.25)]);
        init.quats.push([
            uni(&mut rng, -1.0, 1.0),
            uni(&mut rng, -1.0, 1.0),
            uni(&mut rng, -1.0, 1.0),
            uni(&mut rng, -1.0, 1.0) + 1.5,
        ]);
        init.opacities.push(if colorful { uni(&mut rng, 0.4, 0.95) } else { 0.5 });
        if colorful {
            init.sh.extend([
                uni(&mut rng, -0.9, 0.9),
                uni(&mut rng, -0.9, 0.9),
                uni(&mut rng, -0.9, 0.9),
            ]);
        } else {
            init.sh.extend([0.0, 0.0, 0.0]);
        }
    }
    init
}

fn orbit_camera(angle: f32, height: f32, dist: f32) -> RasterCamera {
    let eye = Vec3::new(angle.sin() * dist, height, angle.cos() * dist);
    let cam = gs_core::Camera::look_at(eye, Vec3::ZERO, Vec3::Y);
    RasterCamera {
        center: cam.position,
        quat: cam.rotation,
        focal: SIZE as f32 * 0.9,
        sh_degree: 0,
    }
}

// ~45 s on the dev GPU — gated behind --ignored so routine development
// doesn't pay for it. Run explicitly (also part of milestone acceptance):
//   cargo test -p gs-train --release --test train_synthetic -- --ignored --nocapture
#[test]
#[ignore = "slow end-to-end training (~45 s GPU); run with --ignored"]
fn trains_synthetic_scene_from_scratch() {
    let Some(ctx) = context() else { return };

    // Ground truth scene + rendered targets on an orbit.
    let gt = random_surfels(0xfeedbeef, N_GT, 1.2, true);
    let gt_raster = Rasterizer::new(&ctx, N_GT as u32, SIZE, SIZE, (N_GT * 64) as u32);
    gt_raster.upload_scene(
        &ctx,
        &SceneInput {
            positions: &gt.positions,
            scales: &gt.scales,
            quats: &gt.quats,
            opacities: &gt.opacities,
            sh: &gt.sh,
            sh_coeffs: 1,
        },
    );
    let render_gt = |camera: &RasterCamera| -> Vec<[f32; 4]> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        gt_raster.forward(&ctx, &mut encoder, camera, N_GT as u32);
        ctx.queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            &gt_raster.out_color,
        ))
        .to_vec()
    };

    let mut train_views = Vec::new();
    let mut eval_views = Vec::new();
    for i in 0..35 {
        let angle = i as f32 / 35.0 * std::f32::consts::TAU;
        let camera = orbit_camera(angle, 1.0 + (i % 3) as f32 * 0.5, 4.0);
        let view = TrainView {
            target: render_gt(&camera),
            camera,
        };
        if i % 7 == 3 {
            eval_views.push(view);
        } else {
            train_views.push(view);
        }
    }

    // Train a different random init from scratch.
    let init = random_surfels(0x12345, N_TRAIN, 1.2, false);
    let config = TrainConfig {
        iters: 6000,
        log_every: 1000,
        ..Default::default()
    };
    let mut trainer = Trainer::new(&ctx, SIZE, SIZE, train_views, init, config);
    let start_psnr = trainer.eval_psnr(&ctx, &eval_views);
    trainer.train(&ctx);
    let psnr = trainer.eval_psnr(&ctx, &eval_views);
    eprintln!("held-out PSNR: {start_psnr:.2} dB → {psnr:.2} dB (gate: > 27 dB)");
    assert!(
        psnr > 27.0,
        "synthetic training gate failed: {psnr:.2} dB (from {start_psnr:.2})"
    );
}

/// M4 gate: with distortion + normal-consistency losses, regularizers, MCMC
/// relocation/noise, and progressive SH all enabled, training must still
/// converge to at least the M3-level bar at the same fixed budget.
#[test]
#[ignore = "slow end-to-end training (~45 s GPU); run with --ignored"]
fn trains_with_m4_features_enabled() {
    let Some(ctx) = context() else { return };

    let gt = random_surfels(0xfeedbeef, N_GT, 1.2, true);
    let gt_raster = Rasterizer::new(&ctx, N_GT as u32, SIZE, SIZE, (N_GT * 64) as u32);
    gt_raster.upload_scene(
        &ctx,
        &SceneInput {
            positions: &gt.positions,
            scales: &gt.scales,
            quats: &gt.quats,
            opacities: &gt.opacities,
            sh: &gt.sh,
            sh_coeffs: 1,
        },
    );
    let render_gt = |camera: &RasterCamera| -> Vec<[f32; 4]> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        gt_raster.forward(&ctx, &mut encoder, camera, N_GT as u32);
        ctx.queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            &gt_raster.out_color,
        ))
        .to_vec()
    };
    let mut train_views = Vec::new();
    let mut eval_views = Vec::new();
    for i in 0..35 {
        let angle = i as f32 / 35.0 * std::f32::consts::TAU;
        let camera = orbit_camera(angle, 1.0 + (i % 3) as f32 * 0.5, 4.0);
        let view = TrainView {
            target: render_gt(&camera),
            camera,
        };
        if i % 7 == 3 {
            eval_views.push(view);
        } else {
            train_views.push(view);
        }
    }

    let init = random_surfels(0x777, N_TRAIN, 1.2, false);
    let config = TrainConfig {
        iters: 6000,
        log_every: 2000,
        lambda_dist: 0.005,
        lambda_normal: 0.02,
        reg_opacity: 0.005,
        reg_scale: 0.005,
        geo_start: 800,
        mcmc_every: 400,
        mcmc_noise: 20.0,
        ..Default::default()
    };
    let mut trainer = Trainer::new(&ctx, SIZE, SIZE, train_views, init, config);
    trainer.train(&ctx);
    let psnr = trainer.eval_psnr(&ctx, &eval_views);
    eprintln!("M4-featured held-out PSNR: {psnr:.2} dB (gate: > 26 dB)");
    assert!(psnr > 26.0, "M4-featured training regressed: {psnr:.2} dB");
}

/// M7 gate: pose refinement recovers deliberately perturbed training poses.
/// Same synthetic scene, but every training camera gets a small rotation +
/// center error (like monocular VO noise). Without refinement this caps PSNR
/// hard; with refinement the trainer should climb back near the clean bar.
#[test]
#[ignore = "slow end-to-end training (~90 s GPU); run with --ignored"]
fn pose_refinement_recovers_perturbed_poses() {
    let Some(ctx) = context() else { return };

    let gt = random_surfels(0xfeedbeef, N_GT, 1.2, true);
    let gt_raster = Rasterizer::new(&ctx, N_GT as u32, SIZE, SIZE, (N_GT * 64) as u32);
    gt_raster.upload_scene(
        &ctx,
        &SceneInput {
            positions: &gt.positions,
            scales: &gt.scales,
            quats: &gt.quats,
            opacities: &gt.opacities,
            sh: &gt.sh,
            sh_coeffs: 1,
        },
    );
    let render_gt = |camera: &RasterCamera| -> Vec<[f32; 4]> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        gt_raster.forward(&ctx, &mut encoder, camera, N_GT as u32);
        ctx.queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            &gt_raster.out_color,
        ))
        .to_vec()
    };

    // Targets rendered from TRUE poses; trainer is given PERTURBED cameras
    // (eval poses stay true — they measure whether the model is right).
    let mut rng = 0xabcdef123u64;
    let mut train_views = Vec::new();
    let mut eval_views = Vec::new();
    for i in 0..35 {
        let angle = i as f32 / 35.0 * std::f32::consts::TAU;
        let camera = orbit_camera(angle, 1.0 + (i % 3) as f32 * 0.5, 4.0);
        let target = render_gt(&camera);
        if i % 7 == 3 {
            eval_views.push(TrainView { target, camera });
        } else {
            let mut bad = camera.clone();
            let axis = glam::Vec3::new(
                uni(&mut rng, -1.0, 1.0),
                uni(&mut rng, -1.0, 1.0),
                uni(&mut rng, -1.0, 1.0),
            )
            .normalize();
            bad.quat = (bad.quat * glam::Quat::from_axis_angle(axis, 0.035)).normalize();
            bad.center += glam::Vec3::new(
                uni(&mut rng, -0.08, 0.08),
                uni(&mut rng, -0.08, 0.08),
                uni(&mut rng, -0.08, 0.08),
            );
            train_views.push(TrainView { target, camera: bad });
        }
    }

    let run = |pose_lr: f32| -> f64 {
        let init = random_surfels(0x12345, N_TRAIN, 1.2, false);
        let config = TrainConfig {
            iters: 6000,
            log_every: 6000,
            pose_refine_lr: pose_lr,
            pose_refine_start: 500,
            ..Default::default()
        };
        let mut trainer = Trainer::new(
            &ctx,
            SIZE,
            SIZE,
            train_views.clone(),
            init,
            config,
        );
        trainer.train(&ctx);
        trainer.eval_psnr(&ctx, &eval_views)
    };

    let without = run(0.0);
    let with = run(2e-3);
    eprintln!("perturbed poses: PSNR {without:.2} dB frozen vs {with:.2} dB refined");
    assert!(
        with > without + 1.5,
        "pose refinement gained too little: {without:.2} -> {with:.2} dB"
    );
    assert!(with > 25.0, "refined PSNR too low: {with:.2} dB");
}
