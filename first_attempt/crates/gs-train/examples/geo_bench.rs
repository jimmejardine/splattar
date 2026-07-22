//! Isolates the geometry-loss slowdown (task: 0.8 → <0.1 it/s at geo_start on
//! Mip-NeRF360 room). Times training iterations at room-like resolution and
//! surfel count under feature combinations:
//!   baseline / +distortion / +normal / +mcmc-noise / all
//! Run: cargo run -p gs-train --release --example geo_bench

use glam::Vec3;
use gs_train::{InitialSurfels, TrainConfig, TrainView, Trainer};

const W: u32 = 780;
const H: u32 = 520;
const N: usize = 300_000;
const ITERS: u32 = 60;
const WARMUP: u32 = 10;

fn xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 11) as f64 / (1u64 << 53) as f64) as f32
}

fn main() {
    env_logger::init();
    let ctx = pollster::block_on(gs_wgpu::GpuContext::new(wgpu::Backends::VULKAN))
        .expect("gpu context");

    // Room-ish synthetic: surfels spread through a box, one fixed camera set.
    let mut rng = 0xbe9c4u64 | 1;
    let mut init = InitialSurfels {
        positions: Vec::new(),
        scales: Vec::new(),
        quats: Vec::new(),
        opacities: Vec::new(),
        sh: Vec::new(),
        sh_coeffs: 1,
    };
    for _ in 0..N {
        let u = |rng: &mut u64, s: f32| (xorshift(rng) * 2.0 - 1.0) * s;
        init.positions.push(Vec3::new(
            u(&mut rng, 4.0),
            u(&mut rng, 2.0),
            u(&mut rng, 4.0),
        ));
        init.scales.push([
            0.01 + xorshift(&mut rng) * 0.04,
            0.01 + xorshift(&mut rng) * 0.04,
        ]);
        init.quats.push([
            u(&mut rng, 1.0),
            u(&mut rng, 1.0),
            u(&mut rng, 1.0),
            u(&mut rng, 1.0) + 1.5,
        ]);
        init.opacities.push(0.3 + xorshift(&mut rng) * 0.6);
        init.sh
            .extend([u(&mut rng, 0.8), u(&mut rng, 0.8), u(&mut rng, 0.8)]);
    }

    // A handful of cameras orbiting the box interior; targets are just a
    // rendered snapshot of the init (content doesn't matter for timing).
    let mut views = Vec::new();
    for k in 0..8 {
        let a = k as f32 / 8.0 * std::f32::consts::TAU;
        let eye = Vec3::new(a.sin() * 2.0, 0.3, a.cos() * 2.0);
        let cam = gs_core::Camera::look_at(eye, Vec3::ZERO, Vec3::Y);
        views.push(TrainView {
            camera: gs_kernels::RasterCamera {
                center: cam.position,
                quat: cam.rotation,
                focal: 790.0,
                sh_degree: 0,
            },
            target: vec![[0.3f32, 0.3, 0.3, 1.0]; (W * H) as usize],
        });
    }

    let combos: &[(&str, f32, f32, f32)] = &[
        // name, lambda_dist, lambda_normal, mcmc_noise
        ("baseline (color only)", 0.0, 0.0, 0.0),
        ("+distortion", 0.01, 0.0, 0.0),
        ("+normal", 0.0, 0.05, 0.0),
        ("+mcmc noise", 0.0, 0.0, 20.0),
        ("all (room config)", 0.01, 0.05, 20.0),
    ];
    for (name, ld, ln, noise) in combos {
        let config = TrainConfig {
            iters: ITERS,
            log_every: 0,
            lambda_dist: *ld,
            lambda_normal: *ln,
            mcmc_noise: *noise,
            mcmc_every: 0,
            geo_start: 0,
            entries_per_surfel: 48,
            ..Default::default()
        };
        let init_clone = InitialSurfels {
            positions: init.positions.clone(),
            scales: init.scales.clone(),
            quats: init.quats.clone(),
            opacities: init.opacities.clone(),
            sh: init.sh.clone(),
            sh_coeffs: init.sh_coeffs,
        };
        let mut trainer = Trainer::new(&ctx, W, H, views.clone(), init_clone, config);
        for i in 0..WARMUP {
            trainer.step(&ctx, i);
        }
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let t0 = std::time::Instant::now();
        for i in WARMUP..ITERS {
            trainer.step(&ctx, i);
        }
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let per = t0.elapsed().as_secs_f64() / (ITERS - WARMUP) as f64;
        println!("{name:>24}: {:.1} ms/iter ({:.2} it/s)", per * 1e3, 1.0 / per);
    }
}
