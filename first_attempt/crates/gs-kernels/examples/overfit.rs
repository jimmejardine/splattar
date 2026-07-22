//! M2 acceptance demo: gradient-descend 50 surfels to match a 128×128 target
//! image rendered from a different random scene. Host-side Adam over GPU
//! gradients (Adam-in-WGSL arrives in M3). Gate: PSNR > 35 dB.
//!
//! `cargo run -p gs-kernels --release --example overfit`

use glam::{Quat, Vec3};
use gs_kernels::{RasterCamera, Rasterizer, SceneInput};
use gs_wgpu::GpuContext;

const N: usize = 50;
const SIZE: u32 = 128;
const ITERS: usize = 3000;

fn xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 11) as f64 / (1u64 << 53) as f64) as f32
}

fn uniform(rng: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * xorshift(rng)
}

struct Params {
    positions: Vec<Vec3>,
    scales: Vec<[f32; 2]>,
    quats: Vec<[f32; 4]>,
    opacities: Vec<f32>,
    sh: Vec<f32>, // deg 0: 3 per surfel
}

impl Params {
    fn random(seed: u64, spread: f32) -> Self {
        let mut rng = seed;
        let mut p = Params {
            positions: Vec::new(),
            scales: Vec::new(),
            quats: Vec::new(),
            opacities: Vec::new(),
            sh: Vec::new(),
        };
        for _ in 0..N {
            let z = -uniform(&mut rng, 2.5, 5.0);
            let ext = -z * 0.55 * spread;
            p.positions.push(Vec3::new(
                uniform(&mut rng, -ext, ext),
                uniform(&mut rng, -ext, ext),
                z,
            ));
            p.scales.push([uniform(&mut rng, 0.15, 0.5), uniform(&mut rng, 0.15, 0.5)]);
            p.quats.push([
                uniform(&mut rng, -1.0, 1.0),
                uniform(&mut rng, -1.0, 1.0),
                uniform(&mut rng, -1.0, 1.0),
                uniform(&mut rng, -1.0, 1.0) + 1.5,
            ]);
            p.opacities.push(uniform(&mut rng, 0.4, 0.9));
            p.sh.extend([
                uniform(&mut rng, -0.9, 0.9),
                uniform(&mut rng, -0.9, 0.9),
                uniform(&mut rng, -0.9, 0.9),
            ]);
        }
        p
    }

    fn as_input(&self) -> SceneInput<'_> {
        SceneInput {
            positions: &self.positions,
            scales: &self.scales,
            quats: &self.quats,
            opacities: &self.opacities,
            sh: &self.sh,
            sh_coeffs: 1,
        }
    }
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.t += 1;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grads[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grads[i] * grads[i];
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= lr * mhat / (vhat.sqrt() + 1e-8);
        }
    }
}

fn main() {
    let ctx = pollster::block_on(GpuContext::new(wgpu::Backends::all())).expect("gpu");
    let camera = RasterCamera {
        center: Vec3::ZERO,
        quat: Quat::IDENTITY,
        focal: SIZE as f32 * 0.9,
        sh_degree: 0,
    };
    let raster = Rasterizer::new(&ctx, N as u32, SIZE, SIZE, (N * 128) as u32);
    let n_px = (SIZE * SIZE) as usize;

    let render = |params: &Params| -> Vec<[f32; 4]> {
        raster.upload_scene(&ctx, &params.as_input());
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        raster.forward(&ctx, &mut encoder, &camera, N as u32);
        ctx.queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, &raster.out_color))
            .to_vec()
    };

    // Ground truth from one random scene; optimize a different random init.
    let target = render(&Params::random(0xbeef, 1.0));
    let mut params = Params::random(0x1234, 0.9);

    let mut adam_pos = Adam::new(N * 3);
    let mut adam_scale = Adam::new(N * 2);
    let mut adam_quat = Adam::new(N * 4);
    let mut adam_op = Adam::new(N);
    let mut adam_sh = Adam::new(N * 3);

    let read_f32 = |buf: &wgpu::Buffer| -> Vec<f32> {
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, buf)).to_vec()
    };

    let mut final_psnr = 0.0;
    for iter in 0..ITERS {
        let out = render(&params);

        // L2 loss and its gradient.
        let mut mse = 0.0f64;
        let dl: Vec<[f32; 4]> = out
            .iter()
            .zip(&target)
            .map(|(o, t)| {
                let mut d = [0.0f32; 4];
                for ch in 0..3 {
                    let e = o[ch] - t[ch];
                    mse += (e * e) as f64;
                    d[ch] = 2.0 * e / (n_px as f32 * 3.0);
                }
                d
            })
            .collect();
        mse /= (n_px * 3) as f64;
        let psnr = -10.0 * (mse.max(1e-12)).log10();
        final_psnr = psnr;
        if iter % 250 == 0 {
            println!("iter {iter:>5}: mse {mse:.6e}  psnr {psnr:.2} dB");
        }

        ctx.queue
            .write_buffer(&raster.dl_dcolor, 0, bytemuck::cast_slice(&dl));
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        raster.backward(&mut encoder, N as u32);
        ctx.queue.submit([encoder.finish()]);

        let g_pos = read_f32(&raster.grad_pos);
        let g_scale = read_f32(&raster.grad_scales);
        let g_quat = read_f32(&raster.grad_quat);
        let g_op = read_f32(&raster.grad_opacity);
        let g_sh = read_f32(&raster.grad_sh);

        // Flatten → Adam → clamp back.
        let mut pos_flat: Vec<f32> = params.positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let pos_grads: Vec<f32> = (0..N).flat_map(|i| [g_pos[i * 4], g_pos[i * 4 + 1], g_pos[i * 4 + 2]]).collect();
        adam_pos.step(&mut pos_flat, &pos_grads, 0.01);
        for i in 0..N {
            params.positions[i] = Vec3::new(pos_flat[i * 3], pos_flat[i * 3 + 1], pos_flat[i * 3 + 2]);
        }

        let mut scale_flat: Vec<f32> = params.scales.iter().flatten().copied().collect();
        adam_scale.step(&mut scale_flat, &g_scale, 0.005);
        for i in 0..N {
            params.scales[i] = [
                scale_flat[i * 2].clamp(0.02, 2.0),
                scale_flat[i * 2 + 1].clamp(0.02, 2.0),
            ];
        }

        let mut quat_flat: Vec<f32> = params.quats.iter().flatten().copied().collect();
        adam_quat.step(&mut quat_flat, &g_quat, 0.02);
        for i in 0..N {
            let q = Quat::from_xyzw(
                quat_flat[i * 4],
                quat_flat[i * 4 + 1],
                quat_flat[i * 4 + 2],
                quat_flat[i * 4 + 3],
            )
            .normalize();
            params.quats[i] = [q.x, q.y, q.z, q.w];
        }

        adam_op.step(&mut params.opacities, &g_op, 0.02);
        for o in &mut params.opacities {
            *o = o.clamp(0.02, 0.98);
        }

        let sh_grads: Vec<f32> = (0..N).flat_map(|i| [g_sh[i * 48], g_sh[i * 48 + 1], g_sh[i * 48 + 2]]).collect();
        adam_sh.step(&mut params.sh, &sh_grads, 0.05);
    }

    println!("final: psnr {final_psnr:.2} dB over {ITERS} iters (gate: > 35 dB)");
    assert!(final_psnr > 35.0, "M2 overfit gate failed: {final_psnr:.2} dB");
    println!("M2 overfit gate PASSED");
}
