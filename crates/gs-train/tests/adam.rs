//! Adam-in-WGSL vs a CPU reference: several steps over random params/grads
//! for each activation mode, including the raw→activated chain rule.

use gs_train::{Activation, Optimizer};
use gs_wgpu::{GpuContext, buffers};

fn context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(wgpu::Backends::all())) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPING GPU adam tests: {e}");
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
    (((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
}

fn dact(mode: Activation, x: f32) -> f32 {
    match mode {
        Activation::Identity => 1.0,
        Activation::Exp => x.exp(),
        Activation::Sigmoid => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        }
    }
}

#[test]
fn adam_matches_cpu_reference() {
    let Some(ctx) = context() else { return };
    const N: usize = 1000;
    const STEPS: usize = 5;
    const LR: f32 = 0.01;

    for mode in [Activation::Identity, Activation::Exp, Activation::Sigmoid] {
        let mut rng = 0xadau64 ^ mode as u64;
        let init: Vec<f32> = (0..N).map(|_| xorshift(&mut rng) * 0.8).collect();
        let grads_per_step: Vec<Vec<f32>> = (0..STEPS)
            .map(|_| (0..N).map(|_| xorshift(&mut rng)).collect())
            .collect();

        // CPU reference.
        let mut p = init.clone();
        let mut m = vec![0f32; N];
        let mut v = vec![0f32; N];
        for (t, grads) in grads_per_step.iter().enumerate() {
            let t = t as i32 + 1;
            for i in 0..N {
                let g = grads[i] * dact(mode, p[i]);
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                let mhat = m[i] / (1.0 - 0.9f32.powi(t));
                let vhat = v[i] / (1.0 - 0.999f32.powi(t));
                p[i] -= LR * mhat / (vhat.sqrt() + 1e-8);
            }
        }

        // GPU.
        let grads_buf = buffers::storage_empty(&ctx.device, "test-grads", N as u64 * 4);
        let act_buf = buffers::storage_empty(&ctx.device, "test-act", N as u64 * 4);
        let mut optim = Optimizer::new(&ctx);
        optim.add_class(&ctx, "test", N as u32, mode, LR, &grads_buf, &act_buf);
        ctx.queue
            .write_buffer(&optim.class("test").raw, 0, bytemuck::cast_slice(&init));
        for grads in &grads_per_step {
            ctx.queue
                .write_buffer(&grads_buf, 0, bytemuck::cast_slice(grads));
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            optim.encode_step(&ctx, &mut encoder);
            ctx.queue.submit([encoder.finish()]);
        }
        let got: Vec<f32> = bytemuck::cast_slice(&buffers::readback(
            &ctx.device,
            &ctx.queue,
            &optim.class("test").raw,
        ))
        .to_vec();
        for i in 0..N {
            assert!(
                (got[i] - p[i]).abs() < 1e-5,
                "{mode:?} param {i}: gpu {} vs cpu {}",
                got[i],
                p[i]
            );
        }
    }
}
