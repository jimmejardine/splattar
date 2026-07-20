//! The M3 training loop: posed views + initial surfels → optimized scene.
//! Random-view batches, exponential position-LR decay, fused L1+D-SSIM loss,
//! Adam-in-WGSL. One command submission per iteration.

use glam::Vec3;
use gs_kernels::{RasterCamera, Rasterizer};
use gs_wgpu::GpuContext;

use crate::optim::{Activation, Optimizer};
use crate::ssim::SsimLoss;

#[derive(Clone)]
pub struct TrainView {
    pub camera: RasterCamera,
    /// rgba-f32 target (w unused), width×height entries, row-major.
    pub target: Vec<[f32; 4]>,
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub iters: u32,
    pub lambda: f32,
    /// Position LR as a fraction of scene extent (3DGS-style), decayed
    /// exponentially to `pos_lr_final_factor`× by the last iteration.
    pub lr_pos: f32,
    pub pos_lr_final_factor: f32,
    pub lr_scales: f32,
    pub lr_quat: f32,
    pub lr_opacity: f32,
    pub lr_sh: f32,
    pub entries_per_surfel: u32,
    pub log_every: u32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            iters: 7000,
            lambda: 0.2,
            lr_pos: 1.6e-4,
            pos_lr_final_factor: 0.01,
            lr_scales: 5e-3,
            lr_quat: 1e-3,
            lr_opacity: 5e-2,
            lr_sh: 2.5e-3,
            entries_per_surfel: 64,
            log_every: 500,
            seed: 0x5eed,
        }
    }
}

/// Initial surfels in ACTIVATED space (world scales, 0..1 opacity).
pub struct InitialSurfels {
    pub positions: Vec<Vec3>,
    pub scales: Vec<[f32; 2]>,
    pub quats: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    /// Coefficient-major rgb-interleaved, `sh_coeffs*3` per surfel.
    pub sh: Vec<f32>,
    pub sh_coeffs: usize,
}

pub struct Trainer {
    pub raster: Rasterizer,
    pub optim: Optimizer,
    pub loss: SsimLoss,
    pub views: Vec<TrainView>,
    pub num_surfels: u32,
    pub config: TrainConfig,
    extent: f32,
    rng: u64,
}

impl Trainer {
    pub fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        views: Vec<TrainView>,
        init: InitialSurfels,
        config: TrainConfig,
    ) -> Self {
        let n = init.positions.len() as u32;
        assert!(n > 0 && !views.is_empty());
        let raster = Rasterizer::new(
            ctx,
            n,
            width,
            height,
            (n * config.entries_per_surfel).max(1024),
        );
        let loss = SsimLoss::new(ctx, width, height, config.lambda, &raster.out_color, &raster.dl_dcolor);

        // Scene extent for position LR scaling.
        let (mut lo, mut hi) = (init.positions[0], init.positions[0]);
        for p in &init.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let extent = (hi - lo).length().max(1e-3) * 0.5;

        let mut optim = Optimizer::new(ctx);
        optim.add_class(ctx, "pos", n * 4, Activation::Identity, config.lr_pos * extent, &raster.grad_pos, &raster.positions);
        optim.add_class(ctx, "scales", n * 2, Activation::Exp, config.lr_scales, &raster.grad_scales, &raster.scales);
        optim.add_class(ctx, "quat", n * 4, Activation::Identity, config.lr_quat, &raster.grad_quat, &raster.quats);
        optim.add_class(ctx, "opacity", n, Activation::Sigmoid, config.lr_opacity, &raster.grad_opacity, &raster.opacities);
        optim.add_class(ctx, "sh", n * 48, Activation::Identity, config.lr_sh, &raster.grad_sh, &raster.sh);

        // Upload raw initial parameters.
        let nn = n as usize;
        let mut pos_raw = Vec::with_capacity(nn * 4);
        for p in &init.positions {
            pos_raw.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
        }
        let scales_raw: Vec<f32> = init
            .scales
            .iter()
            .flat_map(|s| [s[0].max(1e-6).ln(), s[1].max(1e-6).ln()])
            .collect();
        let quat_raw: Vec<f32> = init.quats.iter().flatten().copied().collect();
        let op_raw: Vec<f32> = init
            .opacities
            .iter()
            .map(|&o| {
                let o = o.clamp(1e-4, 1.0 - 1e-4);
                (o / (1.0 - o)).ln()
            })
            .collect();
        let mut sh_raw = vec![0f32; nn * 48];
        let per = init.sh_coeffs * 3;
        for i in 0..nn {
            sh_raw[i * 48..i * 48 + per].copy_from_slice(&init.sh[i * per..(i + 1) * per]);
        }
        ctx.queue.write_buffer(&optim.class("pos").raw, 0, bytemuck::cast_slice(&pos_raw));
        ctx.queue.write_buffer(&optim.class("scales").raw, 0, bytemuck::cast_slice(&scales_raw));
        ctx.queue.write_buffer(&optim.class("quat").raw, 0, bytemuck::cast_slice(&quat_raw));
        ctx.queue.write_buffer(&optim.class("opacity").raw, 0, bytemuck::cast_slice(&op_raw));
        ctx.queue.write_buffer(&optim.class("sh").raw, 0, bytemuck::cast_slice(&sh_raw));

        // Materialize activated buffers once before the first forward.
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        optim.encode_activate(ctx, &mut encoder);
        ctx.queue.submit([encoder.finish()]);

        let seed = config.seed;
        Self {
            raster,
            optim,
            loss,
            views,
            num_surfels: n,
            config,
            extent,
            rng: seed | 1,
        }
    }

    fn next_view(&mut self) -> usize {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        (x % self.views.len() as u64) as usize
    }

    /// One training iteration on a random view. Returns the view used.
    pub fn step(&mut self, ctx: &GpuContext, iter: u32) -> usize {
        // Exponential position LR decay.
        let t = iter as f32 / self.config.iters.max(1) as f32;
        let lr = self.config.lr_pos * self.extent * self.config.pos_lr_final_factor.powf(t);
        self.optim.set_lr("pos", lr);

        let view_idx = self.next_view();
        ctx.queue.write_buffer(
            &self.loss.target,
            0,
            bytemuck::cast_slice(&self.views[view_idx].target),
        );
        let camera = self.views[view_idx].camera.clone();
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.raster.forward(ctx, &mut encoder, &camera, self.num_surfels);
        self.loss.encode(&mut encoder);
        self.raster.backward(&mut encoder, self.num_surfels);
        self.optim.encode_step(ctx, &mut encoder);
        ctx.queue.submit([encoder.finish()]);
        view_idx
    }

    pub fn train(&mut self, ctx: &GpuContext) {
        let start = std::time::Instant::now();
        for iter in 0..self.config.iters {
            self.step(ctx, iter);
            if self.config.log_every > 0
                && (iter % self.config.log_every == 0 || iter + 1 == self.config.iters)
            {
                let (l1, dssim) = self.loss.read_losses(ctx);
                log::info!(
                    "iter {iter:>6}: L1 {l1:.5}  D-SSIM {dssim:.5}  ({:.1} it/s)",
                    (iter as f64 + 1.0) / start.elapsed().as_secs_f64()
                );
            }
        }
    }

    /// Render a view and return its rgb image (readback).
    pub fn render_view(&self, ctx: &GpuContext, camera: &RasterCamera) -> Vec<[f32; 4]> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.raster.forward(ctx, &mut encoder, camera, self.num_surfels);
        ctx.queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            &self.raster.out_color,
        ))
        .to_vec()
    }

    /// Mean PSNR over a set of held-out views.
    pub fn eval_psnr(&self, ctx: &GpuContext, views: &[TrainView]) -> f64 {
        let mut total = 0.0;
        for view in views {
            let out = self.render_view(ctx, &view.camera);
            let mut mse = 0.0f64;
            for (o, t) in out.iter().zip(&view.target) {
                for ch in 0..3 {
                    let e = (o[ch] - t[ch]) as f64;
                    mse += e * e;
                }
            }
            mse /= (out.len() * 3) as f64;
            total += -10.0 * mse.max(1e-12).log10();
        }
        total / views.len() as f64
    }

    /// Read back activated parameters (for export).
    pub fn read_scene(&self, ctx: &GpuContext) -> ExportScene {
        let read = |b: &wgpu::Buffer| -> Vec<f32> {
            bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, b)).to_vec()
        };
        ExportScene {
            positions: read(&self.raster.positions),
            scales: read(&self.raster.scales),
            quats: read(&self.raster.quats),
            opacities: read(&self.raster.opacities),
            sh: read(&self.raster.sh),
            num: self.num_surfels as usize,
        }
    }
}

/// Activated scene arrays as read back from the GPU (positions vec4-strided,
/// sh padded to 48).
pub struct ExportScene {
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    pub quats: Vec<f32>,
    pub opacities: Vec<f32>,
    pub sh: Vec<f32>,
    pub num: usize,
}

