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
    // --- M4: quality & geometry ---
    /// Depth-distortion loss weight (0 disables).
    pub lambda_dist: f32,
    /// Normal-consistency loss weight (0 disables).
    pub lambda_normal: f32,
    /// L1 regularizer weights on activated opacity / scales (MCMC priors).
    pub reg_opacity: f32,
    pub reg_scale: f32,
    /// Iteration at which geometry losses phase in (enabling them from
    /// iteration 0 hurts convergence — CLAUDE.md gotcha).
    pub geo_start: u32,
    /// Raise the active SH degree by one every this many iterations (0 = off).
    pub sh_promote_every: u32,
    /// Run MCMC dead-surfel relocation every this many iterations (0 = off).
    pub mcmc_every: u32,
    /// Activated-opacity threshold below which a surfel counts as dead.
    pub mcmc_dead: f32,
    /// Position exploration noise multiplier (× current position LR; 0 = off).
    pub mcmc_noise: f32,
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
            lambda_dist: 0.0,
            lambda_normal: 0.0,
            reg_opacity: 0.0,
            reg_scale: 0.0,
            geo_start: 500,
            sh_promote_every: 0,
            mcmc_every: 0,
            mcmc_dead: 0.005,
            mcmc_noise: 0.0,
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
    pub normal_loss: crate::NormalLoss,
    pub views: Vec<TrainView>,
    pub num_surfels: u32,
    pub config: TrainConfig,
    extent: f32,
    rng: u64,
    noise_pipeline: wgpu::ComputePipeline,
    noise_bg: wgpu::BindGroup,
    noise_uniform: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseUniform {
    n: u32,
    seed: u32,
    sigma: f32,
    opacity_k: f32,
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

        // Regularizers (mean-based: λ/count added to dL/d(activated)).
        if config.reg_opacity > 0.0 {
            optim.set_reg("opacity", config.reg_opacity / n as f32);
        }
        if config.reg_scale > 0.0 {
            optim.set_reg("scales", config.reg_scale / (2 * n) as f32);
        }

        let normal_loss = crate::NormalLoss::new(
            ctx,
            width,
            height,
            &raster.out_color,
            &raster.out_aux,
            &raster.out_normal,
            &raster.dl_dnormal,
            &raster.dl_dcolor,
        );

        // MCMC exploration-noise kernel over raw positions.
        let noise_module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcmc-noise"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcmc.wgsl").into()),
        });
        let noise_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mcmc-noise"),
                layout: None,
                module: &noise_module,
                entry_point: Some("add_noise"),
                compilation_options: Default::default(),
                cache: None,
            });
        let noise_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcmc-noise-uniform"),
            size: std::mem::size_of::<NoiseUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let noise_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcmc-noise-bg"),
            layout: &noise_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: noise_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: optim.class("pos").raw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: raster.scales.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: raster.opacities.as_entire_binding() },
            ],
        });

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
            normal_loss,
            views,
            num_surfels: n,
            config,
            extent,
            rng: seed | 1,
            noise_pipeline,
            noise_bg,
            noise_uniform,
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

        // Geometry losses phase in after warm-up. The distortion loss sums
        // over all rays — normalize by pixel count so config weights are
        // comparable to the mean-normalized color loss.
        let geo_on = iter >= self.config.geo_start;
        let n_px = (self.raster.width * self.raster.height) as f32;
        self.raster.lambda_dist = if geo_on {
            self.config.lambda_dist / n_px
        } else {
            0.0
        };

        let view_idx = self.next_view();
        ctx.queue.write_buffer(
            &self.loss.target,
            0,
            bytemuck::cast_slice(&self.views[view_idx].target),
        );
        let mut camera = self.views[view_idx].camera.clone();
        // Progressive SH unlock.
        if let Some(deg) = iter.checked_div(self.config.sh_promote_every) {
            camera.sh_degree = deg.min(camera.sh_degree);
        }
        self.normal_loss.set_lambda(
            ctx,
            if geo_on { self.config.lambda_normal } else { 0.0 },
            camera.focal,
        );

        // Split into two submissions so a heavy early-training iteration stays
        // under the Windows GPU watchdog (TDR ~2 s per submission).
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.raster.forward(ctx, &mut encoder, &camera, self.num_surfels);
        self.loss.encode(&mut encoder);
        self.normal_loss.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.raster.backward(&mut encoder, self.num_surfels);
        self.optim.encode_step(ctx, &mut encoder);

        // MCMC exploration noise on raw positions, scaled by the position LR.
        if self.config.mcmc_noise > 0.0 && geo_on {
            let u = NoiseUniform {
                n: self.num_surfels,
                seed: iter.wrapping_mul(0x9e3779b9),
                sigma: self.config.mcmc_noise * lr,
                opacity_k: 5.0,
            };
            ctx.queue
                .write_buffer(&self.noise_uniform, 0, bytemuck::bytes_of(&u));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcmc-noise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.noise_pipeline);
            pass.set_bind_group(0, &self.noise_bg, &[]);
            pass.dispatch_workgroups(self.num_surfels.div_ceil(256), 1, 1);
        }
        ctx.queue.submit([encoder.finish()]);

        // Periodic dead-surfel relocation.
        if self.config.mcmc_every > 0
            && iter > 0
            && iter.is_multiple_of(self.config.mcmc_every)
            && iter + 500 < self.config.iters
        {
            self.mcmc_relocate(ctx);
        }
        view_idx
    }

    /// MCMC relocation: dead surfels (activated opacity below threshold) move
    /// onto opacity-sampled alive surfels; both get the split opacity
    /// α' = 1 − √(1−α) and slightly reduced scales; Adam moments reset.
    fn mcmc_relocate(&mut self, ctx: &GpuContext) {
        let read = |b: &wgpu::Buffer| -> Vec<f32> {
            bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, b)).to_vec()
        };
        let opacity_act = read(&self.raster.opacities);
        let n = self.num_surfels as usize;
        let dead: Vec<u32> = (0..n)
            .filter(|&i| opacity_act[i] < self.config.mcmc_dead)
            .map(|i| i as u32)
            .collect();
        if dead.is_empty() {
            return;
        }

        // Opacity-proportional alive sampling via a cumulative table.
        let mut cum = Vec::with_capacity(n);
        let mut total = 0f64;
        for (i, &o) in opacity_act.iter().enumerate() {
            if o >= self.config.mcmc_dead {
                total += o as f64;
            }
            let _ = i;
            cum.push(total);
        }
        if total <= 0.0 {
            return;
        }

        let mut pos = read(&self.optim.class("pos").raw);
        let mut scales = read(&self.optim.class("scales").raw);
        let mut quats = read(&self.optim.class("quat").raw);
        let mut opac = read(&self.optim.class("opacity").raw);
        let mut sh = read(&self.optim.class("sh").raw);

        let mut touched = dead.clone();
        for &d in &dead {
            // Sample an alive target ∝ opacity.
            let mut x = self.rng;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.rng = x;
            let r = (x >> 11) as f64 / (1u64 << 53) as f64 * total;
            let a = cum.partition_point(|&c| c < r).min(n - 1);
            let (d, a) = (d as usize, a);
            if a == d {
                continue;
            }
            touched.push(a as u32);

            // Copy target params; split opacity; shrink scales ~15%.
            let alpha_a = opacity_act[a].clamp(1e-4, 0.999);
            let alpha_new = (1.0 - (1.0 - alpha_a).sqrt()).clamp(1e-4, 0.999);
            let logit = (alpha_new / (1.0 - alpha_new)).ln();
            opac[d] = logit;
            opac[a] = logit;
            for k in 0..4 {
                pos[d * 4 + k] = pos[a * 4 + k];
                quats[d * 4 + k] = quats[a * 4 + k];
            }
            for k in 0..2 {
                let s = scales[a * 2 + k] - 0.16; // log-space ×0.85
                scales[a * 2 + k] = s;
                scales[d * 2 + k] = s;
            }
            let src: Vec<f32> = sh[a * 48..(a + 1) * 48].to_vec();
            sh[d * 48..(d + 1) * 48].copy_from_slice(&src);
        }

        ctx.queue
            .write_buffer(&self.optim.class("pos").raw, 0, bytemuck::cast_slice(&pos));
        ctx.queue
            .write_buffer(&self.optim.class("scales").raw, 0, bytemuck::cast_slice(&scales));
        ctx.queue
            .write_buffer(&self.optim.class("quat").raw, 0, bytemuck::cast_slice(&quats));
        ctx.queue
            .write_buffer(&self.optim.class("opacity").raw, 0, bytemuck::cast_slice(&opac));
        ctx.queue
            .write_buffer(&self.optim.class("sh").raw, 0, bytemuck::cast_slice(&sh));

        self.optim.zero_moments(ctx, "pos", &touched, 4);
        self.optim.zero_moments(ctx, "scales", &touched, 2);
        self.optim.zero_moments(ctx, "quat", &touched, 4);
        self.optim.zero_moments(ctx, "opacity", &touched, 1);
        self.optim.zero_moments(ctx, "sh", &touched, 48);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.optim.encode_activate(ctx, &mut encoder);
        ctx.queue.submit([encoder.finish()]);
        log::debug!("mcmc: relocated {} dead surfels", dead.len());
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

