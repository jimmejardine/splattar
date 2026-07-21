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
    // --- M7: pose refinement (VO poses are noisy; the rasterizer's camera
    // gradients are three-way verified, so let them polish the cameras) ---
    /// Base LR for per-view pose refinement (0 = off). Rotation uses it
    /// directly; camera center uses it × scene extent.
    pub pose_refine_lr: f32,
    /// Iteration at which pose refinement starts (the model must be good
    /// enough first that pose gradients point somewhere sensible).
    pub pose_refine_start: u32,
    /// Iteration at which pose refinement stops. Long runs otherwise walk
    /// the camera gauge all the way to the end and the model overfits a
    /// moving target: train loss stays great, held-out collapses.
    pub pose_refine_end: u32,
    /// Also refine the (shared) focal length in log-space.
    pub focal_refine: bool,
    /// Iteration at which per-view affine appearance compensation starts
    /// (u32::MAX = off). Phone auto-exposure/AWB sweeps continuously; each
    /// view gets a per-channel gain+bias fitted against its render and the
    /// *target* is inverse-corrected in the loss — the scene stays canonical
    /// (never baked into SH, per CLAUDE.md).
    pub appearance_start: u32,
    /// Below this iteration, forward and backward go in separate queue
    /// submissions (headroom under the Windows GPU watchdog — TDR ~2 s per
    /// submission — for heavy unconverged early iterations); from it on, one
    /// submission per iteration. u32::MAX = always split (escape hatch).
    pub merge_submits_after: u32,
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
            pose_refine_lr: 0.0,
            pose_refine_start: 500,
            pose_refine_end: u32::MAX,
            focal_refine: false,
            appearance_start: u32::MAX,
            merge_submits_after: 1000,
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
    // Pose-refinement Adam state: per-view [rot(3), center(3)] + shared
    // log-focal. Host-side — 6 params per view is nothing.
    pose_m: Vec<[f32; 6]>,
    pose_v: Vec<[f32; 6]>,
    pose_t: Vec<u32>,
    focal_state: (f32, f32, u32),
    /// Multiplier on the initial focal (shared across views), refined in
    /// log-space when `focal_refine` is on.
    pub focal_scale: f32,
    /// Subsampled init positions for per-view depth statistics: the camera
    /// center LR must scale with the *local viewing depth*, not the scene
    /// extent (a long walkthrough's extent is ~30× the room depth).
    probe_positions: Vec<Vec3>,
    pose_depth: Vec<f32>,
    /// Per-view appearance [gain rgb, bias rgb], EMA-updated from LS fits.
    appear: Vec<[f32; 6]>,
    /// GPU-resident targets + GPU affine-fit reduction (async readback).
    appearance: crate::appearance::Appearance,
    /// Async ring for the 64-byte grad_cam readback: pose updates apply 1-2
    /// iterations late instead of stalling the queue every iteration.
    pose_ring: gs_wgpu::ReadbackRing<PendingPose>,
    /// Per-kernel GPU timing, sampled on log iterations (None when the
    /// adapter lacks TIMESTAMP_QUERY).
    timer: Option<gs_wgpu::GpuTimer>,
    /// Host wall-clock per step phase, accumulated between log lines. The
    /// host timers are what reveal CPU↔GPU stalls — GPU timestamps can't.
    phases: PhaseTimes,
}

/// Snapshot taken when a pose-gradient readback is issued — the host Adam
/// step must chain the gradient at the camera state that produced it.
struct PendingPose {
    view_idx: usize,
    /// Camera rotation at render time (grad_cam's dl/dR chains through it).
    quat: glam::Quat,
    /// Focal at render time, for the log-space focal gradient.
    focal: f32,
    /// LR decay factor at issue time.
    decay: f32,
}

/// Millisecond accumulators for the phases of [`Trainer::step`].
#[derive(Default)]
struct PhaseTimes {
    target: f64,
    fwd: f64,
    appear: f64,
    bwd: f64,
    pose: f64,
    mcmc: f64,
    n: u32,
}

impl PhaseTimes {
    fn log_line(&self) -> String {
        let n = self.n.max(1) as f64;
        format!(
            "host ms/iter: target {:.2} | fwd-submit {:.2} | appear {:.2} | bwd-submit {:.2} | pose {:.2} | mcmc {:.2}",
            self.target / n,
            self.fwd / n,
            self.appear / n,
            self.bwd / n,
            self.pose / n,
            self.mcmc / n,
        )
    }
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
        let appearance = crate::appearance::Appearance::new(
            ctx,
            width,
            height,
            &views,
            &loss.target,
            &raster.out_color,
        );

        // Scene extent for position LR scaling.
        let (mut lo, mut hi) = (init.positions[0], init.positions[0]);
        for p in &init.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let extent = (hi - lo).length().max(1e-3) * 0.5;
        // Depth probes for pose-refinement LR scaling.
        let probe_stride = (init.positions.len() / 2048).max(1);
        let probe_positions: Vec<Vec3> = init
            .positions
            .iter()
            .step_by(probe_stride)
            .copied()
            .collect();

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
        let n_views = views.len();
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
            pose_m: vec![[0.0; 6]; n_views],
            pose_v: vec![[0.0; 6]; n_views],
            pose_t: vec![0; n_views],
            focal_state: (0.0, 0.0, 0),
            focal_scale: 1.0,
            probe_positions,
            pose_depth: vec![0.0; n_views],
            appear: vec![[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]; n_views],
            appearance,
            pose_ring: gs_wgpu::ReadbackRing::new(&ctx.device, "pose-grad", 16 * 4, 4),
            timer: ctx
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(|| gs_wgpu::GpuTimer::new(ctx, 64)),
            phases: PhaseTimes::default(),
        }
    }

    /// Median depth of the probe positions in front of this camera (cached).
    fn view_depth(&mut self, view_idx: usize) -> f32 {
        if self.pose_depth[view_idx] > 0.0 {
            return self.pose_depth[view_idx];
        }
        let cam = &self.views[view_idx].camera;
        let inv = cam.quat.conjugate();
        let mut depths: Vec<f32> = self
            .probe_positions
            .iter()
            .map(|p| -(inv * (*p - cam.center)).z)
            .filter(|d| *d > 0.05)
            .collect();
        let d = if depths.is_empty() {
            self.extent * 0.1
        } else {
            let mid = depths.len() / 2;
            *depths
                .select_nth_unstable_by(mid, f32::total_cmp)
                .1
        };
        self.pose_depth[view_idx] = d.max(1e-3);
        self.pose_depth[view_idx]
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
        // GPU per-kernel timing is sampled on log iterations only (scopes
        // accumulate until read, and the read blocks).
        let profile = self.config.log_every > 0 && iter.is_multiple_of(self.config.log_every);
        let mut timer = if profile { self.timer.take() } else { None };
        let t0 = std::time::Instant::now();
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

        // Deliver async results that completed since last iteration:
        // appearance fits and pose updates, both applied 1-2 iterations late
        // by design (appearance models slow auto-exposure and already ran one
        // step lagged; pose LR decays and BARF-style refinement tolerates it).
        let fits = self.appearance.poll_fits(&ctx.device);
        for (v, fit) in fits {
            self.apply_affine_fit(v, fit);
        }
        let pending = self.pose_ring.poll_ready(&ctx.device);
        for (p, bytes) in pending {
            self.apply_pose_grads(&p, &bytes);
        }

        let view_idx = self.next_view();
        // A pose update for the view we are about to render must not land
        // after its camera is snapshotted — force-drain the rare collision
        // (a view recurs every ~n_views iterations; ring latency is 1-2).
        if self.pose_ring.any_pending(|p| p.view_idx == view_idx) {
            let pending = self.pose_ring.drain_blocking(&ctx.device);
            for (p, bytes) in pending {
                self.apply_pose_grads(&p, &bytes);
            }
        }
        let appearance_on = iter >= self.config.appearance_start;
        let mut camera = self.views[view_idx].camera.clone();
        camera.focal *= self.focal_scale;
        // Progressive SH unlock.
        if let Some(deg) = iter.checked_div(self.config.sh_promote_every) {
            camera.sh_degree = deg.min(camera.sh_degree);
        }
        self.normal_loss.set_lambda(
            ctx,
            if geo_on { self.config.lambda_normal } else { 0.0 },
            camera.focal,
        );

        let t_target = std::time::Instant::now();

        // Split into two submissions so a heavy early-training iteration stays
        // under the Windows GPU watchdog (TDR ~2 s per submission).
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // GPU-side target: inverse-correct this view's atlas image with its
        // current affine (identity before appearance starts) into the loss
        // target — no host transform, no per-iteration image upload.
        self.appearance.encode_transform(
            ctx,
            &mut encoder,
            view_idx,
            &self.appear[view_idx],
            timer.as_mut(),
        );
        self.raster
            .forward_timed(ctx, &mut encoder, &camera, self.num_surfels, timer.as_mut());
        self.loss.encode_timed(&mut encoder, timer.as_mut());
        self.normal_loss.encode_timed(&mut encoder, timer.as_mut());
        // Appearance: refit this view's affine from the fresh render vs the
        // ORIGINAL target — GPU-reduced to 48 bytes, read back async through
        // the ring (applied 1-2 iterations later by poll_fits above; the old
        // blocking fit already ran one step lagged by design).
        if appearance_on {
            self.appearance
                .encode_fit(&mut encoder, view_idx, timer.as_mut());
        }
        // Nothing reads between forward and backward anymore — once past the
        // heavy early iterations (TDR headroom), keep everything in one
        // submission.
        let split = iter < self.config.merge_submits_after;
        if split {
            if let Some(t) = timer.as_ref() {
                t.resolve(&mut encoder);
            }
            ctx.queue.submit([encoder.finish()]);
            self.appearance.map_pending();
            encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        }
        let t_fwd = std::time::Instant::now();

        let t_appear = std::time::Instant::now();

        self.raster
            .backward_timed(&mut encoder, self.num_surfels, timer.as_mut());
        self.optim.encode_step_timed(ctx, &mut encoder, timer.as_mut());

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
        // Pose refinement: queue this view's camera gradient for an async
        // readback instead of stalling on it. The host Adam step (rotation/
        // center + shared focal, LR decayed like positions) runs when the
        // readback lands, against the camera snapshot taken here.
        if self.config.pose_refine_lr > 0.0
            && iter >= self.config.pose_refine_start
            && iter < self.config.pose_refine_end
        {
            let tag = PendingPose {
                view_idx,
                quat: camera.quat,
                focal: camera.focal,
                decay: self.config.pos_lr_final_factor.powf(t),
            };
            if !self
                .pose_ring
                .encode_copy(&mut encoder, &self.raster.grad_cam, 0, 16 * 4, tag)
            {
                log::debug!("pose ring full — dropping pose sample for view {view_idx}");
            }
        }
        if let Some(t) = timer.as_ref() {
            t.resolve(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
        if !split {
            self.appearance.map_pending();
        }
        self.pose_ring.map_pending();
        let t_bwd = std::time::Instant::now();
        let t_pose = std::time::Instant::now();

        // Periodic dead-surfel relocation.
        if self.config.mcmc_every > 0
            && iter > 0
            && iter.is_multiple_of(self.config.mcmc_every)
            && iter + 500 < self.config.iters
        {
            self.mcmc_relocate(ctx);
        }

        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        self.phases.target += ms(t0, t_target);
        self.phases.fwd += ms(t_target, t_fwd);
        self.phases.appear += ms(t_fwd, t_appear);
        self.phases.bwd += ms(t_appear, t_bwd);
        self.phases.pose += ms(t_bwd, t_pose);
        self.phases.mcmc += ms(t_pose, std::time::Instant::now());
        self.phases.n += 1;
        if timer.is_some() {
            self.timer = timer;
        }
        view_idx
    }

    /// Fold a completed least-squares fit into a view's appearance affine:
    /// EMA update, clamps, and the global gauge anchor (the affines have a
    /// shared degree of freedom — all views could drift together chasing the
    /// render — so the mean correction is projected back to identity).
    fn apply_affine_fit(&mut self, view_idx: usize, fit: [f32; 6]) {
        let ap = &mut self.appear[view_idx];
        for k in 0..6 {
            ap[k] = 0.8 * ap[k] + 0.2 * fit[k];
        }
        for g in &mut ap[..3] {
            *g = g.clamp(0.5, 2.0);
        }
        for b in &mut ap[3..] {
            *b = b.clamp(-0.5, 0.5);
        }
        let n = self.appear.len() as f32;
        let mut mean = [0.0f32; 6];
        for a in &self.appear {
            for k in 0..6 {
                mean[k] += a[k] / n;
            }
        }
        for a in &mut self.appear {
            for k in 0..3 {
                a[k] /= mean[k].max(0.25);
                a[3 + k] -= mean[3 + k];
            }
        }
    }

    /// Host-side camera update from a completed `grad_cam` readback:
    /// layout [dl/dR (col-major 3×3), dl/dcenter (3), dl/dfocal, …].
    /// Rotation uses the right-perturbation R' = R·exp([δ]×): the Riemannian
    /// gradient is the antisymmetric part of B = Rᵀ·(dl/dR). Chains through
    /// the camera snapshot taken when the readback was issued.
    fn apply_pose_grads(&mut self, p: &PendingPose, bytes: &[u8]) {
        let g: &[f32] = bytemuck::cast_slice(bytes);
        let Some(grad) = camera_step_grad(g, p.quat) else {
            return;
        };
        // Center LR scales with this view's median scene depth: a center
        // move of d·δθ shifts the image about as much as a rotation of δθ.
        let depth = self.view_depth(p.view_idx);
        let t = self.pose_t[p.view_idx] + 1;
        self.pose_t[p.view_idx] = t;
        let lr = self.config.pose_refine_lr * p.decay;
        let step = adam6(
            &mut self.pose_m[p.view_idx],
            &mut self.pose_v[p.view_idx],
            t,
            &grad,
            lr,
            lr * depth,
        );
        let view = &mut self.views[p.view_idx];
        apply_camera_step(&mut view.camera, &step);

        // Shared focal in log-space (dL/d log f = f · dL/df).
        if self.config.focal_refine {
            let gf = g[12] * p.focal;
            let (ref mut m, ref mut v, ref mut tf) = self.focal_state;
            *tf += 1;
            *m = 0.9 * *m + 0.1 * gf;
            *v = 0.999 * *v + 0.001 * gf * gf;
            let mhat = *m / (1.0 - 0.9f32.powi(*tf as i32));
            let vhat = *v / (1.0 - 0.999f32.powi(*tf as i32));
            let lr_f = lr * 0.5;
            self.focal_scale *= (-lr_f * mhat / (vhat.sqrt() + 1e-10)).exp();
        }
    }

    /// Held-out PSNR with test-time pose refinement: each eval camera is
    /// photometrically aligned to the *frozen* model for `steps` iterations
    /// before scoring. This is the honest monocular protocol (BARF-style):
    /// training legitimately drifts the gauge away from the raw VO poses, so
    /// frozen eval cameras measure gauge drift, not model quality. The model
    /// is never touched (the optimizer step is simply not encoded).
    pub fn eval_psnr_refined(
        &mut self,
        ctx: &GpuContext,
        views: &[TrainView],
        steps: u32,
    ) -> f64 {
        let mut refined: Vec<TrainView> = views.to_vec();
        for view in &mut refined {
            let mut m = [0.0f32; 6];
            let mut v = [0.0f32; 6];
            // Depth for LR scaling, from the probe cloud.
            let inv = view.camera.quat.conjugate();
            let mut depths: Vec<f32> = self
                .probe_positions
                .iter()
                .map(|p| -(inv * (*p - view.camera.center)).z)
                .filter(|d| *d > 0.05)
                .collect();
            let depth = if depths.is_empty() {
                self.extent * 0.1
            } else {
                let mid = depths.len() / 2;
                *depths.select_nth_unstable_by(mid, f32::total_cmp).1
            };
            ctx.queue
                .write_buffer(&self.loss.target, 0, bytemuck::cast_slice(&view.target));
            for t in 1..=steps {
                let mut encoder = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                self.raster
                    .forward(ctx, &mut encoder, &view.camera, self.num_surfels);
                self.loss.encode(&mut encoder);
                ctx.queue.submit([encoder.finish()]);
                let mut encoder = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                self.raster.backward(&mut encoder, self.num_surfels);
                ctx.queue.submit([encoder.finish()]);
                let g = read_cam_grads(ctx, &self.raster.grad_cam);
                let Some(grad) = camera_step_grad(&g, view.camera.quat) else {
                    break;
                };
                let step = adam6(
                    &mut m,
                    &mut v,
                    t,
                    &grad,
                    self.config.pose_refine_lr.max(1e-3),
                    self.config.pose_refine_lr.max(1e-3) * depth,
                );
                apply_camera_step(&mut view.camera, &step);
            }
        }
        if self.config.appearance_start != u32::MAX {
            self.eval_psnr_affine(ctx, &refined)
        } else {
            self.eval_psnr(ctx, &refined)
        }
    }

    /// PSNR with a per-view affine (gain+bias) fitted between render and
    /// target before scoring — the eval counterpart of training-time
    /// appearance compensation (auto-exposure is not scene error).
    pub fn eval_psnr_affine(&self, ctx: &GpuContext, views: &[TrainView]) -> f64 {
        let mut total = 0.0;
        for view in views {
            let out = self.render_view(ctx, &view.camera);
            let ap = fit_affine(&out, &view.target)
                .unwrap_or([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
            let mut mse = 0.0f64;
            for (o, t) in out.iter().zip(&view.target) {
                for ch in 0..3 {
                    let e = (ap[ch] * o[ch] + ap[3 + ch] - t[ch]) as f64;
                    mse += e * e;
                }
            }
            mse /= (out.len() * 3) as f64;
            total += -10.0 * mse.max(1e-12).log10();
        }
        total / views.len().max(1) as f64
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

    /// Apply every in-flight async result (pose updates, appearance fits).
    /// Call after the last step before eval/export so nothing queued is lost.
    fn flush_async(&mut self, ctx: &GpuContext) {
        let pending = self.pose_ring.drain_blocking(&ctx.device);
        for (p, bytes) in pending {
            self.apply_pose_grads(&p, &bytes);
        }
        let fits: Vec<_> = self.appearance.drain_fits(&ctx.device);
        for (v, fit) in fits {
            self.apply_affine_fit(v, fit);
        }
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
                log::info!("  {}", self.phases.log_line());
                self.phases = PhaseTimes::default();
                if let Some(timer) = self.timer.as_mut() {
                    // Sampled on this iteration only; merge duplicate labels
                    // (the two radix sorts share theirs).
                    let mut merged: Vec<(String, f64)> = Vec::new();
                    for (label, dt) in timer.read(ctx) {
                        match merged.iter_mut().find(|(l, _)| *l == label) {
                            Some((_, acc)) => *acc += dt,
                            None => merged.push((label, dt)),
                        }
                    }
                    if !merged.is_empty() {
                        let line: Vec<String> = merged
                            .iter()
                            .map(|(l, ms)| format!("{l} {ms:.2}"))
                            .collect();
                        log::info!("  gpu ms (sampled): {}", line.join(" | "));
                    }
                }
            }
        }
        self.flush_async(ctx);
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


// --- Pose-refinement helpers (host-side; grad_cam layout:
// [dl/dR col-major 3x3 | dl/dcenter (3) | dl/dfocal | ...]) ---

fn read_cam_grads(ctx: &GpuContext, buf: &wgpu::Buffer) -> Vec<f32> {
    bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, buf)).to_vec()
}

/// [rot(3), center(3)] gradient. Rotation uses the right-perturbation
/// R' = R·exp([δ]×): the Riemannian gradient is the antisymmetric part of
/// B = Rᵀ·(dl/dR). Returns None on non-finite grads (skip the step).
fn camera_step_grad(g: &[f32], quat: glam::Quat) -> Option<[f32; 6]> {
    if g.iter().take(13).any(|v| !v.is_finite()) {
        return None;
    }
    let a = glam::Mat3::from_cols(
        glam::Vec3::new(g[0], g[1], g[2]),
        glam::Vec3::new(g[3], g[4], g[5]),
        glam::Vec3::new(g[6], g[7], g[8]),
    );
    let b = glam::Mat3::from_quat(quat).transpose() * a;
    Some([
        b.col(1)[2] - b.col(2)[1], // B32 − B23
        b.col(2)[0] - b.col(0)[2], // B13 − B31
        b.col(0)[1] - b.col(1)[0], // B21 − B12
        g[9],
        g[10],
        g[11],
    ])
}

/// One Adam step (β1 0.9, β2 0.999) over [rot, center] with separate LRs.
fn adam6(
    m: &mut [f32; 6],
    v: &mut [f32; 6],
    t: u32,
    grad: &[f32; 6],
    lr_rot: f32,
    lr_cen: f32,
) -> [f32; 6] {
    let bc1 = 1.0 - 0.9f32.powi(t as i32);
    let bc2 = 1.0 - 0.999f32.powi(t as i32);
    let mut step = [0.0f32; 6];
    for k in 0..6 {
        m[k] = 0.9 * m[k] + 0.1 * grad[k];
        v[k] = 0.999 * v[k] + 0.001 * grad[k] * grad[k];
        let lr = if k < 3 { lr_rot } else { lr_cen };
        step[k] = lr * (m[k] / bc1) / ((v[k] / bc2).sqrt() + 1e-10);
    }
    step
}

/// Per-channel least-squares affine target ≈ gain·render + bias, on a pixel
/// subsample. Returns [gain rgb, bias rgb]; None if the render is degenerate.
fn fit_affine(render: &[[f32; 4]], target: &[[f32; 4]]) -> Option<[f32; 6]> {
    let mut out = [1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0];
    for ch in 0..3 {
        let (mut sr, mut st, mut srr, mut srt, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in (0..render.len()).step_by(4) {
            let r = render[i][ch] as f64;
            let t = target[i][ch] as f64;
            sr += r;
            st += t;
            srr += r * r;
            srt += r * t;
            n += 1.0;
        }
        let var = srr - sr * sr / n;
        if var < 1e-9 || !var.is_finite() {
            return None;
        }
        let g = (srt - sr * st / n) / var;
        let b = (st - g * sr) / n;
        if !(g.is_finite() && b.is_finite()) {
            return None;
        }
        out[ch] = g as f32;
        out[3 + ch] = b as f32;
    }
    Some(out)
}

fn apply_camera_step(camera: &mut gs_kernels::RasterCamera, step: &[f32; 6]) {
    let delta = glam::Vec3::new(-step[0], -step[1], -step[2]);
    camera.quat = (camera.quat * glam::Quat::from_scaled_axis(delta)).normalize();
    camera.center -= glam::Vec3::new(step[3], step[4], step[5]);
}
