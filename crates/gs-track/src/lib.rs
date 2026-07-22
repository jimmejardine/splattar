//! Photometric camera tracking.
//!
//! Given a frozen map and an incoming frame, recover the camera pose by
//! descending the difference between the rendered map and the frame. Every
//! pixel contributes, which is the whole reason for the architecture: feature
//! tracking discards all but a thousand corners per frame.
//!
//! The map is never touched here. Tracking and mapping share one loss but run
//! at different times — tracking on every frame with the map frozen, mapping on
//! keyframes with the window's poses in play.

use glam::{Mat3, Quat, Vec3};
use gs_kernels::{RasterCamera, Rasterizer};
use gs_wgpu::GpuContext;

/// How the residual behaved over one tracking call, coarse level first.
#[derive(Clone, Debug, Default)]
pub struct TrackReport {
    /// Residual after each iteration, in order. The shape of this is the
    /// diagnostic: a healthy lock falls steeply then flattens; a diverging one
    /// rises; a stuck one is flat from the first step.
    pub residuals: Vec<f32>,
    /// Fraction of the frame the map covered, from the final iteration.
    pub coverage: f32,
    pub iterations: u32,
    /// Residual of the pose actually returned — the best seen, not the last
    /// visited.
    pub best: f32,
}

impl TrackReport {
    /// Residual of the returned pose. Deliberately the best seen rather than
    /// the last measured: the trace's last entry is taken BEFORE the final
    /// step, so it describes a pose the caller never receives.
    pub fn final_residual(&self) -> f32 {
        self.best
    }

    /// Did the residual actually improve? A tracker that reports a pose
    /// without this having fallen has not tracked anything.
    pub fn improved(&self) -> bool {
        self.residuals.first().is_some_and(|a| self.best < *a)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrackConfig {
    pub iterations: u32,
    /// Learning rate for the rotation triple.
    pub lr_rot: f32,
    /// Learning rate for the camera centre, in scene units. Scaled by the
    /// scene's depth at the call site: a centre move of `d·δθ` shifts the image
    /// about as much as a rotation of `δθ`, so the two rates only balance when
    /// the translation rate carries the depth.
    pub lr_center: f32,
    /// Stop once the residual stops improving by this fraction.
    pub min_improvement: f32,
    /// Consecutive non-improving iterations tolerated before stopping.
    ///
    /// Adam orbits a minimum rather than settling on it, so single steps that
    /// fail to improve are normal, not a plateau. Stopping on the first one
    /// ends the descent while it still has most of its progress left — measured
    /// here as quitting at iteration 18 with 0.063 m of error still on the
    /// table, against 0.001 m when allowed to run.
    pub patience: u32,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            iterations: 60,
            lr_rot: 2e-3,
            lr_center: 2e-3,
            min_improvement: 1e-4,
            patience: 10,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LossParams {
    npix: u32,
    gain: f32,
    bias: f32,
    _pad: u32,
}

/// The photometric loss, and the pose descent that consumes its gradient.
pub struct Tracker {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    target: wgpu::Buffer,
    stats: wgpu::Buffer,
    npix: u32,
}

impl Tracker {
    pub fn new(ctx: &GpuContext, raster: &Rasterizer, width: u32, height: u32) -> Self {
        let device = &ctx.device;
        let npix = width * height;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("photometric"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/photometric.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("photometric"),
            layout: None,
            module: &module,
            entry_point: Some("photometric"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photometric-uni"),
            size: std::mem::size_of::<LossParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let target =
            gs_wgpu::buffers::storage_empty(device, "photometric-target", npix as u64 * 16);
        let stats = gs_wgpu::buffers::storage_empty(device, "photometric-stats", 16);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photometric"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: raster.out_color.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: target.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raster.dl_dcolor.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: stats.as_entire_binding(),
                },
            ],
        });
        Self {
            pipeline,
            bind,
            uniform,
            target,
            stats,
            npix,
        }
    }

    /// Upload the frame to align against. Held GPU-side across iterations —
    /// the target does not change while a pose is being recovered.
    pub fn set_target(&self, ctx: &GpuContext, frame: &[[f32; 4]]) {
        assert_eq!(frame.len(), self.npix as usize, "frame size mismatch");
        ctx.queue
            .write_buffer(&self.target, 0, bytemuck::cast_slice(frame));
    }

    /// Recover `camera` by photometric descent against the frozen map.
    ///
    /// Returns the residual trace rather than just a pose: whether the residual
    /// fell, and how, is the only evidence that the returned pose means
    /// anything.
    pub fn track(
        &self,
        ctx: &GpuContext,
        raster: &Rasterizer,
        surfels: u32,
        camera: &mut RasterCamera,
        depth: f32,
        cfg: &TrackConfig,
    ) -> TrackReport {
        let mut report = TrackReport::default();
        let (mut m, mut v) = ([0.0f32; 6], [0.0f32; 6]);
        let mut best = f32::INFINITY;
        let mut stale = 0u32;
        // Adam does not converge to a point; near a minimum it orbits one at a
        // radius set by the learning rate. Returning the LAST iterate therefore
        // returns an arbitrary member of that orbit, which can be markedly
        // worse than one already visited. Keep the best.
        let mut best_camera = camera.clone();

        for t in 1..=cfg.iterations {
            let (residual, coverage, grad) = self.step(ctx, raster, surfels, camera);
            report.residuals.push(residual);
            report.coverage = coverage;
            report.iterations = t;

            if residual < best * (1.0 - cfg.min_improvement) {
                best = residual;
                best_camera = camera.clone();
                stale = 0;
            } else {
                if residual < best {
                    best = residual;
                    best_camera = camera.clone();
                }
                stale += 1;
            }
            // Stop when it stops paying. Running a converged tracker on is not
            // free — every frame pays for it — but one non-improving step is
            // Adam orbiting, not a plateau.
            if stale >= cfg.patience {
                break;
            }

            let Some(grad) = grad else {
                // Non-finite camera gradient: the render has degenerated (the
                // camera is inside geometry, or sees none of it). Stopping
                // keeps the best pose rather than stepping on garbage.
                log::debug!("tracking: non-finite gradient at iteration {t}");
                break;
            };
            let step = adam6(&mut m, &mut v, t, &grad, cfg.lr_rot, cfg.lr_center * depth);
            apply_camera_step(camera, &step);
        }
        *camera = best_camera;
        report.best = best;
        report
    }

    /// One forward + loss + backward, returning (residual, coverage, camera
    /// gradient).
    fn step(
        &self,
        ctx: &GpuContext,
        raster: &Rasterizer,
        surfels: u32,
        camera: &RasterCamera,
    ) -> (f32, f32, Option<[f32; 6]>) {
        ctx.queue.write_buffer(
            &self.stats,
            0,
            bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 0.0]),
        );
        ctx.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&LossParams {
                npix: self.npix,
                gain: 1.0,
                bias: 0.0,
                _pad: 0,
            }),
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("track-step"),
            });
        raster.forward(ctx, &mut encoder, camera, surfels);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("photometric"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(self.npix.div_ceil(256), 1, 1);
        }
        raster.backward(&mut encoder, surfels);
        ctx.queue.submit([encoder.finish()]);

        let stats: Vec<f32> =
            bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, &self.stats))
                .to_vec();
        let cam: Vec<f32> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            &raster.grad_cam,
        ))
        .to_vec();
        (stats[0], stats[1], camera_step_grad(&cam, camera.quat))
    }
}

/// `grad_cam` holds dL/dR as a column-major 3×3 plus dL/dcentre and dL/dfocal.
/// A rotation lives on a manifold, so the usable gradient is the Riemannian one
/// — the antisymmetric part of `Rᵀ·(dL/dR)` — not the raw matrix entries.
fn camera_step_grad(g: &[f32], quat: Quat) -> Option<[f32; 6]> {
    if g.len() < 12 || g.iter().take(12).any(|v| !v.is_finite()) {
        return None;
    }
    let a = Mat3::from_cols(
        Vec3::new(g[0], g[1], g[2]),
        Vec3::new(g[3], g[4], g[5]),
        Vec3::new(g[6], g[7], g[8]),
    );
    let b = Mat3::from_quat(quat).transpose() * a;
    Some([
        b.col(1)[2] - b.col(2)[1],
        b.col(2)[0] - b.col(0)[2],
        b.col(0)[1] - b.col(1)[0],
        g[9],
        g[10],
        g[11],
    ])
}

/// One Adam step over [rotation, centre] with separate rates.
fn adam6(
    m: &mut [f32; 6],
    v: &mut [f32; 6],
    t: u32,
    grad: &[f32; 6],
    lr_rot: f32,
    lr_center: f32,
) -> [f32; 6] {
    let bc1 = 1.0 - 0.9f32.powi(t as i32);
    let bc2 = 1.0 - 0.999f32.powi(t as i32);
    let mut step = [0.0f32; 6];
    for k in 0..6 {
        m[k] = 0.9 * m[k] + 0.1 * grad[k];
        v[k] = 0.999 * v[k] + 0.001 * grad[k] * grad[k];
        let lr = if k < 3 { lr_rot } else { lr_center };
        step[k] = lr * (m[k] / bc1) / ((v[k] / bc2).sqrt() + 1e-10);
    }
    step
}

fn apply_camera_step(camera: &mut RasterCamera, step: &[f32; 6]) {
    // Right-perturbation: R' = R·exp([δ]×), matching how the gradient above was
    // derived. Applying it on the left would rotate about the world axes and
    // descend a different quantity than the one measured.
    let delta = Vec3::new(-step[0], -step[1], -step[2]);
    camera.quat = (camera.quat * Quat::from_scaled_axis(delta)).normalize();
    camera.center -= Vec3::new(step[3], step[4], step[5]);
}
