//! GPU-resident training targets + GPU-side appearance fitting.
//!
//! All training views are uploaded once into a single storage-buffer atlas at
//! construction; each iteration a tiny `target_transform` dispatch
//! inverse-corrects the chosen view into the loss target buffer (replacing
//! the old per-iteration host transform + full-image `write_buffer`). The
//! per-view affine fit runs as a GPU reduction whose 48-byte sufficient
//! statistics come back through an async [`ReadbackRing`] with 1–2
//! iterations of latency — the hot loop never blocks on a readback.

use gs_wgpu::{GpuContext, ReadbackRing, buffers};

use crate::trainer::TrainView;

/// Fit subsample stride — matches the host fit this replaced (every 4th
/// pixel by linear index).
const FIT_STRIDE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AppearParams {
    gain: [f32; 4],
    bias: [f32; 4],
    npix: u32,
    stride: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct Appearance {
    transform_pipeline: wgpu::ComputePipeline,
    fit_pipeline: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    stats: wgpu::Buffer,
    /// One big buffer of raw targets, 256-byte-aligned stride per view.
    /// Held alive by the per-view bind groups.
    #[allow(dead_code)]
    atlas: wgpu::Buffer,
    transform_bgs: Vec<wgpu::BindGroup>,
    fit_bgs: Vec<wgpu::BindGroup>,
    ring: ReadbackRing<usize>,
    npix: u32,
    /// Byte stride between atlas slots, and how many slots exist.
    stride: u64,
    capacity: usize,
    /// Buffers the per-view bind groups reference; held so views can be
    /// appended after construction (wgpu buffers are refcounted handles).
    loss_target: wgpu::Buffer,
    render: wgpu::Buffer,
    /// Sample count of the subsampled fit (host-known, not summed on GPU).
    n_samples: f64,
}

impl Appearance {
    /// `target` is the loss target buffer the transform writes; `render` the
    /// rasterizer's out_color the fit compares against.
    pub fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        views: &[TrainView],
        target: &wgpu::Buffer,
        render: &wgpu::Buffer,
    ) -> Self {
        Self::with_capacity(ctx, width, height, views, views.len(), target, render)
    }

    /// As [`Self::new`], but the atlas is sized for `capacity` views so more
    /// can be appended later with [`Self::push_view`]. The atlas is uploaded
    /// once and stays GPU-resident (CLAUDE.md); growing it would mean
    /// reallocating and re-uploading every target, so incremental mapping
    /// reserves instead.
    #[allow(clippy::too_many_arguments)]
    pub fn with_capacity(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        views: &[TrainView],
        capacity: usize,
        target: &wgpu::Buffer,
        render: &wgpu::Buffer,
    ) -> Self {
        let capacity = capacity.max(views.len()).max(1);
        let device = &ctx.device;
        let npix = width * height;
        let view_bytes = npix as u64 * 16;
        // Storage-buffer binding offsets must respect the device alignment.
        let align = device.limits().min_storage_buffer_offset_alignment as u64;
        let stride = view_bytes.div_ceil(align) * align;

        let atlas = buffers::storage_empty(device, "appear-atlas", stride * capacity as u64);
        for (v, view) in views.iter().enumerate() {
            assert_eq!(view.target.len(), npix as usize);
            ctx.queue.write_buffer(
                &atlas,
                v as u64 * stride,
                bytemuck::cast_slice(&view.target),
            );
        }

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("appear-uniform"),
            size: std::mem::size_of::<AppearParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats = buffers::storage_empty(device, "appear-stats", 12 * 4);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("appearance"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/appearance.wgsl").into()),
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let transform_pipeline = make("target_transform");
        let fit_pipeline = make("fit_reduce");

        let atlas_slice = |v: usize| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &atlas,
                offset: v as u64 * stride,
                size: std::num::NonZeroU64::new(view_bytes),
            })
        };
        let transform_bgs = (0..views.len())
            .map(|v| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("appear-transform"),
                    layout: &transform_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: atlas_slice(v) },
                        wgpu::BindGroupEntry { binding: 2, resource: target.as_entire_binding() },
                    ],
                })
            })
            .collect();
        let fit_bgs = (0..views.len())
            .map(|v| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("appear-fit"),
                    layout: &fit_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: atlas_slice(v) },
                        wgpu::BindGroupEntry { binding: 3, resource: render.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: stats.as_entire_binding() },
                    ],
                })
            })
            .collect();

        Self {
            transform_pipeline,
            fit_pipeline,
            uniform,
            stats,
            atlas,
            transform_bgs,
            fit_bgs,
            ring: ReadbackRing::new(device, "appear-stats", 12 * 4, 4),
            npix,
            stride,
            capacity,
            loss_target: target.clone(),
            render: render.clone(),
            n_samples: npix.div_ceil(FIT_STRIDE) as f64,
        }
    }

    /// Number of views the atlas can still take.
    pub fn headroom(&self) -> usize {
        self.capacity - self.transform_bgs.len()
    }

    /// Append one view's target to the atlas and build its bind groups.
    /// Returns its index, or None when the reserved capacity is exhausted.
    pub fn push_view(&mut self, ctx: &GpuContext, view: &TrainView) -> Option<usize> {
        let v = self.transform_bgs.len();
        if v >= self.capacity {
            return None;
        }
        assert_eq!(view.target.len(), self.npix as usize);
        ctx.queue.write_buffer(
            &self.atlas,
            v as u64 * self.stride,
            bytemuck::cast_slice(&view.target),
        );
        let view_bytes = self.npix as u64 * 16;
        let slice = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.atlas,
            offset: v as u64 * self.stride,
            size: std::num::NonZeroU64::new(view_bytes),
        });
        self.transform_bgs.push(ctx.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("appear-transform"),
                layout: &self.transform_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: slice.clone() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.loss_target.as_entire_binding() },
                ],
            },
        ));
        self.fit_bgs.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("appear-fit"),
            layout: &self.fit_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: slice },
                wgpu::BindGroupEntry { binding: 3, resource: self.render.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.stats.as_entire_binding() },
            ],
        }));
        Some(v)
    }

    /// Record the target transform for `view_idx` with the given affine
    /// ([gain rgb, bias rgb] — identity before appearance starts). Must be
    /// encoded before the loss reads the target.
    pub fn encode_transform(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view_idx: usize,
        affine: &[f32; 6],
        mut timer: Option<&mut gs_wgpu::GpuTimer>,
    ) {
        let u = AppearParams {
            gain: [affine[0], affine[1], affine[2], 1.0],
            bias: [affine[3], affine[4], affine[5], 0.0],
            npix: self.npix,
            stride: FIT_STRIDE,
            _pad0: 0,
            _pad1: 0,
        };
        ctx.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("appear-transform"),
            timestamp_writes: gs_wgpu::profile::scope(&mut timer, "appear-transform"),
        });
        pass.set_pipeline(&self.transform_pipeline);
        pass.set_bind_group(0, &self.transform_bgs[view_idx], &[]);
        pass.dispatch_workgroups(self.npix.div_ceil(256), 1, 1);
    }

    /// Record the fit reduction (fresh render vs the ORIGINAL target) and a
    /// copy of its statistics into the async ring, tagged with the view.
    /// Skips silently if every ring slot is in flight (a fit sample is lost,
    /// never a stall). Caller must invoke [`Self::map_pending`] after submit.
    pub fn encode_fit(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view_idx: usize,
        mut timer: Option<&mut gs_wgpu::GpuTimer>,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("appear-fit"),
                timestamp_writes: gs_wgpu::profile::scope(&mut timer, "appear-fit"),
            });
            pass.set_pipeline(&self.fit_pipeline);
            pass.set_bind_group(0, &self.fit_bgs[view_idx], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if !self
            .ring
            .encode_copy(encoder, &self.stats, 0, 12 * 4, view_idx)
        {
            log::debug!("appearance fit ring full — dropping sample for view {view_idx}");
        }
    }

    /// Start mapping the fit copies recorded this submit (call right after
    /// `queue.submit`).
    pub fn map_pending(&mut self) {
        self.ring.map_pending();
    }

    /// Deliver completed fits: (view, [gain rgb, bias rgb]). Non-blocking;
    /// degenerate fits are dropped like the host fit did.
    pub fn poll_fits(&mut self, device: &wgpu::Device) -> Vec<(usize, [f32; 6])> {
        let ready = self.ring.poll_ready(device);
        self.to_fits(ready)
    }

    /// Blocking counterpart of [`Self::poll_fits`] — waits for every
    /// in-flight fit (end of training).
    pub fn drain_fits(&mut self, device: &wgpu::Device) -> Vec<(usize, [f32; 6])> {
        let ready = self.ring.drain_blocking(device);
        self.to_fits(ready)
    }

    fn to_fits(&self, ready: Vec<(usize, Vec<u8>)>) -> Vec<(usize, [f32; 6])> {
        ready
            .into_iter()
            .filter_map(|(view, bytes)| {
                let sums: &[f32] = bytemuck::cast_slice(&bytes);
                fit_affine_from_sums(sums.try_into().unwrap(), self.n_samples)
                    .map(|fit| (view, fit))
            })
            .collect()
    }
}

/// Per-channel least-squares affine from GPU sufficient statistics — the
/// sums-form twin of the old host `fit_affine` (layout [Σr, Σt, Σrr, Σrt]
/// per rgb). Returns None on degenerate renders, matching the host fit.
fn fit_affine_from_sums(sums: [f32; 12], n: f64) -> Option<[f32; 6]> {
    let mut out = [1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0];
    for ch in 0..3 {
        let sr = sums[ch] as f64;
        let st = sums[3 + ch] as f64;
        let srr = sums[6 + ch] as f64;
        let srt = sums[9 + ch] as f64;
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
