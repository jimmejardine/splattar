//! Dense surfel initialization from plane-sweep depth.
//!
//! The alternative it replaces ([`crate::upsample_to_budget`]) grows a sparse
//! SfM cloud to the surfel budget by duplicating points with random jitter at
//! low opacity, then relies on thousands of training iterations to discover
//! geometry that is measurable up front. Surfels born on surfaces, with
//! normals from a local plane fit, start where descent would have had to
//! travel to — which is the single biggest convergence lever in PLAN.md.
//!
//! The sweep runs on a coarser grid than the training targets: depth does not
//! need full resolution, and cost is O(pixels x hypotheses x neighbours x
//! patch).

use glam::{Quat, Vec3};
use gs_wgpu::{GpuContext, buffers};

use crate::trainer::{InitialSurfels, TrainView};

const SH_C0: f32 = 0.282_094_79;

#[derive(Clone, Copy, Debug)]
pub struct SweepOptions {
    /// Sweep grid = target resolution / this.
    pub downscale: u32,
    /// Inverse-depth hypotheses per pixel.
    pub hypotheses: u32,
    /// Neighbour views scored per reference view.
    pub neighbours: u32,
    /// NCC patch half-width (1 = 3x3, 2 = 5x5, 3 = 7x7).
    pub patch: i32,
    /// Minimum confidence (score x margin over the runner-up) to spawn.
    pub min_confidence: f32,
    /// Spawn every Nth accepted pixel — one surfel per pixel is far past the
    /// budget and mostly redundant.
    pub spawn_stride: u32,
    /// Depth search range, as multiples of the median scene depth.
    pub depth_range: (f32, f32),
}

impl Default for SweepOptions {
    fn default() -> Self {
        Self {
            // 2, not 4: at 4 the sweep resolved only ~20% of the surfel
            // budget and random jitter filled the rest. Quartering the pixel
            // count costs ~4x the sweep time, which is still seconds against a
            // training run of minutes (RESULTS.md 2026-07-22).
            downscale: 2,
            hypotheses: 64,
            neighbours: 4,
            patch: 2,
            min_confidence: 0.15,
            spawn_stride: 2,
            depth_range: (0.15, 4.0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SweepCamGpu {
    rot0: [f32; 4],
    rot1: [f32; 4],
    rot2: [f32; 4],
    center: [f32; 4],
    focal: f32,
    _p: [f32; 3],
}

impl SweepCamGpu {
    fn new(view: &TrainView) -> Self {
        let m = glam::Mat3::from_quat(view.camera.quat.normalize());
        Self {
            rot0: m.x_axis.extend(0.0).to_array(),
            rot1: m.y_axis.extend(0.0).to_array(),
            rot2: m.z_axis.extend(0.0).to_array(),
            center: view.camera.center.extend(0.0).to_array(),
            focal: view.camera.focal,
            _p: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SweepParams {
    width: u32,
    height: u32,
    full_width: u32,
    full_height: u32,
    stride_texels: u32,
    n_hypotheses: u32,
    n_neighbours: u32,
    patch: i32,
    inv_depth_min: f32,
    inv_depth_max: f32,
    ref_slot: u32,
    _pad: u32,
}

/// Per-reference-view sweep result, in the reference camera's frame.
pub struct DepthMap {
    pub width: usize,
    pub height: usize,
    /// 0 where the sweep found nothing trustworthy.
    pub depth: Vec<f32>,
    pub confidence: Vec<f32>,
}

/// Rank candidate neighbours for a reference view by usable parallax.
///
/// Baseline, not proximity: a neighbour that only rotated carries no depth
/// information at all. This is the same trap the VO bootstrap hit — panning
/// produces hundreds of pixels of flow with zero baseline — and picking
/// neighbours by frame distance would walk straight into it. Views whose
/// optical axes have diverged too far are dropped as well; the patches stop
/// corresponding.
fn pick_neighbours(views: &[TrainView], r: usize, want: usize, depth: f32) -> Vec<usize> {
    let rc = &views[r].camera;
    let r_fwd = rc.quat * Vec3::NEG_Z;
    let mut scored: Vec<(f32, usize)> = views
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != r)
        .filter_map(|(i, v)| {
            let baseline = (v.camera.center - rc.center).length();
            // Parallax angle subtended at the scene, in radians.
            let parallax = (baseline / depth.max(1e-6)).atan();
            let cos_axes = (v.camera.quat * Vec3::NEG_Z).dot(r_fwd);
            // Below ~1 degree of parallax the depth is unconstrained; past
            // ~40 degrees of axis divergence the patches no longer match.
            if parallax < 0.017 || cos_axes < 0.766 {
                return None;
            }
            // Prefer moderate parallax: too little is uninformative, too much
            // breaks the fronto-parallel patch assumption. Peak near 6 deg.
            let target = 0.105_f32;
            Some(((parallax - target).abs(), i))
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().take(want).map(|(_, i)| i).collect()
}

/// Test hook for [`pick_neighbours`] — baseline selection is the piece most
/// likely to silently pick useless views, so it is worth asserting directly.
#[doc(hidden)]
pub fn pick_neighbours_for_test(
    views: &[TrainView],
    r: usize,
    want: usize,
    depth: f32,
) -> Vec<usize> {
    pick_neighbours(views, r, want, depth)
}

/// Median depth of a set of world points along a view's optical axis, used to
/// centre the sweep range. Points behind the camera are ignored.
fn median_depth(views: &[TrainView], points: &[Vec3]) -> f32 {
    let mut d: Vec<f32> = Vec::new();
    let vstride = (views.len() / 16).max(1);
    let pstride = (points.len() / 512).max(1);
    for v in views.iter().step_by(vstride) {
        let fwd = v.camera.quat * Vec3::NEG_Z;
        for p in points.iter().step_by(pstride) {
            let z = (*p - v.camera.center).dot(fwd);
            if z > 0.05 {
                d.push(z);
            }
        }
    }
    if d.is_empty() {
        return 1.0;
    }
    let mid = d.len() / 2;
    *d.select_nth_unstable_by(mid, f32::total_cmp).1
}

/// GPU plane sweep over every view, returning one depth map per view.
pub struct PlaneSweep {
    pipeline: wgpu::ComputePipeline,
    atlas: wgpu::Buffer,
    stride_texels: u32,
    full: (u32, u32),
    n_views: usize,
}

impl PlaneSweep {
    /// Uploads every view's target into one atlas. Views are read many times
    /// (each is a neighbour of several references), so a single resident copy
    /// beats re-uploading per reference.
    pub fn new(ctx: &GpuContext, width: u32, height: u32, views: &[TrainView]) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plane-sweep"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/plane_sweep.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("plane-sweep"),
            layout: None,
            module: &module,
            entry_point: Some("sweep"),
            compilation_options: Default::default(),
            cache: None,
        });
        let npix = (width * height) as u64;
        let view_bytes = npix * 16;
        let align = device.limits().min_storage_buffer_offset_alignment as u64;
        // Indexed as one flat array in the shader, so the stride must be a
        // whole number of texels as well as aligned.
        let stride_bytes = view_bytes.div_ceil(align.max(16)) * align.max(16);
        let stride_texels = (stride_bytes / 16) as u32;
        let atlas = buffers::storage_empty(
            device,
            "sweep-atlas",
            stride_bytes * views.len().max(1) as u64,
        );
        for (i, view) in views.iter().enumerate() {
            assert_eq!(view.target.len(), npix as usize, "view target size mismatch");
            ctx.queue.write_buffer(
                &atlas,
                i as u64 * stride_bytes,
                bytemuck::cast_slice(&view.target),
            );
        }
        Self {
            pipeline,
            atlas,
            stride_texels,
            full: (width, height),
            n_views: views.len(),
        }
    }

    /// Sweep one reference view against its chosen neighbours.
    pub fn run(
        &self,
        ctx: &GpuContext,
        views: &[TrainView],
        r: usize,
        neighbours: &[usize],
        scene_depth: f32,
        opts: &SweepOptions,
    ) -> DepthMap {
        assert_eq!(views.len(), self.n_views);
        let ds = opts.downscale.max(1);
        let (w, h) = (
            (self.full.0 / ds).max(1),
            (self.full.1 / ds).max(1),
        );
        let n = (w * h) as usize;
        if neighbours.is_empty() {
            return DepthMap {
                width: w as usize,
                height: h as usize,
                depth: vec![0.0; n],
                confidence: vec![0.0; n],
            };
        }

        let mut cams = vec![SweepCamGpu::new(&views[r])];
        let mut slots = vec![r as u32];
        for &i in neighbours {
            cams.push(SweepCamGpu::new(&views[i]));
            slots.push(i as u32);
        }
        let near = scene_depth * opts.depth_range.0.max(1e-3);
        let far = scene_depth * opts.depth_range.1.max(opts.depth_range.0 + 1e-3);
        let params = SweepParams {
            width: w,
            height: h,
            full_width: self.full.0,
            full_height: self.full.1,
            stride_texels: self.stride_texels,
            // MAX_H in the shader bounds the per-invocation cost curve.
            n_hypotheses: opts.hypotheses.clamp(2, 128),
            n_neighbours: neighbours.len() as u32,
            patch: opts.patch.clamp(1, 3),
            // Uniform in inverse depth; min/max are swapped relative to depth.
            inv_depth_min: 1.0 / far,
            inv_depth_max: 1.0 / near,
            ref_slot: r as u32,
            _pad: 0,
        };

        let device = &ctx.device;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sweep-uniform"),
            size: std::mem::size_of::<SweepParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
        let cam_buf = buffers::storage_init(device, "sweep-cams", bytemuck::cast_slice(&cams));
        let slot_buf = buffers::storage_init(device, "sweep-slots", bytemuck::cast_slice(&slots));
        let out_depth = buffers::storage_empty(device, "sweep-depth", n as u64 * 4);
        let out_conf = buffers::storage_empty(device, "sweep-conf", n as u64 * 4);

        fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sweep-bg"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &uniform),
                bind(1, &cam_buf),
                bind(2, &slot_buf),
                bind(3, &self.atlas),
                bind(4, &out_depth),
                bind(5, &out_conf),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("plane-sweep"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("plane-sweep"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        ctx.queue.submit([encoder.finish()]);

        DepthMap {
            width: w as usize,
            height: h as usize,
            depth: bytemuck::cast_slice(&buffers::readback(&ctx.device, &ctx.queue, &out_depth))
                .to_vec(),
            confidence: bytemuck::cast_slice(&buffers::readback(
                &ctx.device,
                &ctx.queue,
                &out_conf,
            ))
            .to_vec(),
        }
    }
}

/// Build surfels from a swept depth map: back-project accepted pixels, fit a
/// local plane for the normal, size each surfel to its pixel footprint, and
/// take colour from the target.
fn surfels_from_depth(
    view: &TrainView,
    full: (u32, u32),
    map: &DepthMap,
    opts: &SweepOptions,
    out: &mut InitialSurfels,
) -> (usize, usize) {
    let (fw, fh) = (full.0 as f32, full.1 as f32);
    let (cx, cy) = (fw * 0.5, fh * 0.5);
    let focal = view.camera.focal;
    let rot = view.camera.quat.normalize();
    let center = view.camera.center;
    let (sx, sy) = (fw / map.width as f32, fh / map.height as f32);

    // Reference-frame back-projection of a sweep pixel, in CAMERA space.
    let cam_point = |gx: usize, gy: usize| -> Option<Vec3> {
        let d = map.depth[gy * map.width + gx];
        if d <= 0.0 || map.confidence[gy * map.width + gx] < opts.min_confidence {
            return None;
        }
        let px = (gx as f32 + 0.5) * sx;
        let py = (gy as f32 + 0.5) * sy;
        Some(Vec3::new(
            (px - cx) * d / focal,
            -(py - cy) * d / focal,
            -d,
        ))
    };

    let stride = opts.spawn_stride.max(1) as usize;
    let mut candidates = 0usize;
    let mut spawned = 0usize;
    for gy in (1..map.height.saturating_sub(1)).step_by(stride) {
        for gx in (1..map.width.saturating_sub(1)).step_by(stride) {
            candidates += 1;
            let Some(p) = cam_point(gx, gy) else { continue };
            let foot = p.length() * sx / focal;

            // Normal from a least-squares plane over the accepted 3x3
            // neighbourhood. Requiring two SPECIFIC neighbours (a finite-
            // difference cross product) compounds the per-pixel acceptance
            // rate to roughly p^3 and threw away most of a real interior
            // sweep; needing any 3 of 9 both yields far more and fits a
            // better normal.
            //
            // Fitted in INVERSE depth, which is exactly affine in pixel
            // coordinates for a plane: 1/d = A(u-cx) + B(v-cy) + C, so the
            // fit is linear and the normal falls out in closed form.
            let mut ata = glam::Mat3::ZERO;
            let mut atb = Vec3::ZERO;
            let mut n_pts = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (gx as i32 + dx, gy as i32 + dy);
                    if nx < 0 || ny < 0 || nx as usize >= map.width || ny as usize >= map.height {
                        continue;
                    }
                    let Some(q) = cam_point(nx as usize, ny as usize) else {
                        continue;
                    };
                    // Same-surface gate: a step far larger than the local
                    // pixel footprint is an occlusion edge, and fitting across
                    // it produces a normal describing nothing.
                    if (q.z - p.z).abs() > 8.0 * foot {
                        continue;
                    }
                    let u = (nx as f32 + 0.5) * sx - cx;
                    let v = (ny as f32 + 0.5) * sy - cy;
                    let row = Vec3::new(u, v, 1.0);
                    let inv_d = 1.0 / (-q.z).max(1e-6);
                    ata += glam::Mat3::from_cols(row * row.x, row * row.y, row * row.z);
                    atb += row * inv_d;
                    n_pts += 1;
                }
            }
            if n_pts < 3 || ata.determinant().abs() < 1e-18 {
                continue;
            }
            let abc = ata.inverse() * atb;
            // 1/d = A·u + B·v + C  =>  n ∝ (A·f, −B·f, −C).
            let n_cam = Vec3::new(abc.x * focal, -abc.y * focal, -abc.z);
            if n_cam.length_squared() < 1e-20 {
                continue;
            }
            let mut n_cam = n_cam.normalize();
            // Face the camera (camera looks down -z, so the normal must have
            // a positive z component in camera space).
            if n_cam.z < 0.0 {
                n_cam = -n_cam;
            }

            let pos = rot * p + center;
            let normal = rot * n_cam;
            // A surfel's quat maps its local frame to world with +z the
            // normal, matching the rasterizer's tu/tv/normal basis.
            let quat = Quat::from_rotation_arc(Vec3::Z, normal);
            // One surfel covers its own footprint; sized to the spawn stride
            // so the spawned set tiles the surface without heavy overlap.
            let radius = (foot * stride as f32 * 0.5).max(1e-5);

            let px = ((gx as f32 + 0.5) * sx) as usize;
            let py = ((gy as f32 + 0.5) * sy) as usize;
            let t = view.target[(py.min(full.1 as usize - 1)) * full.0 as usize
                + px.min(full.0 as usize - 1)];

            out.positions.push(pos);
            out.scales.push([radius, radius]);
            out.quats.push(quat.to_array());
            out.opacities.push(0.6);
            // DC only, matching `init_from_sfm_points` — the trainer unlocks
            // higher SH bands progressively.
            out.sh.push((t[0] - 0.5) / SH_C0);
            out.sh.push((t[1] - 0.5) / SH_C0);
            out.sh.push((t[2] - 0.5) / SH_C0);
            spawned += 1;
        }
    }
    (candidates, spawned)
}

/// Dense init: plane-sweep every view and spawn surfels on what it found.
///
/// `points` are the SfM landmarks, used only to centre the depth search range.
/// Returns fewer surfels than the budget by design — [`crate::upsample_to_budget`]
/// tops up whatever the sweep could not resolve.
pub fn init_from_plane_sweep(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    views: &[TrainView],
    points: &[Vec3],
    opts: &SweepOptions,
) -> InitialSurfels {
    let mut out = InitialSurfels {
        positions: Vec::new(),
        scales: Vec::new(),
        quats: Vec::new(),
        opacities: Vec::new(),
        sh: Vec::new(),
        sh_coeffs: 1,
    };
    if views.len() < 2 {
        return out;
    }
    let scene_depth = median_depth(views, points);
    let sweep = PlaneSweep::new(ctx, width, height, views);
    let mut swept = 0usize;
    let mut candidates = 0usize;
    for r in 0..views.len() {
        let nb = pick_neighbours(views, r, opts.neighbours as usize, scene_depth);
        if nb.is_empty() {
            continue;
        }
        let map = sweep.run(ctx, views, r, &nb, scene_depth, opts);
        let (c, s) = surfels_from_depth(&views[r], (width, height), &map, opts, &mut out);
        candidates += c;
        if s > 0 {
            swept += 1;
        }
    }
    // Yield is the number to watch: everything the sweep declines is filled in
    // by random upsampling instead, so a low rate means dense init is barely
    // contributing however good the depth it does produce is.
    log::info!(
        "plane-sweep init: {} surfels from {swept}/{} views, {:.0}% of {candidates} \
         candidates accepted (median scene depth {scene_depth:.2})",
        out.positions.len(),
        views.len(),
        100.0 * out.positions.len() as f64 / candidates.max(1) as f64,
    );
    out
}
