//! Plane-sweep correctness against a scene whose depth is known exactly.
//!
//! The gradient-check rule in CLAUDE.md covers differentiable kernels; the
//! sweep is forward-only, so the oracle here is ground truth rather than a CPU
//! reimplementation. That is deliberately the stronger test: a CPU port of the
//! same projection math would reproduce a convention error rather than catch
//! it, and the projection convention is exactly what is most likely to be
//! wrong (it has to match surfel_prep_fwd.wgsl, not merely be self-consistent).

use glam::{Quat, Vec3};
use gs_kernels::RasterCamera;
use gs_train::{SweepOptions, TrainView, plane_sweep::PlaneSweep};
use gs_wgpu::GpuContext;

const SIZE: u32 = 64;
const FOCAL: f32 = 64.0;
/// The answer the sweep has to find.
const PLANE_DEPTH: f32 = 3.0;

fn context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(wgpu::Backends::all())) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPING GPU plane-sweep test: {e}");
            None
        }
    }
}

/// Smooth, non-repetitive value noise. Repetitive texture would be matched at
/// many depths and correctly rejected by the confidence gate, which would make
/// this test vacuous.
///
/// Two octaves at a deliberately non-harmonic ratio: smoothstep has zero
/// gradient at every lattice boundary, so a single octave leaves a regular
/// grid of untexturable patches that the sweep rightly refuses to match. One
/// octave measured 61% coverage — a property of the test image, not the
/// kernel. The second octave fills those dead zones.
fn noise(x: f32, y: f32) -> f32 {
    0.6 * octave(x, y, 4.0, 0.0, 0.0) + 0.4 * octave(x, y, 9.7, 0.37, 0.71)
}

fn octave(x: f32, y: f32, scale: f32, ox: f32, oy: f32) -> f32 {
    let hash = |i: i32, j: i32| -> f32 {
        let mut h = (i as u32).wrapping_mul(0x9E37_79B9) ^ (j as u32).wrapping_mul(0x85EB_CA6B);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2545_F491);
        h ^= h >> 13;
        (h >> 8) as f32 / (1u32 << 24) as f32
    };
    let (sx, sy) = ((x + ox) * scale, (y + oy) * scale);
    let (i, j) = (sx.floor() as i32, sy.floor() as i32);
    let (fx, fy) = (sx - i as f32, sy - j as f32);
    // Smoothstep so the field is C1 — bilinear alone has creases the NCC
    // patch can lock onto.
    let (ux, uy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let a = hash(i, j);
    let b = hash(i + 1, j);
    let c = hash(i, j + 1);
    let d = hash(i + 1, j + 1);
    (a * (1.0 - ux) + b * ux) * (1.0 - uy) + (c * (1.0 - ux) + d * ux) * uy
}

/// Render a camera's view of a fronto-parallel textured plane at world
/// z = -PLANE_DEPTH. Every pixel of a camera with identity rotation sees the
/// plane at exactly PLANE_DEPTH, which is what the sweep must recover.
fn render_plane(center: Vec3) -> Vec<[f32; 4]> {
    let (cx, cy) = (SIZE as f32 * 0.5, SIZE as f32 * 0.5);
    let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
    for py in 0..SIZE {
        for px in 0..SIZE {
            let d = PLANE_DEPTH;
            let wx = (px as f32 + 0.5 - cx) * d / FOCAL + center.x;
            let wy = -(py as f32 + 0.5 - cy) * d / FOCAL + center.y;
            let v = noise(wx, wy);
            out.push([v, v, v, 1.0]);
        }
    }
    out
}

fn view_at(center: Vec3) -> TrainView {
    TrainView {
        camera: RasterCamera {
            center,
            quat: Quat::IDENTITY,
            focal: FOCAL,
            sh_degree: 0,
        },
        target: render_plane(center),
    }
}

#[test]
fn sweep_recovers_a_known_plane_depth() {
    let Some(ctx) = context() else { return };

    // Reference at the origin; neighbours offset sideways and up. Baseline
    // 0.3 at depth 3 is ~5.7 deg of parallax — enough to constrain depth,
    // little enough that the fronto-parallel patch assumption holds.
    let views = vec![
        view_at(Vec3::ZERO),
        view_at(Vec3::new(0.3, 0.0, 0.0)),
        view_at(Vec3::new(-0.3, 0.0, 0.0)),
        view_at(Vec3::new(0.0, 0.3, 0.0)),
    ];
    let opts = SweepOptions {
        downscale: 1,
        hypotheses: 96,
        neighbours: 3,
        patch: 2,
        ..Default::default()
    };
    let sweep = PlaneSweep::new(&ctx, SIZE, SIZE, &views);
    let map = sweep.run(&ctx, &views, 0, &[1, 2, 3], PLANE_DEPTH, &opts);

    // Interior only: patch sampling clamps at the border, so edge pixels see a
    // distorted window.
    let margin = 6;
    let mut errs: Vec<f32> = Vec::new();
    let mut accepted = 0usize;
    let mut interior = 0usize;
    for y in margin..map.height - margin {
        for x in margin..map.width - margin {
            interior += 1;
            let i = y * map.width + x;
            if map.depth[i] <= 0.0 || map.confidence[i] < opts.min_confidence {
                continue;
            }
            accepted += 1;
            errs.push((map.depth[i] - PLANE_DEPTH).abs() / PLANE_DEPTH);
        }
    }
    assert!(!errs.is_empty(), "sweep accepted no interior pixels at all");
    errs.sort_by(f32::total_cmp);
    let median = errs[errs.len() / 2];
    let p90 = errs[errs.len() * 9 / 10];
    let coverage = accepted as f64 / interior as f64;
    eprintln!(
        "plane sweep: {accepted}/{interior} accepted ({:.0}%), \
         relative depth error median {:.4} / p90 {:.4}",
        coverage * 100.0,
        median,
        p90
    );

    // A textured plane at a well-conditioned baseline should be recovered
    // nearly everywhere and accurately. These bounds are loose enough for the
    // hypothesis quantisation (96 steps over 0.45..12) and tight enough that a
    // projection-convention error cannot pass.
    assert!(
        coverage > 0.7,
        "sweep only accepted {:.0}% of a well-textured plane",
        coverage * 100.0
    );
    assert!(median < 0.02, "median depth error {median:.4} (want < 2%)");
    assert!(p90 < 0.05, "p90 depth error {p90:.4} (want < 5%)");
}

/// A neighbour that only rotated carries no depth information. Selecting by
/// frame proximity instead of baseline walks into exactly the trap the VO
/// bootstrap hit: panning produces huge flow and zero parallax.
#[test]
fn neighbour_selection_rejects_zero_baseline_views() {
    let mut rotated = view_at(Vec3::ZERO);
    rotated.camera.quat = Quat::from_rotation_y(0.15);
    let views = vec![
        view_at(Vec3::ZERO),
        rotated,                             // co-located, rotated: useless
        view_at(Vec3::new(0.3, 0.0, 0.0)),   // real baseline
        view_at(Vec3::new(0.001, 0.0, 0.0)), // baseline far below the gate
    ];
    let picked = gs_train::plane_sweep::pick_neighbours_for_test(&views, 0, 4, PLANE_DEPTH);
    assert!(
        picked.contains(&2),
        "dropped the only usable neighbour: {picked:?}"
    );
    assert!(
        !picked.contains(&1) && !picked.contains(&3),
        "kept a zero-baseline neighbour: {picked:?}"
    );
}
