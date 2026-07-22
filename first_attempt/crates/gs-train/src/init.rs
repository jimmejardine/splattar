//! Surfel initialization from SfM points (COLMAP points3D): position from the
//! point, color into SH c0, scale from mean 3-NN distance via a voxel hash.

use std::collections::HashMap;

use glam::Vec3;

use crate::trainer::InitialSurfels;

const SH_C0: f32 = 0.282_094_79;

/// Grow an SfM init to a fixed MCMC budget by duplicating random surfels with
/// position jitter proportional to their scale, at low opacity (relocation
/// and the optimizer put them to work). No-op if already at/over budget.
pub fn upsample_to_budget(init: &mut InitialSurfels, budget: usize, seed: u64) {
    let base = init.positions.len();
    if base >= budget {
        return;
    }
    let mut rng = seed | 1;
    let mut rand = move || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (((rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let per = init.sh_coeffs * 3;
    for _ in base..budget {
        let src = ((rand().abs() * base as f32) as usize).min(base - 1);
        let s = init.scales[src];
        let jitter = Vec3::new(rand(), rand(), rand()) * (s[0] + s[1]);
        init.positions.push(init.positions[src] + jitter);
        init.scales.push(s);
        init.quats.push(init.quats[src]);
        init.opacities.push(0.05);
        let sh: Vec<f32> = init.sh[src * per..(src + 1) * per].to_vec();
        init.sh.extend(sh);
    }
}

pub fn init_from_sfm_points(points: &[([f32; 3], [u8; 3])], seed: u64) -> InitialSurfels {
    assert!(!points.is_empty());
    let positions: Vec<Vec3> = points.iter().map(|(p, _)| Vec3::from_array(*p)).collect();

    // Voxel hash sized for ~8 points per cell on average.
    let (mut lo, mut hi) = (positions[0], positions[0]);
    for p in &positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let extent = (hi - lo).max_element().max(1e-3);
    let cell = (extent / (positions.len() as f32).cbrt().max(1.0)) * 2.0;
    let key = |p: Vec3| -> (i32, i32, i32) {
        (
            ((p.x - lo.x) / cell) as i32,
            ((p.y - lo.y) / cell) as i32,
            ((p.z - lo.z) / cell) as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (i, p) in positions.iter().enumerate() {
        grid.entry(key(*p)).or_default().push(i as u32);
    }

    // Cap the per-cell scan: SfM clouds have dense clusters (thousands of
    // points per cell in outlier-stretched scenes) and full scans turn this
    // O(n²) — 64 candidates approximate the 3-NN plenty well.
    const MAX_CANDIDATES: usize = 64;
    let mean_3nn = |i: usize| -> f32 {
        let p = positions[i];
        let (kx, ky, kz) = key(p);
        let mut d2: Vec<f32> = Vec::with_capacity(MAX_CANDIDATES);
        'outer: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ids) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                        for &j in ids {
                            if j as usize != i {
                                d2.push(positions[j as usize].distance_squared(p));
                                if d2.len() >= MAX_CANDIDATES {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
        if d2.len() < 3 {
            return cell * 0.5;
        }
        d2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f32 = d2[..3].iter().map(|d| d.sqrt()).sum::<f32>() / 3.0;
        mean.max(1e-4 * extent)
    };

    let mut rng = seed | 1;
    let mut rand = move || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (((rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };

    // Robust cap: 3× the median 3-NN spacing. Outlier-stretched SfM clouds
    // (sky points kilometers out) otherwise produce screen-flooding splats.
    let mut raw_scales: Vec<f32> = (0..points.len()).map(&mean_3nn).collect();
    let mut sorted = raw_scales.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cap = sorted[sorted.len() / 2] * 3.0;
    for s in &mut raw_scales {
        *s = s.min(cap);
    }

    let mut scales = Vec::with_capacity(points.len());
    let mut quats = Vec::with_capacity(points.len());
    let mut sh = Vec::with_capacity(points.len() * 3);
    for (i, (_, rgb)) in points.iter().enumerate() {
        let s = raw_scales[i];
        scales.push([s, s]);
        quats.push([rand() * 0.1, rand() * 0.1, rand() * 0.1, 1.0]);
        for &c in rgb {
            sh.push((c as f32 / 255.0 - 0.5) / SH_C0);
        }
    }
    InitialSurfels {
        positions,
        scales,
        quats,
        opacities: vec![0.1; points.len()],
        sh,
        sh_coeffs: 1,
    }
}
