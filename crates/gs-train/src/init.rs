//! Surfel initialization from SfM points (COLMAP points3D): position from the
//! point, color into SH c0, scale from mean 3-NN distance via a voxel hash.

use std::collections::HashMap;

use glam::Vec3;

use crate::trainer::InitialSurfels;

const SH_C0: f32 = 0.282_094_79;

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

    let mean_3nn = |i: usize| -> f32 {
        let p = positions[i];
        let (kx, ky, kz) = key(p);
        let mut d2: Vec<f32> = Vec::with_capacity(32);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ids) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                        for &j in ids {
                            if j as usize != i {
                                d2.push(positions[j as usize].distance_squared(p));
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
        mean.clamp(1e-4 * extent, 0.05 * extent)
    };

    let mut rng = seed | 1;
    let mut rand = move || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (((rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };

    let mut scales = Vec::with_capacity(points.len());
    let mut quats = Vec::with_capacity(points.len());
    let mut sh = Vec::with_capacity(points.len() * 3);
    for (i, (_, rgb)) in points.iter().enumerate() {
        let s = mean_3nn(i);
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
