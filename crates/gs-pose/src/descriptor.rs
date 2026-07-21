//! BRIEF-style binary patch descriptor (256 bits) for cross-video landmark
//! matching. Sampled on a smoothed image (pyramid level 1) with a fixed
//! pseudo-random pair pattern. Not rotation-invariant — phone walkthroughs
//! are shot roughly upright, and Sim(3) RANSAC downstream tolerates the
//! resulting false negatives. Revisit (oriented pattern) if roll shows up.

use std::sync::OnceLock;

use crate::image::GrayImage;
use crate::twoview::Rng64;

pub const DESC_BYTES: usize = 32;
pub type Descriptor = [u8; DESC_BYTES];

const PATTERN_RADIUS: f32 = 12.0;

fn pattern() -> &'static [(f32, f32, f32, f32); 256] {
    static P: OnceLock<[(f32, f32, f32, f32); 256]> = OnceLock::new();
    P.get_or_init(|| {
        let mut rng = Rng64::new(0xB51EF);
        let mut r = || {
            // Roughly Gaussian via sum of uniforms, clipped to the radius.
            let u: f64 = (0..4)
                .map(|_| rng.next_u64() as f64 / u64::MAX as f64 - 0.5)
                .sum();
            (u * 0.55 * PATTERN_RADIUS as f64).clamp(
                -PATTERN_RADIUS as f64,
                PATTERN_RADIUS as f64,
            ) as f32
        };
        std::array::from_fn(|_| (r(), r(), r(), r()))
    })
}

/// Descriptor at (x, y) — coordinates in the given image's pixel space.
/// The caller should pass a smoothed image (pyramid level 1 works well) with
/// coordinates scaled accordingly.
pub fn describe(img: &GrayImage, x: f32, y: f32) -> Descriptor {
    let mut d = [0u8; DESC_BYTES];
    for (bit, (ax, ay, bx, by)) in pattern().iter().enumerate() {
        let va = img.sample(x + ax, y + ay);
        let vb = img.sample(x + bx, y + by);
        if va > vb {
            d[bit / 8] |= 1 << (bit % 8);
        }
    }
    d
}

#[inline]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Brute-force match with Lowe-style ratio test on Hamming distances, plus a
/// cross-check: each b-side descriptor keeps only its single best a-side
/// partner. Without the cross-check, repeated indoor texture lets hundreds of
/// a-descriptors pile onto the same few b-landmarks, and a degenerate
/// (scale→0) Sim(3) can "explain" the resulting concentrated cluster.
/// Returns (index in a, index in b) pairs.
pub fn match_descriptors(
    a: &[Descriptor],
    b: &[Descriptor],
    max_dist: u32,
    ratio: f32,
) -> Vec<(usize, usize)> {
    let mut fwd: Vec<(usize, usize, u32)> = Vec::new();
    for (i, da) in a.iter().enumerate() {
        let mut best = (u32::MAX, usize::MAX);
        let mut second = u32::MAX;
        for (j, db) in b.iter().enumerate() {
            let d = hamming(da, db);
            if d < best.0 {
                second = best.0;
                best = (d, j);
            } else if d < second {
                second = d;
            }
        }
        if best.0 <= max_dist && (best.0 as f32) < ratio * second as f32 {
            fwd.push((i, best.1, best.0));
        }
    }
    // Cross-check: keep only the best a per b.
    let mut best_for_b: std::collections::HashMap<usize, (usize, u32)> =
        std::collections::HashMap::new();
    for &(i, j, d) in &fwd {
        match best_for_b.get(&j) {
            Some(&(_, dj)) if dj <= d => {}
            _ => {
                best_for_b.insert(j, (i, d));
            }
        }
    }
    let mut out: Vec<(usize, usize)> = best_for_b
        .into_iter()
        .map(|(j, (i, _))| (i, j))
        .collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Pyramid;

    fn texture(w: usize, h: usize, ox: f32, oy: f32) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f32 + ox, y as f32 + oy);
                img.data[y * w + x] = (0.5
                    + 0.2 * (xf * 0.21).sin() * (yf * 0.13).cos()
                    + 0.15 * (xf * 0.043 + yf * 0.071).sin()
                    + 0.1 * (xf * 0.33).cos() * (yf * 0.27).sin())
                .clamp(0.0, 1.0);
            }
        }
        img
    }

    #[test]
    fn same_point_matches_across_translation() {
        let a = Pyramid::build(texture(256, 256, 0.0, 0.0), 2);
        let b = Pyramid::build(texture(256, 256, -6.0, -3.0), 2);
        // Point (100, 100) in a appears at (106, 103) in b; level-1 coords ÷2.
        let da = describe(&a.levels[1], 50.0, 50.0);
        let db = describe(&b.levels[1], 53.0, 51.5);
        let d_true = hamming(&da, &db);
        let d_false = hamming(&da, &describe(&b.levels[1], 90.0, 20.0));
        assert!(d_true < 40, "true match distance {d_true}");
        assert!(d_false > 80, "false match distance {d_false}");
    }

    #[test]
    fn ratio_matching_finds_correspondences() {
        let a_img = Pyramid::build(texture(256, 256, 0.0, 0.0), 2);
        let b_img = Pyramid::build(texture(256, 256, -8.0, 5.0), 2);
        let pts: Vec<(f32, f32)> = (0..20)
            .map(|k| (20.0 + 4.5 * k as f32, 25.0 + 4.0 * k as f32))
            .collect();
        let da: Vec<Descriptor> = pts
            .iter()
            .map(|p| describe(&a_img.levels[1], p.0, p.1))
            .collect();
        let db: Vec<Descriptor> = pts
            .iter()
            .map(|p| describe(&b_img.levels[1], p.0 + 4.0, p.1 - 2.5))
            .collect();
        let matches = match_descriptors(&da, &db, 60, 0.8);
        let correct = matches.iter().filter(|(i, j)| i == j).count();
        assert!(
            correct >= 14 && correct >= matches.len() * 4 / 5,
            "correct {correct} of {} matches",
            matches.len()
        );
    }
}
