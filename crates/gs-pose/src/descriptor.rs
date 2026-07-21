//! Viewpoint-robust binary patch descriptors for cross-video / cross-segment
//! landmark matching (the "AKAZE-class upgrade"): steered BRIEF (256 bits)
//! with intensity-centroid orientation (ORB-style) computed at THREE pyramid
//! levels per observation. Matching takes the minimum Hamming distance over
//! nearby level offsets — a scale search that covers roughly a 4× revisit-
//! distance range. Detection stays KLT corners; robustness lives in the
//! description, which is what cross-view matching actually exercises.
//!
//! The old non-oriented single-scale variant measurably failed across real
//! viewpoint change (whip-pan cuts, second walkthroughs) — see RESULTS.md.

use std::sync::OnceLock;

use crate::image::{GrayImage, Pyramid};
use crate::twoview::Rng64;

pub const DESC_BYTES: usize = 32;
pub type Descriptor = [u8; DESC_BYTES];

/// Pyramid levels a multi-descriptor is computed at (indices into
/// `Pyramid::levels`, clamped to what exists).
pub const DESC_LEVELS: usize = 3;
const FIRST_LEVEL: usize = 1;

/// One descriptor per pyramid level, coarse scale search at match time.
pub type MultiDescriptor = [Descriptor; DESC_LEVELS];

const PATTERN_RADIUS: f32 = 12.0;
/// Radius of the intensity-centroid orientation patch.
const ORI_RADIUS: i32 = 12;

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

/// Intensity-centroid orientation (radians) of the circular patch at (x, y):
/// the angle of the vector from patch center to its brightness centroid.
/// Stable under viewpoint change; steering the pattern by it makes the
/// descriptor rotation-invariant.
pub fn orientation(img: &GrayImage, x: f32, y: f32) -> f32 {
    let (mut m10, mut m01) = (0.0f32, 0.0f32);
    for dy in -ORI_RADIUS..=ORI_RADIUS {
        for dx in -ORI_RADIUS..=ORI_RADIUS {
            if dx * dx + dy * dy > ORI_RADIUS * ORI_RADIUS {
                continue;
            }
            let v = img.sample(x + dx as f32, y + dy as f32);
            m10 += dx as f32 * v;
            m01 += dy as f32 * v;
        }
    }
    m01.atan2(m10)
}

/// Steered BRIEF at (x, y): the sampling pattern is rotated by `angle`.
pub fn describe_oriented(img: &GrayImage, x: f32, y: f32, angle: f32) -> Descriptor {
    let (s, c) = angle.sin_cos();
    let mut d = [0u8; DESC_BYTES];
    for (bit, (ax, ay, bx, by)) in pattern().iter().enumerate() {
        let (rax, ray) = (c * ax - s * ay, s * ax + c * ay);
        let (rbx, rby) = (c * bx - s * by, s * bx + c * by);
        let va = img.sample(x + rax, y + ray);
        let vb = img.sample(x + rbx, y + rby);
        if va > vb {
            d[bit / 8] |= 1 << (bit % 8);
        }
    }
    d
}

/// Backwards-compatible single-level descriptor (level-1 oriented).
pub fn describe(img: &GrayImage, x: f32, y: f32) -> Descriptor {
    let angle = orientation(img, x, y);
    describe_oriented(img, x, y, angle)
}

/// Multi-scale oriented descriptor for a point given in LEVEL-0 pixel
/// coordinates: one steered BRIEF per pyramid level (orientation recomputed
/// per level — coarse levels see coarser structure).
pub fn describe_multi(pyr: &Pyramid, x0: f32, y0: f32) -> MultiDescriptor {
    std::array::from_fn(|k| {
        let lvl = (FIRST_LEVEL + k).min(pyr.levels.len() - 1);
        let s = 1.0 / (1 << lvl) as f32;
        let img = &pyr.levels[lvl];
        let (x, y) = (x0 * s, y0 * s);
        let angle = orientation(img, x, y);
        describe_oriented(img, x, y, angle)
    })
}

#[inline]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Scale-searching distance: minimum Hamming over level offsets −1, 0, +1
/// (each side's level ladder shifted against the other's).
#[inline]
pub fn hamming_multi(a: &MultiDescriptor, b: &MultiDescriptor) -> u32 {
    let mut best = u32::MAX;
    for off in -1i32..=1 {
        for ia in 0..DESC_LEVELS as i32 {
            let ib = ia + off;
            if !(0..DESC_LEVELS as i32).contains(&ib) {
                continue;
            }
            best = best.min(hamming(&a[ia as usize], &b[ib as usize]));
        }
    }
    best
}

/// Brute-force multi-descriptor match with Lowe-style ratio test and a
/// cross-check (best a per b) — same guards as before, scale-searching
/// distance underneath.
pub fn match_descriptors(
    a: &[MultiDescriptor],
    b: &[MultiDescriptor],
    max_dist: u32,
    ratio: f32,
) -> Vec<(usize, usize)> {
    let mut fwd: Vec<(usize, usize, u32)> = Vec::new();
    for (i, da) in a.iter().enumerate() {
        let mut best = (u32::MAX, usize::MAX);
        let mut second = u32::MAX;
        for (j, db) in b.iter().enumerate() {
            let d = hamming_multi(da, db);
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

    /// Rotate an image by `angle` about (cx, cy) with bilinear resampling.
    fn rotate(img: &GrayImage, angle: f32, cx: f32, cy: f32) -> GrayImage {
        let mut out = GrayImage::new(img.width, img.height);
        let (s, c) = angle.sin_cos();
        for y in 0..img.height {
            for x in 0..img.width {
                // Inverse map.
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let sx = c * dx + s * dy + cx;
                let sy = -s * dx + c * dy + cy;
                out.data[y * img.width + x] = img.sample(sx, sy);
            }
        }
        out
    }

    /// Downscale by an arbitrary factor with bilinear sampling.
    fn downscale(img: &GrayImage, f: f32) -> GrayImage {
        let w = (img.width as f32 / f) as usize;
        let h = (img.height as f32 / f) as usize;
        let mut out = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                out.data[y * w + x] = img.sample(x as f32 * f, y as f32 * f);
            }
        }
        out
    }

    #[test]
    fn same_point_matches_across_translation() {
        let a = Pyramid::build(texture(256, 256, 0.0, 0.0), 4);
        let b = Pyramid::build(texture(256, 256, -6.0, -3.0), 4);
        let da = describe_multi(&a, 100.0, 100.0);
        let db = describe_multi(&b, 106.0, 103.0);
        let d_true = hamming_multi(&da, &db);
        let d_false = hamming_multi(&da, &describe_multi(&b, 180.0, 40.0));
        assert!(d_true < 40, "true match distance {d_true}");
        assert!(d_false > 60, "false match distance {d_false}");
    }

    #[test]
    fn survives_in_plane_rotation() {
        // 25° roll — far beyond handheld wobble; non-steered BRIEF dies here.
        let base = texture(256, 256, 0.0, 0.0);
        let rot = rotate(&base, 25f32.to_radians(), 128.0, 128.0);
        let a = Pyramid::build(base, 4);
        let b = Pyramid::build(rot, 4);
        // Points near the center barely move under rotation about the center.
        let mut good = 0;
        for k in 0..12 {
            let (x, y) = (100.0 + 5.0 * k as f32, 118.0 + 2.0 * k as f32);
            // Forward-rotate the point to its position in the rotated image.
            let (s, c) = 25f32.to_radians().sin_cos();
            let (dx, dy) = (x - 128.0, y - 128.0);
            let (rx, ry) = (c * dx - s * dy + 128.0, s * dx + c * dy + 128.0);
            let d = hamming_multi(
                &describe_multi(&a, x, y),
                &describe_multi(&b, rx, ry),
            );
            if d < 55 {
                good += 1;
            }
        }
        assert!(good >= 9, "only {good}/12 survived 25° rotation");
    }

    #[test]
    fn survives_scale_change() {
        // 1.7× closer revisit: the level ladder + offset search absorbs it.
        let base = texture(384, 384, 0.0, 0.0);
        let small = downscale(&base, 1.7);
        let a = Pyramid::build(base, 4);
        let b = Pyramid::build(small, 4);
        let mut good = 0;
        for k in 0..12 {
            let (x, y) = (120.0 + 12.0 * k as f32, 130.0 + 9.0 * k as f32);
            let d = hamming_multi(
                &describe_multi(&a, x, y),
                &describe_multi(&b, x / 1.7, y / 1.7),
            );
            if d < 60 {
                good += 1;
            }
        }
        assert!(good >= 8, "only {good}/12 survived 1.7x scale");
    }

    #[test]
    fn ratio_matching_finds_correspondences() {
        let a_img = Pyramid::build(texture(256, 256, 0.0, 0.0), 4);
        let b_img = Pyramid::build(texture(256, 256, -8.0, 5.0), 4);
        let pts: Vec<(f32, f32)> = (0..20)
            .map(|k| (40.0 + 9.0 * k as f32, 50.0 + 8.0 * k as f32))
            .collect();
        let da: Vec<MultiDescriptor> = pts
            .iter()
            .map(|p| describe_multi(&a_img, p.0, p.1))
            .collect();
        let db: Vec<MultiDescriptor> = pts
            .iter()
            .map(|p| describe_multi(&b_img, p.0 + 8.0, p.1 - 5.0))
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
