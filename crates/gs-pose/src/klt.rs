//! Pyramidal Lucas–Kanade tracking (translation model, iterative, with
//! forward–backward verification). Matching is zero-mean (window means
//! subtracted) so the tracker shrugs off the frame-to-frame exposure drift
//! phone cameras produce. Tracks survive only if the backward track returns
//! to within `fb_tol` of the start — the standard cheap outlier gate before
//! any geometric verification.

use crate::image::Pyramid;

pub struct KltConfig {
    /// Half-window (window is (2h+1)²).
    pub half_win: usize,
    pub max_iters: usize,
    /// Convergence: update norm below this stops iterating (pixels).
    pub eps: f32,
    /// Forward-backward round-trip tolerance in pixels (finest level).
    pub fb_tol: f32,
    /// Minimum structure-tensor determinant to attempt a solve.
    pub min_det: f32,
}

impl Default for KltConfig {
    fn default() -> Self {
        Self {
            half_win: 7,
            max_iters: 30,
            eps: 0.01,
            fb_tol: 0.8,
            min_det: 1e-7,
        }
    }
}

/// Track one point from `prev` to `next`, starting from `guess` (pixel coords
/// in the finest level; pass the previous position when there is no motion
/// prediction). Returns the refined position in `next`.
pub fn track_point(
    prev: &Pyramid,
    next: &Pyramid,
    start: (f32, f32),
    guess: (f32, f32),
    cfg: &KltConfig,
) -> Option<(f32, f32)> {
    let n = prev.levels.len().min(next.levels.len());
    let scale = (1 << (n - 1)) as f32;
    // Displacement carried coarse-to-fine.
    let mut dx = (guess.0 - start.0) / scale;
    let mut dy = (guess.1 - start.1) / scale;

    for lvl in (0..n).rev() {
        let s = (1 << lvl) as f32;
        let px = start.0 / s;
        let py = start.1 / s;
        let (img_p, img_n) = (&prev.levels[lvl], &next.levels[lvl]);
        let hw = cfg.half_win as isize;

        // Template values + gradients at the (fixed) prev position.
        let win = (2 * hw + 1) * (2 * hw + 1);
        let mut tmpl = Vec::with_capacity(win as usize);
        let mut gxs = Vec::with_capacity(win as usize);
        let mut gys = Vec::with_capacity(win as usize);
        let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
        for wy in -hw..=hw {
            for wx in -hw..=hw {
                let sx = px + wx as f32;
                let sy = py + wy as f32;
                let v = img_p.sample(sx, sy);
                let gx = 0.5 * (img_p.sample(sx + 1.0, sy) - img_p.sample(sx - 1.0, sy));
                let gy = 0.5 * (img_p.sample(sx, sy + 1.0) - img_p.sample(sx, sy - 1.0));
                tmpl.push(v);
                gxs.push(gx);
                gys.push(gy);
                a += gx * gx;
                b += gx * gy;
                c += gy * gy;
            }
        }
        let det = a * c - b * b;
        if det < cfg.min_det {
            return None;
        }
        let (ia, ib, ic) = (c / det, -b / det, a / det);

        let tmpl_mean: f32 = tmpl.iter().sum::<f32>() / tmpl.len() as f32;
        let mut cur = vec![0.0f32; tmpl.len()];
        for _ in 0..cfg.max_iters {
            let mut k = 0;
            let mut cur_mean = 0.0f32;
            for wy in -hw..=hw {
                for wx in -hw..=hw {
                    let v = img_n.sample(px + dx + wx as f32, py + dy + wy as f32);
                    cur[k] = v;
                    cur_mean += v;
                    k += 1;
                }
            }
            cur_mean /= cur.len() as f32;
            // Zero-mean residual: cancels uniform exposure offsets.
            let bias = cur_mean - tmpl_mean;
            let (mut bx, mut by) = (0.0f32, 0.0f32);
            for k in 0..cur.len() {
                let diff = cur[k] - tmpl[k] - bias;
                bx += diff * gxs[k];
                by += diff * gys[k];
            }
            let ux = -(ia * bx + ib * by);
            let uy = -(ib * bx + ic * by);
            dx += ux;
            dy += uy;
            if ux * ux + uy * uy < cfg.eps * cfg.eps {
                break;
            }
        }

        if lvl > 0 {
            dx *= 2.0;
            dy *= 2.0;
        }
    }

    let out = (start.0 + dx, start.1 + dy);
    let (w, h) = (prev.levels[0].width as f32, prev.levels[0].height as f32);
    if out.0 < 1.0 || out.1 < 1.0 || out.0 > w - 2.0 || out.1 > h - 2.0 {
        return None;
    }
    Some(out)
}

/// Forward-backward verified track: prev→next, then next→prev must land
/// within `fb_tol` of the original point.
pub fn track_point_fb(
    prev: &Pyramid,
    next: &Pyramid,
    start: (f32, f32),
    guess: (f32, f32),
    cfg: &KltConfig,
) -> Option<(f32, f32)> {
    let fwd = track_point(prev, next, start, guess, cfg)?;
    let back = track_point(next, prev, fwd, start, cfg)?;
    let d2 = (back.0 - start.0).powi(2) + (back.1 - start.1).powi(2);
    (d2 <= cfg.fb_tol * cfg.fb_tol).then_some(fwd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::{DetectConfig, detect};
    use crate::image::{GrayImage, Pyramid};

    /// Smooth random texture: sum of a few sinusoids (trackable everywhere).
    fn texture(w: usize, h: usize, ox: f32, oy: f32) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f32 + ox, y as f32 + oy);
                let v = 0.5
                    + 0.15 * (xf * 0.18).sin() * (yf * 0.11).cos()
                    + 0.12 * (xf * 0.05 + yf * 0.07).sin()
                    + 0.1 * (xf * 0.31).cos() * (yf * 0.23).sin()
                    + 0.08 * (xf * 0.021 - yf * 0.013).cos();
                img.data[y * w + x] = v.clamp(0.0, 1.0);
            }
        }
        img
    }

    #[test]
    fn tracks_pure_translation() {
        let (dx, dy) = (7.3f32, -4.6f32);
        // next(x) = prev(x - d)  ⇒ features move by +d.
        let prev = Pyramid::build(texture(320, 240, 0.0, 0.0), 3);
        let next = Pyramid::build(texture(320, 240, -dx, -dy), 3);
        let corners = detect(&prev.levels[0], &DetectConfig::default(), &[]);
        assert!(corners.len() > 30);
        let cfg = KltConfig::default();
        let mut ok = 0;
        for c in &corners {
            if c.x < 20.0 || c.y < 20.0 || c.x > 300.0 || c.y > 220.0 {
                continue;
            }
            if let Some((tx, ty)) = track_point_fb(&prev, &next, (c.x, c.y), (c.x, c.y), &cfg)
            {
                let ex = (tx - c.x - dx).abs();
                let ey = (ty - c.y - dy).abs();
                assert!(ex < 0.35 && ey < 0.35, "track error ({ex:.3},{ey:.3})");
                ok += 1;
            }
        }
        assert!(ok > 25, "too few surviving tracks: {ok}");
    }

    #[test]
    fn survives_brightness_shift_partially() {
        // KLT translation-only has no gain/bias model; a small offset should
        // still converge because gradients dominate.
        let prev = Pyramid::build(texture(320, 240, 0.0, 0.0), 3);
        let mut shifted = texture(320, 240, -3.0, -2.0);
        for v in &mut shifted.data {
            *v = (*v + 0.03).clamp(0.0, 1.0);
        }
        let next = Pyramid::build(shifted, 3);
        let corners = detect(&prev.levels[0], &DetectConfig::default(), &[]);
        let cfg = KltConfig::default();
        let mut ok = 0;
        for c in corners.iter().take(60) {
            if c.x < 20.0 || c.y < 20.0 || c.x > 300.0 || c.y > 220.0 {
                continue;
            }
            if let Some((tx, ty)) = track_point_fb(&prev, &next, (c.x, c.y), (c.x, c.y), &cfg)
            {
                if (tx - c.x - 3.0).abs() < 0.5 && (ty - c.y - 2.0).abs() < 0.5 {
                    ok += 1;
                }
            }
        }
        assert!(ok > 15, "too few tracks under brightness shift: {ok}");
    }

    #[test]
    fn rejects_untrackable_flat_region() {
        let prev = Pyramid::build(GrayImage::new(160, 120), 3);
        let next = Pyramid::build(GrayImage::new(160, 120), 3);
        assert!(
            track_point_fb(&prev, &next, (80.0, 60.0), (80.0, 60.0), &KltConfig::default())
                .is_none()
        );
    }
}
