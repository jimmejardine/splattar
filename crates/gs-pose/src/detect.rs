//! Shi-Tomasi corner detection with grid-bucketed non-max suppression:
//! uniform spatial coverage matters more for VO conditioning than raw corner
//! strength, so the image is divided into cells and the best corner per cell
//! above threshold is kept.

use crate::image::GrayImage;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct Corner {
    pub x: f32,
    pub y: f32,
    /// Shi-Tomasi min-eigenvalue score.
    pub score: f32,
}

pub struct DetectConfig {
    /// Side of the bucketing cell in pixels.
    pub cell: usize,
    /// Half-window for the structure tensor.
    pub half_win: usize,
    /// Minimum min-eigenvalue (on [0,1] intensities).
    pub min_score: f32,
    /// Skip cells that already contain a tracked feature.
    pub min_dist: f32,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            cell: 24,
            half_win: 3,
            min_score: 1e-4,
            min_dist: 12.0,
        }
    }
}

/// Central-difference gradients (Scharr would be marginally better; central
/// differences keep the structure tensor consistent with the KLT window).
fn gradients(img: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (img.width, img.height);
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    // Rows in parallel — pure per-pixel, deterministic.
    gx.par_chunks_mut(w)
        .zip(gy.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (gxr, gyr))| {
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                gxr[x] = 0.5 * (img.get(x + 1, y) - img.get(x - 1, y));
                gyr[x] = 0.5 * (img.get(x, y + 1) - img.get(x, y - 1));
            }
        });
    (gx, gy)
}

/// Detect corners, avoiding positions within `min_dist` of `existing`.
pub fn detect(img: &GrayImage, cfg: &DetectConfig, existing: &[(f32, f32)]) -> Vec<Corner> {
    let (w, h) = (img.width, img.height);
    let (gx, gy) = gradients(img);
    let cells_x = w.div_ceil(cfg.cell);
    let cells_y = h.div_ceil(cfg.cell);
    let mut best: Vec<Option<Corner>> = vec![None; cells_x * cells_y];

    // Mark occupied cells (and their neighbors within min_dist).
    let mut occupied = vec![false; cells_x * cells_y];
    for &(ex, ey) in existing {
        let r = cfg.min_dist;
        let cx0 = (((ex - r).max(0.0) as usize) / cfg.cell).min(cells_x - 1);
        let cx1 = (((ex + r).max(0.0) as usize) / cfg.cell).min(cells_x - 1);
        let cy0 = (((ey - r).max(0.0) as usize) / cfg.cell).min(cells_y - 1);
        let cy1 = (((ey + r).max(0.0) as usize) / cfg.cell).min(cells_y - 1);
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                occupied[cy * cells_x + cx] = true;
            }
        }
    }

    let hw = cfg.half_win;
    let margin = hw + 1;
    if h <= 2 * margin || w <= 2 * margin {
        return Vec::new();
    }
    // One cell row per band, computed in parallel — each cell's best is an
    // ordered scan of its own pixels, so results are thread-count-invariant.
    // Structure-tensor box sums are separable (sliding sums, ~6 adds/px
    // instead of 3·(2hw+1)² multiplies) and are computed lazily, only over
    // contiguous runs of UNOCCUPIED cells — with healthy track coverage most
    // cells are occupied and skip all work.
    best.par_chunks_mut(cells_x).enumerate().for_each(|(cy, row)| {
        let y0 = (cy * cfg.cell).max(margin);
        let y1 = ((cy + 1) * cfg.cell).min(h - margin);
        if y0 >= y1 {
            return;
        }
        let occ_row = &occupied[cy * cells_x..(cy + 1) * cells_x];
        let mut cx0 = 0usize;
        while cx0 < cells_x {
            if occ_row[cx0] {
                cx0 += 1;
                continue;
            }
            let mut cx1 = cx0;
            while cx1 + 1 < cells_x && !occ_row[cx1 + 1] {
                cx1 += 1;
            }
            // Pixel-column span of this unoccupied run.
            let x0 = (cx0 * cfg.cell).max(margin);
            let x1 = ((cx1 + 1) * cfg.cell).min(w - margin);
            if x0 < x1 {
                let span = x1 - x0;
                // Horizontal sliding tensor sums for the run's rows.
                let rows = y1 + hw - (y0 - hw);
                let mut ha = vec![0.0f32; rows * span];
                let mut hb = vec![0.0f32; rows * span];
                let mut hc = vec![0.0f32; rows * span];
                for (ry, wy) in (y0 - hw..y1 + hw).enumerate() {
                    let r = wy * w;
                    let (mut sa, mut sb, mut sc) = (0.0f32, 0.0f32, 0.0f32);
                    for x in x0 - hw..=x0 + hw {
                        let (ix, iy) = (gx[r + x], gy[r + x]);
                        sa += ix * ix;
                        sb += ix * iy;
                        sc += iy * iy;
                    }
                    for i in 0..span {
                        ha[ry * span + i] = sa;
                        hb[ry * span + i] = sb;
                        hc[ry * span + i] = sc;
                        let x = x0 + i;
                        if x + hw + 1 < w {
                            let (ax, ay) = (gx[r + x + hw + 1], gy[r + x + hw + 1]);
                            let (dx, dy) = (gx[r + x - hw], gy[r + x - hw]);
                            sa += ax * ax - dx * dx;
                            sb += ax * ay - dx * dy;
                            sc += ay * ay - dy * dy;
                        }
                    }
                }
                // Vertical sliding sums over the run's columns.
                let mut va = vec![0.0f32; span];
                let mut vb = vec![0.0f32; span];
                let mut vc = vec![0.0f32; span];
                for ry in 0..2 * hw + 1 {
                    for i in 0..span {
                        va[i] += ha[ry * span + i];
                        vb[i] += hb[ry * span + i];
                        vc[i] += hc[ry * span + i];
                    }
                }
                for y in y0..y1 {
                    for i in 0..span {
                        let x = x0 + i;
                        let cx = x / cfg.cell;
                        let (a, b, c) = (va[i], vb[i], vc[i]);
                        // min eigenvalue of [[a,b],[b,c]]
                        let tr = 0.5 * (a + c);
                        let det = a * c - b * b;
                        let disc = (tr * tr - det).max(0.0).sqrt();
                        let lmin = tr - disc;
                        if lmin < cfg.min_score {
                            continue;
                        }
                        if row[cx].is_none_or(|prev| lmin > prev.score) {
                            row[cx] = Some(Corner {
                                x: x as f32,
                                y: y as f32,
                                score: lmin,
                            });
                        }
                    }
                    if y + 1 < y1 {
                        let add = (y + 1 + hw) - (y0 - hw);
                        let sub = (y - hw) - (y0 - hw);
                        for i in 0..span {
                            va[i] += ha[add * span + i] - ha[sub * span + i];
                            vb[i] += hb[add * span + i] - hb[sub * span + i];
                            vc[i] += hc[add * span + i] - hc[sub * span + i];
                        }
                    }
                }
            }
            cx0 = cx1 + 1;
        }
    });
    best.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: usize, h: usize, sq: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.data[y * w + x] = (((x / sq) + (y / sq)) % 2) as f32;
            }
        }
        img
    }

    #[test]
    fn finds_checkerboard_corners() {
        let img = checkerboard(160, 120, 20);
        let corners = detect(&img, &DetectConfig::default(), &[]);
        assert!(corners.len() >= 20, "got {}", corners.len());
        // Every corner should sit near a checker intersection (multiple of 20).
        for c in &corners {
            let dx = (c.x / 20.0).round() * 20.0 - c.x;
            let dy = (c.y / 20.0).round() * 20.0 - c.y;
            // Peak can sit anywhere the 7px tensor window straddles both edges.
            assert!(
                dx.abs() <= 4.0 && dy.abs() <= 4.0,
                "corner off-grid at ({}, {})",
                c.x,
                c.y
            );
        }
    }

    #[test]
    fn flat_image_yields_nothing() {
        let img = GrayImage::new(160, 120);
        assert!(detect(&img, &DetectConfig::default(), &[]).is_empty());
    }

    #[test]
    fn respects_existing_features() {
        let img = checkerboard(160, 120, 20);
        let all = detect(&img, &DetectConfig::default(), &[]);
        let taken: Vec<(f32, f32)> = all.iter().map(|c| (c.x, c.y)).collect();
        let more = detect(&img, &DetectConfig::default(), &taken);
        for m in &more {
            for t in &taken {
                let d = ((m.x - t.0).powi(2) + (m.y - t.1).powi(2)).sqrt();
                assert!(d >= 12.0, "new corner too close to existing");
            }
        }
    }
}
