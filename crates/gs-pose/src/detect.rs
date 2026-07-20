//! Shi-Tomasi corner detection with grid-bucketed non-max suppression:
//! uniform spatial coverage matters more for VO conditioning than raw corner
//! strength, so the image is divided into cells and the best corner per cell
//! above threshold is kept.

use crate::image::GrayImage;

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
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            gx[y * w + x] = 0.5 * (img.get(x + 1, y) - img.get(x - 1, y));
            gy[y * w + x] = 0.5 * (img.get(x, y + 1) - img.get(x, y - 1));
        }
    }
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
    for y in margin..h - margin {
        for x in margin..w - margin {
            let cell_i = (y / cfg.cell) * cells_x + x / cfg.cell;
            if occupied[cell_i] {
                continue;
            }
            // Structure tensor over the window.
            let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
            for wy in y - hw..=y + hw {
                for wx in x - hw..=x + hw {
                    let ix = gx[wy * w + wx];
                    let iy = gy[wy * w + wx];
                    a += ix * ix;
                    b += ix * iy;
                    c += iy * iy;
                }
            }
            // min eigenvalue of [[a,b],[b,c]]
            let tr = 0.5 * (a + c);
            let det = a * c - b * b;
            let disc = (tr * tr - det).max(0.0).sqrt();
            let lmin = tr - disc;
            if lmin < cfg.min_score {
                continue;
            }
            if best[cell_i].is_none_or(|prev| lmin > prev.score) {
                best[cell_i] = Some(Corner {
                    x: x as f32,
                    y: y as f32,
                    score: lmin,
                });
            }
        }
    }
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
