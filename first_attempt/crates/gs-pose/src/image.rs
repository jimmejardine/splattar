//! Grayscale f32 images + pyramids for tracking. Luma comes straight from the
//! decoder's Y plane (tracking doesn't need color); values are kept in [0, 1].

/// Row-major single-channel f32 image.
#[derive(Debug, Clone)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn from_luma8(y: &[u8], width: usize, height: usize) -> Self {
        assert_eq!(y.len(), width * height);
        Self {
            width,
            height,
            data: y.iter().map(|&v| v as f32 / 255.0).collect(),
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Bilinear sample with border clamp; (x, y) in pixel coordinates where
    /// integer coordinates land on pixel centers.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let a = self.get(x0, y0) * (1.0 - fx) + self.get(x1, y0) * fx;
        let b = self.get(x0, y1) * (1.0 - fx) + self.get(x1, y1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Half-resolution downsample: 2×2 box then a light 1-2-1 pre-blur folded
    /// in by sampling the 3×3 neighborhood (standard KLT pyramid kernel).
    /// Deliberately serial: pyramids are built by per-frame prep workers
    /// (cross-frame parallelism), where in-frame rayon would only add
    /// fork/join overhead and pool contention.
    pub fn downsample(&self) -> GrayImage {
        let w = (self.width / 2).max(1);
        let h = (self.height / 2).max(1);
        let mut out = GrayImage::new(w, h);
        out.data.chunks_mut(w).enumerate().for_each(|(oy, row)| {
            for (ox, px) in row.iter_mut().enumerate() {
                let cx = 2 * ox;
                let cy = 2 * oy;
                let xm = cx.saturating_sub(1);
                let xp = (cx + 1).min(self.width - 1);
                let ym = cy.saturating_sub(1);
                let yp = (cy + 1).min(self.height - 1);
                let v = 0.25 * self.get(cx, cy)
                    + 0.125
                        * (self.get(xm, cy)
                            + self.get(xp, cy)
                            + self.get(cx, ym)
                            + self.get(cx, yp))
                    + 0.0625
                        * (self.get(xm, ym)
                            + self.get(xp, ym)
                            + self.get(xm, yp)
                            + self.get(xp, yp));
                *px = v;
            }
        });
        out
    }
}

/// Coarse-to-fine pyramid; `levels[0]` is full resolution.
pub struct Pyramid {
    pub levels: Vec<GrayImage>,
}

impl Pyramid {
    pub fn build(base: GrayImage, n_levels: usize) -> Self {
        let mut levels = Vec::with_capacity(n_levels);
        levels.push(base);
        for k in 1..n_levels {
            let next = levels[k - 1].downsample();
            if next.width < 16 || next.height < 16 {
                break;
            }
            levels.push(next);
        }
        Self { levels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_interpolates() {
        let img = GrayImage {
            width: 2,
            height: 2,
            data: vec![0.0, 1.0, 0.0, 1.0],
        };
        assert!((img.sample(0.5, 0.5) - 0.5).abs() < 1e-6);
        assert!((img.sample(0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((img.sample(1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pyramid_halves() {
        let base = GrayImage::new(640, 480);
        let p = Pyramid::build(base, 4);
        assert_eq!(p.levels.len(), 4);
        assert_eq!(p.levels[3].width, 80);
        assert_eq!(p.levels[3].height, 60);
    }
}
