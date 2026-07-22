//! Sharpness scoring: variance of a 3×3 Laplacian over the luma plane,
//! computed on a subsampled grid (every 2nd pixel) — plenty for ranking
//! frames within a keyframe window.

pub fn laplacian_variance(y: &[u8], width: usize, height: usize) -> f64 {
    if width < 4 || height < 4 {
        return 0.0;
    }
    let mut sum = 0f64;
    let mut sum2 = 0f64;
    let mut count = 0u64;
    let mut py = 2;
    while py < height - 2 {
        let mut px = 2;
        while px < width - 2 {
            let c = y[py * width + px] as f64;
            let lap = 4.0 * c
                - y[py * width + px - 1] as f64
                - y[py * width + px + 1] as f64
                - y[(py - 1) * width + px] as f64
                - y[(py + 1) * width + px] as f64;
            sum += lap;
            sum2 += lap * lap;
            count += 1;
            px += 2;
        }
        py += 2;
    }
    let mean = sum / count as f64;
    sum2 / count as f64 - mean * mean
}
