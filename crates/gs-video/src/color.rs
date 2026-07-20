//! YUV 4:2:0 → RGB conversion. HD phone footage is BT.709 limited range;
//! SD falls back to BT.601. (HLG/10-bit tone mapping arrives with HEVC.)

/// Convert planar 4:2:0 to interleaved rgba-f32 (a = 1). Picks BT.709 for
/// heights ≥ 720, BT.601 otherwise.
pub fn yuv420_to_rgba_f32(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> Vec<[f32; 4]> {
    let bt709 = height >= 720;
    // Limited-range coefficients.
    let (kr, kb) = if bt709 { (0.2126, 0.0722) } else { (0.299, 0.114) };
    let kg = 1.0 - kr - kb;
    let cw = width.div_ceil(2);

    let mut out = Vec::with_capacity(width * height);
    for py in 0..height {
        for px in 0..width {
            let yy = (y[py * width + px] as f32 - 16.0) * (255.0 / 219.0);
            let ci = (py / 2) * cw + px / 2;
            let cb = (u[ci] as f32 - 128.0) * (255.0 / 224.0);
            let cr = (v[ci] as f32 - 128.0) * (255.0 / 224.0);
            let r = yy + 2.0 * (1.0 - kr) * cr;
            let b = yy + 2.0 * (1.0 - kb) * cb;
            let g = (yy - kr * r - kb * b) / kg;
            out.push([
                (r / 255.0).clamp(0.0, 1.0),
                (g / 255.0).clamp(0.0, 1.0),
                (b / 255.0).clamp(0.0, 1.0),
                1.0,
            ]);
        }
    }
    out
}
