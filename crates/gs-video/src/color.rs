//! YUV 4:2:0 → RGB conversion. HD phone footage is BT.709 limited range;
//! SD falls back to BT.601. 10-bit wide-gamut HEVC (iPhone HLG) is folded to
//! 8-bit BT.709 at decode time via [`yuv2020_10_to_yuv709_8`] so everything
//! downstream sees ordinary SDR planes.

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

/// One 10-bit BT.2020 limited-range YCbCr pixel → 8-bit BT.709 limited-range
/// YCbCr. Gamut mapping runs on the nonlinear (gamma/HLG-encoded) values —
/// the standard quick-path approximation; HLG's SDR backward compatibility
/// makes the result watchable and, more importantly here, photometrically
/// stable across frames (the training loss only needs consistency). A true
/// scene-linear OOTF tone map is a follow-up.
#[inline]
pub fn px2020_10_to_709_8(y10: u16, cb10: u16, cr10: u16) -> (u8, u8, u8) {
    // Normalize limited-range 10-bit.
    let yn = (y10 as f32 - 64.0) / 876.0;
    let cbn = (cb10 as f32 - 512.0) / 896.0;
    let crn = (cr10 as f32 - 512.0) / 896.0;
    // Y'CbCr → R'G'B' with BT.2020 NCL coefficients.
    let (kr, kb) = (0.2627f32, 0.0593f32);
    let kg = 1.0 - kr - kb;
    let r = yn + 2.0 * (1.0 - kr) * crn;
    let b = yn + 2.0 * (1.0 - kb) * cbn;
    let g = (yn - kr * r - kb * b) / kg;
    // BT.2020 → BT.709 primaries (applied on nonlinear values).
    let r7 = (1.6605 * r - 0.5876 * g - 0.0728 * b).clamp(0.0, 1.0);
    let g7 = (-0.1246 * r + 1.1329 * g - 0.0083 * b).clamp(0.0, 1.0);
    let b7 = (-0.0182 * r - 0.1006 * g + 1.1187 * b).clamp(0.0, 1.0);
    // R'G'B' → Y'CbCr with BT.709 coefficients, 8-bit limited range.
    let (kr, kb) = (0.2126f32, 0.0722f32);
    let y = kr * r7 + (1.0 - kr - kb) * g7 + kb * b7;
    let cb = (b7 - y) / (2.0 * (1.0 - kb));
    let cr = (r7 - y) / (2.0 * (1.0 - kr));
    (
        (16.0 + 219.0 * y).round().clamp(0.0, 255.0) as u8,
        (128.0 + 224.0 * cb).round().clamp(0.0, 255.0) as u8,
        (128.0 + 224.0 * cr).round().clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_grey_survives_gamut_map() {
        // Achromatic pixels are identical in any RGB gamut: Y maps through
        // the range conversion, chroma stays neutral.
        for (y10, expect_y8) in [(64u16, 16u8), (940, 235), (502, 125)] {
            let (y8, cb8, cr8) = px2020_10_to_709_8(y10, 512, 512);
            assert!(
                (y8 as i32 - expect_y8 as i32).abs() <= 1,
                "y10 {y10}: got {y8}, want ~{expect_y8}"
            );
            assert!((cb8 as i32 - 128).abs() <= 1);
            assert!((cr8 as i32 - 128).abs() <= 1);
        }
    }

    #[test]
    fn saturated_2020_red_clamps_into_709() {
        // A pure BT.2020 red is outside 709: expect a valid, strongly red
        // (high Cr) result after clamping.
        // R'G'B'2020 = (1, 0, 0) → Y' = kr, Cb = -kr/..., encode to 10-bit:
        let yn = 0.2627f32;
        let y10 = (64.0 + 876.0 * yn) as u16;
        let cb10 = (512.0 + 896.0 * (-yn / (2.0 * (1.0 - 0.0593)))) as u16;
        let cr10 = (512.0 + 896.0 * ((1.0 - yn) / (2.0 * (1.0 - 0.2627)))) as u16;
        let (_y8, cb8, cr8) = px2020_10_to_709_8(y10, cb10, cr10);
        assert!(cr8 > 200, "expected strong red chroma, got Cr {cr8}");
        assert!(cb8 < 128);
    }
}
