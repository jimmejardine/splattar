//! Real spherical-harmonics basis evaluation (degrees 0–3), matching the 3DGS
//! convention: view-dependent color = clamp(0.5 + Σ basisᵢ(dir)·coeffᵢ, ≥0).

use glam::Vec3;

pub const SH_C0: f32 = 0.282_094_79;
pub const SH_C1: f32 = 0.488_602_51;
pub const SH_C2: [f32; 5] = [1.092_548_4, -1.092_548_4, 0.315_391_57, -1.092_548_4, 0.546_274_2];
pub const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_33,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// Coefficients per color channel for a given SH degree: (deg+1)².
pub fn num_coeffs(degree: u8) -> usize {
    (degree as usize + 1) * (degree as usize + 1)
}

/// Infer SH degree from the number of `f_rest_*` properties per channel in an
/// INRIA .ply (0 → deg 0, 9 → deg 1... wait, rest counts are per ALL channels).
/// `rest_count` is the total count of f_rest_* properties (all channels):
/// 0 → deg 0, 9 → deg 1, 24 → deg 2, 45 → deg 3.
pub fn degree_from_rest_count(rest_count: usize) -> Option<u8> {
    match rest_count {
        0 => Some(0),
        9 => Some(1),
        24 => Some(2),
        45 => Some(3),
        _ => None,
    }
}

/// Evaluate the SH basis dot coefficients for a view direction (must be
/// normalized). `coeffs` is coefficient-major (one Vec3 rgb per coefficient),
/// length ≥ num_coeffs(degree). Returns the raw sum (no +0.5 offset).
pub fn eval_sh(degree: u8, coeffs: &[Vec3], dir: Vec3) -> Vec3 {
    debug_assert!(coeffs.len() >= num_coeffs(degree));
    let mut result = SH_C0 * coeffs[0];
    if degree == 0 {
        return result;
    }

    let (x, y, z) = (dir.x, dir.y, dir.z);
    result += -SH_C1 * y * coeffs[1] + SH_C1 * z * coeffs[2] - SH_C1 * x * coeffs[3];
    if degree == 1 {
        return result;
    }

    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, yz, xz) = (x * y, y * z, x * z);
    result += SH_C2[0] * xy * coeffs[4]
        + SH_C2[1] * yz * coeffs[5]
        + SH_C2[2] * (2.0 * zz - xx - yy) * coeffs[6]
        + SH_C2[3] * xz * coeffs[7]
        + SH_C2[4] * (xx - yy) * coeffs[8];
    if degree == 2 {
        return result;
    }

    result += SH_C3[0] * y * (3.0 * xx - yy) * coeffs[9]
        + SH_C3[1] * xy * z * coeffs[10]
        + SH_C3[2] * y * (4.0 * zz - xx - yy) * coeffs[11]
        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffs[12]
        + SH_C3[4] * x * (4.0 * zz - xx - yy) * coeffs[13]
        + SH_C3[5] * z * (xx - yy) * coeffs[14]
        + SH_C3[6] * x * (xx - 3.0 * yy) * coeffs[15];
    result
}

/// Full 3DGS color: 0.5 + eval, clamped non-negative per channel.
pub fn sh_to_color(degree: u8, coeffs: &[Vec3], dir: Vec3) -> Vec3 {
    (eval_sh(degree, coeffs, dir) + Vec3::splat(0.5)).max(Vec3::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coeff_counts() {
        assert_eq!(num_coeffs(0), 1);
        assert_eq!(num_coeffs(1), 4);
        assert_eq!(num_coeffs(2), 9);
        assert_eq!(num_coeffs(3), 16);
        assert_eq!(degree_from_rest_count(0), Some(0));
        assert_eq!(degree_from_rest_count(9), Some(1));
        assert_eq!(degree_from_rest_count(24), Some(2));
        assert_eq!(degree_from_rest_count(45), Some(3));
        assert_eq!(degree_from_rest_count(12), None);
    }

    #[test]
    fn degree0_is_view_independent() {
        let coeffs = [Vec3::new(1.0, -0.5, 0.25)];
        let a = eval_sh(0, &coeffs, Vec3::Z);
        let b = eval_sh(0, &coeffs, Vec3::new(0.577_350_3, -0.577_350_3, 0.577_350_3));
        assert_eq!(a, b);
        assert!((a.x - SH_C0).abs() < 1e-6);
    }

    #[test]
    fn degree1_axis_direction() {
        // dir = +z: only the c2 (z) band-1 term contributes besides DC.
        let mut coeffs = [Vec3::ZERO; 4];
        coeffs[0] = Vec3::splat(0.2);
        coeffs[1] = Vec3::splat(10.0); // y term — must not contribute at dir=+z
        coeffs[2] = Vec3::splat(0.3);
        coeffs[3] = Vec3::splat(10.0); // x term — must not contribute at dir=+z
        let got = eval_sh(1, &coeffs, Vec3::Z);
        let expect = SH_C0 * 0.2 + SH_C1 * 0.3;
        assert!((got.x - expect).abs() < 1e-6, "got {got:?}, expect {expect}");
    }

    #[test]
    fn color_offset_and_clamp() {
        let coeffs = [Vec3::new(-10.0, 0.0, 0.0)];
        let c = sh_to_color(0, &coeffs, Vec3::Z);
        assert_eq!(c.x, 0.0); // clamped
        assert!((c.y - 0.5).abs() < 1e-6); // zero coeff → 0.5 grey
    }
}
