//! SoA containers for splat scenes and plain point clouds.

use glam::{Quat, Vec3};

use crate::sh;

/// A 3DGS splat scene in structure-of-arrays layout, with activations already
/// applied: `opacity` is post-sigmoid, `scales` post-exp (world units),
/// `rotations` normalized (glam xyzw order).
///
/// `sh` is flat and **coefficient-major interleaved**: per splat,
/// `[c0.r, c0.g, c0.b, c1.r, c1.g, c1.b, ...]` for `num_coeffs` coefficients.
/// This differs from the INRIA .ply layout, which stores `f_rest` channel-major
/// (all red coeffs, then green, then blue) — loaders re-interleave at read time
/// so shaders can index linearly. Getting this wrong produces plausible but
/// wrong colors; see the interleave test in gs-io.
#[derive(Debug, Clone, Default)]
pub struct SplatCloud {
    pub positions: Vec<Vec3>,
    pub sh: Vec<f32>,
    pub opacity: Vec<f32>,
    pub scales: Vec<Vec3>,
    pub rotations: Vec<Quat>,
    pub sh_degree: u8,
}

impl SplatCloud {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// SH coefficients per color channel for this cloud's degree.
    pub fn num_coeffs(&self) -> usize {
        sh::num_coeffs(self.sh_degree)
    }

    /// Floats of SH data per splat (all three channels).
    pub fn sh_floats_per_splat(&self) -> usize {
        3 * self.num_coeffs()
    }

    /// Axis-aligned bounding box over splat centers (ignores splat extents).
    /// Returns `None` for an empty cloud.
    pub fn bbox(&self) -> Option<(Vec3, Vec3)> {
        let first = *self.positions.first()?;
        let (min, max) = self
            .positions
            .iter()
            .fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        Some((min, max))
    }

    /// Center of the bounding box; `None` for an empty cloud.
    pub fn center(&self) -> Option<Vec3> {
        self.bbox().map(|(lo, hi)| (lo + hi) * 0.5)
    }
}

/// A plain colored point cloud (e.g. a non-splat .ply). Not renderable by the
/// splat pipeline; kept for loader tests and future point-init tooling.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    pub positions: Vec<Vec3>,
    pub colors: Vec<[u8; 3]>,
}

impl PointCloud {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_and_center() {
        let cloud = SplatCloud {
            positions: vec![Vec3::new(-1.0, 0.0, 2.0), Vec3::new(3.0, -4.0, 0.0)],
            ..Default::default()
        };
        let (lo, hi) = cloud.bbox().unwrap();
        assert_eq!(lo, Vec3::new(-1.0, -4.0, 0.0));
        assert_eq!(hi, Vec3::new(3.0, 0.0, 2.0));
        assert_eq!(cloud.center().unwrap(), Vec3::new(1.0, -2.0, 1.0));
        assert!(SplatCloud::default().bbox().is_none());
    }
}
