//! Synthetic scenes with known geometry.
//!
//! Per CLAUDE.md, ground truth beats a reimplementation as an oracle: a scene
//! whose geometry and camera path are *known* turns "did it recover the right
//! answer" into a fact. Every milestone from here validates against one of
//! these before it is pointed at real video, because on real video a wrong
//! answer and a hard scene look identical.

use glam::{Quat, Vec3};

use crate::{Surfel, SurfelMap};

/// A closed box room, surfels facing inward, with a deterministic colour
/// pattern on every wall.
///
/// A room rather than a plane because the thing being tested is depth: a
/// fronto-parallel plane can be fitted at the wrong scale and still look
/// right from the bootstrap view, so it hides exactly the failure that
/// matters. Walls at different distances cannot.
///
/// `per_side` surfels per axis per wall; `size` is the half-extent.
pub fn room(per_side: usize, size: f32) -> SurfelMap {
    let mut map = SurfelMap::new();
    let n = per_side.max(2);
    let step = 2.0 * size / (n - 1) as f32;
    // Radius slightly over half the spacing so the walls are opaque rather
    // than a grid of dots with gaps the tracker could lock onto.
    let radius = step * 0.75;

    // (outward axis, sign) for the six faces. Normals face INWARD, toward a
    // camera inside the room.
    let faces: [(Vec3, f32); 6] = [
        (Vec3::X, 1.0),
        (Vec3::X, -1.0),
        (Vec3::Y, 1.0),
        (Vec3::Y, -1.0),
        (Vec3::Z, 1.0),
        (Vec3::Z, -1.0),
    ];

    for (fi, (axis, sign)) in faces.iter().enumerate() {
        // Two in-plane axes for this face.
        let u = if axis.x != 0.0 { Vec3::Y } else { Vec3::X };
        let v = axis.cross(u).normalize();
        let normal = -*axis * *sign; // inward
        let orientation = Quat::from_rotation_arc(Vec3::Z, normal);
        for i in 0..n {
            for j in 0..n {
                let a = -size + i as f32 * step;
                let b = -size + j as f32 * step;
                let position = *axis * *sign * size + u * a + v * b;
                map.push(Surfel {
                    position,
                    orientation,
                    scale: [radius, radius],
                    opacity: 0.95,
                    // Deterministic, non-repeating across the room: a
                    // repetitive pattern matches at many poses, which would
                    // make a tracking test pass for the wrong reason.
                    color: swatch(fi, i, j),
                });
            }
        }
    }
    map
}

/// A stable pseudo-random colour per (face, i, j). Hash-based rather than
/// periodic so no two patches of wall look alike.
fn swatch(face: usize, i: usize, j: usize) -> [f32; 3] {
    let mut h = (face as u32).wrapping_mul(0x9E37_79B9)
        ^ (i as u32).wrapping_mul(0x85EB_CA6B)
        ^ (j as u32).wrapping_mul(0xC2B2_AE35);
    let mut next = || {
        h ^= h >> 15;
        h = h.wrapping_mul(0x2545_F491);
        h ^= h >> 13;
        // Keep it mid-bright: pure black or white patches carry no gradient.
        0.25 + 0.6 * ((h >> 8) as f32 / (1u32 << 24) as f32)
    };
    [next(), next(), next()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn room_has_six_faces_of_inward_surfels() {
        let n = 6;
        let map = room(n, 2.0);
        assert_eq!(map.len(), 6 * n * n);
        // Every surfel's normal must point back toward the room centre, or a
        // camera inside sees the backs of the walls.
        for k in 0..map.len() {
            let p = map.positions[k];
            let q = glam::Quat::from_array(map.quats[k]);
            let normal = q * Vec3::Z;
            assert!(
                normal.dot(-p.normalize_or_zero()) > 0.0,
                "surfel {k} at {p:?} faces outward",
            );
        }
    }

    /// Neighbouring patches must differ, or a tracking test could pass by
    /// locking onto the wrong wall.
    #[test]
    fn swatches_are_locally_distinct() {
        let a = swatch(0, 3, 4);
        for (di, dj) in [(1, 0), (0, 1), (1, 1)] {
            let b = swatch(0, 3 + di, 4 + dj);
            let diff: f32 = (0..3).map(|c| (a[c] - b[c]).abs()).sum();
            assert!(diff > 0.05, "neighbouring swatches too similar: {a:?} {b:?}");
        }
    }
}
