//! Streaming binary payload decode. Reads fixed-stride rows through an aligned
//! scratch buffer (never maps or double-buffers the whole file), applying
//! activations and the SH re-interleave during the single transpose pass.

use std::io::Read;

use glam::{Quat, Vec3};
use gs_core::{PointCloud, SplatCloud};

use super::PlyError;
use super::layout::{GaussianLayout, PointsLayout};
use crate::ply::header::PropertyType;

/// Rows decoded per chunk (~8 MiB at the 236-byte degree-3 stride).
const ROWS_PER_CHUNK: usize = 32 * 1024;

pub fn read_splats(r: &mut impl Read, l: &GaussianLayout) -> Result<SplatCloud, PlyError> {
    let floats_per_row = l.stride / 4;
    let rest_per_channel = l.f_rest.len() / 3;
    let coeffs = 1 + rest_per_channel;

    let mut cloud = SplatCloud {
        positions: Vec::with_capacity(l.count),
        sh: Vec::with_capacity(l.count * 3 * coeffs),
        opacity: Vec::with_capacity(l.count),
        scales: Vec::with_capacity(l.count),
        rotations: Vec::with_capacity(l.count),
        sh_degree: l.sh_degree,
    };

    // Scratch is a Vec<f32>, so the byte view handed to read_exact is always
    // 4-byte aligned; the offset-driven path additionally indexes rows as f32s,
    // which requires stride and every used offset to be 4-byte multiples.
    if !l.fast_path {
        let offsets_ok = l.stride.is_multiple_of(4)
            && l
                .pos
                .iter()
                .chain(&l.f_dc)
                .chain(&l.f_rest)
                .chain(std::iter::once(&l.opacity))
                .chain(&l.scale)
                .chain(&l.rot)
                .all(|&o| o.is_multiple_of(4));
        if !offsets_ok {
            return Err(PlyError::BadHeader(
                "gaussian vertex row has non-4-byte-aligned properties".into(),
            ));
        }
    }

    let mut scratch = vec![0f32; ROWS_PER_CHUNK * floats_per_row];
    let mut remaining = l.count;
    let mut next_progress = l.count / 10;
    while remaining > 0 {
        let rows = remaining.min(ROWS_PER_CHUNK);
        let floats = rows * floats_per_row;
        read_exact_or_eof(r, bytemuck::cast_slice_mut(&mut scratch[..floats]), l, remaining)?;

        for row in scratch[..floats].chunks_exact(floats_per_row) {
            if l.fast_path {
                push_row_fast(&mut cloud, row, rest_per_channel);
            } else {
                push_row_offsets(&mut cloud, row, l, rest_per_channel);
            }
        }

        remaining -= rows;
        if l.count - remaining >= next_progress && remaining > 0 {
            log::debug!("ply load: {}%", 100 * (l.count - remaining) / l.count);
            next_progress += l.count / 10;
        }
    }
    Ok(cloud)
}

/// Canonical INRIA row: x y z | dc0..2 | rest.. | opacity | s0..2 | r0..3.
fn push_row_fast(cloud: &mut SplatCloud, row: &[f32], rest_per_channel: usize) {
    let rest = 3 * rest_per_channel;
    cloud.positions.push(Vec3::new(row[0], row[1], row[2]));
    push_sh(&mut cloud.sh, &row[3..6], &row[6..6 + rest], rest_per_channel);
    let o = 6 + rest;
    cloud.opacity.push(sigmoid(row[o]));
    cloud
        .scales
        .push(Vec3::new(row[o + 1].exp(), row[o + 2].exp(), row[o + 3].exp()));
    cloud.rotations.push(quat_from_ply(
        row[o + 4],
        row[o + 5],
        row[o + 6],
        row[o + 7],
    ));
}

/// Offset-driven fallback for reordered rows / extra float properties.
fn push_row_offsets(
    cloud: &mut SplatCloud,
    row: &[f32],
    l: &GaussianLayout,
    rest_per_channel: usize,
) {
    let at = |byte_off: usize| row[byte_off / 4];
    cloud
        .positions
        .push(Vec3::new(at(l.pos[0]), at(l.pos[1]), at(l.pos[2])));
    let dc = [at(l.f_dc[0]), at(l.f_dc[1]), at(l.f_dc[2])];
    let rest: Vec<f32> = l.f_rest.iter().map(|&off| at(off)).collect();
    push_sh(&mut cloud.sh, &dc, &rest, rest_per_channel);
    cloud.opacity.push(sigmoid(at(l.opacity)));
    cloud.scales.push(Vec3::new(
        at(l.scale[0]).exp(),
        at(l.scale[1]).exp(),
        at(l.scale[2]).exp(),
    ));
    cloud.rotations.push(quat_from_ply(
        at(l.rot[0]),
        at(l.rot[1]),
        at(l.rot[2]),
        at(l.rot[3]),
    ));
}

/// Re-interleave SH from the ply's channel-major f_rest (all red coeffs, then
/// green, then blue) into coefficient-major [c0.rgb, c1.rgb, ...].
fn push_sh(sh: &mut Vec<f32>, dc: &[f32], rest: &[f32], rest_per_channel: usize) {
    sh.extend_from_slice(dc);
    for i in 0..rest_per_channel {
        sh.push(rest[i]);
        sh.push(rest[rest_per_channel + i]);
        sh.push(rest[2 * rest_per_channel + i]);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// INRIA convention stores rot_0 = w; glam wants xyzw. Degenerate quats fall
/// back to identity rather than poisoning downstream math with NaNs.
fn quat_from_ply(w: f32, x: f32, y: f32, z: f32) -> Quat {
    let q = Quat::from_xyzw(x, y, z, w);
    if q.length_squared() > 1e-12 {
        q.normalize()
    } else {
        Quat::IDENTITY
    }
}

pub fn read_points(r: &mut impl Read, l: &PointsLayout) -> Result<PointCloud, PlyError> {
    let mut cloud = PointCloud {
        positions: Vec::with_capacity(l.count),
        colors: Vec::with_capacity(l.count),
    };
    // Point rows can mix types (float xyz + uchar rgb) → byte-level scratch.
    let rows_per_chunk = (8 << 20) / l.stride.max(1);
    let mut scratch = vec![0u8; rows_per_chunk * l.stride];
    let mut remaining = l.count;
    while remaining > 0 {
        let rows = remaining.min(rows_per_chunk);
        let bytes = rows * l.stride;
        r.read_exact(&mut scratch[..bytes])
            .map_err(|_| PlyError::UnexpectedEof {
                expected: l.count,
                got: l.count - remaining,
            })?;
        for row in scratch[..bytes].chunks_exact(l.stride) {
            let f = |off: usize| f32::from_le_bytes(row[off..off + 4].try_into().unwrap());
            cloud.positions.push(Vec3::new(f(l.pos[0]), f(l.pos[1]), f(l.pos[2])));
            let color = match l.color {
                Some((offs, PropertyType::UChar)) => [row[offs[0]], row[offs[1]], row[offs[2]]],
                Some((offs, _float)) => offs.map(|o| (f(o).clamp(0.0, 1.0) * 255.0) as u8),
                None => [200, 200, 200],
            };
            cloud.colors.push(color);
        }
        remaining -= rows;
    }
    Ok(cloud)
}

fn read_exact_or_eof(
    r: &mut impl Read,
    buf: &mut [u8],
    l: &GaussianLayout,
    remaining: usize,
) -> Result<(), PlyError> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            PlyError::UnexpectedEof {
                expected: l.count,
                got: l.count - remaining,
            }
        } else {
            PlyError::Io(e)
        }
    })
}
