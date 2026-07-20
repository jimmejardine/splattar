//! Compat 3DGS .ply export: surfels baked into the canonical INRIA layout so
//! every standard viewer (including our own `gs-cli view`) can render trained
//! scenes. The third scale is written as a large negative log (a fully
//! flattened ellipsoid â€” the 2DGS reference convention). Exports are baked
//! snapshots: raw log-scales / logit opacities / channel-major f_rest exactly
//! inverse to the reader's activations.

use std::io::{BufWriter, Write};
use std::path::Path;

/// log(scale) written for the flattened third axis.
const FLAT_LOG_SCALE: f32 = -10.0;

/// Write `count` surfels. Buffer layouts match the trainer's readback:
/// positions vec4-strided, scales 2/surfel (activated), quats xyzw,
/// opacities activated, sh 48/surfel coefficient-major.
pub fn write_3dgs_ply(
    path: &Path,
    count: usize,
    positions: &[f32],
    scales: &[f32],
    quats: &[f32],
    opacities: &[f32],
    sh: &[f32],
) -> std::io::Result<()> {
    assert!(positions.len() >= count * 4);
    assert!(scales.len() >= count * 2);
    assert!(quats.len() >= count * 4);
    assert!(opacities.len() >= count);
    assert!(sh.len() >= count * 48);

    let mut w = BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "ply\nformat binary_little_endian 1.0")?;
    writeln!(w, "comment splattar baked export (surfels flattened to 3DGS)")?;
    writeln!(w, "element vertex {count}")?;
    for name in ["x", "y", "z"] {
        writeln!(w, "property float {name}")?;
    }
    for i in 0..3 {
        writeln!(w, "property float f_dc_{i}")?;
    }
    for i in 0..45 {
        writeln!(w, "property float f_rest_{i}")?;
    }
    writeln!(w, "property float opacity")?;
    for i in 0..3 {
        writeln!(w, "property float scale_{i}")?;
    }
    for i in 0..4 {
        writeln!(w, "property float rot_{i}")?;
    }
    writeln!(w, "end_header")?;

    let mut row = Vec::with_capacity(59);
    for i in 0..count {
        row.clear();
        row.extend_from_slice(&positions[i * 4..i * 4 + 3]);
        // f_dc = coeff 0 rgb.
        row.extend_from_slice(&sh[i * 48..i * 48 + 3]);
        // f_rest: channel-major over coeffs 1..15 from our coeff-major layout.
        for ch in 0..3 {
            for k in 1..16 {
                row.push(sh[i * 48 + k * 3 + ch]);
            }
        }
        // Inverse sigmoid.
        let o = opacities[i].clamp(1e-6, 1.0 - 1e-6);
        row.push((o / (1.0 - o)).ln());
        // Log scales, flattened third axis.
        row.push(scales[i * 2].max(1e-9).ln());
        row.push(scales[i * 2 + 1].max(1e-9).ln());
        row.push(FLAT_LOG_SCALE);
        // INRIA quat order is w-first.
        let q = &quats[i * 4..i * 4 + 4];
        row.extend_from_slice(&[q[3], q[0], q[1], q[2]]);
        w.write_all(bytemuck::cast_slice(&row))?;
    }
    w.flush()
}
