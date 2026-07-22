//! .ply reading: header parsing, layout detection, streaming binary decode.

mod header;
mod layout;
mod reader;
mod writer;

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use gs_core::{PointCloud, SplatCloud};

pub use header::{Element, PlyHeader, Property, PropertyType};
pub use layout::{GaussianLayout, Layout, PointsLayout};
pub use writer::write_3dgs_ply;

/// What a .ply file turned out to contain.
#[derive(Debug)]
pub enum PlyContents {
    Splats(SplatCloud),
    Points(PointCloud),
}

#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a ply file (missing 'ply' magic)")]
    NotPly,
    #[error("unsupported ply format `{0}` (only binary_little_endian 1.0 is supported)")]
    UnsupportedFormat(String),
    #[error("malformed ply header: {0}")]
    BadHeader(String),
    #[error("no `vertex` element in ply header")]
    NoVertexElement,
    #[error("vertex property `{0}` has unsupported type (expected float)")]
    BadPropertyType(String),
    #[error("file ended early: expected {expected} vertices, payload holds {got}")]
    UnexpectedEof { expected: usize, got: usize },
    #[error(
        "not a gaussian splat ply — vertex properties are [{properties}]. \
         Splat files carry f_dc_*/opacity/scale_*/rot_* (see PLAN.md formats)"
    )]
    UnknownLayout { properties: String },
    #[error("f_rest count {0} does not match any SH degree (expected 0, 9, 24, or 45)")]
    BadShCount(usize),
}

/// Load a .ply from disk, auto-detecting splat vs point-cloud layout.
pub fn load_ply(path: impl AsRef<Path>) -> Result<PlyContents, PlyError> {
    let path = path.as_ref();
    let start = std::time::Instant::now();
    let file = File::open(path)?;
    let size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let result = load_ply_from(BufReader::with_capacity(1 << 20, file))?;
    match &result {
        PlyContents::Splats(s) => log::info!(
            "loaded {} ({:.1} MB): {} splats, SH degree {}, in {:.2?}",
            path.display(),
            size as f64 / 1e6,
            s.len(),
            s.sh_degree,
            start.elapsed()
        ),
        PlyContents::Points(p) => log::info!(
            "loaded {} ({:.1} MB): {} points (plain point cloud), in {:.2?}",
            path.display(),
            size as f64 / 1e6,
            p.len(),
            start.elapsed()
        ),
    }
    Ok(result)
}

/// Load a .ply from any reader (used by tests and future streaming sources).
pub fn load_ply_from(mut r: impl Read) -> Result<PlyContents, PlyError> {
    let header = header::parse_header(&mut r)?;
    match layout::detect(&header)? {
        Layout::Gaussian3d(l) => Ok(PlyContents::Splats(reader::read_splats(&mut r, &l)?)),
        Layout::Points(l) => Ok(PlyContents::Points(reader::read_points(&mut r, &l)?)),
    }
}
