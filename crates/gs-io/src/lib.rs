//! File I/O: .ply reading/writing (3DGS and native surfel layouts), .spz export,
//! the appendable project format, scene-manifest sidecar, and dataset harnesses.
//!
//! Exports are lossy baked snapshots — never re-ingested (see CLAUDE.md).

pub mod ply;

pub use ply::{PlyContents, PlyError, load_ply, load_ply_from};
