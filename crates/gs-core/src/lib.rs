//! Core math and types shared by every splattar crate.
//!
//! Splat SoA structs, camera model, and spherical-harmonics evaluation.
//! This crate must stay wasm-safe and dependency-light (glam/bytemuck/half only).

pub mod camera;
pub mod sh;
pub mod splat;

pub use camera::Camera;
pub use splat::{PointCloud, SplatCloud};
