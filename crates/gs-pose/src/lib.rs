//! Visual-odometry front-end and submap registration. Geometry runs anchor-out
//! (never bootstrap at a segment boundary); registration is deferred and
//! continuous (islands merge wherever overlap appears). nalgebra is quarantined
//! in this crate — public APIs speak glam.

pub mod ba;
pub mod detect;
pub mod image;
pub mod klt;
pub mod pnp;
pub mod se3;
pub mod triangulate;
pub mod twoview;

pub use detect::{Corner, DetectConfig, detect};
pub use image::{GrayImage, Pyramid};
pub use klt::{KltConfig, track_point, track_point_fb};
pub use se3::Se3;
