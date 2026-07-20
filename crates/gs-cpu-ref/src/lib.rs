//! CPU oracle: analytic forward/backward reference rasterizer (f64) plus the
//! finite-difference harness for three-way gradient agreement checks
//! (finite-diff ↔ CPU-analytic ↔ GPU WGSL). Never ships in apps.

pub mod backward;
pub mod forward;
pub mod math;
pub mod scene;

pub use backward::gradients;
pub use forward::render;
pub use scene::{Gradients, MicroScene, RefCamera, RenderOutput, Surfel};
