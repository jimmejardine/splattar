//! Viewer rasterizer (separate from the training rasterizer in gs-kernels):
//! preprocess → depth sort → instanced-quad draw, shared by desktop, VR, and web.
//! Shaders live in `src/shaders/`.

pub mod golden;
pub mod offscreen;
pub mod pipeline;
pub mod scene;

pub use pipeline::{RenderSettings, SplatRenderer};
pub use scene::GpuScene;
