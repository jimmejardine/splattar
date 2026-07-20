//! WGSL training kernels (forward + hand-derived backward) and their dispatch layer.
//!
//! Any change here must pass the gs-cpu-ref gradient checks — wrong gradients
//! fail silently (see CLAUDE.md Verification rules). Shaders live in `src/shaders/`
//! as `<stage>_fwd.wgsl` / `<stage>_bwd.wgsl` pairs.
