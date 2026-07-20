//! Viewer core: camera controllers and render-loop orchestration, decoupled from
//! any particular surface. The winit window/event loop lives behind the default
//! `winit` feature so app-web and app-vr can reuse the core without it.
