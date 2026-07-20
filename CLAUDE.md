# Splattar

Pure-Rust pipeline: apartment walkthrough video → Gaussian-surfel (2DGS) model + extracted mesh → first-person walkthrough (desktop, VR, web). NVIDIA GPU via wgpu compute. See `PLAN.md` for the full architecture, milestone roadmap (M0–M12), and acceptance criteria — keep it current as decisions change.

## Hard constraints (do not violate)

- **100% Rust.** No C/C++ build dependencies: no ffmpeg, no COLMAP, no OpenCV, no `-sys` crates that compile or link C/C++. System runtimes loaded at runtime (GPU driver, OpenXR runtime DLL) are the only exception. Check what a crate links before adding it.
- **From scratch.** The differentiable rasterizer (WGSL ray-splat-intersection forward + hand-derived backward, 2DGS surfels), Adam optimizer, and MCMC densification are implemented here — do not depend on Brush or port its code wholesale; reading it as a reference is fine.
- **GPU via wgpu** (Vulkan primary backend on the dev machine). No CUDA.
- **Video is the only product input — one or many.** No photo/image-folder input mode. The pipeline is video-native and SLAM-shaped (tracking front-end → incremental mapping → short global refinement), exploiting temporal structure: KLT flow tracking, PTS-parameterized SE(3) trajectory spline, dense small-baseline depth for surfel init, temporal minibatches with tile-sort reuse, track-new-keyframes-against-the-model. **The scene is a graph of submaps** (overlapping segments of each video; patch videos join via relocalization) aligned by Sim(3) pose-graph relaxation — monocular scale is per-submap until graph alignment, so never assume one global scale before it. Do not reintroduce unordered-photo SfM or random-view batch training as the product path. Posed-frame-sequence loaders are internal validation harnesses only.

## Workspace

Cargo workspace, one crate per concern under `crates/`: `gs-core` (math/types, wasm-safe), `gs-io` (.ply/.spz/datasets), `gs-wgpu` (device, radix sort, prefix sum, bench harness), `gs-kernels` (WGSL training kernels), `gs-cpu-ref` (CPU oracle, never ships), `gs-train`, `gs-video`, `gs-pose`, `gs-render` (viewer rasterizer — separate from training rasterizer), `gs-viewer`, `gs-cli`, `app-desktop`, `app-vr`, `app-web`.

Conventions:
- `glam` for math everywhere; `nalgebra` is quarantined inside `gs-pose` (rust-cv ecosystem needs it) — convert at its API boundary.
- Pin shared versions in `[workspace.dependencies]`; crates use `dep.workspace = true`.
- Every pipeline stage must run headless via `gs-cli` (`extract`, `pose`, `train`, `export`, `view`, `run`) before any GUI work.
- WGSL shaders live in `crates/gs-kernels/src/shaders/` and `crates/gs-render/src/shaders/`; keep forward/backward pairs adjacent and name them `<stage>_fwd.wgsl` / `<stage>_bwd.wgsl`.

## Commands

```
cargo build --workspace                 # native build (excludes app-web)
cargo test --workspace                  # unit + property + gradient-check tests
cargo run -p gs-cli -- view <file.ply>  # render a splat file
cargo run -p gs-cli -- run <video.mp4>  # full pipeline: video → splat model (creates a project)
cargo run -p gs-cli -- add <video.mp4>  # extend an existing project with a patch video (relocalize + merge)
cargo run -p gs-cli -- train <dataset>  # validation harness: posed video-sequence datasets only
cargo clippy --workspace -- -D warnings
```

`app-web` builds only for `wasm32-unknown-unknown` (via trunk or wasm-pack); exclude it from native workspace commands if it breaks them.

## Verification rules

- **Any change to a WGSL kernel or its dispatch code must pass the gradient checks** in `gs-cpu-ref`: finite-difference ↔ CPU-analytic ↔ GPU three-way agreement on randomized micro-scenes. Never land kernel changes on green compile alone — wrong gradients fail silently.
- Rendering changes must pass golden-image tests (PSNR-tolerance comparison against `assets/golden/`, not byte hashes — drivers differ). Regenerate goldens only when a visual change is intended, and say so in the commit.
- Perf-sensitive kernels are measured with the `gs-wgpu` timestamp-query bench harness; budgets are in PLAN.md §Verification. Flag regressions, don't bury them.
- Debug pose and trainer problems separately: validate the trainer on known-pose datasets, validate `gs-pose` against datasets with COLMAP ground truth. Never tune both against the same failing end-to-end run.

## Gotchas

- WGSL has no f32 atomics: gradient accumulation uses per-tile shared-memory accumulation flushed via u32-bitcast CAS. Any new accumulation code must follow this pattern.
- Splat buffers are SoA and pre-allocated to the fixed MCMC budget — no mid-training reallocation.
- Training uses f32 throughout; the viewer path uses packed f16 (`shader-f16` when available). Don't mix.
- Primitives are 2D Gaussian surfels (2 scales + orientation), not 3D ellipsoids. The trainer only produces the surfel layout, but `gs-io`/`gs-render` must also load standard 3DGS .ply files (3 scales) — M0 and golden tests depend on public 3DGS scenes.
- Geometry losses (depth-distortion, normal-consistency) phase in after warm-up iterations — enabling them from iteration 0 hurts convergence. Mesh extraction consumes rendered median depth, not splat centers.
- iPhone footage defaults to HEVC in .mov (10-bit Main 10, often Dolby Vision/HLG). H.264 decode candidates: NihAV, `rusty_h264`; HEVC candidate: `rust_h265` (pure Rust, new 2026-07) — treat all as unproven until validated against reference decodes of real clips. Beware parser-only crates that sound like decoders (`media-codec-h265`, `scuffle-h26x`, `h264-reader` — headers only, no frames). Decode is hard-blocking (video-only input): validate decoders early, and use the internal frame-sequence test harness to keep trainer/VO work unblocked if decode stalls. 10-bit HLG needs tone mapping before training.
- iPhone video is Variable Frame Rate. Never compute timing as frame_index / fps — always use the PTS carried per-frame from the demuxer, and preserve PTS through decode → keyframe selection.
- Submap seams: overlap regions are deduplicated during joint refinement, and cross-submap alignment is Sim(3) (7-DOF — scale included). Any change to registration or merge code must pass the seam golden tests (double-wall check) and the relocalization benchmark (incl. its no-false-positive requirement).
