# Splattar

Pure-Rust pipeline: apartment walkthrough video → Gaussian-surfel (2DGS) model + extracted mesh → first-person walkthrough (desktop, VR, web). NVIDIA GPU via wgpu compute. See `PLAN.md` for the full architecture, milestone roadmap (M0–M11), and acceptance criteria — keep it current as decisions change.

## Hard constraints (do not violate)

- **100% Rust.** No C/C++ build dependencies: no ffmpeg, no COLMAP, no OpenCV, no `-sys` crates that compile or link C/C++. System runtimes loaded at runtime (GPU driver, OpenXR runtime DLL) are the only exception. Check what a crate links before adding it.
- **From scratch.** The differentiable rasterizer (WGSL ray-splat-intersection forward + hand-derived backward, 2DGS surfels), Adam optimizer, and MCMC densification are implemented here — do not depend on Brush or port its code wholesale; reading it as a reference is fine.
- **GPU via wgpu** (Vulkan primary backend on the dev machine). No CUDA.

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
cargo run -p gs-cli -- train <dataset>  # train on a COLMAP/Nerfstudio dataset
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
- iPhone footage defaults to HEVC, which `gs-video` cannot decode (NihAV is H.264). The image-folder input path is the supported fallback; keep it first-class.
