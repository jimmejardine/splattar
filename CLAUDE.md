# Splattar

Pure-Rust, video-only pipeline: walkthrough video(s) of an apartment → 2D-Gaussian-surfel (2DGS) model + extracted mesh → real-time first-person walkthrough on desktop, VR (OpenXR), and web (WASM/WebGPU). All GPU work runs on wgpu compute (Vulkan primary backend on the NVIDIA dev machine). `PLAN.md` holds the full architecture, milestone roadmap (M0–M12), and acceptance criteria — keep it current as decisions change.

## Hard constraints (do not violate)

- **100% Rust.** No C/C++ build dependencies: no ffmpeg, no COLMAP, no OpenCV, no `-sys` crates that compile or link C/C++. System runtimes loaded at runtime (GPU driver, OpenXR runtime DLL) are the only exception. Check what a crate links before adding it.
- **From scratch.** The differentiable 2DGS rasterizer (WGSL ray-splat-intersection forward + hand-derived backward), Adam optimizer, and MCMC densification are implemented in this repo. Brush is reference reading only — never a dependency, never ported wholesale.
- **GPU via wgpu.** No CUDA.
- **Video is the only product input** — one video or several. No photo/image-folder mode. The pipeline is SLAM-shaped: VO front-end → incremental submap mapping → Sim(3) submap-graph alignment → short global refinement. Never reintroduce unordered-photo SfM or random-view batch training as the product path. Posed-frame-sequence loaders exist only as internal validation harnesses.

## Workspace

Cargo workspace, one crate per concern under `crates/`: `gs-core` (math/types, wasm-safe), `gs-io` (.ply/.spz, project persistence, dataset harnesses), `gs-wgpu` (device, radix sort, prefix sum, bench harness), `gs-kernels` (WGSL training kernels), `gs-cpu-ref` (CPU oracle, never ships), `gs-train`, `gs-video`, `gs-pose`, `gs-render` (viewer rasterizer — separate from the training rasterizer), `gs-viewer`, `gs-cli`, `app-desktop`, `app-vr`, `app-web`.

Conventions:
- `glam` for math everywhere; `nalgebra` is quarantined inside `gs-pose` (rust-cv ecosystem needs it) — convert at its API boundary.
- Pin shared versions in `[workspace.dependencies]`; crates use `dep.workspace = true`.
- Every pipeline stage runs headless via `gs-cli` before any GUI work.
- WGSL shaders live in `crates/gs-kernels/src/shaders/` and `crates/gs-render/src/shaders/`; keep forward/backward pairs adjacent, named `<stage>_fwd.wgsl` / `<stage>_bwd.wgsl`.

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

## Test assets

`samples/ply/cactus-{low,med,high}.ply` are standard 3DGS splat files (binary little-endian; x/y/z, `f_dc_0..2`, `f_rest_0..44` = SH degree 3, opacity, 3 scales, quaternion; no normals): ~139k / ~700k / ~1.9M splats. Use `cactus-low` for fast iteration and golden tests, `cactus-high` as the M0 performance fixture. `samples/ply/tree.ply` is **not** a splat file — it is a plain xyz+RGB point cloud (CloudCompare export); only useful for point-cloud loader tests. These files are large — never write derived outputs into `samples/`.

## Verification rules

- **Any change to a WGSL kernel or its dispatch code must pass the gradient checks** in `gs-cpu-ref`: finite-difference ↔ CPU-analytic ↔ GPU three-way agreement on randomized micro-scenes. Never land kernel changes on green compile alone — wrong gradients fail silently.
- Rendering changes must pass golden-image tests (PSNR-tolerance comparison against `assets/golden/`, not byte hashes — drivers differ). Regenerate goldens only when a visual change is intended, and say so in the commit.
- Registration/merge changes must pass the seam golden tests (double-wall check across submap boundaries) and the relocalization benchmark, including its no-false-positive requirement.
- Perf-sensitive kernels are measured with the `gs-wgpu` timestamp-query bench harness; budgets are in PLAN.md §Verification. Flag regressions, don't bury them.
- Debug pose and trainer problems separately: validate the trainer on known-pose datasets, validate `gs-pose` against datasets with GT trajectories. Never tune both against the same failing end-to-end run.

## Gotchas

- WGSL has no f32 atomics: gradient accumulation uses per-tile shared-memory accumulation flushed via u32-bitcast CAS. Any new accumulation code must follow this pattern.
- Splat buffers are SoA and pre-allocated to the fixed MCMC budget — no mid-training reallocation.
- Training uses f32 throughout; the viewer path uses packed f16 (`shader-f16` when available). Don't mix.
- Primitives are 2D Gaussian surfels (2 scales + orientation), not 3D ellipsoids. The trainer only produces the surfel layout, but `gs-io`/`gs-render` must also load standard 3DGS .ply files (3 scales) — M0 and the sample assets depend on it.
- Geometry losses (depth-distortion, normal-consistency) phase in after warm-up iterations — enabling them from iteration 0 hurts convergence. Mesh extraction consumes rendered median depth, not splat centers.
- Monocular scale is per-submap until Sim(3) graph alignment — never assume one global scale before it. Cross-submap alignment is 7-DOF (scale included).
- Submaps are built anchor-out: the causal pass (decode → KLT tracks → sharpness scores → zoom signal) runs once in decode order; geometry (bootstrap, spline, mapping) starts at the segment's best-conditioned anchor window and grows in both temporal directions. Don't conflate decode order with estimation order, and never bootstrap at a segment boundary — boundaries sit at low-quality footage by construction.
- Appearance (auto-exposure + auto white balance) is a per-submap time-varying per-channel affine color spline applied in the training loss — never bake per-view appearance into surfel SH colors; the scene stays canonical.
- Intrinsics are per-submap groups, not per-video: zoom or lens switches force submap boundaries (detected via the radial-flow signal in the causal pass). Never assume one focal length for a whole video.
- wgpu exposes one queue per device — no async-compute streams. Never design for concurrent GPU training jobs; parallelism = CPU-side pipelining (decode/track/VO/relocalize run ahead, across videos) feeding a serial GPU submap-build queue.
- Registration is deferred and continuous: new videos always build as floating islands in their own gauge; Sim(3) constraints attach wherever overlap is found (any time point, any video) and graph components merge when connected. Never gate video ingestion on relocalizing its first frames. Unconnected islands are first-class, not a failure: viewer/export show all islands side by side on a shared floor (archipelago view) so the operator sees what to film to bridge them. Island placement is a presentation-level transform per connected component — never stored in the Sim(3) graph or geometry state; recompute it whenever connectivity changes. Walk mode/collision is per-island.
- Two-tier data model: submap surfels stay in submap-local coordinates forever — world placement is composed at render/export time from the current Sim(3) solution; never bake world positions into internal storage. Exports (.ply/.spz + scene-manifest JSON) are lossy snapshots — never re-ingest them; `add` operates on the project only, then re-export.
- iPhone footage defaults to HEVC in .mov (10-bit Main 10, often Dolby Vision/HLG). Decoder candidates — H.264: NihAV, `rusty_h264`; HEVC: `rust_h265` (new 2026-07) — are unproven until validated against reference decodes of real clips. Beware parser-only crates that sound like decoders (`media-codec-h265`, `scuffle-h26x`, `h264-reader` — headers only, no frames). Decode is hard-blocking; the internal frame-sequence harness keeps trainer/VO work unblocked if it stalls. 10-bit HLG needs tone mapping before training.
- iPhone video is Variable Frame Rate. Never compute timing as frame_index / fps — use the per-frame PTS from the demuxer, preserved through decode → keyframe promotion → trajectory spline.
