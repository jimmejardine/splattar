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
cargo build --workspace                    # native build (excludes app-web)
cargo test --workspace                     # fast gates; slow GPU/real-video tests need -- --ignored
cargo run -p gs-cli -- view <ply|project>  # render a splat file, or compose a project dir (islands offset)
cargo run -p gs-cli -- run <video.mp4>     # full pipeline: video → VO → train → project dir + baked splat
cargo run -p gs-cli -- add <video.mp4> --project <dir>  # extend a project (register via Sim(3) or island)
cargo run -p gs-cli -- pose <video.mp4>    # VO only: keyframe trajectory CSV
cargo run -p gs-cli -- train <dataset>     # validation harness: posed COLMAP datasets only
cargo clippy --workspace -- -D warnings
```

Use `--release` for anything touching real video or training (KLT and the
trainer are unusable in dev profile; gs-pose gets opt-level 3 in dev via a
profile override, the rest does not).

`app-web` builds only for `wasm32-unknown-unknown` (via trunk or wasm-pack); exclude it from native workspace commands if it breaks them.

## Test assets

`samples/ply/cactus-{low,med,high}.ply` are standard 3DGS splat files (binary little-endian; x/y/z, `f_dc_0..2`, `f_rest_0..44` = SH degree 3, opacity, 3 scales, quaternion; no normals): 139,410 / ~700k / 1,935,120 splats. Use `cactus-low` for fast iteration and golden tests, `cactus-high` as the M0 performance fixture. `samples/video/prinsengracht-494-android/{1,2}.mp4` are real Android walkthrough clips — ingest/VO/end-to-end fixtures from M5 on. Samples are gitignored and large — never write derived outputs into `samples/`.

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
- Surfel exports: native surfel .ply has 2 scales; compat exports (3DGS .ply, .spz) append a zero third scale (2DGS reference convention). SPZ writing targets gzip-era SPZ via flate2's default pure-Rust backend (miniz_oxide) — never enable flate2's zlib C backends; SPZ 4 writing needs a pure-Rust ZSTD encoder (blocked as of 2026-07; reading via `ruzstd` is fine).
- Video decode is **hardware-only**: Vulkan Video (NVDEC) via raw ash FFI in `gs-video::nvdec`. The M5 bake-off rejected every software decoder (`rusty_h264` CAVLC-only — phone footage is CABAC; `oxideav-h264` I-slices only; NihAV works but is AGPL-3.0) — never add software codec crates, and beware parser-only crates that sound like decoders (`media-codec-h265`, `scuffle-h26x`, `h264-reader`). iPhone HEVC (10-bit Main 10, HLG/Dolby Vision in .mov) needs a `video_decode_h265` session on the same machinery (not yet written) + tone mapping; the image-folder input path is the supported fallback until then.
- ash 0.38 has no high-level wrappers for video extensions: calls go through `fns.fp().xxx_khr` raw function pointers, and bindgen `StdVideo*` structs must be built with `std::mem::zeroed()` + bitfield setter methods — never positional `new_bitfield_1`.
- iPhone video is Variable Frame Rate. Never compute timing as frame_index / fps — use the per-frame PTS from the demuxer, preserved through decode → keyframe promotion → trajectory spline.
- VO parallelism (rayon, CPU-side only): the anchor-out keyframe loop is inherently serial — parallelism lives *inside* each step (per-feature KLT, per-track PnP-gather/triangulation scans, per-landmark BA assembly/Schur) plus the decode↔track thread split in `gs-cli::run_vo`. Every parallel reduction merges partials in index/chunk order so results are deterministic under any thread count — keep that property when touching these loops (a bare `par_iter().sum()` is not). The dense reduced-camera solve in `ba.rs` (Cholesky, LU fallback) is still serial and is the scaling wall for global BA on 1000+ keyframe segments; the structural fix is a sparse solver, not more threads.
- VO on real footage (all learned the hard way, all now encoded in `gs-pose`): keyframe survival statistics must drop dead tracks at each promotion or the ratio decays monotonically and every frame becomes a keyframe; bootstrap pair selection must measure parallax via a **global-affine residual**, never raw flow (panning creates hundreds of px of flow with zero baseline); dense keyframes during fast pans are correct behavior, not a bug. KLT matching is zero-mean (survives auto-exposure) — keep it that way.
- Trainer pose refinement (M7): the camera-center LR scales with the view's **median scene depth** (probe cloud), never scene extent — a walkthrough's extent is ~30× the room depth and extent-scaled steps destabilize training. Held-out eval must photometrically align eval poses to the frozen model before scoring (`eval_psnr_refined`, BARF-style): training legitimately drifts the monocular gauge, and frozen-VO-pose eval measures that drift, not model quality. Focal is one shared per-submap parameter refined in log-space.
- Cross-video registration (M8): landmarks must be spatially voxel-deduped and descriptor matches cross-checked before Sim(3) RANSAC — KLT respawns re-triangulate the same physical corner ~20× per video, and coincident duplicates let a degenerate scale→0 transform out-vote the true model. Guard every Sim(3) acceptance with scale bounds AND an inlier-spread gate. Current single-scale non-oriented BRIEF descriptors are insufficient across viewpoint change (the flat pair lands as an island); AKAZE-class descriptors are the open upgrade — don't spend time retuning thresholds instead.
- Geometry losses: `examples/geo_bench` measured them at ~zero marginal cost per iteration, and `gs-cli run` enables them at `geo_start: 1500` (the earlier ">10× per iteration" observation tracked scene evolution, not these kernels). The trainer now logs per-phase host ms and sampled per-kernel GPU ms every `log_every` — trust those numbers over folklore when perf shifts.
- **The training hot loop is submit-only — never add a blocking readback to it.** `gs_wgpu::buffers::readback` (map + `poll(wait)`) is for tests, tools, and export; anything read per-iteration goes through `gs_wgpu::ReadbackRing` (persistent staging slots, async map, results delivered 1–2 iterations late — appearance fits and pose updates are designed for that latency). Per-iteration blocking readbacks are exactly what once collapsed training from ~26 to ~2 it/s.
- Training targets are GPU-resident (per-view atlas uploaded once in `gs-train::appearance`); the per-view affine appearance correction and its least-squares fit run as kernels (`appearance.wgsl`), with the affine still applied to the *target* (a constant in the gradient path — applying it to the render would change gradients). Don't reintroduce per-iteration target uploads or host-side image transforms.
- Backward-kernel gradient accumulation is CAS-based, but slots that are **workgroup-uniform** (per-splat slots in `rasterize_bwd`, the 13 `grad_cam` slots) route through `red_add`/`red_cam_add`, whose bodies are string-swapped at pipeline creation for a subgroup pre-reduction when the device has `Features::SUBGROUP` (`SPLATTAR_NO_SUBGROUPS=1` forces the scalar path). Any change to those kernels must keep the marker text in sync with `Rasterizer::new` (an assert guards drift) and must pass gradient checks on **both** paths. naga 30 quirks: no `subgroupElect` (use `subgroupExclusiveAdd(1u)==0u`), and `enable subgroups;` is rejected.
