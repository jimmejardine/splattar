# Splattar — working rules

This file holds only the rules for working in this repo. Architecture, workspace map, pipeline, roadmap, and acceptance criteria: [PLAN.md](PLAN.md) (keep it current as decisions change). Measured numbers per milestone: [RESULTS.md](RESULTS.md) (append dated notes, don't rewrite history). User-facing intro, quickstart, and test-asset rebuild: [README.md](README.md).

## Hard constraints (do not violate)

- **100% Rust.** No C/C++ build deps: no ffmpeg, COLMAP, OpenCV, no `-sys` crates that compile or link C/C++. Runtime-loaded system libs (GPU driver, OpenXR runtime DLL) are the only exception. Check what a crate links before adding it.
- **From scratch.** The differentiable 2DGS rasterizer (WGSL forward + hand-derived backward), Adam optimizer, and MCMC densification are this repo's code. Brush is reference reading only — never a dependency, never ported wholesale.
- **GPU via wgpu.** No CUDA.
- **Video is the only product input.** The pipeline is SLAM-shaped (VO → incremental submaps → Sim(3) graph → global refinement). Never reintroduce unordered-photo SfM or random-view batch training as the product path; posed-frame loaders are internal validation harnesses only.

## Conventions

- Crate-per-concern workspace (map: PLAN.md §Workspace). `glam` for math everywhere; `nalgebra` and `faer` are quarantined inside `gs-pose` — convert at its API boundary.
- Pin shared versions in `[workspace.dependencies]`; crates use `dep.workspace = true`.
- Every pipeline stage runs headless via `gs-cli` before any GUI work.
- WGSL lives in `crates/gs-kernels/src/shaders/` and `crates/gs-render/src/shaders/`; keep forward/backward pairs adjacent, named `<stage>_fwd.wgsl` / `<stage>_bwd.wgsl`.
- `app-web` builds only for `wasm32-unknown-unknown`; exclude it from native workspace commands if it breaks them.

## Commands

```
cargo build --workspace                    # native build (excludes app-web)
cargo test --workspace                     # fast gates; slow GPU/real-video tests need -- --ignored
cargo clippy --workspace --tests -- -D warnings
cargo run -p gs-cli -- add <video.mp4> [--project <dir>]  # the ONLY ingestion command; default project: gs-project next to the video; add order doesn't matter
cargo run -p gs-cli -- view <ply|project>  # render a splat file or compose a project
cargo run -p gs-cli -- pose <video.mp4>    # VO only: keyframe trajectory CSV
cargo run -p gs-cli -- play <video.mp4>    # frame-stepping decode sanity player
cargo run -p gs-cli -- register <project> --submap <n> [--write]  # offline registration lab
cargo run -p gs-cli -- refocal <project> --submap <n>  # re-BA a submap with the trainer-refined focal
cargo run -p gs-cli -- train <dataset>     # validation harness: posed COLMAP datasets only
cargo run -p gs-cli -- export <project>    # baked .ply + scene-manifest sidecar
```

Use `--release` for anything touching real video or training (KLT and the trainer are unusable in dev; gs-pose alone gets opt-level 3 in dev via a profile override).

## Test assets

`samples/` is gitignored and large — **never write derived outputs into `samples/`** (default project dirs land next to the video, so pass `--project` when ingesting sample clips). Fixtures: `samples/ply/cactus-low.ply` = golden tests / fast iteration, `cactus-high` = perf fixture, `samples/video/prinsengracht-494-*` = real phone walkthroughs for ingest/VO/end-to-end/M8-merge work. Formats, sizes, and how to rebuild them on a new machine: README.md §Rebuilding local data.

## Verification rules

- **Any change to a WGSL kernel or its dispatch code must pass the gradient checks** in `gs-cpu-ref`: finite-difference ↔ CPU-analytic ↔ GPU three-way agreement on randomized micro-scenes. Never land kernel changes on green compile alone — wrong gradients fail silently.
- Rendering changes must pass the golden-image tests (PSNR tolerance vs `assets/golden/`, not byte hashes). Regenerate goldens only for intended visual changes, and say so in the commit.
- Registration/merge changes must pass the seam golden tests (double-wall check) and the relocalization benchmark, including its no-false-positive requirement.
- Perf-sensitive kernels are measured with the `gs-wgpu` timestamp-query bench harness; budgets in PLAN.md §Verification. Flag regressions, don't bury them.
- Debug pose and trainer problems separately: trainer on known-pose datasets, `gs-pose` on GT-trajectory datasets. Never tune both against the same failing end-to-end run.

## Gotchas

GPU / kernels:
- WGSL has no f32 atomics: gradient accumulation is per-tile shared memory flushed via u32-bitcast CAS. All new accumulation code follows this pattern. Workgroup-uniform slots (per-splat slots in `rasterize_bwd`, the 13 `grad_cam` slots) route through `red_add`/`red_cam_add`, string-swapped at pipeline creation for a subgroup pre-reduction when `Features::SUBGROUP` is present (`SPLATTAR_NO_SUBGROUPS=1` forces scalar). Keep the marker text in sync with `Rasterizer::new` (an assert guards it) and pass gradient checks on **both** paths. naga 30: no `subgroupElect` (use `subgroupExclusiveAdd(1u)==0u`), `enable subgroups;` rejected.
- Splat buffers are SoA, pre-allocated to the fixed MCMC budget — no mid-training reallocation. Training is f32 throughout; the viewer packs f16 — don't mix.
- Primitives are 2D surfels (2 scales + orientation). The trainer only emits the surfel layout, but `gs-io`/`gs-render` must also load standard 3-scale 3DGS .ply. Compat exports append a zero third scale; SPZ writing stays on flate2's pure-Rust backend (SPZ 4 blocked on a pure-Rust ZSTD encoder; reading via `ruzstd` is fine).
- wgpu exposes one queue per device — no async-compute streams. Parallelism = CPU-side pipelining feeding a serial GPU queue; never design for concurrent GPU training jobs.

Training loop:
- **The hot loop is submit-only — never add a blocking readback.** `gs_wgpu::buffers::readback` is for tests/tools/export; per-iteration reads go through `gs_wgpu::ReadbackRing` (async, results 1–2 iterations late — appearance fits and pose updates are designed for that latency). A per-iteration blocking readback once collapsed training ~26 → ~2 it/s.
- Targets are GPU-resident (per-view atlas uploaded once); the per-view affine appearance correction and its least-squares fit run as kernels, with the affine applied to the *target* (a constant in the gradient path — applying it to the render would change gradients). Never bake per-view appearance into surfel SH; never reintroduce per-iteration target uploads.
- Geometry losses phase in after warm-up (`geo_start`, ~zero marginal cost per `geo_bench`); enabling them at iteration 0 hurts convergence. Mesh extraction consumes rendered median depth, not splat centers.
- Pose refinement: camera-center LR scales with the view's **median scene depth**, never scene extent (extent is ~30× room depth on a walkthrough). Held-out eval must photometrically align eval poses to the frozen model first (`eval_psnr_refined`, BARF-style) — frozen-pose eval measures gauge drift, not model quality. Focal is one shared per-submap parameter, refined in log-space; the trainer persists it as `focal_refined` (feed back via `refocal`).
- The trainer logs per-phase host ms and sampled per-kernel GPU ms every `log_every` — trust those over folklore when perf shifts.

Video:
- Decode is **hardware-only**: Vulkan Video (NVDEC) H.264 + H.265 sessions via raw ash FFI in `gs-video`. The bake-off rejected every software decoder (CAVLC-only / I-only / AGPL) — never add software codec crates, and beware parser-only crates that sound like decoders. ash 0.38 has no video wrappers: raw `fns.fp().xxx_khr` calls; bindgen `StdVideo*` structs via `mem::zeroed()` + bitfield setters, never positional `new_bitfield_1`.
- Phone video is VFR — always carry per-frame PTS (demux → decode → keyframe promotion → spline), never frame_index/fps. Display rotation (tkhd matrix) is baked into decoded pixels; 10-bit HEVC tone-maps BT.2020→709 at decode.

VO (`gs-pose`):
- The anchor-out keyframe loop is serial; rayon parallelism lives *inside* each step, and every parallel reduction merges partials in index order for determinism (a bare `par_iter().sum()` is not deterministic). The reduced-camera solve uses faer sparse Cholesky with a cached symbolic factorization.
- The causal pass is a three-stage pipeline (decode thread → prep pool building pyramids/sharpness, pure per-frame → tracking spine consuming in index order): keep prep stateless and the spine order-strict or determinism breaks. Don't add `with_min_len` to the per-track KLT fan-out — LK effort varies wildly per track and fixed chunks defeat work-stealing (measured 3× regression). HEVC clips are decode-bound (~46 fps NVDEC H.265), H.264 clips spine-bound — profile before optimizing either side (the spine logs its klt/desc/detect split).
- Submaps build anchor-out: the causal pass runs once in decode order; geometry grows both directions from the best-conditioned anchor window. Never bootstrap at a segment boundary (boundaries sit at bad footage by construction) and don't conflate decode order with estimation order.
- Bootstrap is the five-point (Nistér–Stewénius) solver — eight-point is numerically fragile on rotation-dominant low-parallax geometry (broke on a real rotated clip). Pair selection measures parallax via a global-affine residual, never raw flow (panning = huge flow, zero baseline). KLT is zero-mean (survives auto-exposure) — keep it that way. Keyframe survival stats must drop dead tracks at each promotion or every frame becomes a keyframe; dense keyframes during fast pans are correct.
- `solve_segments` recurses into unsolved flanks; each segment is its own gauge with landmark state reset between (gauge leaks otherwise). PnP needs ≥12 landmark support or self-consistent false matches on repetitive texture glue poses across genuine cuts. Ranges that can't bootstrap are dropped honestly.

Projects / registration:
- **No submap has a privileged gauge.** Projects store only pairwise Sim(3) edges (`edge=` lines in meta.txt, submap-local → target-local; no edges = island); placement is resolved per connected component (`project::resolve_placements`, union-find + BFS from lowest index) at compose/register time. Add order affects only submap indices and the +x component layout (presentation-only, never stored), never connectivity.
- Two-tier data model: surfels stay in submap-local coordinates forever; world placement is composed at render/export time. Exports are lossy snapshots — never re-ingest them.
- Registration is deferred and continuous: new videos build as islands, constraints attach wherever overlap appears, components merge when connected. Never gate ingestion on relocalizing first frames; islands are first-class (archipelago view). Walk mode/collision is per-island.
- Sim(3) acceptance needs: spatially voxel-deduped landmarks + cross-checked matches (KLT respawns duplicate corners ~20× and a scale→0 collapse out-votes the truth), scale bounds AND an inlier-spread gate, and per-component grouping of candidates before RANSAC (pooling across components mixes unrelated gauges).
- Monocular scale is per-submap until Sim(3) alignment (7-DOF); intrinsics are per-submap groups (zoom/lens switches force boundaries) — never one focal per video.
- Appearance (auto-exposure/AWB) is a per-submap affine color spline applied in the training loss; the scene stays canonical.
