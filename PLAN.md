# Splattar — Pure-Rust Apartment → Gaussian Splat Walkthrough App

## Context

Build a **100% Rust** application that takes phone video of an apartment walkthrough and produces a high-definition 3D Gaussian Splatting model that can be explored first-person on desktop (WASD/mouse), in VR (OpenXR), and in a shareable web viewer (WASM/WebGPU). Dev machine: Windows 11 + NVIDIA GPU.

**Settled decisions:**
- GPU via **wgpu** (Vulkan/DX12/WebGPU compute) — runs full-speed on NVIDIA; CUDA not required.
- **100% Rust, no C/C++ build deps** (no ffmpeg, no COLMAP, no OpenCV). System drivers/runtimes (GPU driver, OpenXR runtime DLL) are acceptable.
- Training pipeline written **from scratch as a learning project** — do not build on Brush (study it as reference only). We implement the differentiable tile rasterizer (forward + hand-derived backward) in WGSL, the Adam optimizer, and densification ourselves.
- Representation: **2D Gaussian surfels (2DGS) with modern improvements** — ray-splat-intersection rasterizer, depth-distortion + normal-consistency regularization, MCMC densification, Mip-Splatting-style anti-aliasing. Chosen over classic 3DGS because apartments are planar-dominant and surfels give view-consistent depth/normals → accurate surfaces, mesh export, and collision geometry. SuGaR's contribution (surface alignment + mesh) is achieved the 2DGS way: alignment is intrinsic to surfels, and meshes come from TSDF fusion + marching cubes rather than SuGaR's Poisson reconstruction (a poor fit for pure Rust). (Also surveyed: 3DGRT/3DGUT ray tracing, SVRaster sparse voxels, feed-forward VGGT/VicaSplat, PGSR/GOF/RaDe-GS surface methods.)

- **Video is the only product input.** No photo/snapshot mode. The pipeline is designed around temporal structure — sequential frames, small inter-frame baselines, a continuous camera path on a PTS timeline — instead of the legacy unordered-photo pipeline (global SfM + random-view batch training), which we explicitly do not reproduce. Architecture is SLAM-shaped: a tracking front-end feeds an incremental mapping back-end, with a short global refinement at the end. Posed-frame-sequence dataset loaders exist only as internal validation harnesses (trainer/VO benchmarking), never as a user-facing input.

**Hard parts, front-loaded:** (1) hand-derived WGSL backward pass, (2) pure-Rust visual-odometry pose pipeline (no COLMAP), (3) pure-Rust video decode — now hard-blocking since video is the only input (iPhone HEVC mitigations below).

## Workspace Layout

```
Cargo.toml                # [workspace], shared workspace.dependencies
crates/
  gs-core      # splat SoA structs, SE(3)/quat math, camera model, SH eval (glam, bytemuck, half; wasm-safe, dep-light)
  gs-io        # .ply (3DGS layout) + .spz read/write, COLMAP/Nerfstudio dataset loaders, checkpoints
  gs-wgpu      # device init, buffer pools, WGSL composition, GPU radix sort, prefix sum, timestamp-query bench harness
  gs-kernels   # WGSL training kernels: project/cull, tile binning, forward + HAND-DERIVED BACKWARD rasterize, SSIM fwd/bwd, Adam, MCMC relocation/noise
  gs-cpu-ref   # slow obviously-correct CPU fwd/bwd rasterizer + finite-difference harness (test oracle only, never ships)
  gs-train     # training loop, L1+D-SSIM, MCMC controller, regularizers, pose-refinement hooks, eval (PSNR/SSIM)
  gs-video     # streaming MP4/.mov demux + H.264/HEVC decode (PTS everywhere) + YUV→RGB/tone map, sharpness scoring, keyframe promotion; feature-flagged Vulkan Video (ash) later
  gs-pose      # VO front-end: pyramidal KLT flow w/ motion prediction, five-point bootstrap (arrsac), continuous-time SE(3) trajectory spline over PTS, sliding-window BA (factrs or hand-rolled sparse LM), flow-inconsistency transient masks; AKAZE only for bootstrap/loop-closure; nalgebra quarantined here, glam elsewhere
  gs-render    # fast forward-only viewer pipeline: depth radix sort, packed f16 splats, SH deg 0–3, size/distance LOD
  gs-viewer    # FPS camera, scene load/transform, render-loop orchestration (surface-agnostic)
  gs-cli       # headless driver: `extract`, `pose`, `train`, `export`, `view`, `run` subcommands — primary dev entry point
  app-desktop  # winit + egui shell (viewer + pipeline progress UI)
  app-vr       # OpenXR + Vulkan interop (wgpu-hal), stereo, scale-calibration UX
  app-web      # wasm32 + WebGPU, .spz streaming, static single-page bundle
assets/        # fixture scenes, golden images, tiny fixture videos
```

Key rules: training rasterizer (`gs-kernels`) and viewing rasterizer (`gs-render`) are separate pipelines sharing `gs-wgpu` sort/infra; everything runnable headless via `gs-cli` before any GUI.

## Pipeline

Video-native, SLAM-shaped — no global SfM stage, no random-view batch training:

1. **Streaming ingest:** MP4/.mov demux + decode with per-frame PTS; sharpness scoring; keyframe *promotion* (median flow displacement 15–40 px + sharpest-in-window), non-keyframes still feed tracking.
2. **Tracking front-end (VO):** pyramidal KLT optical flow with constant-velocity motion prediction (small search windows — sequential frames make descriptor matching unnecessary); five-point + ARRSAC bootstrap; **continuous-time SE(3) trajectory spline parameterized by PTS** (smoothness prior fights drift on textureless walls, VFR-native, rolling-shutter-ready); sliding-window BA; **flow-inconsistency masks for transients** (people/pets excluded from supervision).
3. **Dense surfel initialization:** small-baseline plane-sweep depth on wgpu across neighboring frames → surfels born *on surfaces with normals from local plane fits* — skips most of densification's geometry discovery, the single biggest convergence lever.
4. **Incremental mapping:** each promoted keyframe is first *tracked against the current splat model* (photometric alignment through our differentiable rasterizer — pose nearly free); surfels spawned only in newly revealed regions; sliding-window optimization with **temporal minibatches** — adjacent keyframes see nearly the same splats, so tile bins/sorts are reused across steps (L1 + D-SSIM λ=0.2, 2DGS depth-distortion + normal-consistency regularizers, mip-style 2D filter).
5. **Global refinement (short):** all keyframes, MCMC budget enforcement (2–4M surfels), progressive SH deg 0→2, joint pose-spline/focal refinement, **smooth exposure spline over PTS** (video auto-exposure varies continuously — fewer parameters, better conditioned than per-frame latents).
6. **Alignment & outputs:** floor-plane RANSAC y-up + metric scale from user reference measurement (default door ≈ 2.03 m) → mesh extraction (median depth → TSDF fusion → marching cubes → simplified OBJ/GLB, collision + export) → export .ply + .spz → viewers.

Because mapping is incremental, a live preview can grow while the video plays; target is a model minutes after ingest ends, not a multi-hour batch job.

## Key Decisions

- **Primitive: 2DGS surfels, single rasterizer — not dual 3DGS/2DGS pipelines.** The differentiable rasterizer uses 2DGS ray-splat intersection (exact perspective-correct evaluation of an oriented planar disk — no EWA affine approximation), which yields per-pixel depth and normals for free. Maintaining a second classic-3DGS training path would double the highest-risk work (M2) for little gain on planar indoor scenes. The *viewer* (`gs-render`) still loads standard 3DGS .ply files (M0 uses public scenes), so it handles both splat layouts.
- **Surface quality: 2DGS regularizers, SuGaR by concept only.** Depth-distortion (compact ray weights) + normal-consistency (rendered normal ↔ depth-gradient normal) losses, phased in after warm-up. SuGaR's explicit SDF-alignment regularizer and Poisson meshing are skipped: surfels are already surface-aligned by construction, and screened Poisson reconstruction is a large pure-Rust implementation with no crate support. Optional later upgrade if walls still ripple: PGSR-style single-view planar + multi-view geometric consistency losses.
- **Mesh extraction: TSDF fusion + marching cubes** (the 2DGS paper's own pipeline) over rendered median-depth maps from training views — both algorithms are simple, well-documented, and pure-Rust-friendly. Output: watertight-ish apartment mesh, decimated for collision (walk mode: gravity + wall collision) and exportable OBJ/GLB for measurement/floor-plan use.
- **Hand-rolled autodiff** (no Burn): backward is hand-derived WGSL anyway; remaining chain rule (SH, sigmoid/exp activations, ray-splat intersection Jacobian, pose grads) is bounded. Escape hatch: Burn glue in `gs-train` only, if stuck.
- **Video decode (hard-blocking — video is the only input):** software decode behind a `VideoDecoder` trait. H.264 ("Most Compatible"): evaluate NihAV (`nihav-itu`) vs `rusty_h264` (pure Rust, bit-exact vs Cisco h264dec, Baseline+B-slices+most High) and keep the winner. HEVC (iPhone "High Efficiency"): evaluate `rust_h265` (pure-Rust Main/Main 10 4:2:0, MIT/Apache, new 2026-07) on real iPhone .mov streams; Vulkan Video NVDEC (via `ash`) is the hardware-decode stretch. A frame-sequence reader exists **for internal test harnesses only** (dataset validation, decoder goldens) — not a product input. Do not confuse parser-only crates for decoders (`media-codec-h265`, `scuffle-h264/h265`, `h264-reader` parse headers; they produce no frames). iPhone notes: .mov demux (hvcC/avcC length-prefixed NALs → Annex B conversion); Dolby Vision clips decode via the backward-compatible HLG base layer but need HLG→SDR tone mapping before training; **iPhone video is Variable Frame Rate — never assume constant fps; carry per-frame PTS from the demuxer through everything temporal**.
- **Pose: video visual odometry, not SfM.** KLT flow tracking with motion prediction (no descriptor matching in the steady state), continuous-time SE(3) spline over PTS, sliding-window BA; once mapping starts, new keyframes are posed by photometric alignment against the current splat model. Loop closure (AKAZE-based place recognition + pose-graph relaxation, cheap candidate selection thanks to temporal ordering) only if apartment loops visibly ghost.
- **Incremental mapping over batch training:** grow the model as the camera walks; temporal minibatches with tile-bin/sort reuse across adjacent keyframes; short global refinement at the end. Rationale: each keyframe starts from an already-good model, dense depth init skips discovery, and redundant adjacent frames become a cache win instead of wasted work.
- **Densification: MCMC** — fixed splat budget = pre-allocated buffers, no realloc/compaction, direct VRAM/quality knob.
- **Precision:** f32 training (per-tile shared-memory gradient accumulation, then atomic flush via u32-CAS — WGSL has no f32 atomics); f16/quantized viewing.
- **One WGSL radix sort, two uses:** training (tileID‖depth 64-bit keys → per-tile ranges); viewer (32-bit depth-only global sort, web-splat style).
- **SH degree 2 default** (deg 3 runtime option) — apartments are diffuse; saves ~40% splat payload for VR/web.
- **Anti-aliasing:** Mip-Splatting-style filtering adapted to surfels (2DGS's object-space low-pass + screen-space 2D mip filter) from the quality milestone — prevents thin-surfel shimmer in VR.

## Milestones (ordered by risk; each ends in a runnable demo)

- **M0 — Render an existing .ply splat** (start here): gs-core/io/wgpu/render/viewer + `gs-cli view`. Accept: ≥120 FPS @1440p, ~1.5M splats; visually matches a reference viewer.
- **M1 — GPU primitives hardened:** radix sort, prefix sum, tile binning as tested standalone kernels. Accept: property tests vs CPU pass; 4M-key sort < 2 ms.
- **M2 — Differentiable 2DGS rasterizer (highest risk):** CPU analytic oracle first, then WGSL fwd/bwd — ray-splat intersection forward; hand-derived backward through intersection, opacity, SH, and pose/focal; depth + normal render targets. Accept: GPU↔CPU-analytic ≤1e-4 rel; analytic↔finite-diff ≤1e-2 rel per parameter class; 50-surfel image overfit PSNR > 35 dB.
- **M3 — Trainer validated on posed video sequences** (isolates trainer from pose errors; internal harness, not a product path): Adam-in-WGSL, SSIM kernels, posed-sequence loader (Replica renders / ScanNet-style). Accept: PSNR within ~1 dB of published 2DGS numbers on comparable indoor scenes; 30k iters ≤ 60 min.
- **M4 — Quality & geometry:** MCMC + mip filters + progressive SH + **depth-distortion & normal-consistency regularizers**. Accept: ≥ M3 PSNR at fixed 2M budget; no zoom shimmer; fewer floaters; rendered normal maps are clean on flat walls (visual golden + normal-vs-depth-gradient agreement metric); Chamfer vs Replica ground-truth mesh in the published 2DGS ballpark.
- **M5 — Video ingest** (moved up — it is now the only input): streaming MP4/.mov demux with per-frame PTS (VFR-safe), H.264 decode (NihAV vs `rusty_h264` bake-off), `rust_h265` iPhone HEVC validation (10-bit + HLG tone mapping) against reference decodes of real clips, color-correct YUV→RGB (golden frames), sharpness scoring.
- **M6 — VO front-end** (second risk item; parallelizable after M2): KLT tracking + motion prediction, five-point bootstrap, PTS-parameterized SE(3) spline, sliding-window BA, transient masks, keyframe promotion. Validate on video datasets with GT trajectories (TUM RGB-D mono / ScanNet / Replica walkthrough renders). Accept: ATE < 1% of trajectory, RPE < 0.5°; training on our poses loses < 1 dB vs GT poses, photometric refinement recovers ≥ half.
- **M7 — Video-native training, end-to-end on a real apartment video:** wgpu plane-sweep dense depth → surface-aligned surfel init; incremental mapping (track-against-model, spawn-in-new-regions, temporal minibatches with sort/bin reuse); global refinement pass; exposure spline. `gs-cli run walkthrough.mp4` → walk your own apartment. Accept: held-out-keyframe PSNR > 24 dB; **total pipeline ≤ 60 min for a 2-min 1080p video (stretch: model ready minutes after ingest ends)**; ablation logged showing dense-init + incremental beats naive batch on time-to-PSNR.
- **M8 — Mesh extraction & collision:** render median depth from training views → TSDF fusion → marching cubes → decimation; `gs-cli export --mesh` (OBJ/GLB); viewer walk mode (gravity + capsule-vs-mesh collision, toggleable vs fly mode). Accept: mesh of the M7 apartment has no holes bigger than a doorway on main surfaces; walk mode can't pass through walls/floor; mesh export opens correctly in Blender.
- **M9 — Web viewer:** wasm32 build, progressive .spz. Accept: ≥60 FPS @1080p in Chrome WebGPU, 1.5M splats.
- **M10 — VR:** OpenXR, shared cyclopean-eye sort, f16 packing, LOD, scale calibration; reuse M8 collision for roomscale comfort bounds. Accept: sustained 90 Hz; world scale ±5%.
- **M11 — Stretch:** live-growing preview during ingest (decode/track/map run concurrently — the incremental architecture already permits it), near-real-time reconstruction, Vulkan Video NVDEC (HEVC/iPhone), egui pipeline UI, loop-closure pose graph, foveation, PGSR-style planar/multi-view consistency losses if wall geometry needs another step up.

## Risks

| Risk | Mitigation |
|---|---|
| Silent wrong gradients / atomics races in WGSL backward | CPU oracle + FD checks in CI gate all kernel work; shared-mem accumulation; test Vulkan and DX12 backends |
| Decode is now hard-blocking (video-only input); `rust_h265` is v0.1, untested on real phone streams; Dolby Vision/HLG color | decoder validation front-loaded to M5 with golden-frame checks; H.264 capture guidance; internal frame-sequence harness keeps trainer/VO development unblocked if decode stalls; Vulkan Video in M11 |
| Pose drift on textureless walls / auto-exposure | trajectory-spline smoothness prior, capture guidance, exposure spline, photometric track-against-model refinement, held-out PSNR canary, pose-graph loop closure contingency |
| Incremental mapping bakes early drift/errors into the model (vs batch's global view) | sliding-window BA before surfels are spawned; global refinement pass at end; loop closure contingency; M7 ablation vs batch harness catches regressions |
| wgpu vs CUDA perf gap | subgroup ops, shared-mem accumulation; accept 2–3× (60-min M3 budget assumes it); profile from M1 |
| VR frame budget (2×90 Hz) | f16, SH deg 2, shared sort, LOD, resolution scale |
| 2DGS fidelity dip on fine/fuzzy detail (plants, fabric) vs 3DGS | Accept ~0.5 dB; apartments are planar-dominant; MCMC budget headroom; PGSR-style losses or per-scene SH deg 3 if needed |
| Solo-dev scope creep | every milestone ends in a `gs-cli` demo; one shared render crate for all three viewers |

## Verification

- **Gradient three-way agreement** (finite-diff ↔ CPU analytic ↔ GPU WGSL) on randomized micro-scenes, in CI forever.
- **Golden-image tests** (PSNR ≥ 45 dB tolerance, not hashes) on deterministic camera paths from M0.
- **Property tests** for sort/binning vs CPU.
- **Dataset benchmarks** on posed video sequences — Replica renders (PSNR/SSIM + Chamfer vs GT mesh), TUM RGB-D / ScanNet trajectories (pose ATE/RPE) — in a checked-in results log. These loaders are internal harnesses; the product input remains video only.
- **Time-to-PSNR ablations** (M7+): incremental + dense-init vs naive batch on the same scene — the video-native speed claims are measured, not assumed.
- **Geometry checks** (M4+): rendered-normal ↔ depth-gradient-normal agreement metric; visual goldens of depth/normal maps on flat-wall fixtures.
- **Failure isolation:** trainer validated on GT poses before our poses; ingest validated on golden frames before feeding the pipeline.
- **Perf gates** via wgpu timestamp queries (sort < 2 ms @4M, forward < 4 ms @1080p/2M, train iter < 100 ms).

**First action:** M0 — scaffold the workspace and get a public pre-trained .ply splat rendering through our own wgpu pipeline with WASD/mouse.
