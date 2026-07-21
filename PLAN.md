# Splattar — Architecture & Roadmap

> **Status (2026-07-21): M0–M6 complete; M7/M8 core landed, quality gates open.** Per-milestone numbers and the story behind each gate: [RESULTS.md](RESULTS.md). Working rules, commands, and gotchas: [CLAUDE.md](CLAUDE.md).

## Goal

A 100% Rust application that turns walkthrough video of an apartment into a high-definition Gaussian-surfel model you can explore first-person — desktop (WASD/mouse), VR (OpenXR), and a shareable web viewer (WASM/WebGPU). Dev machine: Windows 11 + NVIDIA GPU.

## Settled decisions

- **GPU via wgpu** (Vulkan/DX12/WebGPU compute) — no CUDA. **100% Rust, no C/C++ build deps**; system runtimes (GPU driver, OpenXR DLL) acceptable.
- **From scratch, as a learning project** — differentiable rasterizer (WGSL forward + hand-derived backward), Adam, densification all implemented here; Brush is reference reading only.
- **Representation: 2D Gaussian surfels (2DGS)** — ray-splat-intersection rasterizer, depth-distortion + normal-consistency regularizers, MCMC densification, mip-style anti-aliasing. Chosen over classic 3DGS because apartments are planar-dominant and surfels give view-consistent depth/normals → accurate surfaces, mesh export, collision. Surveyed and set aside: SuGaR (goals achieved the 2DGS way; TSDF+marching cubes instead of Poisson), 3DGRT/3DGUT (CUDA-bound), SVRaster, feed-forward VGGT/VicaSplat (below per-scene fidelity), PGSR losses kept as a stretch upgrade.
- **Video is the only product input — one or many videos.** The pipeline exploits temporal structure instead of reproducing unordered-photo SfM + random-view batch training. Architecture is SLAM-shaped: tracking front-end → incremental submap mapping → Sim(3) graph alignment → short global refinement.
- **The scene is a graph of submaps** (bounded drift, multi-video sessions, pipeline overlap). Posed-frame dataset loaders exist only as internal validation harnesses.

Highest-risk items were front-loaded: (1) hand-derived WGSL backward — done, M2; (2) pure-Rust VO — done, M6; (3) pure-Rust video decode — done, M5 (hardware).

## Workspace

```
Cargo.toml                # [workspace], shared workspace.dependencies
crates/
  gs-core      # splat SoA structs, SE(3)/quat math, camera model, SH eval (glam, bytemuck, half; wasm-safe)
  gs-io        # project format + export writers (.ply 3DGS/surfel, .spz, scene-manifest JSON), dataset harnesses
  gs-wgpu      # device init, buffer pools, radix sort, prefix sum, ReadbackRing/FramePacer, timestamp bench harness
  gs-kernels   # WGSL training kernels: project/cull, tile binning, ray-splat fwd + HAND-DERIVED BWD, SSIM, Adam, MCMC, appearance
  gs-cpu-ref   # slow obviously-correct CPU fwd/bwd + finite-difference harness (test oracle only, never ships)
  gs-train     # training loop, losses, MCMC controller, pose/focal refinement, appearance fit, eval
  gs-video     # MP4/.mov demux (PTS everywhere, rotation, B-frame reorder) + Vulkan Video H.264/H.265 decode + YUV→RGB/tone map, sharpness, keyframe promotion
  gs-pose      # VO: pyramidal KLT, five-point bootstrap, anchor-out solve + segmentation, sliding/global BA (faer sparse Schur), steered-BRIEF descriptor DB, Sim(3) registration; nalgebra+faer quarantined here
  gs-render    # forward-only viewer pipeline: depth sort, packed f16, SH 0–3, LOD; loads 3DGS and surfel layouts
  gs-viewer    # FPS camera, walk/fly, render-loop orchestration (surface-agnostic); frame player
  gs-cli       # headless driver: add / view / pose / play / register / refocal / train / export — primary dev entry point
  app-desktop  # winit + egui shell        app-vr  # OpenXR + Vulkan interop        app-web  # wasm32 + WebGPU
assets/        # golden images, fixtures   samples/  # gitignored user data (see README)
```

Design rules: training rasterizer (`gs-kernels`) and viewing rasterizer (`gs-render`) are separate pipelines sharing `gs-wgpu` infra; everything runs headless via `gs-cli` before any GUI; `gs-core` compiles everywhere including wasm32.

## Pipeline

1. **Streaming ingest:** demux + hardware decode with per-frame PTS (VFR-safe, rotation baked, HEVC tone-mapped); sharpness scoring; keyframe *promotion* (median flow 15–40 px + sharpest-in-window) — non-keyframes still feed tracking.
2. **Tracking front-end (VO), two-pass:** the **causal pass** runs once in decode order — pyramidal KLT with constant-velocity prediction, sharpness/PTS bookkeeping, and a zoom/lens-switch signal (coherent radial flow) — storing tracks; **geometry runs anchor-out** from the best-conditioned window, not in decode order. Five-point RANSAC bootstrap; sliding-window + global BA; flow-inconsistency masks for transients. Continuity breaks recurse into per-segment solves, each its own gauge.
3. **Submap segmentation:** overlapping submaps (~20–40 s), boundaries at low-quality moments; a zoom/lens switch **forces** a boundary — each submap owns an intrinsics group. Each submap owns its trajectory, surfels, appearance state, and gauge; keyframe descriptors enter a project-wide relocalization DB.
4. **Dense surfel initialization:** small-baseline plane-sweep depth → surfels born on surfaces with normals from plane fits (biggest convergence lever; designed, not yet built).
5. **Incremental mapping (per submap, anchor-out):** new keyframes tracked photometrically against the current model; surfels spawn in newly revealed regions; temporal minibatches reuse tile bins/sorts (designed; current trainer is per-submap batch over selected views).
6. **Registration & graph alignment (deferred, continuous):** every keyframe matches against the relocalization DB; matches → Sim(3) (7-DOF, per-submap scale) → verification ladder (temporal bridge → covisibility-voted global → pairwise image matching + epipolar verify) → pairwise edge in the project graph; components merge when connected. Photometric Sim(3) polish is the planned final stage.
7. **Global refinement (short):** all keyframes; MCMC budget; progressive SH 0→2; joint pose/focal refinement; per-submap affine appearance splines + cross-submap harmonization at merges.
8. **Alignment & outputs:** per component — floor-plane RANSAC y-up + metric scale from a reference measurement → mesh (median depth → TSDF → marching cubes) → archipelago layout for unconnected components → **export = baked snapshot** (.ply/.spz + scene-manifest sidecar).

**Multi-session model:** `gs-cli add <video>` is the only ingestion command; the project (default `gs-project` next to the video) persists submaps in local coordinates with pairwise Sim(3) edges — no privileged gauge, add order irrelevant. New videos build unconditionally as floating islands; unconnected islands are a normal state, shown side by side (**archipelago view**) as the coverage diagnostic that tells the operator what bridge footage to film. CPU stages (decode/track/VO/relocalize) pipeline ahead across videos; the GPU trains submaps serially from a ready queue.

## Key decisions

- **Single training rasterizer, 2DGS only.** Ray-splat intersection (perspective-correct, no EWA approximation) gives per-pixel depth/normals free. No parallel 3DGS training path. The viewer loads both layouts.
- **Mesh: TSDF fusion + marching cubes** over rendered median depth — pure-Rust-friendly; decimated for collision (walk mode) and OBJ/GLB export.
- **Hand-rolled autodiff — no Burn.** Backward is hand-derived WGSL; the remaining chain rule is bounded. Escape hatch: Burn glue in `gs-train` only.
- **Pose: video VO, not SfM.** Loop closure is not a separate subsystem — revisits register as submap constraints through the same relocalization machinery.
- **Anchor-out submap building:** halves worst-case drift at segment ends (where seam constraints attach), avoids bootstrapping on boundary-quality footage, gives both growth directions a warm map, and confines mid-segment failures to one tail. Decode stays forward-only; the causal pass is decoupled from geometry order.
- **Video decode: hardware (Vulkan Video/NVDEC via ash) — settled in M5, HEVC landed after.** Software bake-off failed across the board (CAVLC-only / I-only / AGPL). H.264+H.265 parsing lives in `gs-video`; the GPU driver is a permitted system runtime. VFR: always PTS, never frame_index/fps.
- **Precision: f32 training (CAS-accumulated gradients, subgroup pre-reduction where available), f16/quantized viewing.**
- **One WGSL radix sort, two uses:** training (tileID‖depth 64-bit) and viewer (32-bit depth-only per frame).
- **SH degree 2 default** (deg 3 runtime option) — apartments are diffuse; ~40% payload saving for VR/web. Progressive unlock during training.
- **Motion blur: model in the trainer (BAD-Gaussians-style exposure-integral rendering), don't pre-deblur.** Backlog after M8; needs per-frame exposure time.
- **Appearance & intrinsics are time-varying:** per-submap affine color splines (auto-exposure + AWB) applied in the loss only; per-submap intrinsics groups with zoom/lens-switch detection forcing boundaries. Local HDR tone mapping is explicitly not inverted in v1.
- **Scheduling: serial GPU training, pipelined CPU preparation.** Training saturates GPU bandwidth and wgpu has one queue — concurrency buys nothing and costs VRAM; build order is correctness-irrelevant (the graph is order-independent).
- **Two-tier data model: appendable project (source of truth, submap-local coords + Sim(3) edges), baked lossy exports** (never re-ingested). The scene-manifest sidecar carries island bounds/spawn points for our viewers; external viewers ignore it.
- **Splat formats:** native surfel .ply (2 scales), compat 3DGS .ply (zero third scale), .spz (gzip-era via pure-Rust flate2; SPZ 4 write blocked on a pure-Rust ZSTD encoder). Stretch: glTF KHR_gaussian_splatting.

## Milestones (ordered by risk; each ends in a runnable demo)

- ✅ **M0 — Render an existing splat file.** ≥120 FPS @1440p on `cactus-high`, visual parity with a reference viewer. *190 FPS, 0.36 s load.*
- ✅ **M1 — GPU primitives.** Sort/prefix/binning property-tested vs CPU; 4M-key sort < 2 ms. *1.68 ms.*
- ✅ **M2 — Differentiable 2DGS rasterizer (risk #1).** CPU oracle → WGSL forward → hand-derived backward incl. pose/focal grads. Gates: GPU↔CPU ≤1e-4 rel, analytic↔FD ≤1e-2, overfit >35 dB. *FD↔analytic 4.4e-7; overfit 39.6 dB.*
- ✅ **M3 — Trainer on posed sequences** (isolates trainer from pose errors). *Synthetic 28.4 dB held-out from cold init; published-parity + wall-clock gates deferred to a real-dataset run (needs densification by design).*
- ✅ **M4 — Quality & geometry.** MCMC + progressive SH + geo regularizers, all gradient-verified; TDR-safe stepping. *All-features-on matches baseline; mip filters beyond low-pass and Chamfer deferred to M9.*
- ✅ **M5 — Video ingest.** Streaming demux (PTS/VFR), hardware H.264 **and H.265/HEVC** decode (B-frames, tiles, 10-bit tone map, display rotation), keyframe promotion; `play` sanity stepper.
- ✅ **M6 — VO front-end (risk #2).** Causal pass, five-point bootstrap (replaced fragile eight-point), anchor-out solve + segment recursion, sliding + global BA (faer sparse, rayon-parallel, deterministic). Gates: ATE <1%, RPE <0.5°. *0.91% / <0.5° synthetic; full 3,605-kf real clip solves; deferred: TUM/ScanNet benches, transient masks.*
- 🔶 **M7 — Video-native training end-to-end.** `gs-cli add walkthrough.mp4` → walkable splat. Accept: held-out PSNR > 24 dB; ≤ 60 min for a 2-min video. *Landed: VO → view selection → training with pose+focal refinement, GPU appearance compensation, geometry losses; trainer restructured submit-only (~45×, 100+ it/s). Best 21.2 dB (sync trainer); async trainer plateaus 16.5–17.8 dB — stale-refinement lag is the open lead, then dense init and incremental mapping (still designed-only).*
- 🔶 **M8 — Multi-video & patch sessions.** Accept: overlapping videos merge with no ghosting; non-overlapping render as archipelago; a bridge video merges components; exports never re-ingested. *Landed: order-independent projects (pairwise Sim(3) edges, per-component placement), registration ladder (temporal bridge → covis-voted global → pairwise image stage with five-point verify), steered-BRIEF descriptors, `register`/`refocal` labs. Open: cross-video merge is data-limited at 10–16 verified matches/pair — next lever full-res pairwise matching; then color harmonization.*
- **M9 — Mesh & collision.** TSDF → marching cubes → decimation per island; `export --mesh`; walk mode with capsule collision. Accept: no doorway-sized holes; can't walk through walls; opens in Blender.
- **M10 — Web viewer.** wasm32 + progressive .spz streaming + manifest support. Accept: ≥60 FPS @1080p Chrome WebGPU, 1.5M splats.
- **M11 — VR.** OpenXR interop, stereo shared sort, f16/LOD/resolution scaling, scale calibration. Accept: sustained 90 Hz; world scale ±5%.
- **M12 — Stretch.** Live-growing preview during ingest; deeper CPU/GPU overlap; egui pipeline UI; foveation; PGSR losses; glTF export; SPZ 4 write.

## Risks

| Risk | Mitigation |
|---|---|
| Silent wrong gradients / atomics races in WGSL | CPU oracle + FD checks gate all kernel work; both subgroup and scalar accumulation paths tested |
| Decode coverage (non-NVIDIA / older GPUs lack Vulkan Video) | capture guidance; image-folder harness as internal fallback; decode layer is session-generic |
| VO poses cap training quality (monocular, gauge drift) | photometric pose+focal refinement (landed); pose-aligned eval; `refocal` re-BA; track-against-model refinement later |
| Async trainer refinement lag (pose/appearance fits applied to advanced state) | open M7 lead — bound staleness or sync fit points; ablations are cheap at 100+ it/s |
| Submap seam misalignment → double walls | photometric Sim(3) polish; overlap dedup in joint refinement; seam golden tests |
| Cross-video registration data-limited (10–16 matches/pair) | full-res pairwise matching next; covis voting + five-point verify already in place; coarse-register + photometric polish as fallback |
| Auto-exposure/AWB and cross-session lighting | per-submap affine appearance splines (landed) + harmonization at merge; local HDR tone mapping explicitly not inverted v1 |
| EIS crop/warp, focus breathing, rolling shutter | absorbed by per-submap focal refinement + spline smoothness; explicit modeling out of scope v1 |
| A video never registers | not a failure: archipelago view is the coverage diagnostic; bridge video merges later; no-false-positive gate on relocalization |
| wgpu perf gap vs CUDA | subgroup ops, shared-mem accumulation, timestamp profiling; measured 100+ it/s at working resolutions |
| VR frame budget | f16, SH deg 2, shared per-frame sort, LOD, resolution scale, foveation stretch |
| 2DGS fidelity dip on fuzzy detail | accept ~0.5 dB; MCMC headroom; PGSR losses or SH deg 3 per scene if needed |
| Solo-dev scope creep | every milestone ends in a runnable `gs-cli` demo; escape hatches documented, not taken by default |

## Verification

- **Gradient three-way agreement** (FD ↔ CPU analytic ↔ GPU) on randomized micro-scenes per parameter class — reruns on any kernel edit, both subgroup paths.
- **Golden-image tests** (PSNR ≥ 45 dB vs `assets/golden/`, not byte hashes) on deterministic camera paths.
- **Property tests:** sort order/stability, binning coverage vs CPU.
- **Dataset benchmarks** (internal harnesses): Replica/Mip-NeRF360 PSNR/SSIM; TUM/ScanNet ATE/RPE — logged in RESULTS.md.
- **Relocalization benchmark:** ≥90% success on held-out overlapping clips; zero false positives on non-overlapping.
- **Seam golden tests** (M8+): no double walls across submap boundaries.
- **End-to-end canary:** fixed real walkthrough clips; held-out PSNR + wall-clock tracked per change; `add` e2e gates (segment bridging, add-order independence) in `gs-cli/tests`.
- **Perf gates** (timestamp-query harness): sort < 2 ms @4M keys; viewer forward < 4 ms @1080p/2M; train iteration < 100 ms (currently ~10 ms).
- **Failure-source isolation:** trainer on GT poses, VO on GT trajectories — never debug both against the same failing run.
