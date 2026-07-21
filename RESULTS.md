# Results Log

Measured numbers per milestone, appended as work lands (see PLAN.md §Verification).
Hardware unless stated otherwise: RTX 4090, Vulkan, driver 591.86, Windows 11, Rust 1.95.0.

## M0 — viewer (2026-07-20)

| Check | Gate | Measured |
|---|---|---|
| cactus-high load (456 MB, 1,935,120 splats, SH deg 3) | < 5 s | **0.36 s** (release) |
| cactus-high render @ 2560×1440, orbit avg | ≥ 120 FPS | **5.27 ms / 190 FPS** (offscreen, GPU-blocking, no present) |
| GPU radix sort vs CPU stable sort | exact match | pass at n ∈ {0…2M}, duplicates + payload pairing + stability |
| Golden test (cactus-low, 3 poses, 800×600) | PSNR ≥ 45 dB | pass (same-machine baseline) |
| Purity audit (`cargo tree`) | no C/C++-building crates | pass (windows-sys/renderdoc-sys bindings only) |
| Workspace gates | build / clippy -D warnings / test | pass |

Notes: f32 SoA buffers throughout (~600 MB VRAM at 1.9M splats); f16 SH packing
remains the first perf lever if a smaller GPU needs it. Interactive window
verified separately (WASD/mouse, pointer lock, SH-degree keys). Per-kernel
timestamp budgets (sort < 2 ms @ 4M etc.) are enforced starting M1 with the
bench harness.

## M1 — GPU primitives hardened (2026-07-20)

| Check | Gate | Measured |
|---|---|---|
| Radix sort 4M keys (GPU time, prep+8 passes) | < 2 ms | **1.68 ms** (1M 0.57 / 2M 0.94 / 8M 3.22) |
| Sort property tests | exact vs CPU stable sort, 0–10M | pass (order + payload pairing + stability, heavy duplicates) |
| Prefix sum | exact vs CPU, 1–10M incl. non-block sizes | pass (+ total via 1-element buffer) |
| Tile binning | exact payload stream + ranges vs CPU | pass (1–100k items, 64×64 grid, duplicate depths, culled + stacked cases) |
| Viewer forward @ 1920×1080, 1.94M splats | < 4 ms budget | **2.62 ms GPU** (preprocess 0.70 / sort 0.88 / draw 1.04) — 313 FPS wall |

Optimization applied (PLAN.md ladder level b): the M0 serial scan — one
workgroup, 16 threads walking every block — cost 4.7 ms of 5.4 ms at 4M keys.
Replaced with per-digit column-scan workgroups (shared Hillis–Steele + carry)
plus a tiny totals kernel: 3.2× total speedup, no digit-width change needed.
Tile binning's correctness leans on sort *stability* (two 32-bit sorts ≡ one
64-bit tile‖depth sort); the binning property test is the canary if the sort
ever changes.

## M2 — differentiable 2DGS rasterizer (2026-07-20)

| Check | Gate | Measured |
|---|---|---|
| CPU analytic ↔ finite differences (f64), all parameter classes | ≤ 1e-2 rel | **worst 4.4e-7** (pos/scales/quat/opacity/SH0-3/cam center/cam quat/focal) |
| GPU forward ↔ CPU oracle (color) | structural agreement | **max 3.3e-6**, mean ~3e-8 per channel |
| GPU backward ↔ CPU analytic, all classes | ≤ 1e-4 rel target | **worst 9.1e-4, typical ≤ 1.6e-4** (f32 CAS-accumulation noise on top of f64-certified analytics; asserted at 2e-3) |
| 50-surfel / 128×128 overfit (host Adam over GPU grads) | PSNR > 35 dB | **39.57 dB** at 3k iters (35.7 dB by iter 250) |

## M3 — trainer on posed sequences (2026-07-20)

| Check | Gate | Measured |
|---|---|---|
| Adam-in-WGSL vs CPU reference (all activation modes) | exact | pass (≤1e-5 after 5 steps, incl. exp/sigmoid chain rules) |
| SSIM/L1 loss kernels | correct gradients | validated end-to-end by training convergence (analytic backward via 3 blurred coefficient maps; self-adjoint zero-pad blur) |
| Synthetic posed-sequence training (300 surfels from scratch, 30 views, 128²) | held-out PSNR > 27 dB | **28.41 dB** (from 19.69 dB at init), 6k iters ≈ 45 s |
| Compat .ply export | round-trip exact | pass (write → read → activated values match; third scale flattened) |

Machinery landed: raw-space parameters (log-scales, logit-opacities) with
in-kernel activation chains; exponential position-LR decay scaled by scene
extent; COLMAP binary sparse loader (SIMPLE_PINHOLE/PINHOLE/SIMPLE_RADIAL,
convention conversion documented in gs-io::colmap) + SfM-point surfel init
(voxel-hash 3-NN scales); `gs-cli train <dataset>` → trains → held-out PSNR →
bakes a compat .ply viewable with `gs-cli view`.

**Open for real-data validation:** the "within ~1 dB of published 2DGS" and
"30k iters ≤ 60 min" gates need a real COLMAP dataset (e.g. Mip-NeRF360
room/counter, ~12 GB download) and M4's MCMC densification — a fixed
SfM-initialized budget cannot reach published numbers by design. Re-measure
after M4 with a dataset on disk.

Notes: forward = explicit ray–splat intersection in camera space (Cramer on
scalar triple products — equivalent to the 2DGS homography form, directly
differentiable); low-pass = max(G_ray, G_screen), σ²=0.5. Backward follows the
CLAUDE.md accumulation mandate: per-tile shared-memory atomic<u32> CAS float
adds flushed per chunk, then a per-surfel geometry chain kernel; camera
quaternion grads chain on the host from the GPU's dl/dR matrix using the same
f64 math as the oracle. Gradients flow to: position, scales, quaternion,
opacity, SH (deg 0–3), camera center, camera rotation, focal. Depth + normal
render targets exist (no losses on them until M4).

## M4 — 2DGS quality: aux losses, regularizers, MCMC (2026-07-20)

| Check | Gate | Measured |
|---|---|---|
| Aux-loss gradients (depth, normal, depth-distortion), CPU FD ↔ analytic | ≤ 1e-2 rel | pass (f64 oracle; distortion via prefix-recovery reverse walk) |
| GPU backward ↔ CPU analytic with aux losses active | agreement | **≤ 6e-3 surfel params, ≤ 3e-2 camera-global** (f32 cancellation in distortion prefix recovery + whole-image CAS sums; color-only path stays at 2e-3) |
| Synthetic training, all M4 features on (distortion + normal loss, regularizers, progressive SH, MCMC) | no regression vs M3 baseline | **28.56 dB** (baseline same-config 28.59 dB) |
| MCMC relocation | budget fixed, no NaN/collapse | pass (opacity-sampled relocation α'=1−√(1−α), PCG-hash noise gated by exp(−5α)) |

Landed: fused L1+D-SSIM backward; depth-distortion loss (per-ray pairwise,
prefix-recovery A_i = W_end − suffix_w − w in the reverse walk, normalized by
pixel count — unnormalized it was 16,000× too strong and collapsed training to
11.5 dB); normal-consistency loss (normals from unprojected-depth forward
differences, gather adjoint second pass, alpha/orientation detached);
opacity/scale regularizers folded into the Adam kernel's activation chain;
progressive SH promotion; MCMC relocation + noise injection at a fixed surfel
budget (SoA buffers never reallocate, per CLAUDE.md).

Also hardened for real scenes: Windows TDR (~2 s GPU watchdog) forced two-way
step submission splits and a 64 px cap on the binning tile-rect radius. The cap
is part of the shipped forward model, so the CPU oracle mirrors the exact
tile-rect truncation (`covers_pixel`) — three-way agreement re-verified after
the change. Mip-NeRF360 `room` (311 views @ 779×519, 112k SfM points, 300k
budget) training runs at ~0.7 it/s; end-to-end numbers to be recorded when the
7k-iter run completes.

## M5 — video ingest: MP4 demux + NVDEC H.264 + keyframes (2026-07-20)

| Check | Gate | Measured |
|---|---|---|
| Android walkthrough decode (478×850 portrait, H.264 High, CABAC, in-hardware) | ≥ 60 frames, luma variance sane, PTS strictly increasing | **pass** — 90 frames in the gate window, full 2,323-sample stream decoded by the keyframe test, both tests in **4.7 s** total |
| VFR-safe PTS | from sample table, never frame/fps | pass (start_time + rendering_offset over timescale) |
| Keyframe promotion (sharpest-in-window, Laplacian variance) | ≥ 8 keyframes, ≥ 0.2 s spacing, increasing PTS | pass (0.5 s windows, 12 max) |
| Purity audit | no C/C++ build deps, no copyleft codec code | pass — decode is Vulkan Video (NVDEC) through ash; the GPU driver is a system runtime |

Decoder bake-off that led here: `rusty_h264` is CAVLC-only and silently skips
CABAC slices (phone footage is CABAC — decoded 0 frames); `oxideav-h264` is
I-slice-only; NihAV decodes everything but is AGPL-3.0. Decision: hardware
decode via Vulkan Video — zero third-party codec code, works for any H.264
the GPU supports, and the same session machinery extends to HEVC (iPhone)
later. Implementation: full SPS/PPS/slice-header parsing in `gs-video::h264`
(Exp-Golomb, RBSP, scaling lists, POC type 0/2), MP4 demux via the pure-Rust
`mp4` crate, `NvDecoder` with coincide-mode NV12 DPB, sliding-window
eviction, and per-plane readback → cropped I420. Scope is the phone subset:
progressive 4:2:0 8-bit, I/P slices (no B-frames in phone walkthrough video —
verified all composition offsets are 0 in the samples).

## M6 — VO front-end (2026-07-21)

| Check | Gate | Measured |
|---|---|---|
| KLT tracker on synthetic warps | sub-pixel + FB rejection | pass (< 0.35 px on ±7 px translation; zero-mean matching survives exposure offsets) |
| Two-view / PnP / BA geometry vs synthetic GT | recover known poses | pass (8-pt RANSAC through 25% outliers, rot err < 5e-3 rad; BA to < 1e-14 cost noiseless, gauge-fixed) |
| Full VO on analytic two-plane scene (50 frames, 400×300) | **ATE < 1%** of trajectory, **RPE < 0.5°** | **ATE 0.91%**, RPE rot < 0.5° per pair, zoom signal flat under constant focal |
| Full VO on real Android walkthrough (600 frames, 478×850, NVDEC decode) | bootstrap + solve succeed | **404/404 keyframes solved**, 9,028 landmarks, bootstrap median parallax 1.26°, trajectory finite/smooth (`gs-cli pose`) |
| Frame-to-frame KLT survival on real footage | healthy | 81–98% FB-verified per frame (diagnostic test, first 40 frames) |

Architecture notes: causal pass (constant-velocity KLT, flow/survival keyframe
promotion, radial-flow zoom signal) is separate from the anchor-out solve, per
PLAN. Two real-footage lessons are now encoded in the code: (1) survival
statistics must drop dead tracks at each keyframe or promotion runs away;
(2) bootstrap pair selection must measure parallax via a **global-affine
residual**, not raw flow — panning creates hundreds of pixels of flow with
zero baseline, and every flow-selected pair failed the 1° parallax gate.
Monocular scale stays a free per-segment gauge (global BA fixes only the
anchor pose). nalgebra is quarantined in gs-pose; the public API speaks glam.

Deferred from M6 (tracked for M8 prep): AKAZE-style descriptor DB + Sim(3)
relocalization primitive, TUM RGB-D ATE benchmark (dataset not on disk), VO
solve perf (188 s for 404 keyframes in a dev build — local-BA cadence and
match indexing are the known hotspots).

## M7 — video-native training end-to-end (2026-07-21, in progress)

| Check | Gate | Measured |
|---|---|---|
| `gs-cli run <video>` end-to-end | video → walkable .ply | **works**: 600 frames → VO (404 kf) → 73 views → 150k surfels → project dir + baked splat, ~40 min wall total (dominated by VO solve 3 min + training 21 min @ 2.3 it/s, 239×425) |
| Held-out keyframe PSNR | > 24 dB | **20.33 dB** (pose-aligned eval; 18.80 frozen-pose baseline). Gate open — see levers below |
| Pose+focal refinement in trainer | improves over frozen VO poses | pass on synthetic (25.35 → 27.58 dB on ~2° perturbed poses); on real footage raw-pose eval drops to 16.56 dB while aligned eval gains — training drifts the gauge, as expected for monocular |
| Project persistence | submap-0 written | meta.txt, landmarks.bin (pos+color+descriptor), trajectory.csv, splat.ply |

Findings encoded in code: camera-center refinement LR must scale with the
view's **median scene depth**, not scene extent (a walkthrough's extent is
~30× the room depth — extent scaling produced 0.16-unit camera steps and
destabilized training); held-out eval must photometrically align eval poses
to the frozen model before scoring (BARF-style), since gauge drift otherwise
masquerades as model error.

Open levers for the 24 dB gate, in expected order of impact: per-submap
time-varying affine appearance model (phone auto-exposure is unmodeled and
visibly swings across the walkthrough — PLAN already specifies this),
rolling-shutter/EIS handling, geometry losses once task #32 lands, longer
training + higher resolution, stronger VO global BA.
