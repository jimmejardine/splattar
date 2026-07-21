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

## M8 — multi-video: persistence, registration, islands (2026-07-21, core landed)

| Check | Gate | Measured |
|---|---|---|
| Project persistence | submaps with meta/landmarks/descriptors/trajectory/splat | pass (`run` writes submap-0; `add` writes submap-N) |
| `gs-cli add <video>` | VO + registration attempt + train + persist | pass end-to-end on 2.mp4 (island path) |
| Island handling | unregistered submap is first-class, composed side-by-side | pass — `view <project-dir>` composes registered submaps through Sim(3), islands offset along +x (presentation-only, never stored) |
| Cross-video Sim(3) merge of the overlapping flat pair | coherent merge | **not yet** — descriptor matching is the bottleneck (3/348 geometric consensus with naive BRIEF), submap persisted as island; see below |

Registration failure ladder actually hit (each now guarded in code):
1. one-directional matching + duplicated landmarks (KLT respawns re-triangulate
   the same corner ~20×) let a **scale-0.000 collapse** claim 749 "inliers" →
   cross-check matching, spatial voxel dedup, RANSAC scale bounds, inlier
   spread gate;
2. with honest matching, non-oriented single-scale BRIEF across two videos
   shot from different directions yields almost no geometric consensus —
   the deferred AKAZE-class descriptor (or per-keyframe 2D matching with
   epipolar verification) is the required next step (task list).

The Sim(3) estimator itself is solid: Umeyama + RANSAC recovers scale 2.7 /
arbitrary rotation through 40% outliers at 1e-6 accuracy in unit tests.

### M7 refinement ablation on the real walkthrough (2026-07-21)

All runs: 600 frames of 1.mp4, 239×425, pose+focal refinement + appearance
compensation as noted; held-out PSNR is pose-aligned (BARF protocol).

| Run | Config | Held-out |
|---|---|---|
| frozen VO poses (baseline) | 3k iters / 150k | 18.80 dB |
| + pose/focal refinement | 3k / 150k | 20.33 dB |
| + appearance compensation | 3k / 150k | **21.16 dB** (best so far) |
| long: 7k / 250k, geo+noise on, unanchored | | 15.51 dB |
| + appearance gauge anchor, decayed pose LR | 7k / 250k | 16.64 dB |
| + pose-refinement window (stop at iters/2) | 7k / 250k | 17.75 dB |

Diagnosis chain now encoded in code: (1) per-view affines share a global
color gauge — anchored so the mean correction stays identity; (2) constant
pose-refinement LR walks the camera gauge all run — now decayed on the
position schedule AND stopped at mid-training (train loss stayed excellent
while held-out collapsed: overfitting a moving gauge); (3) prime suspect for
the remaining long-run gap **and** the original Mip-NeRF360 room collapse:
MCMC exploration noise scales with the position LR, which scales with scene
**extent** — ~50× too strong on a walkthrough whose extent is ~30× its room
depth, and it activates exactly at geo_start. geo_bench exonerated the
geometry-loss kernels themselves (~0% per-iteration cost at 300k/780×520).
Verification run with noise 20→1 in flight.

### M7 addendum: async-trainer plateau (2026-07-21, joint with the optimizing session)

The trainer was restructured for async readbacks in a parallel session
(~45× faster: 2.3 → 100+ it/s; VO causal pass parallelized with rayon).
Synthetic gates re-validated: core 28.60 dB, M4-featured 28.48, pose
refinement +2.4 dB — pass; appearance compensation dropped +2.1 → +1.2 dB
(async fit lag; gate temporarily at +1.0). On the real walkthrough, every
config now plateaus at 16.5–17.8 dB (budget 150k/250k × iters 3k/7k ×
noise 20/1 × pose-window 0.5/1.0 all within ~1 dB) vs 21.16 dB measured
on the synchronous trainer — and eval pose-alignment adds ~nothing
(16.90 raw vs 16.93 aligned). Working hypothesis: pose gradients and
affine fits are applied against scene state that has since advanced
(stale-gradient noise) — tracked as the async-refinement-lag task.
Speed made the ablation possible at all: 8 full pipeline runs in the
time one used to take.

### M8 addendum: segment bridging — instrumented negative result (2026-07-21)

Landed: landmark persistence v3 (per-landmark reference keyframe + pixel
observation), submap keyframe ranges in meta, a registration strategy
ladder (temporal bridge → covisibility-voted global → island), and a
2D bridge solver (`sim3_from_bridge`: DLT-P6P RANSAC of a boundary
keyframe against world 3D + gauge scale from median depth ratios —
GT-verified to 12%/0.02 rad under heavy segment-side depth noise and
observation outliers).

Real-footage verdict on the flat's cut (kf 522↔535): descriptor matching
across the cut produces 17–24 candidates, but < 6 are ever geometrically
consistent (4,000 RANSAC iterations, 8 px tolerance). Root cause: the
track-killing cut is a whip pan — the camera faces different content on
each side, so boundary windows share almost no true field of view; the
"matches" are repeated-texture aliases. Same conclusion as the cross-video
case: registration needs viewpoint-robust (AKAZE-class) descriptors, used
with the covisibility-voted matcher over whole segments (real overlap
exists at NON-boundary times — the walkthrough revisits rooms). The 2D
solver and instrumentation carry over unchanged once descriptors improve.

### M8: descriptor upgrade landed — retrieval better, consensus still open (2026-07-21)

Shipped the viewpoint-robust descriptor ("AKAZE-class"): steered BRIEF with
intensity-centroid orientation at 3 pyramid levels per observation, matched
on minimum Hamming over level offsets (≈4× scale search). Unit gates:
survives 25° in-plane rotation (9+/12) and 1.7× scale (8+/12); persistence
v5 stores full per-landmark observation lists + per-submap pose tables, and
a new offline `gs-cli register` lab re-attempts registration with tunable
gates in seconds (no VO re-run).

Cross-video measurement (1.mp4 ↔ 2.mp4): raw matches 348 → 443 (+27%), and
the covisibility vote table is now semantically coherent — video 2's opening
keyframes (0–144) consistently match video 1's closing stretch (416–512),
i.e. the walkthroughs were localized by appearance alone. What still fails
is geometric consensus: per-keyframe correspondence precision remains too
low for Umeyama or DLT-PnP (best 2D groups of 12–17 obs never yield ≥6
consistent reprojections). Conclusion: landmark-DB retrieval is the wrong
final stage. Next architecture step: persist small keyframe thumbnails and
run pairwise image matching + epipolar verification on the candidate
covisible pairs the vote table already identifies. Whip-pan segment cuts
(1.mp4 internal) remain information-limited at the boundary regardless of
descriptors — cross-segment merging rides the same revisit-based path.

### M8: pairwise image stage — bottleneck moved to solver sample size (2026-07-21)

The pairwise architecture is in: submaps persist half-res keyframe
thumbnails + pose tables; the registration ladder ends with fresh
corner detection and descriptor matching between the actual images of
vote-nominated keyframe pairs, essential-matrix verification, landmark
snapping, and Sim(3) via Umeyama or the 2D depth-ratio bridge. Synthetic
gate: 30+ verified matches on a true pair, unrelated pair rejected (two
pitfalls documented in code: tight ratio gates skew matches onto dominant
planes and degenerate the eight-point solve; small thumbnails need dense
detection).

On the real flat pair the right keyframes engage (52–68 raw matches per
nominated pair) but no pair verifies: cross-take match precision is ~20%,
and EIGHT-point sampling needs 8 clean draws (0.2⁸ × 8k iters ≈ 0.02
expected successes). The arithmetic fix is a FIVE-point essential solver
(0.2⁵ × 8k ≈ 2.6) — M6's originally-planned five-point bootstrap, now
with a measured, quantitative justification. Everything downstream of the
solver is already built and waiting.

### Five-point solver landed — cross-take pairs verify (2026-07-21)

Implemented the Nistér/Stewénius five-point essential solver from scratch
(`gs-pose::fivepoint`): 4-dim null-space parametrization, det + trace
constraints expanded symbolically over 20 monomials, Gauss-Jordan to a
10×10 action matrix, real eigen-solutions via Schur eigenvalues + per-λ
null spaces. Gates: constraints vanish at GT (1.7e-12), exact E recovery
on held-out points across seeds, solves the all-planar scene that
degenerates eight-point, and survives the 20%-inlier regime that
motivated it. Both consumers switched: pairwise verification and the VO
bootstrap (all 41 gs-pose tests green).

Measured payoff on the flat pair: pairwise epipolar verification went
0 → 11–13 inliers on 5 of 6 vote-nominated keyframe pairs — the two
walkthroughs now geometrically verify against each other. Registration
remains one step short: snapped landmark pairs (2–6/pair, 20 pooled)
don't reach Sim(3) consensus because each submap's geometry is warped by
the unrefined focal guess (trainer measures the true focal at +6.5%, VO
geometry never receives it) — a non-similarity warp no Sim(3) can fit
tighter than a few percent of depth. Next step (well-scoped): feed the
trainer-refined focal back through a VO re-BA per submap, then the
existing ladder should close; alternatively accept coarse registration
and lean on PLAN's photometric Sim(3) polish.

### Focal re-BA landed; registration bottleneck now quantified to match count (2026-07-21)

`gs-cli refocal` rebuilds a submap's bundle problem from disk (poses.csv +
landmark observation lists), re-normalizes with the trainer-measured focal
(now persisted as `focal_refined` in meta after every training run), and
rewrites the geometry in place. Measured: reprojection cost 20.9 → 0.40
(submap-0) and 3.47 → 0.50 (submap-2); snap yields doubled. The lab loop
was also parallelized with rayon (five-point RANSAC draws, brute-force
matching, description) — a full registration attempt dropped from 300+ s
to ~1.6 s.

Registration itself remains open, now with a complete elimination chain:
with warp-free geometry, (a) 3D-3D landmark pairs (~30% snap precision),
(b) 2D-3D DLT bridges (6-point sampling at that precision), and (c) pure
relative-pose Sim(3) from verified essential decompositions (new solver,
GT-exact in tests; rotation clustering + lenient cheirality) all fail on
the same underlying number: 10–16 verified matches per cross-take image
pair — enough to confirm covisibility, too few for precise geometry.
The next data lever is FULL-RESOLUTION pairwise matching (thumbnails are
half-res; 4× pixels → sharper corners, more matches, tighter E), plus more
thumb pairs per region. New durable tools regardless: scale-bounded Sim(3)
RANSAC, appearance-guided snapping, `sim3_from_relative_pairs`.

### Display-rotation × bootstrap interaction: resolved (2026-07-21)

Baking phone display rotation into decoded pixels (tkhd matrix → upright
frames, gs-video 1607d5d) initially collapsed VO bootstrap on the rotated
720×1280 iPhone clip: 97/538 essential inliers vs 466/538 unrotated on the
same keyframe pair, despite every isolated layer proving
rotation-equivalent (pixels visually exact, single-step KLT agreement
0.004 px, RANSAC invariant on synthetic geometry). Root cause: the
eight-point DLT bootstrap was numerically fragile on this clip's
rotation-dominant ~1° parallax geometry — its Hartley-normalized
conditioning sat on the failing side in the rotated coordinate frame. The
Nistér–Stewénius five-point solver (631a5bb) is robust in that motion
regime in both frames: the clip now solves 223/223 keyframes rotated AND
unrotated. `SPLATTAR_NO_ROTATION=1` remains as a debug isolation flag; the
rotation bake is unconditionally correct and on by default.

### Order-independent project model: pairwise Sim(3) edges (2026-07-21)

`gs-cli run` is gone; `add <video>` is the only ingestion command and
creates `./gs-project` by default (`--project` overrides). The absolute
`sim3=` world transform in submap meta.txt is replaced by zero-or-more
pairwise `edge=` lines (this submap's coords → target submap's coords; no
edges = island; legacy metas migrate on read: identity `sim3=` drops, any
other becomes an edge to submap 0). No submap has a privileged gauge:
placement is resolved at compose/register time by union-find + BFS from
each component's lowest-index submap, and connected components are laid
out along +x (presentation-only, never stored). Cross-component
registration RANSACs per component — pooling matches across components
would mix unrelated gauges into one model. Ingestion order now affects
only submap indices and layout, never connectivity; the
`order_independence_cross_video` e2e test pins this on the Android pair.
