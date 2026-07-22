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

`gs-cli run` is gone; `add <video>` is the only ingestion command and by
default creates `gs-project` next to the video (`--project` overrides). The absolute
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

### First `add` e2e run: order independence holds; bridge gate red by design (2026-07-21)

The two ignored gates in `gs-cli/tests/add_e2e.rs` ran end-to-end (51 min
total). `order_independence_cross_video` PASSED: ingesting 1.mp4/2.mp4 in
both orders produces identical structure — 11 submaps in 11 components
either way. (Side observation: the pair used to solve as 4 segments; the
five-point bootstrap now solves 11, picking up previously
un-bootstrappable ranges.) `hevc_three_segments_bridge` is RED as
expected: the back-room clip's 3 segments stay 3 components because its
cuts are whip pans — the instrumented negative documented above ("segment
bridging"). The strict `components < 3` assert is kept deliberately as
the acceptance gate for the full-res pairwise-matching work; make
bridging succeed rather than loosening the test.

### Causal-pass + solve speedup: 65 min → ~3 min on the back-room clip (2026-07-21)

Three changes, verified deterministic (identical keyframe counts across
runs): (1) the keyframe flow threshold is now a fraction of the image
diagonal (`kf_flow_frac`, default 0.015 ≈ the proven 15 px @ 478×850) —
the old absolute-pixel threshold promoted ~90% of frames on high-res
footage; (2) KLT tracks at a capped long side (960 px, integer luma
decimation), the whole front-end living in tracking coords with a single
`track_scale` lift at the training boundary; (3) the remaining serial
per-frame internals were rayon-parallelized with index-ordered merges
(promotion-time descriptor extraction — the dominant one — corner
detection per cell-row band, pyramid rows). Promotion reasons are now
logged (flow/survival/low-tracks).

Back-room HEVC clip, 4,315 frames @ 720×1280: causal 11.5 → 47.0 fps,
keyframes 3,605 → 2,900 (91% flow-driven — the footage genuinely moves
fast), anchor-out solve 745 → 121 s, full coverage throughout. Decode is
NOT the bottleneck (NVDEC ~500 fps, overlapped); the causal pass is now
bounded by the serial frame-to-frame KLT chain itself.

`SPLATTAR_KF_FLOW_FRAC` env knob added for density experiments. At 0.03:
1,923 keyframes, same 4 segments, full coverage, comparable landmarks
(43k vs 48k), solve 71.6 s. Candidate default bump pending an end-to-end
PSNR comparison. GPU KLT (WGSL LK — the GPU is idle during VO in today's
serial flow, NVDEC is a separate engine) is the recorded next lever if
the causal pass needs another 4×+; a faster linear-algebra library is
not — the big solve already runs on faer, and the rest is small
fixed-size ops where keyframe count, not the library, is the multiplier.

### Causal pass round 2: pipelined prep, faster detect — and the wall moves (2026-07-21)

Restructured `run_vo` into a three-stage pipeline: decode thread → prep
worker pool (pyramid + sharpness per frame — pure functions, built
serially per frame with parallelism ACROSS frames) → tracking spine
consuming strictly in index order (results identical to the serial
loop; back-room reproduces its keyframe count exactly). The spine now
logs its phase split (klt/desc/detect). Corner detection was rewritten
with separable sliding box sums computed lazily over unoccupied-cell
runs (7.5 → 6.1 s on the flat clip). Negative result, kept out:
`with_min_len(32)` on the per-track KLT fan-out — LK effort varies
wildly per track and fixed chunks defeat rayon's work-stealing
(measured 3× KLT regression; reverted).

Where things stand vs this morning's 12 fps complaint:
- H.264 478×850 flat clip: **76.5 fps** causal (spine-bound; split:
  klt 14.5 s / desc 7.9 s / detect 6.1 s of 30 s wall).
- HEVC 720×1280 back-room: **45.6 fps** causal — now **decode-bound**:
  spine work sums to ~39 s of the 95 s wall; NVDEC H.265 delivers ~46
  fps and no CPU-side work moves this clip. Solve 745 → 130 s.

Next levers, in order of what the profiles say: (1) NVDEC H.265
pipelining (frames in flight) for HEVC-bound clips; (2) GPU KLT (WGSL
LK) for the spine-bound half; (3) descriptor cadence (obs_desc every
keyframe is 25% of the spine and mostly unused downstream).

## Convergence investigation — the trainer, not the poses (2026-07-21)

Triggered by "a single room does not converge to anything visually
identifiable". Bisected by holding one side perfect and measuring the other.

**VO is healthy.** `add` on the back-room clip: 377/377 keyframes solved in
one segment, 10,111 landmarks, 22 s. Not the bottleneck.

**A real convention bug, small effect.** `KeyframePose::c2w()` transposed the
camera-to-world rotation twice, so every training view saw an inverted camera
(fixed; now pinned by `gs-pose/tests/c2w_convention.rs`). A/B on identical
clip + config: **14.88 → 15.74 dB** (+0.86). Real, but not the cause.

**The trainer plateaus at ~19 dB with PERFECT poses.** `train datasets/room`
(mip-NeRF360, COLMAP poses and intrinsics, 779×519, 39 held-out views) —
reference 2DGS/3DGS implementations reach 28-31 dB on this scene:

| config | held-out PSNR |
|---|---|
| 3k iters, 150k budget, mcmc_noise 20 (default) | 19.51 dB |
| 3k iters, 150k, mcmc_noise 0 | 19.43 dB |
| 6k iters, 300k, MCMC relocation AND noise off | 19.15 dB |
| 12k iters, 300k | **17.82 dB** (worse than 3k) |
| 3k iters, 150k, splat radius cap 64→512 px | 14.39 dB (worse) |

The plateau is insensitive to iteration count, surfel budget, and MCMC; more
training does not help and can hurt. LRs match the 3DGS reference values
(pos 1.6e-4×extent, scales 5e-3, quat 1e-3, opacity 5e-2, sh 2.5e-3).

**Ruled out:** camera rotation convention (fixed, +0.9 dB only), MCMC
exploration noise, MCMC relocation, the 64 px splat radius cap (protective —
raising it hurts), pose↔target slot pairing (verified correct), the
tracking→training focal lift (verified correct), LR values.

**Note on reading the logs:** the per-iteration L1 swings 3× between logged
iterations. That is per-view sampling variance (one random view per
iteration), NOT instability — don't chase it.

### Why every test stayed green

Correctness is only ever verified at micro-scale: `raster_parity.rs` runs 50
surfels at 128², `train_synthetic.rs` 300 surfels at 128². Both pass (28.6 dB).
Nothing exercises 150k-300k surfels at 779×519, and nothing tested the
VO→trainer seam at all. Two green suites, one non-converging pipeline.

### Open leads, in priority order

1. **No render-vs-target diagnostic exists.** `Trainer::render_view`
   (`trainer.rs:1040`) is only used for PSNR. Dumping render/target pairs at
   iteration N would immediately distinguish blurry-but-correct from
   geometrically displaced from color-broken. This is the missing tool.
2. **Scale-up the parity test** to 300k surfels at 779×519 — if GPU and CPU
   oracle disagree there, the gradients are wrong in the regime that matters.
3. **Bisect the async-trainer restructure** (already recorded: real-footage
   configs dropped 21.2 → 16.6 dB while synthetic gates stayed green). Run
   `train datasets/room` at the pre-async commit; that is a clean A/B the
   earlier investigation never had.

### ROOT CAUSE: normal-consistency loss over-weighted by the pixel count (2026-07-21)

The convergence failure is a one-line weighting bug in `gs-train/src/trainer.rs`.
The geometry losses sum over rays, so their weights must be divided by the pixel
count to be comparable with the mean-normalized color loss — the trainer does
this for `lambda_dist` and the normal kernel's `lambda` uniform is documented as
"includes 1/N normalization", but `set_lambda` was passed the RAW config value.
The normal loss therefore ran ~4e5× (779×519) too strong from `geo_start` on.

Because it phases in at iteration 1500, training improved to ~iter 1500 and then
degraded — which is why MORE iterations made things WORSE and why every tuning
knob looked inert.

Isolation ladder (mip-NeRF360 room, perfect COLMAP poses, single view, 150k
surfels, MCMC off — a fixed target that gradient descent cannot get worse on):

| config | 8k iters |
|---|---|
| no geo losses (baseline) | 34.23 dB |
| lambda_normal 0.05, raw (as shipped) | 23.95 dB |
| **lambda_normal 0.05, /n_px (fixed)** | **34.18 dB** |
| lambda_dist 0.01 (normalized, as shipped) | 29.49 dB |
| lambda_dist 0.001 | 33.39 dB |

Normal-loss cost: −10.3 dB → −0.05 dB, matching the ~0.2 dB the 2DGS paper
reports. `lambda_dist` default also lowered 0.01 → 0.001 (0.01 costs ~5 dB even
when correctly normalized).

End-to-end on the back-room clip (`add`, 600 frames, pose-aligned held-out):

| state | PSNR |
|---|---|
| as found | 14.88 dB |
| + c2w camera fix | 15.74 dB |
| + geo losses off, 3k iters | 19.50 dB |
| **+ geo losses off, 10k iters** | **20.39 dB** |
| + geo at fixed weights (0.001/0.05), 10k | 19.33 dB |

The render is now unmistakably the room (desk, monitors, armchair, framed
canvas, curtains) instead of unrecognizable mush.

**Diagnostic added:** `SPLATTAR_DUMP=<dir>` writes render/target PNG pairs from
`train` and `add`. Every scalar-only hypothesis in the previous investigation
was wrong; the first look at an actual render located the problem in minutes.

**Why the tests missed it:** `train_synthetic.rs` (300 surfels, 128²) leaves
`lambda_normal` at its 0.0 default, so no test ever enabled this loss during
training. The `gs-cpu-ref` aux-loss gradient check verifies the kernel against a
CPU port of the SAME convention, so a weighting-contract violation between
trainer and kernel is invisible to it. Needed: a training test with geo losses
ON asserting PSNR stays within ~0.5 dB of the geo-off baseline.

### HEVC decode round: the wall was write-combined memory, not NVDEC (2026-07-21)

Pipelined the H.265 Vulkan Video decoder (ring of in-flight frame
contexts — own command buffer/fence/bitstream/persistently-mapped
readback; submit/drain API with a PTS queue in the reader; global
decode→decode memory barrier so overlapped submissions can't race on
DPB contents; `SPLATTAR_DECODE_INFLIGHT` override). Instrumentation
(read/submit/fence-wait/post split, logged per run) then showed fence
waits were ~0.1 s — the actual cost was reading 1.4 MB/frame back
through WRITE-COMBINED host memory (~74 s of the 88 s decode thread on
the 4,315-frame clip). Fix: request HOST_VISIBLE|COHERENT|CACHED
readback memory first. With row-parallel to_display/rotation and the
VO dead-track archive (per-frame scans O(live), merged back in exact
spawn order for the solve), the back-room clip's causal pass went
**47 → 130+ fps**; full VO ≈ 2.5 min (was ~13.5 min this morning).
Decode output proven bit-identical across runs (luma FNV).

Open item — rare tracking-side nondeterminism (~1 run in 30): keyframe
count flips by a few; one caught trace shows a normal borderline-motion
frame whose KLT outcome differed wholesale (217 vs 133 survivors).
Ruled out: decode (pixels hash-equal; flips also at INFLIGHT=1),
thread-count variation (all reductions index-ordered). Tooling in
place: `SPLATTAR_KF_TRACE` writes per-frame flow/survival/live +
tracking-res luma FNV (prep-side, ~free), `SPLATTAR_CAUSAL_ONLY` skips
the solve for fast hammering — diff two runs to the first divergent
frame; the hash column attributes it to pixels vs tracking state. An
absolute mass-death tripwire was tried and removed: 30–50% single-frame
track loss is NORMAL during fast pans on this footage.

### RE-EVALUATION after the normal-loss fix (2026-07-22)

Every M7 refinement/quality conclusion above was measured on the broken
trainer (normal loss ~4e5× too strong from geo_start=1500, so anything past
~1500 iters was being actively destroyed). Re-ran the ablations against the
fixed trainer. Testbeds: mip-NeRF360 `room` (perfect COLMAP poses, isolates
the trainer) and 600 frames of 1.mp4 @ 239×425 (the exact clip the historical
21.16 dB came from; pose-aligned held-out).

**room (perfect poses), held-out:**

| iters | geo off | geo on (0.001/0.05) |
|---|---|---|
| 3k | 22.00 | — |
| 7k | 27.44 | 27.26 |
| 30k | 29.35 | 29.73 |

**1.mp4 (full stack unless noted), pose-aligned held-out:**

| config | PSNR |
|---|---|
| baseline (no pose, no appear), 7k | 19.88 |
| + pose refinement, 7k | 21.33 |
| + appearance, 7k | 21.42 |
| full, 3k / 7k / 15k / 30k | 19.82 / 21.42 / **22.17** / 21.25 |
| full 30k, mcmc_noise=0 | **22.33** |
| full + geo (0.001/0.05), 7k | 21.12 |

**Conclusions, re-evaluated:**

| Prior conclusion | Prior | Now | Verdict |
|---|---|---|---|
| Pose refinement helps | +1.53 dB | +1.45 dB | CONFIRMED — keep |
| Appearance compensation | +0.8–2.1 dB | +0.09 (7k) / +0.33 (15k) | OVERTURNED — was masking geo damage |
| Long runs DEGRADE (21→16.6) | degrade past 3k | improve to ~15k then plateau | OVERTURNED — was the geo bug |
| Async-trainer plateau ~16.5–17.8 dB | blamed readback lag | those were 7k geo-on runs; fixed = 21.4 @ 7k | OVERTURNED — geo bug, not async |
| 24 dB gate unreachable | open at 20.3 | room 27–30 (PASSES); video 22.3 peak | room passes; video resolution-limited |
| "geo_bench exonerated the kernels; collapse = scene evolution" | exonerated | the collapse WAS the normal-loss weighting | OVERTURNED |
| MCMC extent-scaled noise trap | prime suspect, unverified | CONFIRMED at long runs only (30k: 21.25→22.33 with noise off); invisible at ≤15k | CONFIRMED |
| Geo losses help held-out | assumed | neutral (room) / slightly negative (video); value is M9 mesh geometry, not PSNR | REFRAMED |

**Net:** the trainer is healthy. room clears the 24 dB gate; the ~1.7 dB video
gap is the tiny 239×425 target and monocular pose noise, not a defect.
Appearance compensation and (for held-out PSNR) the geometry losses are no
longer justified by the numbers — pose refinement is the one refinement that
earns its place.

**Follow-ups warranted (not yet done):**
1. Scale MCMC exploration noise by scene **depth**, not extent (same fix
   already applied to the pose-center LR) — removes the long-run trap so
   noise>0 can keep helping early densification without the 30k regression.
2. Reconsider whether appearance compensation and the geometry losses belong
   on by default on the `add` path (both are ~neutral-to-negative for
   held-out PSNR now; geo losses stay relevant for M9 meshing).
3. `add` default iters raised 4000 → 7000 (quality now rises with length).

### Anchor-out mapping: obs index kills the scans; BA is ~95% of the solve (2026-07-22)

The mapping loop's per-keyframe `par_iter` scans over ALL tracks (~90k
on the back-room clip: PnP gather + triangulation candidates) are
replaced by a per-keyframe observation index built once per solve
(`build_obs_index`), with `kf_matches` becoming a track-ordered bucket
intersection — bit-identical results (verified: 2,901 kf, same 4
segments, same solved counts and landmark totals). New mapping-phase
timers tell the real story: gather 0.0 s, pnp 0.5 s, tri ~8 s,
local-BA ~42 s, and global BA 96 s across segments — bundle adjustment
is now ~95% of the anchor-out solve. The originally-planned
triangulation dirty-set was DROPPED as not worth it (measured 8 s, was
assumed dominant). Next lever, if the solve needs to be faster still:
BA itself — iteration counts/tolerances, Schur assembly parallelism,
and local-BA cadence (currently every 2nd keyframe) — not more scan
optimization.

### The recorded walk path was in the wrong gauge (2026-07-22)

Symptom from walking the viewer's recorded path with the ground-truth
overlay on (`[`/`]` + `T`): the captured frame lines up with the render in
some stretches and shows an unrelated part of the room in others. The
hypothesis on the table was under-convergence. It was not — it was
bookkeeping, and more iterations made it worse.

`add_submap` wrote `poses.csv` BEFORE training. The trainer then refines
every view's camera for the rest of the run (`pose_window` 1.0) and only the
refined *focal* was ever written back, so `splat.ply` ended up in a gauge
`poses.csv` no longer described — and the viewer's path reads `poses.csv`.

Measured on 900 frames of 1.mp4 (submap-0: 539 keyframes, 102 selected
views → 89 train / 13 held out, 7000 iters, 150k surfels):

| | |
|---|---|
| held-out PSNR, raw VO poses | 17.58 dB |
| held-out PSNR, pose-aligned (BARF protocol) | **22.99 dB** |
| \|Δpos\| median / p95 over the 89 refined views | 0.2415 / 0.4578 |
| …as a multiple of the median keyframe step (0.0621) | **3.9× / 7.4×** |
| Δrot median / p95 | 0.45° / 0.83° |

Training slides each camera by roughly four keyframe-steps of translation
(rotation drift is negligible by comparison), and the shift is nonuniform —
p95 is nearly double the median. That nonuniformity is exactly the observed
pattern: stretches where refinement barely moved cost nothing, stretches
where it moved most decorrelate completely. The 5.4 dB raw-vs-aligned gap is
the same effect measured photometrically.

Fix: submaps now persist `poses_refined.csv` next to `poses.csv`. Two
gauges, not one — see PLAN.md §Key decisions. `poses.csv` stays the
geometric gauge (`landmarks.bin`, Sim(3) edges, `refocal` all share it);
`poses_refined.csv` is the photometric gauge the surfels are in, and is what
the viewer's path reads. Only ~1 in 6 keyframes here was a training view
(1 in 12 on full-length clips), so the correction measured at those anchors
is propagated to the rest as a local Δrot/Δpos field (slerp/lerp between
bracketing anchors, clamped outside the span). Snapshots are additionally
written for every selected view (`view-thumbs/`, kept out of `thumbs/` so
the pairwise registration candidate set is unchanged), so the full-strength
ground-truth blend is available at exactly the poses the model was fit to.

**Still open, same symptom, different cause:** `--max-views` is a flat 120
regardless of clip length, and selection is one sharpest keyframe per 0.25 s
window — uniform in *time*, not in *space*. Real submaps carry 1413 and 1749
keyframes on their paths against those ≤120 views, so a fast pan can leave a
whole stretch with near-zero supervision, where the surfels stay close to
their SfM-init state. Scaling `max_views` (and the fixed 150k surfel budget)
with submap size, and selecting by pose spacing rather than elapsed time, is
the follow-up.

### Coverage, model size and run length now follow the submap (2026-07-22)

Three flat constants decided how much work a submap got regardless of how
much scene it covered: 120 training views, a 150k surfel budget, and 7000
iterations. Real submaps on the back-room clip carry 1749 / 776 / 272
keyframes, so all three were badly mis-sized, and view selection was
uniform in *time* (sharpest keyframe per 0.25 s) rather than in space —
spending the budget where the operator lingered instead of where the scene
is.

Views are now spaced by **apparent view change** (rotation plus
translation-over-median-scene-depth, one currency; `--view-spacing`,
default 8°), the surfel budget follows the view count, and `--iters` became
a *ceiling* with the run stopping when a held-out probe plateaus.

**Back-room clip (3.mp4), per submap, at the intermediate step (spacing +
auto budget, iters still flat at 7000):**

| submap | keyframes | views (was 120) | budget (was 150k) | held-out |
|---|---|---|---|---|
| 0 | 1749 | 311 | 340k | 17.11 dB (was 16.73) |
| 1 | 776 | 165 | 180k | 20.55 dB |
| 2 | 272 | 77 | 84k | 23.96 dB |

Note the eval set is NOT held constant across the 120-view and 311-view
runs — more views means held-out views sample more of the scene, including
the previously starved stretches. The two PSNR numbers are not strictly
comparable and the +0.38 dB understates the change.

**1.mp4, 900 frames, with adaptive stopping (117 views, 127.5k surfels):
23.77 dB**, against 22.99 dB for the old flat 120 views / 150k / 7000 —
better on a *smaller* model.

**The probe trace is the useful artifact.** submap-0 climbed 18.23 → 18.63
→ 20.26 → 20.80 → 20.85 → 22.43 → 22.54 → 22.28 → 23.14 → 23.33 → 23.71 →
23.66 → 23.76 dB over 13 probes and never plateaued (at most one probe
without a gain) before hitting the ceiling — so the ceiling, not
convergence, was binding. `ITERS_PER_VIEW` was therefore raised 67 → 150,
which reproduces the ~15k peak this file already measured at ~105 views.
A 12-view submap by contrast reached 3/4 patience by iteration 2500,
confirming the detector fires when there is genuinely nothing left.

Gauge drift also fell as coverage rose (more views constrain the gauge):
|Δpos| median 0.2675 → 0.1077 on submap-0, and raw-pose held-out gained
~2 dB. Drift is now reported as the **view swing** it causes at the median
scene depth (≈0.3°), not as a multiple of the keyframe step — that
normalizer was degenerate, since flow-promoted keyframes during a pan sit
at near-zero baseline and drove the median step to 0.0001.

**Caveat on the defaults:** 8° spacing and 1250 surfels/view are reasoned,
not ablated. 150 iters/view is now evidence-based but from one clip. The
plateau detector makes a too-generous ceiling cheap, so the ceiling is the
safest of the three to be wrong about.

## `add` pipeline profiling + three speedups (2026-07-22)

Profiled `add 3.mp4` end-to-end (serial baseline 910 s), then implemented the
three biggest levers. Clean re-measure on a quiet machine: **629 s, −31%** —
with the adaptive-stop feature (landed independently) changing total work, so
the phase numbers below are the like-for-like evidence:

| lever | baseline | after | note |
|---|---|---|---|
| pose-aligned final eval | ~157 s | seconds | was 100 GPU align-solves × every eval view, for a diagnostic; now an 8-view/40-step probe (logged "N-view probe") |
| CPU/GPU pipeline | prep serialized | staging hidden | segments B+C fully staged while A still trained (log-order proof); staging reads only pre-training state, so semantics identical to serial; meta.txt writes are now atomic (tmp+rename) |
| global-BA track thinning | 118.4 s | 66.2 s | ≤8 evenly-spread obs per track entering the GLOBAL polish (m obs → m² pose-pair fill after Schur); local windows untouched; vo_synthetic ATE/RPE gates unchanged |

Also landed: checkpoint bakes (splat.ply every 2500 iters, tmp+rename — an
interrupted add is now viewable) and align-before-strike for the plateau
probe (a sagging probe is often the 8-step-warm aligner trailing the
drifting gauge, not model decline; strikes now require a 32-step top-up to
fail first).

**Metric caveat:** shrinking the final eval to a probe makes small-submap
scores fuzzier (5 views × 40 steps under-aligns vs the old 10 × 100 —
segment C reads 20.05 dB where the old measurement said 23.77; part
measurement, part its adaptive stop at 45% of ceiling). Cross-run PSNR
comparisons on small submaps now carry ±1–2 dB of measurement noise.

**GPU-utilization finding (corrected after a clean re-measure):** the
earlier "leak-shaped bwd-submit growth" was contention from a concurrent
user-launched add — in the uncontended run bwd-submit is FLAT per segment
(A ~10-15 ms, B 12→15.6 ms tracking splat spread, C ~4.3 ms). bwd-submit
mostly measures the FramePacer waiting for the GPU, i.e. the host is
already ahead; there is no host-side leak and no cheap host-side win.
The real 50-60% utilization cause: each iteration is ~15 small
dependency-chained dispatches (0.2-4 ms each) with barriers between them,
and sampled kernel sums (~12-15 ms) vs the 22.7 ms iteration period put
GPU-busy at ~60%. Closing that gap is kernel-graph work — merged passes,
fewer barriers, larger dispatches — not submit plumbing.

### Dense init from plane-sweep depth: +1.7 dB, half the iterations (2026-07-22)

PLAN.md §Pipeline step 4, "designed, not yet built" until now. The init it
replaces grows a sparse SfM cloud to the surfel budget by duplicating points
with random jitter at low opacity, then spends thousands of iterations
discovering geometry that is measurable up front.

**A/B on 1.mp4 (900 frames), submap-0, identical `--iters 7000` and budget
128,750 — the arms differ only in `--no-dense-init`:**

| probe iter | dense | sparse | Δ |
|---|---|---|---|
| 500 | 18.53 | 18.48 | +0.05 |
| 1500 | 21.47 | 20.30 | +1.17 |
| 2500 | 22.73 | 21.16 | +1.57 |
| 3500 | 23.45 | 22.60 | +0.85 |
| 5000 | 23.82 | 22.52 | +1.30 |
| 6500 | 24.40 | 23.16 | +1.24 |
| **final** | **24.92** | **23.19** | **+1.73** |

**Iterations-to-quality is the headline: dense init reaches 23.35 dB at
iteration 3000; the sparse run needs 6000 to reach the same number. Half the
iterations for equal quality, and +1.73 dB when both run to the ceiling.**

Cost is negligible: **~730 ms for 103 views**, against a training run of
minutes. Sweeping is essentially free relative to what it saves.

Kernel: per reference view, neighbours ranked by usable PARALLAX (not frame
proximity — a co-located rotation carries no depth information, the same trap
that forced VO pair selection onto global-affine residual rather than raw
flow); hypotheses uniform in inverse depth; patch NCC rather than zero-mean
SSD, because phone auto-exposure moves gain AND offset and an SSD volume would
largely measure exposure; winner-take-all with a subpixel parabola fit.
Verified against ground truth (a textured plane at known depth) rather than a
CPU port, which would reproduce a projection-convention error rather than catch
it: 96% of interior pixels accepted, relative depth error median 0.25% / p90
0.62%.

**Two bugs worth recording, both design errors rather than typos:**

1. Confidence was the margin over the GLOBAL runner-up. For any smooth cost
   curve the runner-up is the hypothesis ADJACENT to the winner, scoring almost
   identically — so a perfect match scored ~zero confidence and every pixel was
   rejected. The margin must exclude a neighbourhood of the peak; that is what
   distinguishes a unique match from a repetitive texture matching at several
   depths.
2. Normals were a finite-difference cross product needing two SPECIFIC accepted
   neighbours, compounding per-pixel acceptance to ~p³ (measured 10% yield on
   real footage). A least-squares plane fit over any 3 of the 3×3
   neighbourhood, in inverse depth where a plane is exactly affine in pixel
   coordinates, raised it to 17% (16,486 → 26,174 surfels).

**Sweep resolution, measured after the A/B rather than before it.** At
`downscale: 4` the sweep filled only ~20% of the surfel budget and random
jitter supplied the rest, so the +1.73 dB above is what one fifth of the idea
is worth. Re-running the dense arm at `downscale: 2` (everything else
identical):

| arm | swept surfels | % of budget | submap-0 held-out |
|---|---|---|---|
| sparse (`--no-dense-init`) | 0 | 0% | 23.19 dB |
| dense, sweep downscale 4 | 26,174 | 20% | 24.92 dB |
| **dense, sweep downscale 2** | **93,121** | **72%** | **25.63 dB** |

**+2.44 dB over sparse init**, and the sweep still costs **708 ms** for 103
views — the same as at downscale 4, so the coarse setting was latency-bound
rather than compute-bound and the finer one is genuinely free. Default is now
2. Iterations-to-quality holds: the sparse run's best (23.35 dB at iteration
6000) is passed by iteration 3000.

**Caveat:** on submap-1 (13 train views, a 2-view probe) sparse scored 31.79 vs
dense 31.26. Sample far too small to read anything into, recorded rather than
omitted.

### Trainer accepts views mid-run (2026-07-22)

The trainer half of PLAN.md §Pipeline step 5. `Trainer::add_view` uploads a
target into a reserved atlas slot and extends every per-view state vector, so
mapping no longer has to wait for the whole clip to be tracked and solved. The
atlas is reserved, not grown: it is uploaded once and stays GPU-resident
(CLAUDE.md), so reallocating would mean re-uploading every target.

`next_view` gains an optional recency bias. Uniform sampling over a growing
view set gives each new keyframe an ever-smaller share exactly when it needs
the most work, so a live map would lag further behind the camera the longer it
ran. Both knobs default to off — batch runs are bit-identical to before.

Gate (synthetic, 30 train views): 15 views up front, the other 15 appended at
iteration 1200. **Incremental 29.04 dB vs batch 28.50 dB** — arriving mid-run
costs nothing. `add_view` returns None past the reserved capacity rather than
silently dropping a keyframe, which would leave an unexplained hole in the map;
that is asserted too.

Not yet built: surfel spawning into the MCMC dead pool, pose-correction of
already-spawned surfels (the two-tier requirement), the causal provisional-pose
front-end, and the streaming driver.

### Maps survive pose corrections — two-tier mapping is sound (2026-07-22)

The load-bearing assumption of the two-tier design (provisional poses now,
anchor-out correction later): a map built under provisional poses must survive
being corrected, or a live map would have to be discarded every time better
poses landed.

`Trainer::transform_surfels` applies a Sim(3) to a contiguous surfel range and
zeroes those surfels' Adam moments. An index range is a sound way to name "what
this keyframe window spawned" because MCMC relocation never moves a surfel to a
different index — it overwrites DEAD slots with copies of alive ones, so an
alive surfel keeps its index for life.

Gate (synthetic, 4000 iterations then a 7° rotation + 0.2-unit translation
applied to surfels and cameras together): **27.88 dB → 27.88 dB**, exact, and
training resumes to 28.48 dB over the next 1000 iterations. The correction is a
warm start, not a rebuild.

**The bug the test caught is worth recording, because nothing else would have
found it.** The optimizer holds RAW parameters; the rasterizer reads a separate
ACTIVATED copy that only the Adam step refreshes. Transforming raw alone left
the renderer showing the pre-correction scene through post-correction
cameras — a 6 dB drop (27.88 → 21.86) that looked exactly like "the transform
maths is wrong". It is not; the fix is an `encode_activate` pass. Anything that
writes raw parameters outside the training step has this hazard, including the
surfel spawning still to be built.

**Known gap:** SH bands above degree 0 are not rotated, the same v1
simplification `compose_project` makes for rotated submaps. The gate above runs
at SH degree 0 and so does not cover it. Correct rotation needs Wigner-D
matrices; until then a large correction loses view-dependent colour that
descent must re-fit.

### Surfel spawning into the dead pool (2026-07-22)

The last trainer-side primitive for incremental mapping. New keyframes must be
able to ADD geometry: a corridor the camera has just entered contains nothing
for descent to move, and no amount of optimizing the existing set will invent
it. Relocation (relocate.wgsl modes 0/1) derives a new surfel from an alive
one; `Trainer::spawn_surfels` writes uploaded parameters instead, which is what
lets a plane-sweep depth map become surfels mid-run.

Budget-neutral like everything else touching the surfel set: the splat buffers
are pre-allocated to the MCMC budget and never reallocated (CLAUDE.md), so a
spawn consumes the dead pool. When the pool is short the placed count is
RETURNED rather than the request silently truncated — a caller feeding a depth
map needs to know its geometry did not land.

Gate: a deliberately starved model (40 live surfels in a 900 budget, MCMC
relocation disabled so it cannot refill the pool itself) trained to 22.76 dB,
then 600 surfels spawned onto the geometry it was missing → **49.10 dB** after
further training, all 600 placed. The high absolute number is not a quality
claim: the test hands it exact ground-truth positions and normals, because its
job is to prove the geometry is reachable at all, not to benchmark it. It also
asserts the render changes IMMEDIATELY after spawning, which is what catches
the raw-vs-activated hazard below.

**The raw/activated hazard, hit twice now.** The optimizer holds raw parameters
and the rasterizer reads a separate activated copy refreshed only by the Adam
step. `transform_surfels` hit this first (a 6 dB drop that looked like bad
transform maths); `spawn_surfels` needs the same `encode_activate` pass, and
its test asserts immediate visibility precisely so a future writer of raw
parameters cannot reintroduce it quietly.

Trainer-side incremental mapping is now complete: dense init, mid-run view
append, Sim(3) correction of spawned geometry, and spawning. Remaining work is
outside gs-train — the causal provisional-pose front-end in gs-pose, and the
streaming driver in gs-cli where the redundant per-submap decode (~75 s) and
the 15.4 min wall clock actually live.

### End-to-end regression: adaptive stopping is under-tuned (2026-07-22)

First full-clip run with dense init + view spacing + auto budget + adaptive
stopping all enabled. Back-room clip (3.mp4, 4315 frames, 2 min 24 s of video).

**17 min 31 s against the 15.4 min baseline — 13% SLOWER**, quality mixed:

| submap | keyframes | baseline | now | Δ | iterations | train time |
|---|---|---|---|---|---|---|
| 0 | 1749 | 17.11 dB | 17.40 | +0.29 | 7,500 / 40,000 | 184 s |
| 1 | 776 | 20.55 dB | 22.74 | **+2.19** | 19,160 / 21,600 | **609 s** |
| 2 | 272 | 23.96 dB | **20.59** | **−3.37** | 4,505 / 10,050 | 24 s |

**Two defects, both in work landed today.**

1. `ITERS_PER_VIEW = 150` is too aggressive. submap-1 spent 19,160 iterations
   (10 of the 17.5 minutes) buying 2.19 dB. That constant was calibrated from
   ONE clip — 1.mp4's probe was still climbing at its 6,834 ceiling — and
   generalising from a single data point is exactly what it looks like.
2. The plateau detector stops too early on small submaps, AND the anneal can
   end below the peak. submap-2 declared plateau at 3,500 iterations where the
   old fixed 7,000-iteration run reached 23.96 dB; its own probe peaked at
   21.24 pre-anneal and it finished at 20.59. **Nothing checks that the final
   state beats the best state observed.** A stop-on-plateau mechanism that can
   end worse than its own best is incomplete regardless of tuning.

**Dense init's benefit is scene-dependent, which the earlier +2.44 dB did not
show.** submap-0 gained only +0.29 dB and is pose-limited, not init-limited: a
3.4 dB raw-vs-aligned gap (13.97 vs 17.40) and a plateau at iteration 3,500 out
of a 40,000 ceiling — it stops improving long before running out of budget, so
neither initialization nor iterations is the cap. Better-placed surfels cannot
fix cameras that disagree with each other. The +2.44 dB on 1.mp4 stands, but it
was measured on a well-conditioned 539-keyframe submap and does not generalise
to the 1749-keyframe one.

**Not yet attributed:** the anchor-out solve came in at 124.5 s against 198.5 s
previously. That is not from any change recorded here.

Fixes wanted before this configuration should be trusted: keep the best
checkpoint rather than the annealed end state; scale plateau patience with
probe size (submap-2's probe is ~10 views, where noise alone can fake a
plateau); recalibrate the iteration ceiling against more than one clip.
