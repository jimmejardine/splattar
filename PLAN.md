# Splattar — architecture & roadmap (second attempt)

## Context

The first attempt is a batch pipeline: decode the whole clip → track features →
solve all poses globally → select a sparse subset of views → train a model from
scratch against them. Measured on a 2 min 24 s clip it takes 17.5 minutes, and
today's session established that its two worst symptoms are structural, not
tuning:

- **It discards most of the video.** Feature VO tracks ~1000 corners per frame;
  view selection then keeps ~300 of 2794 keyframes. Almost every pixel captured
  is thrown away.
- **Poses and scene live in separate gauges.** The trainer refines cameras
  after the solver has committed poses to disk, so the two disagree — the root
  cause of the incoherent ground-truth overlay that opened this session, and of
  the 3.4 dB raw-vs-aligned gap that caps the back-room clip at ~17 dB.

The proposed replacement is **direct visual SLAM**: track the camera by
photometrically aligning the incoming frame against the rendered model, then
use the same residual to update the model. This is the mainstream design for
splat-based SLAM (MonoGS, Photo-SLAM). It uses every pixel, and it collapses
tracking and mapping into ONE loss, which deletes the two-gauge bug class by
construction.

**A second, equally important goal: the developer must be able to see what is
wrong.** dB is a scalar summary that hides where and why. Diagnostics are
therefore a first-class output of the pipeline, designed in from the start, not
a reporting layer bolted on.

## What the first attempt proved

Worth stating because it justifies the architecture rather than the code:

- The hand-derived differentiable 2DGS rasterizer works, including camera
  gradients (`grad_cam` = `[dl/dR 3×3, dl/dcenter, dl/dfocal]`),
  three-way verified against a CPU oracle and finite differences.
- Photometric pose descent against a frozen model **already works** —
  `eval_psnr_refined` recovers perturbed cameras this way. That is exactly the
  tracking step of the new design.
- Surfels beat 3DGS here for the same reason as before: view-consistent depth
  and real normals.
- Hardware NVDEC decode (H.264 + H.265, VFR-safe) works.
- Islands + Sim(3) merge is the right structure for disconnected coverage.

## Boundary with `first_attempt/`

`git mv` the entire current tree into `first_attempt/` (preserves history), give
it its own workspace manifest, and **freeze it** — it is never edited again. The
new workspace starts at the repo root, empty.

**Nothing carries over by default.** Each piece crosses the boundary only when a
milestone actually needs it, as a deliberate per-crate decision recorded in the
commit that does it. This is chosen over bulk carry-over so no batch-era
assumption arrives unexamined. It is emphatically *not* a mandate to re-derive
the verified gradient kernels: when M1 needs a differentiable render, the
expected decision is to bring `gs-kernels` and `gs-cpu-ref` across intact.

## Architecture

Per frame, in one loop:

1. **Decode** the next frame (skip frames with negligible motion — a cheap
   global-residual test, not feature counting).
2. **Predict** the pose by constant velocity.
3. **Track** — photometric descent of the camera pose alone against the frozen
   model, coarse-to-fine over an image pyramid. This is `advance_aligner`'s job
   in the old tree.
4. **Decide** whether this frame is a keyframe (viewpoint change since the last
   one, plus how much of the frame the model fails to explain).
5. **Map**, on keyframes only — joint descent over the scene *and* the poses in
   the sliding window.
6. **Spawn** surfels where the model does not explain the frame (plane-sweep
   depth from the window gives position and normal).
7. **Retain** — sliding window with exponential-decay history: recent frames
   dense, a thin tail of old anchors kept to hold long-range coherence.

**Tracking runs on every frame; mapping only on keyframes.** Small baselines
make photometric alignment reliable, so tracking wants density; mapping wants
spacing. Conflating them is a known failure mode.

**Photometric affine per frame.** Auto-exposure moves gain and offset
continuously and direct methods are far more sensitive to it than feature
methods. Every frame carries an `(a, b)` brightness transform inside the
tracking loss (as DSO does).

**On tracking loss** (blur, occlusion, white tear): the camera is probably near
where it was, so search locally over orientation around the last good pose,
coarse pyramid level first. If nothing locks on, start a new island.

### Relocalization and island merging

**Relocalization is the same operation as tracking.** "Where is this frame in
island A?" and "where is this frame in the model?" are the same question, so
merging reuses the tracking code rather than being a separate subsystem. This is
the largest simplification the architecture buys: in the first attempt
registration was its own descriptor/RANSAC/Sim(3) stack and it failed at 10–16
verified matches per pair.

Only candidate generation is genuinely new, because photometric descent has a
narrow basin and needs a starting pose. Tiered, cheapest first:

1. **Gravity prior** — estimate each island's up-vector from the modal surfel
   normal (floors and ceilings dominate an apartment). Collapses the search from
   6-DOF to yaw plus translation. Available only because surfels carry real
   normals.
2. **Coarse candidates** — render an island from a coarse pose grid at low
   resolution, compare against the other island's keyframes by cheap global
   similarity.
3. **Photometric refinement** of the surviving candidates — the tracking code,
   unchanged.
4. **Acceptance** on photometric residual and coverage across several frames,
   biased hard against false positives (a wrong merge is far worse than a missed
   one; the first attempt's relocalization benchmark had the same requirement).

### Overlap: duplicate surfels must be fused explicitly

Descent will **not** clean up an overlap on its own. Two coincident
semi-transparent surfels at 50% opacity render almost identically to one at 75%,
so the photometric loss has no gradient pushing them together; they persist as a
fuzzy double surface and can drift apart instead of converging.

After a merge: an explicit fuse pass (spatial hash, agreement on normal as well
as position, then fuse or kill), killed surfels returning to the MCMC dead pool
so it stays budget-neutral, followed by joint re-optimization over frames from
both islands. Acceptance is the double-wall seam test.

### Bending: a deformation graph, used for all drift

Rigidly placing each island is not enough. Drift accumulates *continuously along
the trajectory*, not in discrete jumps at island boundaries, so the map has to
deform.

Keyframes are graph nodes carrying a pose correction; each surfel binds to the k
keyframes that observed it, with weights. When a loop closes, Sim(3) pose-graph
relaxation distributes the error around the cycle and every surfel moves by the
weighted blend of its nodes' corrections. `transform_surfels` in the first
attempt — one rigid transform over one index range — is the degenerate case.

Two points that matter for the design:

- **A completed circle is what makes bending both possible and necessary.**
  Without a cycle there is no constraint: a chain of islands carries no
  information about its own drift and nothing to bend toward. The first attempt
  would never have bent — `resolve_placements` is a BFS spanning tree that
  stores cycle-closing edges and then ignores them.
- **This is the general drift-correction subsystem, not a merge feature.** The
  same graph closes loops *within* a single island, which is the more common
  case. Merging is one trigger among several.

Surfels suit this unusually well: no connectivity to maintain, so bending is
just moving independent primitives. It is why ElasticFusion chose them, and it
is a third argument for surfels alongside view-consistent depth and real
normals.

### Bootstrap

Planar init plus joint descent, then densify:

1. Seed surfels on a plane at a guessed depth from the first frame.
2. Run joint photometric descent over the first ~second of frames, optimizing
   poses AND surfel depths together. Real translation produces parallax, and
   parallax separates near from far — this is the "let the next frame tease out
   nearer/further" idea, and it is the principled resolution of the
   chicken-and-egg (depth needs pose, pose needs depth) because neither is
   assumed.
3. Once poses are roughly right, plane-sweep the window to densify properly.

No feature matching anywhere in the bootstrap. Same procedure starts each new
island.

## Diagnostics are a design constraint

Every frame emits a diagnostic record — incoming frame, render from the
estimated pose, error map, pose, per-level tracking residual, surfel count,
window contents, island id. Two consumers of the same stream:

- **Live window** while processing: frame | render | error heatmap, plus camera
  trail and island map. Pausable and steppable, so a run can be stopped at the
  frame where it visibly goes wrong.
- **Recorded trace** to disk, so any run can be replayed, scrubbed, and compared
  against another run.

**Signals that replace dB as the primary marker:** the error heatmap (where the
model disagrees), coverage (fraction of the frame the model explains at all),
the per-frame residual curve (when tracking degrades), pose-trail smoothness
(drift and jumps are visible as kinks), and window coherence (do old anchors
still line up). PSNR remains computable but stops being the steering signal.

## Milestones

Each ends in something visually checkable — that is the point of the ordering.

- **M0 — Move and empty workspace.** `first_attempt/` frozen; new root
  workspace; decode a video and display it with a scrub bar. *See: the video
  plays.*
- **M1 — Differentiable render + the three-pane view.** Bring `gs-kernels`,
  `gs-cpu-ref`, `gs-wgpu` across; render a hand-built scene; live frame |
  render | error panes. *See: a render, and where it disagrees.*
- **M2 — Track against a frozen model.** Synthetic scene, known camera path,
  recover the pose photometrically from a perturbed start. *See: the render
  snap onto the frame; residual curve fall.*
- **M3 — Bootstrap.** Planar init + joint descent over the first second. *See: a
  flat wall bend into a room.*
- **M4 — The loop.** Track + map + spawn + sliding window over a real clip.
  *See: the model grow as the video plays.*
- **M5 — Tears and relocalization.** Detect loss, search locally, resume. *See:
  it recover after a white tear.*
- **M6 — Loop closure and bending.** Deformation graph; Sim(3) pose-graph
  relaxation over all edges including cycle-closers; surfels follow their
  nodes. Built first for loops *within* one island, which is testable on a
  single clip that revisits a room. *See: a drifted corridor straighten when the
  loop closes.*
- **M7 — Islands and merge.** New islands on failure; gravity-primed candidate
  search; photometric verification; explicit duplicate fusion; merge reuses M6's
  graph. *See: two islands snap together, with no double wall.*

## Verification

Per-milestone visual acceptance as above, plus, carried over from the first
attempt because they caught real bugs:

- **Gradient three-way agreement** (finite difference ↔ CPU analytic ↔ GPU) on
  any kernel that crosses into the new tree. This is why `gs-cpu-ref` comes with
  `gs-kernels`.
- **Synthetic scenes with known ground truth** for tracking and bootstrap:
  a known camera path and known geometry make "did it recover the right answer"
  a fact rather than an opinion. Ground truth beat a CPU reimplementation for
  the plane-sweep kernel this session and would again.
- **Every stage runs headless** before it gets a GUI, so it can be tested.

Known hazards to design against, learned the expensive way:
- Photometric alignment is local — coarse-to-fine and motion prediction are not
  optional extras.
- Monocular scale drifts without loop constraints; the long-history window
  anchors exist for this.
- Any code writing raw optimizer parameters must re-run activation before the
  renderer reads them — this cost 6 dB once and nearly shipped twice.
