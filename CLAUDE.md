# Splattar — working rules (second attempt)

Architecture and milestones: [PLAN.md](PLAN.md). Measured results per
milestone: [RESULTS.md](RESULTS.md) — append dated notes, never rewrite history.

## The first attempt is frozen

`first_attempt/` holds the entire batch pipeline: decode-everything → feature VO
→ global solve → select a sparse view subset → train from scratch. It works and
reaches ~17.5 min on a 2.4-minute clip. It is **reference only and never
edited.** It has its own workspace and is excluded from the root one.

Why it was abandoned, so the mistakes are not repeated:
- **It discarded most of the video.** Feature VO tracks ~1000 corners per frame;
  view selection then kept ~300 of 2794 keyframes.
- **Poses and scene lived in separate gauges.** The trainer refined cameras
  after the solver had committed poses to disk, so the two disagreed. This
  architecture makes that impossible: one loss owns both.

**Nothing crosses from `first_attempt/` by default.** Each piece is carried over
only when a milestone needs it, as a deliberate decision recorded in the commit
that does it, so no batch-era assumption arrives unexamined. Carrying the
verified gradient kernels over intact when they are needed is expected — it is
bulk copying that is disallowed, not reuse.

## Hard constraints (unchanged, do not violate)

- **100% Rust.** No C/C++ build deps: no ffmpeg, COLMAP, OpenCV, no `-sys`
  crates that compile or link C/C++. Runtime-loaded system libs (GPU driver,
  OpenXR runtime DLL) are the only exception. Check what a crate links before
  adding it.
- **From scratch.** The differentiable 2DGS rasterizer (WGSL forward + hand-
  derived backward), the optimizer, and densification are this repo's code.
- **GPU via wgpu.** No CUDA.
- **Video is the only product input.** Posed-photo datasets are validation
  harnesses only, never the product path.
- **2D surfels** (2 scales + orientation), not 3D gaussians: view-consistent
  depth, real normals, and they deform cleanly under loop closure.

## Architecture in one paragraph

Direct visual SLAM. Per frame: predict pose by constant velocity → track by
photometric descent of the pose against the rendered model (coarse-to-fine) →
on keyframes, jointly descend scene *and* window poses → spawn surfels where the
model fails to explain the frame → retain a sliding window with exponential
decay so a thin tail of old anchors holds long-range coherence. Tracking and
mapping share ONE loss. Tracking runs every frame; mapping only on keyframes.

## Conventions

- Crate-per-concern workspace; `glam` for math everywhere.
- Pin shared versions in `[workspace.dependencies]`; crates use
  `dep.workspace = true`.
- Every stage runs headless before it gets a GUI.
- WGSL lives beside its dispatch code; forward/backward pairs adjacent, named
  `<stage>_fwd.wgsl` / `<stage>_bwd.wgsl`.
- Use `--release` for anything touching real video or optimization.

## Diagnostics are a design constraint, not a feature

**dB is not the steering signal.** It is a scalar that hides where and why. Every
frame emits a diagnostic record (frame, render, error map, pose, per-level
residual, surfel count, window, island id), consumed both by a live window
during processing and by a trace on disk that can be replayed and compared.

Steer by: the error heatmap (where the model disagrees), coverage (what fraction
of the frame the model explains at all), the residual curve (when tracking
degrades), pose-trail smoothness (drift shows as kinks), and window coherence.
PSNR stays computable; it stops being the thing decisions are made on.

## Verification rules

- **Any change to a WGSL kernel or its dispatch must pass the gradient checks**:
  finite-difference ↔ CPU-analytic ↔ GPU three-way agreement on randomized
  micro-scenes. Wrong gradients fail silently — green compile proves nothing.
- **Prefer ground truth to a reimplementation.** A synthetic scene with known
  geometry and a known camera path makes "did it recover the right answer" a
  fact. A CPU port of the same maths reproduces a convention error rather than
  catching it.
- Every stage runs headless via the CLI before any GUI work.
- Flag regressions, never bury them. RESULTS.md records what was measured,
  including the runs that went backwards.

## Gotchas (carried forward — these cost real time)

- WGSL has no f32 atomics: gradient accumulation is per-tile shared memory
  flushed via u32-bitcast CAS.
- **Any code writing raw optimizer parameters must re-run activation before the
  renderer reads them.** The optimizer holds raw values; the rasterizer reads a
  separate activated copy that only the optimizer step refreshes. This cost 6 dB
  once and nearly shipped twice.
- Splat buffers are pre-allocated to a fixed budget — growth consumes a dead
  pool, never reallocates. Training is f32; the viewer packs f16.
- Phone video is VFR — always carry per-frame PTS, never frame_index/fps.
  Display rotation is baked into decoded pixels; 10-bit HEVC tone-maps at decode.
- **Photometric alignment is local.** Coarse-to-fine pyramids and motion
  prediction are not optional extras; without them fast motion breaks tracking.
- **Auto-exposure moves gain AND offset.** Direct methods are far more sensitive
  to this than feature methods — every frame needs an affine brightness term in
  the tracking loss. Zero-mean alone absorbs only the offset.
- Monocular scale drifts without loop constraints. The long-history window
  anchors exist for exactly this.
- wgpu exposes one queue per device — no async compute. Parallelism is CPU-side
  pipelining feeding a serial GPU queue.

## Test assets

`samples/` (gitignored, shared with `first_attempt/`) holds the real phone
walkthroughs: `samples/video/prinsengracht-494-*`. Never write derived outputs
into `samples/` — pass an explicit output directory.
