# Splattar results — second attempt

Measured numbers per milestone, newest last. Append dated notes; never rewrite
history. Runs that went backwards are recorded too — the first attempt's most
useful entries were the negative ones.

First-attempt results are preserved at [first_attempt/RESULTS.md](first_attempt/RESULTS.md).
The findings worth carrying into this attempt:

- **The differentiable 2DGS rasterizer is sound**, including camera gradients
  (`grad_cam` = `[dl/dR 3×3, dl/dcenter, dl/dfocal]`), three-way verified
  against a CPU oracle and finite differences. Photometric pose descent against
  a frozen model recovers perturbed cameras — that is this attempt's tracking
  step, already proven to work.
- **Surfels earn their place**: view-consistent depth, real normals, and clean
  deformation under loop closure.
- **Hardware NVDEC decode works** for H.264 and H.265, VFR-safe.
- **Dense init beats random upsampling where init is the bottleneck** (+2.44 dB
  on a well-conditioned submap) but **not where pose quality is** (+0.29 dB on a
  drifted one). Better-placed geometry cannot fix cameras that disagree.
- **A batch pipeline that separates pose solving from scene fitting produces two
  gauges that disagree**, which capped real clips at ~17 dB and made the
  ground-truth overlay incoherent. This attempt exists to remove that split.

---

## M0 — Repo split (2026-07-22)

First attempt moved to `first_attempt/` with history preserved (151 files, all
recorded as renames), frozen, and excluded from the root workspace. New empty
workspace at the root. `samples/`, `datasets/` and `target/` stay at the root
and are shared.

## M0 — Decode + diagnostic window (2026-07-22)

`gs-cli play <video>` decodes and displays. Measured on
`samples/video/prinsengracht-494-android/1.mp4`: **~73 fps decode** with the
window open, ~70 fps headless, PTS advancing correctly (VFR-safe path intact).

Carried across the `first_attempt/` boundary: **`gs-video`**, unchanged, less a
declared dependency on `gs-core` it never used. Nothing else.

New: **`gs-diag`** — the diagnostic stream. `FrameRecord` carries
`frame | render | error` panels plus pose, per-pyramid-level residual, coverage,
surfel count and island id. The render and error panels are `Option` because at
M0 there is no model to render or difference against; an absent panel is not
drawn rather than drawn black. The record is meant to be filled in progressively
across M1–M7 rather than redesigned at each.

Two properties that exist to keep diagnostics usable rather than decorative:

- **The window is a passive consumer.** The pipeline pushes records and never
  learns whether anything is watching, so every stage stays runnable headless
  (`--headless`, which is how CI will run it) and the same records can later be
  replayed from a trace with no pipeline at all.
- **Pausing holds the pipeline.** The decoder blocks while paused instead of
  racing ahead. A viewer that cannot stop the pipeline is useless for finding
  the frame where something went wrong.

Deferred with reasons, not forgotten:
- **No scrub bar.** Stepping works (`←`/`→`, `Home`/`End`); a draggable timeline
  needs UI rendering that does not exist yet, and is better added alongside the
  residual curve at M1 so there is one overlay system rather than two.
- **No disk trace.** The record's shape should settle before a format is
  committed to; writing it now would mean migrating it at M1 and again at M2.

## M1 — Differentiable render + three-pane diagnostics (2026-07-22)

`gs-cli render` builds a synthetic room, renders it from a known camera and
from a deliberately displaced one, and shows **frame | render | error**. This is
what M2's tracker has to close, made visible before the tracker exists.

Carried across the boundary: **`gs-kernels`, `gs-cpu-ref`, `gs-core`,
`gs-wgpu`**, unchanged. `gs-kernels` needs `gs-core` and `gs-wgpu`; the oracle
crosses WITH the kernels rather than later, because a kernel without its oracle
cannot be checked. The gradient checks pass in the new tree: forward, backward,
aux-loss backward against the CPU oracle, plus binning against its CPU
reference.

New: **`gs-map`** — the surfel map, and the only thing holding geometry. The
CPU-side map is separate from its GPU-resident form so it can be built and
tested with no GPU at all.

**Measured, 3456-surfel room at 480×480 on the RTX 4090:**

| cameras | mean residual | peak |
|---|---|---|
| identical | **0.000000** | **0.000000** |
| offset 0.03 m, 0.4° | 0.017343 | 0.045939 |
| offset 0.15 m, 2.0° | 0.063867 | 0.074248 |

Two properties this establishes, both prerequisites for M2:

- **Identical cameras agree exactly.** Enforced as a hard check in the headless
  path, not merely observed: the two renders are the same call with the same
  inputs, so a non-zero residual would mean the render is not deterministic and
  every later measurement would be built on sand.
- **The residual is monotone in displacement.** 0.017 at 0.03 m against 0.064 at
  0.15 m. Photometric descent needs the residual to fall as the pose improves;
  if it did not order displacements correctly there would be nothing to descend.

~65 fps for two full renders per step, so the render is not the bottleneck in
anything that follows.

The synthetic scene is a closed ROOM, not a plane: a fronto-parallel plane can
be fitted at the wrong scale and still look correct from the bootstrap view,
which hides exactly the failure that matters. Wall colours are hash-based and
locally distinct, asserted in a test — a repetitive pattern matches at many
poses and would let a tracking test pass for the wrong reason.

## M2 — Photometric tracking recovers a camera (2026-07-22)

**The milestone the architecture rests on.** Everything after it assumes a frame
can be located against the map by descending the photometric residual. It can.

Ground truth, not a reimplementation: a known 3456-surfel room at 320×320, a
known camera, a deliberately displaced start.

| | start | recovered |
|---|---|---|
| position error | 0.1118 m | **0.0047 m** (24×) |
| rotation error | 1.50° | **0.07°** (21×) |
| residual | 0.01994 | **0.00065** (31×) |

Two supporting properties, each its own test because each is a way the headline
result could be true by accident:

- **A correct pose is held.** Starting at the answer, drift is 0.00000 m /
  0.000° and the descent stops after 11 iterations. A tracker that wandered off
  a good pose would corrupt the map on every already-correct frame.
- **The residual orders poses by correctness.** 0.00000 at the true pose,
  0.01670 at 0.05 m, 0.07112 at 0.25 m. If it did not, there would be nothing to
  descend and the recovery above could pass by luck.

New: **`gs-track`** — the photometric loss kernel and the pose descent that
consumes its camera gradient. The loss produces the residual and dL/d(render)
in one pass on purpose: a diagnostic that disagreed with the quantity being
optimized would lie about what the optimizer is doing. L1 rather than L2,
because the outliers here are occlusions and specularities — real content the
map does not explain — and L2 lets a handful of them dominate the update.

**Three bugs found, all worth recording because none was a typo.**

1. `target` is a reserved WGSL keyword (the same class of trap as `patch` in the
   first attempt).
2. **`final_residual()` described a pose the caller never receives.** The trace's
   last entry is measured BEFORE the final step. Worse, returning the last Adam
   iterate is arbitrary: Adam does not converge to a point, it orbits one at a
   radius set by the learning rate, so the last iterate can be markedly worse
   than one already visited. Measured: a clean descent to 0.00067 followed by a
   final measurement of 0.043. The tracker now returns the BEST pose seen — the
   same lesson the first attempt learned about adaptively-stopped training.
3. **The early stop fired on a single non-improving step.** With Adam orbiting,
   that is normal rather than a plateau: it quit at iteration 18 leaving 0.063 m
   of error on the table, against 0.0047 m when allowed to run. Now requires
   `patience` consecutive non-improving iterations.

Cost: ~32 s for 60 iterations at 320×320 in a debug-profile test binary, i.e.
~0.5 s per iteration under test settings. Not representative of release
performance, but a reminder that tracking must be coarse-to-fine before it meets
real video — full-resolution descent from a cold start is not affordable per
frame.
