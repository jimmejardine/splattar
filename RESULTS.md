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
