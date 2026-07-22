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
