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
