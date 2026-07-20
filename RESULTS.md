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
