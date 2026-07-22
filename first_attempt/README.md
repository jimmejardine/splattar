# Splattar

A 100% Rust pipeline that turns walkthrough video of an apartment into a
high-definition Gaussian-surfel (2DGS) model you can explore first-person —
desktop, VR, and web. No C/C++ dependencies (no ffmpeg, no COLMAP, no CUDA):
all GPU work runs on [wgpu](https://wgpu.rs) compute, the differentiable
rasterizer and trainer are written from scratch, and video decode is hardware
(NVDEC via Vulkan Video through pure-Rust `ash` bindings).

Docs: **[PLAN.md](PLAN.md)** — architecture, settled decisions, roadmap with
acceptance criteria (source of truth) · **[RESULTS.md](RESULTS.md)** —
measured numbers per milestone · **[CLAUDE.md](CLAUDE.md)** — constraints,
conventions, and verification rules for contributors.

## Status

| Milestone | State | Headline |
|---|---|---|
| M0 viewer | ✅ | 190 FPS @1440p on 1.9M splats |
| M1 GPU primitives | ✅ | radix sort 1.68 ms @4M keys; property-tested vs CPU |
| M2 differentiable 2DGS rasterizer | ✅ | gradients certified FD ↔ CPU ↔ GPU (worst 4.4e-7 analytic) |
| M3 trainer | ✅ | Adam-in-WGSL, fused L1+D-SSIM; 28.4 dB synthetic held-out (published-parity run pending) |
| M4 quality (MCMC, geo losses) | ✅ | aux-loss gradients verified; fixed-budget MCMC; TDR-safe |
| M5 video ingest | ✅ | hardware H.264 **and HEVC/H.265** decode (B-frames, tiles, 10-bit tone map, rotation, VFR); `play` sanity stepper |
| M6 VO front-end | ✅ | five-point bootstrap; full 3,605-keyframe real clip solves as segments; causal pass + BA parallelized (600-frame VO 232 s → 38 s) |
| M7 `add`: video → walkable splat | 🔶 | end-to-end works; trainer ~45× faster (2.3 → 100+ it/s); best 21.2 dB vs 24 dB gate — async-refinement lag is the open lead |
| M8 multi-video projects | 🔶 | order-independent projects (pairwise Sim(3) edges, islands, archipelago view); cross-video merge blocked at 10–16 verified matches/pair — full-res pairwise matching next |
| M9–M12 mesh/web/VR | — | not started |

## Quickstart

```
cargo build --workspace --release

# Fly around any 3DGS/2DGS splat file
cargo run --release -p gs-cli -- view samples/ply/cactus-high.ply

# Walkthrough video → project + walkable splat
# (creates gs-project next to the video; add order doesn't matter)
cargo run --release -p gs-cli -- add clips/walkthrough.mp4

# Extend with more videos (each registers into the graph or becomes an island)
cargo run --release -p gs-cli -- add clips/patch.mp4

# View the composed project (connected submaps merged, islands side by side)
cargo run --release -p gs-cli -- view clips/gs-project
```

More subcommands (`pose`, `play`, `train`, `register`, `refocal`, `export`)
are listed in CLAUDE.md §Commands. Decode requires an NVIDIA GPU with Vulkan
Video and a recent driver; training/viewing run on any wgpu-Vulkan GPU.

## Rebuilding local data on a new machine

`samples/` and `datasets/` are **gitignored** (multi-GB binaries). Tests that
depend on them skip gracefully when absent. To restore a full working setup:

| Path | What | How to rebuild |
|---|---|---|
| `datasets/room/` | Mip-NeRF360 *room* scene (COLMAP sparse + images) — trainer validation | Download the official bundle (~12 GB): `https://storage.googleapis.com/gresearch/refraw360/360_v2.zip`, extract, copy `room/` in. The loader wants `room/sparse/0/*.bin` plus `images/` (or `images_4/`). |
| `samples/ply/cactus-{low,med,high}.ply` | Standard 3DGS splats (binary LE, SH deg 3; 139k / ~700k / 1.9M splats) — `cactus-low` drives golden tests, `cactus-high` the M0 perf gate | Any standard 3DGS `.ply` works (same names; keep `cactus-low` small). After replacing `cactus-low`, regenerate goldens: `cargo run -p gs-cli --release -- view samples/ply/cactus-low.ply --render-golden assets/golden`, eyeball, commit as an intended visual change. |
| `samples/video/prinsengracht-494-*/` | Real phone walkthrough clips (Android H.264 + HEVC, iPhone HEVC, WhatsApp transcodes) — decode/VO/end-to-end/M8-merge fixtures; `prinsengracht-494-android/{1,2}.mp4` overlap → the merge pair | Record fresh clips on any phone (H.264 or HEVC, portrait or landscape both fine) and drop them in matching directories. |
| `assets/golden/` | Golden render images | In git — regenerate only for intended visual changes, and say so in the commit. |

Nothing else is machine-local: `cargo build --workspace` fetches all Rust
dependencies.
