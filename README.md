# Splattar

A 100% Rust pipeline that turns walkthrough video of an apartment into a
high-definition Gaussian-surfel (2DGS) model you can explore first-person —
desktop, VR, and web. No C/C++ dependencies (no ffmpeg, no COLMAP, no CUDA):
all GPU work runs on [wgpu](https://wgpu.rs) compute, the differentiable
rasterizer and trainer are written from scratch, and video decode is hardware
(NVDEC via Vulkan Video through pure-Rust `ash` bindings).

## Status

| Milestone | State | Headline |
|---|---|---|
| M0 viewer | ✅ | 190 FPS @1440p on 1.9M splats, 0.36 s load |
| M1 GPU primitives | ✅ | radix sort 1.68 ms @4M keys; property-tested vs CPU |
| M2 differentiable 2DGS rasterizer | ✅ | gradients certified FD ↔ CPU ↔ GPU (worst 4.4e-7 analytic) |
| M3 trainer | ✅ | Adam-in-WGSL, fused L1+D-SSIM; 28.4 dB synthetic held-out |
| M4 quality (MCMC, geo losses) | ✅ | aux-loss gradients verified; fixed-budget MCMC; TDR-safe |
| M5 video ingest | ✅ | hardware H.264 decode; 2,323 frames + keyframes in <5 s |
| M6 VO front-end | ✅ | ATE 0.91% / RPE <0.5° synthetic; 404/404 kf on real footage |
| M7 `run`: video → walkable splat | 🔶 | end-to-end works (~40 min); 20.3 dB vs 24 dB gate — appearance model next |
| M8 multi-video projects | 🔶 | persistence, `add`, Sim(3), islands work; merging awaits AKAZE-class descriptors |
| M9–M12 mesh/web/VR | — | not started |

Measured numbers and the story behind each gate: **[RESULTS.md](RESULTS.md)**.

## Quickstart

```
cargo build --workspace --release

# Fly around any 3DGS/2DGS splat file
cargo run --release -p gs-cli -- view samples/ply/cactus-high.ply

# Full pipeline: walkthrough video → project + walkable splat
cargo run --release -p gs-cli -- run walkthrough.mp4

# Extend the project with another video (registers or becomes an island)
cargo run --release -p gs-cli -- add patch.mp4 --project walkthrough.project

# View the composed project (registered submaps merged, islands side by side)
cargo run --release -p gs-cli -- view walkthrough.project

# Visual odometry only: trajectory CSV from a video
cargo run --release -p gs-cli -- pose walkthrough.mp4

# Validation harness: train on a posed COLMAP dataset
cargo run --release -p gs-cli -- train datasets/room

cargo test --workspace          # fast gates; add -- --ignored for GPU-heavy ones
cargo clippy --workspace -- -D warnings
```

Requires an NVIDIA GPU with Vulkan Video for decode (any wgpu-Vulkan GPU for
training/viewing) and a recent driver.

- **[PLAN.md](PLAN.md)** — architecture, settled decisions, milestone roadmap with acceptance criteria (source of truth)
- **[RESULTS.md](RESULTS.md)** — measured numbers per milestone
- **[CLAUDE.md](CLAUDE.md)** — hard constraints, conventions, and verification rules for contributors

## Workspace map

One crate per concern under `crates/`: `gs-core` (math/types), `gs-io`
(.ply/.spz, COLMAP harness), `gs-wgpu` (device, sort, prefix sum, bench),
`gs-kernels` (WGSL training rasterizer), `gs-cpu-ref` (f64 oracle, never
ships), `gs-train` (trainer + pose refinement), `gs-video` (MP4 demux + NVDEC
decode + keyframes), `gs-pose` (KLT/VO/BA/Sim(3); nalgebra quarantined here),
`gs-render`/`gs-viewer` (viewer path), `gs-cli`, `app-desktop`, `app-vr`,
`app-web`.

## Rebuilding local data on a new machine

`samples/` and `datasets/` are **gitignored** (multi-GB binaries). Tests that
depend on them skip gracefully when absent. To restore a full working setup:

| Path | What | How to rebuild |
|---|---|---|
| `datasets/room/` | Mip-NeRF360 *room* scene (COLMAP sparse + images) — trainer validation | Download the official bundle (~12 GB): `https://storage.googleapis.com/gresearch/refraw360/360_v2.zip`, extract, copy the `room/` folder in. The loader wants `room/sparse/0/*.bin` plus an `images/` (or downscaled `images_4/`) directory. |
| `samples/ply/cactus-high.ply`, `cactus-low.ply` | Reference 3DGS splats for the viewer + golden-image tests | Any standard 3DGS `.ply` viewable in a reference viewer works (name them the same, keep `cactus-low` small so golden tests stay fast). After placing a new `cactus-low`, regenerate the committed goldens: `cargo run -p gs-cli --release -- view samples/ply/cactus-low.ply --render-golden assets/golden`, eyeball the PNGs, commit them as an intended visual change. |
| `samples/video/prinsengracht-494-android/*.mp4` | Real Android H.264 walkthrough clips — decode + VO + end-to-end gates (`1.mp4`/`2.mp4` overlap → M8 merge fixture) | Record fresh clips on any Android phone (default camera app, H.264/AVC). Portrait or landscape both fine. Drop the `.mp4` files in; tests pick up `1.mp4`. iPhone users: set Camera → Formats → **Most Compatible** (H.264) until HEVC decode lands. |
| `assets/golden/` | Golden render images (committed) | In git — regenerate only for intended visual changes, and say so in the commit. |

Nothing else is machine-local: `cargo build --workspace` fetches all Rust
dependencies.
