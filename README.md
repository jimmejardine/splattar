# Splattar

A 100% Rust pipeline that turns walkthrough video of an apartment into a
high-definition Gaussian-surfel (2DGS) model you can explore first-person —
desktop, VR, and web. No C/C++ dependencies (no ffmpeg, no COLMAP, no CUDA);
all GPU work runs on [wgpu](https://wgpu.rs) compute, trained from scratch by
a hand-derived differentiable rasterizer. Video decode is hardware (NVDEC via
Vulkan Video).

**Status:** milestones **M0–M5 of M12 complete** — real-time viewer (190 FPS
@1440p on 1.9M splats), hardened GPU primitives, the differentiable 2DGS
rasterizer (gradients certified against finite differences at 4×10⁻⁷), the
full training loop (Adam-in-WGSL, fused L1+D-SSIM, MCMC densification,
geometry regularizers), and video ingest (MP4 → NVDEC H.264 → keyframes).
In progress: M6, the visual-odometry front-end.

```
cargo run --release -p gs-cli -- view samples/ply/cactus-high.ply   # fly around a splat
cargo run --release -p gs-cli -- train <colmap-dataset>             # validation harness
cargo test --workspace --release                                    # all gates
```

- **[PLAN.md](PLAN.md)** — architecture, settled decisions, milestone roadmap with acceptance criteria (source of truth)
- **[RESULTS.md](RESULTS.md)** — measured numbers per milestone
- **[CLAUDE.md](CLAUDE.md)** — hard constraints, conventions, and verification rules for contributors

## Rebuilding local data on a new machine

`samples/` and `datasets/` are **gitignored** (multi-GB binaries). Tests that
depend on them skip gracefully when absent. To restore a full working setup:

| Path | What | How to rebuild |
|---|---|---|
| `datasets/room/` | Mip-NeRF360 *room* scene (COLMAP sparse + images) — trainer validation | Download the official bundle (~12 GB): `https://storage.googleapis.com/gresearch/refraw360/360_v2.zip`, extract, copy the `room/` folder in. The loader wants `room/sparse/0/*.bin` plus an `images/` (or downscaled `images_4/`) directory. |
| `samples/ply/cactus-high.ply`, `cactus-low.ply` | Reference 3DGS splats for the viewer + golden-image tests | Any standard 3DGS `.ply` viewable in a reference viewer works (name them the same, keep `cactus-low` small so golden tests stay fast). After placing a new `cactus-low`, regenerate the committed goldens: `cargo run -p gs-cli --release -- view samples/ply/cactus-low.ply --render-golden assets/golden`, eyeball the PNGs, commit them as an intended visual change. |
| `samples/video/prinsengracht-494-android/*.mp4` | Real Android H.264 walkthrough clips — decode + VO gates | Record fresh clips on any Android phone (default camera app, H.264/AVC). Portrait or landscape both fine. Drop the `.mp4` files in; tests pick up `1.mp4`. iPhone users: set Camera → Formats → **Most Compatible** (H.264) until HEVC decode lands. |
| `assets/golden/` | Golden render images (committed) | In git — regenerate only for intended visual changes, and say so in the commit. |

Nothing else is machine-local: `cargo build --workspace` fetches all Rust
dependencies, and the GPU path needs only an up-to-date NVIDIA driver (Vulkan
Video for decode; any wgpu-Vulkan GPU for training/viewing).
