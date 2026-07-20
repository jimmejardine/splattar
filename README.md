# Splattar

A 100% Rust pipeline that turns walkthrough video of an apartment into a
high-definition Gaussian-surfel (2DGS) model you can explore first-person —
desktop, VR, and web. No C/C++ dependencies (no ffmpeg, no COLMAP, no CUDA);
all GPU work runs on [wgpu](https://wgpu.rs) compute, trained from scratch by
a hand-derived differentiable rasterizer.

**Status:** milestones **M0–M3 of M12 complete** — real-time viewer (190 FPS
@1440p on 1.9M splats), hardened GPU primitives, the differentiable 2DGS
rasterizer (gradients certified against finite differences at 4×10⁻⁷), and
the full training loop (Adam-in-WGSL, fused L1+D-SSIM). Next: M4, MCMC
densification + geometry regularizers.

```
cargo run --release -p gs-cli -- view samples/ply/cactus-high.ply   # fly around a splat
cargo run --release -p gs-cli -- train <colmap-dataset>             # validation harness
cargo test --workspace --release                                    # all gates
```

- **[PLAN.md](PLAN.md)** — architecture, settled decisions, milestone roadmap with acceptance criteria (source of truth)
- **[RESULTS.md](RESULTS.md)** — measured numbers per milestone
- **[CLAUDE.md](CLAUDE.md)** — hard constraints, conventions, and verification rules for contributors
