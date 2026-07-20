//! Golden-image regression test: renders cactus-low from the three canonical
//! poses and compares against committed goldens with a PSNR tolerance.
//! Regenerate intentionally via:
//!   cargo run -p gs-cli --release -- view samples/ply/cactus-low.ply --render-golden assets/golden
//! and say so in the commit message (see CLAUDE.md).

use std::path::PathBuf;

use gs_render::{GpuScene, SplatRenderer, golden, offscreen};
use gs_wgpu::GpuContext;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

#[test]
fn cactus_low_matches_goldens() {
    let root = repo_root();
    let sample = root.join("samples/ply/cactus-low.ply");
    if !sample.exists() {
        eprintln!("SKIPPING golden test: {} not present", sample.display());
        return;
    }
    let golden_paths: Vec<PathBuf> = (0..3)
        .map(|i| root.join(format!("assets/golden/cactus-low-pose{i}.png")))
        .collect();
    if golden_paths.iter().any(|p| !p.exists()) {
        eprintln!("SKIPPING golden test: goldens not generated yet (see test header)");
        return;
    }
    let Ok(ctx) = pollster::block_on(GpuContext::new(wgpu::Backends::all())) else {
        eprintln!("SKIPPING golden test: no GPU adapter");
        return;
    };

    let gs_io::PlyContents::Splats(cloud) = gs_io::load_ply(&sample).expect("load") else {
        panic!("cactus-low is not a splat ply?");
    };
    let scene = GpuScene::upload(&ctx, &cloud);
    let renderer = SplatRenderer::new(&ctx, &scene, offscreen::OFFSCREEN_FORMAT);
    let rendered = golden::render_goldens(&ctx, &scene, &renderer);

    for (i, (image, path)) in rendered.iter().zip(&golden_paths).enumerate() {
        let reference = image::open(path).expect("read golden").to_rgba8();
        assert_eq!(
            (reference.width(), reference.height()),
            golden::GOLDEN_SIZE,
            "golden {i} has wrong size"
        );
        let psnr = golden::psnr_rgba(image, reference.as_raw());
        eprintln!("golden pose {i}: PSNR {psnr:.1} dB");
        assert!(
            psnr >= golden::GOLDEN_PSNR_DB,
            "pose {i} PSNR {psnr:.1} dB below the {} dB gate — visual regression \
             (or an intended change: regenerate goldens per the test header)",
            golden::GOLDEN_PSNR_DB
        );
    }
}
