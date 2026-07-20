//! Headless pipeline driver. Every pipeline stage runs here before any GUI work.

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gs-cli", about = "Splattar: video → gaussian-surfel walkthrough", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Render a splat .ply file in an interactive window (WASD + mouse).
    View {
        /// Path to a .ply splat file.
        file: PathBuf,
        /// GPU backend: vulkan (default), dx12, or gl.
        #[arg(long)]
        backend: Option<String>,
        /// Cap to the monitor refresh rate (Fifo). Off by default so FPS is measurable.
        #[arg(long)]
        vsync: bool,
        #[arg(long, default_value_t = 1600)]
        width: u32,
        #[arg(long, default_value_t = 900)]
        height: u32,
        /// Active spherical-harmonics degree 0-3 (keys 0-3 switch at runtime).
        #[arg(long, default_value_t = 3)]
        sh_degree: u8,
        /// Debug multiplier on splat extents.
        #[arg(long, default_value_t = 1.0)]
        splat_scale: f32,
        /// Disable the 180° upright flip applied to COLMAP-convention scenes.
        #[arg(long)]
        no_flip: bool,
        /// Headless: render the three canonical golden poses (800×600 PNGs)
        /// into this directory and exit. Used to (re)generate assets/golden/.
        #[arg(long, value_name = "DIR")]
        render_golden: Option<PathBuf>,
    },
    /// Full pipeline: video → splat model (creates a project). Arrives in M7.
    Run { video: PathBuf },
    /// Extend an existing project with another video (relocalize + merge). Arrives in M8.
    Add { video: PathBuf },
    /// Validation harness: train on a posed COLMAP-format dataset.
    Train {
        /// Dataset root (contains sparse/0 and images/).
        dataset: PathBuf,
        #[arg(long, default_value_t = 7000)]
        iters: u32,
        /// Image downscale factor (prefers a pre-scaled images_N directory).
        #[arg(long, default_value_t = 4)]
        downscale: u32,
        /// Hold out every Nth view for evaluation (0 = train on everything).
        #[arg(long, default_value_t = 8)]
        holdout: u32,
        /// Output path for the baked compat .ply.
        #[arg(long, default_value = "trained.ply")]
        out: PathBuf,
        /// Fixed surfel budget (MCMC); SfM init is upsampled to this count.
        #[arg(long, default_value_t = 300_000)]
        budget: u32,
        /// Depth-distortion loss weight (per-pixel normalized).
        #[arg(long, default_value_t = 0.01)]
        lambda_dist: f32,
        /// Normal-consistency loss weight.
        #[arg(long, default_value_t = 0.05)]
        lambda_normal: f32,
        /// MCMC relocation interval in iterations (0 = off).
        #[arg(long, default_value_t = 300)]
        mcmc_every: u32,
        /// MCMC exploration-noise multiplier (× position LR).
        #[arg(long, default_value_t = 20.0)]
        mcmc_noise: f32,
        /// Promote SH degree every N iterations (0 = full degree from start).
        #[arg(long, default_value_t = 1000)]
        sh_promote: u32,
    },
    /// Export the project as baked .ply/.spz (+ scene manifest). Arrives in M7.
    Export { project: PathBuf },
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.command {
        Command::View {
            file,
            backend,
            vsync,
            width,
            height,
            sh_degree,
            splat_scale,
            no_flip,
            render_golden,
        } => {
            let contents = gs_io::load_ply(&file)
                .with_context(|| format!("loading {}", file.display()))?;
            let cloud = match contents {
                gs_io::PlyContents::Splats(c) => c,
                gs_io::PlyContents::Points(p) => bail!(
                    "'{}' is a plain point cloud ({} points, xyz+rgb), not a gaussian \
                     splat file. Point rendering is not part of M0.",
                    file.display(),
                    p.len()
                ),
            };
            if let Some(dir) = render_golden {
                return render_goldens(&file, &cloud, &dir, backend.as_deref());
            }
            let options = gs_viewer::windowed::ViewOptions {
                backends: gs_wgpu::backends_from_str(backend.as_deref())?,
                vsync,
                width,
                height,
                sh_degree: sh_degree.min(3),
                splat_scale,
                flip_scene: !no_flip,
                title: format!(
                    "splattar — {}",
                    file.file_name().unwrap_or_default().to_string_lossy()
                ),
            };
            gs_viewer::windowed::run(cloud, options).map_err(|e| anyhow::anyhow!(e))
        }
        Command::Run { .. } => bail!("`run` arrives in M7 — see PLAN.md"),
        Command::Add { .. } => bail!("`add` arrives in M8 — see PLAN.md"),
        Command::Train {
            dataset,
            iters,
            downscale,
            holdout,
            out,
            budget,
            lambda_dist,
            lambda_normal,
            mcmc_every,
            mcmc_noise,
            sh_promote,
        } => train(
            &dataset,
            TrainCliOpts {
                iters,
                downscale,
                holdout,
                out,
                budget,
                lambda_dist,
                lambda_normal,
                mcmc_every,
                mcmc_noise,
                sh_promote,
            },
        ),
        Command::Export { .. } => bail!("`export` arrives in M7 — see PLAN.md"),
    }
}

struct TrainCliOpts {
    iters: u32,
    downscale: u32,
    holdout: u32,
    out: PathBuf,
    budget: u32,
    lambda_dist: f32,
    lambda_normal: f32,
    mcmc_every: u32,
    mcmc_noise: f32,
    sh_promote: u32,
}

fn train(dataset: &std::path::Path, opts: TrainCliOpts) -> anyhow::Result<()> {
    use gs_kernels::RasterCamera;
    use gs_train::{TrainConfig, TrainView, Trainer};
    let TrainCliOpts {
        iters,
        downscale,
        holdout,
        out,
        budget,
        lambda_dist,
        lambda_normal,
        mcmc_every,
        mcmc_noise,
        sh_promote,
    } = opts;
    let out = out.as_path();

    let ctx = pollster::block_on(gs_wgpu::GpuContext::new(gs_wgpu::backends_from_str(None)?))?;
    let ds = gs_io::load_colmap(dataset, downscale)?;

    let mut train_views = Vec::new();
    let mut eval_views = Vec::new();
    for (i, v) in ds.views.iter().enumerate() {
        let view = TrainView {
            camera: RasterCamera {
                center: glam::Vec3::from_array(v.center),
                quat: glam::Quat::from_array(v.quat),
                focal: ds.focal,
                sh_degree: 3,
            },
            target: v.image.clone(),
        };
        if holdout > 0 && (i as u32).is_multiple_of(holdout) {
            eval_views.push(view);
        } else {
            train_views.push(view);
        }
    }
    log::info!(
        "training on {} views, evaluating on {} held-out",
        train_views.len(),
        eval_views.len()
    );

    let mut init = gs_train::init_from_sfm_points(&ds.points, 0x5eed);
    gs_train::upsample_to_budget(&mut init, budget as usize, 0xb00);
    log::info!(
        "init: {} SfM points upsampled to a {} surfel budget",
        ds.points.len(),
        init.positions.len()
    );
    let config = TrainConfig {
        iters,
        log_every: 500,
        lambda_dist,
        lambda_normal,
        reg_opacity: 0.01,
        reg_scale: 0.005,
        geo_start: 1500,
        sh_promote_every: sh_promote,
        mcmc_every,
        mcmc_noise,
        entries_per_surfel: 48,
        ..Default::default()
    };
    let mut trainer = Trainer::new(&ctx, ds.width, ds.height, train_views, init, config);
    let start = std::time::Instant::now();
    trainer.train(&ctx);
    let elapsed = start.elapsed();

    if !eval_views.is_empty() {
        let psnr = trainer.eval_psnr(&ctx, &eval_views);
        log::info!("held-out PSNR: {psnr:.2} dB over {} views", eval_views.len());
    }
    log::info!(
        "trained {iters} iters in {elapsed:.0?} ({:.1} it/s)",
        iters as f64 / elapsed.as_secs_f64()
    );

    let scene = trainer.read_scene(&ctx);
    gs_io::write_3dgs_ply(
        out,
        scene.num,
        &scene.positions,
        &scene.scales,
        &scene.quats,
        &scene.opacities,
        &scene.sh,
    )?;
    log::info!("wrote {} ({} surfels, flattened compat layout)", out.display(), scene.num);
    println!("done: {} — view it with `gs-cli view {}`", out.display(), out.display());
    Ok(())
}

/// Headless golden regeneration: renders the three canonical poses through the
/// exact viewer pipeline and writes `<stem>-pose{0,1,2}.png` into `dir`.
fn render_goldens(
    file: &std::path::Path,
    cloud: &gs_core::SplatCloud,
    dir: &std::path::Path,
    backend: Option<&str>,
) -> anyhow::Result<()> {
    use gs_render::{GpuScene, SplatRenderer, golden, offscreen};

    std::fs::create_dir_all(dir)?;
    let ctx = pollster::block_on(gs_wgpu::GpuContext::new(gs_wgpu::backends_from_str(
        backend,
    )?))?;
    let scene = GpuScene::upload(&ctx, cloud);
    let renderer = SplatRenderer::new(&ctx, &scene, offscreen::OFFSCREEN_FORMAT);
    let images = golden::render_goldens(&ctx, &scene, &renderer);

    let stem = file.file_stem().unwrap_or_default().to_string_lossy();
    let (w, h) = golden::GOLDEN_SIZE;
    for (i, rgba) in images.into_iter().enumerate() {
        let out = dir.join(format!("{stem}-pose{i}.png"));
        image::RgbaImage::from_raw(w, h, rgba)
            .context("image size")?
            .save(&out)
            .with_context(|| format!("writing {}", out.display()))?;
        println!("wrote {}", out.display());
    }
    Ok(())
}
