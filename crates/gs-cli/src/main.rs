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
    /// Validation harness: train on a posed video-sequence dataset. Arrives in M3.
    Train { dataset: PathBuf },
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
        Command::Train { .. } => bail!("`train` arrives in M3 — see PLAN.md"),
        Command::Export { .. } => bail!("`export` arrives in M7 — see PLAN.md"),
    }
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
