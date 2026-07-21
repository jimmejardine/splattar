//! Headless pipeline driver. Every pipeline stage runs here before any GUI work.

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod project;

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
    /// Full pipeline: video → VO → train → baked .ply splat model.
    Run {
        /// Path to an H.264 .mp4 walkthrough video.
        video: PathBuf,
        /// Focal length guess in pixels (default: 0.85 × the long side).
        #[arg(long)]
        focal: Option<f64>,
        /// Stop after this many decoded frames (0 = whole video).
        #[arg(long, default_value_t = 0)]
        max_frames: u32,
        #[arg(long, default_value_t = 4000)]
        iters: u32,
        #[arg(long, default_value_t = 150_000)]
        budget: u32,
        /// Integer downscale applied to training images.
        #[arg(long, default_value_t = 2)]
        downscale: u32,
        /// Cap on training views (sharpest-per-window selection).
        #[arg(long, default_value_t = 120)]
        max_views: u32,
        /// Fraction of training during which pose refinement runs (1.0 =
        /// full run; the LR decays on the position schedule either way).
        #[arg(long, default_value_t = 1.0)]
        pose_window: f32,
        /// Output path for the baked compat .ply.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Visual odometry: decode a video, track, solve keyframe poses, and
    /// write the trajectory + sparse landmarks as CSV next to the video.
    Pose {
        /// Path to an H.264 .mp4 walkthrough video.
        video: PathBuf,
        /// Focal length guess in pixels (default: 0.85 × the long side).
        #[arg(long)]
        focal: Option<f64>,
        /// Stop after this many decoded frames (0 = whole video).
        #[arg(long, default_value_t = 0)]
        max_frames: u32,
        /// Output CSV path (default: <video>.trajectory.csv).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Extend an existing project with another video: VO, cross-video Sim(3)
    /// registration against the project landmarks, train, persist as a new
    /// submap (registered, or an island if no overlap is found).
    Add {
        /// Path to an H.264 .mp4 walkthrough video.
        video: PathBuf,
        /// Project directory created by `run`.
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        focal: Option<f64>,
        #[arg(long, default_value_t = 0)]
        max_frames: u32,
        #[arg(long, default_value_t = 4000)]
        iters: u32,
        #[arg(long, default_value_t = 150_000)]
        budget: u32,
        #[arg(long, default_value_t = 2)]
        downscale: u32,
        #[arg(long, default_value_t = 120)]
        max_views: u32,
    },
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
    /// Bake the composed project to a single compat .ply + scene-manifest
    /// JSON sidecar (lossy snapshot — never re-ingested; `add` grows the
    /// project, then re-export).
    Export {
        project: PathBuf,
        /// Output .ply path (default: <project>/export.ply; manifest sits
        /// next to it as <stem>.manifest.json).
        #[arg(long)]
        out: Option<PathBuf>,
    },
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
            let cloud = if file.is_dir() {
                // Project directory: compose submaps (registered ones in the
                // shared world, islands offset side by side).
                compose_project(&file)?.0
            } else {
                let contents = gs_io::load_ply(&file)
                    .with_context(|| format!("loading {}", file.display()))?;
                match contents {
                    gs_io::PlyContents::Splats(c) => c,
                    gs_io::PlyContents::Points(p) => bail!(
                        "'{}' is a plain point cloud ({} points, xyz+rgb), not a gaussian \
                         splat file. Point rendering is not part of M0.",
                        file.display(),
                        p.len()
                    ),
                }
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
        Command::Run {
            video,
            focal,
            max_frames,
            iters,
            budget,
            downscale,
            max_views,
            pose_window,
            out,
        } => run_pipeline(
            &video, focal, max_frames, iters, budget, downscale, max_views, pose_window, out,
        ),
        Command::Pose {
            video,
            focal,
            max_frames,
            out,
        } => run_pose(&video, focal, max_frames, out),
        Command::Add {
            video,
            project,
            focal,
            max_frames,
            iters,
            budget,
            downscale,
            max_views,
        } => run_add(
            &video, &project, focal, max_frames, iters, budget, downscale, max_views,
        ),
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
        Command::Export { project, out } => run_export(&project, out),
    }
}

/// Shared VO stage: decode → causal pass → anchor-out solve.
struct VoOutput {
    result: gs_pose::VoResult,
    keyframes: Vec<gs_pose::vo::Keyframe>,
    intrinsics: gs_pose::vo::Intrinsics,
    video_size: (u32, u32),
}

fn run_vo(
    video: &std::path::Path,
    focal: Option<f64>,
    max_frames: u32,
) -> anyhow::Result<VoOutput> {
    use gs_pose::vo::{Intrinsics, VoConfig, VoFrontEnd};
    let mut reader = gs_video::Mp4H264Reader::open(video).context("open video")?;
    let t0 = std::time::Instant::now();

    let mut vo: Option<(VoFrontEnd, Intrinsics, (u32, u32))> = None;
    let mut n = 0u32;
    while let Some(frame) = reader.next_frame().context("decode")? {
        let (fe, ..) = vo.get_or_insert_with(|| {
            let f = focal.unwrap_or(0.85 * frame.width.max(frame.height) as f64);
            log::info!(
                "video {}x{}, focal guess {f:.0}px",
                frame.width,
                frame.height
            );
            let intr = Intrinsics {
                focal: f,
                cx: frame.width as f64 / 2.0,
                cy: frame.height as f64 / 2.0,
            };
            (
                VoFrontEnd::new(VoConfig {
                    intrinsics: intr,
                    ..Default::default()
                }),
                intr,
                (frame.width, frame.height),
            )
        });
        let gray = gs_pose::GrayImage::from_luma8(
            &frame.y,
            frame.width as usize,
            frame.height as usize,
        );
        fe.push_frame(gray, frame.pts);
        n += 1;
        if n.is_multiple_of(200) {
            log::info!("tracked {n} frames, {} keyframes", fe.keyframes.len());
        }
        if max_frames > 0 && n >= max_frames {
            break;
        }
    }
    let (mut fe, intrinsics, video_size) = vo.context("no frames decoded")?;
    let decode_track_s = t0.elapsed().as_secs_f64();
    log::info!(
        "causal pass: {n} frames, {} keyframes, {:.1} fps",
        fe.keyframes.len(),
        n as f64 / decode_track_s
    );

    let t1 = std::time::Instant::now();
    let result = fe.solve().context("VO solve failed (not enough parallax?)")?;
    log::info!(
        "anchor-out solve: {}/{} keyframes solved (anchor at kf {}), {} landmarks, {:.2}s",
        result.keyframe_poses.iter().flatten().count(),
        result.keyframe_poses.len(),
        result.anchor,
        result.landmarks.len(),
        t1.elapsed().as_secs_f64()
    );
    Ok(VoOutput {
        result,
        keyframes: std::mem::take(&mut fe.keyframes),
        intrinsics,
        video_size,
    })
}

fn run_pose(
    video: &std::path::Path,
    focal: Option<f64>,
    max_frames: u32,
    out: Option<PathBuf>,
) -> anyhow::Result<()> {
    let vo = run_vo(video, focal, max_frames)?;
    let solved: Vec<_> = vo.result.keyframe_poses.iter().flatten().collect();

    let out = out.unwrap_or_else(|| {
        let mut p = video.to_path_buf();
        p.set_extension("trajectory.csv");
        p
    });
    let mut csv = String::from("pts,cx,cy,cz,qw,qx,qy,qz\n");
    for kp in &solved {
        let c = kp.pose.center();
        let q = kp.pose.r.inverse(); // camera-to-world rotation
        csv.push_str(&format!(
            "{:.6},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8}\n",
            kp.pts, c[0], c[1], c[2], q.w, q.i, q.j, q.k
        ));
    }
    std::fs::write(&out, csv)?;
    log::info!("trajectory written to {}", out.display());
    Ok(())
}

/// Everything the trainer needs from a video, plus the persistence payload.
struct Prepared {
    train_views: Vec<gs_train::TrainView>,
    eval_views: Vec<gs_train::TrainView>,
    points: Vec<gs_io::SfmPoint>,
    /// Landmarks with descriptors for the project registration DB.
    landmarks: Vec<project::Landmark>,
    tw: u32,
    th: u32,
}

/// View selection + second decode pass + landmark assembly.
fn prepare_training(
    video: &std::path::Path,
    vo: &VoOutput,
    downscale: u32,
    max_views: u32,
) -> anyhow::Result<Prepared> {
    use gs_kernels::RasterCamera;
    use gs_train::TrainView;

    let ds = downscale.max(1) as usize;
    let (vw, vh) = (vo.video_size.0 as usize, vo.video_size.1 as usize);
    let (tw, th) = (vw / ds, vh / ds);

    // View selection: solved keyframes, sharpest per 0.25 s window, capped.
    let mut chosen: Vec<usize> = Vec::new();
    {
        let mut window_start = f64::NEG_INFINITY;
        let mut best_in_window: Option<usize> = None;
        for (k, kp) in vo.result.keyframe_poses.iter().enumerate() {
            if kp.is_none() {
                continue;
            }
            let kf = &vo.keyframes[k];
            if kf.pts - window_start > 0.25 {
                if let Some(b) = best_in_window.take() {
                    chosen.push(b);
                }
                window_start = kf.pts;
            }
            if best_in_window.is_none_or(|b| kf.sharpness > vo.keyframes[b].sharpness) {
                best_in_window = Some(k);
            }
        }
        if let Some(b) = best_in_window {
            chosen.push(b);
        }
    }
    if chosen.len() > max_views as usize {
        let stride = chosen.len() as f64 / max_views as f64;
        chosen = (0..max_views as usize)
            .map(|i| chosen[(i as f64 * stride) as usize])
            .collect();
    }
    anyhow::ensure!(chosen.len() >= 12, "too few usable views: {}", chosen.len());
    log::info!(
        "selected {} training views (downscale {ds} -> {tw}x{th})",
        chosen.len()
    );

    // Second decode pass: collect RGBA for the chosen keyframes only.
    let want: std::collections::HashMap<usize, usize> = chosen
        .iter()
        .enumerate()
        .map(|(slot, &k)| (vo.keyframes[k].frame_idx, slot))
        .collect();
    let mut images: Vec<Option<Vec<[f32; 4]>>> = vec![None; chosen.len()];
    {
        let mut reader = gs_video::Mp4H264Reader::open(video)?;
        let mut idx = 0usize;
        let mut remaining = want.len();
        while let Some(frame) = reader.next_frame()? {
            if let Some(&slot) = want.get(&idx) {
                let full =
                    gs_video::color::yuv420_to_rgba_f32(&frame.y, &frame.u, &frame.v, vw, vh);
                images[slot] = Some(box_downsample(&full, vw, vh, ds));
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
            idx += 1;
        }
        anyhow::ensure!(remaining == 0, "second decode pass missed {remaining} keyframes");
    }

    // Landmark init points: robust distance filter, color sampled from the
    // reference observation when it lands on a collected view.
    let centroid = {
        let mut c = glam::DVec3::ZERO;
        for l in &vo.result.landmarks {
            c += *l;
        }
        c / vo.result.landmarks.len().max(1) as f64
    };
    let mut dists: Vec<f64> = vo
        .result
        .landmarks
        .iter()
        .map(|l| (*l - centroid).length())
        .collect();
    dists.sort_by(f64::total_cmp);
    let med_dist = dists.get(dists.len() / 2).copied().unwrap_or(1.0);
    let slot_of_kf: std::collections::HashMap<usize, usize> =
        chosen.iter().enumerate().map(|(s, &k)| (k, s)).collect();
    let mut points: Vec<gs_io::SfmPoint> = Vec::new();
    let mut landmarks: Vec<project::Landmark> = Vec::new();
    for (li, l) in vo.result.landmarks.iter().enumerate() {
        if (*l - centroid).length() > 8.0 * med_dist {
            continue; // low-parallax runaway triangulation
        }
        let (kf, (px, py)) = vo.result.landmark_obs[li];
        let color = slot_of_kf
            .get(&kf)
            .and_then(|&slot| images[slot].as_ref())
            .map(|img| {
                let x = ((px as usize) / ds).min(tw - 1);
                let y = ((py as usize) / ds).min(th - 1);
                let p = img[y * tw + x];
                [
                    (p[0] * 255.0) as u8,
                    (p[1] * 255.0) as u8,
                    (p[2] * 255.0) as u8,
                ]
            })
            .unwrap_or([128, 128, 128]);
        let pos = [l.x as f32, l.y as f32, l.z as f32];
        points.push((pos, color));
        landmarks.push(project::Landmark {
            pos,
            color,
            desc: vo.result.landmark_desc[li],
        });
    }
    anyhow::ensure!(points.len() >= 500, "too few init points: {}", points.len());
    log::info!(
        "init: {} landmarks kept of {} (median-distance filter)",
        points.len(),
        vo.result.landmarks.len()
    );

    // Cameras in the renderer convention; every 8th view held out.
    let focal_t = (vo.intrinsics.focal / ds as f64) as f32;
    let mut train_views = Vec::new();
    let mut eval_views = Vec::new();
    for (slot, &k) in chosen.iter().enumerate() {
        let kp = vo.result.keyframe_poses[k].as_ref().unwrap();
        let c2w = kp.c2w();
        let m = glam::Mat4::from_cols_array(&c2w.to_cols_array().map(|v| v as f32));
        let (_, rot, trans) = m.to_scale_rotation_translation();
        let view = TrainView {
            camera: RasterCamera {
                center: trans,
                quat: rot,
                focal: focal_t,
                sh_degree: 3,
            },
            target: images[slot].take().unwrap(),
        };
        if slot % 8 == 0 {
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
    Ok(Prepared {
        train_views,
        eval_views,
        points,
        landmarks,
        tw: tw as u32,
        th: th as u32,
    })
}

/// Train on prepared views and bake a compat .ply. Returns (PSNR, surfels).
fn train_and_bake(
    prepared: Prepared,
    iters: u32,
    budget: u32,
    pose_window: f32,
    out: &std::path::Path,
) -> anyhow::Result<(f64, usize)> {
    use gs_train::{TrainConfig, Trainer};

    let ctx = pollster::block_on(gs_wgpu::GpuContext::new(gs_wgpu::backends_from_str(None)?))?;
    let mut init = gs_train::init_from_sfm_points(&prepared.points, 0x5eed);
    gs_train::upsample_to_budget(&mut init, budget as usize, 0xb00);
    let config = TrainConfig {
        iters,
        log_every: 500,
        // Geo losses measured at ~zero marginal cost (examples/geo_bench);
        // the earlier room-run collapse tracks scene evolution, not these
        // kernels — watch it/s in logs, not this config.
        geo_start: 1500,
        lambda_dist: 0.01,
        lambda_normal: 0.05,
        reg_opacity: 0.01,
        reg_scale: 0.005,
        sh_promote_every: 1000,
        mcmc_every: 300,
        // Noise sigma = mcmc_noise × pos LR, and pos LR scales with scene
        // EXTENT — on a walkthrough (extent ≈ 30× room depth) 20.0 jitters
        // surfels by ~0.1 world units per iter and wrecks held-out quality.
        mcmc_noise: 1.0,
        entries_per_surfel: 48,
        // VO poses + a guessed focal are noisy — let the verified camera
        // gradients polish them (validated: recovers +2.2 dB on synthetic
        // perturbed poses).
        pose_refine_lr: 2e-3,
        pose_refine_start: 500,
        // Window over which pose refinement runs (LR decays on the position
        // schedule regardless; the window matters on very long runs).
        pose_refine_end: ((iters as f32 * pose_window.clamp(0.05, 1.0)) as u32).max(1),
        focal_refine: true,
        // Phone auto-exposure sweeps continuously — compensate per view.
        appearance_start: 300,
        ..Default::default()
    };
    let mut eval_views = prepared.eval_views;
    let mut trainer = Trainer::new(
        &ctx,
        prepared.tw,
        prepared.th,
        prepared.train_views,
        init,
        config,
    );
    let start = std::time::Instant::now();
    trainer.train(&ctx);
    let elapsed = start.elapsed();
    let psnr = if eval_views.is_empty() {
        f64::NAN
    } else {
        // Focal is shared — apply the refined value to the held-out cameras.
        if trainer.focal_scale != 1.0 {
            log::info!("refined focal scale: {:.4}", trainer.focal_scale);
            for v in &mut eval_views {
                v.camera.focal *= trainer.focal_scale;
            }
        }
        // Training legitimately drifts the gauge away from raw VO poses, so
        // frozen eval cameras measure gauge drift, not model quality — align
        // each eval pose photometrically to the frozen model before scoring.
        let raw = trainer.eval_psnr(&ctx, &eval_views);
        let refined = trainer.eval_psnr_refined(&ctx, &eval_views, 100);
        log::info!("held-out PSNR: {raw:.2} dB raw poses, {refined:.2} dB pose-aligned");
        refined
    };
    log::info!(
        "trained {iters} iters in {elapsed:.0?} ({:.1} it/s); held-out PSNR {psnr:.2} dB",
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
    Ok((psnr, scene.num))
}

fn write_trajectory_csv(
    path: &std::path::Path,
    vo: &VoOutput,
) -> anyhow::Result<()> {
    let mut csv = String::from("pts,cx,cy,cz,qw,qx,qy,qz\n");
    for kp in vo.result.keyframe_poses.iter().flatten() {
        let c = kp.pose.center();
        let q = kp.pose.r.inverse();
        csv.push_str(&format!(
            "{:.6},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8}\n",
            kp.pts, c[0], c[1], c[2], q.w, q.i, q.j, q.k
        ));
    }
    std::fs::write(path, csv)?;
    Ok(())
}

/// M7 pipeline: video -> VO -> train -> project dir with submap-0 + baked ply.
#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    video: &std::path::Path,
    focal: Option<f64>,
    max_frames: u32,
    iters: u32,
    budget: u32,
    downscale: u32,
    max_views: u32,
    pose_window: f32,
    out: Option<PathBuf>,
) -> anyhow::Result<()> {
    let vo = run_vo(video, focal, max_frames)?;
    let prepared = prepare_training(video, &vo, downscale, max_views)?;

    // Project layout: submap-0 in its own gauge (= project world).
    let project_root = video.with_extension("project");
    let (idx, dir) = project::Project::next_submap_dir(&project_root)?;
    anyhow::ensure!(idx == 0, "project {} already exists — use `add`", project_root.display());
    project::write_meta(
        &dir.join("meta.txt"),
        &project::SubmapMeta {
            video: video.display().to_string(),
            focal: vo.intrinsics.focal,
            width: vo.video_size.0,
            height: vo.video_size.1,
            world_from_submap: Some(project::WorldFromSubmap::identity()),
        },
    )?;
    project::write_landmarks(&dir.join("landmarks.bin"), &prepared.landmarks)?;
    write_trajectory_csv(&dir.join("trajectory.csv"), &vo)?;

    let ply = dir.join("splat.ply");
    let (psnr, num) = train_and_bake(prepared, iters, budget, pose_window, &ply)?;
    if let Some(extra) = out {
        std::fs::copy(&ply, &extra)?;
    }
    println!(
        "done: {} ({num} surfels, held-out PSNR {psnr:.2} dB) — walk it with `gs-cli view {}`",
        project_root.display(),
        ply.display()
    );
    Ok(())
}

/// M8: extend a project with another video — register via descriptor-matched
/// 3D-3D landmark correspondences (RANSAC Sim(3)), or persist as an island.
#[allow(clippy::too_many_arguments)]
fn run_add(
    video: &std::path::Path,
    project_root: &std::path::Path,
    focal: Option<f64>,
    max_frames: u32,
    iters: u32,
    budget: u32,
    downscale: u32,
    max_views: u32,
) -> anyhow::Result<()> {
    use glam::DVec3;
    use gs_pose::descriptor::match_descriptors;
    use gs_pose::sim3::{Sim3G, register_point_sets};

    let proj = project::Project::load(project_root)?;

    // Pool the registered submaps' landmarks in project-world coordinates.
    let mut world_pts: Vec<DVec3> = Vec::new();
    let mut world_desc: Vec<gs_pose::descriptor::Descriptor> = Vec::new();
    for (i, meta) in proj.submaps.iter().enumerate() {
        let Some(w) = &meta.world_from_submap else { continue };
        let lms = project::read_landmarks(
            &project::Project::submap_dir(project_root, i).join("landmarks.bin"),
        )?;
        let s = Sim3G {
            scale: w.scale,
            rot: glam::DQuat::from_xyzw(w.quat[1], w.quat[2], w.quat[3], w.quat[0]),
            trans: DVec3::from_array(w.trans),
        };
        for l in lms {
            world_pts.push(s.apply(DVec3::new(
                l.pos[0] as f64,
                l.pos[1] as f64,
                l.pos[2] as f64,
            )));
            world_desc.push(l.desc);
        }
    }
    // Spatial dedup: KLT respawns re-triangulate the same physical corner
    // dozens of times across a long video; coincident duplicates let a
    // degenerate collapse transform out-vote the true registration.
    let (world_pts, world_desc) = dedup_landmarks(world_pts, world_desc);
    log::info!("project DB: {} registered landmarks after dedup", world_pts.len());

    let vo = run_vo(video, focal, max_frames)?;
    let prepared = prepare_training(video, &vo, downscale, max_views)?;

    // Descriptor matching new-submap -> world, then Sim(3) RANSAC on 3D pairs.
    let new_pts_all: Vec<DVec3> = prepared
        .landmarks
        .iter()
        .map(|l| DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64))
        .collect();
    let new_desc_all: Vec<gs_pose::descriptor::Descriptor> =
        prepared.landmarks.iter().map(|l| l.desc).collect();
    let (new_pts, new_desc) = dedup_landmarks(new_pts_all, new_desc_all);
    let pairs = match_descriptors(&new_desc, &world_desc, 55, 0.85);
    log::info!("descriptor matches: {}", pairs.len());
    let mut registration = None;
    if pairs.len() >= 20 {
        let a: Vec<DVec3> = pairs.iter().map(|&(i, _)| new_pts[i]).collect();
        let b: Vec<DVec3> = pairs.iter().map(|&(_, j)| world_pts[j]).collect();
        // Threshold relative to the world scene scale.
        let centroid = b.iter().copied().sum::<DVec3>() / b.len() as f64;
        let mut d: Vec<f64> = b.iter().map(|p| (*p - centroid).length()).collect();
        d.sort_by(f64::total_cmp);
        let thresh = 0.05 * d[d.len() / 2];
        if let Some((sim3, inliers)) = register_point_sets(&a, &b, 800, thresh, 0x5133) {
            // Inlier spread gate: a genuine overlap spans structure, it isn't
            // a tight cluster barely wider than the RANSAC threshold.
            let inl_b: Vec<DVec3> = inliers.iter().map(|&i| b[i]).collect();
            let cen = inl_b.iter().copied().sum::<DVec3>() / inl_b.len() as f64;
            let spread = (inl_b.iter().map(|p| (*p - cen).length_squared()).sum::<f64>()
                / inl_b.len() as f64)
                .sqrt();
            log::info!(
                "Sim(3): {} of {} matches agree (scale {:.3}, inlier spread {:.2} vs thresh {:.2})",
                inliers.len(),
                pairs.len(),
                sim3.scale,
                spread,
                thresh
            );
            if inliers.len() >= 25 && (0.05..=20.0).contains(&sim3.scale) && spread > 4.0 * thresh
            {
                registration = Some(sim3);
            }
        }
    }

    let (idx, dir) = project::Project::next_submap_dir(project_root)?;
    let world_from_submap = registration.map(|s| project::WorldFromSubmap {
        scale: s.scale,
        quat: [s.rot.w, s.rot.x, s.rot.y, s.rot.z],
        trans: s.trans.to_array(),
    });
    let registered = world_from_submap.is_some();
    project::write_meta(
        &dir.join("meta.txt"),
        &project::SubmapMeta {
            video: video.display().to_string(),
            focal: vo.intrinsics.focal,
            width: vo.video_size.0,
            height: vo.video_size.1,
            world_from_submap,
        },
    )?;
    project::write_landmarks(&dir.join("landmarks.bin"), &prepared.landmarks)?;
    write_trajectory_csv(&dir.join("trajectory.csv"), &vo)?;

    let ply = dir.join("splat.ply");
    let (psnr, num) = train_and_bake(prepared, iters, budget, 1.0, &ply)?;
    println!(
        "submap-{idx}: {num} surfels, held-out PSNR {psnr:.2} dB — {}",
        if registered {
            "REGISTERED into the project world (overlap found)"
        } else {
            "no overlap found: kept as an ISLAND (film a bridge clip to connect it)"
        }
    );
    println!("view the composed project with `gs-cli view {}`", project_root.display());
    Ok(())
}

/// Keep one landmark per voxel (0.5% of the median centroid distance): the
/// registration wants distinct physical corners, not every re-triangulation.
fn dedup_landmarks(
    pts: Vec<glam::DVec3>,
    desc: Vec<gs_pose::descriptor::Descriptor>,
) -> (Vec<glam::DVec3>, Vec<gs_pose::descriptor::Descriptor>) {
    if pts.is_empty() {
        return (pts, desc);
    }
    let centroid = pts.iter().copied().sum::<glam::DVec3>() / pts.len() as f64;
    let mut d: Vec<f64> = pts.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(f64::total_cmp);
    let voxel = (0.005 * d[d.len() / 2]).max(1e-6);
    let mut seen = std::collections::HashSet::new();
    let mut out_p = Vec::new();
    let mut out_d = Vec::new();
    for (p, dsc) in pts.into_iter().zip(desc) {
        let key = (
            (p.x / voxel).floor() as i64,
            (p.y / voxel).floor() as i64,
            (p.z / voxel).floor() as i64,
        );
        if seen.insert(key) {
            out_p.push(p);
            out_d.push(dsc);
        }
    }
    (out_p, out_d)
}

/// Bake the composed project into one compat .ply + a manifest sidecar.
fn run_export(project_root: &std::path::Path, out: Option<PathBuf>) -> anyhow::Result<()> {
    let (cloud, placements) = compose_project(project_root)?;
    let n = cloud.positions.len();
    let coeffs = if n > 0 { cloud.sh.len() / (n * 3) } else { 0 };

    // Flatten to the writer's layout (vec4 positions, 2 activated scales,
    // xyzw quats, 48-coeff-major SH padded/truncated to degree 3).
    let mut positions = Vec::with_capacity(n * 4);
    let mut scales = Vec::with_capacity(n * 2);
    let mut quats = Vec::with_capacity(n * 4);
    let mut sh = vec![0f32; n * 48];
    for i in 0..n {
        let p = cloud.positions[i];
        positions.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
        let s = cloud.scales[i];
        scales.extend_from_slice(&[s.x, s.y]);
        let q = cloud.rotations[i];
        quats.extend_from_slice(&[q.x, q.y, q.z, q.w]);
        let per = coeffs.min(16) * 3;
        sh[i * 48..i * 48 + per]
            .copy_from_slice(&cloud.sh[i * coeffs * 3..i * coeffs * 3 + per]);
    }

    let out = out.unwrap_or_else(|| project_root.join("export.ply"));
    gs_io::write_3dgs_ply(&out, n, &positions, &scales, &quats, &cloud.opacity, &sh)?;

    // Scene-manifest sidecar (hand-rolled JSON — no serde in the workspace).
    let mut manifest = String::from("{\n  \"submaps\": [\n");
    for (k, pl) in placements.iter().enumerate() {
        manifest.push_str(&format!(
            "    {{\"video\": {:?}, \"registered\": {}, \"surfel_start\": {}, \
             \"surfel_count\": {}, \"island_offset_x\": {:.4}, \
             \"bbox_min\": [{:.4}, {:.4}, {:.4}], \"bbox_max\": [{:.4}, {:.4}, {:.4}]}}{}\n",
            pl.video,
            pl.registered,
            pl.surfels.start,
            pl.surfels.len(),
            pl.island_offset_x,
            pl.bbox.0[0], pl.bbox.0[1], pl.bbox.0[2],
            pl.bbox.1[0], pl.bbox.1[1], pl.bbox.1[2],
            if k + 1 < placements.len() { "," } else { "" }
        ));
    }
    manifest.push_str("  ],\n  \"note\": \"baked snapshot — never re-ingest; grow the project with `gs-cli add` and re-export\"\n}\n");
    let manifest_path = out.with_extension("manifest.json");
    std::fs::write(&manifest_path, manifest)?;
    println!(
        "exported {n} surfels from {} submaps: {} (+ {})",
        placements.len(),
        out.display(),
        manifest_path.display()
    );
    Ok(())
}

/// Per-submap placement info produced during composition (for the manifest).
struct SubmapPlacement {
    video: String,
    registered: bool,
    surfels: std::ops::Range<usize>,
    /// Presentation x-offset applied to islands (0 for registered submaps).
    island_offset_x: f32,
    /// World-space AABB after placement.
    bbox: ([f32; 3], [f32; 3]),
}

/// Compose a project's submaps into one SplatCloud: registered submaps go
/// through their Sim(3) into the shared world; islands are placed side by
/// side along +x (presentation-only offset, never stored — per the two-tier
/// data-model rule).
fn compose_project(
    root: &std::path::Path,
) -> anyhow::Result<(gs_core::SplatCloud, Vec<SubmapPlacement>)> {
    use glam::{DQuat, DVec3, Quat, Vec3};

    let proj = project::Project::load(root)?;
    let mut merged: Option<gs_core::SplatCloud> = None;
    let mut island_cursor: Option<f32> = None; // starts at world max x + gap
    let mut placements: Vec<SubmapPlacement> = Vec::new();

    for (i, meta) in proj.submaps.iter().enumerate() {
        let ply = project::Project::submap_dir(root, i).join("splat.ply");
        let contents = gs_io::load_ply(&ply)
            .with_context(|| format!("loading {}", ply.display()))?;
        let gs_io::PlyContents::Splats(mut cloud) = contents else {
            bail!("{} is not a splat file", ply.display());
        };

        // Transform into the world (or island placement).
        let mut island_offset_x = 0.0f32;
        match &meta.world_from_submap {
            Some(w) => {
                let s = w.scale as f32;
                let rot = Quat::from_xyzw(
                    w.quat[1] as f32,
                    w.quat[2] as f32,
                    w.quat[3] as f32,
                    w.quat[0] as f32,
                );
                let t = Vec3::new(
                    w.trans[0] as f32,
                    w.trans[1] as f32,
                    w.trans[2] as f32,
                );
                let _ = (DQuat::IDENTITY, DVec3::ZERO); // (glam f64 kept for future precision)
                for p in &mut cloud.positions {
                    *p = s * (rot * *p) + t;
                }
                for q in &mut cloud.rotations {
                    *q = rot * *q;
                }
                for sc in &mut cloud.scales {
                    *sc *= s;
                }
                // Note: SH bands deg>0 are not rotated (v1 simplification) —
                // view-dependent color is slightly wrong for rotated submaps.
            }
            None => {
                // Island: normalize to sit next to the current world extent.
                let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
                for p in &cloud.positions {
                    min_x = min_x.min(p.x);
                    max_x = max_x.max(p.x);
                }
                let world_max = island_cursor.unwrap_or_else(|| {
                    merged
                        .as_ref()
                        .map(|m| {
                            m.positions
                                .iter()
                                .fold(f32::MIN, |acc, p| acc.max(p.x))
                        })
                        .unwrap_or(0.0)
                });
                let gap = (max_x - min_x) * 0.15 + 1.0;
                let dx = world_max + gap - min_x;
                for p in &mut cloud.positions {
                    p.x += dx;
                }
                island_cursor = Some(world_max + gap + (max_x - min_x));
                island_offset_x = dx;
                log::info!("submap-{i} is an unregistered island — placed at +x offset {dx:.1}");
            }
        }

        let start = merged.as_ref().map_or(0, |m| m.positions.len());
        let mut bbox = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &cloud.positions {
            for a in 0..3 {
                bbox.0[a] = bbox.0[a].min(p[a]);
                bbox.1[a] = bbox.1[a].max(p[a]);
            }
        }
        placements.push(SubmapPlacement {
            video: meta.video.clone(),
            registered: meta.world_from_submap.is_some(),
            surfels: start..start + cloud.positions.len(),
            island_offset_x,
            bbox,
        });

        merged = Some(match merged {
            None => cloud,
            Some(mut m) => {
                anyhow::ensure!(
                    m.sh_degree == cloud.sh_degree,
                    "submap SH degrees differ"
                );
                m.positions.extend(cloud.positions);
                m.sh.extend(cloud.sh);
                m.opacity.extend(cloud.opacity);
                m.scales.extend(cloud.scales);
                m.rotations.extend(cloud.rotations);
                m
            }
        });
    }
    Ok((merged.context("empty project")?, placements))
}

/// Integer box downsample for RGBA f32 images.
fn box_downsample(src: &[[f32; 4]], w: usize, h: usize, ds: usize) -> Vec<[f32; 4]> {
    if ds <= 1 {
        return src.to_vec();
    }
    let (ow, oh) = (w / ds, h / ds);
    let mut out = vec![[0.0f32; 4]; ow * oh];
    let inv = 1.0 / (ds * ds) as f32;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = [0.0f32; 4];
            for sy in 0..ds {
                for sx in 0..ds {
                    let p = src[(oy * ds + sy) * w + ox * ds + sx];
                    for c in 0..4 {
                        acc[c] += p[c];
                    }
                }
            }
            out[oy * ow + ox] = [acc[0] * inv, acc[1] * inv, acc[2] * inv, 1.0];
        }
    }
    out
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
