//! Headless pipeline driver. Every pipeline stage runs here before any GUI work.

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod play;
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
    /// Sanity-check player: step through decoded frames in a window
    /// (→ +1 frame, Shift+→ +10, ↑ +1 s, Shift+↑ +10 s, R restart, Esc quit).
    Play {
        /// Path to an H.264/H.265 .mp4 video.
        video: PathBuf,
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
    /// Offline registration lab: re-attempt Sim(3) registration of one
    /// persisted submap against the rest of the project, with tunable
    /// matching parameters and diagnostics. Writes nothing unless --write.
    Register {
        project: PathBuf,
        /// Submap index to (re-)register.
        #[arg(long)]
        submap: usize,
        #[arg(long, default_value_t = 55)]
        max_dist: u32,
        #[arg(long, default_value_t = 0.85)]
        ratio: f32,
        #[arg(long, default_value_t = 5)]
        min_votes: usize,
        /// Persist a successful registration into the submap's meta.
        #[arg(long)]
        write: bool,
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
        Command::Play { video } => play::run_play(video),
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
        Command::Register {
            project,
            submap,
            max_dist,
            ratio,
            min_votes,
            write,
        } => run_register(&project, submap, max_dist, ratio, min_votes, write),
        Command::Export { project, out } => run_export(&project, out),
    }
}

/// Shared VO stage: decode → causal pass → anchor-out solve.
struct VoOutput {
    /// Solved segments in temporal order — each in its own monocular gauge
    /// (track-continuity breaks split the clip; Sim(3) registration or
    /// island placement joins the submaps built from them).
    segments: Vec<gs_pose::VoResult>,
    keyframes: Vec<gs_pose::vo::Keyframe>,
    intrinsics: gs_pose::vo::Intrinsics,
    video_size: (u32, u32),
    /// Half-res snapshots of every 4th keyframe (pairwise registration).
    thumbs: Vec<gs_pose::vo::Thumb>,
}

fn solved_count(seg: &gs_pose::VoResult) -> usize {
    seg.keyframe_poses.iter().flatten().count()
}

fn run_vo(
    video: &std::path::Path,
    focal: Option<f64>,
    max_frames: u32,
) -> anyhow::Result<VoOutput> {
    use gs_pose::vo::{Intrinsics, VoConfig, VoFrontEnd};
    let mut reader = gs_video::Mp4H264Reader::open(video).context("open video")?;
    let t0 = std::time::Instant::now();

    // Decode and tracking overlap: NVDEC decode (mostly blocking GPU fence
    // waits) stays on this thread — the Vulkan session isn't Send — while
    // KLT tracking runs on a worker fed through a small bounded channel, so
    // causal-pass wall clock is max(decode, track) instead of their sum.
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<u8>, u32, u32, f64)>(3);
    let worker = std::thread::spawn(move || {
        let mut vo: Option<(VoFrontEnd, Intrinsics, (u32, u32))> = None;
        let mut n = 0u32;
        while let Ok((y, width, height, pts)) = rx.recv() {
            let (fe, ..) = vo.get_or_insert_with(|| {
                let f = focal.unwrap_or(0.85 * width.max(height) as f64);
                log::info!("video {width}x{height}, focal guess {f:.0}px");
                let intr = Intrinsics {
                    focal: f,
                    cx: width as f64 / 2.0,
                    cy: height as f64 / 2.0,
                };
                (
                    VoFrontEnd::new(VoConfig {
                        intrinsics: intr,
                        ..Default::default()
                    }),
                    intr,
                    (width, height),
                )
            });
            let gray =
                gs_pose::GrayImage::from_luma8(&y, width as usize, height as usize);
            fe.push_frame(gray, pts);
            n += 1;
            if n.is_multiple_of(200) {
                log::info!("tracked {n} frames, {} keyframes", fe.keyframes.len());
            }
        }
        (vo, n)
    });
    let mut sent = 0u32;
    while let Some(frame) = reader.next_frame().context("decode")? {
        if tx
            .send((frame.y, frame.width, frame.height, frame.pts))
            .is_err()
        {
            break; // worker died; its panic resurfaces at join
        }
        sent += 1;
        if max_frames > 0 && sent >= max_frames {
            break;
        }
    }
    drop(tx);
    let (vo, n) = worker.join().expect("tracking worker panicked");
    let (mut fe, intrinsics, video_size) = vo.context("no frames decoded")?;
    let decode_track_s = t0.elapsed().as_secs_f64();
    log::info!(
        "causal pass: {n} frames, {} keyframes, {:.1} fps",
        fe.keyframes.len(),
        n as f64 / decode_track_s
    );

    let t1 = std::time::Instant::now();
    let segments = fe.solve_segments();
    anyhow::ensure!(
        !segments.is_empty(),
        "VO solve failed (not enough parallax?)"
    );
    let total_solved: usize = segments.iter().map(solved_count).sum();
    log::info!(
        "anchor-out solve: {} segment(s), {}/{} keyframes solved, {:.2}s",
        segments.len(),
        total_solved,
        fe.keyframes.len(),
        t1.elapsed().as_secs_f64()
    );
    for (i, seg) in segments.iter().enumerate() {
        log::info!(
            "  segment {i}: {} keyframes solved (anchor kf {}), {} landmarks",
            solved_count(seg),
            seg.anchor,
            seg.landmarks.len()
        );
    }
    Ok(VoOutput {
        segments,
        keyframes: std::mem::take(&mut fe.keyframes),
        intrinsics,
        video_size,
        thumbs: std::mem::take(&mut fe.thumbs),
    })
}

/// Persist this segment's keyframe thumbnails as grayscale PNGs.
fn write_thumbs(
    dir: &std::path::Path,
    thumbs: &[gs_pose::vo::Thumb],
    range: Option<(u32, u32)>,
) -> anyhow::Result<()> {
    let Some((lo, hi)) = range else { return Ok(()) };
    let tdir = dir.join("thumbs");
    std::fs::create_dir_all(&tdir)?;
    for t in thumbs {
        let kf = t.kf as u32;
        if kf < lo || kf > hi {
            continue;
        }
        let img = image::GrayImage::from_raw(t.width as u32, t.height as u32, t.data.clone())
            .context("thumb size")?;
        img.save(tdir.join(format!("{kf:05}.png")))?;
    }
    Ok(())
}

/// Load one persisted thumbnail as an f32 gray image.
fn load_thumb(dir: &std::path::Path, kf: u32) -> Option<gs_pose::GrayImage> {
    let img = image::open(dir.join("thumbs").join(format!("{kf:05}.png"))).ok()?;
    let g = img.to_luma8();
    Some(gs_pose::GrayImage {
        width: g.width() as usize,
        height: g.height() as usize,
        data: g.as_raw().iter().map(|&v| v as f32 / 255.0).collect(),
    })
}

/// Thumbnail keyframe indices available in a submap dir.
fn list_thumbs(dir: &std::path::Path) -> Vec<u32> {
    let mut out: Vec<u32> = std::fs::read_dir(dir.join("thumbs"))
        .map(|rd| {
            rd.filter_map(|e| {
                e.ok()?
                    .path()
                    .file_stem()?
                    .to_str()?
                    .parse::<u32>()
                    .ok()
            })
            .collect()
        })
        .unwrap_or_default();
    out.sort_unstable();
    out
}

fn run_pose(
    video: &std::path::Path,
    focal: Option<f64>,
    max_frames: u32,
    out: Option<PathBuf>,
) -> anyhow::Result<()> {
    let vo = run_vo(video, focal, max_frames)?;

    let out = out.unwrap_or_else(|| {
        let mut p = video.to_path_buf();
        p.set_extension("trajectory.csv");
        p
    });
    std::fs::write(&out, trajectory_csv(&vo))?;
    log::info!("trajectory written to {}", out.display());
    Ok(())
}

/// Multi-segment trajectory CSV. The `seg` column matters: each segment is
/// its own monocular gauge, so positions are only comparable within one.
fn trajectory_csv(vo: &VoOutput) -> String {
    let mut csv = String::from("seg,pts,cx,cy,cz,qw,qx,qy,qz\n");
    for (si, seg) in vo.segments.iter().enumerate() {
        for kp in seg.keyframe_poses.iter().flatten() {
            let c = kp.pose.center();
            let q = kp.pose.r.inverse(); // camera-to-world rotation
            csv.push_str(&format!(
                "{si},{:.6},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8}\n",
                kp.pts, c[0], c[1], c[2], q.w, q.i, q.j, q.k
            ));
        }
    }
    csv
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

/// View selection + second decode pass + landmark assembly, for one VO
/// segment (poses and landmarks live in that segment's gauge).
fn prepare_training(
    video: &std::path::Path,
    vo: &VoOutput,
    seg: &gs_pose::VoResult,
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
        for (k, kp) in seg.keyframe_poses.iter().enumerate() {
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
    anyhow::ensure!(chosen.len() >= 4, "too few usable views: {}", chosen.len());
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
        for l in &seg.landmarks {
            c += *l;
        }
        c / seg.landmarks.len().max(1) as f64
    };
    let mut dists: Vec<f64> = seg
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
    for (li, l) in seg.landmarks.iter().enumerate() {
        if (*l - centroid).length() > 8.0 * med_dist {
            continue; // low-parallax runaway triangulation
        }
        let (kf, (px, py)) = seg.landmark_obs[li];
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
            desc: seg.landmark_desc[li],
            kf: kf as u32,
            obs: [px, py],
            obs_all: seg.landmark_obs_all[li]
                .iter()
                .map(|(k, p)| (*k as u32, [p.0, p.1]))
                .collect(),
        });
    }
    anyhow::ensure!(points.len() >= 150, "too few init points: {}", points.len());
    log::info!(
        "init: {} landmarks kept of {} (median-distance filter)",
        points.len(),
        seg.landmarks.len()
    );

    // Cameras in the renderer convention; every 8th view held out.
    let focal_t = (vo.intrinsics.focal / ds as f64) as f32;
    let mut train_views = Vec::new();
    let mut eval_views = Vec::new();
    for (slot, &k) in chosen.iter().enumerate() {
        let kp = seg.keyframe_poses[k].as_ref().unwrap();
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
    std::fs::write(path, trajectory_csv(vo))?;
    Ok(())
}

/// Per-submap solved keyframe poses (kf index + center + c2w quat), the
/// offline registration lab's camera source.
fn write_seg_poses(path: &std::path::Path, seg: &gs_pose::VoResult) -> anyhow::Result<()> {
    let mut csv = String::from("kf,cx,cy,cz,qw,qx,qy,qz\n");
    for (k, kp) in seg.keyframe_poses.iter().enumerate() {
        let Some(kp) = kp else { continue };
        let c = kp.pose.center();
        let q = kp.pose.r.inverse(); // camera-to-world rotation
        csv.push_str(&format!(
            "{k},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8}\n",
            c[0], c[1], c[2], q.w, q.i, q.j, q.k
        ));
    }
    std::fs::write(path, csv)?;
    Ok(())
}

/// Load a poses.csv back as world→camera transforms per keyframe.
fn load_seg_poses(
    path: &std::path::Path,
) -> anyhow::Result<std::collections::HashMap<u32, (glam::DQuat, glam::DVec3)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut out = std::collections::HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<f64> = line.split(',').filter_map(|v| v.parse().ok()).collect();
        if f.len() != 8 {
            continue;
        }
        let center = glam::DVec3::new(f[1], f[2], f[3]);
        let q_c2w = glam::DQuat::from_xyzw(f[5], f[6], f[7], f[4]);
        let r_wc = q_c2w.conjugate();
        out.insert(f[0] as u32, (r_wc, -(r_wc * center)));
    }
    Ok(out)
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
    // Largest segment becomes submap-0 (its gauge = project world); the
    // clip beyond track-continuity breaks goes through the same
    // register-or-island path another video would.
    let mut order: Vec<usize> = (0..vo.segments.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(solved_count(&vo.segments[i])));
    let primary = &vo.segments[order[0]];
    let prepared = prepare_training(video, &vo, primary, downscale, max_views)?;

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
            kf_range: seg_kf_range(primary),
            world_from_submap: Some(project::WorldFromSubmap::identity()),
        },
    )?;
    project::write_landmarks(&dir.join("landmarks.bin"), &prepared.landmarks)?;
    write_trajectory_csv(&dir.join("trajectory.csv"), &vo)?;
    write_seg_poses(&dir.join("poses.csv"), primary)?;
    write_thumbs(&dir, &vo.thumbs, seg_kf_range(primary))?;

    let ply = dir.join("splat.ply");
    let (psnr, num) = train_and_bake(prepared, iters, budget, pose_window, &ply)?;
    if let Some(extra) = out {
        std::fs::copy(&ply, &extra)?;
    }
    for &si in &order[1..] {
        if let Err(e) = add_segment_submap(
            &project_root,
            video,
            &vo,
            &vo.segments[si],
            iters,
            budget,
            downscale,
            max_views,
        ) {
            log::warn!("segment {si}: skipped ({e:#})");
        }
    }
    println!(
        "done: {} ({} submap(s), submap-0: {num} surfels, held-out PSNR {psnr:.2} dB) — walk it with `gs-cli view {}`",
        project_root.display(),
        order.len(),
        project_root.display()
    );
    Ok(())
}

/// M8: extend a project with another video — each VO segment is registered
/// via descriptor-matched 3D-3D landmark correspondences (RANSAC Sim(3)),
/// or persisted as an island.
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
    let vo = run_vo(video, focal, max_frames)?;
    // Largest segment first: it has the best registration odds and, once
    // landed, its landmarks help the smaller ones register.
    let mut order: Vec<usize> = (0..vo.segments.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(solved_count(&vo.segments[i])));
    for &si in &order {
        if let Err(e) = add_segment_submap(
            project_root,
            video,
            &vo,
            &vo.segments[si],
            iters,
            budget,
            downscale,
            max_views,
        ) {
            log::warn!("segment {si}: skipped ({e:#})");
        }
    }
    println!("view the composed project with `gs-cli view {}`", project_root.display());
    Ok(())
}

/// Register one VO segment against the project's pooled landmark DB and
/// persist it as a new trained submap (or an island when no overlap is
/// found). Reloads the project each call so segments landed earlier extend
/// the DB for later ones.
#[allow(clippy::too_many_arguments)]
fn add_segment_submap(
    project_root: &std::path::Path,
    video: &std::path::Path,
    vo: &VoOutput,
    seg: &gs_pose::VoResult,
    iters: u32,
    budget: u32,
    downscale: u32,
    max_views: u32,
) -> anyhow::Result<()> {
    use glam::DVec3;
    use gs_pose::sim3::Sim3G;

    let proj = project::Project::load(project_root)?;

    // Registered submaps' landmarks in project-world coordinates, retaining
    // per-submap identity + keyframe index (both matter for registration
    // strategies), spatially deduped per submap (KLT respawns re-triangulate
    // the same corner dozens of times; coincident duplicates let a collapse
    // transform out-vote the true registration).
    let mut world: Vec<DbLandmark> = Vec::new();
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
        let entries: Vec<DbLandmark> = lms
            .into_iter()
            .map(|l| DbLandmark {
                pos: s.apply(DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64)),
                desc: l.desc,
                kf: l.kf,
                obs: (l.obs[0], l.obs[1]),
                submap: i,
            })
            .collect();
        world.extend(dedup_db_landmarks(entries));
    }
    log::info!("project DB: {} registered landmarks after dedup", world.len());

    let prepared = prepare_training(video, vo, seg, downscale, max_views)?;

    let new_all: Vec<DbLandmark> = prepared
        .landmarks
        .iter()
        .map(|l| DbLandmark {
            pos: DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64),
            desc: l.desc,
            kf: l.kf,
            obs: (l.obs[0], l.obs[1]),
            submap: usize::MAX,
        })
        .collect();
    let new = dedup_db_landmarks(new_all);

    // Strategy ladder: temporal bridge (same-video neighbor segments — small
    // viewpoint change, descriptors reliable), then covisibility-voted global
    // matching. No success → island (first-class, per PLAN).
    let seg_range = seg_kf_range(seg);
    let mut registration =
        try_bridge_registration(&proj, video, seg, &vo.intrinsics, seg_range, &world, &new);
    if registration.is_none() {
        registration = try_global_registration(&world, &new);
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
            kf_range: seg_kf_range(seg),
            world_from_submap,
        },
    )?;
    project::write_landmarks(&dir.join("landmarks.bin"), &prepared.landmarks)?;
    write_trajectory_csv(&dir.join("trajectory.csv"), vo)?;
    write_seg_poses(&dir.join("poses.csv"), seg)?;
    write_thumbs(&dir, &vo.thumbs, seg_kf_range(seg))?;

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
    Ok(())
}

/// A registration-DB landmark: world/segment position + descriptor +
/// reference keyframe + owning submap (usize::MAX for the new side).
struct DbLandmark {
    pos: glam::DVec3,
    desc: gs_pose::descriptor::MultiDescriptor,
    kf: u32,
    /// Reference observation pixel (NaN when unknown, e.g. pre-v3 files).
    /// Persisted for the world side of future 2D-based registration; the
    /// current bridge reads new-side observations from the VO result instead.
    #[allow(dead_code)]
    obs: (f32, f32),
    submap: usize,
}

/// Voxel-dedup keep-mask (0.5% of median centroid distance): true for the
/// first landmark in each voxel. Mask form so parallel arrays stay aligned.
fn dedup_mask(pts: &[glam::DVec3]) -> Vec<bool> {
    if pts.is_empty() {
        return Vec::new();
    }
    let centroid = pts.iter().copied().sum::<glam::DVec3>() / pts.len() as f64;
    let mut d: Vec<f64> = pts.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(f64::total_cmp);
    let voxel = (0.005 * d[d.len() / 2]).max(1e-6);
    let mut seen = std::collections::HashSet::new();
    pts.iter()
        .map(|p| {
            seen.insert((
                (p.x / voxel).floor() as i64,
                (p.y / voxel).floor() as i64,
                (p.z / voxel).floor() as i64,
            ))
        })
        .collect()
}

/// Voxel dedup keeping full records.
fn dedup_db_landmarks(lms: Vec<DbLandmark>) -> Vec<DbLandmark> {
    let mask = dedup_mask(&lms.iter().map(|l| l.pos).collect::<Vec<_>>());
    lms.into_iter()
        .zip(mask)
        .filter_map(|(l, keep)| keep.then_some(l))
        .collect()
}

/// Shared Sim(3) attempt over candidate 3D-3D pairs with degeneracy gates.
/// `thresh_frac` scales the RANSAC threshold off the world-side cloud spread;
/// `min_inliers` differs per strategy (a temporal bridge has a strong prior).
fn attempt_sim3(
    a: &[glam::DVec3],
    b: &[glam::DVec3],
    thresh_frac: f64,
    min_inliers: usize,
    label: &str,
) -> Option<gs_pose::sim3::Sim3G> {
    use gs_pose::sim3::register_point_sets;
    if a.len() < min_inliers {
        return None;
    }
    let centroid = b.iter().copied().sum::<glam::DVec3>() / b.len() as f64;
    let mut d: Vec<f64> = b.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(f64::total_cmp);
    let thresh = (thresh_frac * d[d.len() / 2]).max(1e-9);
    let (sim3, inliers) = register_point_sets(a, b, 1000, thresh, 0x5133)?;
    let inl_b: Vec<glam::DVec3> = inliers.iter().map(|&i| b[i]).collect();
    let cen = inl_b.iter().copied().sum::<glam::DVec3>() / inl_b.len() as f64;
    let spread = (inl_b.iter().map(|p| (*p - cen).length_squared()).sum::<f64>()
        / inl_b.len() as f64)
        .sqrt();
    log::info!(
        "Sim(3) [{label}]: {}/{} agree (scale {:.3}, spread {:.2} vs thresh {:.2})",
        inliers.len(),
        a.len(),
        sim3.scale,
        spread,
        thresh
    );
    (inliers.len() >= min_inliers
        && (0.05..=20.0).contains(&sim3.scale)
        && spread > 4.0 * thresh)
        .then_some(sim3)
}

/// Temporal bridge: consecutive segments of the SAME video are separated by
/// a track-loss cut, but their boundary keyframes view the same space seconds
/// apart — small viewpoint change, where the patch descriptors are reliable.
/// Match only boundary-window landmarks against each temporally adjacent
/// registered submap.
fn try_bridge_registration(
    proj: &project::Project,
    video: &std::path::Path,
    seg: &gs_pose::VoResult,
    intr: &gs_pose::vo::Intrinsics,
    seg_range: Option<(u32, u32)>,
    world: &[DbLandmark],
    _new: &[DbLandmark],
) -> Option<gs_pose::sim3::Sim3G> {
    use gs_pose::descriptor::match_descriptors;
    const WINDOW: u32 = 60; // keyframes each side of the cut
    const MAX_GAP: u32 = 200; // dropped stretches between segments can be long

    let (s0, s1) = seg_range?;
    let video_str = video.display().to_string();
    for (i, meta) in proj.submaps.iter().enumerate() {
        if meta.world_from_submap.is_none() || meta.video != video_str {
            continue;
        }
        let Some((o0, o1)) = meta.kf_range else { continue };
        // Which side is adjacent? (other before seg, or after)
        let (world_edge, new_edge) = if o1 <= s0 && s0 - o1 <= MAX_GAP {
            (o1, s0)
        } else if s1 <= o0 && o0 - s1 <= MAX_GAP {
            (o0, s1)
        } else {
            continue;
        };
        let w_sub: Vec<&DbLandmark> = world
            .iter()
            .filter(|l| l.submap == i && l.kf.abs_diff(world_edge) <= WINDOW)
            .collect();
        // New side straight from the segment (indices retained — the 2D stage
        // needs each landmark's full observation list).
        let n_sub: Vec<(usize, glam::DVec3)> = seg
            .landmark_obs
            .iter()
            .enumerate()
            .filter(|(_, (kf, _))| {
                *kf != usize::MAX && (*kf as u32).abs_diff(new_edge) <= WINDOW
            })
            .map(|(li, _)| (li, seg.landmarks[li]))
            .collect();
        if w_sub.len() < 12 || n_sub.len() < 12 {
            continue;
        }
        let wd: Vec<_> = w_sub.iter().map(|l| l.desc).collect();
        let nd: Vec<_> = n_sub
            .iter()
            .map(|(li, _)| seg.landmark_desc[*li])
            .collect();
        // TIGHT matching for the 3D-3D attempt: indoor texture repeats, and
        // false matches drown Umeyama RANSAC (measured: 131 loose, 3 true).
        let pairs = match_descriptors(&nd, &wd, 40, 0.75);
        log::info!(
            "bridge to submap-{i} (kf {world_edge}↔{new_edge}): {} window landmarks vs {}, {} tight matches",
            n_sub.len(),
            w_sub.len(),
            pairs.len()
        );
        if pairs.len() >= 8 {
            let a: Vec<glam::DVec3> = pairs.iter().map(|&(x, _)| n_sub[x].1).collect();
            let b: Vec<glam::DVec3> = pairs.iter().map(|&(_, y)| w_sub[y].pos).collect();
            if let Some(s) = attempt_sim3(&a, &b, 0.15, 8, &format!("bridge s{i}")) {
                return Some(s);
            }
        }
        // 2D bridge: boundary landmarks on the new side come from short
        // tracks truncated by the cut — depths are the noisiest in the map,
        // but the 2D observations are exact. DLT-PnP a boundary keyframe
        // against the world 3D; gauge scale via median depth ratio. Looser
        // matches are fine here — reprojection RANSAC absorbs outliers.
        // Tight matches first (high inlier fraction — best for the 6-point
        // minimal sample), loose as fallback for sparse boundaries.
        let loose = match_descriptors(&nd, &wd, 55, 0.85);
        log::info!(
            "2D-bridge stage: {} tight / {} loose matches",
            pairs.len(),
            loose.len()
        );
        let match_sets = [&pairs, &loose];
        let mut cand: Vec<usize> = seg
            .keyframe_poses
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .map(|(k, _)| k)
            .collect();
        cand.sort_by_key(|&k| (k as u32).abs_diff(new_edge));
        for (set_idx, match_set) in match_sets.iter().enumerate() {
            let min_obs = if set_idx == 0 { 6 } else { 8 };
            for &k in cand.iter().take(10) {
                let kp = seg.keyframe_poses[k].as_ref().unwrap();
                let obs: Vec<gs_pose::sim3::BridgeObs> = match_set
                    .iter()
                    .filter_map(|&(x, y)| {
                        let li = n_sub[x].0;
                        let px = seg.landmark_obs_all[li]
                            .iter()
                            .find(|(kk, _)| *kk == k)?
                            .1;
                        Some(gs_pose::sim3::BridgeObs {
                            obs: (
                                (px.0 as f64 - intr.cx) / intr.focal,
                                (px.1 as f64 - intr.cy) / intr.focal,
                            ),
                            world: w_sub[y].pos,
                            seg: n_sub[x].1,
                        })
                    })
                    .collect();
                log::debug!(
                    "2D bridge candidate kf {k} (set {set_idx}): {} co-observed",
                    obs.len()
                );
                if obs.len() < min_obs {
                    continue;
                }
                let q = kp.pose.r.quaternion();
                let t = &kp.pose.t;
                let result = gs_pose::sim3::sim3_from_bridge(
                    glam::DQuat::from_xyzw(q.i, q.j, q.k, q.w),
                    glam::DVec3::new(t[0], t[1], t[2]),
                    &obs,
                    8.0 / intr.focal, // ~8 px reprojection tolerance
                    0x2b71d6e,
                );
                match result {
                    Some((s, inl)) => {
                        log::info!(
                            "2D bridge via kf {k} (set {set_idx}): {inl}/{} inliers (scale {:.3})",
                            obs.len(),
                            s.scale
                        );
                        if inl >= min_obs && (0.05..=20.0).contains(&s.scale) {
                            return Some(s);
                        }
                    }
                    None => {
                        log::debug!(
                            "2D bridge kf {k} (set {set_idx}): solver rejected ({} obs)",
                            obs.len()
                        );
                    }
                }
            }
        }
    }
    None
}

/// Covisibility-voted global matching: match everything loosely, then keep
/// only matches whose (new-kf, world-kf) neighborhood pair gathers multiple
/// independent votes — genuine overlap concentrates on covisible keyframe
/// pairs, noise scatters. This replaces blind pooled matching.
fn try_global_registration(
    world: &[DbLandmark],
    new: &[DbLandmark],
) -> Option<gs_pose::sim3::Sim3G> {
    use gs_pose::descriptor::match_descriptors;
    const BUCKET: u32 = 8; // keyframes per vote bucket
    const MIN_VOTES: usize = 5;

    let wd: Vec<_> = world.iter().map(|l| l.desc).collect();
    let nd: Vec<_> = new.iter().map(|l| l.desc).collect();
    let pairs = match_descriptors(&nd, &wd, 55, 0.85);
    log::info!("global descriptor matches: {}", pairs.len());
    if pairs.len() < 20 {
        return None;
    }
    // Vote by (new bucket, world submap, world bucket).
    let mut votes: std::collections::HashMap<(u32, usize, u32), usize> =
        std::collections::HashMap::new();
    for &(x, y) in &pairs {
        if new[x].kf == u32::MAX || world[y].kf == u32::MAX {
            continue;
        }
        *votes
            .entry((new[x].kf / BUCKET, world[y].submap, world[y].kf / BUCKET))
            .or_default() += 1;
    }
    let good: std::collections::HashSet<(u32, usize, u32)> = votes
        .into_iter()
        .filter(|(_, v)| *v >= MIN_VOTES)
        .map(|(k, _)| k)
        .collect();
    let filtered: Vec<(usize, usize)> = pairs
        .into_iter()
        .filter(|&(x, y)| {
            new[x].kf != u32::MAX
                && world[y].kf != u32::MAX
                && good.contains(&(new[x].kf / BUCKET, world[y].submap, world[y].kf / BUCKET))
        })
        .collect();
    log::info!(
        "covisibility vote: {} matches in {} agreeing keyframe-pair buckets",
        filtered.len(),
        good.len()
    );
    if filtered.len() < 20 {
        return None;
    }
    let a: Vec<glam::DVec3> = filtered.iter().map(|&(x, _)| new[x].pos).collect();
    let b: Vec<glam::DVec3> = filtered.iter().map(|&(_, y)| world[y].pos).collect();
    attempt_sim3(&a, &b, 0.05, 25, "covis")
}

/// Solved keyframe index range (inclusive) of a VO segment.
fn seg_kf_range(seg: &gs_pose::VoResult) -> Option<(u32, u32)> {
    let solved: Vec<u32> = seg
        .keyframe_poses
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_some())
        .map(|(k, _)| k as u32)
        .collect();
    Some((*solved.first()?, *solved.last()?))
}


/// Offline registration lab over persisted submaps: full diagnostics
/// (distance percentiles, vote histogram), tunable gates, optional write.
fn run_register(
    project_root: &std::path::Path,
    submap: usize,
    max_dist: u32,
    ratio: f32,
    min_votes: usize,
    write: bool,
) -> anyhow::Result<()> {
    use glam::DVec3;
    use gs_pose::descriptor::{hamming_multi, match_descriptors};
    use gs_pose::sim3::Sim3G;

    let proj = project::Project::load(project_root)?;
    anyhow::ensure!(submap < proj.submaps.len(), "no submap-{submap}");

    let load = |i: usize| -> anyhow::Result<Vec<project::Landmark>> {
        project::read_landmarks(
            &project::Project::submap_dir(project_root, i).join("landmarks.bin"),
        )
    };
    // World: all registered submaps except the target. Raw per-submap lists
    // are retained for observation snapping in the pairwise stage.
    let mut world: Vec<DbLandmark> = Vec::new();
    let mut world_raw: Vec<(usize, Vec<project::Landmark>)> = Vec::new();
    let mut world_sims: std::collections::HashMap<usize, Sim3G> =
        std::collections::HashMap::new();
    for (i, meta) in proj.submaps.iter().enumerate() {
        if i == submap {
            continue;
        }
        let Some(w) = &meta.world_from_submap else { continue };
        let s = Sim3G {
            scale: w.scale,
            rot: glam::DQuat::from_xyzw(w.quat[1], w.quat[2], w.quat[3], w.quat[0]),
            trans: DVec3::from_array(w.trans),
        };
        let raw = load(i)?;
        let entries: Vec<DbLandmark> = raw
            .iter()
            .map(|l| DbLandmark {
                pos: s.apply(DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64)),
                desc: l.desc,
                kf: l.kf,
                obs: (l.obs[0], l.obs[1]),
                submap: i,
            })
            .collect();
        world.extend(dedup_db_landmarks(entries));
        world_sims.insert(i, s);
        world_raw.push((i, raw));
    }
    let new_raw = load(submap)?;
    let new_pts_all: Vec<DVec3> = new_raw
        .iter()
        .map(|l| DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64))
        .collect();
    let mask = dedup_mask(&new_pts_all);
    let new: Vec<&project::Landmark> = new_raw
        .iter()
        .zip(&mask)
        .filter_map(|(l, &keep)| keep.then_some(l))
        .collect();
    let new_pts: Vec<DVec3> = new
        .iter()
        .map(|l| DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64))
        .collect();
    println!("world {} landmarks, submap-{submap} {} landmarks", world.len(), new.len());

    // Distance diagnostics on the raw matches.
    let wd: Vec<_> = world.iter().map(|l| l.desc).collect();
    let nd: Vec<_> = new.iter().map(|l| l.desc).collect();
    let pairs = match_descriptors(&nd, &wd, max_dist, ratio);
    let mut dists: Vec<u32> = pairs
        .iter()
        .map(|&(x, y)| hamming_multi(&nd[x], &wd[y]))
        .collect();
    dists.sort_unstable();
    let pct = |p: usize| dists.get(dists.len() * p / 100).copied().unwrap_or(0);
    println!(
        "matches: {} (dist p10 {} / p50 {} / p90 {})",
        pairs.len(),
        pct(10),
        pct(50),
        pct(90)
    );

    // Vote histogram (top buckets).
    const BUCKET: u32 = 8;
    let mut votes: std::collections::HashMap<(u32, usize, u32), usize> =
        std::collections::HashMap::new();
    for &(x, y) in &pairs {
        if new[x].kf != u32::MAX && world[y].kf != u32::MAX {
            *votes
                .entry((new[x].kf / BUCKET, world[y].submap, world[y].kf / BUCKET))
                .or_default() += 1;
        }
    }
    let mut top: Vec<_> = votes.iter().collect();
    top.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    for ((nk, ws, wk), v) in top.iter().take(8) {
        println!(
            "  votes {v}: submap kf~{}..{} <-> {ws} kf~{}..{}",
            nk * BUCKET,
            (nk + 1) * BUCKET,
            wk * BUCKET,
            (wk + 1) * BUCKET
        );
    }
    // Vote-nominated regions for the pairwise image stage (votes ≥ 2 —
    // pairwise verification supplies its own precision).
    let top_regions: Vec<((u32, usize, u32), usize)> = top
        .iter()
        .filter(|(_, v)| **v >= 2)
        .take(6)
        .map(|(k, v)| (**k, **v))
        .collect();
    let good: std::collections::HashSet<(u32, usize, u32)> = votes
        .into_iter()
        .filter(|(_, v)| *v >= min_votes)
        .map(|(k, _)| k)
        .collect();
    let filtered: Vec<(usize, usize)> = pairs
        .into_iter()
        .filter(|&(x, y)| {
            good.contains(&(new[x].kf / BUCKET, world[y].submap, world[y].kf / BUCKET))
        })
        .collect();
    println!("covis-filtered: {} matches in {} buckets", filtered.len(), good.len());

    let mut registration = None;
    if filtered.len() >= 15 {
        let a: Vec<DVec3> = filtered.iter().map(|&(x, _)| new_pts[x]).collect();
        let b: Vec<DVec3> = filtered.iter().map(|&(_, y)| world[y].pos).collect();
        registration = attempt_sim3(&a, &b, 0.05, 15, "register-lab 3D");
    }

    // 2D stage: DLT-PnP a covisible keyframe of the target submap against
    // the world 3D of the filtered matches (dodges this submap's own noisy
    // landmark depths; gauge scale from median depth ratios).
    if registration.is_none() && filtered.len() >= 8 {
        let meta = &proj.submaps[submap];
        let (cx, cy) = (meta.width as f64 / 2.0, meta.height as f64 / 2.0);
        let poses = load_seg_poses(
            &project::Project::submap_dir(project_root, submap).join("poses.csv"),
        )?;
        // Rank keyframes by how many filtered matches they observe.
        let mut per_kf: std::collections::HashMap<u32, Vec<(usize, usize, [f32; 2])>> =
            std::collections::HashMap::new();
        for &(x, y) in &filtered {
            for (k, p) in &new[x].obs_all {
                per_kf.entry(*k).or_default().push((x, y, *p));
            }
        }
        let mut ranked: Vec<_> = per_kf.into_iter().collect();
        ranked.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
        for (k, group) in ranked.into_iter().take(8) {
            // One camera sees ONE world region: keep only this keyframe's
            // dominant world neighborhood (mode over 16-kf world buckets).
            let mut region_votes: std::collections::HashMap<(usize, u32), usize> =
                std::collections::HashMap::new();
            for &(_, y, _) in &group {
                *region_votes
                    .entry((world[y].submap, world[y].kf / 16))
                    .or_default() += 1;
            }
            let Some((&dom, _)) = region_votes.iter().max_by_key(|(_, v)| **v) else {
                continue;
            };
            let group: Vec<_> = group
                .into_iter()
                .filter(|&(_, y, _)| {
                    world[y].submap == dom.0 && (world[y].kf / 16).abs_diff(dom.1) <= 1
                })
                .collect();
            if group.len() < 6 {
                continue;
            }
            let Some((r_wc, t_wc)) = poses.get(&k) else { continue };
            let obs: Vec<gs_pose::sim3::BridgeObs> = group
                .iter()
                .map(|&(x, y, p)| gs_pose::sim3::BridgeObs {
                    obs: (
                        (p[0] as f64 - cx) / meta.focal,
                        (p[1] as f64 - cy) / meta.focal,
                    ),
                    world: world[y].pos,
                    seg: new_pts[x],
                })
                .collect();
            let result = gs_pose::sim3::sim3_from_bridge(
                *r_wc,
                *t_wc,
                &obs,
                8.0 / meta.focal,
                0x2b71d6e,
            );
            match result {
                Some((s, inl)) => {
                    println!(
                        "2D stage kf {k}: {inl}/{} reprojection inliers (scale {:.3})",
                        obs.len(),
                        s.scale
                    );
                    if inl >= 10 && (0.05..=20.0).contains(&s.scale) {
                        registration = Some(s);
                        break;
                    }
                }
                None => println!("2D stage kf {k}: rejected ({} obs)", group.len()),
            }
        }
    }
    // Pairwise image stage: for vote-nominated keyframe regions, match the
    // two actual images and epipolar-verify, then snap verified corners to
    // stored landmark observations — precision comes from the geometry of
    // the specific pair, not from global descriptor uniqueness.
    if registration.is_none() && !top_regions.is_empty() {
        use gs_pose::pairwise::{PairwiseConfig, match_image_pair};
        let sub_dir = |i: usize| project::Project::submap_dir(project_root, i);
        let new_thumbs = list_thumbs(&sub_dir(submap));
        let new_meta = &proj.submaps[submap];
        let poses = load_seg_poses(&sub_dir(submap).join("poses.csv")).unwrap_or_default();

        // Observation snap indices: (submap, kf) → [(px, py, world pos)] and
        // kf → [(px, py, seg pos)] for the target.
        type ObsIndex = std::collections::HashMap<(usize, u32), Vec<(f32, f32, glam::DVec3)>>;
        let mut w_index: ObsIndex = std::collections::HashMap::new();
        for (i, raw) in &world_raw {
            let s = world_sims[i];
            for l in raw {
                let wpos =
                    s.apply(DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64));
                for (kf, p) in &l.obs_all {
                    w_index.entry((*i, *kf)).or_default().push((p[0], p[1], wpos));
                }
            }
        }
        let mut n_index: std::collections::HashMap<u32, Vec<(f32, f32, DVec3)>> =
            std::collections::HashMap::new();
        for l in &new_raw {
            let spos = DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64);
            for (kf, p) in &l.obs_all {
                n_index.entry(*kf).or_default().push((p[0], p[1], spos));
            }
        }
        let snap = |list: Option<&Vec<(f32, f32, DVec3)>>, px: (f32, f32)| -> Option<DVec3> {
            let list = list?;
            let (mut best, mut best_d2) = (None, 8.0f32 * 8.0);
            for &(x, y, p) in list {
                let d2 = (x - px.0).powi(2) + (y - px.1).powi(2);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = Some(p);
                }
            }
            best
        };
        let nearest = |kfs: &[u32], target: u32| -> Option<u32> {
            kfs.iter().copied().min_by_key(|k| k.abs_diff(target))
        };

        'regions: for &((nk, ws, wk), v) in &top_regions {
            let w_thumbs = list_thumbs(&sub_dir(ws));
            let (Some(kn), Some(kw)) = (
                nearest(&new_thumbs, nk * BUCKET + BUCKET / 2),
                nearest(&w_thumbs, wk * BUCKET + BUCKET / 2),
            ) else {
                continue;
            };
            let (Some(n_img), Some(w_img)) =
                (load_thumb(&sub_dir(submap), kn), load_thumb(&sub_dir(ws), kw))
            else {
                continue;
            };
            let cfg2 = PairwiseConfig {
                image_scale: 0.5,
                focal: new_meta.focal,
                cx: new_meta.width as f64 / 2.0,
                cy: new_meta.height as f64 / 2.0,
                ..Default::default()
            };
            let verified = match_image_pair(&w_img, &n_img, &cfg2);
            println!(
                "pairwise ({v} votes): world s{ws} kf {kw} <-> kf {kn}: {} verified",
                verified.len()
            );
            if verified.len() < 15 {
                continue;
            }
            // Snap verified endpoints to landmark observations.
            let mut a3 = Vec::new(); // new/seg side
            let mut b3 = Vec::new(); // world side
            let mut bridge = Vec::new();
            for m in &verified {
                let wsnap = snap(w_index.get(&(ws, kw)), m.a_px);
                let nsnap = snap(n_index.get(&kn), m.b_px);
                if let (Some(wp), Some(np)) = (wsnap, nsnap) {
                    a3.push(np);
                    b3.push(wp);
                    bridge.push(gs_pose::sim3::BridgeObs {
                        obs: (
                            (m.b_px.0 as f64 - new_meta.width as f64 / 2.0) / new_meta.focal,
                            (m.b_px.1 as f64 - new_meta.height as f64 / 2.0) / new_meta.focal,
                        ),
                        world: wp,
                        seg: np,
                    });
                }
            }
            println!("  snapped to {} landmark pairs", a3.len());
            if a3.len() >= 10
                && let Some(s) = attempt_sim3(&a3, &b3, 0.08, 10, "pairwise 3D")
            {
                registration = Some(s);
                break 'regions;
            }
            if bridge.len() >= 8
                && let Some((r_wc, t_wc)) = poses.get(&kn)
                && let Some((s, inl)) = gs_pose::sim3::sim3_from_bridge(
                    *r_wc,
                    *t_wc,
                    &bridge,
                    8.0 / new_meta.focal,
                    0x9a1f,
                )
            {
                println!(
                    "  pairwise 2D: {inl}/{} inliers (scale {:.3})",
                    bridge.len(),
                    s.scale
                );
                if inl >= 8 && (0.05..=20.0).contains(&s.scale) {
                    registration = Some(s);
                    break 'regions;
                }
            }
        }
    }

    match &registration {
        Some(s) => println!("REGISTERED: scale {:.3}", s.scale),
        None => println!("no registration"),
    }
    if write
        && let Some(s) = registration
    {
        let dir = project::Project::submap_dir(project_root, submap);
        let mut meta = project::read_meta(&dir.join("meta.txt"))?;
        meta.world_from_submap = Some(project::WorldFromSubmap {
            scale: s.scale,
            quat: [s.rot.w, s.rot.x, s.rot.y, s.rot.z],
            trans: s.trans.to_array(),
        });
        project::write_meta(&dir.join("meta.txt"), &meta)?;
        println!("written to submap-{submap}/meta.txt");
    }
    Ok(())
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
