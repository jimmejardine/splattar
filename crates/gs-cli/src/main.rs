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
    /// Ingest a video into a project (created on first use): VO → per-segment
    /// Sim(3) edge registration or island → train → new submap(s). The only
    /// ingestion command — order of adds doesn't matter (no submap has a
    /// privileged gauge; placement is resolved per connected component).
    Add {
        /// Path to an H.264/H.265 .mp4 walkthrough video.
        video: PathBuf,
        /// Project directory (created if absent; default: `gs-project`
        /// next to the video).
        #[arg(long)]
        project: Option<PathBuf>,
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
        /// Fraction of training during which pose refinement runs (1.0 =
        /// full run; the LR decays on the position schedule either way).
        #[arg(long, default_value_t = 1.0)]
        pose_window: f32,
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
        #[arg(long, default_value_t = 0.001)]
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
    /// Re-bundle a submap's persisted geometry with the trainer-measured
    /// focal (removes the guess-focal warp that blocks cross-submap Sim(3)
    /// registration). Rewrites landmarks.bin / poses.csv / meta in place.
    Refocal {
        project: PathBuf,
        #[arg(long)]
        submap: usize,
        /// Corrected focal in pixels (default: focal_refined from meta).
        #[arg(long)]
        focal: Option<f64>,
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
                // shared world, islands offset side by side). A directory
                // that merely CONTAINS a default `gs-project` (e.g. the
                // video folder) works too.
                let root = if !file.join("submap-0").is_dir()
                    && file.join("gs-project").join("submap-0").is_dir()
                {
                    file.join("gs-project")
                } else {
                    file.clone()
                };
                compose_project(&root)?.0
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
            pose_window,
        } => {
            // Default project dir lives next to the video, not in the cwd.
            let project = project.unwrap_or_else(|| {
                video
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(""))
                    .join("gs-project")
            });
            run_add(
                &video, &project, focal, max_frames, iters, budget, downscale, max_views,
                pose_window,
            )
        }
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
        Command::Refocal {
            project,
            submap,
            focal,
        } => run_refocal(&project, submap, focal),
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
    /// Tracking-resolution convention: focal/cx/cy, landmark observations,
    /// poses, and thumbs all live at `track_size`; `track_scale` maps back
    /// to decoded pixels (applied once, at the training boundary).
    intrinsics: gs_pose::vo::Intrinsics,
    /// Decoded frame size (full resolution).
    video_size: (u32, u32),
    /// Size KLT actually tracked at (long side capped; ≤ video_size).
    track_size: (u32, u32),
    /// video_size / track_size integer factor.
    track_scale: u32,
    /// Half-res snapshots of every 4th keyframe (pairwise registration).
    thumbs: Vec<gs_pose::vo::Thumb>,
}

fn solved_count(seg: &gs_pose::VoResult) -> usize {
    seg.keyframe_poses.iter().flatten().count()
}

/// KLT tracks at most this long-side resolution (integer decimation of the
/// decoded luma) — subpixel LK doesn't need full phone resolution, and the
/// causal pass is tracking-bound, not decode-bound.
const TRACK_MAX_SIDE: u32 = 960;

/// Integer box-average decimation of a tight-packed luma plane.
fn downscale_luma(y: &[u8], w: usize, f: usize) -> Vec<u8> {
    let h = y.len() / w;
    let (tw, th) = (w / f, h / f);
    let mut out = Vec::with_capacity(tw * th);
    let norm = (f * f) as u32;
    for ty in 0..th {
        for tx in 0..tw {
            let mut acc = 0u32;
            for sy in 0..f {
                let row = (ty * f + sy) * w + tx * f;
                for sx in 0..f {
                    acc += y[row + sx] as u32;
                }
            }
            out.push((acc / norm) as u8);
        }
    }
    out
}

/// Pure per-frame prep (everything the tracker needs that does NOT depend
/// on tracking state): decimated pyramid + sharpness.
struct PrepFrame {
    pyr: gs_pose::Pyramid,
    sharpness: f32,
    pts: f64,
    /// FNV of the tracking-res luma when SPLATTAR_KF_TRACE is set (isolates
    /// pixel nondeterminism from tracking nondeterminism per frame).
    luma_fnv: u64,
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn run_vo(
    video: &std::path::Path,
    focal: Option<f64>,
    max_frames: u32,
) -> anyhow::Result<VoOutput> {
    use gs_pose::vo::{Intrinsics, VoConfig, VoFrontEnd};
    use std::sync::{Arc, Mutex, mpsc};
    let mut reader = gs_video::Mp4H264Reader::open(video).context("open video")?;
    let t0 = std::time::Instant::now();

    // Three-stage pipeline: NVDEC decode stays on this thread (the Vulkan
    // session isn't Send) → a pool of prep workers builds each frame's
    // pyramid + sharpness (pure per-frame work, serial per frame — the
    // parallelism is ACROSS frames) → the tracking spine consumes prepared
    // frames strictly in index order, so results are identical to the old
    // serial loop. Only the KLT step truly depends on the previous frame.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .saturating_sub(2)
        .clamp(1, 12);
    let (raw_tx, raw_rx) = mpsc::sync_channel::<(u64, Vec<u8>, u32, u32, f64)>(workers * 2);
    let raw_rx = Arc::new(Mutex::new(raw_rx));
    let (prep_tx, prep_rx) = mpsc::sync_channel::<(u64, PrepFrame)>(workers * 2);
    let n_levels = VoConfig::default().pyramid_levels;
    let prep_pool: Vec<_> = (0..workers)
        .map(|_| {
            let rx = Arc::clone(&raw_rx);
            let tx = prep_tx.clone();
            std::thread::spawn(move || {
                loop {
                    let msg = { rx.lock().unwrap().recv() };
                    let Ok((idx, y, width, height, pts)) = msg else { break };
                    // Tracking-resolution cap is a pure function of the frame
                    // size, so workers need no shared state.
                    let s = width.max(height).div_ceil(TRACK_MAX_SIDE) as usize;
                    let trace = std::env::var_os("SPLATTAR_KF_TRACE").is_some();
                    let (gray, luma_fnv) = if s > 1 {
                        let small = downscale_luma(&y, width as usize, s);
                        let h = if trace { fnv64(&small) } else { 0 };
                        (
                            gs_pose::GrayImage::from_luma8(
                                &small,
                                width as usize / s,
                                height as usize / s,
                            ),
                            h,
                        )
                    } else {
                        let h = if trace { fnv64(&y) } else { 0 };
                        (
                            gs_pose::GrayImage::from_luma8(
                                &y,
                                width as usize,
                                height as usize,
                            ),
                            h,
                        )
                    };
                    let pyr = gs_pose::Pyramid::build(gray, n_levels);
                    let sharpness = gs_pose::vo::gradient_energy(
                        &pyr.levels[1.min(pyr.levels.len() - 1)],
                    );
                    if tx
                        .send((idx, PrepFrame { pyr, sharpness, pts, luma_fnv }))
                        .is_err()
                    {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(prep_tx); // spine's channel closes when the last worker exits

    let spine = std::thread::spawn(move || {
        let mut fe: Option<VoFrontEnd> = None;
        let mut pending: std::collections::HashMap<u64, PrepFrame> =
            std::collections::HashMap::new();
        let mut next = 0u64;
        let mut luma_fnvs: Vec<u64> = Vec::new();
        while let Ok((idx, frame)) = prep_rx.recv() {
            pending.insert(idx, frame);
            while let Some(f) = pending.remove(&next) {
                luma_fnvs.push(f.luma_fnv);
                let fe = fe.get_or_insert_with(|| {
                    let (tw, th) = (f.pyr.levels[0].width, f.pyr.levels[0].height);
                    let fpx = focal.unwrap_or(0.85 * tw.max(th) as f64);
                    log::info!("tracking at {tw}x{th}, focal guess {fpx:.0}px");
                    let mut cfg = VoConfig {
                        intrinsics: Intrinsics {
                            focal: fpx,
                            cx: tw as f64 / 2.0,
                            cy: th as f64 / 2.0,
                        },
                        ..Default::default()
                    };
                    // Tuning override for keyframe density experiments.
                    if let Ok(v) = std::env::var("SPLATTAR_KF_FLOW_FRAC")
                        && let Ok(frac) = v.parse::<f32>()
                    {
                        log::info!("kf_flow_frac override: {frac}");
                        cfg.kf_flow_frac = frac;
                    }
                    VoFrontEnd::new(cfg)
                });
                fe.push_prepared(f.pyr, f.sharpness, f.pts);
                next += 1;
                if next.is_multiple_of(200) {
                    log::info!("tracked {next} frames, {} keyframes", fe.keyframes.len());
                }
            }
        }
        (fe, next, luma_fnvs)
    });

    let mut sent = 0u64;
    let mut sent_size = (0u32, 0u32);
    let (mut decode_s, mut send_s) = (0.0f64, 0.0f64);
    // Debug isolation: FNV over every decoded luma plane, to tell decode
    // nondeterminism from tracking nondeterminism.
    let hash_decode = std::env::var_os("SPLATTAR_HASH_DECODE").is_some();
    let mut fnv = 0xcbf29ce484222325u64;
    loop {
        let t = std::time::Instant::now();
        let Some(frame) = reader.next_frame().context("decode")? else { break };
        decode_s += t.elapsed().as_secs_f64();
        if hash_decode {
            for &b in &frame.y {
                fnv = (fnv ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        sent_size = (frame.width, frame.height);
        if sent == 0 {
            log::info!("video {}x{}", frame.width, frame.height);
        }
        let t = std::time::Instant::now();
        let sent_ok = raw_tx
            .send((sent, frame.y, frame.width, frame.height, frame.pts))
            .is_ok();
        send_s += t.elapsed().as_secs_f64();
        if !sent_ok {
            break; // pipeline died; the panic resurfaces at join
        }
        sent += 1;
        if max_frames > 0 && sent >= max_frames as u64 {
            break;
        }
    }
    if hash_decode {
        log::info!("decoded-luma FNV: {fnv:016x}");
    }
    if let Some((read, submit, wait, post)) = reader.decode_timing() {
        log::info!(
            "decode thread: {decode_s:.1}s in next_frame ({read:.1} read / {submit:.1} submit / {wait:.1} fence-wait / {post:.1} post), {send_s:.1}s blocked on send"
        );
    } else {
        log::info!("decode thread: {decode_s:.1}s in next_frame, {send_s:.1}s blocked on send");
    }
    let video_size = (sent_size.0, sent_size.1);
    drop(raw_tx);
    for w in prep_pool {
        w.join().expect("prep worker panicked");
    }
    let (vo, n, luma_fnvs) = spine.join().expect("tracking spine panicked");
    let mut fe = vo.context("no frames decoded")?;
    let intrinsics = fe.intrinsics();
    let track_scale = video_size.0.max(video_size.1).div_ceil(TRACK_MAX_SIDE);
    let track_size = (video_size.0 / track_scale, video_size.1 / track_scale);
    let decode_track_s = t0.elapsed().as_secs_f64();
    let p = fe.promotions;
    let tm = fe.timing;
    log::info!(
        "causal pass: {n} frames, {} keyframes ({} flow / {} survival / {} low-tracks), {:.1} fps \
         [spine: klt {:.1}s, desc {:.1}s, detect {:.1}s]",
        fe.keyframes.len(),
        p.flow,
        p.survival,
        p.tracks,
        n as f64 / decode_track_s,
        tm.klt_s,
        tm.desc_s,
        tm.detect_s
    );
    if let (Ok(path), Some(trace)) = (std::env::var("SPLATTAR_KF_TRACE"), &fe.kf_trace) {
        let mut s = String::new();
        for (i, (flow, surv, live)) in trace.iter().enumerate() {
            let fnv = luma_fnvs.get(i).copied().unwrap_or(0);
            s.push_str(&format!("{i},{flow},{surv},{live},{fnv:016x}\n"));
        }
        let _ = std::fs::write(&path, s);
        log::info!("kf trace written to {path}");
    }
    if std::env::var_os("SPLATTAR_CAUSAL_ONLY").is_some() {
        anyhow::bail!("causal-only run (SPLATTAR_CAUSAL_ONLY set) — no solve");
    }

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
        track_size,
        track_scale,
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
                // obs is in tracking coords; targets are decode-res / ds.
                let ts = vo.track_scale as usize;
                let x = ((px as usize * ts) / ds).min(tw - 1);
                let y = ((py as usize * ts) / ds).min(th - 1);
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

    // Cameras in the renderer convention; every 8th view held out. The VO
    // focal is in tracking-res pixels — lift to decode res, then downscale.
    let focal_t = (vo.intrinsics.focal * vo.track_scale as f64 / ds as f64) as f32;
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
) -> anyhow::Result<(f64, usize, f64)> {
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
        // 0.01 costs ~5 dB even correctly normalized (RESULTS.md).
        lambda_dist: 0.001,
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
    Ok((psnr, scene.num, trainer.focal_scale as f64))
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

/// The ingestion command: video → VO → per-segment registration-or-island →
/// train → new submap(s). Creates the project directory on first use. No
/// submap has a privileged gauge; each successful registration is stored as
/// a pairwise Sim(3) EDGE to a concrete existing submap, and world placement
/// is resolved per connected component at compose time — so the order in
/// which videos are added doesn't matter (submap indices and archipelago
/// layout follow add order, which is presentation-only).
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
    pose_window: f32,
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
            pose_window,
        ) {
            log::warn!("segment {si}: skipped ({e:#})");
        }
    }
    let proj = project::Project::load_or_empty(project_root)?;
    let placements = project::resolve_placements(&proj);
    let components = placements.iter().map(|p| p.component).max().map_or(0, |c| c + 1);
    println!(
        "project {}: {} submap(s) in {} component(s) — walk it with `gs-cli view {}`",
        project_root.display(),
        proj.submaps.len(),
        components,
        project_root.display()
    );
    Ok(())
}

/// Load every submap's landmarks in LOCAL coordinates, deduped per submap.
fn load_project_landmarks(
    project_root: &std::path::Path,
    proj: &project::Project,
) -> anyhow::Result<Vec<Vec<DbLandmark>>> {
    let mut locals = Vec::with_capacity(proj.submaps.len());
    for i in 0..proj.submaps.len() {
        let lms = project::read_landmarks(
            &project::Project::submap_dir(project_root, i).join("landmarks.bin"),
        )?;
        let entries: Vec<DbLandmark> = lms
            .into_iter()
            .map(|l| DbLandmark {
                pos: glam::DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64),
                desc: l.desc,
                kf: l.kf,
                obs: (l.obs[0], l.obs[1]),
                submap: i,
            })
            .collect();
        locals.push(dedup_db_landmarks(entries));
    }
    Ok(locals)
}

/// Pool every submap's landmarks (islands included — no privileged gauge)
/// into its component-root frame via the resolved placements.
fn build_component_db(
    locals: &[Vec<DbLandmark>],
    placements: &[project::Placement],
) -> Vec<DbLandmark> {
    let mut db = Vec::new();
    for (i, lms) in locals.iter().enumerate() {
        for l in lms {
            db.push(DbLandmark {
                pos: placements[i].world.apply(l.pos),
                desc: l.desc,
                kf: l.kf,
                obs: l.obs,
                submap: i,
            });
        }
    }
    db
}

/// Register one VO segment against the project and persist it as a new
/// trained submap — with a pairwise Sim(3) edge to an existing submap when
/// registration succeeds, as an island otherwise. Reloads the project each
/// call so segments landed earlier are bridge targets for later ones. The
/// first submap of a fresh project is trivially an island.
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
    pose_window: f32,
) -> anyhow::Result<()> {
    use glam::DVec3;

    let proj = project::Project::load_or_empty(project_root)?;
    let placements = project::resolve_placements(&proj);
    // Per-submap LOCAL landmarks (bridge targets) + the component-frame DB
    // (global matching). Spatial dedup per submap: KLT respawns
    // re-triangulate the same corner dozens of times, and coincident
    // duplicates let a collapse transform out-vote the true registration.
    let locals = load_project_landmarks(project_root, &proj)?;
    let db = build_component_db(&locals, &placements);
    log::info!("project DB: {} landmarks after dedup", db.len());

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
    // viewpoint change, descriptors reliable; yields a seg→submap-local edge
    // directly), then covisibility-voted global matching (yields a
    // seg→component-root transform, folded to an edge on the winning
    // submap). No success → island (first-class, per PLAN).
    let seg_range = seg_kf_range(seg);
    let mut registration: Option<(usize, gs_pose::sim3::Sim3G)> =
        try_bridge_registration(&proj, video, seg, &vo.intrinsics, seg_range, &locals);
    if registration.is_none() {
        registration = try_global_registration(&db, &new, &placements).map(|(target, s)| {
            // s maps seg → component-root frame; the stored edge must map
            // seg → target-submap-local.
            (target, placements[target].world.inverse().compose(&s))
        });
    }

    let (idx, dir) = project::Project::next_submap_dir(project_root)?;
    let edges = registration
        .map(|(to, s)| vec![project::Sim3Edge::from_sim3(to as u32, &s)])
        .unwrap_or_default();
    let edge_to = edges.first().map(|e| e.to);
    project::write_meta(
        &dir.join("meta.txt"),
        &project::SubmapMeta {
            video: video.display().to_string(),
            // Meta stays in the VO tracking-res convention (matches the
            // persisted observations, poses, and thumbs).
            focal: vo.intrinsics.focal,
            width: vo.track_size.0,
            height: vo.track_size.1,
            kf_range: seg_kf_range(seg),
            focal_refined: None,
            edges,
        },
    )?;
    project::write_landmarks(&dir.join("landmarks.bin"), &prepared.landmarks)?;
    write_trajectory_csv(&dir.join("trajectory.csv"), vo)?;
    write_seg_poses(&dir.join("poses.csv"), seg)?;
    write_thumbs(&dir, &vo.thumbs, seg_kf_range(seg))?;

    let ply = dir.join("splat.ply");
    let (psnr, num, fscale) = train_and_bake(prepared, iters, budget, pose_window, &ply)?;
    update_refined_focal(&dir, fscale)?;
    println!(
        "submap-{idx}: {num} surfels, held-out PSNR {psnr:.2} dB — {}",
        match edge_to {
            Some(t) => format!("EDGE to submap-{t} (overlap found)"),
            None => "no overlap found: ISLAND (film a bridge clip to connect it)".into(),
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
) -> Option<(gs_pose::sim3::Sim3G, Vec<usize>)> {
    // Same phone, comparable walking pace: cross-submap gauge scale beyond
    // [0.2, 5] is physically implausible, and unconstrained search lets
    // near-collapse models out-vote the truth on ambiguous pools.
    use gs_pose::sim3::register_point_sets_bounded;
    if a.len() < min_inliers {
        return None;
    }
    let centroid = b.iter().copied().sum::<glam::DVec3>() / b.len() as f64;
    let mut d: Vec<f64> = b.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(f64::total_cmp);
    let thresh = (thresh_frac * d[d.len() / 2]).max(1e-9);
    let (sim3, inliers) =
        register_point_sets_bounded(a, b, 1000, thresh, 0x5133, (0.2, 5.0))?;
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
        .then_some((sim3, inliers))
}

/// Temporal bridge: consecutive segments of the SAME video are separated by
/// a track-loss cut, but their boundary keyframes view the same space seconds
/// apart — small viewpoint change, where the patch descriptors are reliable.
/// Match only boundary-window landmarks against each temporally adjacent
/// submap (islands included — no privileged gauge). Candidate landmarks are
/// in submap-LOCAL coordinates, so a success is directly the pairwise edge
/// (segment coords → submap-i coords).
fn try_bridge_registration(
    proj: &project::Project,
    video: &std::path::Path,
    seg: &gs_pose::VoResult,
    intr: &gs_pose::vo::Intrinsics,
    seg_range: Option<(u32, u32)>,
    locals: &[Vec<DbLandmark>],
) -> Option<(usize, gs_pose::sim3::Sim3G)> {
    use gs_pose::descriptor::match_descriptors;
    const WINDOW: u32 = 60; // keyframes each side of the cut
    const MAX_GAP: u32 = 200; // dropped stretches between segments can be long

    let (s0, s1) = seg_range?;
    let video_str = video.display().to_string();
    for (i, meta) in proj.submaps.iter().enumerate() {
        if meta.video != video_str {
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
        let w_sub: Vec<&DbLandmark> = locals[i]
            .iter()
            .filter(|l| l.kf.abs_diff(world_edge) <= WINDOW)
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
            if let Some((s, _)) = attempt_sim3(&a, &b, 0.15, 8, &format!("bridge s{i}")) {
                return Some((i, s));
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
                            return Some((i, s));
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
/// pairs, noise scatters. The Sim(3) is estimated PER COMPONENT (pooling
/// across components would feed points from unrelated frames into one
/// RANSAC). Returns the target submap (mode of the inliers' owners) and the
/// transform seg → that submap's COMPONENT-ROOT frame.
fn try_global_registration(
    world: &[DbLandmark],
    new: &[DbLandmark],
    placements: &[project::Placement],
) -> Option<(usize, gs_pose::sim3::Sim3G)> {
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
    // Group by connected component, biggest match pool first.
    let mut by_comp: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for &(x, y) in &filtered {
        by_comp
            .entry(placements[world[y].submap].component)
            .or_default()
            .push((x, y));
    }
    let mut comps: Vec<(usize, Vec<(usize, usize)>)> = by_comp.into_iter().collect();
    comps.sort_by_key(|(c, v)| (std::cmp::Reverse(v.len()), *c));
    for (comp, set) in comps {
        if set.len() < 20 {
            continue;
        }
        let a: Vec<glam::DVec3> = set.iter().map(|&(x, _)| new[x].pos).collect();
        let b: Vec<glam::DVec3> = set.iter().map(|&(_, y)| world[y].pos).collect();
        if let Some((sim3, inliers)) =
            attempt_sim3(&a, &b, 0.05, 25, &format!("covis c{comp}"))
        {
            // Target = the submap owning the most inlier landmarks.
            let mut counts: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &i in &inliers {
                *counts.entry(world[set[i].1].submap).or_default() += 1;
            }
            let target = counts
                .into_iter()
                .max_by_key(|&(s, n)| (n, std::cmp::Reverse(s)))
                .map(|(s, _)| s)?;
            return Some((target, sim3));
        }
    }
    None
}

/// Record the trainer-measured focal in the submap meta (geometry itself
/// is corrected separately by `refocal`).
fn update_refined_focal(dir: &std::path::Path, focal_scale: f64) -> anyhow::Result<()> {
    let meta_path = dir.join("meta.txt");
    let mut meta = project::read_meta(&meta_path)?;
    meta.focal_refined = Some(meta.focal * focal_scale);
    project::write_meta(&meta_path, &meta)
}

/// The focal re-BA: rebuild the submap's bundle problem from disk (poses +
/// landmarks + full observation lists ARE the BA graph), re-normalize the
/// observations with the corrected focal, re-optimize, and rewrite the
/// geometry. Removes the guess-focal warp that makes independently built
/// submaps non-Sim(3)-comparable (measured: no registration consensus at
/// ~6% focal error; see RESULTS.md).
fn run_refocal(
    project_root: &std::path::Path,
    submap: usize,
    focal_override: Option<f64>,
) -> anyhow::Result<()> {
    let dir = project::Project::submap_dir(project_root, submap);
    let mut meta = project::read_meta(&dir.join("meta.txt"))?;
    let new_focal = focal_override
        .or(meta.focal_refined)
        .context("no refined focal in meta — pass --focal or retrain")?;
    log::info!(
        "refocal submap-{submap}: {:.1} -> {:.1} px",
        meta.focal,
        new_focal
    );

    let mut landmarks = project::read_landmarks(&dir.join("landmarks.bin"))?;
    let pose_map = load_seg_poses(&dir.join("poses.csv"))?;
    let mut kfs: Vec<u32> = pose_map.keys().copied().collect();
    kfs.sort_unstable();
    let cam_of_kf: std::collections::HashMap<u32, usize> =
        kfs.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut poses: Vec<(glam::DQuat, glam::DVec3)> =
        kfs.iter().map(|k| pose_map[k]).collect();

    let (cx, cy) = (meta.width as f64 / 2.0, meta.height as f64 / 2.0);
    let mut lm_pos: Vec<glam::DVec3> = landmarks
        .iter()
        .map(|l| glam::DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64))
        .collect();
    let mut obs: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (li, l) in landmarks.iter().enumerate() {
        for (kf, p) in &l.obs_all {
            let Some(&cam) = cam_of_kf.get(kf) else { continue };
            obs.push((
                cam,
                li,
                (p[0] as f64 - cx) / new_focal,
                (p[1] as f64 - cy) / new_focal,
            ));
        }
    }
    log::info!(
        "re-BA: {} poses, {} landmarks, {} observations",
        poses.len(),
        lm_pos.len(),
        obs.len()
    );
    let (before, after) =
        gs_pose::ba::refine_submap_glam(&mut poses, &mut lm_pos, &obs, 30);
    log::info!("re-BA cost: {before:.3e} -> {after:.3e}");

    // Rewrite geometry: landmarks, poses, meta focal.
    for (l, p) in landmarks.iter_mut().zip(&lm_pos) {
        l.pos = [p.x as f32, p.y as f32, p.z as f32];
    }
    project::write_landmarks(&dir.join("landmarks.bin"), &landmarks)?;
    let mut csv = String::from("kf,cx,cy,cz,qw,qx,qy,qz\n");
    for (i, k) in kfs.iter().enumerate() {
        let (r_wc, t_wc) = poses[i];
        // Stored form: center + camera-to-world rotation.
        let q_c2w = r_wc.conjugate();
        let center = -(q_c2w * t_wc);
        csv.push_str(&format!(
            "{k},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8}\n",
            center.x, center.y, center.z, q_c2w.w, q_c2w.x, q_c2w.y, q_c2w.z
        ));
    }
    std::fs::write(dir.join("poses.csv"), csv)?;
    meta.focal = new_focal;
    meta.focal_refined = Some(new_focal);
    project::write_meta(&dir.join("meta.txt"), &meta)?;
    println!(
        "submap-{submap} re-bundled at focal {new_focal:.1} px (cost {before:.2e} -> {after:.2e})"
    );
    Ok(())
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
    let placements = project::resolve_placements(&proj);

    let load = |i: usize| -> anyhow::Result<Vec<project::Landmark>> {
        project::read_landmarks(
            &project::Project::submap_dir(project_root, i).join("landmarks.bin"),
        )
    };
    // World: every other submap (islands included), each placed in its
    // component-root frame. Raw per-submap lists are retained for
    // observation snapping in the pairwise stage.
    let mut world: Vec<DbLandmark> = Vec::new();
    let mut world_raw: Vec<(usize, Vec<project::Landmark>)> = Vec::new();
    let mut world_sims: std::collections::HashMap<usize, Sim3G> =
        std::collections::HashMap::new();
    for (i, pl) in placements.iter().enumerate() {
        if i == submap {
            continue;
        }
        let s = pl.world;
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

    // Registration result: (target submap, Sim3 submap-local → target's
    // component-root frame).
    let mut registration: Option<(usize, Sim3G)> = None;
    if filtered.len() >= 15 {
        // Per-component RANSAC — pooling across components would mix
        // unrelated placement frames into one model.
        let mut by_comp: std::collections::HashMap<usize, Vec<(usize, usize)>> =
            std::collections::HashMap::new();
        for &(x, y) in &filtered {
            by_comp
                .entry(placements[world[y].submap].component)
                .or_default()
                .push((x, y));
        }
        let mut comps: Vec<_> = by_comp.into_iter().collect();
        comps.sort_by_key(|(c, v)| (std::cmp::Reverse(v.len()), *c));
        for (comp, set) in comps {
            if set.len() < 15 {
                continue;
            }
            let a: Vec<DVec3> = set.iter().map(|&(x, _)| new_pts[x]).collect();
            let b: Vec<DVec3> = set.iter().map(|&(_, y)| world[y].pos).collect();
            if let Some((s, inliers)) =
                attempt_sim3(&a, &b, 0.05, 15, &format!("register-lab 3D c{comp}"))
            {
                let mut counts: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::new();
                for &i in &inliers {
                    *counts.entry(world[set[i].1].submap).or_default() += 1;
                }
                if let Some(target) = counts
                    .into_iter()
                    .max_by_key(|&(t, n)| (n, std::cmp::Reverse(t)))
                    .map(|(t, _)| t)
                {
                    registration = Some((target, s));
                    break;
                }
            }
        }
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
                        registration = Some((dom.0, s));
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
        use gs_pose::pairwise::PairwiseConfig;
        let sub_dir = |i: usize| project::Project::submap_dir(project_root, i);
        let new_thumbs = list_thumbs(&sub_dir(submap));
        let new_meta = &proj.submaps[submap];
        let poses = load_seg_poses(&sub_dir(submap).join("poses.csv")).unwrap_or_default();

        // Snap sets come from PROJECTING every landmark into the keyframe
        // through its known (gauge-local) pose — matching against the sparse
        // stored KLT observations starved the stage (measured 0-2 snaps per
        // 11-13 verified corners; fresh dense corners rarely coincide with
        // tracking corners).
        let per_submap_poses = |i: usize| load_seg_poses(&sub_dir(i).join("poses.csv"));
        let project_set = |raw: &[project::Landmark],
                           pose: &(glam::DQuat, glam::DVec3),
                           meta: &project::SubmapMeta,
                           to_world: Option<&gs_pose::sim3::Sim3G>|
         -> Vec<(f32, f32, DVec3, gs_pose::descriptor::MultiDescriptor)> {
            let (r, t) = pose;
            let mut out = Vec::new();
            for l in raw {
                let p = DVec3::new(l.pos[0] as f64, l.pos[1] as f64, l.pos[2] as f64);
                let c = *r * p + *t;
                if c.z < 0.1 {
                    continue;
                }
                let px = (c.x / c.z * meta.focal + meta.width as f64 / 2.0) as f32;
                let py = (c.y / c.z * meta.focal + meta.height as f64 / 2.0) as f32;
                if px < 0.0 || py < 0.0 || px >= meta.width as f32 || py >= meta.height as f32
                {
                    continue;
                }
                let stored = to_world.map(|s| s.apply(p)).unwrap_or(p);
                out.push((px, py, stored, l.desc));
            }
            out
        };
        // Appearance-guided snap: among landmark projections within a
        // generous radius, take the one whose stored descriptor matches the
        // verified corner's — distance alone grabs the wrong landmark in
        // repeated indoor texture (measured: 41 pooled snaps, 6 consistent).
        let snap = |list: &[(f32, f32, DVec3, gs_pose::descriptor::MultiDescriptor)],
                    px: (f32, f32),
                    desc: &gs_pose::descriptor::MultiDescriptor|
         -> Option<DVec3> {
            let (mut best, mut best_d) = (None, 61u32);
            for (x, y, p, ld) in list {
                let d2 = (x - px.0).powi(2) + (y - px.1).powi(2);
                if d2 > 18.0 * 18.0 {
                    continue;
                }
                let hd = gs_pose::descriptor::hamming_multi(desc, ld);
                if hd < best_d {
                    best_d = hd;
                    best = Some(*p);
                }
            }
            best
        };

        // Verified+snapped pairs pooled across ALL regions (they share the
        // same two gauges — per-pair counts are small but they add up).
        let mut pool_a: Vec<DVec3> = Vec::new();
        let mut pool_b: Vec<DVec3> = Vec::new();
        let mut pool_submap: Option<usize> = None;

        let nearest_two = |kfs: &[u32], target: u32| -> Vec<u32> {
            let mut v: Vec<u32> = kfs.to_vec();
            v.sort_by_key(|k| k.abs_diff(target));
            v.truncate(2);
            v
        };
        // 2D bridge observations pooled per new-side keyframe across every
        // region and thumb combo — one camera, exact 2D, robust-median scale.
        let mut bridge_by_kn: std::collections::HashMap<u32, Vec<gs_pose::sim3::BridgeObs>> =
            std::collections::HashMap::new();
        // Relative camera poses from verified pairs — the snap-free Sim(3)
        // route (one entry per verified image pair).
        let mut rel_pairs: Vec<gs_pose::sim3::RelPair> = Vec::new();

        'regions: for &((nk, ws, wk), v) in &top_regions {
            let w_thumbs = list_thumbs(&sub_dir(ws));
            let kns = nearest_two(&new_thumbs, nk * BUCKET + BUCKET / 2);
            let kws = nearest_two(&w_thumbs, wk * BUCKET + BUCKET / 2);
            for &kn in &kns {
            for &kw in &kws {
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
            let (verified, rel) =
                gs_pose::pairwise::match_image_pair_with_pose(&w_img, &n_img, &cfg2);
            println!(
                "pairwise ({v} votes): world s{ws} kf {kw} <-> kf {kn}: {} verified",
                verified.len()
            );
            // Projected snap sets for this specific keyframe pair.
            let w_meta = &proj.submaps[ws];
            let Some(w_pose) = per_submap_poses(ws).ok().and_then(|p| p.get(&kw).copied())
            else {
                continue;
            };
            let Some(n_pose) = poses.get(&kn).copied() else { continue };
            let w_raw = world_raw
                .iter()
                .find(|(i, _)| *i == ws)
                .map(|(_, r)| r.as_slice())
                .unwrap_or(&[]);
            let w_proj = project_set(w_raw, &w_pose, w_meta, Some(&world_sims[&ws]));
            let n_proj = project_set(&new_raw, &n_pose, new_meta, None);
            if log::log_enabled!(log::Level::Debug) {
                let min_d = |list: &[(f32, f32, DVec3, gs_pose::descriptor::MultiDescriptor)], px: (f32, f32)| -> f32 {
                    list.iter()
                        .map(|&(x, y, _, _)| ((x - px.0).powi(2) + (y - px.1).powi(2)).sqrt())
                        .fold(f32::INFINITY, f32::min)
                };
                let dw: Vec<i32> =
                    verified.iter().map(|m| min_d(&w_proj, m.a_px) as i32).collect();
                let dn: Vec<i32> =
                    verified.iter().map(|m| min_d(&n_proj, m.b_px) as i32).collect();
                log::debug!(
                    "snap diag: {} w-proj / {} n-proj; corner→proj px dists w{dw:?} n{dn:?}",
                    w_proj.len(),
                    n_proj.len()
                );
            }
            if verified.len() < 10 {
                continue;
            }
            // Collect the relative-pose constraint (world camera = submap
            // gauge pose pushed through its Sim(3) as an effective camera:
            // R^w = R^g·R_sᵀ, t^w = σ·t^g − R^w·t_s).
            if let Some(rp) = rel
                && (pool_submap.is_none() || pool_submap == Some(ws))
            {
                let s = &world_sims[&ws];
                let (rg, tg) = w_pose;
                let rw = rg * s.rot.conjugate();
                let tw = s.scale * tg - rw * s.trans;
                rel_pairs.push(gs_pose::sim3::RelPair {
                    cam_a_world: (rw, tw),
                    cam_b_seg: n_pose,
                    rel_rot: rp.rot,
                    rel_tdir: rp.tdir,
                });
            }
            // Snap verified endpoints to landmark observations.
            let mut a3 = Vec::new(); // new/seg side
            let mut b3 = Vec::new(); // world side
            let mut bridge = Vec::new();
            for m in &verified {
                let wsnap = snap(&w_proj, m.a_px, &m.a_desc);
                let nsnap = snap(&n_proj, m.b_px, &m.b_desc);
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
            if pool_submap.is_none() || pool_submap == Some(ws) {
                pool_submap = Some(ws);
                pool_a.extend_from_slice(&a3);
                pool_b.extend_from_slice(&b3);
                bridge_by_kn.entry(kn).or_default().extend(bridge);
            }
            if a3.len() >= 8
                && let Some((s, _)) = attempt_sim3(&a3, &b3, 0.08, 8, "pairwise 3D")
            {
                registration = Some((ws, s));
                break 'regions;
            }
            } // kw
            } // kn
        }

        // Snap-free first: Sim(3) from relative camera poses alone.
        if registration.is_none() && rel_pairs.len() >= 2 {
            if let Some((s, rms)) = gs_pose::sim3::sim3_from_relative_pairs(&rel_pairs) {
                println!(
                    "relative-pose Sim(3): {} pairs, rms {rms:.3}, scale {:.3}",
                    rel_pairs.len(),
                    s.scale
                );
                // Residual gate relative to the world scene scale.
                let cen =
                    pool_b.iter().copied().sum::<DVec3>() / pool_b.len().max(1) as f64;
                let scene = pool_b
                    .iter()
                    .map(|p| (*p - cen).length())
                    .fold(0.0f64, f64::max)
                    .max(1.0);
                // Rel-pair collection is gated to a single world submap, so
                // pool_submap is the attribution. NOTE: the rms gate scales
                // off that single submap's extent — if the single-submap
                // gate is ever loosened, per-component monocular scales
                // break this silently.
                if rms < 0.08 * scene
                    && (0.2..=5.0).contains(&s.scale)
                    && let Some(ps) = pool_submap
                {
                    registration = Some((ps, s));
                }
            } else {
                println!(
                    "relative-pose Sim(3): {} pairs, rotations inconsistent",
                    rel_pairs.len()
                );
            }
        }

        // Pooled attempts across all regions: 3D Umeyama first, then the 2D
        // depth-ratio bridge per new-side keyframe with enough observations.
        if registration.is_none()
            && pool_a.len() >= 10
            && let Some(ps) = pool_submap
        {
            println!("pooled pairwise: {} pairs across regions", pool_a.len());
            registration = attempt_sim3(&pool_a, &pool_b, 0.18, 8, "pairwise pooled")
                .map(|(s, _)| (ps, s));
        }
        if registration.is_none() {
            let mut kns: Vec<_> = bridge_by_kn.iter().collect();
            kns.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
            for (kn, bridge) in kns {
                if bridge.len() < 8 {
                    break;
                }
                let Some((r_wc, t_wc)) = poses.get(kn) else { continue };
                if let Some((s, inl)) = gs_pose::sim3::sim3_from_bridge(
                    *r_wc,
                    *t_wc,
                    bridge,
                    8.0 / new_meta.focal,
                    0x9a1f,
                ) {
                    println!(
                        "pooled 2D bridge via kf {kn}: {inl}/{} inliers (scale {:.3})",
                        bridge.len(),
                        s.scale
                    );
                    if inl >= 8
                        && (0.2..=5.0).contains(&s.scale)
                        && let Some(ps) = pool_submap
                    {
                        registration = Some((ps, s));
                        break;
                    }
                }
            }
        }
    }

    match &registration {
        Some((t, s)) => println!("REGISTERED to submap-{t}: scale {:.3}", s.scale),
        None => println!("no registration"),
    }
    if write
        && let Some((t, s)) = registration
    {
        // s maps submap-local → target's component-root frame; fold the
        // target's own placement out to get the pairwise edge submap → t.
        let edge = placements[t].world.inverse().compose(&s);
        let dir = project::Project::submap_dir(project_root, submap);
        let mut meta = project::read_meta(&dir.join("meta.txt"))?;
        meta.edges = vec![project::Sim3Edge::from_sim3(t as u32, &edge)];
        project::write_meta(&dir.join("meta.txt"), &meta)?;
        println!("edge to submap-{t} written to submap-{submap}/meta.txt");
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
            "    {{\"video\": {:?}, \"component\": {}, \"surfel_start\": {}, \
             \"surfel_count\": {}, \"offset_x\": {:.4}, \
             \"bbox_min\": [{:.4}, {:.4}, {:.4}], \"bbox_max\": [{:.4}, {:.4}, {:.4}]}}{}\n",
            pl.video,
            pl.component,
            pl.surfels.start,
            pl.surfels.len(),
            pl.offset_x,
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
    /// Connected component this submap belongs to (archipelago view).
    component: usize,
    surfels: std::ops::Range<usize>,
    /// Presentation x-offset applied to this submap's whole component
    /// (0 for component 0; never stored — recomputed every composition).
    offset_x: f32,
    /// World-space AABB after placement (offset included).
    bbox: ([f32; 3], [f32; 3]),
}

/// Compose a project's submaps into one SplatCloud: each submap goes through
/// its resolved placement into its component-root frame; connected
/// components are then laid side by side along +x (presentation-only offset,
/// never stored — per the two-tier data-model rule).
fn compose_project(
    root: &std::path::Path,
) -> anyhow::Result<(gs_core::SplatCloud, Vec<SubmapPlacement>)> {
    use glam::{Quat, Vec3};

    let proj = project::Project::load(root)?;
    let resolved = project::resolve_placements(&proj);

    // Pass 1: load every submap, apply its placement transform, gather
    // per-component x extents for the archipelago layout.
    let mut clouds: Vec<gs_core::SplatCloud> = Vec::new();
    let mut comp_extent: std::collections::BTreeMap<usize, (f32, f32)> =
        std::collections::BTreeMap::new();
    for (i, pl) in resolved.iter().enumerate() {
        let ply = project::Project::submap_dir(root, i).join("splat.ply");
        let contents = gs_io::load_ply(&ply)
            .with_context(|| format!("loading {}", ply.display()))?;
        let gs_io::PlyContents::Splats(mut cloud) = contents else {
            bail!("{} is not a splat file", ply.display());
        };

        let w = &pl.world;
        let s = w.scale as f32;
        let rot = Quat::from_xyzw(
            w.rot.x as f32,
            w.rot.y as f32,
            w.rot.z as f32,
            w.rot.w as f32,
        );
        let t = Vec3::new(w.trans.x as f32, w.trans.y as f32, w.trans.z as f32);
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

        let e = comp_extent
            .entry(pl.component)
            .or_insert((f32::MAX, f32::MIN));
        for p in &cloud.positions {
            e.0 = e.0.min(p.x);
            e.1 = e.1.max(p.x);
        }
        clouds.push(cloud);
    }

    // Component layout: component 0 keeps its frame; each further component
    // is shifted so its min-x sits one gap past the previous one's max-x.
    let mut comp_offset: std::collections::HashMap<usize, f32> =
        std::collections::HashMap::new();
    let mut cursor: Option<f32> = None;
    for (&c, &(min_x, max_x)) in &comp_extent {
        let width = (max_x - min_x).max(0.0);
        let dx = match cursor {
            None => 0.0,
            Some(right) => {
                let gap = width * 0.15 + 1.0;
                right + gap - min_x
            }
        };
        comp_offset.insert(c, dx);
        cursor = Some(max_x + dx);
        if dx != 0.0 {
            log::info!("component {c} placed at +x offset {dx:.1} (archipelago view)");
        }
    }

    // Pass 2: apply offsets, merge in submap order, record placements.
    let mut merged: Option<gs_core::SplatCloud> = None;
    let mut placements: Vec<SubmapPlacement> = Vec::new();
    for (i, mut cloud) in clouds.into_iter().enumerate() {
        let comp = resolved[i].component;
        let dx = comp_offset[&comp];
        if dx != 0.0 {
            for p in &mut cloud.positions {
                p.x += dx;
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
            video: proj.submaps[i].video.clone(),
            component: comp,
            surfels: start..start + cloud.positions.len(),
            offset_x: dx,
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
