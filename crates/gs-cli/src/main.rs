//! Headless driver for the direct-SLAM pipeline.
//!
//! Every stage runs here before it gets a GUI (CLAUDE.md). The diagnostic
//! window is optional on every command: the pipeline pushes records into a
//! [`DiagStream`] and never learns whether anything is watching.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
use glam::{Quat, Vec3};
use gs_diag::{DiagStream, FrameRecord, Panel};

#[derive(Parser)]
#[command(name = "gs-cli", about = "splattar — direct SLAM over walkthrough video")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Decode a video and show it in the diagnostic window. The M0 sanity
    /// check: proves decode, PTS handling and the diagnostic path end to end.
    Play {
        /// Path to an H.264/H.265 .mp4.
        video: PathBuf,
        /// Stop after this many frames (0 = all).
        #[arg(long, default_value_t = 0)]
        max_frames: usize,
        /// Decode without opening a window — for CI and timing.
        #[arg(long)]
        headless: bool,
        /// Records kept for scrubbing. Each is a full RGBA frame, so this
        /// bounds memory, not history: the disk trace preserves history.
        #[arg(long, default_value_t = 600)]
        history: usize,
    },
    /// Render a synthetic room and show frame | render | error, with the
    /// render's camera deliberately displaced. The M1 sanity check: proves the
    /// differentiable rasterizer and all three diagnostic panes, and previews
    /// exactly what M2's tracker has to close.
    Render {
        /// Camera displacement, metres. 0 makes the error pane go black —
        /// which is itself the test that the two paths agree.
        #[arg(long, default_value_t = 0.15)]
        offset: f32,
        /// Camera rotation error, degrees.
        #[arg(long, default_value_t = 2.0)]
        rotate: f32,
        /// Orbit steps rendered, so the panes animate rather than freeze.
        #[arg(long, default_value_t = 240)]
        steps: usize,
        #[arg(long)]
        headless: bool,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match Cli::parse().command {
        Command::Play {
            video,
            max_frames,
            headless,
            history,
        } => play(video, max_frames, headless, history),
        Command::Render {
            offset,
            rotate,
            steps,
            headless,
        } => render(offset, rotate, steps, headless),
    }
}

/// M1: render a known room from a known path, and from a displaced camera,
/// and show the disagreement.
fn render(offset: f32, rotate: f32, steps: usize, headless: bool) -> anyhow::Result<()> {
    let stream = DiagStream::new(steps.max(2));
    if headless {
        return render_into(offset, rotate, steps, &stream);
    }
    let producer = {
        let stream = Arc::clone(&stream);
        std::thread::spawn(move || {
            if let Err(e) = render_into(offset, rotate, steps, &stream) {
                log::error!("render failed: {e:#}");
            }
        })
    };
    gs_diag::run(Arc::clone(&stream))?;
    let _ = producer.join();
    Ok(())
}

fn render_into(
    offset: f32,
    rotate: f32,
    steps: usize,
    stream: &DiagStream,
) -> anyhow::Result<()> {
    const W: u32 = 480;
    const H: u32 = 480;
    let ctx = gs_map::gpu()?;
    let map = gs_map::synthetic::room(24, 3.0);
    let gpu = gs_map::GpuMap::new(&ctx, &map, map.len() as u32, W, H);
    log::info!("synthetic room: {} surfels at {W}x{H}", map.len());

    // Focal for a ~60 degree vertical field of view.
    let focal = (H as f32 * 0.5) / (30.0_f32.to_radians()).tan();
    let start = std::time::Instant::now();
    let mut residuals: Vec<f32> = Vec::with_capacity(steps);
    for i in 0..steps {
        if stream.is_closed() {
            break;
        }
        // A slow orbit inside the room, looking outward at the walls.
        let t = i as f32 / steps as f32 * std::f32::consts::TAU;
        let truth = gs_kernels::RasterCamera {
            center: Vec3::new(t.cos() * 0.8, 0.0, t.sin() * 0.8),
            quat: Quat::from_rotation_y(-t),
            focal,
            sh_degree: 0,
        };
        // The displaced camera: what a tracker would see before it converges.
        let wrong = gs_kernels::RasterCamera {
            center: truth.center + Vec3::new(offset, offset * 0.5, 0.0),
            quat: truth.quat * Quat::from_rotation_x(rotate.to_radians()),
            ..truth
        };

        let truth_rgba = gpu.render_f32(&ctx, &truth);
        let wrong_rgba = gpu.render_f32(&ctx, &wrong);
        let err = gs_map::abs_error(&truth_rgba, &wrong_rgba);
        let mean: f32 = err.iter().sum::<f32>() / err.len() as f32;

        let mut rec = FrameRecord::captured(
            i,
            i as f64 / 30.0,
            gs_map::to_panel(W, H, &truth_rgba),
        );
        rec.render = Some(gs_map::to_panel(W, H, &wrong_rgba));
        // x4 so a small residual is still visible: the errors that matter for
        // tracking are the ones too small to see on a linear scale.
        rec.error = Some(Panel::heatmap(W, H, &err, 4.0));
        rec.pose = Some((wrong.center, wrong.quat));
        rec.residual = vec![mean];
        rec.surfels = gpu.live();
        residuals.push(mean);
        stream.push(rec);
    }
    let n = residuals.len().max(1) as f32;
    let mean = residuals.iter().sum::<f32>() / n;
    let peak = residuals.iter().copied().fold(0.0f32, f32::max);
    log::info!(
        "rendered {} frames in {:.1?} ({:.0} fps, 2 renders each)",
        residuals.len(),
        start.elapsed(),
        residuals.len() as f64 / start.elapsed().as_secs_f64().max(1e-9)
    );
    // The headless equivalent of looking at the error pane. With no camera
    // displacement this must be zero: the two renders are the same call with
    // the same inputs, so anything else means the render is not deterministic
    // and every later measurement would be built on sand.
    log::info!("photometric residual: mean {mean:.6}, peak {peak:.6}");
    if offset == 0.0 && rotate == 0.0 {
        anyhow::ensure!(
            peak == 0.0,
            "identical cameras produced a non-zero residual ({peak:.6}) — the              render is not deterministic"
        );
        log::info!("identical cameras agree exactly, as they must");
    }
    Ok(())
}

fn play(video: PathBuf, max_frames: usize, headless: bool, history: usize) -> anyhow::Result<()> {
    let stream = DiagStream::new(history);

    if headless {
        return decode_into(&video, max_frames, &stream);
    }

    // winit needs the event loop on the main thread, so decoding runs on a
    // worker. The window owns nothing the decoder needs and vice versa — they
    // share only the record stream.
    let producer = {
        let stream = Arc::clone(&stream);
        let video = video.clone();
        std::thread::spawn(move || {
            if let Err(e) = decode_into(&video, max_frames, &stream) {
                log::error!("decode failed: {e:#}");
            }
        })
    };
    gs_diag::run(Arc::clone(&stream))?;
    // The window has closed; `decode_into` notices via `is_closed` and returns.
    let _ = producer.join();
    Ok(())
}

/// Decode every frame into diagnostic records.
///
/// `push` applies the backpressure: it blocks while the buffer holds records
/// the viewer has not reached. That keeps the decoder from racing ahead and
/// evicting frames nobody has looked at, without this function needing to know
/// anything about playback state.
fn decode_into(video: &std::path::Path, max_frames: usize, stream: &DiagStream) -> anyhow::Result<()> {
    let mut reader = gs_video::VideoReader::open(video)
        .with_context(|| format!("opening {}", video.display()))?;
    let total = reader.sample_count();
    log::info!("{}: {total} samples", video.display());

    let start = std::time::Instant::now();
    let mut n = 0usize;
    let mut last_pts = f64::NAN;
    while let Some(frame) = reader.next_frame()? {
        if stream.is_closed() {
            break;
        }
        let panel = rgba_panel(&frame);
        stream.push(FrameRecord::captured(n, frame.pts, panel));
        last_pts = frame.pts;
        n += 1;
        if max_frames > 0 && n >= max_frames {
            break;
        }
        if n.is_multiple_of(200) {
            log::info!(
                "decoded {n} frames ({:.1} fps), pts {:.2}s",
                n as f64 / start.elapsed().as_secs_f64(),
                frame.pts
            );
        }
    }
    log::info!(
        "decoded {n} frames in {:.1?} ({:.1} fps); last pts {last_pts:.2}s",
        start.elapsed(),
        n as f64 / start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// YUV 4:2:0 → RGBA8 for display.
///
/// Deliberately separate from the f32 path the tracker will need: this one is
/// for looking at, and stays 8-bit so a record is cheap enough to keep hundreds
/// of in memory for scrubbing.
fn rgba_panel(frame: &gs_video::DecodedFrame) -> Panel {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let rgba_f32 = gs_video::color::yuv420_to_rgba_f32(&frame.y, &frame.u, &frame.v, w, h);
    let rgba: Vec<u8> = rgba_f32
        .iter()
        .flat_map(|p| {
            [
                (p[0] * 255.0).clamp(0.0, 255.0) as u8,
                (p[1] * 255.0).clamp(0.0, 255.0) as u8,
                (p[2] * 255.0).clamp(0.0, 255.0) as u8,
                255,
            ]
        })
        .collect();
    Panel::new(frame.width, frame.height, rgba)
}
