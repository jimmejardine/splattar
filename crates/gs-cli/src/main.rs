//! Headless driver for the direct-SLAM pipeline.
//!
//! Every stage runs here before it gets a GUI (CLAUDE.md). The diagnostic
//! window is optional on every command: the pipeline pushes records into a
//! [`DiagStream`] and never learns whether anything is watching.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
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
    }
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
/// Pull-based on purpose: the decoder does not race ahead of a paused window,
/// because a viewer that cannot hold the pipeline still is useless for finding
/// the frame where something went wrong.
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
        while stream.is_paused() && !stream.is_closed() {
            std::thread::sleep(std::time::Duration::from_millis(16));
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
