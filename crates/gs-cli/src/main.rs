//! Headless pipeline driver. Every pipeline stage runs here before any GUI work.

use anyhow::bail;
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
    /// Render a splat .ply file in an interactive window (M0).
    View {
        /// Path to a .ply splat file.
        file: PathBuf,
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
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::View { file } => {
            bail!(
                "`view` lands with the M0 renderer (next commits) — asked to view {}",
                file.display()
            )
        }
        Command::Run { .. } => bail!("`run` arrives in M7 — see PLAN.md"),
        Command::Add { .. } => bail!("`add` arrives in M8 — see PLAN.md"),
        Command::Train { .. } => bail!("`train` arrives in M3 — see PLAN.md"),
        Command::Export { .. } => bail!("`export` arrives in M7 — see PLAN.md"),
    }
}
