//! Diagnostics for the direct-SLAM pipeline.
//!
//! Per CLAUDE.md these are a design constraint, not a reporting layer: dB is a
//! scalar that hides where and why, and what goes wrong in direct SLAM is
//! spatial (which part of the frame the model fails to explain) and temporal
//! (when tracking began to slip). Neither survives averaging.
//!
//! The pipeline emits one [`FrameRecord`] per frame into a [`DiagStream`] and
//! never learns whether anything is watching, so every stage stays runnable
//! headless.

pub mod record;
pub mod window;

pub use record::{FrameRecord, Panel, TrackState};
pub use window::{DiagStream, run};
