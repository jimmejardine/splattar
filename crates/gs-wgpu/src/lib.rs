//! Shared GPU infrastructure on wgpu: device/queue init, buffer helpers,
//! GPU radix sort and prefix sum (reused by both the trainer and the viewers),
//! and the timestamp-query bench harness (behind the `profile` feature).
//!
//! wgpu exposes one queue per device — parallelism lives CPU-side (see CLAUDE.md).
