//! Shared GPU infrastructure on wgpu: device/queue init, buffer helpers,
//! GPU radix sort (reused by both the trainer and the viewers), and the
//! timestamp-query bench harness (behind the `profile` feature).
//!
//! wgpu exposes one queue per device — parallelism lives CPU-side (see CLAUDE.md).

pub mod buffers;
pub mod context;
pub mod prefix_sum;
pub mod profile;
pub mod sort;

pub use buffers::ReadbackRing;
pub use context::{GpuContext, backends_from_str};
pub use prefix_sum::PrefixSum;
pub use profile::GpuTimer;
pub use sort::RadixSorter;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no compatible GPU adapter found: {0}")]
    NoAdapter(String),
    #[error("device request failed: {0}")]
    DeviceRequest(String),
    #[error("unknown backend `{0}` (expected vulkan, dx12, or gl)")]
    BadBackend(String),
}
