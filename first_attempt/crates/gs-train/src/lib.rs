//! Training: Adam-in-WGSL over raw parameter classes, fused L1 + D-SSIM loss,
//! and the training loop for posed-sequence datasets (the M3 validation
//! harness; incremental anchor-out submap building arrives in M7).

pub mod appearance;
pub mod init;
pub mod normal_loss;
pub mod optim;
pub mod plane_sweep;
pub mod ssim;
pub mod trainer;

pub use init::{init_from_sfm_points, upsample_to_budget};
pub use plane_sweep::{DepthMap, PlaneSweep, SweepOptions, init_from_plane_sweep};
pub use normal_loss::NormalLoss;
pub use optim::{Activation, Optimizer};
pub use ssim::SsimLoss;
pub use trainer::{ExportScene, InitialSurfels, NewSurfel, TrainConfig, TrainView, Trainer};
