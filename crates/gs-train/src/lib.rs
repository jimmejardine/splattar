//! Training: Adam-in-WGSL over raw parameter classes, fused L1 + D-SSIM loss,
//! and the training loop for posed-sequence datasets (the M3 validation
//! harness; incremental anchor-out submap building arrives in M7).

pub mod init;
pub mod optim;
pub mod ssim;
pub mod trainer;

pub use init::init_from_sfm_points;
pub use optim::{Activation, Optimizer};
pub use ssim::SsimLoss;
pub use trainer::{ExportScene, InitialSurfels, TrainConfig, TrainView, Trainer};
