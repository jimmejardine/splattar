//! Uploads a SplatCloud into the SoA GPU buffers the viewer pipeline consumes.
//! f32 throughout for M0 (see PLAN.md — f16 SH packing is the first perf lever).

use glam::Vec3;
use gs_core::SplatCloud;
use gs_wgpu::{GpuContext, buffers};

/// Floats of SH storage per splat on the GPU: always padded to degree 3
/// (16 coeffs × 3 channels) so the shader layout is fixed; the active degree
/// is a uniform.
pub const SH_FLOATS_PER_SPLAT: usize = 48;

pub struct GpuScene {
    pub num_splats: u32,
    pub sh_degree: u8,
    pub bbox: (Vec3, Vec3),
    pub positions: wgpu::Buffer,
    pub sh: wgpu::Buffer,
    pub opacity: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub rotations: wgpu::Buffer,
}

impl GpuScene {
    pub fn upload(ctx: &GpuContext, cloud: &SplatCloud) -> Self {
        let n = cloud.len();
        assert!(n > 0, "refusing to upload an empty splat cloud");
        let device = &ctx.device;

        let mut positions = Vec::with_capacity(n * 4);
        for p in &cloud.positions {
            positions.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
        }

        let coeffs = cloud.sh_floats_per_splat();
        let mut sh = vec![0f32; n * SH_FLOATS_PER_SPLAT];
        for i in 0..n {
            sh[i * SH_FLOATS_PER_SPLAT..i * SH_FLOATS_PER_SPLAT + coeffs]
                .copy_from_slice(&cloud.sh[i * coeffs..(i + 1) * coeffs]);
        }

        let mut scales = Vec::with_capacity(n * 4);
        for s in &cloud.scales {
            scales.extend_from_slice(&[s.x, s.y, s.z, 0.0]);
        }
        let mut rotations = Vec::with_capacity(n * 4);
        for q in &cloud.rotations {
            rotations.extend_from_slice(&[q.x, q.y, q.z, q.w]);
        }

        Self {
            num_splats: n as u32,
            sh_degree: cloud.sh_degree,
            bbox: cloud.bbox().expect("non-empty"),
            positions: buffers::storage_init(device, "scene-positions", bytemuck::cast_slice(&positions)),
            sh: buffers::storage_init(device, "scene-sh", bytemuck::cast_slice(&sh)),
            opacity: buffers::storage_init(device, "scene-opacity", bytemuck::cast_slice(&cloud.opacity)),
            scales: buffers::storage_init(device, "scene-scales", bytemuck::cast_slice(&scales)),
            rotations: buffers::storage_init(device, "scene-rotations", bytemuck::cast_slice(&rotations)),
        }
    }
}
