//! Device/queue initialization. Vulkan is the primary backend on the dev
//! machine; DX12 stays one flag away as an escape hatch (and to smoke out
//! driver-specific WGSL miscompiles, per CLAUDE.md).

use crate::GpuError;

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a headless context (no surface). `backends` usually comes from
    /// [`backends_from_str`].
    pub async fn new(backends: wgpu::Backends) -> Result<Self, GpuError> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle().with_env();
        desc.backends = backends;
        let instance = wgpu::Instance::new(desc);
        Self::with_instance(instance, None).await
    }

    /// Create a context against an existing instance, optionally compatible
    /// with a surface (the windowed path hands its surface in here).
    pub async fn with_instance(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, GpuError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::NoAdapter(e.to_string()))?;

        let info = adapter.get_info();
        log::info!(
            "adapter: {} ({:?}, driver: {} {})",
            info.name,
            info.backend,
            info.driver,
            info.driver_info
        );

        // The degree-3 SH buffer for ~2M splats is ~372 MB — far beyond the
        // 128 MiB default binding limit. Request what the adapter offers.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            // The viewer preprocess binds 9+ storage buffers (default limit: 8).
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .min(16),
            ..wgpu::Limits::default()
        };

        // Optional features: request whatever the adapter offers. Timestamps
        // feed the per-kernel training/bench timers (GpuTimer no-ops without
        // them); subgroups let the backward rasterizer pre-reduce gradient
        // adds warp-wide before hitting the shared-memory CAS loop (kernels
        // select their variant from device.features()). Code paths must
        // handle either feature being absent. SPLATTAR_NO_SUBGROUPS=1 forces
        // the scalar-CAS fallback — the emergency disable, and how the
        // fallback path is exercised in gradient checks on subgroup-capable
        // hardware.
        let mut required_features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let no_subgroups = std::env::var_os("SPLATTAR_NO_SUBGROUPS").is_some_and(|v| v != "0");
        if adapter.features().contains(wgpu::Features::SUBGROUP) && !no_subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("splattar-device"),
                required_features,
                required_limits,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

/// Parse a `--backend` CLI value. Empty/None → Vulkan primary with DX12+GL
/// fallback so a broken Vulkan ICD doesn't brick the viewer.
pub fn backends_from_str(s: Option<&str>) -> Result<wgpu::Backends, GpuError> {
    Ok(match s {
        None => wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL,
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("dx12") => wgpu::Backends::DX12,
        Some("gl") => wgpu::Backends::GL,
        Some(other) => return Err(GpuError::BadBackend(other.to_string())),
    })
}
