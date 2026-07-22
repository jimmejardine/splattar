//! GPU exclusive prefix sum over u32, multi-level (handles up to `capacity`
//! elements; three levels cover 1024³ ≈ 1e9). The grand total lands in
//! [`PrefixSum::total`] (a 1-element buffer) — tile binning copies it into
//! the sorter's count buffer without a CPU round-trip.

use crate::GpuContext;

const BLOCK: u64 = 1024;

struct Level {
    /// Elements this level scans (its data buffer) and its block-sums output.
    data: wgpu::Buffer,
    block_sums: wgpu::Buffer,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: u64,
}

pub struct PrefixSum {
    levels: Vec<Level>,
    scan_pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
}

impl PrefixSum {
    pub fn new(ctx: &GpuContext, capacity: u32) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefix-sum"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/prefix_sum.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix-sum-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
                storage(1),
                storage(2),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefix-sum-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let scan_pipeline = make("scan_blocks");
        let add_pipeline = make("add_back");

        // Build levels: data_0 (capacity) → sums_0 → sums_1 → … until one block.
        let mut levels = Vec::new();
        let mut cap = capacity.max(1) as u64;
        let mut data: Option<wgpu::Buffer> = None;
        loop {
            let nb = cap.div_ceil(BLOCK);
            let data_buf = data.take().unwrap_or_else(|| {
                crate::buffers::storage_empty(device, "prefix-data", cap.max(1) * 4)
            });
            let block_sums = crate::buffers::storage_empty(device, "prefix-sums", nb.max(1) * 4);
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("prefix-params"),
                size: 4,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prefix-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: data_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: block_sums.as_entire_binding(),
                    },
                ],
            });
            let next_data = block_sums.clone();
            levels.push(Level {
                data: data_buf,
                block_sums,
                params,
                bind_group,
                capacity: cap,
            });
            if nb <= 1 {
                break;
            }
            cap = nb;
            data = Some(next_data);
        }

        Self {
            levels,
            scan_pipeline,
            add_pipeline,
        }
    }

    /// The buffer whose first `n` elements get scanned in place.
    pub fn data(&self) -> &wgpu::Buffer {
        &self.levels[0].data
    }

    /// 1-element buffer holding the grand total after `encode`.
    pub fn total(&self) -> &wgpu::Buffer {
        &self.levels.last().unwrap().block_sums
    }

    /// Record an exclusive scan of the first `n` elements of [`data`].
    pub fn encode(&self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, n: u32) {
        assert!(n as u64 <= self.levels[0].capacity, "n exceeds capacity");
        // Per-level element counts.
        let mut counts = Vec::with_capacity(self.levels.len());
        let mut cur = n as u64;
        for _ in &self.levels {
            counts.push(cur as u32);
            cur = cur.div_ceil(BLOCK);
        }
        for (level, &count) in self.levels.iter().zip(&counts) {
            ctx.queue
                .write_buffer(&level.params, 0, bytemuck::bytes_of(&count));
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prefix-sum"),
            timestamp_writes: None,
        });
        // Downward: scan each level, producing the next level's input.
        pass.set_pipeline(&self.scan_pipeline);
        for (level, &count) in self.levels.iter().zip(&counts) {
            pass.set_bind_group(0, &level.bind_group, &[]);
            pass.dispatch_workgroups((count as u64).div_ceil(BLOCK).max(1) as u32, 1, 1);
        }
        // Upward: add scanned block sums back into each level below the top.
        pass.set_pipeline(&self.add_pipeline);
        for (level, &count) in self.levels.iter().zip(&counts).rev().skip(1) {
            pass.set_bind_group(0, &level.bind_group, &[]);
            pass.dispatch_workgroups((count as u64).div_ceil(BLOCK).max(1) as u32, 1, 1);
        }
    }
}

fn storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
