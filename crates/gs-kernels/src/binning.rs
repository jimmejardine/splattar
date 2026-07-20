//! Tile binning for the training rasterizer: items (tile-space AABB + depth
//! key) → per-tile entry lists, grouped by tile and depth-ascending within
//! each tile.
//!
//! Invariant this module depends on: the gs-wgpu radix sort is **stable**, so
//! sorting by depth first and then by tile id is equivalent to one 64-bit
//! (tile ‖ depth) sort. If sort stability is ever broken, binning breaks —
//! the binning property test is the canary.

use gs_wgpu::{GpuContext, PrefixSum, RadixSorter};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileRect {
    /// Inclusive tile-space AABB: min x/y, max x/y. max < min ⇒ culled item.
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

pub struct TileBinner {
    max_items: u32,
    max_entries: u32,
    num_tiles: u32,
    params: wgpu::Buffer,
    pub rects: wgpu::Buffer,
    pub depths: wgpu::Buffer,
    /// (start, end) per tile after `encode`.
    pub ranges: wgpu::Buffer,
    prefix: PrefixSum,
    sorter: RadixSorter,
    count_pipeline: wgpu::ComputePipeline,
    count_bg: wgpu::BindGroup,
    expand_pipeline: wgpu::ComputePipeline,
    expand_bg: wgpu::BindGroup,
    gather_pipeline: wgpu::ComputePipeline,
    gather_bg: wgpu::BindGroup,
    ranges_pipeline: wgpu::ComputePipeline,
    ranges_bg: wgpu::BindGroup,
}

impl TileBinner {
    pub fn new(ctx: &GpuContext, max_items: u32, max_entries: u32, num_tiles: u32) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile-binning"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tile_binning.wgsl").into()),
        });
        let prefix = PrefixSum::new(ctx, max_items);
        let sorter = RadixSorter::new(ctx, max_entries);

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binning-params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rects = gs_wgpu::buffers::storage_empty(device, "binning-rects", max_items as u64 * 16);
        let depths = gs_wgpu::buffers::storage_empty(device, "binning-depths", max_items as u64 * 4);
        let tile_keys =
            gs_wgpu::buffers::storage_empty(device, "binning-tile-keys", max_entries as u64 * 4);
        let ranges =
            gs_wgpu::buffers::storage_empty(device, "binning-ranges", num_tiles as u64 * 8);

        // Auto layouts: each entry point binds exactly what it uses.
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let count_pipeline = make("count_tiles");
        let expand_pipeline = make("expand");
        let gather_pipeline = make("gather_keys");
        let ranges_pipeline = make("tile_ranges");

        let count_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binning-count-bg"),
            layout: &count_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &params),
                bind(1, &rects),
                bind(4, prefix.data()), // counts land in the prefix-sum input
            ],
        });
        let expand_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binning-expand-bg"),
            layout: &expand_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(0, &params),
                bind(1, &rects),
                bind(2, &depths),
                bind(3, prefix.data()), // scanned offsets
                bind(4, &tile_keys),
                bind(5, sorter.keys()),
                bind(6, sorter.payloads()),
            ],
        });
        let gather_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binning-gather-bg"),
            layout: &gather_pipeline.get_bind_group_layout(0),
            entries: &[
                bind(4, &tile_keys),
                bind(5, sorter.keys()),
                bind(6, sorter.payloads()),
                bind(7, sorter.counts()),
            ],
        });
        let ranges_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binning-ranges-bg"),
            layout: &ranges_pipeline.get_bind_group_layout(0),
            entries: &[bind(5, sorter.keys()), bind(7, sorter.counts()), bind(8, &ranges)],
        });

        Self {
            max_items,
            max_entries,
            num_tiles,
            params,
            rects,
            depths,
            ranges,
            prefix,
            sorter,
            count_pipeline,
            count_bg,
            expand_pipeline,
            expand_bg,
            gather_pipeline,
            gather_bg,
            ranges_pipeline,
            ranges_bg,
        }
    }

    /// Sorted entry payloads (entry indices, grouped by tile, depth-ascending).
    pub fn sorted_entries(&self) -> &wgpu::Buffer {
        self.sorter.payloads()
    }

    /// Entry count buffer (single u32, filled from the prefix-sum total).
    pub fn entry_count(&self) -> &wgpu::Buffer {
        self.sorter.counts()
    }

    /// Record the full binning pipeline for `num_items` items on a grid
    /// `tiles_x` wide. Caller has already written `rects` and `depths`.
    pub fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        num_items: u32,
        tiles_x: u32,
    ) {
        assert!(num_items <= self.max_items);
        assert!(tiles_x > 0 && self.num_tiles.is_multiple_of(tiles_x));
        ctx.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&[num_items, tiles_x]));

        let item_groups = num_items.div_ceil(256).max(1);
        let entry_groups = self.max_entries.div_ceil(256).max(1);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binning-count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &self.count_bg, &[]);
            pass.dispatch_workgroups(item_groups, 1, 1);
        }
        self.prefix.encode(ctx, encoder, num_items);
        // Total entry count → the sorter's GPU-side count.
        encoder.copy_buffer_to_buffer(self.prefix.total(), 0, self.sorter.counts(), 0, 4);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binning-expand"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.expand_pipeline);
            pass.set_bind_group(0, &self.expand_bg, &[]);
            pass.dispatch_workgroups(item_groups, 1, 1);
        }
        self.sorter.encode(encoder); // stable sort by depth key
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binning-gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &self.gather_bg, &[]);
            pass.dispatch_workgroups(entry_groups, 1, 1);
        }
        self.sorter.encode(encoder); // stable sort by tile id — groups tiles
        encoder.clear_buffer(&self.ranges, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binning-ranges"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ranges_pipeline);
            pass.set_bind_group(0, &self.ranges_bg, &[]);
            pass.dispatch_workgroups(entry_groups, 1, 1);
        }
    }
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
