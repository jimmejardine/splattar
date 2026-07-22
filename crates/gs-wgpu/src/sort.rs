//! GPU radix sort: 32-bit keys + 32-bit payloads, 4-bit digits, 8 passes.
//! The element count lives in a GPU buffer (written by whoever produced the
//! keys — the viewer's preprocess pass, or the CPU in tests); dispatch sizes
//! are derived on-GPU via `prep_dispatch`, so counts never round-trip.
//!
//! After `encode`, the sorted keys/payloads are back in the `keys()`/
//! `payloads()` buffers (8 passes = even number of ping-pongs).

use crate::GpuContext;

const BLOCK: u32 = 1024; // must match radix_sort.wgsl (WG 128 × EPT 8)
const DIGITS: u64 = 16;
const PASSES: u32 = 8;
/// Dynamic-offset stride for the per-pass shift uniform.
const UNIFORM_STRIDE: u64 = 256;

pub struct RadixSorter {
    capacity: u32,
    keys: [wgpu::Buffer; 2],
    payloads: [wgpu::Buffer; 2],
    counts: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    prep_bind_group: wgpu::BindGroup,
    prep_pipeline: wgpu::ComputePipeline,
    histogram_pipeline: wgpu::ComputePipeline,
    scan_columns_pipeline: wgpu::ComputePipeline,
    scan_totals_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
}

impl RadixSorter {
    pub fn new(ctx: &GpuContext, capacity: u32) -> Self {
        let device = &ctx.device;
        let max_blocks = capacity.div_ceil(BLOCK).max(1) as u64;

        let elems = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (capacity.max(1) as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let keys = [elems("sort-keys-a"), elems("sort-keys-b")];
        let payloads = [elems("sort-payloads-a"), elems("sort-payloads-b")];

        let counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort-counts"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let block_hists = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort-block-hists"),
            size: max_blocks * DIGITS * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let digit_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort-digit-offsets"),
            size: DIGITS * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let digit_totals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort-digit-totals"),
            size: DIGITS * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dispatch_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort-dispatch-args"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        // Per-pass shift values at 256-byte dynamic offsets.
        let mut shifts = vec![0u8; (PASSES as u64 * UNIFORM_STRIDE) as usize];
        for p in 0..PASSES {
            let shift = p * 4;
            shifts[(p as u64 * UNIFORM_STRIDE) as usize..][..4]
                .copy_from_slice(&shift.to_le_bytes());
        }
        let params = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sort-params"),
                contents: &shifts,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort-bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // counts
                storage_entry(2, true),  // keys_in
                storage_entry(3, true),  // payload_in
                storage_entry(4, false), // keys_out
                storage_entry(5, false), // payload_out
                storage_entry(6, false), // block_hists
                storage_entry(7, false), // digit_offsets
                storage_entry(8, false), // digit_totals
            ],
        });

        let make_bg = |label: &str, in_idx: usize, out_idx: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &params,
                            offset: 0,
                            size: wgpu::BufferSize::new(4),
                        }),
                    },
                    bind(1, &counts),
                    bind(2, &keys[in_idx]),
                    bind(3, &payloads[in_idx]),
                    bind(4, &keys[out_idx]),
                    bind(5, &payloads[out_idx]),
                    bind(6, &block_hists),
                    bind(7, &digit_offsets),
                    bind(8, &digit_totals),
                ],
            })
        };
        let bind_groups = [make_bg("sort-bg-even", 0, 1), make_bg("sort-bg-odd", 1, 0)];

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix-sort"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/radix_sort.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sort-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let histogram_pipeline = make_pipeline("histogram");
        let scan_columns_pipeline = make_pipeline("scan_columns");
        let scan_totals_pipeline = make_pipeline("scan_totals");
        let scatter_pipeline = make_pipeline("scatter");

        // prep_dispatch has its own tiny layout (counts + args).
        let prep_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sort-prep"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sort_prep.wgsl").into()),
        });
        let prep_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort-prep-bgl"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let prep_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sort-prep-pl"),
            bind_group_layouts: &[Some(&prep_layout)],
            immediate_size: 0,
        });
        let prep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prep_dispatch"),
            layout: Some(&prep_pl),
            module: &prep_module,
            entry_point: Some("prep_dispatch"),
            compilation_options: Default::default(),
            cache: None,
        });
        let prep_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort-prep-bg"),
            layout: &prep_layout,
            entries: &[bind(0, &counts), bind(1, &dispatch_args)],
        });

        Self {
            capacity,
            keys,
            payloads,
            counts,
            dispatch_args,
            bind_groups,
            prep_bind_group,
            prep_pipeline,
            histogram_pipeline,
            scan_columns_pipeline,
            scan_totals_pipeline,
            scatter_pipeline,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Input/output keys buffer (sorted in place across the 8 passes).
    pub fn keys(&self) -> &wgpu::Buffer {
        &self.keys[0]
    }

    /// Input/output payloads buffer.
    pub fn payloads(&self) -> &wgpu::Buffer {
        &self.payloads[0]
    }

    /// Element-count buffer (single u32). Written by the key producer.
    pub fn counts(&self) -> &wgpu::Buffer {
        &self.counts
    }

    /// Records the full sort. Caller submits the encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_inner(encoder, None, None);
    }

    /// Like [`encode`], with the two passes wrapped in GpuTimer scopes.
    pub fn encode_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timer: &mut crate::GpuTimer,
    ) {
        let prep = timer.compute_scope("sort-prep");
        // Scopes borrow the timer serially; materialize prep first.
        self.encode_prep(encoder, prep);
        let main = timer.compute_scope("sort");
        self.encode_main(encoder, main);
    }

    /// Diagnostic encoding: every stage of every digit pass in its own
    /// timestamped compute pass. Slower than `encode` (pass overhead) — for
    /// attributing cost only, never for production timing.
    pub fn encode_stage_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timer: &mut crate::GpuTimer,
    ) {
        let ts = timer.compute_scope("prep");
        self.encode_prep(encoder, ts);
        for p in 0..PASSES {
            let offset = (p as u64 * UNIFORM_STRIDE) as u32;
            let bg = &self.bind_groups[(p % 2) as usize];
            for (stage, pipeline) in [
                ("histogram", &self.histogram_pipeline),
                ("scan", &self.scan_columns_pipeline),
                ("totals", &self.scan_totals_pipeline),
                ("scatter", &self.scatter_pipeline),
            ] {
                let ts = timer.compute_scope(&format!("{stage}{p}"));
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(stage),
                    timestamp_writes: ts,
                });
                pass.set_bind_group(0, bg, &[offset]);
                pass.set_pipeline(pipeline);
                match stage {
                    "scan" => pass.dispatch_workgroups(DIGITS as u32, 1, 1),
                    "totals" => pass.dispatch_workgroups(1, 1, 1),
                    _ => pass.dispatch_workgroups_indirect(&self.dispatch_args, 0),
                }
            }
        }
    }

    fn encode_inner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prep_ts: Option<wgpu::ComputePassTimestampWrites<'_>>,
        main_ts: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        self.encode_prep(encoder, prep_ts);
        self.encode_main(encoder, main_ts);
    }

    fn encode_prep(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort-prep"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.prep_pipeline);
        pass.set_bind_group(0, &self.prep_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn encode_main(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("radix-sort"),
            timestamp_writes,
        });
        for p in 0..PASSES {
            let offset = (p as u64 * UNIFORM_STRIDE) as u32;
            let bg = &self.bind_groups[(p % 2) as usize];
            pass.set_bind_group(0, bg, &[offset]);
            pass.set_pipeline(&self.histogram_pipeline);
            pass.dispatch_workgroups_indirect(&self.dispatch_args, 0);
            pass.set_pipeline(&self.scan_columns_pipeline);
            pass.dispatch_workgroups(DIGITS as u32, 1, 1);
            pass.set_pipeline(&self.scan_totals_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.scatter_pipeline);
            pass.dispatch_workgroups_indirect(&self.dispatch_args, 0);
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(4),
        },
        count: None,
    }
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
