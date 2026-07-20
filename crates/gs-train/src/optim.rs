//! Adam-in-WGSL over the trainer's parameter classes. Each class owns raw
//! parameters (identity/log/logit space), Adam moment buffers, and a view of
//! the rasterizer's activated buffer it feeds.

use gs_wgpu::{GpuContext, buffers};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity = 0,
    /// raw = log(activated) — scales.
    Exp = 1,
    /// raw = logit(activated) — opacity.
    Sigmoid = 2,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamUniform {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1_inv: f32,
    bc2_inv: f32,
    n: u32,
    activation: u32,
}

#[allow(dead_code)] // moment buffers are held to keep the bind group valid
pub struct ParamClass {
    pub name: &'static str,
    pub n: u32,
    pub activation: Activation,
    pub lr: f32,
    /// Raw parameters (identity classes alias the activated buffer's content
    /// layout but live in their own buffer; `activate` copies through).
    pub raw: wgpu::Buffer,
    m: wgpu::Buffer,
    v: wgpu::Buffer,
    uniform: wgpu::Buffer,
    step_bg: wgpu::BindGroup,
    act_bg: wgpu::BindGroup,
}

pub struct Optimizer {
    classes: Vec<ParamClass>,
    step_pipeline: wgpu::ComputePipeline,
    activate_pipeline: wgpu::ComputePipeline,
    t: u32,
}

impl Optimizer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adam"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/adam.wgsl").into()),
        });
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
        Self {
            classes: Vec::new(),
            step_pipeline: make("adam_step"),
            activate_pipeline: make("activate"),
            t: 0,
        }
    }

    /// Register a class. `grads` is the rasterizer's gradient buffer for the
    /// ACTIVATED values; `activated` is the buffer the rasterizer reads.
    /// Initial raw values are uploaded by the caller into the returned class's
    /// `raw` buffer (same element count as `activated`, flat f32).
    #[allow(clippy::too_many_arguments)] // one call site per parameter class
    pub fn add_class(
        &mut self,
        ctx: &GpuContext,
        name: &'static str,
        n: u32,
        activation: Activation,
        lr: f32,
        grads: &wgpu::Buffer,
        activated: &wgpu::Buffer,
    ) {
        let device = &ctx.device;
        let bytes = n as u64 * 4;
        let raw = buffers::storage_empty(device, &format!("adam-raw-{name}"), bytes);
        let m = buffers::storage_empty(device, &format!("adam-m-{name}"), bytes);
        let v = buffers::storage_empty(device, &format!("adam-v-{name}"), bytes);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adam-uniform"),
            size: std::mem::size_of::<AdamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &self.step_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: raw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grads.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: m.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: v.as_entire_binding() },
            ],
        });
        let act_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &self.activate_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: raw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: activated.as_entire_binding() },
            ],
        });
        self.classes.push(ParamClass {
            name,
            n,
            activation,
            lr,
            raw,
            m,
            v,
            uniform,
            step_bg,
            act_bg,
        });
    }

    pub fn class(&self, name: &str) -> &ParamClass {
        self.classes.iter().find(|c| c.name == name).unwrap()
    }

    pub fn set_lr(&mut self, name: &str, lr: f32) {
        self.classes.iter_mut().find(|c| c.name == name).unwrap().lr = lr;
    }

    fn write_uniforms(&self, ctx: &GpuContext, t: u32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        for c in &self.classes {
            let u = AdamUniform {
                lr: c.lr,
                beta1,
                beta2,
                eps: 1e-8,
                bc1_inv: 1.0 / (1.0 - beta1.powi(t as i32)),
                bc2_inv: 1.0 / (1.0 - beta2.powi(t as i32)),
                n: c.n,
                activation: c.activation as u32,
            };
            ctx.queue.write_buffer(&c.uniform, 0, bytemuck::bytes_of(&u));
        }
    }

    /// One Adam step for every class, then refresh activated buffers.
    pub fn encode_step(&mut self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        self.t += 1;
        self.write_uniforms(ctx, self.t);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("adam"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.step_pipeline);
        for c in &self.classes {
            pass.set_bind_group(0, &c.step_bg, &[]);
            pass.dispatch_workgroups(c.n.div_ceil(256), 1, 1);
        }
        pass.set_pipeline(&self.activate_pipeline);
        for c in &self.classes {
            pass.set_bind_group(0, &c.act_bg, &[]);
            pass.dispatch_workgroups(c.n.div_ceil(256), 1, 1);
        }
    }

    /// Refresh activated buffers without stepping (after uploading raw params).
    pub fn encode_activate(&self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        self.write_uniforms(ctx, self.t.max(1));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("activate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.activate_pipeline);
        for c in &self.classes {
            pass.set_bind_group(0, &c.act_bg, &[]);
            pass.dispatch_workgroups(c.n.div_ceil(256), 1, 1);
        }
    }
}
