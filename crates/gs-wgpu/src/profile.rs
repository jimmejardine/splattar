//! Timestamp-query bench harness (feature `profile`). Wraps compute/render
//! passes in labeled scopes; timings come back in milliseconds. Degrades to a
//! silent no-op when the adapter lacks TIMESTAMP_QUERY.

use crate::GpuContext;

/// Reborrow an optional timer into a compute-pass timestamp descriptor —
/// `timestamp_writes: scope(&mut timer, "label")` in pass descriptors.
pub fn scope<'a>(
    timer: &'a mut Option<&mut GpuTimer>,
    label: &str,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    timer.as_mut().and_then(|t| t.compute_scope(label))
}

pub struct GpuTimer {
    query_set: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    labels: Vec<String>,
    max_scopes: u32,
    period_ns: f32,
}

impl GpuTimer {
    pub fn new(ctx: &GpuContext, max_scopes: u32) -> Self {
        let enabled = ctx.device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if !enabled {
            log::warn!("TIMESTAMP_QUERY unavailable — GpuTimer disabled, timings will be empty");
        }
        let query_set = enabled.then(|| {
            ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu-timer"),
                ty: wgpu::QueryType::Timestamp,
                count: max_scopes * 2,
            })
        });
        let size = (max_scopes as u64 * 2) * 8;
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-timer-resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-timer-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve,
            readback,
            labels: Vec::new(),
            max_scopes,
            period_ns: ctx.queue.get_timestamp_period(),
        }
    }

    /// Begin a labeled compute scope; attach the result to the pass descriptor.
    pub fn compute_scope(&mut self, label: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let base = self.alloc(label)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: self.query_set.as_ref().unwrap(),
            beginning_of_pass_write_index: Some(base),
            end_of_pass_write_index: Some(base + 1),
        })
    }

    /// Begin a labeled render scope.
    pub fn render_scope(&mut self, label: &str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let base = self.alloc(label)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: self.query_set.as_ref().unwrap(),
            beginning_of_pass_write_index: Some(base),
            end_of_pass_write_index: Some(base + 1),
        })
    }

    fn alloc(&mut self, label: &str) -> Option<u32> {
        self.query_set.as_ref()?;
        if self.labels.len() as u32 >= self.max_scopes {
            log::warn!("GpuTimer: out of scopes, dropping '{label}'");
            return None;
        }
        self.labels.push(label.to_string());
        Some((self.labels.len() as u32 - 1) * 2)
    }

    /// Record query resolution; call after the timed passes, before submit.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(qs) = &self.query_set else { return };
        let n = self.labels.len() as u32 * 2;
        if n == 0 {
            return;
        }
        encoder.resolve_query_set(qs, 0..n, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, n as u64 * 8);
    }

    /// Blocking readback of all scopes recorded since the last read, in
    /// (label, milliseconds) order. Clears the scope list.
    pub fn read(&mut self, ctx: &GpuContext) -> Vec<(String, f64)> {
        if self.query_set.is_none() || self.labels.is_empty() {
            self.labels.clear();
            return Vec::new();
        }
        let n = self.labels.len() as u64 * 2;
        let slice = self.readback.slice(..n * 8);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        rx.recv().expect("map callback").expect("map failed");
        let mapped = slice.get_mapped_range().expect("mapped range");
        let ticks: Vec<u64> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.readback.unmap();

        self.labels
            .drain(..)
            .enumerate()
            .map(|(i, label)| {
                let dt = ticks[i * 2 + 1].saturating_sub(ticks[i * 2]);
                (label, dt as f64 * self.period_ns as f64 / 1e6)
            })
            .collect()
    }
}
