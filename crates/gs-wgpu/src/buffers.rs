//! Small buffer helpers shared by the sort, the renderer, and tests.

use wgpu::util::DeviceExt;

pub fn storage_init(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    })
}

pub fn storage_empty(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Blocking readback of a whole buffer (tests and golden paths only — never
/// on the interactive frame loop).
pub fn readback(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer) -> Vec<u8> {
    let size = src.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    rx.recv()
        .expect("map_async callback dropped")
        .expect("buffer map failed");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
}

/// Persistent asynchronous readback ring: a fixed set of reusable MAP_READ
/// staging buffers with tagged, FIFO-ordered completion. This is the hot-loop
/// counterpart of [`readback`]: record a copy into a slot inside the frame's
/// encoder, call [`ReadbackRing::map_pending`] right after submit, and collect
/// results a later iteration via the non-blocking [`ReadbackRing::poll_ready`]
/// — the GPU pipeline never drains. Results arrive with 1–2 iterations of
/// latency; callers must be designed to tolerate that.
pub struct ReadbackRing<T> {
    slots: Vec<RingSlot<T>>,
    /// Slot indices in issue order — completions are delivered FIFO so
    /// consumers see readbacks in the order they were encoded.
    pending: std::collections::VecDeque<usize>,
}

struct RingSlot<T> {
    buf: wgpu::Buffer,
    state: SlotState<T>,
}

enum SlotState<T> {
    Free,
    /// Copy recorded into an encoder that has not been submitted yet.
    Copied(T),
    /// map_async in flight.
    Mapping(T, std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>),
}

impl<T> ReadbackRing<T> {
    pub fn new(device: &wgpu::Device, label: &str, slot_size: u64, depth: usize) -> Self {
        assert!(depth > 0 && slot_size > 0);
        let slots = (0..depth)
            .map(|i| RingSlot {
                buf: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{label}-ring-{i}")),
                    size: slot_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: SlotState::Free,
            })
            .collect();
        Self {
            slots,
            pending: std::collections::VecDeque::new(),
        }
    }

    /// Number of readbacks issued but not yet delivered.
    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    /// Whether any pending tag satisfies `pred` (e.g. "is this view's pose
    /// update still in flight?").
    pub fn any_pending(&self, mut pred: impl FnMut(&T) -> bool) -> bool {
        self.slots.iter().any(|s| match &s.state {
            SlotState::Copied(t) | SlotState::Mapping(t, _) => pred(t),
            SlotState::Free => false,
        })
    }

    /// Record a copy of `size` bytes from `src` into a free slot. Returns
    /// false — recording nothing — when every slot is in flight; hot-loop
    /// callers should either skip the sample or drain first.
    pub fn encode_copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        src_offset: u64,
        size: u64,
        tag: T,
    ) -> bool {
        let Some(idx) = self
            .slots
            .iter()
            .position(|s| matches!(s.state, SlotState::Free))
        else {
            return false;
        };
        encoder.copy_buffer_to_buffer(src, src_offset, &self.slots[idx].buf, 0, size);
        self.slots[idx].state = SlotState::Copied(tag);
        self.pending.push_back(idx);
        true
    }

    /// Kick off `map_async` for every slot copied since the last call. Must
    /// run AFTER the encoder holding those copies has been submitted.
    pub fn map_pending(&mut self) {
        for slot in &mut self.slots {
            if matches!(slot.state, SlotState::Copied(_)) {
                let SlotState::Copied(tag) = std::mem::replace(&mut slot.state, SlotState::Free)
                else {
                    unreachable!()
                };
                let (tx, rx) = std::sync::mpsc::channel();
                slot.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                slot.state = SlotState::Mapping(tag, rx);
            }
        }
    }

    /// Non-blocking: advance the device and deliver completed readbacks in
    /// issue order. Stops at the first still-in-flight slot to preserve FIFO.
    pub fn poll_ready(&mut self, device: &wgpu::Device) -> Vec<(T, Vec<u8>)> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let _ = device.poll(wgpu::PollType::Poll);
        self.collect_ready()
    }

    /// Blocking: wait until every mapped slot has completed and deliver all
    /// of them (issue order). Slots still merely Copied (missing a
    /// `map_pending` call) are a caller bug and panic.
    pub fn drain_blocking(&mut self, device: &wgpu::Device) -> Vec<(T, Vec<u8>)> {
        assert!(
            !self
                .slots
                .iter()
                .any(|s| matches!(s.state, SlotState::Copied(_))),
            "ReadbackRing::drain_blocking with unmapped copies — call map_pending after submit"
        );
        let mut out = Vec::new();
        while !self.pending.is_empty() {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll failed");
            out.extend(self.collect_ready());
        }
        out
    }

    fn collect_ready(&mut self) -> Vec<(T, Vec<u8>)> {
        let mut out = Vec::new();
        while let Some(&idx) = self.pending.front() {
            let slot = &mut self.slots[idx];
            let SlotState::Mapping(_, rx) = &slot.state else {
                break; // Copied but not yet submitted — nothing to collect.
            };
            match rx.try_recv() {
                Ok(Ok(())) => {
                    let SlotState::Mapping(tag, _) =
                        std::mem::replace(&mut slot.state, SlotState::Free)
                    else {
                        unreachable!()
                    };
                    let data = slot
                        .buf
                        .slice(..)
                        .get_mapped_range()
                        .expect("mapped range")
                        .to_vec();
                    slot.buf.unmap();
                    self.pending.pop_front();
                    out.push((tag, data));
                }
                Ok(Err(e)) => panic!("ring readback map failed: {e:?}"),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(e) => panic!("ring readback channel closed: {e:?}"),
            }
        }
        out
    }
}
