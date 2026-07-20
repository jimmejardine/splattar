//! GPU radix sort correctness: exact agreement with a CPU stable sort on
//! order, payload pairing, and stability (duplicate keys keep input order).
//! Skips with a visible warning when no adapter is available.

use gs_wgpu::{GpuContext, RadixSorter};

fn context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(wgpu::Backends::all())) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPING GPU sort tests: {e}");
            None
        }
    }
}

/// Deterministic xorshift so failures reproduce.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

#[test]
fn sorts_match_cpu_stable_sort() {
    let Some(ctx) = context() else { return };
    const CAPACITY: u32 = 2_000_000;
    let sorter = RadixSorter::new(&ctx, CAPACITY);

    for &n in &[0u32, 1, 255, 256, 1024, 4097, 100_000, 2_000_000] {
        let mut rng = 0x1234_5678u32 ^ n.wrapping_mul(2654435761);
        // Heavy duplicates (key % 8192) so stability is actually exercised;
        // payload = original index makes stability checkable exactly.
        let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut rng) % 8192).collect();
        let payloads: Vec<u32> = (0..n).collect();

        ctx.queue
            .write_buffer(sorter.keys(), 0, bytemuck::cast_slice(&keys));
        ctx.queue
            .write_buffer(sorter.payloads(), 0, bytemuck::cast_slice(&payloads));
        ctx.queue
            .write_buffer(sorter.counts(), 0, bytemuck::bytes_of(&n));

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        sorter.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);

        let got_keys: Vec<u32> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            sorter.keys(),
        ))[..n as usize]
            .to_vec();
        let got_payloads: Vec<u32> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
            &ctx.device,
            &ctx.queue,
            sorter.payloads(),
        ))[..n as usize]
            .to_vec();

        let mut expect: Vec<(u32, u32)> =
            keys.iter().copied().zip(payloads.iter().copied()).collect();
        expect.sort_by_key(|&(k, _)| k); // Vec::sort_by_key is stable

        for i in 0..n as usize {
            assert_eq!(
                (got_keys[i], got_payloads[i]),
                expect[i],
                "mismatch at index {i} of n={n}"
            );
        }
    }
}

#[test]
fn full_range_keys_sort() {
    let Some(ctx) = context() else { return };
    let n = 65_537u32; // crosses block boundaries, odd size
    let sorter = RadixSorter::new(&ctx, n);
    let mut rng = 0xdead_beefu32;
    let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut rng)).collect(); // full 32-bit range
    let payloads: Vec<u32> = (0..n).collect();

    ctx.queue
        .write_buffer(sorter.keys(), 0, bytemuck::cast_slice(&keys));
    ctx.queue
        .write_buffer(sorter.payloads(), 0, bytemuck::cast_slice(&payloads));
    ctx.queue
        .write_buffer(sorter.counts(), 0, bytemuck::bytes_of(&n));

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    sorter.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);

    let got: Vec<u32> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
        &ctx.device,
        &ctx.queue,
        sorter.keys(),
    ))[..n as usize]
        .to_vec();
    let mut expect = keys.clone();
    expect.sort_unstable();
    assert_eq!(got, expect);
}
