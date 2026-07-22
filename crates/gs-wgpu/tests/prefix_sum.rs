//! Exclusive-scan correctness vs CPU at sizes spanning 1..10M, including
//! non-multiples of the 1024 block, plus total verification.

use gs_wgpu::{GpuContext, PrefixSum};

fn context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(wgpu::Backends::all())) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPING GPU prefix-sum tests: {e}");
            None
        }
    }
}

fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn run_case(ctx: &GpuContext, scan: &PrefixSum, input: &[u32]) {
    let n = input.len() as u32;
    ctx.queue
        .write_buffer(scan.data(), 0, bytemuck::cast_slice(input));
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    scan.encode(ctx, &mut encoder, n);
    ctx.queue.submit([encoder.finish()]);

    let got: Vec<u32> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
        &ctx.device,
        &ctx.queue,
        scan.data(),
    ))[..n as usize]
        .to_vec();
    let total: u32 =
        bytemuck::cast_slice(&gs_wgpu::buffers::readback(&ctx.device, &ctx.queue, scan.total()))
            [0];

    let mut expect = Vec::with_capacity(input.len());
    let mut acc = 0u32;
    for &v in input {
        expect.push(acc);
        acc = acc.wrapping_add(v);
    }
    assert_eq!(total, acc, "total mismatch at n={n}");
    for i in 0..input.len() {
        assert_eq!(got[i], expect[i], "scan mismatch at index {i} of n={n}");
    }
}

#[test]
fn exclusive_scan_matches_cpu() {
    let Some(ctx) = context() else { return };
    let scan = PrefixSum::new(&ctx, 10_000_000);

    for &n in &[1usize, 2, 255, 1024, 1025, 4096, 100_003, 1_048_576, 10_000_000] {
        let mut rng = 0x5eed_0001u32 ^ n as u32;
        // Small values so 10M sums stay far from u32 overflow.
        let input: Vec<u32> = (0..n).map(|_| xorshift(&mut rng) % 16).collect();
        run_case(&ctx, &scan, &input);
    }

    // Degenerate shapes.
    run_case(&ctx, &scan, &vec![0u32; 4096]);
    run_case(&ctx, &scan, &vec![1u32; 2049]);
    run_case(&ctx, &scan, &[42]);
}
