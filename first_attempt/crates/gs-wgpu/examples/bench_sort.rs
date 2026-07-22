//! Sort perf gate (PLAN.md budget: 4M keys < 2 ms).
//! `cargo run -p gs-wgpu --release --features profile --example bench_sort`

use gs_wgpu::{GpuContext, GpuTimer, RadixSorter};

fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let ctx = pollster::block_on(GpuContext::new(wgpu::Backends::all())).expect("gpu");

    // --stages: one-shot per-stage cost attribution at 4M.
    if std::env::args().any(|a| a == "--stages") {
        let n = 4_000_000u32;
        let sorter = RadixSorter::new(&ctx, n);
        let mut timer = GpuTimer::new(&ctx, 32);
        let mut rng = 0xc0ffeeu32;
        let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut rng)).collect();
        let payloads: Vec<u32> = (0..n).collect();
        ctx.queue.write_buffer(sorter.counts(), 0, bytemuck::bytes_of(&n));
        for _ in 0..3 {
            ctx.queue.write_buffer(sorter.keys(), 0, bytemuck::cast_slice(&keys));
            ctx.queue
                .write_buffer(sorter.payloads(), 0, bytemuck::cast_slice(&payloads));
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            sorter.encode_stage_profiled(&mut encoder, &mut timer);
            timer.resolve(&mut encoder);
            ctx.queue.submit([encoder.finish()]);
            let timings = timer.read(&ctx);
            let mut hist = 0.0;
            let mut scan = 0.0;
            let mut scatter = 0.0;
            for (label, ms) in &timings {
                if label.starts_with("histogram") {
                    hist += ms;
                } else if label.starts_with("scan") {
                    scan += ms;
                } else if label.starts_with("scatter") {
                    scatter += ms;
                }
            }
            println!("4M stage totals: histogram {hist:.3} ms, scan {scan:.3} ms, scatter {scatter:.3} ms");
        }
        return;
    }

    const WARMUP: usize = 5;
    const ITERS: usize = 20;
    for &n in &[1_000_000u32, 2_000_000, 4_000_000, 8_000_000] {
        let sorter = RadixSorter::new(&ctx, n);
        let mut timer = GpuTimer::new(&ctx, 4);
        let mut rng = 0xc0ffee ^ n;
        let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut rng)).collect();
        let payloads: Vec<u32> = (0..n).collect();
        ctx.queue
            .write_buffer(sorter.counts(), 0, bytemuck::bytes_of(&n));

        let mut best = f64::INFINITY;
        let mut sum = 0.0;
        for i in 0..WARMUP + ITERS {
            // Restore unsorted input (sorting sorted data would flatter the numbers).
            ctx.queue
                .write_buffer(sorter.keys(), 0, bytemuck::cast_slice(&keys));
            ctx.queue
                .write_buffer(sorter.payloads(), 0, bytemuck::cast_slice(&payloads));
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            sorter.encode_profiled(&mut encoder, &mut timer);
            timer.resolve(&mut encoder);
            ctx.queue.submit([encoder.finish()]);
            let timings = timer.read(&ctx);
            if i >= WARMUP {
                let total: f64 = timings.iter().map(|(_, ms)| ms).sum();
                best = best.min(total);
                sum += total;
            }
        }
        println!(
            "n = {n:>9}: best {best:.3} ms, avg {:.3} ms (GPU time, prep+sort)",
            sum / ITERS as f64
        );
    }
}
