//! Tile-binning property test vs a CPU reference: every (item, tile) pair
//! appears exactly once inside that tile's range, depth keys ascend within
//! each range, equal depths preserve expansion order (sort stability), and
//! empty tiles have empty ranges.

use gs_kernels::TileBinner;
use gs_wgpu::GpuContext;

const TILES_X: u32 = 64;
const TILES_Y: u32 = 64;

fn context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(wgpu::Backends::all())) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPING GPU binning tests: {e}");
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

#[derive(Clone, Copy)]
struct Item {
    rect: [u32; 4], // min_x, min_y, max_x, max_y (inclusive)
    depth: u32,
}

fn run_case(ctx: &GpuContext, items: &[Item], seed_label: &str) {
    // CPU reference: expansion order is item-major, row-major within the rect.
    let mut entries: Vec<(u32, u32, u32)> = Vec::new(); // (tile, depth, entry_idx)
    for item in items {
        let [x0, y0, x1, y1] = item.rect;
        if x1 < x0 || y1 < y0 {
            continue;
        }
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                entries.push((ty * TILES_X + tx, item.depth, entries.len() as u32));
            }
        }
    }
    let total = entries.len() as u32;
    // Expected final order: by tile, then depth, then expansion order (stability).
    let mut expected = entries.clone();
    expected.sort_by_key(|&(tile, depth, idx)| (tile, depth, idx));

    let max_entries = total.max(1);
    let binner = TileBinner::new(ctx, items.len() as u32, max_entries, TILES_X * TILES_Y);
    let rects: Vec<u32> = items.iter().flat_map(|i| i.rect).collect();
    let depths: Vec<u32> = items.iter().map(|i| i.depth).collect();
    ctx.queue
        .write_buffer(&binner.rects, 0, bytemuck::cast_slice(&rects));
    ctx.queue
        .write_buffer(&binner.depths, 0, bytemuck::cast_slice(&depths));

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    binner.encode(ctx, &mut encoder, items.len() as u32, TILES_X);
    ctx.queue.submit([encoder.finish()]);

    let got_count: u32 = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
        &ctx.device,
        &ctx.queue,
        binner.entry_count(),
    ))[0];
    assert_eq!(got_count, total, "{seed_label}: entry count");
    if total == 0 {
        return;
    }

    let payloads: Vec<u32> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
        &ctx.device,
        &ctx.queue,
        binner.sorted_entries(),
    ))[..total as usize]
        .to_vec();
    let ranges: Vec<[u32; 2]> = bytemuck::cast_slice(&gs_wgpu::buffers::readback(
        &ctx.device,
        &ctx.queue,
        &binner.ranges,
    ))
    .to_vec();

    // Payload stream must match the expected (tile, depth, entry) order exactly.
    for (i, (&got_entry, &(_, _, want_entry))) in
        payloads.iter().zip(expected.iter()).enumerate()
    {
        assert_eq!(
            got_entry, want_entry,
            "{seed_label}: payload mismatch at sorted position {i}"
        );
    }

    // Ranges must exactly bracket each tile's run in the expected stream.
    let mut expected_ranges = vec![[0u32; 2]; (TILES_X * TILES_Y) as usize];
    for (i, &(tile, _, _)) in expected.iter().enumerate() {
        let r = &mut expected_ranges[tile as usize];
        if r[1] == 0 && r[0] == 0 && (i == 0 || expected[i - 1].0 != tile) {
            r[0] = i as u32;
        }
        r[1] = i as u32 + 1;
    }
    for (tile, (got, want)) in ranges.iter().zip(expected_ranges.iter()).enumerate() {
        assert_eq!(got, want, "{seed_label}: range mismatch for tile {tile}");
    }
}

#[test]
fn binning_matches_cpu_reference() {
    let Some(ctx) = context() else { return };

    // Random rects across sizes, heavy depth duplication to exercise stability.
    for &(num_items, seed) in &[(1u32, 7u32), (100, 11), (5_000, 13), (100_000, 17)] {
        let mut rng = seed;
        let items: Vec<Item> = (0..num_items)
            .map(|_| {
                let x0 = xorshift(&mut rng) % TILES_X;
                let y0 = xorshift(&mut rng) % TILES_Y;
                let w = xorshift(&mut rng) % 4;
                let h = xorshift(&mut rng) % 4;
                Item {
                    rect: [
                        x0,
                        y0,
                        (x0 + w).min(TILES_X - 1),
                        (y0 + h).min(TILES_Y - 1),
                    ],
                    depth: xorshift(&mut rng) % 32, // heavy duplicates
                }
            })
            .collect();
        run_case(&ctx, &items, &format!("random n={num_items}"));
    }

    // Degenerate cases: culled item (max < min), all items in one tile.
    let culled = Item {
        rect: [5, 5, 4, 4],
        depth: 1,
    };
    let stacked: Vec<Item> = (0..500)
        .map(|i| Item {
            rect: [10, 10, 10, 10],
            depth: (i % 3) as u32,
        })
        .collect();
    run_case(&ctx, &[culled], "culled item");
    run_case(&ctx, &stacked, "stacked single tile");
}
