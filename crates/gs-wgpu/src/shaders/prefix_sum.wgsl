// Exclusive prefix sum over u32. Block-level shared-memory scan plus block
// sums; the host recurses over levels (scan the block sums with the same
// kernel, then add back down). WG 256 × 4 elements = 1024 per block.

const WG: u32 = 256u;
const EPT: u32 = 4u;
const BLOCK: u32 = 1024u;

struct Params { n: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

var<workgroup> sums: array<u32, WG>;

@compute @workgroup_size(WG)
fn scan_blocks(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = lid.x;
    let base = wg_id.x * BLOCK + t * EPT;

    var v: array<u32, EPT>;
    var total = 0u;
    for (var i = 0u; i < EPT; i++) {
        let idx = base + i;
        v[i] = select(0u, data[idx], idx < params.n);
        total += v[i];
    }
    sums[t] = total;
    workgroupBarrier();

    // Inclusive Hillis–Steele over per-thread totals.
    for (var step = 1u; step < WG; step = step << 1u) {
        var add = 0u;
        if t >= step {
            add = sums[t - step];
        }
        workgroupBarrier();
        sums[t] += add;
        workgroupBarrier();
    }

    // Exclusive base for this thread, then exclusive within its elements.
    var run = sums[t] - total;
    for (var i = 0u; i < EPT; i++) {
        let idx = base + i;
        if idx < params.n {
            data[idx] = run;
        }
        run += v[i];
    }
    if t == WG - 1u {
        block_sums[wg_id.x] = sums[WG - 1u];
    }
}

@compute @workgroup_size(WG)
fn add_back(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // block_sums has been exclusively scanned by the level above; block 0 adds 0.
    let add = block_sums[wg_id.x];
    let base = wg_id.x * BLOCK + lid.x * EPT;
    for (var i = 0u; i < EPT; i++) {
        let idx = base + i;
        if idx < params.n {
            data[idx] += add;
        }
    }
}
