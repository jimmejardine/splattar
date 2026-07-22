// LSD radix sort for 32-bit keys with 32-bit payloads.
// 4-bit digits → 8 passes (host ping-pongs buffers; even pass count means the
// final order lands back in the "a" buffers). Per pass: histogram → scan →
// scatter. Stability comes from consecutive element assignment per thread,
// per-digit exclusive scans across threads, and per-block exclusive prefixes
// across workgroups — every level preserves original order within a digit.
//
// Workgroup memory stays under wgpu's 16 KiB default limit: the scatter rank
// table is WG(128) × DIGITS(16) × 4 B = 8 KiB.

const WG: u32 = 128u;
const EPT: u32 = 8u;       // elements per thread (consecutive → stability)
const BLOCK: u32 = 1024u;  // WG * EPT
const DIGITS: u32 = 16u;
const OOB: u32 = 0xffffffffu;

struct Counts { n: u32 }
struct SortParams { shift: u32 }

@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> counts: Counts;
@group(0) @binding(2) var<storage, read> keys_in: array<u32>;
@group(0) @binding(3) var<storage, read> payload_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> payload_out: array<u32>;
// block_hists[b*16+d]: per-block digit counts (histogram) → per-block
// exclusive prefixes within each digit (after scan).
@group(0) @binding(6) var<storage, read_write> block_hists: array<u32>;
// digit_offsets[d]: exclusive scan of global digit totals.
@group(0) @binding(7) var<storage, read_write> digit_offsets: array<u32>;
// digit_totals[d]: per-digit column totals (scan_columns → scan_totals).
@group(0) @binding(8) var<storage, read_write> digit_totals: array<u32>;

fn digit_of(key: u32) -> u32 {
    return (key >> params.shift) & 0xfu;
}

// ---------------------------------------------------------------- histogram
var<workgroup> wg_hist: array<atomic<u32>, DIGITS>;

@compute @workgroup_size(WG)
fn histogram(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if lid.x < DIGITS {
        atomicStore(&wg_hist[lid.x], 0u);
    }
    workgroupBarrier();

    let base = wg_id.x * BLOCK + lid.x * EPT;
    for (var i = 0u; i < EPT; i++) {
        let idx = base + i;
        if idx < counts.n {
            atomicAdd(&wg_hist[digit_of(keys_in[idx])], 1u);
        }
    }
    workgroupBarrier();

    if lid.x < DIGITS {
        block_hists[wg_id.x * DIGITS + lid.x] = atomicLoad(&wg_hist[lid.x]);
    }
}

// --------------------------------------------------------------------- scan
// Parallelized in two kernels (the M0 serial version cost ~4.7 ms of the
// 5.4 ms total at 4M keys):
//   scan_columns — one workgroup per digit; 256 threads chunk-scan that
//   digit's column across all blocks (shared Hillis–Steele + running carry),
//   leaving per-block exclusive bases in block_hists and the column total in
//   digit_totals.
//   scan_totals — one tiny workgroup exclusive-scans the 16 totals.
const SCAN_WG: u32 = 256u;
var<workgroup> chunk: array<u32, SCAN_WG>;

@compute @workgroup_size(SCAN_WG)
fn scan_columns(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let d = wg_id.x; // digit column owned by this workgroup
    let t = lid.x;
    let num_blocks = (counts.n + BLOCK - 1u) / BLOCK;
    var carry = 0u;
    var base = 0u;
    while base < num_blocks {
        let b = base + t;
        var v = 0u;
        if b < num_blocks {
            v = block_hists[b * DIGITS + d];
        }
        chunk[t] = v;
        workgroupBarrier();
        // Inclusive Hillis–Steele scan over the chunk.
        for (var step = 1u; step < SCAN_WG; step = step << 1u) {
            var add = 0u;
            if t >= step {
                add = chunk[t - step];
            }
            workgroupBarrier();
            chunk[t] += add;
            workgroupBarrier();
        }
        if b < num_blocks {
            // Exclusive base = inclusive - own value, plus carry from prior chunks.
            block_hists[b * DIGITS + d] = carry + chunk[t] - v;
        }
        carry += chunk[SCAN_WG - 1u];
        workgroupBarrier();
        base += SCAN_WG;
    }
    if t == 0u {
        digit_totals[d] = carry;
    }
}

@compute @workgroup_size(1)
fn scan_totals() {
    var acc = 0u;
    for (var i = 0u; i < DIGITS; i++) {
        digit_offsets[i] = acc;
        acc += digit_totals[i];
    }
}

// ------------------------------------------------------------------ scatter
// rank_table[t*16+d]: thread t's private count of digit d, scanned in place
// into the exclusive prefix over threads 0..t within this block.
var<workgroup> rank_table: array<u32, 2048>; // WG * DIGITS

@compute @workgroup_size(WG)
fn scatter(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = lid.x;
    let base = wg_id.x * BLOCK + t * EPT;

    var digits: array<u32, EPT>;
    var local_counts: array<u32, DIGITS>;
    for (var d = 0u; d < DIGITS; d++) {
        local_counts[d] = 0u;
    }
    for (var i = 0u; i < EPT; i++) {
        let idx = base + i;
        if idx < counts.n {
            let d = digit_of(keys_in[idx]);
            digits[i] = d;
            local_counts[d] += 1u;
        } else {
            digits[i] = OOB;
        }
    }
    for (var d = 0u; d < DIGITS; d++) {
        rank_table[t * DIGITS + d] = local_counts[d];
    }
    workgroupBarrier();

    if t < DIGITS {
        var sum = 0u;
        for (var tt = 0u; tt < WG; tt++) {
            let v = rank_table[tt * DIGITS + t];
            rank_table[tt * DIGITS + t] = sum;
            sum += v;
        }
    }
    workgroupBarrier();

    var running: array<u32, DIGITS>;
    for (var d = 0u; d < DIGITS; d++) {
        running[d] = 0u;
    }
    for (var i = 0u; i < EPT; i++) {
        let idx = base + i;
        if idx < counts.n {
            let d = digits[i];
            let local_rank = rank_table[t * DIGITS + d] + running[d];
            running[d] += 1u;
            let dst = digit_offsets[d] + block_hists[wg_id.x * DIGITS + d] + local_rank;
            keys_out[dst] = keys_in[idx];
            payload_out[dst] = payload_in[idx];
        }
    }
}
