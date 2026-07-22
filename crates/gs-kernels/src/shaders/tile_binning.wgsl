// Tile binning: expand items (tile-space AABB + depth key) into per-tile
// entries, and extract per-tile ranges after sorting. The sort itself is the
// shared gs-wgpu radix sort, run twice (depth, then tile) — stability makes
// the two 32-bit sorts equivalent to one 64-bit (tile‖depth) sort.

struct Params {
    num_items: u32,
    tiles_x: u32,
}
struct PlainCount { n: u32 }

@group(0) @binding(0) var<uniform> params: Params;
// Item inputs: tile-space AABB (min.xy, max.xy inclusive) and depth key.
@group(0) @binding(1) var<storage, read> rects: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> depths: array<u32>;
// Exclusive-scanned entry offsets per item (PrefixSum output).
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
// Expansion outputs: tile key per entry, plus the sorter's key/payload buffers.
@group(0) @binding(4) var<storage, read_write> tile_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> sort_payloads: array<u32>;
// entry → source item id (consumers map sorted entries back to items).
@group(0) @binding(9) var<storage, read_write> entry_items: array<u32>;

// ---------------------------------------------------------------- counting
// counts[i] = number of tiles item i overlaps (written into the PrefixSum
// data buffer, bound here as `tile_keys` slot... no: separate entry point uses
// binding 4 as the counts destination — see host wiring).

@compute @workgroup_size(256)
fn count_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.num_items {
        return;
    }
    let r = rects[i];
    var count = 0u;
    if r.z >= r.x && r.w >= r.y {
        count = (r.z - r.x + 1u) * (r.w - r.y + 1u);
    }
    // Binding 4 doubles as the counts buffer for this entry point.
    tile_keys[i] = count;
}

// --------------------------------------------------------------- expansion
@compute @workgroup_size(256)
fn expand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.num_items {
        return;
    }
    let r = rects[i];
    if r.z < r.x || r.w < r.y {
        return;
    }
    var e = offsets[i];
    for (var ty = r.y; ty <= r.w; ty++) {
        for (var tx = r.x; tx <= r.z; tx++) {
            tile_keys[e] = ty * params.tiles_x + tx;
            sort_keys[e] = depths[i];
            sort_payloads[e] = e;
            entry_items[e] = i;
            e += 1u;
        }
    }
}

// ------------------------------------------------------------------ gather
// After the depth sort, sort_payloads holds the depth-ordered permutation of
// entry indices. Load each entry's tile key through that permutation into the
// sorter's key buffer for the second (tile) sort; payloads stay as-is, so the
// final payload stream is entry indices grouped by tile, depth-ascending.
struct GatherCount { n: u32 }
@group(0) @binding(7) var<storage, read> entry_count: GatherCount;

@compute @workgroup_size(256)
fn gather_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= entry_count.n {
        return;
    }
    sort_keys[i] = tile_keys[sort_payloads[i]];
}

// ------------------------------------------------------------- tile ranges
// ranges[t] = (start, end) of tile t's run in the sorted entry stream.
@group(0) @binding(8) var<storage, read_write> ranges: array<vec2<u32>>;

@compute @workgroup_size(256)
fn tile_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = entry_count.n;
    if i >= n {
        return;
    }
    let tile = sort_keys[i];
    if i == 0u || sort_keys[i - 1u] != tile {
        ranges[tile].x = i;
    }
    if i == n - 1u || sort_keys[i + 1u] != tile {
        ranges[tile].y = i + 1u;
    }
}
