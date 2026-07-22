// Writes the indirect dispatch args for the sort's histogram/scatter kernels
// from the GPU-side element count (the count never round-trips to the CPU).

const BLOCK: u32 = 1024u;

struct Counts { n: u32 }
struct DispatchArgs { x: u32, y: u32, z: u32 }

@group(0) @binding(0) var<storage, read> counts: Counts;
@group(0) @binding(1) var<storage, read_write> args: DispatchArgs;

@compute @workgroup_size(1)
fn prep_dispatch() {
    args.x = (counts.n + BLOCK - 1u) / BLOCK;
    args.y = 1u;
    args.z = 1u;
}
