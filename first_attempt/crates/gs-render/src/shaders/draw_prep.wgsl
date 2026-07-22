// Emits indirect draw args from the GPU-side visible-splat count:
// 4-vertex triangle strip, one instance per visible splat.

struct PlainCount { n: u32 }
struct DrawArgs { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32 }

@group(0) @binding(0) var<storage, read> counts: PlainCount;
@group(0) @binding(1) var<storage, read_write> draw_args: DrawArgs;

@compute @workgroup_size(1)
fn draw_prep() {
    draw_args = DrawArgs(4u, counts.n, 0u, 0u);
}
