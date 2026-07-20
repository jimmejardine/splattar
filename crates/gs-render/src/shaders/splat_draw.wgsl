// Instanced splat draw: 4-vertex triangle strip per visible splat, fetched
// through the depth-sorted payload buffer (back-to-front), gaussian falloff
// in the fragment stage, premultiplied-alpha OVER blending, no depth buffer.

struct Splat2D {
    center_axis1: vec4<f32>,
    axis2: vec4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> splats_2d: array<Splat2D>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>, // σ-units
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VsOut {
    let splat = splats_2d[sorted_indices[ii]];
    // Strip corners (−1,−1) (1,−1) (−1,1) (1,1).
    let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    let center = splat.center_axis1.xy;
    let axis1 = splat.center_axis1.zw;
    let axis2 = splat.axis2.xy;

    var out: VsOut;
    out.pos = vec4<f32>(center + corner.x * axis1 + corner.y * axis2, 0.0, 1.0);
    out.local = corner * 3.0; // axes span 3σ
    out.color = splat.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let alpha = in.color.a * exp(-0.5 * dot(in.local, in.local));
    if alpha < 0.0039 {
        discard;
    }
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
