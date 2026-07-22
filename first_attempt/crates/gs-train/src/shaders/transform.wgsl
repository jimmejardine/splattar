// Rigid/Sim(3) transform of a contiguous surfel range, in place.
//
// The two-tier mapping requirement: surfels spawned while a keyframe window
// only had PROVISIONAL poses are in the wrong frame once anchor-out re-solves
// that window. Discarding them would throw away every iteration spent on them;
// moving them with the correction turns the re-solve into a warm start.
//
// Index ranges are a valid way to address "the surfels this window spawned"
// because MCMC relocation never moves a surfel to a different index — it
// overwrites DEAD slots with copies of alive ones. An alive surfel keeps its
// index for life.
//
// Adam moments of the moved range are zeroed: the accumulated gradient history
// describes a position the surfel no longer occupies, and carrying it across
// the jump would fire the optimizer off in a direction that made sense only in
// the old frame.

struct Xf {
    // Camera-to-world style rotation, as a quaternion (xyzw).
    rot: vec4<f32>,
    // Translation, w unused.
    trans: vec4<f32>,
    scale: f32,
    lo: u32,
    hi: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> xf: Xf;
@group(0) @binding(1) var<storage, read_write> pos_raw: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> scales_raw: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> quat_raw: array<vec4<f32>>;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    // xyzw convention, matching glam.
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

@compute @workgroup_size(256)
fn transform_params(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = xf.lo + gid.x;
    if i >= xf.hi {
        return;
    }
    let q = normalize(xf.rot);
    let p = pos_raw[i];
    pos_raw[i] = vec4<f32>(xf.scale * quat_rotate(q, p.xyz) + xf.trans.xyz, p.w);
    // Scales are stored in log space, so a Sim(3) scale is an additive shift.
    scales_raw[i] = scales_raw[i] + vec2<f32>(log(max(xf.scale, 1e-8)));
    quat_raw[i] = quat_mul(q, normalize(quat_raw[i]));
}

// Moment zeroing is a separate entry point for the same reason relocate's is:
// the raw parameter buffers and the ten Adam moment buffers together exceed
// the storage-buffer-per-stage limit. Bindings continue past the params ones
// rather than reusing 1..3 — one module cannot declare two different types at
// the same binding, even in different entry points.
@group(0) @binding(4) var<storage, read_write> pos_m: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> pos_v: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> scales_m: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> scales_v: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> quat_m: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> quat_v: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> opac_m: array<f32>;
@group(0) @binding(11) var<storage, read_write> opac_v: array<f32>;
@group(0) @binding(12) var<storage, read_write> sh_m: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> sh_v: array<vec4<f32>>;

@compute @workgroup_size(256)
fn transform_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = xf.lo + gid.x;
    if i >= xf.hi {
        return;
    }
    pos_m[i] = vec4<f32>(0.0);
    pos_v[i] = vec4<f32>(0.0);
    scales_m[i] = vec2<f32>(0.0);
    scales_v[i] = vec2<f32>(0.0);
    quat_m[i] = vec4<f32>(0.0);
    quat_v[i] = vec4<f32>(0.0);
    opac_m[i] = 0.0;
    opac_v[i] = 0.0;
    for (var k = 0u; k < 12u; k++) {
        sh_m[i * 12u + k] = vec4<f32>(0.0);
        sh_v[i * 12u + k] = vec4<f32>(0.0);
    }
}
