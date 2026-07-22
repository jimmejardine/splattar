// Spawn surfels into dead slots.
//
// Incremental mapping needs new keyframes to ADD geometry, not merely re-fit
// what already exists — a corridor the camera has just walked into contains
// nothing for descent to move. Relocation (relocate.wgsl modes 0 and 1)
// derives the new surfel from an alive one; spawning instead writes uploaded
// parameters, which is what lets a plane-sweep depth map become surfels.
//
// Budget-neutral, like everything else that touches the surfel set: the splat
// buffers are pre-allocated to the MCMC budget and never reallocated
// (CLAUDE.md), so a spawn consumes a dead slot or does not happen.

struct SpawnParams {
    n_spawn: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

// Raw (pre-activation) parameters: scales are log-space, opacity is logit.
struct SpawnSurfel {
    pos: vec4<f32>,        // xyz = world position
    quat: vec4<f32>,       // xyzw, local->world
    scale_opac: vec4<f32>, // xy = log scales, z = logit opacity
    sh: vec4<f32>,         // xyz = degree-0 SH
}

@group(0) @binding(0) var<uniform> sp: SpawnParams;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read> incoming: array<SpawnSurfel>;

@group(0) @binding(3) var<storage, read_write> pos_raw: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> scales_raw: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> quat_raw: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> opac_raw: array<f32>;
@group(0) @binding(7) var<storage, read_write> sh_raw: array<vec4<f32>>;

@compute @workgroup_size(256)
fn spawn_params(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sp.n_spawn {
        return;
    }
    let d = slots[i];
    let s = incoming[i];
    pos_raw[d] = vec4<f32>(s.pos.xyz, 0.0);
    quat_raw[d] = s.quat;
    scales_raw[d] = s.scale_opac.xy;
    opac_raw[d] = s.scale_opac.z;
    // SH is coefficient-major rgb-interleaved, 48 floats padded into 12 vec4s.
    // Only degree 0 is seeded (the trainer unlocks higher bands progressively);
    // every other coefficient must be cleared or the slot inherits the
    // view-dependent colour of whatever died in it.
    sh_raw[d * 12u] = vec4<f32>(s.sh.xyz, 0.0);
    for (var k = 1u; k < 12u; k++) {
        sh_raw[d * 12u + k] = vec4<f32>(0.0);
    }
}

// Separate entry point: the raw buffers and the ten Adam moment buffers
// together exceed the storage-buffer-per-stage limit. Bindings continue past
// the params ones — one module cannot declare two types at one binding.
@group(0) @binding(8) var<storage, read_write> pos_m: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> pos_v: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> scales_m: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read_write> scales_v: array<vec2<f32>>;
@group(0) @binding(12) var<storage, read_write> quat_m: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> quat_v: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> opac_m: array<f32>;
@group(0) @binding(15) var<storage, read_write> opac_v: array<f32>;
@group(0) @binding(16) var<storage, read_write> sh_m: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read_write> sh_v: array<vec4<f32>>;

@compute @workgroup_size(256)
fn spawn_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sp.n_spawn {
        return;
    }
    // A dead slot carries the Adam history of the surfel that died in it.
    // Inheriting that would fire the newborn off in a direction inferred from
    // geometry somewhere else in the scene.
    let d = slots[i];
    pos_m[d] = vec4<f32>(0.0);
    pos_v[d] = vec4<f32>(0.0);
    scales_m[d] = vec2<f32>(0.0);
    scales_v[d] = vec2<f32>(0.0);
    quat_m[d] = vec4<f32>(0.0);
    quat_v[d] = vec4<f32>(0.0);
    opac_m[d] = 0.0;
    opac_v[d] = 0.0;
    for (var k = 0u; k < 12u; k++) {
        sh_m[d * 12u + k] = vec4<f32>(0.0);
        sh_v[d * 12u + k] = vec4<f32>(0.0);
    }
}
