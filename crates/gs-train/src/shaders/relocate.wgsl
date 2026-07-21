// MCMC dead-surfel relocation, GPU side. The host still owns the sampling
// (opacity-proportional alive pick from a readback of activated opacities —
// once per mcmc_every iterations) and uploads (dead, alive) index pairs;
// these kernels do the bulk data movement that used to be a 6-readback /
// 5-upload host round-trip: copy the alive surfel's params onto the dead
// one, write the split opacity to both, shrink scales, and zero the Adam
// moments of both indices. Alive indices are unique within a batch (host
// samples without replacement), so pairs never race.
//
// Split into two entry points to stay under the storage-buffer-per-stage
// limit: params (raw parameter buffers) and moments (Adam m/v x5 classes).

struct RelocParams {
    n_pairs: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> rp: RelocParams;
@group(0) @binding(1) var<storage, read> pairs: array<vec2<u32>>; // (dead, alive)

// relocate_params bindings.
@group(0) @binding(2) var<storage, read> opac_act: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_raw: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> scales_raw: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> quat_raw: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> opac_raw: array<f32>;
@group(0) @binding(7) var<storage, read_write> sh_raw: array<vec4<f32>>; // 12 vec4 per surfel

@compute @workgroup_size(256)
fn relocate_params(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= rp.n_pairs {
        return;
    }
    let pr = pairs[i];
    let d = pr.x;
    let a = pr.y;

    // Split opacity α' = 1 − √(1−α), applied to both, in logit space.
    let alpha_a = clamp(opac_act[a], 1e-4, 0.999);
    let alpha_new = clamp(1.0 - sqrt(1.0 - alpha_a), 1e-4, 0.999);
    let logit = log(alpha_new / (1.0 - alpha_new));
    opac_raw[d] = logit;
    opac_raw[a] = logit;

    pos_raw[d] = pos_raw[a];
    quat_raw[d] = quat_raw[a];
    let s = scales_raw[a] - vec2<f32>(0.16, 0.16); // log-space ×0.85
    scales_raw[a] = s;
    scales_raw[d] = s;
    for (var k = 0u; k < 12u; k++) {
        sh_raw[d * 12u + k] = sh_raw[a * 12u + k];
    }
}

// relocate_moments bindings (uniform + pairs shared above).
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

fn zero_moments_at(idx: u32) {
    pos_m[idx] = vec4<f32>(0.0);
    pos_v[idx] = vec4<f32>(0.0);
    scales_m[idx] = vec2<f32>(0.0);
    scales_v[idx] = vec2<f32>(0.0);
    quat_m[idx] = vec4<f32>(0.0);
    quat_v[idx] = vec4<f32>(0.0);
    opac_m[idx] = 0.0;
    opac_v[idx] = 0.0;
    for (var k = 0u; k < 12u; k++) {
        sh_m[idx * 12u + k] = vec4<f32>(0.0);
        sh_v[idx * 12u + k] = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(256)
fn relocate_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= rp.n_pairs {
        return;
    }
    let pr = pairs[i];
    zero_moments_at(pr.x);
    zero_moments_at(pr.y);
}
