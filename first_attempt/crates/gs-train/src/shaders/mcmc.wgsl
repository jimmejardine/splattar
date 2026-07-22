// MCMC exploration noise (3DGS-MCMC style, simplified): every step, raw
// positions receive Gaussian-ish noise scaled by the surfel's mean scale and
// gated by opacity (opaque, well-supported surfels barely move; near-dead
// ones explore). Relocation of dead surfels is CPU-orchestrated.

struct NoiseParams {
    n: u32,
    seed: u32,
    sigma: f32,     // world-units multiplier (host: noise_cfg × current pos lr)
    opacity_k: f32, // gate sharpness: gate = exp(−k·α)
}

@group(0) @binding(0) var<uniform> np: NoiseParams;
@group(0) @binding(1) var<storage, read_write> pos_raw: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> scales_act: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> opac_act: array<f32>;

fn pcg(v: u32) -> u32 {
    var state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn unit_uniform(v: u32) -> f32 {
    return f32(v & 0x00ffffffu) / 16777216.0;
}

// Sum of 4 uniforms − 2: mean 0, var 1/3 — cheap near-Gaussian.
fn noise1(seed: u32) -> f32 {
    var acc = 0.0;
    var s = seed;
    for (var k = 0u; k < 4u; k++) {
        s = pcg(s);
        acc += unit_uniform(s);
    }
    return acc - 2.0;
}

@compute @workgroup_size(256)
fn add_noise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= np.n {
        return;
    }
    let s = scales_act[i];
    let mean_scale = 0.5 * (s.x + s.y);
    let gate = exp(-np.opacity_k * opac_act[i]);
    let amp = np.sigma * mean_scale * gate;
    if amp <= 0.0 {
        return;
    }
    let base = pcg(i ^ np.seed);
    var p = pos_raw[i];
    p.x += amp * noise1(base ^ 0x68bc21ebu);
    p.y += amp * noise1(base ^ 0x2c1b3c6du);
    p.z += amp * noise1(base ^ 0x1f123bb5u);
    pos_raw[i] = p;
}
