// Fused Adam step over a flat f32 parameter buffer, with the raw→activated
// chain rule applied in-kernel. Parameters are stored RAW (log-scales, logit
// opacities); the rasterizer consumes ACTIVATED copies produced by the
// `activate` entry point. Padded gradient slots are zero, so padded parameter
// lanes never move (m = v = 0 ⇒ step 0).

struct AdamParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    // Bias corrections precomputed on the host: 1/(1−β1ᵗ), 1/(1−β2ᵗ).
    bc1_inv: f32,
    bc2_inv: f32,
    n: u32,
    // 0 = identity, 1 = exp (scales), 2 = sigmoid (opacity).
    activation: u32,
    // L1 regularizer on the ACTIVATED value: adds `reg` to dL/dact
    // (host passes λ/count for a mean-based penalty; MCMC opacity/scale priors).
    reg: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> ap: AdamParams;
@group(0) @binding(1) var<storage, read_write> raw: array<f32>;
@group(0) @binding(2) var<storage, read> grad_act: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> activated: array<f32>;

fn dact(x: f32) -> f32 {
    if ap.activation == 1u {
        return exp(x);
    }
    if ap.activation == 2u {
        let s = 1.0 / (1.0 + exp(-x));
        return s * (1.0 - s);
    }
    return 1.0;
}

@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= ap.n {
        return;
    }
    let g = (grad_act[i] + ap.reg) * dact(raw[i]);
    let mi = ap.beta1 * m[i] + (1.0 - ap.beta1) * g;
    let vi = ap.beta2 * v[i] + (1.0 - ap.beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    let mhat = mi * ap.bc1_inv;
    let vhat = vi * ap.bc2_inv;
    raw[i] -= ap.lr * mhat / (sqrt(vhat) + ap.eps);
}

@compute @workgroup_size(256)
fn activate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= ap.n {
        return;
    }
    let x = raw[i];
    if ap.activation == 1u {
        activated[i] = exp(x);
    } else if ap.activation == 2u {
        activated[i] = 1.0 / (1.0 + exp(-x));
    } else {
        activated[i] = x;
    }
}
