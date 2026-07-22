// Per-view affine appearance compensation, GPU side (M7 perf rework).
//
// `target_transform` inverse-corrects one atlas view with the current affine
// and writes the loss target — replacing the per-iteration host transform +
// full-image upload. `fit_reduce` computes the sufficient statistics of the
// per-channel least-squares fit target ≈ gain·render + bias (Σr, Σt, Σrr,
// Σrt per channel over the same stride-subsampled pixels the host fit used);
// the host turns the 12 sums into gains/biases after an async 48-byte
// readback. Not a gradient kernel — the affine is applied to the target,
// which is a constant in the gradient path.

struct AppearParams {
    gain: vec4<f32>, // rgb gain (w unused)
    bias: vec4<f32>, // rgb bias
    npix: u32,
    stride: u32,     // fit subsample stride (host fit used 4)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> ap: AppearParams;
@group(0) @binding(1) var<storage, read> atlas_view: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> loss_target: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> render: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>; // 12 sums

@compute @workgroup_size(256)
fn target_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= ap.npix {
        return;
    }
    let t = atlas_view[i];
    let corrected = clamp(
        (t.rgb - ap.bias.rgb) / ap.gain.rgb,
        vec3<f32>(0.0),
        vec3<f32>(4.0),
    );
    loss_target[i] = vec4<f32>(corrected, t.w);
}

const FIT_WG: u32 = 128u;

var<workgroup> red_sr: array<vec3<f32>, FIT_WG>;
var<workgroup> red_st: array<vec3<f32>, FIT_WG>;
var<workgroup> red_srr: array<vec3<f32>, FIT_WG>;
var<workgroup> red_srt: array<vec3<f32>, FIT_WG>;

// Single-workgroup strided reduction: ~25k subsampled pixels at product
// resolutions, a few hundred global loads per thread — far below 0.1 ms.
@compute @workgroup_size(128)
fn fit_reduce(@builtin(local_invocation_index) li: u32) {
    var sr = vec3<f32>(0.0);
    var st = vec3<f32>(0.0);
    var srr = vec3<f32>(0.0);
    var srt = vec3<f32>(0.0);
    let n_samples = (ap.npix + ap.stride - 1u) / ap.stride;
    var k = li;
    while k < n_samples {
        let idx = k * ap.stride;
        let r = render[idx].rgb;
        let t = atlas_view[idx].rgb;
        sr += r;
        st += t;
        srr += r * r;
        srt += r * t;
        k += FIT_WG;
    }
    red_sr[li] = sr;
    red_st[li] = st;
    red_srr[li] = srr;
    red_srt[li] = srt;
    workgroupBarrier();
    for (var off = FIT_WG / 2u; off > 0u; off >>= 1u) {
        if li < off {
            red_sr[li] += red_sr[li + off];
            red_st[li] += red_st[li + off];
            red_srr[li] += red_srr[li + off];
            red_srt[li] += red_srt[li + off];
        }
        workgroupBarrier();
    }
    if li == 0u {
        stats[0] = red_sr[0].x;
        stats[1] = red_sr[0].y;
        stats[2] = red_sr[0].z;
        stats[3] = red_st[0].x;
        stats[4] = red_st[0].y;
        stats[5] = red_st[0].z;
        stats[6] = red_srr[0].x;
        stats[7] = red_srr[0].y;
        stats[8] = red_srr[0].z;
        stats[9] = red_srt[0].x;
        stats[10] = red_srt[0].y;
        stats[11] = red_srt[0].z;
    }
}
