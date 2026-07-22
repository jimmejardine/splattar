// Photometric loss between a render and a target frame.
//
// Produces two things from one pass: the scalar residual (what the diagnostic
// curve plots) and dL/d(render) (what the rasterizer's backward consumes to
// reach the camera gradient). They come from the same pass on purpose — a
// residual that disagreed with the gradient being descended would make the
// diagnostic lie about what the optimizer is doing.
//
// L1 rather than L2: outliers here are occlusions, specularities and moving
// objects — real content the map does not explain — and L2 lets a handful of
// them dominate the camera update.

struct Params {
    npix: u32,
    // Per-frame brightness transform applied to the TARGET, so the render is
    // left untouched and the gradient path stays exact. Auto-exposure moves
    // gain and offset continuously, and a direct method that ignores that is
    // measuring the exposure ramp as if it were motion.
    gain: f32,
    bias: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> render: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> frame: array<vec4<f32>>;  // `target` is a reserved WGSL keyword
@group(0) @binding(3) var<storage, read_write> dl_dcolor: array<vec4<f32>>;
// [sum |err|, covered pixel count, unused, unused] as bitcast u32 for atomics.
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>;

const EPS: f32 = 1e-4;

// WGSL has no f32 atomics: accumulate through a bitcast compare-exchange, the
// same pattern the gradient kernels use.
fn atomic_add_f32(idx: u32, value: f32) {
    var old = atomicLoad(&stats[idx]);
    loop {
        let sum = bitcast<f32>(old) + value;
        let res = atomicCompareExchangeWeak(&stats[idx], old, bitcast<u32>(sum));
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}

@compute @workgroup_size(256)
fn photometric(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.npix {
        return;
    }
    let r = render[i];
    let t = frame[i];
    var grad = vec3<f32>(0.0);
    var err = 0.0;
    for (var c = 0u; c < 3u; c++) {
        let tc = p.gain * t[c] + p.bias;
        let d = r[c] - tc;
        err += abs(d);
        // d|x|/dx, softened near zero so a pixel that already matches does not
        // contribute a full-magnitude sign flip every iteration.
        grad[c] = clamp(d / EPS, -1.0, 1.0) / (3.0 * f32(p.npix));
    }
    dl_dcolor[i] = vec4<f32>(grad, 0.0);
    atomic_add_f32(0u, err / (3.0 * f32(p.npix)));
    // Alpha says whether the map covered this pixel at all. Distinct from the
    // residual: a map can fit what it covers perfectly and still explain only
    // part of the frame, and only coverage shows that.
    if r.w > 0.5 {
        atomic_add_f32(1u, 1.0 / f32(p.npix));
    }
}
