// Fused L1 + D-SSIM loss: SSIM with an 11-tap sigma=1.5 Gaussian window
// (zero-padded - the kernel is symmetric, so the blur is self-adjoint and the
// backward pass reuses it), analytic backward via three blurred coefficient
// maps:
//   dS/dx(q) = blur(Cmu)(q) + 2*x(q)*blur(Cx2)(q) + y(q)*blur(Cxy)(q)
// with S = l*cs expressed in the blurred raw moments (mu_x, mu_y, bx2, by2, bxy).
// Final gradient: dL/dx = (1-lambda)*sign(x-y)/N3 + lambda*(-1/(2*N3))*dS/dx.
// (`target` is a reserved WGSL word; the reference image binds as ref_img.)

struct SsimParams {
    width: u32,
    height: u32,
    // 1/(num_pixels*3)
    inv_n3: f32,
    lambda: f32,
}

@group(0) @binding(0) var<uniform> sp: SsimParams;
@group(0) @binding(1) var<storage, read> img: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ref_img: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> in_a: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> in_b: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> in_c: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> out_a: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_b: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> out_c: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> out_d: array<vec4<f32>>;

const C1: f32 = 0.0001; // (0.01)^2
const C2: f32 = 0.0009; // (0.03)^2

const W: array<f32, 6> = array<f32, 6>(
    0.26601818, 0.21300237, 0.10936069, 0.03599398, 0.00759866, 0.00102838,
);

fn px(x: u32, y: u32) -> u32 {
    return y * sp.width + x;
}

// products: out_a = img^2, out_b = ref^2, out_c = img*ref.
@compute @workgroup_size(256)
fn products(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sp.width * sp.height {
        return;
    }
    let x = img[i].rgb;
    let y = ref_img[i].rgb;
    out_a[i] = vec4<f32>(x * x, 0.0);
    out_b[i] = vec4<f32>(y * y, 0.0);
    out_c[i] = vec4<f32>(x * y, 0.0);
}

// Horizontal blur: in_a -> out_a (zero padding).
@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= sp.width || gid.y >= sp.height {
        return;
    }
    var acc = in_a[px(gid.x, gid.y)].rgb * W[0];
    for (var k = 1u; k <= 5u; k++) {
        if gid.x >= k {
            acc += in_a[px(gid.x - k, gid.y)].rgb * W[k];
        }
        if gid.x + k < sp.width {
            acc += in_a[px(gid.x + k, gid.y)].rgb * W[k];
        }
    }
    out_a[px(gid.x, gid.y)] = vec4<f32>(acc, 0.0);
}

// Vertical blur: in_a -> out_a.
@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= sp.width || gid.y >= sp.height {
        return;
    }
    var acc = in_a[px(gid.x, gid.y)].rgb * W[0];
    for (var k = 1u; k <= 5u; k++) {
        if gid.y >= k {
            acc += in_a[px(gid.x, gid.y - k)].rgb * W[k];
        }
        if gid.y + k < sp.height {
            acc += in_a[px(gid.x, gid.y + k)].rgb * W[k];
        }
    }
    out_a[px(gid.x, gid.y)] = vec4<f32>(acc, 0.0);
}

// partials. Host binds: in_a=mu_x, in_b=mu_y, in_c=bx2, out_c=by2, out_d=bxy.
// Writes: out_a=Cmu, out_b=Cx2, out_c=Cxy (in place over by2), out_d=SSIM map
// (in place over bxy). In-place is safe: each thread reads its own pixel
// before writing it.
@compute @workgroup_size(256)
fn partials(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sp.width * sp.height {
        return;
    }
    let mu_x = in_a[i].rgb;
    let mu_y = in_b[i].rgb;
    let bx2 = in_c[i].rgb;
    let by2 = out_c[i].rgb;
    let bxy = out_d[i].rgb;

    let sx2 = bx2 - mu_x * mu_x;
    let sy2 = by2 - mu_y * mu_y;
    let sxy = bxy - mu_x * mu_y;

    let p = mu_x * mu_x + mu_y * mu_y + C1; // den_l
    let q = sx2 + sy2 + C2;                 // den_cs
    let l = (2.0 * mu_x * mu_y + C1) / p;
    let cs = (2.0 * sxy + C2) / q;
    let s = l * cs;

    let dw = -sp.lambda * 0.5 * sp.inv_n3; // dLoss/dS per pixel-channel
    let ds_dmux = cs * (2.0 * mu_y - 2.0 * mu_x * l) / p
        + l * (-2.0 * mu_y + 2.0 * mu_x * cs) / q;
    let ds_dbx2 = -l * cs / q;
    let ds_dbxy = 2.0 * l / q;

    out_a[i] = vec4<f32>(ds_dmux * dw, 0.0); // Cmu
    out_b[i] = vec4<f32>(ds_dbx2 * dw, 0.0); // Cx2
    out_c[i] = vec4<f32>(ds_dbxy * dw, 0.0); // Cxy
    out_d[i] = vec4<f32>(s, 0.0);            // SSIM map
}

// combine: dl = (1-lambda)*sign(x-y)/N3 + bCmu + 2x*bCx2 + y*bCxy.
// Bindings: in_a=bCmu, in_b=bCx2, in_c=bCxy; out_a=dl_dcolor, out_b=l1_map.
@compute @workgroup_size(256)
fn combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sp.width * sp.height {
        return;
    }
    let x = img[i].rgb;
    let y = ref_img[i].rgb;
    let diff = x - y;
    let l1_grad = sign(diff) * ((1.0 - sp.lambda) * sp.inv_n3);
    let ssim_grad = in_a[i].rgb + 2.0 * x * in_b[i].rgb + y * in_c[i].rgb;
    out_a[i] = vec4<f32>(l1_grad + ssim_grad, 0.0);
    out_b[i] = vec4<f32>(abs(diff), 0.0);
}
