// Normal-consistency loss (2DGS): rendered normals must agree with normals
// implied by the rendered depth map. Per pixel q with mean depth
// tbar = depth_acc/alpha: P(q) = tbar * ray(q); N_d = normalize(dPx x dPy)
// from forward differences, oriented toward the camera;
// L = lambda * sum_q alpha_q * (1 - nhat . N_d), alpha and the orientation
// sign treated as detached.
//
// pass1 computes dl_dnormal (into the rasterizer's input) plus the
// coefficient maps cpx = dL/d(dPx), cpy = dL/d(dPy) at each stencil center.
// pass2 is the gather adjoint of the forward differences: each pixel sums the
// stencils that touch it and writes dL/d(depth_acc) into dl_dcolor.w.

struct NormalLossParams {
    width: u32,
    height: u32,
    focal: f32,
    lambda: f32, // includes 1/N normalization; 0 disables
    alpha_min: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

@group(0) @binding(0) var<uniform> np: NormalLossParams;
@group(0) @binding(1) var<storage, read> out_color: array<vec4<f32>>;  // alpha in .w
@group(0) @binding(2) var<storage, read> out_aux: array<vec4<f32>>;    // depth_acc in .y
@group(0) @binding(3) var<storage, read> out_normal: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> cpx: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> cpy: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> dl_dnormal: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> loss_map: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> dl_dcolor: array<vec4<f32>>;

fn ray_at(x: u32, y: u32) -> vec3<f32> {
    let cx = f32(np.width) * 0.5;
    let cy = f32(np.height) * 0.5;
    return vec3<f32>(
        (f32(x) + 0.5 - cx) / np.focal,
        -(f32(y) + 0.5 - cy) / np.focal,
        -1.0,
    );
}

fn tbar(i: u32) -> f32 {
    let a = out_color[i].w;
    if a < np.alpha_min {
        return 0.0;
    }
    return out_aux[i].y / a;
}

@compute @workgroup_size(256)
fn pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n_px = np.width * np.height;
    if i >= n_px {
        return;
    }
    cpx[i] = vec4<f32>(0.0);
    cpy[i] = vec4<f32>(0.0);
    dl_dnormal[i] = vec4<f32>(0.0);
    loss_map[i] = vec4<f32>(0.0);
    if np.lambda == 0.0 {
        return;
    }
    let x = i % np.width;
    let y = i / np.width;
    if x + 1u >= np.width || y + 1u >= np.height {
        return;
    }
    let a_q = out_color[i].w;
    let ix = i + 1u;
    let iy = i + np.width;
    if a_q < np.alpha_min || out_color[ix].w < np.alpha_min || out_color[iy].w < np.alpha_min {
        return;
    }

    let p = tbar(i) * ray_at(x, y);
    let px_p = tbar(ix) * ray_at(x + 1u, y);
    let py_p = tbar(iy) * ray_at(x, y + 1u);
    let dpx = px_p - p;
    let dpy = py_p - p;
    let m = cross(dpx, dpy);
    let len2 = dot(m, m);
    if len2 < 1e-18 {
        return;
    }
    let len = sqrt(len2);
    let mhat = m / len;
    var sign = 1.0;
    if dot(mhat, p) > 0.0 {
        sign = -1.0;
    }
    let n_d = sign * mhat;

    let a_vec = out_normal[i].xyz;
    let la2 = dot(a_vec, a_vec);
    if la2 < 1e-18 {
        return;
    }
    let la = sqrt(la2);
    let nhat = a_vec / la;

    loss_map[i] = vec4<f32>((1.0 - dot(nhat, n_d)) * a_q, 0.0, 0.0, 0.0);

    // Gradients (alpha weight and orientation sign detached).
    let dl_dnd = -np.lambda * a_q * nhat;
    let dl_dnhat = -np.lambda * a_q * n_d;
    let dl_da_vec = (dl_dnhat - nhat * dot(nhat, dl_dnhat)) / la;
    dl_dnormal[i] = vec4<f32>(dl_da_vec, 0.0);

    let dl_dm = sign * (dl_dnd - mhat * dot(mhat, dl_dnd)) / len;
    cpx[i] = vec4<f32>(cross(dpy, dl_dm), 0.0);
    cpy[i] = vec4<f32>(cross(dl_dm, dpx), 0.0);
}

@compute @workgroup_size(256)
fn pass2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n_px = np.width * np.height;
    if i >= n_px {
        return;
    }
    var c = dl_dcolor[i];
    c.w = 0.0;
    if np.lambda != 0.0 {
        let x = i % np.width;
        let y = i / np.width;
        // Adjoint of the forward differences: center contributes −cpx −cpy;
        // left neighbor's cpx and upper neighbor's cpy contribute +.
        var acc = -cpx[i].xyz - cpy[i].xyz;
        if x > 0u {
            acc += cpx[i - 1u].xyz;
        }
        if y > 0u {
            acc += cpy[i - np.width].xyz;
        }
        let a_q = out_color[i].w;
        if a_q >= np.alpha_min {
            // dL/d(depth_acc) = dL/dtbar / alpha (alpha detached).
            c.w = dot(ray_at(x, y), acc) / a_q;
        }
    }
    dl_dcolor[i] = c;
}
