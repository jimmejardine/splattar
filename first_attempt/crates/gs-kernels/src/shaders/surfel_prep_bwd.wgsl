// Geometry backward: chain the per-surfel camera-space gradients
// (dc, dτu, dτv, dcolor, dopacity — accumulated by rasterize_bwd) into
// parameter space: position, scales, quaternion, opacity, SH, plus the camera
// rotation/center accumulators. One thread per surfel; parameter outputs are
// plain writes, camera accumulators are CAS float adds.
// (raster_common.wgsl is prepended by the host.)

@group(0) @binding(0) var<uniform> cam: RasterCamera;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> scales: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> quats: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> sh_coeffs: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> surf_cam: array<SurfelCam>;
@group(0) @binding(6) var<storage, read> grad_geom: array<f32>; // 13 per surfel
@group(0) @binding(7) var<storage, read_write> grad_pos: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> grad_scales: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> grad_quat: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> grad_opacity: array<f32>;
@group(0) @binding(11) var<storage, read_write> grad_sh: array<vec4<f32>>; // 12 per surfel
@group(0) @binding(12) var<storage, read_write> grad_cam: array<atomic<u32>>;

const GS: u32 = 16u; // == GRAD_STRIDE
const SH_C0: f32 = 0.28209479;
const SH_C1: f32 = 0.48860252;

fn cam_add(idx: u32, v: f32) {
    if v == 0.0 {
        return;
    }
    var old = atomicLoad(&grad_cam[idx]);
    loop {
        let newv = bitcast<u32>(bitcast<f32>(old) + v);
        let r = atomicCompareExchangeWeak(&grad_cam[idx], old, newv);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}

// Uniform-slot camera accumulation: every surfel thread adds into the SAME
// 13 global slots, so the host swaps this body for a subgroup pre-reduction
// when the device has subgroups (see Rasterizer::new) — one CAS per subgroup
// instead of one per thread on a single hot address.
fn red_cam_add(idx: u32, v: f32) {
    cam_add(idx, v);
}

// dL/dR_cam accumulation for a = R_camᵀ y chains: dR_{j,k} += y_j · da_k.
// grad_cam layout: [R00 R10 R20 R01 R11 R21 R02 R12 R22, dC.xyz, dfocal, pad].
fn cam_rot_chain(y: vec3<f32>, da: vec3<f32>) {
    for (var k = 0u; k < 3u; k++) {
        red_cam_add(k * 3u + 0u, y.x * da[k]);
        red_cam_add(k * 3u + 1u, y.y * da[k]);
        red_cam_add(k * 3u + 2u, y.z * da[k]);
    }
}

fn cam_rot_mul(v: vec3<f32>) -> vec3<f32> {
    // R · v (columns rot0..2).
    return cam.rot0.xyz * v.x + cam.rot1.xyz * v.y + cam.rot2.xyz * v.z;
}

fn quat_mat(q_in: vec4<f32>) -> mat3x3<f32> {
    let q = normalize(q_in);
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
    );
}

// dL/dq (unnormalized) from dL/dR, mirroring gs-cpu-ref::math::quat_grad.
fn quat_grad(q_in: vec4<f32>, dl_dr: mat3x3<f32>) -> vec4<f32> {
    let n = length(q_in);
    let q = q_in / n;
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;

    var dr: array<mat3x3<f32>, 4>;
    dr[0] = mat3x3<f32>(
        vec3<f32>(0.0, 2.0 * y, 2.0 * z),
        vec3<f32>(2.0 * y, -4.0 * x, 2.0 * w),
        vec3<f32>(2.0 * z, -2.0 * w, -4.0 * x),
    );
    dr[1] = mat3x3<f32>(
        vec3<f32>(-4.0 * y, 2.0 * x, -2.0 * w),
        vec3<f32>(2.0 * x, 0.0, 2.0 * z),
        vec3<f32>(2.0 * w, 2.0 * z, -4.0 * y),
    );
    dr[2] = mat3x3<f32>(
        vec3<f32>(-4.0 * z, 2.0 * w, 2.0 * x),
        vec3<f32>(-2.0 * w, -4.0 * z, 2.0 * y),
        vec3<f32>(2.0 * x, 2.0 * y, 0.0),
    );
    dr[3] = mat3x3<f32>(
        vec3<f32>(0.0, 2.0 * z, -2.0 * y),
        vec3<f32>(-2.0 * z, 0.0, 2.0 * x),
        vec3<f32>(2.0 * y, -2.0 * x, 0.0),
    );

    var dqh = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        var acc = 0.0;
        for (var col = 0u; col < 3u; col++) {
            acc += dot(dl_dr[col], dr[k][col]);
        }
        dqh[k] = acc;
    }
    let d = dot(dqh, q);
    return (dqh - q * d) / n;
}

fn sh_coeff(base: u32, k: u32) -> vec3<f32> {
    let f = k * 3u;
    let r = sh_coeffs[base + f / 4u][f % 4u];
    let g = sh_coeffs[base + (f + 1u) / 4u][(f + 1u) % 4u];
    let b = sh_coeffs[base + (f + 2u) / 4u][(f + 2u) % 4u];
    return vec3<f32>(r, g, b);
}

fn grad_sh_write(base_out: u32, k: u32, v: vec3<f32>) {
    let f = k * 3u;
    grad_sh[base_out + f / 4u][f % 4u] = v.x;
    grad_sh[base_out + (f + 1u) / 4u][(f + 1u) % 4u] = v.y;
    grad_sh[base_out + (f + 2u) / 4u][(f + 2u) % 4u] = v.z;
}

@compute @workgroup_size(256)
fn surfel_prep_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cam.num_surfels {
        return;
    }
    let g = i * GS;
    let dc = vec3<f32>(grad_geom[g], grad_geom[g + 1u], grad_geom[g + 2u]);
    var dtu = vec3<f32>(grad_geom[g + 3u], grad_geom[g + 4u], grad_geom[g + 5u]);
    var dtv = vec3<f32>(grad_geom[g + 6u], grad_geom[g + 7u], grad_geom[g + 8u]);
    var dcolor = vec3<f32>(grad_geom[g + 9u], grad_geom[g + 10u], grad_geom[g + 11u]);
    let dopacity = grad_geom[g + 12u];
    let dnormal = vec3<f32>(grad_geom[g + 13u], grad_geom[g + 14u], grad_geom[g + 15u]);

    // Normal chain: n = sign · normalize(τu_cam × τv_cam); τ and c come from
    // the forward prep record.
    if dnormal.x != 0.0 || dnormal.y != 0.0 || dnormal.z != 0.0 {
        let sc = surf_cam[i];
        let m = cross(sc.tu.xyz, sc.tv.xyz);
        let len2 = dot(m, m);
        if len2 > 1e-24 {
            let len = sqrt(len2);
            let mhat = m / len;
            var sign = 1.0;
            if dot(mhat, sc.c.xyz) > 0.0 {
                sign = -1.0;
            }
            let dl_dm = (dnormal - mhat * dot(mhat, dnormal)) * (sign / len);
            dtu += cross(sc.tv.xyz, dl_dm);
            dtv += cross(dl_dm, sc.tu.xyz);
        }
    }

    var dpos = vec3<f32>(0.0);
    var dcenter = vec3<f32>(0.0);

    // c = R_camᵀ (p − C)
    let p = positions[i].xyz;
    let x = p - cam.center.xyz;
    let dx = cam_rot_mul(dc);
    dpos += dx;
    dcenter -= dx;
    cam_rot_chain(x, dc);

    // τ axes: τ = R_camᵀ (s · Rs·e_axis)
    let rs = quat_mat(quats[i]);
    let s = scales[i];
    var dl_drs = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    let dy_u = cam_rot_mul(dtu);
    let axis_u = rs[0];
    let dscale_u = dot(dy_u, axis_u);
    dl_drs[0] = dy_u * s.x;
    cam_rot_chain(axis_u * s.x, dtu);

    let dy_v = cam_rot_mul(dtv);
    let axis_v = rs[1];
    let dscale_v = dot(dy_v, axis_v);
    dl_drs[1] = dy_v * s.y;
    cam_rot_chain(axis_v * s.y, dtv);

    let dq = quat_grad(quats[i], dl_drs);

    // Color → SH + view-direction path (clamp mask from the forward prep).
    let clamp_bits = u32(surf_cam[i].normal_fl.w);
    if (clamp_bits & 1u) != 0u { dcolor.x = 0.0; }
    if (clamp_bits & 2u) != 0u { dcolor.y = 0.0; }
    if (clamp_bits & 4u) != 0u { dcolor.z = 0.0; }

    let sh_base = i * 12u;
    if dcolor.x != 0.0 || dcolor.y != 0.0 || dcolor.z != 0.0 {
        let len = length(x);
        let dir = x / len;
        let dxd = dir.x; let dyd = dir.y; let dzd = dir.z;
        var ddir = vec3<f32>(0.0);

        // Degree 0.
        grad_sh_write(sh_base, 0u, dcolor * SH_C0);
        if cam.sh_degree >= 1u {
            grad_sh_write(sh_base, 1u, dcolor * (-SH_C1 * dyd));
            grad_sh_write(sh_base, 2u, dcolor * (SH_C1 * dzd));
            grad_sh_write(sh_base, 3u, dcolor * (-SH_C1 * dxd));
            ddir += vec3<f32>(0.0, -SH_C1, 0.0) * dot(sh_coeff(sh_base, 1u), dcolor);
            ddir += vec3<f32>(0.0, 0.0, SH_C1) * dot(sh_coeff(sh_base, 2u), dcolor);
            ddir += vec3<f32>(-SH_C1, 0.0, 0.0) * dot(sh_coeff(sh_base, 3u), dcolor);
        }
        if cam.sh_degree >= 2u {
            let xx = dxd * dxd; let yy = dyd * dyd; let zz = dzd * dzd;
            let b4 = 1.0925484 * dxd * dyd;
            let b5 = -1.0925484 * dyd * dzd;
            let b6 = 0.31539157 * (2.0 * zz - xx - yy);
            let b7 = -1.0925484 * dxd * dzd;
            let b8 = 0.5462742 * (xx - yy);
            grad_sh_write(sh_base, 4u, dcolor * b4);
            grad_sh_write(sh_base, 5u, dcolor * b5);
            grad_sh_write(sh_base, 6u, dcolor * b6);
            grad_sh_write(sh_base, 7u, dcolor * b7);
            grad_sh_write(sh_base, 8u, dcolor * b8);
            ddir += 1.0925484 * vec3<f32>(dyd, dxd, 0.0) * dot(sh_coeff(sh_base, 4u), dcolor);
            ddir += -1.0925484 * vec3<f32>(0.0, dzd, dyd) * dot(sh_coeff(sh_base, 5u), dcolor);
            ddir += 0.31539157 * vec3<f32>(-2.0 * dxd, -2.0 * dyd, 4.0 * dzd)
                * dot(sh_coeff(sh_base, 6u), dcolor);
            ddir += -1.0925484 * vec3<f32>(dzd, 0.0, dxd) * dot(sh_coeff(sh_base, 7u), dcolor);
            ddir += 0.5462742 * vec3<f32>(2.0 * dxd, -2.0 * dyd, 0.0)
                * dot(sh_coeff(sh_base, 8u), dcolor);
        }
        if cam.sh_degree >= 3u {
            let xx = dxd * dxd; let yy = dyd * dyd; let zz = dzd * dzd;
            let b9 = -0.5900436 * dyd * (3.0 * xx - yy);
            let b10 = 2.8906114 * dxd * dyd * dzd;
            let b11 = -0.4570458 * dyd * (4.0 * zz - xx - yy);
            let b12 = 0.37317634 * dzd * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            let b13 = -0.4570458 * dxd * (4.0 * zz - xx - yy);
            let b14 = 1.4453057 * dzd * (xx - yy);
            let b15 = -0.5900436 * dxd * (xx - 3.0 * yy);
            grad_sh_write(sh_base, 9u, dcolor * b9);
            grad_sh_write(sh_base, 10u, dcolor * b10);
            grad_sh_write(sh_base, 11u, dcolor * b11);
            grad_sh_write(sh_base, 12u, dcolor * b12);
            grad_sh_write(sh_base, 13u, dcolor * b13);
            grad_sh_write(sh_base, 14u, dcolor * b14);
            grad_sh_write(sh_base, 15u, dcolor * b15);
            ddir += -0.5900436 * vec3<f32>(6.0 * dxd * dyd, 3.0 * xx - 3.0 * yy, 0.0)
                * dot(sh_coeff(sh_base, 9u), dcolor);
            ddir += 2.8906114 * vec3<f32>(dyd * dzd, dxd * dzd, dxd * dyd)
                * dot(sh_coeff(sh_base, 10u), dcolor);
            ddir += -0.4570458 * vec3<f32>(-2.0 * dxd * dyd, 4.0 * zz - xx - 3.0 * yy, 8.0 * dyd * dzd)
                * dot(sh_coeff(sh_base, 11u), dcolor);
            ddir += 0.37317634
                * vec3<f32>(-6.0 * dxd * dzd, -6.0 * dyd * dzd, 6.0 * zz - 3.0 * xx - 3.0 * yy)
                * dot(sh_coeff(sh_base, 12u), dcolor);
            ddir += -0.4570458 * vec3<f32>(4.0 * zz - 3.0 * xx - yy, -2.0 * dxd * dyd, 8.0 * dxd * dzd)
                * dot(sh_coeff(sh_base, 13u), dcolor);
            ddir += 1.4453057 * vec3<f32>(2.0 * dxd * dzd, -2.0 * dyd * dzd, xx - yy)
                * dot(sh_coeff(sh_base, 14u), dcolor);
            ddir += -0.5900436 * vec3<f32>(3.0 * xx - 3.0 * yy, -6.0 * dxd * dyd, 0.0)
                * dot(sh_coeff(sh_base, 15u), dcolor);
        }
        let dv = (ddir - dir * dot(dir, ddir)) / len;
        dpos += dv;
        dcenter -= dv;
    } else {
        // Zero grads still need deterministic outputs.
        for (var k = 0u; k < 16u; k++) {
            grad_sh_write(sh_base, k, vec3<f32>(0.0));
        }
    }

    grad_pos[i] = vec4<f32>(dpos, 0.0);
    grad_scales[i] = vec2<f32>(dscale_u, dscale_v);
    grad_quat[i] = dq;
    grad_opacity[i] = dopacity;
    red_cam_add(9u, dcenter.x);
    red_cam_add(10u, dcenter.y);
    red_cam_add(11u, dcenter.z);
}
