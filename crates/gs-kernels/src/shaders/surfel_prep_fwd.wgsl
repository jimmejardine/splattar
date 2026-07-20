// Surfel preprocess: world parameters → camera-space SurfelCam records plus
// tile rects + depth keys for the binner. One thread per surfel.
// (raster_common.wgsl is prepended by the host.)

@group(0) @binding(0) var<uniform> cam: RasterCamera;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> scales: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> quats: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read> sh_coeffs: array<vec4<f32>>; // 12 per surfel, deg-3 padded
@group(0) @binding(6) var<storage, read_write> surf_cam: array<SurfelCam>;
@group(0) @binding(7) var<storage, read_write> rects: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read_write> depth_keys: array<u32>;

const SH_C0: f32 = 0.28209479;
const SH_C1: f32 = 0.48860252;

fn sh_coeff(base: u32, k: u32) -> vec3<f32> {
    let f = k * 3u;
    let r = sh_coeffs[base + f / 4u][f % 4u];
    let g = sh_coeffs[base + (f + 1u) / 4u][(f + 1u) % 4u];
    let b = sh_coeffs[base + (f + 2u) / 4u][(f + 2u) % 4u];
    return vec3<f32>(r, g, b);
}

fn eval_sh_color(idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let base = idx * 12u;
    var c = SH_C0 * sh_coeff(base, 0u);
    if cam.sh_degree >= 1u {
        let x = dir.x; let y = dir.y; let z = dir.z;
        c += -SH_C1 * y * sh_coeff(base, 1u) + SH_C1 * z * sh_coeff(base, 2u)
            - SH_C1 * x * sh_coeff(base, 3u);
        if cam.sh_degree >= 2u {
            let xx = x * x; let yy = y * y; let zz = z * z;
            c += 1.0925484 * x * y * sh_coeff(base, 4u)
                + (-1.0925484) * y * z * sh_coeff(base, 5u)
                + 0.31539157 * (2.0 * zz - xx - yy) * sh_coeff(base, 6u)
                + (-1.0925484) * x * z * sh_coeff(base, 7u)
                + 0.5462742 * (xx - yy) * sh_coeff(base, 8u);
            if cam.sh_degree >= 3u {
                c += -0.5900436 * y * (3.0 * xx - yy) * sh_coeff(base, 9u)
                    + 2.8906114 * x * y * z * sh_coeff(base, 10u)
                    + (-0.4570458) * y * (4.0 * zz - xx - yy) * sh_coeff(base, 11u)
                    + 0.37317634 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeff(base, 12u)
                    + (-0.4570458) * x * (4.0 * zz - xx - yy) * sh_coeff(base, 13u)
                    + 1.4453057 * z * (xx - yy) * sh_coeff(base, 14u)
                    + (-0.5900436) * x * (xx - 3.0 * yy) * sh_coeff(base, 15u);
            }
        }
    }
    return c + vec3<f32>(0.5);
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

@compute @workgroup_size(256)
fn surfel_prep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cam.num_surfels {
        return;
    }
    // Degenerate rect (max < min) = culled for the binner.
    var rect = vec4<u32>(1u, 1u, 0u, 0u);

    let p = positions[i].xyz;
    let x = p - cam.center.xyz;
    let c = cam_rot_transpose_mul(cam, x);
    let depth = -c.z;
    if depth < NEAR_DEPTH {
        rects[i] = rect;
        depth_keys[i] = 0u;
        return;
    }

    let rs = quat_mat(quats[i]);
    let s = scales[i];
    let tu = cam_rot_transpose_mul(cam, rs[0] * s.x);
    let tv = cam_rot_transpose_mul(cam, rs[1] * s.y);

    let dir = normalize(x);
    let raw = eval_sh_color(i, dir);
    let clamp_mask = f32(u32(raw.x < 0.0) | (u32(raw.y < 0.0) << 1u) | (u32(raw.z < 0.0) << 2u));
    let color = max(raw, vec3<f32>(0.0));

    var normal = cross(tu, tv);
    if dot(normal, normal) > 0.0 {
        normal = normalize(normal);
        if dot(normal, c) > 0.0 {
            normal = -normal;
        }
    }

    let w = depth;
    let cx = f32(cam.width) * 0.5;
    let cy = f32(cam.height) * 0.5;
    let px = cx + cam.focal * c.x / w;
    let py = cy - cam.focal * c.y / w;

    surf_cam[i] = SurfelCam(
        vec4<f32>(c, depth),
        vec4<f32>(tu, px),
        vec4<f32>(tv, py),
        vec4<f32>(color, opacities[i]),
        vec4<f32>(normal, clamp_mask),
    );

    // Conservative pixel radius: 3σ of both axes projected + low-pass reach.
    let r_px = 3.0 * cam.focal * (length(tu) + length(tv)) / depth + 3.0;
    let x0 = u32(clamp((px - r_px) / f32(TILE), 0.0, f32(cam.tiles_x - 1u)));
    let y0 = u32(clamp((py - r_px) / f32(TILE), 0.0, f32(cam.tiles_y - 1u)));
    let x1 = u32(clamp((px + r_px) / f32(TILE), 0.0, f32(cam.tiles_x - 1u)));
    let y1 = u32(clamp((py + r_px) / f32(TILE), 0.0, f32(cam.tiles_y - 1u)));
    // Fully off-screen splats keep the degenerate rect.
    if px + r_px >= 0.0 && px - r_px < f32(cam.width) && py + r_px >= 0.0
        && py - r_px < f32(cam.height)
    {
        rect = vec4<u32>(x0, y0, x1, y1);
    }
    rects[i] = rect;
    depth_keys[i] = bitcast<u32>(depth);
}
