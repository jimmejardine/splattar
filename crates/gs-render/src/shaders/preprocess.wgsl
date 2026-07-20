// Viewer preprocess: frustum/near cull, EWA projection of 3D gaussians to
// screen-space ellipses, SH color evaluation, and compaction into the sort's
// key/payload buffers. One thread per splat.

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
    num_splats: u32,
    sh_degree: u32,
    splat_scale: f32,
    _pad: u32,
}

// Two vec4s per projected splat: [center.xy, axis1.xy], [axis2.xy, unused],
// plus a color vec4 — packed as 3 vec4s for linear indexing in the vertex stage.
struct Splat2D {
    center_axis1: vec4<f32>, // ndc center.xy, ndc axis1.xy (3σ)
    axis2: vec4<f32>,        // ndc axis2.xy (3σ), zw unused
    color: vec4<f32>,        // rgb, opacity
}

struct VisibleCount { n: atomic<u32> }

@group(0) @binding(0) var<uniform> cam: CameraUniform;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sh_coeffs: array<vec4<f32>>; // 12 per splat (48 f32, deg-3 padded)
@group(0) @binding(3) var<storage, read> opacities: array<f32>;
@group(0) @binding(4) var<storage, read> scales: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> rotations: array<vec4<f32>>; // glam xyzw
@group(0) @binding(6) var<storage, read_write> splats_2d: array<Splat2D>;
@group(0) @binding(7) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(8) var<storage, read_write> sort_payloads: array<u32>;
@group(0) @binding(9) var<storage, read_write> visible: VisibleCount;

const SH_C0: f32 = 0.28209479;
const SH_C1: f32 = 0.48860252;

fn eval_sh(base: u32, dir: vec3<f32>, degree: u32) -> vec3<f32> {
    // Coefficient-major rgb-interleaved: 48 floats = 12 vec4s per splat.
    // coeff k channel c lives at float index k*3+c → vec4 (k*3+c)/4, comp %4.
    var result = SH_C0 * coeff(base, 0u);
    if degree == 0u {
        return result;
    }
    let x = dir.x; let y = dir.y; let z = dir.z;
    result += -SH_C1 * y * coeff(base, 1u) + SH_C1 * z * coeff(base, 2u) - SH_C1 * x * coeff(base, 3u);
    if degree == 1u {
        return result;
    }
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let yz = y * z; let xz = x * z;
    result += 1.0925484 * xy * coeff(base, 4u)
        + (-1.0925484) * yz * coeff(base, 5u)
        + 0.31539157 * (2.0 * zz - xx - yy) * coeff(base, 6u)
        + (-1.0925484) * xz * coeff(base, 7u)
        + 0.5462742 * (xx - yy) * coeff(base, 8u);
    if degree == 2u {
        return result;
    }
    result += -0.5900436 * y * (3.0 * xx - yy) * coeff(base, 9u)
        + 2.8906114 * xy * z * coeff(base, 10u)
        + (-0.4570458) * y * (4.0 * zz - xx - yy) * coeff(base, 11u)
        + 0.37317634 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeff(base, 12u)
        + (-0.4570458) * x * (4.0 * zz - xx - yy) * coeff(base, 13u)
        + 1.4453057 * z * (xx - yy) * coeff(base, 14u)
        + (-0.5900436) * x * (xx - 3.0 * yy) * coeff(base, 15u);
    return result;
}

fn coeff(base: u32, k: u32) -> vec3<f32> {
    let f = k * 3u; // float index of this coeff's r within the splat's 48
    let r = sh_coeffs[base + f / 4u][f % 4u];
    let g = sh_coeffs[base + (f + 1u) / 4u][(f + 1u) % 4u];
    let b = sh_coeffs[base + (f + 2u) / 4u][(f + 2u) % 4u];
    return vec3<f32>(r, g, b);
}

fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
    );
}

@compute @workgroup_size(256)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= cam.num_splats {
        return;
    }

    let p_world = positions[idx].xyz;
    let p_view = (cam.view * vec4<f32>(p_world, 1.0)).xyz;
    let depth = -p_view.z; // camera looks down −z; positive in front
    if depth < 0.05 {
        return;
    }
    let clip = cam.proj * vec4<f32>(p_view, 1.0);
    let ndc = clip.xy / clip.w;
    if any(abs(ndc) > vec2<f32>(1.3, 1.3)) {
        return;
    }

    let opacity = opacities[idx];
    if opacity < 0.0039 { // 1/255
        return;
    }

    // 3D covariance Σ = R S Sᵀ Rᵀ.
    let r = quat_to_mat(rotations[idx]);
    let s = scales[idx].xyz * cam.splat_scale;
    let m = mat3x3<f32>(r[0] * s.x, r[1] * s.y, r[2] * s.z);
    let cov3d = m * transpose(m);

    // EWA: project through the view rotation and the perspective Jacobian.
    let w3 = mat3x3<f32>(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz);
    // Clamp the tangent-plane position like gsplat to tame edge distortion.
    let lim = 1.3 * cam.viewport / (2.0 * cam.focal);
    let tz = p_view.z;
    let tx = clamp(p_view.x / tz, -lim.x, lim.x) * tz;
    let ty = clamp(p_view.y / tz, -lim.y, lim.y) * tz;
    let j = mat3x3<f32>(
        vec3<f32>(cam.focal.x / tz, 0.0, 0.0),
        vec3<f32>(0.0, cam.focal.y / tz, 0.0),
        vec3<f32>(-cam.focal.x * tx / (tz * tz), -cam.focal.y * ty / (tz * tz), 0.0),
    );
    let t = j * w3;
    let cov2d_full = t * cov3d * transpose(t);
    // Low-pass: every splat covers at least ~a pixel (0.3 like INRIA).
    let a = cov2d_full[0][0] + 0.3;
    let d = cov2d_full[1][1] + 0.3;
    let b = cov2d_full[0][1];

    let det = a * d - b * b;
    if det <= 0.0 {
        return;
    }
    let mid = 0.5 * (a + d);
    let disc = sqrt(max(mid * mid - det, 1e-9));
    let l1 = mid + disc;
    let l2 = max(mid - disc, 1e-9);

    var evec1: vec2<f32>;
    if abs(b) > 1e-9 {
        evec1 = normalize(vec2<f32>(b, l1 - a));
    } else if a >= d {
        evec1 = vec2<f32>(1.0, 0.0);
    } else {
        evec1 = vec2<f32>(0.0, 1.0);
    }
    let evec2 = vec2<f32>(-evec1.y, evec1.x);

    let r1 = 3.0 * sqrt(l1); // pixels
    let r2 = 3.0 * sqrt(l2);
    if r1 < 0.3 || r1 > 4096.0 || !(det == det) {
        return; // sub-pixel after low-pass shouldn't happen; NaN guard
    }

    // Pixel-space axes → NDC (signs are irrelevant for a symmetric ellipse).
    let px_to_ndc = 2.0 / cam.viewport;
    let axis1 = evec1 * r1 * px_to_ndc;
    let axis2 = evec2 * r2 * px_to_ndc;

    let dir = normalize(p_world - cam.cam_pos.xyz);
    let rgb = max(eval_sh(idx * 12u, dir, cam.sh_degree) + vec3<f32>(0.5), vec3<f32>(0.0));

    let slot = atomicAdd(&visible.n, 1u);
    splats_2d[slot] = Splat2D(
        vec4<f32>(ndc, axis1),
        vec4<f32>(axis2, 0.0, 0.0),
        vec4<f32>(rgb, opacity),
    );
    // Ascending sort on the inverted depth bits ⇒ back-to-front instances.
    sort_keys[slot] = 0xffffffffu - bitcast<u32>(depth);
    sort_payloads[slot] = slot;
}
