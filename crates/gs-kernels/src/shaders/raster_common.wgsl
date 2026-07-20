// Shared declarations for the training rasterizer kernels. This file is
// concatenated in front of surfel_prep/rasterize kernels at pipeline build
// (include_str! + format in the host) — WGSL has no #include.

struct RasterCamera {
    // Columns of the camera-to-world rotation R (x_cam = Rᵀ (p − C)).
    rot0: vec4<f32>,
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    center: vec4<f32>,
    focal: f32,
    width: u32,
    height: u32,
    sh_degree: u32,
    num_surfels: u32,
    tiles_x: u32,
    tiles_y: u32,
    _pad: u32,
}

// Camera-space surfel produced by surfel_prep, consumed by fwd/bwd kernels.
struct SurfelCam {
    c: vec4<f32>,        // xyz = center (camera space), w = center depth
    tu: vec4<f32>,       // xyz = τu, w = projected center px
    tv: vec4<f32>,       // xyz = τv, w = projected center py
    color_op: vec4<f32>, // rgb = SH color (clamped), w = opacity
    normal_fl: vec4<f32>,// xyz = camera-space normal, w = clamp mask bits
}

const LOWPASS_SIGMA2: f32 = 0.5;
const ALPHA_SKIP: f32 = 0.0039215686; // 1/255
const ALPHA_CLAMP: f32 = 0.995;
const T_TERMINATE: f32 = 1e-4;
const NEAR_DEPTH: f32 = 0.05;
const TILE: u32 = 16u;

fn cam_rot_transpose_mul(cam: RasterCamera, v: vec3<f32>) -> vec3<f32> {
    // Rᵀ v where rot columns are R's columns.
    return vec3<f32>(
        dot(cam.rot0.xyz, v),
        dot(cam.rot1.xyz, v),
        dot(cam.rot2.xyz, v),
    );
}

fn pixel_ray(cam: RasterCamera, px: f32, py: f32) -> vec3<f32> {
    let cx = f32(cam.width) * 0.5;
    let cy = f32(cam.height) * 0.5;
    return vec3<f32>((px - cx) / cam.focal, -(py - cy) / cam.focal, -1.0);
}

fn triple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    return dot(a, cross(b, c));
}

// Evaluate ĝ = max(G_ray, G_screen) exactly like the CPU oracle.
struct HitEval {
    ghat: f32,
    ray_branch: bool,
    u: f32,
    v: f32,
    t: f32,
}

fn evaluate_hit(sc: SurfelCam, d: vec3<f32>, pix_x: f32, pix_y: f32) -> HitEval {
    let s_vec = -d;
    let det = triple(sc.tu.xyz, sc.tv.xyz, s_vec);
    var g_ray = 0.0;
    var u = 0.0;
    var v = 0.0;
    var t = 0.0;
    if abs(det) > 1e-12 {
        u = triple(-sc.c.xyz, sc.tv.xyz, s_vec) / det;
        v = triple(sc.tu.xyz, -sc.c.xyz, s_vec) / det;
        t = triple(sc.tu.xyz, sc.tv.xyz, -sc.c.xyz) / det;
        if t > NEAR_DEPTH {
            g_ray = exp(-(u * u + v * v) * 0.5);
        }
    }
    let dx = pix_x - sc.tu.w;
    let dy = pix_y - sc.tv.w;
    let g_scr = exp(-(dx * dx + dy * dy) / (2.0 * LOWPASS_SIGMA2));

    var out: HitEval;
    if g_ray >= g_scr {
        out = HitEval(g_ray, true, u, v, t);
    } else {
        out = HitEval(g_scr, false, u, v, sc.c.w);
    }
    return out;
}
