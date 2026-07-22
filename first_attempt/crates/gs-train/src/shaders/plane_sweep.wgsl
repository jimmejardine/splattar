// Plane-sweep depth for surfel initialization.
//
// For one reference view, sweep a set of fronto-parallel depth hypotheses and
// score each against 2-4 neighbouring views by zero-mean normalized cross
// correlation over a small patch. Winner-take-all with a subpixel parabola fit
// on the cost curve; the margin between the best and second-best hypothesis
// becomes a confidence the host uses to drop unreliable pixels rather than
// spawn them as floaters.
//
// NCC, not SSD: phone auto-exposure sweeps gain AND offset continuously
// through a walkthrough. Zero-mean alone (what the KLT tracker uses) absorbs
// the offset; normalizing by the patch standard deviation absorbs the gain
// too. A raw SSD cost volume on this footage would mostly measure exposure.
//
// Camera convention matches surfel_prep_fwd.wgsl exactly:
//   x_cam = Rᵀ (p_world − C), camera looks down −z,
//   px = cx + f·x/depth,  py = cy − f·y/depth,  depth = −z_cam.

struct SweepCam {
    // Columns of the camera-to-world rotation R.
    rot0: vec4<f32>,
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    center: vec4<f32>,
    focal: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

struct SweepParams {
    // Sweep grid (coarser than the targets; depth does not need full res).
    width: u32,
    height: u32,
    // Target atlas image size, and its stride in vec4 elements.
    full_width: u32,
    full_height: u32,
    stride_texels: u32,
    n_hypotheses: u32,
    n_neighbours: u32,
    patch_r: i32,
    // Hypotheses are uniform in INVERSE depth — equal disparity steps, which
    // is where the photometric signal is linear. Uniform depth steps waste
    // most of the volume on far geometry nothing can resolve.
    inv_depth_min: f32,
    inv_depth_max: f32,
    ref_slot: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> pp: SweepParams;
// [0] is the reference camera, [1..] the neighbours, parallel to slots.
@group(0) @binding(1) var<storage, read> cams: array<SweepCam>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
@group(0) @binding(3) var<storage, read> atlas: array<vec4<f32>>;
// depth (0 = rejected) and confidence per sweep pixel.
@group(0) @binding(4) var<storage, read_write> out_depth: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_conf: array<f32>;

const NEAR: f32 = 0.05;
const MAX_PATCH: i32 = 3;
// Cost curve is kept per invocation so the peak can be analysed after the
// sweep; the host clamps n_hypotheses to this.
const MAX_H: u32 = 128u;
// Hypotheses within this many steps of the winner are part of the SAME peak
// and are excluded from the runner-up. Without this the "margin" measures the
// curve's smoothness (adjacent hypotheses always score nearly the same) rather
// than its uniqueness, and rejects every correct match.
const PEAK_EXCLUDE: u32 = 3u;

fn luma(c: vec4<f32>) -> f32 {
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

// Bilinear sample of one atlas slot at full-res pixel coordinates.
fn sample_luma(slot: u32, x: f32, y: f32) -> f32 {
    let w = f32(pp.full_width);
    let h = f32(pp.full_height);
    let cx = clamp(x, 0.0, w - 1.001);
    let cy = clamp(y, 0.0, h - 1.001);
    let x0 = u32(cx);
    let y0 = u32(cy);
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let base = slot * pp.stride_texels;
    let row0 = base + y0 * pp.full_width;
    let row1 = row0 + pp.full_width;
    let a = luma(atlas[row0 + x0]);
    let b = luma(atlas[row0 + x0 + 1u]);
    let c = luma(atlas[row1 + x0]);
    let d = luma(atlas[row1 + x0 + 1u]);
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

fn cam_rot_t_mul(cm: SweepCam, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(cm.rot0.xyz, v), dot(cm.rot1.xyz, v), dot(cm.rot2.xyz, v));
}

fn cam_rot_mul(cm: SweepCam, v: vec3<f32>) -> vec3<f32> {
    return cm.rot0.xyz * v.x + cm.rot1.xyz * v.y + cm.rot2.xyz * v.z;
}

// Back-project a reference-view pixel at `depth` to world space.
fn unproject(cm: SweepCam, px: f32, py: f32, depth: f32) -> vec3<f32> {
    let cx = f32(pp.full_width) * 0.5;
    let cy = f32(pp.full_height) * 0.5;
    let c = vec3<f32>(
        (px - cx) * depth / cm.focal,
        -(py - cy) * depth / cm.focal,
        -depth,
    );
    return cam_rot_mul(cm, c) + cm.center.xyz;
}

// Project a world point into a view. Returns (px, py, depth); depth <= 0 means
// behind the camera.
fn project(cm: SweepCam, p: vec3<f32>) -> vec3<f32> {
    let c = cam_rot_t_mul(cm, p - cm.center.xyz);
    let depth = -c.z;
    if depth <= NEAR {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    let cx = f32(pp.full_width) * 0.5;
    let cy = f32(pp.full_height) * 0.5;
    return vec3<f32>(cx + cm.focal * c.x / depth, cy - cm.focal * c.y / depth, depth);
}

@compute @workgroup_size(8, 8)
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= pp.width || gid.y >= pp.height {
        return;
    }
    let out_i = gid.y * pp.width + gid.x;
    out_depth[out_i] = 0.0;
    out_conf[out_i] = 0.0;

    // Sweep grid pixel -> full-res reference pixel (cell centre).
    let sx = f32(pp.full_width) / f32(pp.width);
    let sy = f32(pp.full_height) / f32(pp.height);
    let rx = (f32(gid.x) + 0.5) * sx;
    let ry = (f32(gid.y) + 0.5) * sy;

    let ref_cam = cams[0];
    let ref_slot = slots[0];
    let p = clamp(pp.patch_r, 1, MAX_PATCH);
    let n_samples = f32((2 * p + 1) * (2 * p + 1));

    // Reference patch statistics, computed once for all hypotheses. Patch
    // spacing follows the sweep grid so the window covers the same solid
    // angle it would at full resolution.
    var ref_sum = 0.0;
    var ref_sum2 = 0.0;
    for (var dy = -p; dy <= p; dy++) {
        for (var dx = -p; dx <= p; dx++) {
            let v = sample_luma(ref_slot, rx + f32(dx) * sx, ry + f32(dy) * sy);
            ref_sum += v;
            ref_sum2 += v * v;
        }
    }
    let ref_mean = ref_sum / n_samples;
    let ref_var = max(ref_sum2 / n_samples - ref_mean * ref_mean, 0.0);
    // A textureless patch cannot be matched at any depth. Reject rather than
    // let noise pick a winner — this is most of a blank wall, and spawning
    // surfels there from a noise-driven argmin is how floaters are born.
    if ref_var < 1e-6 {
        return;
    }
    let ref_std = sqrt(ref_var);

    var costs: array<f32, MAX_H>;
    let n_h = min(pp.n_hypotheses, MAX_H);
    let denom = max(f32(n_h) - 1.0, 1.0);
    for (var h = 0u; h < n_h; h++) {
        costs[h] = -2.0;
        let inv_d = mix(pp.inv_depth_min, pp.inv_depth_max, f32(h) / denom);
        let depth = 1.0 / max(inv_d, 1e-8);

        // Mean NCC across neighbours that see this hypothesis.
        var score_sum = 0.0;
        var seen = 0u;
        for (var n = 0u; n < pp.n_neighbours; n++) {
            let ncam = cams[n + 1u];
            let nslot = slots[n + 1u];
            var nb_sum = 0.0;
            var nb_sum2 = 0.0;
            var cross = 0.0;
            var ok = true;
            for (var dy = -p; dy <= p; dy++) {
                for (var dx = -p; dx <= p; dx++) {
                    let world = unproject(
                        ref_cam,
                        rx + f32(dx) * sx,
                        ry + f32(dy) * sy,
                        depth,
                    );
                    let q = project(ncam, world);
                    if q.z <= 0.0 {
                        ok = false;
                        break;
                    }
                    let rv = sample_luma(ref_slot, rx + f32(dx) * sx, ry + f32(dy) * sy);
                    let nv = sample_luma(nslot, q.x, q.y);
                    nb_sum += nv;
                    nb_sum2 += nv * nv;
                    cross += rv * nv;
                }
                if !ok {
                    break;
                }
            }
            if !ok {
                continue;
            }
            let nb_mean = nb_sum / n_samples;
            let nb_var = max(nb_sum2 / n_samples - nb_mean * nb_mean, 0.0);
            if nb_var < 1e-6 {
                continue;
            }
            // NCC in [-1, 1]: gain- and offset-invariant.
            let cov = cross / n_samples - ref_mean * nb_mean;
            score_sum += cov / (ref_std * sqrt(nb_var));
            seen += 1u;
        }
        if seen == 0u {
            continue;
        }
        costs[h] = score_sum / f32(seen);
    }

    // Winner.
    var best = -2.0;
    var best_h = 0u;
    for (var h = 0u; h < n_h; h++) {
        if costs[h] > best {
            best = costs[h];
            best_h = h;
        }
    }
    if best <= -2.0 {
        return;
    }

    // Runner-up, measured OUTSIDE the winner's peak. A repetitive texture
    // produces a second, distant peak that scores nearly as well, and that
    // ambiguity is exactly what must be rejected; adjacent hypotheses are the
    // same peak and say nothing about uniqueness.
    var second = -2.0;
    for (var h = 0u; h < n_h; h++) {
        let far = max(h, best_h) - min(h, best_h) > PEAK_EXCLUDE;
        if far && costs[h] > second {
            second = costs[h];
        }
    }

    // Subpixel refinement in inverse-depth: parabola through the winner and
    // its two immediate neighbours on the cost curve.
    var offset = 0.0;
    if best_h > 0u && best_h + 1u < n_h {
        let a = costs[best_h - 1u];
        let c = costs[best_h + 1u];
        if a > -2.0 && c > -2.0 {
            let den = a - 2.0 * best + c;
            if abs(den) > 1e-8 {
                offset = clamp(0.5 * (a - c) / den, -1.0, 1.0);
            }
        }
    }
    let h_ref = clamp(f32(best_h) + offset, 0.0, denom);
    let inv_d = mix(pp.inv_depth_min, pp.inv_depth_max, h_ref / denom);

    out_depth[out_i] = 1.0 / max(inv_d, 1e-8);
    // Absolute correlation AND uniqueness: a good score at one depth only.
    out_conf[out_i] = clamp(best, 0.0, 1.0) * clamp(best - second, 0.0, 1.0);
}
