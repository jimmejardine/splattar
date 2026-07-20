// Backward rasterization: reverse the forward walk per pixel, produce
// camera-space gradients per surfel (dc, dτu, dτv, dcolor, dopacity) plus the
// focal gradient. Per-tile shared-memory accumulation (atomic<u32> CAS float
// adds) flushed to global once per chunk — the accumulation pattern mandated
// by CLAUDE.md (WGSL has no f32 atomics).
// (raster_common.wgsl is prepended by the host.)

@group(0) @binding(0) var<uniform> cam: RasterCamera;
@group(0) @binding(1) var<storage, read> surf_cam: array<SurfelCam>;
@group(0) @binding(2) var<storage, read> sorted_entries: array<u32>;
@group(0) @binding(3) var<storage, read> entry_items: array<u32>;
@group(0) @binding(4) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> dl_dcolor: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> out_aux: array<vec4<f32>>;   // T_final in .x
@group(0) @binding(7) var<storage, read> out_ncontrib: array<u32>;
// Per-surfel camera-space grads: 13 floats [dc3, dtu3, dtv3, dcolor3, dop1].
@group(0) @binding(8) var<storage, read_write> grad_geom: array<atomic<u32>>;
// Camera accumulator: [dl_dR 9, dcam_center 3, dfocal 1, pad 3].
@group(0) @binding(9) var<storage, read_write> grad_cam: array<atomic<u32>>;

const CHUNK: u32 = 128u;
const GS: u32 = 13u; // grad stride

// Shared copies of the chunk's surfels (compact) + shared gradient slots.
var<workgroup> sh_c: array<vec4<f32>, CHUNK>;     // xyz c, w depth
var<workgroup> sh_tu: array<vec4<f32>, CHUNK>;    // xyz τu, w px
var<workgroup> sh_tv: array<vec4<f32>, CHUNK>;    // xyz τv, w py
var<workgroup> sh_col: array<vec4<f32>, CHUNK>;   // rgb color, w opacity
var<workgroup> sh_id: array<u32, CHUNK>;          // surfel index
var<workgroup> sh_grad: array<atomic<u32>, 1664>; // CHUNK * GS

fn shared_add(slot: u32, v: f32) {
    if v == 0.0 {
        return;
    }
    var old = atomicLoad(&sh_grad[slot]);
    loop {
        let newv = bitcast<u32>(bitcast<f32>(old) + v);
        let r = atomicCompareExchangeWeak(&sh_grad[slot], old, newv);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}

fn global_add(arr_index: u32, v: f32) {
    if v == 0.0 {
        return;
    }
    var old = atomicLoad(&grad_geom[arr_index]);
    loop {
        let newv = bitcast<u32>(bitcast<f32>(old) + v);
        let r = atomicCompareExchangeWeak(&grad_geom[arr_index], old, newv);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}

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

fn add_vec3(entry: u32, base: u32, v: vec3<f32>) {
    shared_add(entry * GS + base, v.x);
    shared_add(entry * GS + base + 1u, v.y);
    shared_add(entry * GS + base + 2u, v.z);
}

@compute @workgroup_size(16, 16)
fn rasterize_bwd(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) local_i: u32,
) {
    let tile_id = wg_id.y * cam.tiles_x + wg_id.x;
    let range = ranges[tile_id];
    if range.y <= range.x {
        return; // empty tile: uniform early-out, no barriers below are reached
    }
    let px_x = wg_id.x * TILE + lid.x;
    let px_y = wg_id.y * TILE + lid.y;
    let inside = px_x < cam.width && px_y < cam.height;
    let pix_i = px_y * cam.width + px_x;

    let d = pixel_ray(cam, f32(px_x) + 0.5, f32(px_y) + 0.5);
    let pix_x = f32(px_x) + 0.5;
    let pix_y = f32(px_y) + 0.5;

    var dl_dc_pix = vec3<f32>(0.0);
    var t_running = 1.0;
    var n_contrib = 0u;
    if inside {
        dl_dc_pix = dl_dcolor[pix_i].rgb;
        t_running = out_aux[pix_i].x;
        n_contrib = out_ncontrib[pix_i];
    }
    var suffix = vec3<f32>(0.0);
    var dfocal = 0.0;

    // Chunks from the back of the range toward the front.
    let total = range.y - range.x;
    let num_chunks = (total + CHUNK - 1u) / CHUNK;
    var chunk = num_chunks;
    while chunk > 0u {
        chunk -= 1u;
        let base = range.x + chunk * CHUNK;
        let count = min(CHUNK, range.y - base);

        if local_i < count {
            let sid = entry_items[sorted_entries[base + local_i]];
            let sc = surf_cam[sid];
            sh_c[local_i] = sc.c;
            sh_tu[local_i] = sc.tu;
            sh_tv[local_i] = sc.tv;
            sh_col[local_i] = sc.color_op;
            sh_id[local_i] = sid;
        }
        // Zero the shared grad slots (13 per entry).
        for (var s = local_i; s < count * GS; s += 256u) {
            atomicStore(&sh_grad[s], 0u);
        }
        workgroupBarrier();

        var j = count;
        while j > 0u {
            j -= 1u;
            let contrib_idx = base - range.x + j;
            if !inside || contrib_idx >= n_contrib
                || (dl_dc_pix.x == 0.0 && dl_dc_pix.y == 0.0 && dl_dc_pix.z == 0.0)
            {
                continue;
            }
            var sc: SurfelCam;
            sc.c = sh_c[j];
            sc.tu = sh_tu[j];
            sc.tv = sh_tv[j];
            sc.color_op = sh_col[j];
            let hit = evaluate_hit(sc, d, pix_x, pix_y);
            let raw_alpha = sc.color_op.w * hit.ghat;
            let alpha = min(raw_alpha, ALPHA_CLAMP);
            if alpha < ALPHA_SKIP {
                continue;
            }
            let t_before = t_running / (1.0 - alpha);
            let w = t_before * alpha;

            add_vec3(j, 9u, dl_dc_pix * w); // dcolor slots 9..11
            let dl_dalpha = dot(dl_dc_pix, sc.color_op.rgb) * t_before
                - dot(dl_dc_pix, suffix) / (1.0 - alpha);
            suffix += sc.color_op.rgb * w;
            t_running = t_before;

            if raw_alpha > ALPHA_CLAMP {
                continue;
            }
            shared_add(j * GS + 12u, dl_dalpha * hit.ghat); // dopacity
            let dl_dghat = dl_dalpha * sc.color_op.w;

            if hit.ray_branch {
                let s_vec = -d;
                let det = triple(sc.tu.xyz, sc.tv.xyz, s_vec);
                let u = hit.u;
                let v = hit.v;
                let dl_du = -u * hit.ghat * dl_dghat;
                let dl_dv = -v * hit.ghat * dl_dghat;
                let dl_dnu = dl_du / det;
                let dl_dnv = dl_dv / det;
                let dl_ddet = -(u * dl_du + v * dl_dv) / det;

                var dc = vec3<f32>(0.0);
                var dtu = vec3<f32>(0.0);
                var dtv = vec3<f32>(0.0);
                var ds = vec3<f32>(0.0);
                // Nu = det[−c, τv, S]: ∇a=τv×S, ∇b=S×(−c), ∇c=(−c)×τv
                dc -= cross(sc.tv.xyz, s_vec) * dl_dnu;
                dtv += cross(s_vec, -sc.c.xyz) * dl_dnu;
                ds += cross(-sc.c.xyz, sc.tv.xyz) * dl_dnu;
                // Nv = det[τu, −c, S]
                dtu += cross(-sc.c.xyz, s_vec) * dl_dnv;
                dc -= cross(s_vec, sc.tu.xyz) * dl_dnv;
                ds += cross(sc.tu.xyz, -sc.c.xyz) * dl_dnv;
                // D = det[τu, τv, S]
                dtu += cross(sc.tv.xyz, s_vec) * dl_ddet;
                dtv += cross(s_vec, sc.tu.xyz) * dl_ddet;
                ds += cross(sc.tu.xyz, sc.tv.xyz) * dl_ddet;

                add_vec3(j, 0u, dc);
                add_vec3(j, 3u, dtu);
                add_vec3(j, 6u, dtv);
                let dd = -ds;
                dfocal += dd.x * (-d.x / cam.focal) + dd.y * (-d.y / cam.focal);
            } else {
                let dx = pix_x - sc.tu.w;
                let dy = pix_y - sc.tv.w;
                let dl_ddx = -(dx / LOWPASS_SIGMA2) * hit.ghat * dl_dghat;
                let dl_ddy = -(dy / LOWPASS_SIGMA2) * hit.ghat * dl_dghat;
                let dpx = -dl_ddx;
                let dpy = -dl_ddy;
                let wz = sc.c.w; // depth = −c.z
                var dc = vec3<f32>(0.0);
                dc.x = dpx * cam.focal / wz;
                dc.y = dpy * (-cam.focal / wz);
                dc.z = dpx * cam.focal * sc.c.x / (wz * wz) + dpy * (-cam.focal * sc.c.y / (wz * wz));
                add_vec3(j, 0u, dc);
                dfocal += dpx * sc.c.x / wz + dpy * (-sc.c.y / wz);
            }
        }
        workgroupBarrier();

        // Flush this chunk's shared grads to the global per-surfel buffer.
        for (var s = local_i; s < count * GS; s += 256u) {
            let v = bitcast<f32>(atomicLoad(&sh_grad[s]));
            if v != 0.0 {
                let entry = s / GS;
                let comp = s % GS;
                global_add(sh_id[entry] * GS + comp, v);
            }
        }
        workgroupBarrier();
    }

    cam_add(12u, dfocal);
}
