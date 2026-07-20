// Tile-based forward rasterization: one 16×16 workgroup per tile, one thread
// per pixel, entries streamed through shared memory in chunks of 128,
// front-to-back alpha compositing. Outputs everything the backward pass needs
// to replay the walk. (raster_common.wgsl is prepended by the host.)

@group(0) @binding(0) var<uniform> cam: RasterCamera;
@group(0) @binding(1) var<storage, read> surf_cam: array<SurfelCam>;
@group(0) @binding(2) var<storage, read> sorted_entries: array<u32>;
@group(0) @binding(3) var<storage, read> entry_items: array<u32>;
@group(0) @binding(4) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> out_color: array<vec4<f32>>;  // rgb + accumulated alpha
@group(0) @binding(6) var<storage, read_write> out_aux: array<vec4<f32>>;    // T_final, depth_acc, unused, unused
@group(0) @binding(7) var<storage, read_write> out_normal: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> out_ncontrib: array<u32>;     // entries consumed within tile range

const CHUNK: u32 = 128u;

var<workgroup> sh_surfels: array<SurfelCam, CHUNK>;

@compute @workgroup_size(16, 16)
fn rasterize_fwd(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) local_i: u32,
) {
    let tile_id = wg_id.y * cam.tiles_x + wg_id.x;
    let range = ranges[tile_id];
    let px_x = wg_id.x * TILE + lid.x;
    let px_y = wg_id.y * TILE + lid.y;
    let inside = px_x < cam.width && px_y < cam.height;

    let d = pixel_ray(cam, f32(px_x) + 0.5, f32(px_y) + 0.5);
    let pix_x = f32(px_x) + 0.5;
    let pix_y = f32(px_y) + 0.5;

    var transmittance = 1.0;
    var color = vec3<f32>(0.0);
    var alpha_acc = 0.0;
    var depth_acc = 0.0;
    var normal_acc = vec3<f32>(0.0);
    var n_contrib = 0u; // index (within range) just past the last contributor
    var done = !inside;

    var base = range.x;
    while base < range.y {
        let count = min(CHUNK, range.y - base);
        // Cooperative load: 256 threads, up to 128 entries.
        if local_i < count {
            sh_surfels[local_i] = surf_cam[entry_items[sorted_entries[base + local_i]]];
        }
        workgroupBarrier();

        for (var j = 0u; j < count; j++) {
            if done {
                continue;
            }
            let sc = sh_surfels[j];
            let hit = evaluate_hit(sc, d, pix_x, pix_y);
            let alpha = min(sc.color_op.w * hit.ghat, ALPHA_CLAMP);
            if alpha < ALPHA_SKIP {
                continue;
            }
            let w = transmittance * alpha;
            color += sc.color_op.rgb * w;
            alpha_acc += w;
            depth_acc += hit.t * w;
            normal_acc += sc.normal_fl.xyz * w;
            transmittance *= 1.0 - alpha;
            n_contrib = base - range.x + j + 1u;
            if transmittance < T_TERMINATE {
                done = true;
            }
        }
        workgroupBarrier();
        base += CHUNK;
    }

    if inside {
        let i = px_y * cam.width + px_x;
        out_color[i] = vec4<f32>(color, alpha_acc);
        out_aux[i] = vec4<f32>(transmittance, depth_acc, 0.0, 0.0);
        out_normal[i] = vec4<f32>(normal_acc, 0.0);
        out_ncontrib[i] = n_contrib;
    }
}
