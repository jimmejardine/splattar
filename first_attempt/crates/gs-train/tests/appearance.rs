//! GPU appearance kernels vs host reference: the target transform must match
//! the old host inverse-correction bit-for-bit-ish, and the GPU-reduced fit
//! statistics must reproduce the host least-squares fit.

use glam::Vec3;
use gs_kernels::RasterCamera;
use gs_train::TrainView;
use gs_train::appearance::Appearance;
use gs_wgpu::{GpuContext, buffers};

const W: u32 = 63; // deliberately not a multiple of the fit stride
const H: u32 = 41;

fn ctx() -> GpuContext {
    pollster::block_on(GpuContext::new(gs_wgpu::backends_from_str(None).unwrap())).unwrap()
}

fn rand_img(seed: u64) -> Vec<[f32; 4]> {
    let mut x = seed | 1;
    (0..(W * H) as usize)
        .map(|_| {
            let mut n = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32
            };
            [n(), n(), n(), 1.0]
        })
        .collect()
}

fn dummy_view(target: Vec<[f32; 4]>) -> TrainView {
    TrainView {
        camera: RasterCamera {
            center: Vec3::ZERO,
            quat: glam::Quat::IDENTITY,
            focal: 50.0,
            sh_degree: 0,
        },
        target,
    }
}

/// The old host fit this replaced (trainer::fit_affine), as reference.
fn fit_affine_host(render: &[[f32; 4]], target: &[[f32; 4]]) -> Option<[f32; 6]> {
    let mut out = [1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0];
    for ch in 0..3 {
        let (mut sr, mut st, mut srr, mut srt, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in (0..render.len()).step_by(4) {
            let r = render[i][ch] as f64;
            let t = target[i][ch] as f64;
            sr += r;
            st += t;
            srr += r * r;
            srt += r * t;
            n += 1.0;
        }
        let var = srr - sr * sr / n;
        if var < 1e-9 || !var.is_finite() {
            return None;
        }
        let g = (srt - sr * st / n) / var;
        let b = (st - g * sr) / n;
        out[ch] = g as f32;
        out[3 + ch] = b as f32;
    }
    Some(out)
}

#[test]
fn gpu_transform_and_fit_match_host() {
    let ctx = ctx();
    let px = (W * H) as u64;
    let targets = [rand_img(0xa11ce), rand_img(0xb0b)];
    let render_img = rand_img(0xc0ffee);

    let target_buf = buffers::storage_empty(&ctx.device, "test-target", px * 16);
    let render_buf = buffers::storage_init(
        &ctx.device,
        "test-render",
        bytemuck::cast_slice(&render_img),
    );
    let views: Vec<TrainView> = targets.iter().cloned().map(dummy_view).collect();
    let mut app = Appearance::new(&ctx, W, H, &views, &target_buf, &render_buf);

    // --- target_transform: GPU inverse-correction matches the host formula.
    let affine = [1.2f32, 0.9, 1.05, 0.05, -0.02, 0.1];
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    app.encode_transform(&ctx, &mut encoder, 1, &affine, None);
    ctx.queue.submit([encoder.finish()]);
    let got: Vec<[f32; 4]> =
        bytemuck::cast_slice(&buffers::readback(&ctx.device, &ctx.queue, &target_buf)).to_vec();
    for (i, (g, t)) in got.iter().zip(&targets[1]).enumerate() {
        for ch in 0..3 {
            let want = ((t[ch] - affine[3 + ch]) / affine[ch]).clamp(0.0, 4.0);
            assert!(
                (g[ch] - want).abs() < 1e-6,
                "transform mismatch at px {i} ch {ch}: {} vs {want}",
                g[ch]
            );
        }
        assert_eq!(g[3], t[3]);
    }

    // --- fit_reduce: async GPU fit matches the host least-squares fit.
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    app.encode_fit(&mut encoder, 0, None);
    ctx.queue.submit([encoder.finish()]);
    app.map_pending();
    let mut fits = Vec::new();
    for _ in 0..1000 {
        fits = app.poll_fits(&ctx.device);
        if !fits.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(fits.len(), 1, "fit never completed");
    let (view, got_fit) = fits[0];
    assert_eq!(view, 0);
    let want_fit = fit_affine_host(&render_img, &targets[0]).unwrap();
    for k in 0..6 {
        assert!(
            (got_fit[k] - want_fit[k]).abs() < 1e-3,
            "fit mismatch at {k}: {} vs {}",
            got_fit[k],
            want_fit[k]
        );
    }
}
