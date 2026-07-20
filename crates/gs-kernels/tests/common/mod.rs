//! Shared micro-scene generation for GPU↔CPU parity tests: build the scene in
//! f64 (gs-cpu-ref types), convert to f32 for the GPU rasterizer.

use glam::DVec3;
use gs_cpu_ref::{MicroScene, RefCamera, Surfel};

pub fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

pub fn uniform(rng: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * xorshift(rng)
}

pub fn make_scene(seed: u64, n_surfels: usize, sh_degree: u8, size: usize) -> MicroScene {
    let mut rng = seed;
    let camera = RefCamera {
        center: DVec3::new(
            uniform(&mut rng, -0.3, 0.3),
            uniform(&mut rng, -0.3, 0.3),
            uniform(&mut rng, -0.3, 0.3),
        ),
        quat: [
            uniform(&mut rng, -0.08, 0.08),
            uniform(&mut rng, -0.08, 0.08),
            uniform(&mut rng, -0.08, 0.08),
            1.0,
        ],
        focal: size as f64 * 0.9,
        width: size,
        height: size,
    };
    let r = gs_cpu_ref::math::quat_to_mat(camera.quat);

    let n_coeffs = gs_cpu_ref::math::num_coeffs(sh_degree);
    let surfels = (0..n_surfels)
        .map(|_| {
            let z = -uniform(&mut rng, 2.0, 6.0);
            let extent = -z * (size as f64 * 0.5) / camera.focal * 0.7;
            let p_cam = DVec3::new(
                uniform(&mut rng, -extent, extent),
                uniform(&mut rng, -extent, extent),
                z,
            );
            let mut sh = vec![DVec3::new(
                uniform(&mut rng, -0.8, 0.8),
                uniform(&mut rng, -0.8, 0.8),
                uniform(&mut rng, -0.8, 0.8),
            )];
            for _ in 1..n_coeffs {
                sh.push(DVec3::new(
                    uniform(&mut rng, -0.25, 0.25),
                    uniform(&mut rng, -0.25, 0.25),
                    uniform(&mut rng, -0.25, 0.25),
                ));
            }
            Surfel {
                pos: camera.center + r * p_cam,
                scales: [uniform(&mut rng, 0.1, 0.5), uniform(&mut rng, 0.1, 0.5)],
                quat: [
                    uniform(&mut rng, -1.0, 1.0),
                    uniform(&mut rng, -1.0, 1.0),
                    uniform(&mut rng, -1.0, 1.0),
                    uniform(&mut rng, -1.0, 1.0) + 1.5,
                ],
                opacity: uniform(&mut rng, 0.25, 0.85),
                sh,
            }
        })
        .collect();

    MicroScene {
        surfels,
        camera,
        sh_degree,
    }
}

/// f32 conversion for the GPU rasterizer.
pub struct GpuSceneData {
    pub positions: Vec<glam::Vec3>,
    pub scales: Vec<[f32; 2]>,
    pub quats: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh: Vec<f32>,
    pub sh_coeffs: usize,
    pub camera: gs_kernels::RasterCamera,
}

pub fn to_gpu(scene: &MicroScene) -> GpuSceneData {
    let n_coeffs = gs_cpu_ref::math::num_coeffs(scene.sh_degree);
    let mut sh = Vec::with_capacity(scene.surfels.len() * n_coeffs * 3);
    for s in &scene.surfels {
        for c in &s.sh {
            sh.extend([c.x as f32, c.y as f32, c.z as f32]);
        }
    }
    let q = scene.camera.quat;
    GpuSceneData {
        positions: scene
            .surfels
            .iter()
            .map(|s| glam::Vec3::new(s.pos.x as f32, s.pos.y as f32, s.pos.z as f32))
            .collect(),
        scales: scene
            .surfels
            .iter()
            .map(|s| [s.scales[0] as f32, s.scales[1] as f32])
            .collect(),
        quats: scene
            .surfels
            .iter()
            .map(|s| [s.quat[0] as f32, s.quat[1] as f32, s.quat[2] as f32, s.quat[3] as f32])
            .collect(),
        opacities: scene.surfels.iter().map(|s| s.opacity as f32).collect(),
        sh,
        sh_coeffs: n_coeffs,
        camera: gs_kernels::RasterCamera {
            center: glam::Vec3::new(
                scene.camera.center.x as f32,
                scene.camera.center.y as f32,
                scene.camera.center.z as f32,
            ),
            quat: glam::Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32),
            focal: scene.camera.focal as f32,
            sh_degree: scene.sh_degree as u32,
        },
    }
}
