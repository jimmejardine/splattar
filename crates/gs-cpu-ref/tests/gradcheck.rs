//! The M2 oracle gate: central finite differences vs the analytic backward,
//! per parameter class, on randomized micro-scenes. The acceptance bar is
//! ≤1e-2 relative; in f64 the real agreement should be orders tighter, so we
//! assert 1e-3 with an absolute floor (rare α-threshold crossings excepted by
//! seed choice — these are deterministic scenes).

use glam::DVec3;
use gs_cpu_ref::{MicroScene, RefCamera, Surfel, gradients, render};

fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / (1u64 << 53) as f64 // [0,1)
}

fn uniform(rng: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * xorshift(rng)
}

fn make_scene(seed: u64, n_surfels: usize, sh_degree: u8, size: usize) -> MicroScene {
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
        focal: 30.0,
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
                scales: [uniform(&mut rng, 0.1, 0.4), uniform(&mut rng, 0.1, 0.4)],
                quat: [
                    uniform(&mut rng, -1.0, 1.0),
                    uniform(&mut rng, -1.0, 1.0),
                    uniform(&mut rng, -1.0, 1.0),
                    uniform(&mut rng, -1.0, 1.0) + 1.5, // keep away from zero norm
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

fn make_weights(seed: u64, n: usize) -> Vec<DVec3> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            DVec3::new(
                uniform(&mut rng, -1.0, 1.0),
                uniform(&mut rng, -1.0, 1.0),
                uniform(&mut rng, -1.0, 1.0),
            )
        })
        .collect()
}

fn loss(scene: &MicroScene, weights: &[DVec3]) -> f64 {
    render(scene)
        .color
        .iter()
        .zip(weights)
        .map(|(c, w)| c.dot(*w))
        .sum()
}

struct Checker {
    h: f64,
    worst: f64,
    worst_label: String,
}

impl Checker {
    fn new() -> Self {
        Self {
            h: 1e-6,
            worst: 0.0,
            worst_label: String::new(),
        }
    }

    fn check(&mut self, label: &str, analytic: f64, fd: f64) {
        let denom = analytic.abs().max(fd.abs());
        let err = (analytic - fd).abs();
        let rel = if denom > 1e-7 { err / denom } else { 0.0 };
        if err > 1e-7 && rel > self.worst {
            self.worst = rel;
            self.worst_label = label.to_string();
        }
        assert!(
            err <= 1e-7 + 1e-3 * denom,
            "{label}: analytic {analytic:.9e} vs fd {fd:.9e} (rel {rel:.2e})"
        );
    }
}

fn grad_check(seed: u64, n_surfels: usize, sh_degree: u8, size: usize) {
    let scene = make_scene(seed, n_surfels, sh_degree, size);
    let weights = make_weights(seed ^ 0xabcdef, size * size);
    let grads = gradients(&scene, &weights);
    let mut ck = Checker::new();
    let h = ck.h;

    let fd = |s: &MicroScene| loss(s, &weights);

    for i in 0..n_surfels {
        for dim in 0..3 {
            let mut sp = scene.clone();
            let mut sm = scene.clone();
            sp.surfels[i].pos[dim] += h;
            sm.surfels[i].pos[dim] -= h;
            ck.check(&format!("pos[{i}][{dim}]"), grads.pos[i][dim], (fd(&sp) - fd(&sm)) / (2.0 * h));
        }
        for dim in 0..2 {
            let mut sp = scene.clone();
            let mut sm = scene.clone();
            sp.surfels[i].scales[dim] += h;
            sm.surfels[i].scales[dim] -= h;
            ck.check(&format!("scale[{i}][{dim}]"), grads.scales[i][dim], (fd(&sp) - fd(&sm)) / (2.0 * h));
        }
        for dim in 0..4 {
            let mut sp = scene.clone();
            let mut sm = scene.clone();
            sp.surfels[i].quat[dim] += h;
            sm.surfels[i].quat[dim] -= h;
            ck.check(&format!("quat[{i}][{dim}]"), grads.quat[i][dim], (fd(&sp) - fd(&sm)) / (2.0 * h));
        }
        {
            let mut sp = scene.clone();
            let mut sm = scene.clone();
            sp.surfels[i].opacity += h;
            sm.surfels[i].opacity -= h;
            ck.check(&format!("opacity[{i}]"), grads.opacity[i], (fd(&sp) - fd(&sm)) / (2.0 * h));
        }
        for k in 0..scene.surfels[i].sh.len() {
            for ch in 0..3 {
                let mut sp = scene.clone();
                let mut sm = scene.clone();
                sp.surfels[i].sh[k][ch] += h;
                sm.surfels[i].sh[k][ch] -= h;
                ck.check(&format!("sh[{i}][{k}][{ch}]"), grads.sh[i][k][ch], (fd(&sp) - fd(&sm)) / (2.0 * h));
            }
        }
    }
    for dim in 0..3 {
        let mut sp = scene.clone();
        let mut sm = scene.clone();
        sp.camera.center[dim] += h;
        sm.camera.center[dim] -= h;
        ck.check(&format!("cam_center[{dim}]"), grads.cam_center[dim], (fd(&sp) - fd(&sm)) / (2.0 * h));
    }
    for dim in 0..4 {
        let mut sp = scene.clone();
        let mut sm = scene.clone();
        sp.camera.quat[dim] += h;
        sm.camera.quat[dim] -= h;
        ck.check(&format!("cam_quat[{dim}]"), grads.cam_quat[dim], (fd(&sp) - fd(&sm)) / (2.0 * h));
    }
    {
        let mut sp = scene.clone();
        let mut sm = scene.clone();
        sp.camera.focal += h;
        sm.camera.focal -= h;
        ck.check("focal", grads.focal, (fd(&sp) - fd(&sm)) / (2.0 * h));
    }
    eprintln!(
        "gradcheck seed={seed} n={n_surfels} deg={sh_degree}: worst rel {:.2e} at {}",
        ck.worst, ck.worst_label
    );
}

#[test]
fn gradcheck_deg0_scenes() {
    grad_check(11, 12, 0, 24);
    grad_check(23, 12, 0, 24);
}

#[test]
fn gradcheck_deg1_scene() {
    grad_check(37, 10, 1, 24);
}

#[test]
fn gradcheck_deg3_scene() {
    grad_check(53, 6, 3, 20);
}

#[test]
fn forward_renders_something() {
    let scene = make_scene(99, 10, 1, 32);
    let out = render(&scene);
    let energy: f64 = out.alpha.iter().sum();
    assert!(energy > 1.0, "scene should cover pixels (alpha sum {energy})");
}
