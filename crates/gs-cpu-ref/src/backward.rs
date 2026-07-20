//! Analytic backward pass for the reference renderer. Color loss only (M2):
//! given dL/d(color image), produce gradients for every surfel parameter
//! (position, scales, quaternion, opacity, SH) and the camera (center,
//! quaternion, focal).
//!
//! Structure mirrors what the GPU will do: a per-pixel reverse compositing
//! walk yielding dL/dα and dL/dcolor per contribution, chained through the
//! ray–splat intersection (triple-product gradients) or the screen-space
//! low-pass branch into camera-space accumulators (dc, dτu, dτv), which are
//! then chained once per surfel into parameter space.

use glam::{DMat3, DVec3};

use crate::forward::{SurfelCam, evaluate, pixel_ray, prepare};
use crate::math::{self, quat_grad, quat_to_mat, triple_grads};
use crate::scene::*;

#[derive(Default, Clone)]
struct CamSpaceAcc {
    dc: DVec3,
    dtu: DVec3,
    dtv: DVec3,
    dcolor: DVec3,
    dopacity: f64,
}

struct Contribution {
    /// Index into the prepared (sorted) surfel list.
    list_idx: usize,
    alpha: f64,
    clamped_alpha: bool,
    t_before: f64,
    hit: crate::forward::Hit,
}

pub fn gradients(scene: &MicroScene, dl_dcolor: &[DVec3]) -> Gradients {
    let cam = &scene.camera;
    assert_eq!(dl_dcolor.len(), cam.width * cam.height);
    let surfels = prepare(scene);
    let mut grads = Gradients::zeros(scene);
    let mut acc = vec![CamSpaceAcc::default(); surfels.len()];
    let mut dfocal = 0.0;

    let mut contribs: Vec<Contribution> = Vec::with_capacity(64);
    for y in 0..cam.height {
        for x in 0..cam.width {
            let dl_dc_pix = dl_dcolor[y * cam.width + x];
            if dl_dc_pix == DVec3::ZERO {
                continue;
            }
            let d = pixel_ray(cam, x, y);
            let (pix_x, pix_y) = (x as f64 + 0.5, y as f64 + 0.5);

            // Forward replay to collect this pixel's contribution list.
            contribs.clear();
            let mut transmittance = 1.0;
            for (list_idx, sc) in surfels.iter().enumerate() {
                let hit = evaluate(sc, d, pix_x, pix_y);
                let raw_alpha = sc.opacity * hit.ghat;
                let alpha = raw_alpha.min(ALPHA_CLAMP);
                if alpha < ALPHA_SKIP {
                    continue;
                }
                contribs.push(Contribution {
                    list_idx,
                    alpha,
                    clamped_alpha: raw_alpha > ALPHA_CLAMP,
                    t_before: transmittance,
                    hit,
                });
                transmittance *= 1.0 - alpha;
                if transmittance < T_TERMINATE {
                    break;
                }
            }

            // Reverse walk: dL/dα_i = ⟨dL/dC, c_i·T_i⟩ − ⟨dL/dC, S_i⟩/(1−α_i)
            // with S_i the color already accumulated behind i.
            let mut suffix = DVec3::ZERO;
            for contrib in contribs.iter().rev() {
                let sc = &surfels[contrib.list_idx];
                let a = &mut acc[contrib.list_idx];
                let t_i = contrib.t_before;
                let alpha = contrib.alpha;

                a.dcolor += dl_dc_pix * (alpha * t_i);
                let dl_dalpha = dl_dc_pix.dot(sc.color) * t_i
                    - dl_dc_pix.dot(suffix) / (1.0 - alpha);
                suffix += sc.color * (alpha * t_i);
                if contrib.clamped_alpha {
                    continue; // clamp kills the α chain
                }

                a.dopacity += dl_dalpha * contrib.hit.ghat;
                let dl_dghat = dl_dalpha * sc.opacity;

                if contrib.hit.ray_branch {
                    chain_ray_branch(sc, d, &contrib.hit, dl_dghat, a, &mut dfocal, cam.focal);
                } else {
                    chain_screen_branch(
                        sc, pix_x, pix_y, dl_dghat, a, &mut dfocal, cam,
                    );
                }
            }
        }
    }

    // Camera-space accumulators → parameter gradients, once per surfel.
    let r_cam = quat_to_mat(cam.quat);
    let mut dl_dr_cam = DMat3::ZERO;
    for (sc, a) in surfels.iter().zip(&acc) {
        let s = &scene.surfels[sc.idx];
        let rs = quat_to_mat(s.quat);

        // c = R_camᵀ (p − C)
        let x = s.pos - cam.center;
        let dx = r_cam * a.dc;
        grads.pos[sc.idx] += dx;
        grads.cam_center -= dx;
        add_transpose_chain(&mut dl_dr_cam, x, a.dc);

        // τu = R_camᵀ (s_u · Rs·e0), τv analogous.
        for (axis, (dtau, scale)) in [(0usize, (a.dtu, s.scales[0])), (1, (a.dtv, s.scales[1]))]
        {
            let axis_world = rs.col(axis);
            let y_vec = axis_world * scale;
            let dy = r_cam * dtau;
            grads.scales[sc.idx][axis] += dy.dot(axis_world);
            let daxis_world = dy * scale;
            let mut dl_drs = DMat3::ZERO;
            *dl_drs.col_mut(axis) = daxis_world;
            let dq = quat_grad(s.quat, &dl_drs);
            for (slot, dqk) in grads.quat[sc.idx].iter_mut().zip(dq) {
                *slot += dqk;
            }
            add_transpose_chain(&mut dl_dr_cam, y_vec, dtau);
        }

        // Color → SH coefficients + view-direction path.
        let mut dlc = a.dcolor;
        for ch in 0..3 {
            if sc.clamped[ch] {
                dlc[ch] = 0.0;
            }
        }
        if dlc != DVec3::ZERO {
            let v = s.pos - cam.center;
            let len = v.length();
            let dir = v / len;
            let basis = math::sh_basis(scene.sh_degree, dir);
            let basis_grad = math::sh_basis_grad(scene.sh_degree, dir);
            let mut ddir = DVec3::ZERO;
            for k in 0..basis.len() {
                grads.sh[sc.idx][k] += dlc * basis[k];
                ddir += basis_grad[k] * s.sh[k].dot(dlc);
            }
            let dv = (ddir - dir * dir.dot(ddir)) / len;
            grads.pos[sc.idx] += dv;
            grads.cam_center -= dv;
        }

        grads.opacity[sc.idx] += a.dopacity;
    }
    grads.cam_quat = quat_grad(cam.quat, &dl_dr_cam);
    grads.focal = dfocal;
    grads
}

/// For a = Rᵀ y with dL/da known: dL/dR_{j,k} += y_j · (dL/da)_k.
fn add_transpose_chain(dl_dr: &mut DMat3, y: DVec3, dl_da: DVec3) {
    for k in 0..3 {
        let col = dl_dr.col(k) + y * dl_da[k];
        *dl_dr.col_mut(k) = col;
    }
}

/// Chain dL/dĝ through the ray–splat intersection: u = Nu/D, v = Nv/D with
/// Nu = det[−c, τv, −d], Nv = det[τu, −c, −d], D = det[τu, τv, −d].
fn chain_ray_branch(
    sc: &SurfelCam,
    d: DVec3,
    hit: &crate::forward::Hit,
    dl_dghat: f64,
    a: &mut CamSpaceAcc,
    dfocal: &mut f64,
    focal: f64,
) {
    let s_vec = -d;
    let det = crate::math::triple(sc.tu, sc.tv, s_vec);
    debug_assert!(det.abs() > 1e-12);
    let (u, v) = (hit.u, hit.v);
    let dl_du = -u * hit.ghat * dl_dghat;
    let dl_dv = -v * hit.ghat * dl_dghat;
    let dl_dnu = dl_du / det;
    let dl_dnv = dl_dv / det;
    let dl_ddet = -(u * dl_du + v * dl_dv) / det;

    let mut ds_vec = DVec3::ZERO;
    // Nu = det[−c, τv, S]
    let (ga, gb, gc) = triple_grads(-sc.c, sc.tv, s_vec);
    a.dc -= ga * dl_dnu;
    a.dtv += gb * dl_dnu;
    ds_vec += gc * dl_dnu;
    // Nv = det[τu, −c, S]
    let (ga, gb, gc) = triple_grads(sc.tu, -sc.c, s_vec);
    a.dtu += ga * dl_dnv;
    a.dc -= gb * dl_dnv;
    ds_vec += gc * dl_dnv;
    // D = det[τu, τv, S]
    let (ga, gb, gc) = triple_grads(sc.tu, sc.tv, s_vec);
    a.dtu += ga * dl_ddet;
    a.dtv += gb * dl_ddet;
    ds_vec += gc * dl_ddet;

    // S = −d; d = ((px−cx)/f, −(py−cy)/f, −1) ⇒ ∂d/∂f = (−d.x/f, −d.y/f, 0).
    let dd = -ds_vec;
    *dfocal += dd.x * (-d.x / focal) + dd.y * (-d.y / focal);
}

/// Chain dL/dĝ through the screen-space low-pass gaussian around the
/// projected center.
fn chain_screen_branch(
    sc: &SurfelCam,
    pix_x: f64,
    pix_y: f64,
    dl_dghat: f64,
    a: &mut CamSpaceAcc,
    dfocal: &mut f64,
    cam: &RefCamera,
) {
    let dx = pix_x - sc.px;
    let dy = pix_y - sc.py;
    let g = (-(dx * dx + dy * dy) / (2.0 * LOWPASS_SIGMA2)).exp();
    let dl_ddx = -(dx / LOWPASS_SIGMA2) * g * dl_dghat;
    let dl_ddy = -(dy / LOWPASS_SIGMA2) * g * dl_dghat;
    let dpx = -dl_ddx;
    let dpy = -dl_ddy;

    // px = cx + f·c.x/w, py = cy − f·c.y/w, w = −c.z.
    let f = cam.focal;
    let w = -sc.c.z;
    a.dc.x += dpx * f / w;
    a.dc.z += dpx * f * sc.c.x / (w * w);
    a.dc.y += dpy * (-f / w);
    a.dc.z += dpy * (-f * sc.c.y / (w * w));
    *dfocal += dpx * sc.c.x / w + dpy * (-sc.c.y / w);
}
