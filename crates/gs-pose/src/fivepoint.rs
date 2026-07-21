//! Nistér/Stewénius five-point essential-matrix solver.
//!
//! Why it exists (measured, see RESULTS.md): pairwise cross-take matching
//! runs at ~20% inlier fraction, where EIGHT-point RANSAC needs 8 clean
//! draws per sample (0.2⁸ ≈ 3e-6) — hopeless at any iteration budget. The
//! five-point minimal sample turns the same data into a routine solve
//! (0.2⁵ ≈ 3e-4). It is also the correct minimal solver for planar scenes,
//! where the eight-point linear system degenerates (documented limitation
//! in `twoview.rs`).
//!
//! Method (Stewénius variant): the 4-dim null space of the 5×9 epipolar
//! system parametrizes E = xX + yY + zZ + W. det(E) = 0 and the trace
//! constraint 2EEᵀE − tr(EEᵀ)E = 0 yield 10 cubics in (x, y, z). Gauss-
//! Jordan over the 10 cubic monomials leaves a 10×10 remainder, from which
//! the action matrix of multiplication-by-x on the degree-≤2 quotient basis
//! is assembled. Its real eigenpairs are the solutions; eigenvectors are
//! recovered per real eigenvalue as SVD null vectors of (A − λI), because
//! nalgebra exposes only eigenvalues for general real matrices.

use nalgebra::{DMatrix, Matrix3};

use crate::twoview::{Match2, RansacResult, Rng64, eight_point, sampson_sq};

// Monomial orderings.
// Linear:  [x, y, z, 1]
// Quad:    [x², xy, xz, y², yz, z², x, y, z, 1]
// Cubic:   [x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³ | quad basis ↑]
type Lin = [f64; 4];
type Quad = [f64; 10];
type Cubic = [f64; 20];

/// lin·lin → quad index map (symmetric).
const LL: [[usize; 4]; 4] = [
    [0, 1, 2, 6],
    [1, 3, 4, 7],
    [2, 4, 5, 8],
    [6, 7, 8, 9],
];

/// quad·lin → cubic index map.
const QL: [[usize; 4]; 10] = [
    [0, 1, 2, 10],   // x²·{x,y,z,1}
    [1, 3, 4, 11],   // xy
    [2, 4, 5, 12],   // xz
    [3, 6, 7, 13],   // y²
    [4, 7, 8, 14],   // yz
    [5, 8, 9, 15],   // z²
    [10, 11, 12, 16], // x
    [11, 13, 14, 17], // y
    [12, 14, 15, 18], // z
    [16, 17, 18, 19], // 1
];

fn mul_ll(a: &Lin, b: &Lin) -> Quad {
    let mut out = [0.0; 10];
    for i in 0..4 {
        if a[i] == 0.0 {
            continue;
        }
        for j in 0..4 {
            out[LL[i][j]] += a[i] * b[j];
        }
    }
    out
}

fn mul_ql(a: &Quad, b: &Lin) -> Cubic {
    let mut out = [0.0; 20];
    for i in 0..10 {
        if a[i] == 0.0 {
            continue;
        }
        for j in 0..4 {
            out[QL[i][j]] += a[i] * b[j];
        }
    }
    out
}

fn quad_add(a: &mut Quad, b: &Quad, s: f64) {
    for i in 0..10 {
        a[i] += s * b[i];
    }
}

fn cubic_add(a: &mut Cubic, b: &Cubic, s: f64) {
    for i in 0..20 {
        a[i] += s * b[i];
    }
}

/// Diagnostic hook for tests: returns (constraint rows, gt residuals when a
/// probe E is supplied projected onto the null basis).
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn diagnose(pts: &[Match2], probe: &Matrix3<f64>) -> (Vec<f64>, [f64; 3], f64) {
    let (rows, basis) = build_system(pts).expect("system");
    // Project the probe onto the orthonormal basis.
    let flat = |m: &Matrix3<f64>| -> [f64; 9] {
        let mut v = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                v[i * 3 + j] = m[(i, j)];
            }
        }
        v
    };
    let pv = flat(probe);
    let mut coef = [0.0f64; 4];
    for k in 0..4 {
        let b = flat(&basis[k]);
        coef[k] = (0..9).map(|i| pv[i] * b[i]).sum();
    }
    let (x, y, z) = (coef[0] / coef[3], coef[1] / coef[3], coef[2] / coef[3]);
    let mono = |c: &Cubic| -> f64 {
        let m = [
            x * x * x,
            x * x * y,
            x * x * z,
            x * y * y,
            x * y * z,
            x * z * z,
            y * y * y,
            y * y * z,
            y * z * z,
            z * z * z,
            x * x,
            x * y,
            x * z,
            y * y,
            y * z,
            z * z,
            x,
            y,
            z,
            1.0,
        ];
        (0..20).map(|i| c[i] * m[i]).sum()
    };
    let resid: Vec<f64> = rows.iter().map(mono).collect();
    // Reconstruction error of the probe from the projection.
    let mut rec = basis[3] * coef[3];
    rec += basis[0] * coef[0] + basis[1] * coef[1] + basis[2] * coef[2];
    let rec_err = (rec - probe).norm();
    (resid, [x, y, z], rec_err)
}

/// Build the 10 cubic constraint rows + null basis (X, Y, Z, W).
#[allow(clippy::needless_range_loop)] // cross-indexed tensor math
fn build_system(pts: &[Match2]) -> Option<(Vec<Cubic>, [Matrix3<f64>; 4])> {
    // Null space of the epipolar system (rows: x'ᵀ E x = 0).
    let n = pts.len();
    let mut q = DMatrix::<f64>::zeros(n, 9);
    for (r, m) in pts.iter().enumerate() {
        let (x1, y1) = m.a;
        let (x2, y2) = m.b;
        // E row-major: e00..e22; constraint Σ e_ij · x2_i · x1_j with
        // homogeneous coords (x, y, 1).
        let row = [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            1.0,
        ];
        for c in 0..9 {
            q[(r, c)] = row[c];
        }
    }
    // Null space via the symmetric eigendecomposition of QᵀQ (nalgebra's
    // SVD is thin — a 5×9 system yields only 5 right singular vectors, so
    // the 4-dim null space must come from the 9×9 Gram matrix instead).
    let gram = q.transpose() * &q;
    let se = gram.symmetric_eigen();
    let mut order: Vec<usize> = (0..9).collect();
    order.sort_by(|&i, &j| se.eigenvalues[i].total_cmp(&se.eigenvalues[j]));
    let basis: [Matrix3<f64>; 4] = std::array::from_fn(|k| {
        let col = se.eigenvectors.column(order[k]);
        Matrix3::from_row_slice(col.as_slice())
    });
    // E = xX + yY + zZ + W (any basis assignment works — the solver finds
    // the right combination).
    let (xm, ym, zm, wm) = (&basis[0], &basis[1], &basis[2], &basis[3]);

    // E entries as linear forms in (x, y, z, 1).
    let lin = |i: usize, j: usize| -> Lin { [xm[(i, j)], ym[(i, j)], zm[(i, j)], wm[(i, j)]] };

    // det(E) = 0 (one cubic).
    let det2 = |a: Lin, b: Lin, c: Lin, d: Lin| -> Quad {
        // a*d − b*c
        let mut q = mul_ll(&a, &d);
        let bc = mul_ll(&b, &c);
        quad_add(&mut q, &bc, -1.0);
        q
    };
    let m00 = det2(lin(1, 1), lin(1, 2), lin(2, 1), lin(2, 2));
    let m01 = det2(lin(1, 0), lin(1, 2), lin(2, 0), lin(2, 2));
    let m02 = det2(lin(1, 0), lin(1, 1), lin(2, 0), lin(2, 1));
    let mut det = mul_ql(&m00, &lin(0, 0));
    let t01 = mul_ql(&m01, &lin(0, 1));
    let t02 = mul_ql(&m02, &lin(0, 2));
    cubic_add(&mut det, &t01, -1.0);
    cubic_add(&mut det, &t02, 1.0);

    // Trace constraint: 2·E·Eᵀ·E − tr(E·Eᵀ)·E = 0 (nine cubics).
    // EEt[i][j] = Σ_k E[i][k]·E[j][k]  (quadratic).
    let mut eet = [[[0.0f64; 10]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = [0.0; 10];
            for k in 0..3 {
                let p = mul_ll(&lin(i, k), &lin(j, k));
                quad_add(&mut acc, &p, 1.0);
            }
            eet[i][j] = acc;
        }
    }
    let mut trace = [0.0f64; 10];
    for i in 0..3 {
        quad_add(&mut trace, &eet[i][i], 1.0);
    }
    let mut rows: Vec<Cubic> = Vec::with_capacity(10);
    rows.push(det);
    for i in 0..3 {
        for j in 0..3 {
            // 2 Σ_k EEt[i][k]·E[k][j]  −  trace·E[i][j]
            let mut c = [0.0f64; 20];
            for k in 0..3 {
                let p = mul_ql(&eet[i][k], &lin(k, j));
                cubic_add(&mut c, &p, 2.0);
            }
            let tp = mul_ql(&trace, &lin(i, j));
            cubic_add(&mut c, &tp, -1.0);
            rows.push(c);
        }
    }
    Some((rows, [*xm, *ym, *zm, *wm]))
}

/// Solve for up to 10 essential matrices from exactly 5 correspondences.
pub fn five_point(pts: &[Match2]) -> Vec<Matrix3<f64>> {
    let Some((rows, basis)) = build_system(pts) else {
        return Vec::new();
    };
    let (xm, ym, zm, wm) = (&basis[0], &basis[1], &basis[2], &basis[3]);

    // Gauss-Jordan over the 10 cubic-monomial columns (0..10).
    let mut a = DMatrix::<f64>::zeros(10, 20);
    for (r, row) in rows.iter().enumerate() {
        for c in 0..20 {
            a[(r, c)] = row[c];
        }
    }
    for col in 0..10 {
        // Partial pivot.
        let mut piv = col;
        for r in col + 1..10 {
            if a[(r, col)].abs() > a[(piv, col)].abs() {
                piv = r;
            }
        }
        if a[(piv, col)].abs() < 1e-10 {
            return Vec::new(); // degenerate sample
        }
        if piv != col {
            a.swap_rows(piv, col);
        }
        let d = a[(col, col)];
        for c in col..20 {
            a[(col, c)] /= d;
        }
        for r in 0..10 {
            if r != col && a[(r, col)].abs() > 0.0 {
                let f = a[(r, col)];
                for c in col..20 {
                    a[(r, c)] -= f * a[(col, c)];
                }
            }
        }
    }
    // Remainder R: cubic monomial M3[i] = −Σ_j R[i][j]·B[j].
    // Action matrix of multiplication by x on B = [x²,xy,xz,y²,yz,z²,x,y,z,1].
    // Row i encodes x·B[i] in the basis: (A·m)(p) = x(p)·m(p) for the
    // monomial vector m(p), so solutions are RIGHT eigenpairs of A.
    let mut act = DMatrix::<f64>::zeros(10, 10);
    for (brow, m3row) in [(0usize, 0usize), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)] {
        // x·B[brow] = M3[m3row] = −R[m3row][:]
        for j in 0..10 {
            act[(brow, j)] = -a[(m3row, 10 + j)];
        }
    }
    act[(6, 0)] = 1.0; // x·x = x²
    act[(7, 1)] = 1.0; // x·y = xy
    act[(8, 2)] = 1.0; // x·z = xz
    act[(9, 6)] = 1.0; // x·1 = x

    // Real eigenvalues → eigenvectors via null spaces of (A − λI).
    let eigs = act.clone().complex_eigenvalues();
    let mut out = Vec::new();
    for e in eigs.iter() {
        if e.im.abs() > 1e-8 * (1.0 + e.re.abs()) {
            continue;
        }
        let mut shifted = act.clone();
        for d in 0..10 {
            shifted[(d, d)] -= e.re;
        }
        let svd = shifted.svd(false, true);
        let Some(vt) = svd.v_t else { continue };
        let v = vt.row(9);
        if v[9].abs() < 1e-12 {
            continue; // solution at infinity
        }
        let (x, y, z) = (v[6] / v[9], v[7] / v[9], v[8] / v[9]);
        let em = xm * x + ym * y + zm * z + wm;
        let norm = em.norm();
        if norm > 1e-12 && em.iter().all(|c| c.is_finite()) {
            out.push(em / norm);
        }
    }
    out
}

/// Five-point RANSAC: minimal samples of 5 (up to 10 candidate E each),
/// Sampson-scored, refit on the consensus with the linear solver. `thresh`
/// is the Sampson distance in normalized coordinates.
pub fn ransac_essential_5pt(
    matches: &[Match2],
    iters: usize,
    thresh: f64,
    seed: u64,
) -> Option<RansacResult> {
    if matches.len() < 5 {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Rng64::new(seed);
    let mut best: Option<(usize, Matrix3<f64>)> = None;
    for _ in 0..iters {
        let mut idx = [0usize; 5];
        let mut k = 0;
        while k < 5 {
            let cand = rng.below(matches.len());
            if !idx[..k].contains(&cand) {
                idx[k] = cand;
                k += 1;
            }
        }
        let sample: Vec<Match2> = idx.iter().map(|&i| matches[i]).collect();
        for e in five_point(&sample) {
            let count = matches.iter().filter(|m| sampson_sq(&e, m) < t2).count();
            if best.is_none_or(|(bc, _)| count > bc) {
                best = Some((count, e));
            }
        }
    }
    let (best_count, e0) = best?;
    if best_count < 5 {
        return None;
    }
    // Refit on the consensus set (linear least squares is fine once the
    // set is clean), then reclassify once.
    let inl: Vec<Match2> = matches
        .iter()
        .filter(|m| sampson_sq(&e0, m) < t2)
        .copied()
        .collect();
    // Linear refit — but keep it only if it doesn't LOSE consensus: on
    // planar scenes the eight-point least squares is itself degenerate and
    // would destroy a perfectly good minimal solution.
    let mut e1 = e0;
    if inl.len() >= 8
        && let Some(er) = eight_point(&inl)
    {
        let refit_count = matches.iter().filter(|m| sampson_sq(&er, m) < t2).count();
        if refit_count >= best_count {
            e1 = er;
        }
    }
    let inliers: Vec<usize> = matches
        .iter()
        .enumerate()
        .filter(|(_, m)| sampson_sq(&e1, m) < t2)
        .map(|(i, _)| i)
        .collect();
    if inliers.len() < 5 {
        return None;
    }
    Some(RansacResult { e: e1, inliers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::se3::Se3;
    use crate::twoview::recover_pose;
    use nalgebra::{Vector3, Vector6};

    fn scene(n: usize, seed: u64, planar: bool) -> (Vec<Match2>, Se3) {
        let mut rng = Rng64::new(seed);
        let mut r = |s: f64| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) * s;
        let pose = Se3::exp(&Vector6::new(0.4, 0.06, 0.12, 0.03, -0.05, 0.02));
        let mut matches = Vec::with_capacity(n);
        for _ in 0..n {
            let p = if planar {
                // All points on one plane — the eight-point killer.
                let (u, v) = (r(2.0), r(1.5));
                Vector3::new(u, v, 4.0 + 0.3 * u + 0.2 * v)
            } else {
                Vector3::new(r(2.0), r(1.5), 3.0 + r(1.2))
            };
            let a = p;
            let b = pose.act(&p);
            matches.push(Match2 {
                a: (a[0] / a[2], a[1] / a[2]),
                b: (b[0] / b[2], b[1] / b[2]),
            });
        }
        (matches, pose)
    }

    /// Ground-truth E from (R, t): E = [t]× R  (for x2ᵀ E x1 = 0 with
    /// x2 = R x1 + t).
    fn gt_e(pose: &Se3) -> Matrix3<f64> {
        let r = pose.r.to_rotation_matrix();
        let t = pose.t;
        let tx = crate::se3::skew(&t);
        let e = tx * r.matrix();
        e / e.norm()
    }

    #[test]
    fn constraint_rows_vanish_at_ground_truth() {
        let (matches, pose) = scene(5, 3, false);
        let egt = gt_e(&pose);
        let (resid, xyz, rec_err) = diagnose(&matches, &egt);
        eprintln!("gt (x,y,z) = {xyz:?}, reconstruction err {rec_err:.2e}");
        let worst = resid.iter().fold(0.0f64, |a, r| a.max(r.abs()));
        eprintln!("worst constraint residual at GT: {worst:.2e}");
        assert!(rec_err < 1e-9, "GT not in null space: {rec_err}");
        assert!(worst < 1e-9, "constraints don't vanish at GT: {worst}");
    }

    #[test]
    fn recovers_exact_essential_from_five_points() {
        for seed in [3u64, 17, 99] {
            let (matches, pose) = scene(8, seed, false);
            let egt = gt_e(&pose);
            let cands = five_point(&matches[..5]);
            assert!(!cands.is_empty(), "seed {seed}: no candidates");
            // The right candidate must fit the three HELD-OUT points too.
            let best = cands
                .iter()
                .map(|e| {
                    matches[5..]
                        .iter()
                        .map(|m| sampson_sq(e, m))
                        .fold(0.0, f64::max)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-14, "seed {seed}: held-out residual {best}");
            // And match GT up to sign.
            let ok = cands.iter().any(|e| {
                let d1 = (e - egt).norm();
                let d2 = (e + egt).norm();
                d1.min(d2) < 1e-6
            });
            assert!(ok, "seed {seed}: no candidate matches GT essential");
        }
    }

    #[test]
    fn solves_planar_scene_where_eight_point_degenerates() {
        let (matches, pose) = scene(40, 7, true);
        let res = ransac_essential_5pt(&matches, 100, 1e-4, 11).expect("5pt ransac");
        assert!(res.inliers.len() >= 38, "inliers {}", res.inliers.len());
        let boot = recover_pose(&res.e, &matches, &res.inliers).expect("pose");
        let dr = boot.pose_ba.r.inverse() * pose.r;
        assert!(dr.angle() < 2e-3, "rot err {}", dr.angle());
    }

    #[test]
    fn survives_twenty_percent_inliers() {
        // The motivating regime: 20% precision, where eight-point sampling
        // has ~zero probability of an all-inlier draw.
        let (mut matches, pose) = scene(60, 23, false);
        let mut rng = Rng64::new(41);
        let mut r = |s: f64| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) * s;
        let n = matches.len();
        for i in 0..n {
            if i % 5 != 0 {
                // 80% gross outliers.
                matches[i].b = (r(0.6), r(0.6));
            }
        }
        let res = ransac_essential_5pt(&matches, 4000, 1e-4, 5).expect("5pt ransac");
        let true_inliers: Vec<usize> = (0..n).filter(|i| i % 5 == 0).collect();
        let found: Vec<usize> = res
            .inliers
            .iter()
            .copied()
            .filter(|i| true_inliers.contains(i))
            .collect();
        assert!(
            found.len() >= 5,
            "found only {} of {} true inliers",
            found.len(),
            true_inliers.len()
        );
        let boot = recover_pose(&res.e, &matches, &res.inliers).expect("pose");
        let dr = boot.pose_ba.r.inverse() * pose.r;
        assert!(dr.angle() < 5e-2, "rot err {}", dr.angle());
    }
}
