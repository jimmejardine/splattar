//! Compat ply writer round-trip: write baked surfels, read back with the
//! standard loader, and compare activated values (positions, scales, quat,
//! opacity, SH interleave) exactly.

use gs_io::{PlyContents, load_ply, write_3dgs_ply};

#[test]
fn write_then_read_roundtrip() {
    let count = 3;
    // Positions vec4-strided; scales activated; quats xyzw; opacities 0..1;
    // sh 48/surfel coefficient-major.
    let positions: Vec<f32> = (0..count * 4).map(|i| i as f32 * 0.25).collect();
    let scales: Vec<f32> = (0..count * 2).map(|i| 0.05 + i as f32 * 0.01).collect();
    let quats: Vec<f32> = (0..count)
        .flat_map(|i| {
            let q = glam::Quat::from_rotation_y(0.3 + i as f32).normalize();
            [q.x, q.y, q.z, q.w]
        })
        .collect();
    let opacities: Vec<f32> = vec![0.25, 0.5, 0.9];
    let sh: Vec<f32> = (0..count * 48).map(|i| (i as f32 * 0.013).sin() * 0.4).collect();

    let dir = std::env::temp_dir().join("splattar-writer-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip.ply");
    write_3dgs_ply(&path, count, &positions, &scales, &quats, &opacities, &sh).unwrap();

    let PlyContents::Splats(cloud) = load_ply(&path).unwrap() else {
        panic!("expected splats");
    };
    assert_eq!(cloud.len(), count);
    assert_eq!(cloud.sh_degree, 3);
    for i in 0..count {
        for d in 0..3 {
            assert!((cloud.positions[i][d] - positions[i * 4 + d]).abs() < 1e-6);
        }
        // Reader applies exp() to the log scales we wrote.
        assert!((cloud.scales[i].x - scales[i * 2]).abs() < 1e-6);
        assert!((cloud.scales[i].y - scales[i * 2 + 1]).abs() < 1e-6);
        assert!(cloud.scales[i].z < 1e-4, "third scale must be flattened");
        // Reader applies sigmoid to the logits we wrote.
        assert!((cloud.opacity[i] - opacities[i]).abs() < 1e-5);
        // Quat: reader normalizes and converts w-first → xyzw.
        let q = cloud.rotations[i];
        for d in 0..4 {
            assert!((q.to_array()[d] - quats[i * 4 + d]).abs() < 1e-5, "quat {i}[{d}]");
        }
        // SH round-trips through the channel-major re-interleave.
        for k in 0..16 {
            for ch in 0..3 {
                let want = sh[i * 48 + k * 3 + ch];
                let got = cloud.sh[i * 48 + k * 3 + ch];
                assert!((got - want).abs() < 1e-6, "sh {i}[{k}][{ch}]");
            }
        }
    }
}
