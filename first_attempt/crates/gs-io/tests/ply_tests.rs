//! gs-io ply reader tests against in-memory fixtures with hand-verifiable
//! values. The builder writes exactly the INRIA binary-LE layout.

use glam::{Quat, Vec3};
use gs_io::{PlyContents, PlyError, load_ply_from};

/// Build a splat ply: header + rows. `rest_per_channel` ∈ {0, 3, 8, 15}.
fn splat_ply(rows: &[Vec<f32>], rest_per_channel: usize, extra_prop: Option<&str>) -> Vec<u8> {
    let mut names: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
    if let Some(extra) = extra_prop {
        names.push(extra.to_string()); // exercises the offset-driven path
    }
    for i in 0..3 {
        names.push(format!("f_dc_{i}"));
    }
    for i in 0..rest_per_channel * 3 {
        names.push(format!("f_rest_{i}"));
    }
    names.push("opacity".into());
    for i in 0..3 {
        names.push(format!("scale_{i}"));
    }
    for i in 0..4 {
        names.push(format!("rot_{i}"));
    }

    let mut out = format!(
        "ply\nformat binary_little_endian 1.0\ncomment fixture\nelement vertex {}\n",
        rows.len()
    )
    .into_bytes();
    for n in &names {
        out.extend_from_slice(format!("property float {n}\n").as_bytes());
    }
    out.extend_from_slice(b"end_header\n");
    for row in rows {
        assert_eq!(row.len(), names.len(), "fixture row width mismatch");
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// One canonical degree-1 splat row with distinct, recognizable values.
/// rest (channel-major on disk): red=[1,2,3], green=[4,5,6], blue=[7,8,9] (×0.1).
fn deg1_row() -> Vec<f32> {
    let mut row = vec![1.0, 2.0, 3.0]; // pos
    row.extend([0.5, -0.5, 0.25]); // dc
    row.extend((1..=9).map(|i| i as f32 * 0.1)); // f_rest_0..8
    row.push(0.0); // opacity logit → 0.5
    row.extend([0.1f32.ln(), 0.2f32.ln(), 0.3f32.ln()]); // scales → 0.1, 0.2, 0.3
    row.extend([1.0, 0.0, 0.0, 0.0]); // rot: w=1 identity (INRIA w-first)
    row
}

#[test]
fn splat_deg1_values_and_interleave() {
    let data = splat_ply(&[deg1_row()], 3, None);
    let PlyContents::Splats(s) = load_ply_from(data.as_slice()).unwrap() else {
        panic!("expected splats");
    };
    assert_eq!(s.len(), 1);
    assert_eq!(s.sh_degree, 1);
    assert_eq!(s.positions[0], Vec3::new(1.0, 2.0, 3.0));
    // Coeff-major interleave: c0.rgb, then c1..c3 as (red_i, green_i, blue_i).
    let expect = [
        0.5, -0.5, 0.25, // c0
        0.1, 0.4, 0.7, // c1
        0.2, 0.5, 0.8, // c2
        0.3, 0.6, 0.9, // c3
    ];
    for (got, want) in s.sh.iter().zip(expect.iter()) {
        assert!((got - want).abs() < 1e-6, "sh interleave: got {got}, want {want}");
    }
    assert!((s.opacity[0] - 0.5).abs() < 1e-6, "sigmoid(0) = 0.5");
    assert!((s.scales[0] - Vec3::new(0.1, 0.2, 0.3)).length() < 1e-6, "exp(ln x) = x");
    assert!((s.rotations[0] - Quat::IDENTITY).length() < 1e-6, "rot_0 is w");
}

#[test]
fn splat_nonidentity_quat_is_normalized_and_reordered() {
    // ply order (w,x,y,z) = (2,0,0,2) → normalized (w,x,y,z) = (0.707.., 0, 0, 0.707..)
    let mut row = deg1_row();
    let n = row.len();
    row[n - 4..].copy_from_slice(&[2.0, 0.0, 0.0, 2.0]);
    let data = splat_ply(&[row], 3, None);
    let PlyContents::Splats(s) = load_ply_from(data.as_slice()).unwrap() else {
        panic!("expected splats");
    };
    let q = s.rotations[0];
    let inv = std::f32::consts::FRAC_1_SQRT_2;
    assert!((q.w - inv).abs() < 1e-6 && (q.z - inv).abs() < 1e-6);
    assert!((q.length() - 1.0).abs() < 1e-6);
}

#[test]
fn splat_deg0_no_rest() {
    let mut row = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5];
    row.push(10.0); // opacity → ~1.0
    row.extend([0.0, 0.0, 0.0]); // scales → 1.0
    row.extend([1.0, 0.0, 0.0, 0.0]);
    let data = splat_ply(&[row], 0, None);
    let PlyContents::Splats(s) = load_ply_from(data.as_slice()).unwrap() else {
        panic!("expected splats");
    };
    assert_eq!(s.sh_degree, 0);
    assert_eq!(s.sh.len(), 3);
    assert!(s.opacity[0] > 0.99);
    assert_eq!(s.scales[0], Vec3::ONE);
}

#[test]
fn extra_property_takes_offset_path_with_same_values() {
    // Insert a dummy float property after xyz → not canonical order anymore.
    let mut row = deg1_row();
    row.insert(3, 42.0);
    let data = splat_ply(&[row], 3, Some("nx"));
    let PlyContents::Splats(s) = load_ply_from(data.as_slice()).unwrap() else {
        panic!("expected splats");
    };
    assert_eq!(s.positions[0], Vec3::new(1.0, 2.0, 3.0));
    assert!((s.sh[3] - 0.1).abs() < 1e-6 && (s.sh[4] - 0.4).abs() < 1e-6);
    assert!((s.scales[0] - Vec3::new(0.1, 0.2, 0.3)).length() < 1e-6);
}

#[test]
fn point_cloud_uchar_rgb() {
    let mut out = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n".to_vec();
    for n in ["x", "y", "z"] {
        out.extend_from_slice(format!("property float {n}\n").as_bytes());
    }
    for n in ["red", "green", "blue"] {
        out.extend_from_slice(format!("property uchar {n}\n").as_bytes());
    }
    out.extend_from_slice(b"end_header\n");
    for (p, c) in [([0.0f32, 1.0, 2.0], [10u8, 20, 30]), ([3.0, 4.0, 5.0], [200, 150, 100])] {
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&c);
    }
    let PlyContents::Points(pc) = load_ply_from(out.as_slice()).unwrap() else {
        panic!("expected points");
    };
    assert_eq!(pc.len(), 2);
    assert_eq!(pc.positions[1], Vec3::new(3.0, 4.0, 5.0));
    assert_eq!(pc.colors[0], [10, 20, 30]);
    assert_eq!(pc.colors[1], [200, 150, 100]);
}

#[test]
fn ascii_format_is_rejected() {
    let data = b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n";
    match load_ply_from(data.as_slice()) {
        Err(PlyError::UnsupportedFormat(f)) => assert!(f.contains("ascii")),
        other => panic!("expected UnsupportedFormat, got {other:?}"),
    }
}

#[test]
fn truncated_payload_reports_eof() {
    let mut data = splat_ply(&[deg1_row(), deg1_row()], 3, None);
    data.truncate(data.len() - 40); // cut into the second row
    match load_ply_from(data.as_slice()) {
        Err(PlyError::UnexpectedEof { expected: 2, .. }) => {}
        other => panic!("expected UnexpectedEof, got {other:?}"),
    }
}

#[test]
fn missing_rot_is_unknown_layout() {
    // Build a splat header but drop rot_3 (truncate names manually).
    let mut out = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n".to_vec();
    for n in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2"] {
        out.extend_from_slice(format!("property float {n}\n").as_bytes());
    }
    out.extend_from_slice(b"end_header\n");
    match load_ply_from(out.as_slice()) {
        Err(PlyError::UnknownLayout { properties }) => assert!(properties.contains("rot_2")),
        other => panic!("expected UnknownLayout, got {other:?}"),
    }
}

#[test]
fn garbage_is_not_ply() {
    assert!(matches!(
        load_ply_from(&b"not a ply at all\n"[..]),
        Err(PlyError::NotPly)
    ));
}
