//! End-to-end `gs-cli add` gates on real sample videos. Slow (VO + a short
//! training run per submap) and sample-dependent — all `#[ignore]`; run with
//! `cargo test -p gs-cli --release --test add_e2e -- --ignored`.
//!
//! Projects are created under temp dirs — never inside samples/ (repo rule).

use std::path::{Path, PathBuf};
use std::process::Command;

fn sample(rel: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/video")
        .join(rel);
    p.exists().then_some(p)
}

fn temp_project(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "splattar-add-e2e-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Run `gs-cli add` with fast training knobs; panic on failure.
fn add(video: &Path, project: &Path) {
    let status = Command::new(env!("CARGO_BIN_EXE_gs-cli"))
        .args([
            "add",
            &video.to_string_lossy(),
            "--project",
            &project.to_string_lossy(),
            "--iters",
            "200",
            "--max-views",
            "40",
            "--downscale",
            "4",
        ])
        .status()
        .expect("spawn gs-cli");
    assert!(status.success(), "gs-cli add {} failed", video.display());
}

/// Minimal meta.txt reader: (submap index, edge targets).
fn read_edges(project: &Path) -> Vec<(usize, Vec<usize>)> {
    let mut out = Vec::new();
    for i in 0.. {
        let meta = project.join(format!("submap-{i}")).join("meta.txt");
        let Ok(text) = std::fs::read_to_string(&meta) else { break };
        let targets: Vec<usize> = text
            .lines()
            .filter_map(|l| l.strip_prefix("edge="))
            .filter_map(|rest| rest.split_whitespace().next())
            .filter_map(|t| t.parse().ok())
            .collect();
        out.push((i, targets));
    }
    out
}

/// Union-find component count over the edge lists.
fn component_count(edges: &[(usize, Vec<usize>)]) -> usize {
    let n = edges.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (i, targets) in edges {
        for &t in targets {
            if t < n {
                let (a, b) = (find(&mut parent, *i), find(&mut parent, t));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count()
}

/// The native HEVC walkthrough solves as 3 VO segments; same-video temporal
/// bridging must connect at least one pair (components < submaps).
#[test]
#[ignore = "slow; needs samples/video/prinsengracht-494-back-room-android-raw"]
fn hevc_three_segments_bridge() {
    let Some(video) =
        sample("prinsengracht-494-back-room-android-raw/20260721_125547.mp4")
    else {
        eprintln!("sample video missing — skipping");
        return;
    };
    let project = temp_project("hevc");
    add(&video, &project);

    let edges = read_edges(&project);
    assert_eq!(edges.len(), 3, "expected 3 submaps (3 VO segments)");
    let comps = component_count(&edges);
    assert!(
        comps < 3,
        "temporal bridges should connect same-video segments (got {comps} components)"
    );
    let _ = std::fs::remove_dir_all(&project);
}

/// Adding two overlapping videos in either order must yield the same submap
/// and component counts. Today both orders produce islands (cross-video
/// registration is an open item — RESULTS.md); the assert stays valid once
/// registration starts working, because connectivity is order-independent.
#[test]
#[ignore = "slow; needs samples/video/prinsengracht-494-android"]
fn order_independence_cross_video() {
    let (Some(v1), Some(v2)) = (
        sample("prinsengracht-494-android/1.mp4"),
        sample("prinsengracht-494-android/2.mp4"),
    ) else {
        eprintln!("sample videos missing — skipping");
        return;
    };

    let pa = temp_project("order-a");
    add(&v1, &pa);
    add(&v2, &pa);
    let ea = read_edges(&pa);

    let pb = temp_project("order-b");
    add(&v2, &pb);
    add(&v1, &pb);
    let eb = read_edges(&pb);

    assert_eq!(ea.len(), eb.len(), "submap counts differ across add order");
    assert_eq!(
        component_count(&ea),
        component_count(&eb),
        "component counts differ across add order"
    );
    let _ = std::fs::remove_dir_all(&pa);
    let _ = std::fs::remove_dir_all(&pb);
}
