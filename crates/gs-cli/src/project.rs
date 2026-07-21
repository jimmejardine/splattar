//! Project persistence (M8): a project directory holds one submap per
//! ingested video segment. Submap surfels and landmarks stay in submap-local
//! coordinates forever (CLAUDE.md two-tier rule). Connectivity is stored as
//! **pairwise Sim(3) edges** between submaps — no submap has a privileged
//! gauge, so ingestion order doesn't matter. World placement is resolved per
//! connected component at compose time ([`resolve_placements`]): component
//! root = lowest submap index, transforms chained along edges; per-component
//! presentation offsets are computed at view/export time, never stored.
//! Submaps with no edges are islands.
//!
//! Formats are deliberately tiny and hand-rolled (no serde in the workspace):
//! `meta.txt` is `key=value` lines; `landmarks.bin` is a fixed-record binary
//! with a magic header. Legacy metas carrying an absolute `sim3=` line (the
//! old submap-0-gauge model) are migrated on read: identity → no edge (that
//! was the old gauge marker), anything else → an edge to submap 0.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

pub const LANDMARK_MAGIC_V1: &[u8; 8] = b"SPLLM01\n";
pub const LANDMARK_MAGIC_V2: &[u8; 8] = b"SPLLM02\n";
pub const LANDMARK_MAGIC_V3: &[u8; 8] = b"SPLLM03\n";
pub const LANDMARK_MAGIC_V4: &[u8; 8] = b"SPLLM04\n";
pub const LANDMARK_MAGIC: &[u8; 8] = b"SPLLM05\n";
pub const DESC_BYTES: usize = 32;
/// Pyramid levels per multi-scale descriptor (mirrors gs_pose::descriptor).
pub const DESC_LEVELS: usize = 3;

pub struct Landmark {
    pub pos: [f32; 3],
    pub color: [u8; 3],
    /// Oriented multi-scale descriptor (one steered BRIEF per pyramid
    /// level). Pre-v4 files carry a single level, replicated on load.
    pub desc: [[u8; DESC_BYTES]; DESC_LEVELS],
    /// Reference keyframe index (VO-global within the source video); enables
    /// temporal-bridge and covisibility-grouped registration. `u32::MAX` for
    /// landmarks loaded from v1 files.
    pub kf: u32,
    /// Pixel position of the reference observation in keyframe `kf` (2D
    /// anchor for PnP-style bridging). NaN when loaded from pre-v3 files.
    pub obs: [f32; 2],
    /// ALL keyframe observations (kf index, pixel) — single-camera PnP
    /// registration needs co-observed sets. Ref-obs only on pre-v5 files.
    pub obs_all: Vec<(u32, [f32; 2])>,
}

/// One pairwise Sim(3) constraint: maps THIS submap's coordinates into
/// submap `to`'s coordinates. Never an identity gauge marker — a submap
/// with no edges is an island.
#[derive(Debug, Clone, Copy)]
pub struct Sim3Edge {
    pub to: u32,
    pub scale: f64,
    /// Unit quaternion, wxyz.
    pub quat: [f64; 4],
    pub trans: [f64; 3],
}

impl Sim3Edge {
    pub fn to_sim3(self) -> gs_pose::sim3::Sim3G {
        gs_pose::sim3::Sim3G {
            scale: self.scale,
            rot: glam::DQuat::from_xyzw(self.quat[1], self.quat[2], self.quat[3], self.quat[0]),
            trans: glam::DVec3::from_array(self.trans),
        }
    }

    pub fn from_sim3(to: u32, s: &gs_pose::sim3::Sim3G) -> Self {
        Self {
            to,
            scale: s.scale,
            quat: [s.rot.w, s.rot.x, s.rot.y, s.rot.z],
            trans: s.trans.to_array(),
        }
    }
}

pub struct SubmapMeta {
    pub video: String,
    pub focal: f64,
    pub width: u32,
    pub height: u32,
    /// Solved keyframe index range of the source VO segment (inclusive) —
    /// lets same-video segments find their temporal neighbors.
    pub kf_range: Option<(u32, u32)>,
    /// Focal measured by the trainer's photometric refinement (pixels).
    /// When it differs from `focal`, the persisted geometry still carries
    /// the guess-focal warp until `gs-cli refocal` re-bundles it.
    pub focal_refined: Option<f64>,
    /// Pairwise Sim(3) constraints to other submaps. Empty = island.
    pub edges: Vec<Sim3Edge>,
}

pub struct Project {
    pub submaps: Vec<SubmapMeta>,
}

/// Resolved world placement of one submap: `world` maps its local
/// coordinates into its connected component's root frame. Presentation
/// layout (side-by-side component offsets) is computed by the composer,
/// not here — this is pure graph algebra.
pub struct Placement {
    /// Component number, ordered by ascending root index.
    pub component: usize,
    /// Lowest submap index in the component (the frame `world` maps into).
    pub root: usize,
    pub world: gs_pose::sim3::Sim3G,
}

/// Union-find + BFS transform chaining over the pairwise edge graph.
/// Deterministic: components ordered by root (lowest member index);
/// neighbors visited in ascending index; first BFS visit wins on redundant
/// edges/cycles. Self-loops and out-of-range edge targets are ignored.
pub fn resolve_placements(proj: &Project) -> Vec<Placement> {
    let n = proj.submaps.len();
    // Undirected adjacency: (neighbor, edge sim3, edge_is_from_this_node).
    let mut adj: Vec<Vec<(usize, gs_pose::sim3::Sim3G, bool)>> = vec![Vec::new(); n];
    for (i, m) in proj.submaps.iter().enumerate() {
        for e in &m.edges {
            let t = e.to as usize;
            if t == i || t >= n {
                continue;
            }
            let s = e.to_sim3();
            adj[i].push((t, s, true));
            adj[t].push((i, s, false));
        }
    }
    for a in &mut adj {
        a.sort_by_key(|&(t, ..)| t);
    }

    let mut component = vec![usize::MAX; n];
    let mut placements: Vec<Placement> = (0..n)
        .map(|i| Placement {
            component: 0,
            root: i,
            world: gs_pose::sim3::Sim3G::identity(),
        })
        .collect();
    let mut next_component = 0;
    for root in 0..n {
        if component[root] != usize::MAX {
            continue;
        }
        let comp = next_component;
        next_component += 1;
        let mut queue = std::collections::VecDeque::from([root]);
        component[root] = comp;
        placements[root].component = comp;
        placements[root].root = root;
        placements[root].world = gs_pose::sim3::Sim3G::identity();
        while let Some(i) = queue.pop_front() {
            let world_i = placements[i].world;
            for &(t, s, forward) in &adj[i] {
                if component[t] != usize::MAX {
                    continue;
                }
                component[t] = comp;
                placements[t].component = comp;
                placements[t].root = root;
                // We are at placed node `i` visiting neighbor `t`.
                // forward: the edge is owned by `i` and maps i→t, so
                //   t→root = (i→root) ∘ (t→i) = world_i ∘ s⁻¹.
                // reverse: the edge is owned by `t` and maps t→i, so
                //   t→root = world_i ∘ s.
                placements[t].world = if forward {
                    world_i.compose(&s.inverse())
                } else {
                    world_i.compose(&s)
                };
                queue.push_back(t);
            }
        }
    }
    placements
}

impl Project {
    pub fn submap_dir(root: &Path, idx: usize) -> PathBuf {
        root.join(format!("submap-{idx}"))
    }

    pub fn load(root: &Path) -> anyhow::Result<Project> {
        let proj = Self::load_or_empty(root)?;
        if proj.submaps.is_empty() {
            bail!("no submaps found in {}", root.display());
        }
        Ok(proj)
    }

    /// Like [`Project::load`] but an absent/empty project is fine — the
    /// first `add` starts from nothing.
    pub fn load_or_empty(root: &Path) -> anyhow::Result<Project> {
        let mut submaps = Vec::new();
        for idx in 0.. {
            let dir = Self::submap_dir(root, idx);
            if !dir.exists() {
                break;
            }
            submaps.push(read_meta(&dir.join("meta.txt"))?);
        }
        Ok(Project { submaps })
    }

    /// Create the directory for the next submap and return (index, dir).
    pub fn next_submap_dir(root: &Path) -> anyhow::Result<(usize, PathBuf)> {
        std::fs::create_dir_all(root)?;
        for idx in 0.. {
            let dir = Self::submap_dir(root, idx);
            if !dir.exists() {
                std::fs::create_dir(&dir)?;
                return Ok((idx, dir));
            }
        }
        unreachable!()
    }
}

pub fn write_meta(path: &Path, meta: &SubmapMeta) -> anyhow::Result<()> {
    let mut s = format!(
        "video={}\nfocal={}\nwidth={}\nheight={}\n",
        meta.video, meta.focal, meta.width, meta.height
    );
    if let Some((a, b)) = meta.kf_range {
        s.push_str(&format!("kf_range={a} {b}\n"));
    }
    if let Some(f) = meta.focal_refined {
        s.push_str(&format!("focal_refined={f}\n"));
    }
    for e in &meta.edges {
        s.push_str(&format!(
            "edge={} {} {} {} {} {} {} {} {}\n",
            e.to,
            e.scale,
            e.quat[0],
            e.quat[1],
            e.quat[2],
            e.quat[3],
            e.trans[0],
            e.trans[1],
            e.trans[2]
        ));
    }
    std::fs::write(path, s)?;
    Ok(())
}

pub fn read_meta(path: &Path) -> anyhow::Result<SubmapMeta> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut video = String::new();
    let mut focal = 0.0f64;
    let (mut width, mut height) = (0u32, 0u32);
    let mut edges = Vec::new();
    let mut kf_range = None;
    let mut focal_refined = None;
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        match k {
            "video" => video = v.to_string(),
            "focal" => focal = v.parse()?,
            "width" => width = v.parse()?,
            "height" => height = v.parse()?,
            "kf_range" => {
                if let Some((a, b)) = v.split_once(' ') {
                    kf_range = Some((a.trim().parse()?, b.trim().parse()?));
                }
            }
            "focal_refined" => focal_refined = v.parse().ok(),
            "edge" => {
                let nums: Vec<f64> = v
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                if nums.len() != 9 {
                    bail!("bad edge line in {}", path.display());
                }
                edges.push(Sim3Edge {
                    to: nums[0] as u32,
                    scale: nums[1],
                    quat: [nums[2], nums[3], nums[4], nums[5]],
                    trans: [nums[6], nums[7], nums[8]],
                });
            }
            // Legacy absolute submap→world transform (world = submap-0's
            // gauge). Identity was the old privileged-gauge marker on
            // submap 0 → no edge; anything else migrates to an edge → 0.
            // (Exact float compare is safe: identity was written verbatim.)
            "sim3" => {
                let nums: Vec<f64> = v
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                if nums.len() != 8 {
                    bail!("bad sim3 line in {}", path.display());
                }
                let identity = nums == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                if !identity {
                    edges.push(Sim3Edge {
                        to: 0,
                        scale: nums[0],
                        quat: [nums[1], nums[2], nums[3], nums[4]],
                        trans: [nums[5], nums[6], nums[7]],
                    });
                }
            }
            _ => {}
        }
    }
    if video.is_empty() || focal == 0.0 {
        bail!("incomplete meta at {}", path.display());
    }
    Ok(SubmapMeta {
        video,
        focal,
        width,
        height,
        kf_range,
        focal_refined,
        edges,
    })
}

pub fn write_landmarks(path: &Path, landmarks: &[Landmark]) -> anyhow::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(LANDMARK_MAGIC)?;
    f.write_all(&(landmarks.len() as u64).to_le_bytes())?;
    for l in landmarks {
        for c in l.pos {
            f.write_all(&c.to_le_bytes())?;
        }
        f.write_all(&l.color)?;
        for lvl in &l.desc {
            f.write_all(lvl)?;
        }
        f.write_all(&l.kf.to_le_bytes())?;
        f.write_all(&l.obs[0].to_le_bytes())?;
        f.write_all(&l.obs[1].to_le_bytes())?;
        let n_obs = l.obs_all.len().min(u16::MAX as usize) as u16;
        f.write_all(&n_obs.to_le_bytes())?;
        for (kf, p) in l.obs_all.iter().take(n_obs as usize) {
            f.write_all(&kf.to_le_bytes())?;
            f.write_all(&p[0].to_le_bytes())?;
            f.write_all(&p[1].to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn read_landmarks(path: &Path) -> anyhow::Result<Vec<Landmark>> {
    let mut f = std::io::BufReader::new(
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
    );
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    let version = match &magic {
        m if m == LANDMARK_MAGIC => 5,
        m if m == LANDMARK_MAGIC_V4 => 4,
        m if m == LANDMARK_MAGIC_V3 => 3,
        m if m == LANDMARK_MAGIC_V2 => 2,
        m if m == LANDMARK_MAGIC_V1 => 1,
        _ => bail!("bad landmark file magic in {}", path.display()),
    };
    let mut n8 = [0u8; 8];
    f.read_exact(&mut n8)?;
    let n = u64::from_le_bytes(n8) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut pos = [0f32; 3];
        for c in &mut pos {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            *c = f32::from_le_bytes(b);
        }
        let mut color = [0u8; 3];
        f.read_exact(&mut color)?;
        let mut desc = [[0u8; DESC_BYTES]; DESC_LEVELS];
        if version >= 4 {
            for lvl in &mut desc {
                f.read_exact(lvl)?;
            }
        } else {
            // Single-level legacy descriptor: replicate across levels so the
            // scale-searching distance degrades to plain Hamming.
            let mut one = [0u8; DESC_BYTES];
            f.read_exact(&mut one)?;
            desc = [one; DESC_LEVELS];
        }
        let kf = if version >= 2 {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            u32::from_le_bytes(b)
        } else {
            u32::MAX
        };
        let obs = if version >= 3 {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            [
                f32::from_le_bytes(b[0..4].try_into().unwrap()),
                f32::from_le_bytes(b[4..8].try_into().unwrap()),
            ]
        } else {
            [f32::NAN; 2]
        };
        let obs_all = if version >= 5 {
            let mut n2 = [0u8; 2];
            f.read_exact(&mut n2)?;
            let cnt = u16::from_le_bytes(n2) as usize;
            let mut list = Vec::with_capacity(cnt);
            for _ in 0..cnt {
                let mut b = [0u8; 12];
                f.read_exact(&mut b)?;
                list.push((
                    u32::from_le_bytes(b[0..4].try_into().unwrap()),
                    [
                        f32::from_le_bytes(b[4..8].try_into().unwrap()),
                        f32::from_le_bytes(b[8..12].try_into().unwrap()),
                    ],
                ));
            }
            list
        } else if kf != u32::MAX && obs[0].is_finite() {
            vec![(kf, obs)]
        } else {
            Vec::new()
        };
        out.push(Landmark {
            pos,
            color,
            desc,
            kf,
            obs,
            obs_all,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_roundtrip_with_edges() {
        let dir = std::env::temp_dir().join("splattar-meta-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("meta.txt");
        let meta = SubmapMeta {
            video: "a/b.mp4".into(),
            focal: 722.5,
            width: 478,
            height: 850,
            kf_range: Some((535, 1290)),
            focal_refined: Some(769.4),
            edges: vec![
                Sim3Edge {
                    to: 0,
                    scale: 2.5,
                    quat: [0.9, 0.1, -0.2, 0.3],
                    trans: [1.0, -2.0, 3.0],
                },
                Sim3Edge {
                    to: 3,
                    scale: 0.7,
                    quat: [1.0, 0.0, 0.0, 0.0],
                    trans: [-4.5, 0.0, 0.25],
                },
            ],
        };
        write_meta(&p, &meta).unwrap();
        let back = read_meta(&p).unwrap();
        assert_eq!(back.video, meta.video);
        assert_eq!(back.width, 478);
        assert_eq!(back.kf_range, Some((535, 1290)));
        assert_eq!(back.edges.len(), 2);
        assert_eq!(back.edges[0].to, 0);
        assert_eq!(back.edges[0].scale, 2.5);
        assert_eq!(back.edges[1].to, 3);
        assert_eq!(back.edges[1].trans, [-4.5, 0.0, 0.25]);

        // Island: no edges, no edge= lines.
        let island = SubmapMeta { edges: Vec::new(), ..meta };
        write_meta(&p, &island).unwrap();
        assert!(!std::fs::read_to_string(&p).unwrap().contains("edge="));
        assert!(read_meta(&p).unwrap().edges.is_empty());
    }

    #[test]
    fn legacy_sim3_migration() {
        let dir = std::env::temp_dir().join("splattar-meta-migrate");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("meta.txt");
        // Non-identity legacy transform → edge to submap 0.
        std::fs::write(
            &p,
            "video=x.mp4\nfocal=700\nwidth=100\nheight=100\nsim3=2 0.9 0.1 -0.2 0.3 1 -2 3\n",
        )
        .unwrap();
        let back = read_meta(&p).unwrap();
        assert_eq!(back.edges.len(), 1);
        assert_eq!(back.edges[0].to, 0);
        assert_eq!(back.edges[0].scale, 2.0);
        assert_eq!(back.edges[0].trans, [1.0, -2.0, 3.0]);
        // Identity legacy transform = the old submap-0 gauge marker → no edge.
        std::fs::write(
            &p,
            "video=x.mp4\nfocal=700\nwidth=100\nheight=100\nsim3=1 1 0 0 0 0 0 0\n",
        )
        .unwrap();
        assert!(read_meta(&p).unwrap().edges.is_empty());
    }

    fn dummy_meta(edges: Vec<Sim3Edge>) -> SubmapMeta {
        SubmapMeta {
            video: "v.mp4".into(),
            focal: 700.0,
            width: 100,
            height: 100,
            kf_range: None,
            focal_refined: None,
            edges,
        }
    }

    #[test]
    fn resolver_components_and_transforms() {
        use gs_pose::sim3::Sim3G;
        let sa = Sim3G {
            scale: 2.0,
            rot: glam::DQuat::from_axis_angle(glam::DVec3::Z, 0.7),
            trans: glam::DVec3::new(1.0, -2.0, 0.5),
        };
        let sb = Sim3G {
            scale: 0.5,
            rot: glam::DQuat::from_axis_angle(glam::DVec3::Y, -0.4),
            trans: glam::DVec3::new(0.0, 3.0, -1.0),
        };
        // Submap 1 carries edges to 0 (sa: 1→0) and to 2 (sb: 1→2);
        // submap 3 is an island.
        let proj = Project {
            submaps: vec![
                dummy_meta(vec![]),
                dummy_meta(vec![
                    Sim3Edge::from_sim3(0, &sa),
                    Sim3Edge::from_sim3(2, &sb),
                ]),
                dummy_meta(vec![]),
                dummy_meta(vec![]),
            ],
        };
        let pl = resolve_placements(&proj);
        assert_eq!(pl.len(), 4);
        // Components: {0,1,2} rooted at 0, {3} rooted at 3.
        assert_eq!(
            (pl[0].component, pl[1].component, pl[2].component, pl[3].component),
            (0, 0, 0, 1)
        );
        assert_eq!((pl[0].root, pl[2].root, pl[3].root), (0, 0, 3));
        let p = glam::DVec3::new(0.3, 1.7, -0.9);
        // world[1] maps 1-local → 0-frame = sa.
        assert!((pl[1].world.apply(p) - sa.apply(p)).length() < 1e-12);
        // world[2] maps 2-local → 0-frame = sa ∘ sb⁻¹.
        let expect = sa.compose(&sb.inverse());
        assert!((pl[2].world.apply(p) - expect.apply(p)).length() < 1e-12);
        // Roots are identity.
        assert!((pl[0].world.apply(p) - p).length() < 1e-15);
        assert!((pl[3].world.apply(p) - p).length() < 1e-15);
    }

    #[test]
    fn landmarks_roundtrip() {
        let dir = std::env::temp_dir().join("splattar-lm-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("landmarks.bin");
        let lms: Vec<Landmark> = (0..10)
            .map(|k| Landmark {
                pos: [k as f32, -(k as f32), 0.5 * k as f32],
                color: [k as u8, 2 * k as u8, 255 - k as u8],
                desc: [[k as u8; DESC_BYTES]; DESC_LEVELS],
                kf: 100 + k as u32,
                obs: [10.0 * k as f32, 5.0 + k as f32],
                obs_all: vec![
                    (100 + k as u32, [10.0 * k as f32, 5.0 + k as f32]),
                    (101 + k as u32, [11.0 * k as f32, 6.0 + k as f32]),
                ],
            })
            .collect();
        write_landmarks(&p, &lms).unwrap();
        let back = read_landmarks(&p).unwrap();
        assert_eq!(back.len(), 10);
        assert_eq!(back[3].pos, [3.0, -3.0, 1.5]);
        assert_eq!(back[7].desc, [[7u8; DESC_BYTES]; DESC_LEVELS]);
        assert_eq!(back[7].kf, 107);
    }
}
