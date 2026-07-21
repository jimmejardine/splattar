//! Project persistence (M8): a project directory holds one submap per
//! ingested video. Submap surfels and landmarks stay in submap-local
//! coordinates forever (CLAUDE.md two-tier rule); each submap carries an
//! optional Sim(3) into the project world (= submap 0's gauge). Islands are
//! submaps without a registration — presentation offsets are computed at
//! view/export time, never stored.
//!
//! Formats are deliberately tiny and hand-rolled (no serde in the workspace):
//! `meta.txt` is `key=value` lines; `landmarks.bin` is a fixed-record binary
//! with a magic header.

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

/// Sim(3) mapping submap coordinates into project-world coordinates.
#[derive(Debug, Clone, Copy)]
pub struct WorldFromSubmap {
    pub scale: f64,
    /// Unit quaternion, wxyz.
    pub quat: [f64; 4],
    pub trans: [f64; 3],
}

impl WorldFromSubmap {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            quat: [1.0, 0.0, 0.0, 0.0],
            trans: [0.0; 3],
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
    /// None = unregistered island.
    pub world_from_submap: Option<WorldFromSubmap>,
}

pub struct Project {
    pub submaps: Vec<SubmapMeta>,
}

impl Project {
    pub fn submap_dir(root: &Path, idx: usize) -> PathBuf {
        root.join(format!("submap-{idx}"))
    }

    pub fn load(root: &Path) -> anyhow::Result<Project> {
        let mut submaps = Vec::new();
        for idx in 0.. {
            let dir = Self::submap_dir(root, idx);
            if !dir.exists() {
                break;
            }
            submaps.push(read_meta(&dir.join("meta.txt"))?);
        }
        if submaps.is_empty() {
            bail!("no submaps found in {}", root.display());
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
    if let Some(w) = &meta.world_from_submap {
        s.push_str(&format!(
            "sim3={} {} {} {} {} {} {} {}\n",
            w.scale,
            w.quat[0],
            w.quat[1],
            w.quat[2],
            w.quat[3],
            w.trans[0],
            w.trans[1],
            w.trans[2]
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
    let mut sim3 = None;
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
            "sim3" => {
                let nums: Vec<f64> = v
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                if nums.len() != 8 {
                    bail!("bad sim3 line in {}", path.display());
                }
                sim3 = Some(WorldFromSubmap {
                    scale: nums[0],
                    quat: [nums[1], nums[2], nums[3], nums[4]],
                    trans: [nums[5], nums[6], nums[7]],
                });
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
        world_from_submap: sim3,
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
    fn meta_roundtrip() {
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
            world_from_submap: Some(WorldFromSubmap {
                scale: 2.5,
                quat: [0.9, 0.1, -0.2, 0.3],
                trans: [1.0, -2.0, 3.0],
            }),
        };
        write_meta(&p, &meta).unwrap();
        let back = read_meta(&p).unwrap();
        assert_eq!(back.video, meta.video);
        assert_eq!(back.width, 478);
        assert_eq!(back.kf_range, Some((535, 1290)));
        let w = back.world_from_submap.unwrap();
        assert_eq!(w.scale, 2.5);
        assert_eq!(w.trans, [1.0, -2.0, 3.0]);
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
