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

pub const LANDMARK_MAGIC: &[u8; 8] = b"SPLLM01\n";
pub const DESC_BYTES: usize = 32;

pub struct Landmark {
    pub pos: [f32; 3],
    pub color: [u8; 3],
    pub desc: [u8; DESC_BYTES],
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
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        match k {
            "video" => video = v.to_string(),
            "focal" => focal = v.parse()?,
            "width" => width = v.parse()?,
            "height" => height = v.parse()?,
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
        f.write_all(&l.desc)?;
    }
    Ok(())
}

pub fn read_landmarks(path: &Path) -> anyhow::Result<Vec<Landmark>> {
    let mut f = std::io::BufReader::new(
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
    );
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != LANDMARK_MAGIC {
        bail!("bad landmark file magic in {}", path.display());
    }
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
        let mut desc = [0u8; DESC_BYTES];
        f.read_exact(&mut desc)?;
        out.push(Landmark { pos, color, desc });
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
                desc: [k as u8; DESC_BYTES],
            })
            .collect();
        write_landmarks(&p, &lms).unwrap();
        let back = read_landmarks(&p).unwrap();
        assert_eq!(back.len(), 10);
        assert_eq!(back[3].pos, [3.0, -3.0, 1.5]);
        assert_eq!(back[7].desc, [7u8; DESC_BYTES]);
    }
}
