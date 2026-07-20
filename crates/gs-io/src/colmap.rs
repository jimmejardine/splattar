//! COLMAP binary sparse-model loader (cameras.bin / images.bin /
//! points3D.bin) — the posed-sequence validation harness input (PLAN.md M3).
//! Not a product input path: the product ingests video only.
//!
//! Convention conversion: COLMAP is x-right/y-down/z-forward with
//! world→camera (R, t); ours is y-up looking down −z with camera-to-world
//! (quat, center). ours_c2w = colmap_c2wᵀ… i.e. Rᵀ·diag(1,−1,−1), C = −Rᵀt.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use glam::{DMat3, DQuat, DVec3};

#[derive(Debug, thiserror::Error)]
pub enum ColmapError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed colmap file: {0}")]
    Malformed(String),
    #[error("unsupported camera model id {0} (supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL)")]
    UnsupportedModel(u32),
    #[error("no sparse model found under {0} (looked for sparse/0 and sparse)")]
    NoSparseModel(PathBuf),
    #[error("image load failed for {0}: {1}")]
    Image(PathBuf, String),
}

pub struct ColmapView {
    pub name: String,
    /// Camera-to-world rotation (our convention), xyzw.
    pub quat: [f32; 4],
    pub center: [f32; 3],
    /// rgba-f32 pixels (w = 1), row-major, loaded at the dataset resolution.
    pub image: Vec<[f32; 4]>,
}

pub struct ColmapDataset {
    pub width: u32,
    pub height: u32,
    pub focal: f32,
    pub views: Vec<ColmapView>,
    /// SfM points: position + rgb color.
    pub points: Vec<SfmPoint>,
}

struct Cursor<R: Read> {
    r: R,
}

impl<R: Read> Cursor<R> {
    fn u32(&mut self) -> Result<u32, ColmapError> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, ColmapError> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn f64(&mut self) -> Result<f64, ColmapError> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(f64::from_le_bytes(b))
    }
    fn u8(&mut self) -> Result<u8, ColmapError> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn cstring(&mut self) -> Result<String, ColmapError> {
        let mut s = Vec::new();
        loop {
            let c = self.u8()?;
            if c == 0 {
                break;
            }
            s.push(c);
        }
        String::from_utf8(s).map_err(|_| ColmapError::Malformed("non-utf8 image name".into()))
    }
    fn skip(&mut self, n: u64) -> Result<(), ColmapError> {
        std::io::copy(&mut (&mut self.r).take(n), &mut std::io::sink())?;
        Ok(())
    }
}

struct CameraIntrinsics {
    width: u64,
    height: u64,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

fn read_cameras(path: &Path) -> Result<Vec<(u32, CameraIntrinsics)>, ColmapError> {
    let mut c = Cursor {
        r: BufReader::new(std::fs::File::open(path)?),
    };
    let n = c.u64()?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let id = c.u32()?;
        let model = c.u32()?;
        let width = c.u64()?;
        let height = c.u64()?;
        let cam = match model {
            0 => {
                // SIMPLE_PINHOLE: f, cx, cy
                let f = c.f64()?;
                let cx = c.f64()?;
                let cy = c.f64()?;
                CameraIntrinsics { width, height, fx: f, fy: f, cx, cy }
            }
            1 => {
                // PINHOLE: fx, fy, cx, cy
                let fx = c.f64()?;
                let fy = c.f64()?;
                let cx = c.f64()?;
                let cy = c.f64()?;
                CameraIntrinsics { width, height, fx, fy, cx, cy }
            }
            2 => {
                // SIMPLE_RADIAL: f, cx, cy, k — distortion ignored with a warning.
                let f = c.f64()?;
                let cx = c.f64()?;
                let cy = c.f64()?;
                let k = c.f64()?;
                if k.abs() > 1e-3 {
                    log::warn!("SIMPLE_RADIAL k={k:.4} ignored (undistortion not implemented)");
                }
                CameraIntrinsics { width, height, fx: f, fy: f, cx, cy }
            }
            other => return Err(ColmapError::UnsupportedModel(other)),
        };
        out.push((id, cam));
    }
    Ok(out)
}

struct RawImage {
    name: String,
    quat_wxyz: [f64; 4], // world→camera rotation
    t: [f64; 3],
    camera_id: u32,
}

fn read_images(path: &Path) -> Result<Vec<RawImage>, ColmapError> {
    let mut c = Cursor {
        r: BufReader::new(std::fs::File::open(path)?),
    };
    let n = c.u64()?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let _image_id = c.u32()?;
        let quat_wxyz = [c.f64()?, c.f64()?, c.f64()?, c.f64()?];
        let t = [c.f64()?, c.f64()?, c.f64()?];
        let camera_id = c.u32()?;
        let name = c.cstring()?;
        let num_points = c.u64()?;
        c.skip(num_points * 24)?; // x, y (f64) + point3D id (i64)
        out.push(RawImage {
            name,
            quat_wxyz,
            t,
            camera_id,
        });
    }
    Ok(out)
}

/// SfM point: position + rgb.
pub type SfmPoint = ([f32; 3], [u8; 3]);

fn read_points(path: &Path) -> Result<Vec<SfmPoint>, ColmapError> {
    let mut c = Cursor {
        r: BufReader::new(std::fs::File::open(path)?),
    };
    let n = c.u64()?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let _id = c.u64()?;
        let p = [c.f64()? as f32, c.f64()? as f32, c.f64()? as f32];
        let rgb = [c.u8()?, c.u8()?, c.u8()?];
        let _error = c.f64()?;
        let track_len = c.u64()?;
        c.skip(track_len * 8)?;
        out.push((p, rgb));
    }
    Ok(out)
}

fn sparse_dir(root: &Path) -> Result<PathBuf, ColmapError> {
    for candidate in [root.join("sparse/0"), root.join("sparse")] {
        if candidate.join("cameras.bin").exists() {
            return Ok(candidate);
        }
    }
    Err(ColmapError::NoSparseModel(root.to_path_buf()))
}

/// Load a COLMAP dataset directory (sparse model + images). `downscale`
/// prefers a pre-scaled `images_{downscale}` directory (Mip-NeRF360 layout)
/// and falls back to resizing.
pub fn load_colmap(root: &Path, downscale: u32) -> Result<ColmapDataset, ColmapError> {
    let sparse = sparse_dir(root)?;
    let cameras = read_cameras(&sparse.join("cameras.bin"))?;
    let images = read_images(&sparse.join("images.bin"))?;
    let points = read_points(&sparse.join("points3D.bin"))?;

    let images_dir = if downscale > 1 && root.join(format!("images_{downscale}")).exists() {
        root.join(format!("images_{downscale}"))
    } else {
        root.join("images")
    };
    let pre_scaled = images_dir != root.join("images");

    let intrinsics = |id: u32| cameras.iter().find(|(cid, _)| *cid == id).map(|(_, c)| c);
    let first = intrinsics(images[0].camera_id)
        .ok_or_else(|| ColmapError::Malformed("image references unknown camera".into()))?;

    let scale = downscale.max(1) as f64;
    let width = (first.width as f64 / scale).round() as u32;
    let height = (first.height as f64 / scale).round() as u32;
    let focal = (first.fx / scale) as f32;
    if (first.fx - first.fy).abs() / first.fx > 0.01 {
        log::warn!("fx≠fy ({:.1} vs {:.1}) — using fx; expect slight distortion", first.fx, first.fy);
    }
    let (ecx, ecy) = (first.width as f64 * 0.5, first.height as f64 * 0.5);
    if (first.cx - ecx).abs() / first.width as f64 > 0.01
        || (first.cy - ecy).abs() / first.height as f64 > 0.01
    {
        log::warn!(
            "principal point ({:.1}, {:.1}) off-center — loader assumes centered; expect bias",
            first.cx,
            first.cy
        );
    }

    let flip = DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0));
    let mut views = Vec::with_capacity(images.len());
    for raw in &images {
        let [w, x, y, z] = raw.quat_wxyz;
        let r_w2c = DMat3::from_quat(DQuat::from_xyzw(x, y, z, w).normalize());
        let t = DVec3::new(raw.t[0], raw.t[1], raw.t[2]);
        let center = -(r_w2c.transpose() * t);
        // ours_c2w = colmap_c2w (= Rᵀ) with y/z axes flipped.
        let c2w = r_w2c.transpose() * flip;
        let q = DQuat::from_mat3(&c2w).normalize();

        let path = images_dir.join(&raw.name);
        let img = image::open(&path)
            .map_err(|e| ColmapError::Image(path.clone(), e.to_string()))?;
        let img = if !pre_scaled && downscale > 1 {
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        } else {
            img
        };
        let rgb = img.to_rgb8();
        if rgb.width() != width || rgb.height() != height {
            return Err(ColmapError::Malformed(format!(
                "{}: size {}x{} != dataset {}x{}",
                raw.name,
                rgb.width(),
                rgb.height(),
                width,
                height
            )));
        }
        let image = rgb
            .pixels()
            .map(|p| {
                [
                    p.0[0] as f32 / 255.0,
                    p.0[1] as f32 / 255.0,
                    p.0[2] as f32 / 255.0,
                    1.0,
                ]
            })
            .collect();
        views.push(ColmapView {
            name: raw.name.clone(),
            quat: [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
            center: [center.x as f32, center.y as f32, center.z as f32],
            image,
        });
    }
    views.sort_by(|a, b| a.name.cmp(&b.name));

    log::info!(
        "colmap dataset: {} views at {width}x{height} (focal {focal:.1}), {} SfM points",
        views.len(),
        points.len()
    );
    Ok(ColmapDataset {
        width,
        height,
        focal,
        views,
        points,
    })
}
