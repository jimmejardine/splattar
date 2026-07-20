//! Detects what a .ply's vertex element actually contains: a 3DGS gaussian
//! splat layout, a plain colored point cloud, or something we don't support.

use gs_core::sh;

use super::PlyError;
use super::header::{Element, PlyHeader, PropertyType};

#[derive(Debug)]
pub enum Layout {
    Gaussian3d(GaussianLayout),
    Points(PointsLayout),
}

/// Byte offsets (within one vertex row) for every property the splat reader
/// needs. Offset-driven so reordered or extra properties (e.g. nx/ny/nz) are
/// tolerated; `fast_path` is set when the row is byte-identical to the
/// canonical INRIA order, enabling bulk f32 decoding.
#[derive(Debug)]
pub struct GaussianLayout {
    pub count: usize,
    pub stride: usize,
    pub sh_degree: u8,
    pub fast_path: bool,
    pub pos: [usize; 3],
    pub f_dc: [usize; 3],
    /// Offsets of f_rest_0..N in index order (channel-major on disk).
    pub f_rest: Vec<usize>,
    pub opacity: usize,
    pub scale: [usize; 3],
    pub rot: [usize; 4],
}

#[derive(Debug)]
pub struct PointsLayout {
    pub count: usize,
    pub stride: usize,
    pub pos: [usize; 3],
    /// Offset and type of r/g/b (uchar or float supported).
    pub color: Option<([usize; 3], PropertyType)>,
}

pub fn detect(header: &PlyHeader) -> Result<Layout, PlyError> {
    let vertex = header.vertex().ok_or(PlyError::NoVertexElement)?;
    if vertex.has_list_property {
        return Err(PlyError::BadHeader("vertex element has list properties".into()));
    }

    if vertex.offset_of("f_dc_0").is_some() {
        Ok(Layout::Gaussian3d(detect_gaussian(vertex)?))
    } else if vertex.offset_of("x").is_some() {
        Ok(Layout::Points(detect_points(vertex)?))
    } else {
        Err(unknown_layout(vertex))
    }
}

fn unknown_layout(vertex: &Element) -> PlyError {
    PlyError::UnknownLayout {
        properties: vertex
            .properties
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    }
}

fn float_offset(vertex: &Element, name: &str) -> Result<usize, PlyError> {
    match vertex.offset_of(name) {
        Some((off, PropertyType::Float)) => Ok(off),
        Some(_) => Err(PlyError::BadPropertyType(name.to_string())),
        None => Err(unknown_layout(vertex)),
    }
}

fn detect_gaussian(vertex: &Element) -> Result<GaussianLayout, PlyError> {
    let pos = [
        float_offset(vertex, "x")?,
        float_offset(vertex, "y")?,
        float_offset(vertex, "z")?,
    ];
    let f_dc = [
        float_offset(vertex, "f_dc_0")?,
        float_offset(vertex, "f_dc_1")?,
        float_offset(vertex, "f_dc_2")?,
    ];
    let rest_count = vertex
        .properties
        .iter()
        .filter(|p| p.name.starts_with("f_rest_"))
        .count();
    let sh_degree =
        sh::degree_from_rest_count(rest_count).ok_or(PlyError::BadShCount(rest_count))?;
    let mut f_rest = Vec::with_capacity(rest_count);
    for i in 0..rest_count {
        f_rest.push(float_offset(vertex, &format!("f_rest_{i}"))?);
    }
    let opacity = float_offset(vertex, "opacity")?;
    let scale = [
        float_offset(vertex, "scale_0")?,
        float_offset(vertex, "scale_1")?,
        float_offset(vertex, "scale_2")?,
    ];
    let rot = [
        float_offset(vertex, "rot_0")?,
        float_offset(vertex, "rot_1")?,
        float_offset(vertex, "rot_2")?,
        float_offset(vertex, "rot_3")?,
    ];

    // Fast path: all properties float, in exactly canonical INRIA order.
    let canonical = canonical_order(rest_count);
    let fast_path = vertex.properties.len() == canonical.len()
        && vertex.properties.iter().all(|p| p.ty == PropertyType::Float)
        && vertex
            .properties
            .iter()
            .zip(canonical.iter())
            .all(|(p, c)| p.name == *c);

    Ok(GaussianLayout {
        count: vertex.count,
        stride: vertex.stride(),
        sh_degree,
        fast_path,
        pos,
        f_dc,
        f_rest,
        opacity,
        scale,
        rot,
    })
}

fn canonical_order(rest_count: usize) -> Vec<String> {
    let mut names = vec!["x".into(), "y".into(), "z".into()];
    for i in 0..3 {
        names.push(format!("f_dc_{i}"));
    }
    for i in 0..rest_count {
        names.push(format!("f_rest_{i}"));
    }
    names.push("opacity".into());
    for i in 0..3 {
        names.push(format!("scale_{i}"));
    }
    for i in 0..4 {
        names.push(format!("rot_{i}"));
    }
    names
}

fn detect_points(vertex: &Element) -> Result<PointsLayout, PlyError> {
    let pos = [
        float_offset(vertex, "x")?,
        float_offset(vertex, "y")?,
        float_offset(vertex, "z")?,
    ];
    let color = match (
        vertex.offset_of("red"),
        vertex.offset_of("green"),
        vertex.offset_of("blue"),
    ) {
        (Some((r, tr)), Some((g, tg)), Some((b, tb))) if tr == tg && tg == tb => match tr {
            PropertyType::UChar | PropertyType::Float => Some(([r, g, b], tr)),
            _ => return Err(PlyError::BadPropertyType("red".into())),
        },
        _ => None,
    };
    Ok(PointsLayout {
        count: vertex.count,
        stride: vertex.stride(),
        pos,
        color,
    })
}
