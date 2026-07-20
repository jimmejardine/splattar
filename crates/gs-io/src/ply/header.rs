//! .ply header parsing. Reads exactly up to the byte after `end_header\n`,
//! leaving the reader positioned at the binary payload.

use std::io::Read;

use super::PlyError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyType {
    Float,
    Double,
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
}

impl PropertyType {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "float" | "float32" => Self::Float,
            "double" | "float64" => Self::Double,
            "char" | "int8" => Self::Char,
            "uchar" | "uint8" => Self::UChar,
            "short" | "int16" => Self::Short,
            "ushort" | "uint16" => Self::UShort,
            "int" | "int32" => Self::Int,
            "uint" | "uint32" => Self::UInt,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Float | Self::Int | Self::UInt => 4,
            Self::Double => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Property {
    pub name: String,
    pub ty: PropertyType,
}

#[derive(Debug, Clone)]
pub struct Element {
    pub name: String,
    pub count: usize,
    pub properties: Vec<Property>,
    /// True if any `property list` was declared on this element.
    pub has_list_property: bool,
}

impl Element {
    /// Byte stride of one row (only valid when `has_list_property` is false).
    pub fn stride(&self) -> usize {
        self.properties.iter().map(|p| p.ty.size()).sum()
    }

    /// Byte offset of a named property within a row, with its type.
    pub fn offset_of(&self, name: &str) -> Option<(usize, PropertyType)> {
        let mut off = 0;
        for p in &self.properties {
            if p.name == name {
                return Some((off, p.ty));
            }
            off += p.ty.size();
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct PlyHeader {
    pub elements: Vec<Element>,
}

impl PlyHeader {
    pub fn vertex(&self) -> Option<&Element> {
        self.elements.iter().find(|e| e.name == "vertex")
    }
}

/// Reads header lines byte-at-a-time (small reads against a BufReader) so the
/// reader lands exactly at the start of the binary payload.
pub fn parse_header(r: &mut impl Read) -> Result<PlyHeader, PlyError> {
    let magic = read_line(r)?;
    if magic.trim_end() != "ply" {
        return Err(PlyError::NotPly);
    }

    let mut elements: Vec<Element> = Vec::new();
    let mut format_seen = false;
    loop {
        let line = read_line(r)?;
        let line = line.trim_end();
        let mut tokens = line.split_ascii_whitespace();
        match tokens.next() {
            Some("format") => {
                let rest = line["format".len()..].trim().to_string();
                if rest != "binary_little_endian 1.0" {
                    return Err(PlyError::UnsupportedFormat(rest));
                }
                format_seen = true;
            }
            Some("comment") | Some("obj_info") | None => {}
            Some("element") => {
                let name = tokens
                    .next()
                    .ok_or_else(|| PlyError::BadHeader(format!("bad element line: {line}")))?;
                let count: usize = tokens
                    .next()
                    .and_then(|c| c.parse().ok())
                    .ok_or_else(|| PlyError::BadHeader(format!("bad element count: {line}")))?;
                elements.push(Element {
                    name: name.to_string(),
                    count,
                    properties: Vec::new(),
                    has_list_property: false,
                });
            }
            Some("property") => {
                let element = elements
                    .last_mut()
                    .ok_or_else(|| PlyError::BadHeader("property before element".into()))?;
                let ty_tok = tokens
                    .next()
                    .ok_or_else(|| PlyError::BadHeader(format!("bad property line: {line}")))?;
                if ty_tok == "list" {
                    element.has_list_property = true;
                    continue;
                }
                let ty = PropertyType::parse(ty_tok)
                    .ok_or_else(|| PlyError::BadHeader(format!("unknown type: {ty_tok}")))?;
                let name = tokens
                    .next()
                    .ok_or_else(|| PlyError::BadHeader(format!("bad property line: {line}")))?;
                element.properties.push(Property {
                    name: name.to_string(),
                    ty,
                });
            }
            Some("end_header") => break,
            Some(other) => {
                return Err(PlyError::BadHeader(format!("unexpected keyword: {other}")));
            }
        }
    }
    if !format_seen {
        return Err(PlyError::BadHeader("missing format line".into()));
    }
    Ok(PlyHeader { elements })
}

/// Read one `\n`-terminated header line (headers are tiny; byte reads hit the
/// BufReader). Bounded to catch binary garbage posing as a header.
fn read_line(r: &mut impl Read) -> Result<String, PlyError> {
    const MAX: usize = 4096;
    let mut buf = Vec::with_capacity(64);
    let mut byte = [0u8; 1];
    loop {
        let n = r.read(&mut byte)?;
        if n == 0 {
            return Err(PlyError::BadHeader("eof inside header".into()));
        }
        if byte[0] == b'\n' {
            break;
        }
        if buf.len() >= MAX {
            return Err(PlyError::BadHeader("header line too long".into()));
        }
        buf.push(byte[0]);
    }
    String::from_utf8(buf).map_err(|_| PlyError::BadHeader("non-utf8 header line".into()))
}
