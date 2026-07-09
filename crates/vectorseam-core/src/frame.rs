//! Protocol v1 frame parsing.

use std::str;

use thiserror::Error;

/// The protocol v1 magic bytes.
pub const FRAME_MAGIC: [u8; 4] = *b"VQS1";

/// The protocol v1 version number.
pub const FRAME_VERSION: u32 = 1;

/// The fixed protocol v1 header length in bytes, including `frame_len`.
pub const FIXED_FRAME_HEADER_LEN: usize = 28;

const FRAME_LEN_FIELD_LEN: usize = 4;
const FRAME_LEN_REMAINING_HEADER: usize = FIXED_FRAME_HEADER_LEN - FRAME_LEN_FIELD_LEN;

/// Errors returned when parsing a protocol v1 frame.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FrameError {
    /// The buffer does not contain enough bytes for the declared field.
    #[error("truncated frame: need at least {needed} bytes, got {actual}")]
    Truncated {
        /// Number of bytes required.
        needed: usize,
        /// Number of bytes available.
        actual: usize,
    },
    /// The frame magic was not `VQS1`.
    #[error("bad frame magic: expected VQS1")]
    BadMagic,
    /// The frame version is not supported.
    #[error("unsupported frame version {0}")]
    BadVersion(u32),
    /// The `name_len` field points past the declared frame.
    #[error("name_len exceeds frame length")]
    NameLenExceedsFrame,
    /// The declared and actual frame lengths do not agree.
    #[error("frame_len mismatch: declared total {declared_total} bytes, actual {actual} bytes")]
    FrameLenMismatch {
        /// Declared total frame bytes, including the `frame_len` field.
        declared_total: usize,
        /// Actual bytes in the supplied buffer.
        actual: usize,
    },
    /// The cohort name bytes are not valid UTF-8.
    #[error("frame name is not valid UTF-8")]
    InvalidNameUtf8,
    /// Frame length arithmetic overflowed the local platform size.
    #[error("frame length overflows usize")]
    LengthOverflow,
}

/// A protocol v1 frame header and cohort name.
///
/// The parser validates the self-delimiting frame envelope and exposes the
/// cohort name without reading or interpreting the vector payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader<'a> {
    /// Protocol `frame_len`, which excludes the length field itself.
    pub frame_len: u32,
    /// Vector data type.
    pub dtype: u32,
    /// Cohort name length in bytes.
    pub name_len: u32,
    /// Declared vector dimension.
    pub dimension: u32,
    /// Vector byte length.
    pub vector_len: u32,
    /// UTF-8 cohort name.
    pub name: &'a str,
}

/// Parses a protocol v1 frame from `buffer`.
///
/// The buffer must contain exactly one complete frame. The vector payload is
/// left untouched; only length fields and the cohort name bytes are inspected.
pub fn parse_frame_header(buffer: &[u8]) -> Result<FrameHeader<'_>, FrameError> {
    if buffer.len() < FIXED_FRAME_HEADER_LEN {
        return Err(FrameError::Truncated {
            needed: FIXED_FRAME_HEADER_LEN,
            actual: buffer.len(),
        });
    }

    let frame_len = read_u32(buffer, 0)?;
    let declared_total = usize::try_from(frame_len)
        .map_err(|_| FrameError::LengthOverflow)?
        .checked_add(FRAME_LEN_FIELD_LEN)
        .ok_or(FrameError::LengthOverflow)?;
    if declared_total != buffer.len() {
        return Err(FrameError::FrameLenMismatch {
            declared_total,
            actual: buffer.len(),
        });
    }
    if usize::try_from(frame_len).map_err(|_| FrameError::LengthOverflow)?
        < FRAME_LEN_REMAINING_HEADER
    {
        return Err(FrameError::Truncated {
            needed: FIXED_FRAME_HEADER_LEN,
            actual: declared_total,
        });
    }

    if buffer[4..8] != FRAME_MAGIC {
        return Err(FrameError::BadMagic);
    }

    let version = read_u32(buffer, 8)?;
    if version != FRAME_VERSION {
        return Err(FrameError::BadVersion(version));
    }

    let dtype = read_u32(buffer, 12)?;
    let name_len = read_u32(buffer, 16)?;
    let dimension = read_u32(buffer, 20)?;
    let vector_len = read_u32(buffer, 24)?;

    let name_len_usize = usize::try_from(name_len).map_err(|_| FrameError::LengthOverflow)?;
    let vector_len_usize = usize::try_from(vector_len).map_err(|_| FrameError::LengthOverflow)?;
    let name_start = FIXED_FRAME_HEADER_LEN;
    let vector_start = name_start
        .checked_add(name_len_usize)
        .ok_or(FrameError::LengthOverflow)?;
    if vector_start > declared_total {
        return Err(FrameError::NameLenExceedsFrame);
    }

    let expected_total = vector_start
        .checked_add(vector_len_usize)
        .ok_or(FrameError::LengthOverflow)?;
    if expected_total != declared_total {
        return Err(FrameError::FrameLenMismatch {
            declared_total,
            actual: expected_total,
        });
    }

    let name = str::from_utf8(&buffer[name_start..vector_start])
        .map_err(|_| FrameError::InvalidNameUtf8)?;

    Ok(FrameHeader {
        frame_len,
        dtype,
        name_len,
        dimension,
        vector_len,
        name,
    })
}

/// Returns the total frame length from a buffer containing at least the
/// `frame_len` field.
pub fn total_frame_len(buffer: &[u8]) -> Result<usize, FrameError> {
    if buffer.len() < FRAME_LEN_FIELD_LEN {
        return Err(FrameError::Truncated {
            needed: FRAME_LEN_FIELD_LEN,
            actual: buffer.len(),
        });
    }
    usize::try_from(read_u32(buffer, 0)?)
        .map_err(|_| FrameError::LengthOverflow)?
        .checked_add(FRAME_LEN_FIELD_LEN)
        .ok_or(FrameError::LengthOverflow)
}

fn read_u32(buffer: &[u8], offset: usize) -> Result<u32, FrameError> {
    let end = offset
        .checked_add(FRAME_LEN_FIELD_LEN)
        .ok_or(FrameError::LengthOverflow)?;
    let bytes = buffer.get(offset..end).ok_or(FrameError::Truncated {
        needed: end,
        actual: buffer.len(),
    })?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(name: &str, vector: &[u8]) -> Vec<u8> {
        let name_bytes = name.as_bytes();
        let frame_len = (FIXED_FRAME_HEADER_LEN - 4 + name_bytes.len() + vector.len()) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&frame_len.to_le_bytes());
        out.extend_from_slice(&FRAME_MAGIC);
        out.extend_from_slice(&FRAME_VERSION.to_le_bytes());
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&((vector.len() / 4) as u32).to_le_bytes());
        out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(vector);
        out
    }

    #[test]
    fn parses_valid_frame_without_touching_vector_bytes() {
        let vector = [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
        let bytes = frame("prod", &vector);

        let parsed = parse_frame_header(&bytes).unwrap();

        assert_eq!(parsed.frame_len, 36);
        assert_eq!(parsed.dtype, 1);
        assert_eq!(parsed.name_len, 4);
        assert_eq!(parsed.dimension, 2);
        assert_eq!(parsed.vector_len, 8);
        assert_eq!(parsed.name, "prod");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = frame("prod", &[0, 1, 2, 3]);
        bytes[4..8].copy_from_slice(b"NOPE");

        assert_eq!(parse_frame_header(&bytes), Err(FrameError::BadMagic));
    }

    #[test]
    fn rejects_bad_version() {
        let mut bytes = frame("prod", &[0, 1, 2, 3]);
        bytes[8..12].copy_from_slice(&2_u32.to_le_bytes());

        assert_eq!(parse_frame_header(&bytes), Err(FrameError::BadVersion(2)));
    }

    #[test]
    fn rejects_truncated_frame() {
        let bytes = vec![0; FIXED_FRAME_HEADER_LEN - 1];

        assert_eq!(
            parse_frame_header(&bytes),
            Err(FrameError::Truncated {
                needed: FIXED_FRAME_HEADER_LEN,
                actual: FIXED_FRAME_HEADER_LEN - 1,
            })
        );
    }

    #[test]
    fn rejects_name_len_exceeding_frame() {
        let mut bytes = frame("prod", &[0, 1, 2, 3]);
        bytes[16..20].copy_from_slice(&99_u32.to_le_bytes());

        assert_eq!(
            parse_frame_header(&bytes),
            Err(FrameError::NameLenExceedsFrame)
        );
    }

    #[test]
    fn rejects_frame_len_mismatch() {
        let mut bytes = frame("prod", &[0, 1, 2, 3]);
        bytes[0..4].copy_from_slice(&99_u32.to_le_bytes());

        assert_eq!(
            parse_frame_header(&bytes),
            Err(FrameError::FrameLenMismatch {
                declared_total: 103,
                actual: bytes.len(),
            })
        );
    }
}
