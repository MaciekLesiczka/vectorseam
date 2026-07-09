//! `.vseam` segment serialization and parsing.

use bytes::{BufMut, Bytes, BytesMut};
use thiserror::Error;

use crate::binary::{ByteReadError, read_u16_le, read_u32_le, read_u64_le};
use crate::cohort::{CohortNameError, MAX_COHORT_NAME_BYTES, validate_cohort_name};
use crate::frame::{FrameError, parse_frame_header, total_frame_len};

/// The `.vseam` segment magic bytes.
pub const SEGMENT_MAGIC: [u8; 4] = *b"VSG1";

const BASE_HEADER_LEN: usize = 46;
const FIXED_PREFIX_LEN: usize = 8;
const RECORD_TIME_LEN: usize = 8;

/// Maximum bytes added around record data by a serialized `.vseam` segment.
pub const MAX_SEGMENT_OVERHEAD_BYTES: usize =
    FIXED_PREFIX_LEN + BASE_HEADER_LEN + MAX_COHORT_NAME_BYTES;

/// Metadata stored at the front of every `.vseam` segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentHeader {
    /// Aligned window start as unix seconds UTC.
    pub window_start: u64,
    /// Window duration in seconds.
    pub window_seconds: u32,
    /// Receive time of the first kept frame as unix microseconds.
    pub first_receive: u64,
    /// Receive time of the last kept frame as unix microseconds.
    pub last_receive: u64,
    /// Frames received for this cohort in this segment part.
    pub received_frame_count: u64,
    /// Records stored in this segment part.
    pub record_count: u64,
    /// Validated cohort name.
    pub cohort: String,
}

/// A borrowed record to serialize into a segment.
#[derive(Debug, Clone, Copy)]
pub struct SegmentRecordRef<'a> {
    /// Collector receive time as unix microseconds.
    pub receive_time: u64,
    /// Raw protocol v1 frame bytes, including the frame length field.
    pub frame: &'a [u8],
}

/// A `.vseam` segment record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentRecord {
    /// Collector receive time as unix microseconds.
    pub receive_time: u64,
    /// Raw protocol v1 frame bytes, including the frame length field.
    pub frame: Bytes,
}

/// A parsed `.vseam` segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Segment header fields.
    pub header: SegmentHeader,
    /// Records parsed from the segment body.
    pub records: Vec<SegmentRecord>,
}

/// Errors returned when writing or reading a `.vseam` segment.
#[derive(Debug, Error)]
pub enum SegmentError {
    /// The segment magic was not `VSG1`.
    #[error("bad segment magic: expected VSG1")]
    BadMagic,
    /// The segment is too short for the declared field.
    #[error("truncated segment: need at least {needed} bytes, got {actual}")]
    Truncated {
        /// Number of bytes required.
        needed: usize,
        /// Number of bytes available.
        actual: usize,
    },
    /// The header length cannot contain the required known fields.
    #[error("segment header_len is shorter than known header fields")]
    HeaderTooShort,
    /// The cohort name length does not fit in the segment format.
    #[error("cohort name is too long for segment header")]
    CohortTooLong,
    /// The header's cohort bytes are not valid UTF-8.
    #[error("segment cohort is not valid UTF-8")]
    InvalidCohortUtf8,
    /// The cohort name does not match the VectorSeam grammar.
    #[error("invalid cohort name in segment header")]
    InvalidCohort(#[from] CohortNameError),
    /// A record contains an invalid protocol frame.
    #[error("invalid protocol frame in segment record")]
    InvalidFrame(#[from] FrameError),
    /// Segment length arithmetic overflowed the local platform size.
    #[error("segment length overflows usize")]
    LengthOverflow,
}

impl From<ByteReadError> for SegmentError {
    fn from(error: ByteReadError) -> Self {
        match error {
            ByteReadError::Truncated { needed, actual } => Self::Truncated { needed, actual },
            ByteReadError::LengthOverflow => Self::LengthOverflow,
        }
    }
}

/// Serializes a segment header and borrowed records into immutable bytes.
///
/// Each record is stored as an 8-byte receive timestamp followed by the
/// byte-exact protocol v1 frame supplied by the caller.
pub fn write_segment(
    header: &SegmentHeader,
    records: &[SegmentRecordRef<'_>],
) -> Result<Bytes, SegmentError> {
    validate_cohort_name(&header.cohort)?;
    let cohort_len = u16::try_from(header.cohort.len()).map_err(|_| SegmentError::CohortTooLong)?;
    let header_len = u32::try_from(BASE_HEADER_LEN + usize::from(cohort_len))
        .map_err(|_| SegmentError::LengthOverflow)?;

    let records_len = records.iter().try_fold(0_usize, |sum, record| {
        parse_frame_header(record.frame)?;
        sum.checked_add(RECORD_TIME_LEN)
            .and_then(|value| value.checked_add(record.frame.len()))
            .ok_or(SegmentError::LengthOverflow)
    })?;

    let capacity = FIXED_PREFIX_LEN
        .checked_add(usize::try_from(header_len).map_err(|_| SegmentError::LengthOverflow)?)
        .and_then(|value| value.checked_add(records_len))
        .ok_or(SegmentError::LengthOverflow)?;
    let mut out = BytesMut::with_capacity(capacity);
    out.put_slice(&SEGMENT_MAGIC);
    out.put_u32_le(header_len);
    out.put_u64_le(header.window_start);
    out.put_u32_le(header.window_seconds);
    out.put_u64_le(header.first_receive);
    out.put_u64_le(header.last_receive);
    out.put_u64_le(header.received_frame_count);
    out.put_u64_le(header.record_count);
    out.put_u16_le(cohort_len);
    out.put_slice(header.cohort.as_bytes());

    for record in records {
        out.put_u64_le(record.receive_time);
        out.put_slice(record.frame);
    }

    Ok(out.freeze())
}

/// Parses a `.vseam` segment into its header and owned records.
///
/// Unknown bytes appended to the header are skipped according to `header_len`.
pub fn read_segment(buffer: &[u8]) -> Result<Segment, SegmentError> {
    if buffer.len() < FIXED_PREFIX_LEN {
        return Err(SegmentError::Truncated {
            needed: FIXED_PREFIX_LEN,
            actual: buffer.len(),
        });
    }
    if buffer[0..4] != SEGMENT_MAGIC {
        return Err(SegmentError::BadMagic);
    }

    let header_len = read_u32_le(buffer, 4)? as usize;
    if header_len < BASE_HEADER_LEN {
        return Err(SegmentError::HeaderTooShort);
    }
    let header_end = FIXED_PREFIX_LEN
        .checked_add(header_len)
        .ok_or(SegmentError::LengthOverflow)?;
    if buffer.len() < header_end {
        return Err(SegmentError::Truncated {
            needed: header_end,
            actual: buffer.len(),
        });
    }

    let mut offset = FIXED_PREFIX_LEN;
    let window_start = read_u64_le(buffer, offset)?;
    offset += 8;
    let window_seconds = read_u32_le(buffer, offset)?;
    offset += 4;
    let first_receive = read_u64_le(buffer, offset)?;
    offset += 8;
    let last_receive = read_u64_le(buffer, offset)?;
    offset += 8;
    let received_frame_count = read_u64_le(buffer, offset)?;
    offset += 8;
    let record_count = read_u64_le(buffer, offset)?;
    offset += 8;
    let cohort_len = usize::from(read_u16_le(buffer, offset)?);
    offset += 2;
    let cohort_end = offset
        .checked_add(cohort_len)
        .ok_or(SegmentError::LengthOverflow)?;
    if cohort_end > header_end {
        return Err(SegmentError::HeaderTooShort);
    }
    let cohort = std::str::from_utf8(&buffer[offset..cohort_end])
        .map_err(|_| SegmentError::InvalidCohortUtf8)?
        .to_owned();
    validate_cohort_name(&cohort)?;

    let mut records = Vec::new();
    let mut record_offset = header_end;
    while record_offset < buffer.len() {
        let receive_time_end = record_offset
            .checked_add(RECORD_TIME_LEN)
            .ok_or(SegmentError::LengthOverflow)?;
        if receive_time_end > buffer.len() {
            return Err(SegmentError::Truncated {
                needed: receive_time_end,
                actual: buffer.len(),
            });
        }
        let receive_time = read_u64_le(buffer, record_offset)?;
        let frame_start = receive_time_end;
        let frame_total_len = total_frame_len(&buffer[frame_start..])?;
        let frame_end = frame_start
            .checked_add(frame_total_len)
            .ok_or(SegmentError::LengthOverflow)?;
        if frame_end > buffer.len() {
            return Err(SegmentError::Truncated {
                needed: frame_end,
                actual: buffer.len(),
            });
        }
        parse_frame_header(&buffer[frame_start..frame_end])?;
        records.push(SegmentRecord {
            receive_time,
            frame: Bytes::copy_from_slice(&buffer[frame_start..frame_end]),
        });
        record_offset = frame_end;
    }

    Ok(Segment {
        header: SegmentHeader {
            window_start,
            window_seconds,
            first_receive,
            last_receive,
            received_frame_count,
            record_count,
            cohort,
        },
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FIXED_FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_VERSION};

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

    fn header() -> SegmentHeader {
        SegmentHeader {
            window_start: 1_783_513_800,
            window_seconds: 600,
            first_receive: 1_000_000,
            last_receive: 2_000_000,
            received_frame_count: 5,
            record_count: 2,
            cohort: "prod/tenant-a".to_owned(),
        }
    }

    #[test]
    fn round_trips_segment_header_and_records() {
        let first = frame("prod/tenant-a", &[1, 2, 3, 4]);
        let second = frame("prod/tenant-a", &[5, 6, 7, 8]);
        let records = [
            SegmentRecordRef {
                receive_time: 1_000_000,
                frame: &first,
            },
            SegmentRecordRef {
                receive_time: 2_000_000,
                frame: &second,
            },
        ];

        let bytes = write_segment(&header(), &records).unwrap();
        let parsed = read_segment(&bytes).unwrap();

        assert_eq!(parsed.header, header());
        assert_eq!(parsed.records.len(), 2);
        assert_eq!(parsed.records[0].receive_time, 1_000_000);
        assert_eq!(parsed.records[0].frame, Bytes::from(first));
        assert_eq!(parsed.records[1].receive_time, 2_000_000);
        assert_eq!(parsed.records[1].frame, Bytes::from(second));
    }

    #[test]
    fn honors_header_len_with_unknown_tail_bytes() {
        let raw_frame = frame("prod/tenant-a", &[1, 2, 3, 4]);
        let records = [SegmentRecordRef {
            receive_time: 1_000_000,
            frame: &raw_frame,
        }];
        let mut bytes = write_segment(&header(), &records).unwrap().to_vec();
        let old_header_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let extra = [9_u8, 8, 7, 6];
        let header_end = 8 + old_header_len as usize;
        bytes.splice(header_end..header_end, extra);
        let new_header_len = old_header_len + u32::try_from(extra.len()).unwrap();
        bytes[4..8].copy_from_slice(&new_header_len.to_le_bytes());

        let parsed = read_segment(&bytes).unwrap();

        assert_eq!(parsed.header, header());
        assert_eq!(parsed.records.len(), 1);
        assert_eq!(parsed.records[0].frame, Bytes::from(raw_frame));
    }
}
