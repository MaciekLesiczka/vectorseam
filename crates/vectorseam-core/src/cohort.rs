//! Cohort name validation.

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

/// Maximum UTF-8 byte length of a cohort name.
pub const MAX_COHORT_NAME_BYTES: usize = 255;

/// Maximum number of slash-separated segments in a cohort name.
pub const MAX_COHORT_SEGMENTS: usize = 8;

/// Maximum byte length of one cohort segment.
pub const MAX_COHORT_SEGMENT_BYTES: usize = 63;

/// Errors returned when validating a cohort name.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CohortNameError {
    /// The cohort name is empty.
    #[error("cohort name must not be empty")]
    Empty,
    /// The cohort name exceeds the whole-name byte limit.
    #[error("cohort name exceeds {MAX_COHORT_NAME_BYTES} bytes")]
    NameTooLong,
    /// The cohort name contains non-ASCII bytes.
    #[error("cohort name must be ASCII")]
    NonAscii,
    /// The cohort name has more than the allowed number of segments.
    #[error("cohort name has more than {MAX_COHORT_SEGMENTS} segments")]
    TooManySegments,
    /// A segment is empty, which includes leading, trailing, or repeated `/`.
    #[error("cohort name must not contain empty segments")]
    EmptySegment,
    /// A segment exceeds the segment byte limit.
    #[error("cohort segment exceeds {MAX_COHORT_SEGMENT_BYTES} bytes")]
    SegmentTooLong,
    /// A segment is `.` or `..`, which is ambiguous in local filesystem paths.
    #[error("cohort segment must not be . or ..")]
    DotSegment,
    /// A segment starts with the reserved storage window marker.
    #[error("cohort segment must not start with window=")]
    ReservedWindowSegment,
    /// A segment contains a byte outside the path-safe cohort alphabet.
    #[error("cohort segment contains invalid characters")]
    InvalidSegment,
}

/// A valid Cohort name.
///
/// `CohortName` is constructed only after applying the VectorSeam cohort
/// grammar.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CohortName(String);

impl CohortName {
    /// Validates and stores an owned cohort name.
    pub fn new(name: String) -> Result<Self, CohortNameError> {
        validate_cohort_name(&name)?;
        Ok(Self(name))
    }

    /// Returns the validated cohort name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Converts the cohort name back into its owned string.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for CohortName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for CohortName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for CohortName {
    type Err = CohortNameError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::try_from(name)
    }
}

impl TryFrom<String> for CohortName {
    type Error = CohortNameError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::new(name)
    }
}

impl TryFrom<&str> for CohortName {
    type Error = CohortNameError;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        Self::new(name.to_owned())
    }
}

/// Validates a cohort name against the VectorSeam grammar.
///
/// The grammar allows 1 to 8 slash-separated ASCII path segments. Each segment
/// must be 1 to 63 bytes and contain only letters, digits, `.`, `_`, `-`, or
/// `=`. The exact segments `.` and `..` are rejected for local filesystem
/// safety, and segments starting with `window=` are reserved by the storage
/// layout.
pub fn validate_cohort_name(name: &str) -> Result<(), CohortNameError> {
    if name.is_empty() {
        return Err(CohortNameError::Empty);
    }
    if name.len() > MAX_COHORT_NAME_BYTES {
        return Err(CohortNameError::NameTooLong);
    }
    if !name.is_ascii() {
        return Err(CohortNameError::NonAscii);
    }

    let segments: Vec<&str> = name.split('/').collect();
    if segments.len() > MAX_COHORT_SEGMENTS {
        return Err(CohortNameError::TooManySegments);
    }

    for segment in segments {
        validate_segment(segment)?;
    }

    Ok(())
}

/// Returns `true` when `name` is a valid cohort name.
pub fn is_valid_cohort_name(name: &str) -> bool {
    validate_cohort_name(name).is_ok()
}

fn validate_segment(segment: &str) -> Result<(), CohortNameError> {
    if segment.is_empty() {
        return Err(CohortNameError::EmptySegment);
    }
    if segment.len() > MAX_COHORT_SEGMENT_BYTES {
        return Err(CohortNameError::SegmentTooLong);
    }
    if segment == "." || segment == ".." {
        return Err(CohortNameError::DotSegment);
    }
    if segment.starts_with("window=") {
        return Err(CohortNameError::ReservedWindowSegment);
    }

    if segment.bytes().all(is_allowed_segment_byte) {
        Ok(())
    } else {
        Err(CohortNameError::InvalidSegment)
    }
}

fn is_allowed_segment_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'=')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_names_matching_cohort_grammar() {
        let segment_63_bytes = "a".repeat(63);
        let pair_segment_63_bytes = format!("{}={}", "k".repeat(31), "v".repeat(31));
        let name_255_bytes = [
            "a".repeat(31),
            "b".repeat(31),
            "c".repeat(31),
            "d".repeat(31),
            "e".repeat(31),
            "f".repeat(31),
            "g".repeat(31),
            "h".repeat(31),
        ]
        .join("/");

        for name in [
            "prod",
            "prod/tenant-a/products",
            "a1/b_2/c-3",
            "prod.tenant",
            "env=prod",
            "env=prod/tenant=a/index=products",
            "env=Prod",
            "env=te.nant",
            "env==prod",
            "=prod",
            "env=",
            "a=b=c",
            "-prod",
            "_prod",
            "part=x",
            "cohorts=x",
            "prod/tenant=a",
            segment_63_bytes.as_str(),
            pair_segment_63_bytes.as_str(),
            name_255_bytes.as_str(),
        ] {
            assert!(is_valid_cohort_name(name), "{name}");
            assert_eq!(CohortName::try_from(name).unwrap().as_str(), name);
        }
    }

    #[test]
    fn rejects_names_outside_cohort_grammar() {
        let segment_64_bytes = "a".repeat(64);
        let pair_segment_64_bytes = format!("{}={}", "k".repeat(32), "v".repeat(31));
        let name_256_bytes = [
            "a".repeat(63),
            "b".repeat(63),
            "c".repeat(63),
            "d".repeat(62),
            "e".to_owned(),
        ]
        .join("/");

        for name in [
            "",
            "prod//tenant",
            "/prod",
            "prod/",
            "a/a/a/a/a/a/a/a/a",
            segment_64_bytes.as_str(),
            name_256_bytes.as_str(),
            "café",
            "window=x",
            "prod/window=x",
            ".",
            "..",
            "prod/.",
            "prod/..",
            "prod tenant",
            "prod+tenant",
            pair_segment_64_bytes.as_str(),
        ] {
            assert!(!is_valid_cohort_name(name), "{name}");
        }
    }
}
