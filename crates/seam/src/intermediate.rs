//! Synchronous IO glue for durable Phase B intermediates.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, RecordBatch, UInt64Array};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::ParquetError;
use parquet::file::reader::{FileReader, SerializedFileReader};
use thiserror::Error;

use crate::model::{IntermediateMetadata, IntermediatePart, MeasuredSample, SweepMeasurement};

/// Invalid or unreadable durable intermediate data.
#[derive(Debug, Error)]
pub enum IntermediateError {
    /// A local file could not be opened.
    #[error("read intermediate file: {0}")]
    Io(#[from] std::io::Error),
    /// A parquet file could not be decoded.
    #[error("decode intermediate parquet: {0}")]
    Parquet(#[from] ParquetError),
    /// An Arrow record batch could not be decoded.
    #[error("decode intermediate Arrow batch: {0}")]
    Arrow(#[from] ArrowError),
    /// The durable Arrow schema did not exactly match the frozen schema.
    #[error("{kind} parquet schema does not match the frozen schema")]
    Schema {
        /// `truth` or `sweep`.
        kind: &'static str,
    },
    /// Required parquet key-value metadata was absent.
    #[error("missing parquet metadata key {0:?}")]
    MissingMetadata(String),
    /// A parquet metadata key occurred more than once.
    #[error("duplicate parquet metadata key {0:?}")]
    DuplicateMetadata(String),
    /// A parquet metadata value did not parse to its frozen type.
    #[error("invalid parquet metadata {key:?} value {value:?}")]
    InvalidMetadata {
        /// Metadata key.
        key: String,
        /// Stored value.
        value: String,
    },
    /// Truth and sweep metadata differed.
    #[error("truth and sweep parquet metadata do not match")]
    MetadataMismatch,
    /// A truth record ordinal occurred more than once.
    #[error("duplicate truth record_index {0}")]
    DuplicateTruthRecord(i32),
    /// Metadata measured count did not equal the actual truth-row count.
    #[error("parquet measured_count {metadata_count} does not match {truth_row_count} truth rows")]
    MeasuredCountMismatch {
        /// Metadata value.
        metadata_count: u64,
        /// Decoded truth-row count.
        truth_row_count: usize,
    },
    /// A sweep row had no truth row with the same record ordinal.
    #[error("sweep record_index {0} has no matching truth row")]
    OrphanSweepRecord(i32),
    /// A sweep ef occurred more than once for a record ordinal.
    #[error("duplicate sweep row for record_index {record_index}, ef {ef}")]
    DuplicateSweep {
        /// Truth record ordinal.
        record_index: i32,
        /// Duplicate ef value.
        ef: i32,
    },
}

/// Reads and joins one durable truth/sweep parquet pair.
pub fn read_intermediate_pair(
    truth_path: &Path,
    sweep_path: &Path,
) -> Result<IntermediatePart, IntermediateError> {
    let truth_metadata = read_metadata(truth_path)?;
    let sweep_metadata = read_metadata(sweep_path)?;
    if truth_metadata != sweep_metadata {
        return Err(IntermediateError::MetadataMismatch);
    }

    let mut samples = read_truth_rows(truth_path)?;
    if usize::try_from(truth_metadata.measured_count) != Ok(samples.len()) {
        return Err(IntermediateError::MeasuredCountMismatch {
            metadata_count: truth_metadata.measured_count,
            truth_row_count: samples.len(),
        });
    }
    read_sweep_rows(sweep_path, &mut samples)?;
    Ok(IntermediatePart {
        metadata: truth_metadata,
        samples: samples.into_values().collect(),
    })
}

/// Frozen truth parquet Arrow schema.
fn truth_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("record_index", DataType::Int32, false),
        Field::new("vector_hash", DataType::UInt64, false),
        Field::new("dup_count", DataType::Int32, false),
        Field::new("receive_time_us", DataType::Int64, false),
        Field::new(
            "gt_keys",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
        Field::new(
            "gt_distances",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ]))
}

/// Frozen sweep parquet Arrow schema.
fn sweep_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("record_index", DataType::Int32, false),
        Field::new("ef", DataType::Int32, false),
        Field::new(
            "returned_keys",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
        Field::new("recall", DataType::Float64, false),
        Field::new("latency_ms", DataType::Float64, false),
        Field::new("result_count", DataType::Int32, false),
    ]))
}

fn read_metadata(path: &Path) -> Result<IntermediateMetadata, IntermediateError> {
    let reader = SerializedFileReader::new(File::open(path)?)?;
    let entries = reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .map_or(&[][..], Vec::as_slice);
    let mut values = BTreeMap::<String, String>::new();
    for entry in entries {
        let Some(value) = &entry.value else {
            continue;
        };
        if values.insert(entry.key.clone(), value.clone()).is_some() {
            return Err(IntermediateError::DuplicateMetadata(entry.key.clone()));
        }
    }

    Ok(IntermediateMetadata {
        format_version: parse_metadata(&values, "format_version")?,
        cohort: metadata_value(&values, "cohort")?.to_owned(),
        part_ulid: metadata_value(&values, "part_ulid")?.to_owned(),
        window_start: parse_metadata(&values, "window_start")?,
        window_seconds: parse_metadata(&values, "window_seconds")?,
        received_frame_count: parse_metadata(&values, "received_frame_count")?,
        record_count: parse_metadata(&values, "record_count")?,
        index: metadata_value(&values, "index")?.to_owned(),
        table: metadata_value(&values, "table")?.to_owned(),
        column: metadata_value(&values, "column")?.to_owned(),
        key: metadata_value(&values, "key")?.to_owned(),
        k: parse_metadata(&values, "k")?,
        ef_grid: parse_ef_grid(metadata_value(&values, "ef_grid")?)?,
        failed_count: parse_metadata(&values, "failed_count")?,
        measured_count: parse_metadata(&values, "measured_count")?,
        computed_at_us: parse_metadata(&values, "computed_at_us")?,
    })
}

fn metadata_value<'a>(
    values: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, IntermediateError> {
    values
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| IntermediateError::MissingMetadata(key.to_owned()))
}

fn parse_metadata<T>(values: &BTreeMap<String, String>, key: &str) -> Result<T, IntermediateError>
where
    T: std::str::FromStr,
{
    let value = metadata_value(values, key)?;
    value
        .parse::<T>()
        .map_err(|_error| IntermediateError::InvalidMetadata {
            key: key.to_owned(),
            value: value.to_owned(),
        })
}

fn parse_ef_grid(value: &str) -> Result<Vec<i32>, IntermediateError> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|item| {
            item.parse::<i32>()
                .map_err(|_error| IntermediateError::InvalidMetadata {
                    key: "ef_grid".to_owned(),
                    value: value.to_owned(),
                })
        })
        .collect()
}

fn read_truth_rows(path: &Path) -> Result<BTreeMap<i32, MeasuredSample>, IntermediateError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    if builder.schema().fields() != truth_schema().fields() {
        return Err(IntermediateError::Schema { kind: "truth" });
    }
    let mut samples = BTreeMap::new();
    for batch in builder.build()? {
        let batch = batch?;
        let record_indexes = int32_column(&batch, 0, "truth")?;
        let vector_hashes = uint64_column(&batch, 1, "truth")?;
        let dup_counts = int32_column(&batch, 2, "truth")?;
        for row in 0..batch.num_rows() {
            let record_index = record_indexes.value(row);
            let sample = MeasuredSample {
                record_index,
                vector_hash: vector_hashes.value(row),
                dup_count: dup_counts.value(row),
                sweeps: BTreeMap::new(),
            };
            if samples.insert(record_index, sample).is_some() {
                return Err(IntermediateError::DuplicateTruthRecord(record_index));
            }
        }
    }
    Ok(samples)
}

fn read_sweep_rows(
    path: &Path,
    samples: &mut BTreeMap<i32, MeasuredSample>,
) -> Result<(), IntermediateError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    if builder.schema().fields() != sweep_schema().fields() {
        return Err(IntermediateError::Schema { kind: "sweep" });
    }
    for batch in builder.build()? {
        let batch = batch?;
        let record_indexes = int32_column(&batch, 0, "sweep")?;
        let ef_values = int32_column(&batch, 1, "sweep")?;
        let recalls = float64_column(&batch, 3, "sweep")?;
        let latencies = float64_column(&batch, 4, "sweep")?;
        for row in 0..batch.num_rows() {
            let record_index = record_indexes.value(row);
            let ef = ef_values.value(row);
            let sample = samples
                .get_mut(&record_index)
                .ok_or(IntermediateError::OrphanSweepRecord(record_index))?;
            if sample
                .sweeps
                .insert(
                    ef,
                    SweepMeasurement {
                        recall: recalls.value(row),
                        latency_ms: latencies.value(row),
                    },
                )
                .is_some()
            {
                return Err(IntermediateError::DuplicateSweep { record_index, ef });
            }
        }
    }
    Ok(())
}

fn int32_column<'a>(
    batch: &'a RecordBatch,
    index: usize,
    kind: &'static str,
) -> Result<&'a Int32Array, IntermediateError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or(IntermediateError::Schema { kind })
}

fn uint64_column<'a>(
    batch: &'a RecordBatch,
    index: usize,
    kind: &'static str,
) -> Result<&'a UInt64Array, IntermediateError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(IntermediateError::Schema { kind })
}

fn float64_column<'a>(
    batch: &'a RecordBatch,
    index: usize,
    kind: &'static str,
) -> Result<&'a Float64Array, IntermediateError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(IntermediateError::Schema { kind })
}
