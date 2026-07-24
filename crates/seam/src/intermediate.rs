//! Synchronous IO glue for durable Phase B intermediates.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, ListArray, RecordBatch, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::errors::ParquetError;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TruthMeasurement {
    pub(crate) record_index: i32,
    pub(crate) vector_hash: u64,
    pub(crate) dup_count: i32,
    pub(crate) receive_time_us: i64,
    pub(crate) latency_ms: f64,
    pub(crate) gt_keys: Vec<i64>,
    pub(crate) gt_distances: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SweepMeasurementRow {
    pub(crate) record_index: i32,
    pub(crate) ef: i32,
    pub(crate) returned_keys: Vec<i64>,
    pub(crate) recall: f64,
    pub(crate) latency_ms: f64,
    pub(crate) result_count: i32,
}

/// Reads and joins one durable truth/sweep parquet pair.
pub fn read_intermediate_pair(
    truth_path: &Path,
    sweep_path: &Path,
) -> Result<IntermediatePart, IntermediateError> {
    let truth = Bytes::from(std::fs::read(truth_path)?);
    let sweep = Bytes::from(std::fs::read(sweep_path)?);
    read_intermediate_pair_bytes(truth, sweep)
}

/// Reads and joins one durable truth/sweep parquet pair from object bytes.
pub fn read_intermediate_pair_bytes(
    truth: Bytes,
    sweep: Bytes,
) -> Result<IntermediatePart, IntermediateError> {
    let truth_metadata = read_metadata(truth.clone())?;
    let sweep_metadata = read_metadata(sweep.clone())?;
    if truth_metadata != sweep_metadata {
        return Err(IntermediateError::MetadataMismatch);
    }

    let mut samples = read_truth_rows(truth)?;
    if usize::try_from(truth_metadata.measured_count) != Ok(samples.len()) {
        return Err(IntermediateError::MeasuredCountMismatch {
            metadata_count: truth_metadata.measured_count,
            truth_row_count: samples.len(),
        });
    }
    read_sweep_rows(sweep, &mut samples)?;
    Ok(IntermediatePart {
        metadata: truth_metadata,
        samples: samples.into_values().collect(),
    })
}

pub(crate) fn encode_intermediate_pair(
    metadata: &IntermediateMetadata,
    truth_rows: &[TruthMeasurement],
    sweep_rows: &[SweepMeasurementRow],
) -> Result<(Bytes, Bytes), IntermediateError> {
    let truth = encode_parquet(
        truth_schema(),
        truth_record_batch(truth_rows)?,
        metadata_key_values(metadata),
    )?;
    let sweep = encode_parquet(
        sweep_schema(),
        sweep_record_batch(sweep_rows)?,
        metadata_key_values(metadata),
    )?;
    Ok((truth, sweep))
}

/// Frozen truth parquet Arrow schema.
fn truth_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("record_index", DataType::Int32, false),
        Field::new("vector_hash", DataType::UInt64, false),
        Field::new("dup_count", DataType::Int32, false),
        Field::new("receive_time_us", DataType::Int64, false),
        Field::new("latency_ms", DataType::Float64, false),
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

fn read_metadata(bytes: Bytes) -> Result<IntermediateMetadata, IntermediateError> {
    let reader = SerializedFileReader::new(bytes)?;
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

fn read_truth_rows(bytes: Bytes) -> Result<BTreeMap<i32, MeasuredSample>, IntermediateError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    if builder.schema().fields() != truth_schema().fields() {
        return Err(IntermediateError::Schema { kind: "truth" });
    }
    let mut samples = BTreeMap::new();
    for batch in builder.build()? {
        let batch = batch?;
        let record_indexes = int32_column(&batch, 0, "truth")?;
        let vector_hashes = uint64_column(&batch, 1, "truth")?;
        let dup_counts = int32_column(&batch, 2, "truth")?;
        let latencies = float64_column(&batch, 4, "truth")?;
        for row in 0..batch.num_rows() {
            let record_index = record_indexes.value(row);
            let sample = MeasuredSample {
                record_index,
                vector_hash: vector_hashes.value(row),
                dup_count: dup_counts.value(row),
                ground_truth_latency_ms: latencies.value(row),
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
    bytes: Bytes,
    samples: &mut BTreeMap<i32, MeasuredSample>,
) -> Result<(), IntermediateError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
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

fn metadata_key_values(metadata: &IntermediateMetadata) -> Vec<KeyValue> {
    let ef_grid = metadata
        .ef_grid
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    [
        ("format_version", metadata.format_version.to_string()),
        ("cohort", metadata.cohort.clone()),
        ("part_ulid", metadata.part_ulid.clone()),
        ("window_start", metadata.window_start.to_string()),
        ("window_seconds", metadata.window_seconds.to_string()),
        (
            "received_frame_count",
            metadata.received_frame_count.to_string(),
        ),
        ("record_count", metadata.record_count.to_string()),
        ("index", metadata.index.clone()),
        ("table", metadata.table.clone()),
        ("column", metadata.column.clone()),
        ("key", metadata.key.clone()),
        ("k", metadata.k.to_string()),
        ("ef_grid", ef_grid),
        ("failed_count", metadata.failed_count.to_string()),
        ("measured_count", metadata.measured_count.to_string()),
        ("computed_at_us", metadata.computed_at_us.to_string()),
    ]
    .into_iter()
    .map(|(key, value)| KeyValue::new(key.to_owned(), Some(value)))
    .collect()
}

fn truth_record_batch(rows: &[TruthMeasurement]) -> Result<RecordBatch, IntermediateError> {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.record_index),
        )),
        Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|row| row.vector_hash),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.dup_count),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.receive_time_us),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.latency_ms),
        )),
        Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
            rows.iter()
                .map(|row| Some(row.gt_keys.iter().copied().map(Some))),
        )),
        Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(
            rows.iter()
                .map(|row| Some(row.gt_distances.iter().copied().map(Some))),
        )),
    ];
    Ok(RecordBatch::try_new(truth_schema(), columns)?)
}

fn sweep_record_batch(rows: &[SweepMeasurementRow]) -> Result<RecordBatch, IntermediateError> {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.record_index),
        )),
        Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.ef))),
        Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
            rows.iter()
                .map(|row| Some(row.returned_keys.iter().copied().map(Some))),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.recall),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.latency_ms),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.result_count),
        )),
    ];
    Ok(RecordBatch::try_new(sweep_schema(), columns)?)
}

fn encode_parquet(
    schema: SchemaRef,
    batch: RecordBatch,
    metadata: Vec<KeyValue>,
) -> Result<Bytes, IntermediateError> {
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_key_value_metadata(Some(metadata))
        .build();
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, Some(properties))?;
    writer.write(&batch)?;
    Ok(Bytes::from(writer.into_inner()?))
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
