use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, ListArray, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use ulid::Ulid;
use vectorseam_core::cohort::CohortName;
use vectorseam_core::frame::{FIXED_FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_VERSION};
use vectorseam_core::segment::{SegmentHeader, SegmentRecordRef, write_segment};
use vectorseam_core::window::format_window_timestamp;

pub const DEFAULT_COHORT: &str = "acceptance/f-agg";
pub const DEFAULT_PART_ULID: &str = "01J00000000000000000000000";
pub const B12_SECOND_PART_ULID: &str = "01J00000000000000000000001";
pub const DEFAULT_WINDOW_START: u64 = 1_783_512_000;
pub const DEFAULT_WINDOW_SECONDS: u32 = 600;

#[derive(Clone, Debug)]
pub struct PartMetadata {
    pub cohort: String,
    pub part_ulid: String,
    pub window_start: u64,
    pub window_seconds: u32,
    pub received_frame_count: u64,
    pub record_count: u64,
    pub index: String,
    pub table: String,
    pub column: String,
    pub key: String,
    pub k: u32,
    pub ef_grid: Vec<i32>,
    pub failed_count: u64,
    pub measured_count: u64,
    pub computed_at_us: u64,
}

impl Default for PartMetadata {
    fn default() -> Self {
        Self {
            cohort: DEFAULT_COHORT.to_owned(),
            part_ulid: DEFAULT_PART_ULID.to_owned(),
            window_start: DEFAULT_WINDOW_START,
            window_seconds: DEFAULT_WINDOW_SECONDS,
            received_frame_count: 1,
            record_count: 1,
            index: "fixture".to_owned(),
            table: "docs_fixture".to_owned(),
            column: "embedding".to_owned(),
            key: "doc_id".to_owned(),
            k: 10,
            ef_grid: vec![10, 20, 40, 80, 160],
            failed_count: 0,
            measured_count: 1,
            computed_at_us: 1_783_512_000_000_000,
        }
    }
}

impl PartMetadata {
    fn parquet_key_values(&self) -> Vec<KeyValue> {
        let ef_grid = self
            .ef_grid
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        [
            ("format_version", "1".to_owned()),
            ("cohort", self.cohort.clone()),
            ("part_ulid", self.part_ulid.clone()),
            ("window_start", self.window_start.to_string()),
            ("window_seconds", self.window_seconds.to_string()),
            (
                "received_frame_count",
                self.received_frame_count.to_string(),
            ),
            ("record_count", self.record_count.to_string()),
            ("index", self.index.clone()),
            ("table", self.table.clone()),
            ("column", self.column.clone()),
            ("key", self.key.clone()),
            ("k", self.k.to_string()),
            ("ef_grid", ef_grid),
            ("failed_count", self.failed_count.to_string()),
            ("measured_count", self.measured_count.to_string()),
            ("computed_at_us", self.computed_at_us.to_string()),
        ]
        .into_iter()
        .map(|(key, value)| KeyValue::new(key.to_owned(), Some(value)))
        .collect()
    }
}

#[derive(Clone, Debug)]
pub struct TruthRow {
    pub record_index: i32,
    pub vector_hash: u64,
    pub dup_count: i32,
    pub receive_time_us: i64,
    pub gt_keys: Vec<i64>,
    pub gt_distances: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct SweepRow {
    pub record_index: i32,
    pub ef: i32,
    pub returned_keys: Vec<i64>,
    pub recall: f64,
    pub latency_ms: f64,
    pub result_count: i32,
}

#[derive(Debug)]
pub struct IntermediatePair {
    pub truth_path: PathBuf,
    pub sweep_path: PathBuf,
}

#[derive(Debug)]
pub struct B12Fixture {
    pub first: IntermediatePair,
    pub second: IntermediatePair,
    pub vector_hash: u64,
}

pub fn write_intermediate_pair(
    root: &Path,
    metadata: &PartMetadata,
    truth_rows: &[TruthRow],
    sweep_rows: &[SweepRow],
) -> Result<IntermediatePair> {
    let truth_path = write_truth_intermediate(root, metadata, truth_rows)?;
    let sweep_path = write_sweep_intermediate(root, metadata, sweep_rows)?;
    Ok(IntermediatePair {
        truth_path,
        sweep_path,
    })
}

pub fn write_b12_cross_part_fixture(root: &Path) -> Result<B12Fixture> {
    let vector_hash = 0xaf63_dc4c_8601_ec8c;
    let first_metadata = PartMetadata {
        received_frame_count: 3,
        record_count: 3,
        ..PartMetadata::default()
    };
    let second_metadata = PartMetadata {
        part_ulid: B12_SECOND_PART_ULID.to_owned(),
        ..PartMetadata::default()
    };
    assert!(first_metadata.part_ulid < second_metadata.part_ulid);

    let truth = |record_index, dup_count| TruthRow {
        record_index,
        vector_hash,
        dup_count,
        receive_time_us: 1_783_512_000_000_000 + i64::from(record_index),
        gt_keys: (1..=10).collect(),
        gt_distances: (0..10).map(|value| f64::from(value) / 100.0).collect(),
    };
    let sweep = |record_index| {
        first_metadata
            .ef_grid
            .iter()
            .map(|ef| SweepRow {
                record_index,
                ef: *ef,
                returned_keys: (1..=10).collect(),
                recall: 1.0,
                latency_ms: 0.5,
                result_count: 10,
            })
            .collect::<Vec<_>>()
    };
    let first = write_intermediate_pair(root, &first_metadata, &[truth(3, 3)], &sweep(3))?;
    let second = write_intermediate_pair(root, &second_metadata, &[truth(0, 1)], &sweep(0))?;
    Ok(B12Fixture {
        first,
        second,
        vector_hash,
    })
}

pub fn independently_count_truth_rows(paths: &[PathBuf]) -> Result<u64> {
    paths.iter().try_fold(0_u64, |total, path| {
        let reader = SerializedFileReader::new(
            File::open(path).with_context(|| format!("open {}", path.display()))?,
        )?;
        let rows = u64::try_from(reader.metadata().file_metadata().num_rows())?;
        total.checked_add(rows).context("truth row count overflow")
    })
}

pub fn write_truth_intermediate(
    root: &Path,
    metadata: &PartMetadata,
    rows: &[TruthRow],
) -> Result<PathBuf> {
    let path = intermediate_path(root, metadata, "truth.parquet")?;
    write_parquet(&path, truth_schema(), truth_batch(rows)?, metadata)?;
    Ok(path)
}

pub fn write_sweep_intermediate(
    root: &Path,
    metadata: &PartMetadata,
    rows: &[SweepRow],
) -> Result<PathBuf> {
    let path = intermediate_path(root, metadata, "sweep.parquet")?;
    write_parquet(&path, sweep_schema(), sweep_batch(rows)?, metadata)?;
    Ok(path)
}

pub fn write_segment_fixture(
    root: &Path,
    metadata: &PartMetadata,
    received_frame_count: u64,
    records: &[(u64, Vec<f32>)],
) -> Result<PathBuf> {
    let cohort = CohortName::try_from(metadata.cohort.as_str())?;
    let frames = records
        .iter()
        .map(|(_receive_time, vector)| encode_f32_frame(&cohort, vector))
        .collect::<Result<Vec<_>>>()?;
    let segment_records = records
        .iter()
        .zip(&frames)
        .map(|((receive_time, _vector), frame)| SegmentRecordRef {
            receive_time: *receive_time,
            frame,
        })
        .collect::<Vec<_>>();
    let header = SegmentHeader {
        window_start: metadata.window_start,
        window_seconds: metadata.window_seconds,
        first_receive: records.first().map_or(0, |record| record.0),
        last_receive: records.last().map_or(0, |record| record.0),
        received_frame_count,
        record_count: u64::try_from(records.len())?,
        cohort,
    };
    let bytes = write_segment(&header, &segment_records)?;
    let timestamp = format_window_timestamp(metadata.window_start)?;
    let directory = root
        .join("cohorts")
        .join(&metadata.cohort)
        .join(format!("window={timestamp}"));
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create fixture directory {}", directory.display()))?;
    let path = directory.join(format!("part-{}.vseam", metadata.part_ulid));
    std::fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

pub fn truth_schema() -> SchemaRef {
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

pub fn sweep_schema() -> SchemaRef {
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

fn measurement_directory(root: &Path, metadata: &PartMetadata) -> Result<PathBuf> {
    let timestamp = format_window_timestamp(metadata.window_start)?;
    Ok(root
        .join("measurements")
        .join(&metadata.cohort)
        .join(format!("window={timestamp}")))
}

fn intermediate_path(root: &Path, metadata: &PartMetadata, suffix: &str) -> Result<PathBuf> {
    let directory = measurement_directory(root, metadata)?;
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create fixture directory {}", directory.display()))?;
    Ok(directory.join(format!("part-{}.{}", metadata.part_ulid, suffix)))
}

fn truth_batch(rows: &[TruthRow]) -> Result<RecordBatch> {
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

fn sweep_batch(rows: &[SweepRow]) -> Result<RecordBatch> {
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

fn write_parquet(
    path: &Path,
    schema: SchemaRef,
    batch: RecordBatch,
    metadata: &PartMetadata,
) -> Result<()> {
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_key_value_metadata(Some(metadata.parquet_key_values()))
        .build();
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn encode_f32_frame(cohort: &CohortName, vector: &[f32]) -> Result<Vec<u8>> {
    let name = cohort.as_str().as_bytes();
    let vector_len = vector
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .context("fixture vector byte length overflow")?;
    let total_len = FIXED_FRAME_HEADER_LEN
        .checked_add(name.len())
        .and_then(|value| value.checked_add(vector_len))
        .context("fixture frame length overflow")?;
    let frame_len = u32::try_from(total_len.checked_sub(4).context("invalid frame length")?)?;
    let mut frame = Vec::with_capacity(total_len);
    frame.extend_from_slice(&frame_len.to_le_bytes());
    frame.extend_from_slice(&FRAME_MAGIC);
    frame.extend_from_slice(&FRAME_VERSION.to_le_bytes());
    frame.extend_from_slice(&1_u32.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(name.len())?.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(vector.len())?.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(vector_len)?.to_le_bytes());
    frame.extend_from_slice(name);
    for value in vector {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    Ok(frame)
}

pub fn metadata_with_part(part_ulid: &str, window_start: u64) -> PartMetadata {
    let _: Ulid = part_ulid
        .parse()
        .expect("test fixture part ULID must be valid");
    PartMetadata {
        part_ulid: part_ulid.to_owned(),
        window_start,
        ..PartMetadata::default()
    }
}
