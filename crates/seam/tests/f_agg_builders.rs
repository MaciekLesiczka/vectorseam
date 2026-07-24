mod support;

use std::collections::HashMap;
use std::fs::File;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::reader::{FileReader, SerializedFileReader};
use seam::intermediate::{IntermediateError, read_intermediate_pair};
use support::f_agg::{
    B12_SECOND_PART_ULID, PartMetadata, SweepRow, TruthRow, independently_count_truth_rows,
    sweep_schema, truth_schema, write_b12_cross_part_fixture, write_intermediate_pair,
    write_segment_fixture, write_truth_intermediate,
};
use tempfile::tempdir;
use vectorseam_core::segment::read_segment;

#[test]
fn f_agg_builders_emit_spec_schemas_and_metadata() {
    let root = tempdir().unwrap();
    let metadata = PartMetadata::default();
    let truth = TruthRow {
        record_index: 0,
        vector_hash: 0xaf63_dc4c_8601_ec8c,
        dup_count: 1,
        receive_time_us: 1_783_512_000_000_000,
        latency_ms: 400.5,
        gt_keys: (1..=10).collect(),
        gt_distances: (0..10).map(|value| f64::from(value) / 100.0).collect(),
    };
    let sweep = SweepRow {
        record_index: 0,
        ef: 10,
        returned_keys: vec![1, 2, 3, 11, 12, 13, 14, 15, 16, 17],
        recall: 0.3,
        latency_ms: 0.5,
        result_count: 10,
    };

    let pair = write_intermediate_pair(root.path(), &metadata, &[truth], &[sweep]).unwrap();

    let truth_reader =
        ParquetRecordBatchReaderBuilder::try_new(File::open(&pair.truth_path).unwrap()).unwrap();
    let sweep_reader =
        ParquetRecordBatchReaderBuilder::try_new(File::open(&pair.sweep_path).unwrap()).unwrap();
    assert_eq!(truth_reader.schema().fields(), truth_schema().fields());
    assert_eq!(sweep_reader.schema().fields(), sweep_schema().fields());

    let file_reader = SerializedFileReader::new(File::open(&pair.truth_path).unwrap()).unwrap();
    assert!(
        file_reader
            .metadata()
            .row_groups()
            .iter()
            .flat_map(|row_group| row_group.columns())
            .all(|column| matches!(column.compression(), Compression::ZSTD(_)))
    );
    let key_values = file_reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .as_ref()
        .unwrap()
        .iter()
        .map(|entry| {
            (
                entry.key.as_str(),
                entry.value.as_deref().unwrap_or_default(),
            )
        })
        .collect::<HashMap<_, _>>();
    assert_eq!(key_values["format_version"], "1");
    assert_eq!(key_values["cohort"], metadata.cohort);
    assert_eq!(key_values["ef_grid"], "10,20,40,80,160");
    assert_eq!(key_values["measured_count"], "1");
}

#[test]
fn f_agg_builder_can_create_truth_without_sweep_for_crash_resume() {
    let root = tempdir().unwrap();
    let metadata = PartMetadata::default();
    let truth = TruthRow {
        record_index: 0,
        vector_hash: 1,
        dup_count: 1,
        receive_time_us: 1_783_512_000_000_000,
        latency_ms: 400.5,
        gt_keys: (1..=10).collect(),
        gt_distances: vec![0.0; 10],
    };

    let truth_path = write_truth_intermediate(root.path(), &metadata, &[truth]).unwrap();
    let sweep_path =
        truth_path.with_file_name(format!("part-{}.sweep.parquet", metadata.part_ulid));

    assert!(truth_path.exists());
    assert!(!sweep_path.exists());
}

#[test]
fn f_agg_b12_second_part_ulid_is_lexicographically_greater() {
    let root = tempdir().unwrap();

    let fixture = write_b12_cross_part_fixture(root.path()).unwrap();

    assert_eq!(B12_SECOND_PART_ULID, "01J00000000000000000000001");
    assert!(
        fixture
            .first
            .truth_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            < fixture
                .second
                .truth_path
                .file_name()
                .unwrap()
                .to_string_lossy()
    );
    assert_eq!(fixture.vector_hash, 0xaf63_dc4c_8601_ec8c);
    assert_eq!(
        independently_count_truth_rows(&[fixture.first.truth_path, fixture.second.truth_path,])
            .unwrap(),
        2
    );
}

#[test]
fn f_agg_builder_segment_header_round_trips_through_core() {
    let root = tempdir().unwrap();
    let metadata = PartMetadata {
        received_frame_count: 100,
        record_count: 80,
        ..PartMetadata::default()
    };
    let records = vec![
        (1_783_512_000_000_000, vec![1.0, 0.0]),
        (1_783_512_000_000_001, vec![0.0, 1.0]),
    ];

    let path = write_segment_fixture(root.path(), &metadata, 100, &records).unwrap();
    let segment = read_segment(&std::fs::read(path).unwrap()).unwrap();

    assert_eq!(segment.header.window_start, metadata.window_start);
    assert_eq!(segment.header.window_seconds, 600);
    assert_eq!(segment.header.received_frame_count, 100);
    assert_eq!(segment.header.record_count, 2);
    assert_eq!(segment.records.len(), 2);
}

#[test]
fn f_agg_reader_rejects_measured_count_different_from_truth_rows() {
    let root = tempdir().unwrap();
    let metadata = PartMetadata {
        measured_count: 2,
        ..PartMetadata::default()
    };
    let truth = TruthRow {
        record_index: 0,
        vector_hash: 1,
        dup_count: 1,
        receive_time_us: 1_783_512_000_000_000,
        latency_ms: 400.5,
        gt_keys: (1..=10).collect(),
        gt_distances: vec![0.0; 10],
    };
    let sweep = SweepRow {
        record_index: 0,
        ef: 10,
        returned_keys: (1..=10).collect(),
        recall: 1.0,
        latency_ms: 0.5,
        result_count: 10,
    };
    let pair = write_intermediate_pair(root.path(), &metadata, &[truth], &[sweep]).unwrap();

    let error = read_intermediate_pair(&pair.truth_path, &pair.sweep_path).unwrap_err();

    assert!(matches!(
        error,
        IntermediateError::MeasuredCountMismatch {
            metadata_count: 2,
            truth_row_count: 1,
        }
    ));
}
