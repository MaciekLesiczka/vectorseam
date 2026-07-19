mod support;

use std::collections::HashSet;

use seam::aggregate::{aggregate, round_json_bytes};
use seam::config::Config;
use seam::intermediate::read_intermediate_pair;
use seam::model::{
    AggregationConfig, AggregationInput, IntermediateMetadata, IntermediatePart, ListedPart,
    MeasuredSample, PhaseAAbort, RoundStatus, SweepMeasurement,
};
use seam::tuner::Tuner;
use support::f_agg::{
    DEFAULT_PART_ULID, DEFAULT_WINDOW_SECONDS, DEFAULT_WINDOW_START, PartMetadata, SweepRow,
    TruthRow, independently_count_truth_rows, write_intermediate_pair, write_segment_fixture,
};
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn c2_database_down_still_publishes_cached_phase_b_and_exits_zero() {
    let root = tempdir().unwrap();
    let metadata = PartMetadata {
        ef_grid: vec![10, 20, 40],
        ..PartMetadata::default()
    };
    write_segment_fixture(
        root.path(),
        &metadata,
        1,
        &[(DEFAULT_WINDOW_START * 1_000_000, vec![0.25, 0.5])],
    )
    .unwrap();
    let truth = TruthRow {
        record_index: 0,
        vector_hash: 0xaf63_dc4c_8601_ec8c,
        dup_count: 1,
        receive_time_us: 1_783_512_000_000_000,
        gt_keys: (1..=10).collect(),
        gt_distances: (0..10).map(|value| f64::from(value) / 100.0).collect(),
    };
    let sweeps = metadata
        .ef_grid
        .iter()
        .map(|ef| SweepRow {
            record_index: 0,
            ef: *ef,
            returned_keys: (1..=10).collect(),
            recall: 1.0,
            latency_ms: 0.5,
            result_count: 10,
        })
        .collect::<Vec<_>>();
    let pair = write_intermediate_pair(root.path(), &metadata, &[truth], &sweeps).unwrap();
    // This expected side reads raw parquet metadata independently and never
    // calls the tuner's intermediate reader.
    let cached_measured =
        independently_count_truth_rows(std::slice::from_ref(&pair.truth_path)).unwrap();

    let yaml = valid_yaml()
        .replace("/tmp/vectorseam", root.path().to_str().unwrap())
        .replace("localhost:5432", "127.0.0.1:1");
    let config = Config::from_yaml_str(&yaml).unwrap();
    let mut tuner = Tuner::start(config).await.unwrap();
    let report = tuner
        .run_round(
            DEFAULT_WINDOW_START + u64::from(DEFAULT_WINDOW_SECONDS),
            "2026-07-08T12:10:00Z".to_owned(),
            1_783_512_600_000_000,
            &CancellationToken::new(),
        )
        .await;

    let published = &report.published["acceptance/f-agg"];
    assert_eq!(published.samples.measured, cached_measured);
    assert!(report.failed_cohorts.is_empty());
    assert!(!report.cancelled);
    tuner.shutdown().await;
}

#[test]
fn c3_config_fingerprint_k_change_ignores_incompatible_intermediate() {
    let mut input = aggregation_input(Vec::new());
    input.config.k = 20;
    input.config.ef_grid = vec![20, 40];
    input.intermediates = vec![IntermediatePart {
        metadata: IntermediateMetadata {
            k: 10,
            ef_grid: vec![20, 40],
            ..metadata(1)
        },
        samples: Vec::new(),
    }];

    let observed = aggregate(&input).unwrap();

    assert_eq!(observed.incompatible_parts, 1);
    assert_eq!(observed.parts_used, 0);
    assert_eq!(observed.samples.measured, 0);
}

#[test]
fn c4_empty_round_reports_insufficient_samples_and_full_gap() {
    let mut input = aggregation_input(Vec::new());
    input.listed_parts.clear();

    let observed = aggregate(&input).unwrap();

    assert_eq!(observed.status, RoundStatus::InsufficientSamples);
    assert_eq!(observed.samples.available, 0);
    assert_eq!(observed.coverage.empty_window_fraction, 1.0);
    assert_eq!(observed.recommended_ef, None);
    assert_eq!(observed.confidence, None);
    assert_eq!(observed.transferred, None);
    assert_eq!(observed.train_quantile_recall, None);
    assert_eq!(observed.test_quantile_recall, None);
    assert_eq!(observed.per_ef, []);
    assert_eq!(observed.dropped_frame_fraction, 0.0);
}

#[test]
fn c5_config_validation_distinct_errors_and_password_env_guidance() {
    let cases = [
        valid_yaml().replace("    server:", "    password: secret\n    server:"),
        valid_yaml().replace("server: localhost:5432", "server: user:pass@localhost:5432"),
        valid_yaml().replace("data_source: primary", "data_source: absent"),
        valid_yaml().replace("ef_search: [10, 20, 40]", "ef_search: [9, 20, 40]"),
        valid_yaml().replace("ef_search: [10, 20, 40]", "ef_search: [10, 20, 1001]"),
        valid_yaml().replace("ef_search: [10, 20, 40]", "ef_search: [10, 40, 20]"),
        valid_yaml().replace("index: fixture", "index: absent"),
        valid_yaml().replace("target: recall", "target: absent"),
        valid_yaml().replace("percentile: 0.95", "percentile: 1.0"),
        valid_yaml().replace("window: 1h", "window: 5min"),
        valid_yaml().replace("storage:", "budget:\n  client_timeout: 0s\nstorage:"),
    ];
    let errors = cases
        .iter()
        .map(|yaml| Config::from_yaml_str(yaml).unwrap_err().to_string())
        .collect::<Vec<_>>();

    assert_eq!(errors.len(), 11);
    assert_eq!(errors.iter().collect::<HashSet<_>>().len(), 11);
    assert!(errors[0].contains("password_env"));
    assert!(errors[1].contains("password_env"));
    assert!(errors[2].contains("unknown data source"));
    assert!(errors[3].contains("ef_search"));
    assert!(errors[4].contains("ef_search"));
    assert!(errors[5].contains("ef_search"));
    assert!(errors[6].contains("unknown"));
    assert!(errors[7].contains("unknown"));
    assert!(errors[8].contains("percentile"));
    assert!(errors[9].contains("window"));
    assert!(errors[10].contains("client_timeout"));
}

#[test]
fn c6_phase_a_abort_forces_insufficient_despite_cached_min_samples() {
    let mut input = aggregation_input(vec![IntermediatePart {
        metadata: metadata(100),
        samples: measured_samples(100),
    }]);
    input.listed_parts[0].received_frame_count = 100;
    input.listed_parts[0].record_count = 100;
    input.phase_a_abort = Some(PhaseAAbort::TableSmallerThanK {
        error: "table has fewer than k visible rows".to_owned(),
    });

    let observed = aggregate(&input).unwrap();

    assert_eq!(observed.samples.unique, 100);
    assert_eq!(observed.status, RoundStatus::InsufficientSamples);
    assert_eq!(
        observed.error.as_deref(),
        Some("table has fewer than k visible rows")
    );
    assert_eq!(observed.recommended_ef, None);
    assert_eq!(observed.confidence, None);
    assert_eq!(observed.transferred, None);
}

#[test]
fn c8_phase_b_reproducible_except_computed_at() {
    let root = tempdir().unwrap();
    let fixture_metadata = PartMetadata::default();
    let truth = TruthRow {
        record_index: 0,
        vector_hash: 0xaf63_dc4c_8601_ec8c,
        dup_count: 1,
        receive_time_us: 1_783_512_000_000_000,
        gt_keys: (1..=10).collect(),
        gt_distances: (0..10).map(|value| f64::from(value) / 100.0).collect(),
    };
    let sweeps = fixture_metadata
        .ef_grid
        .iter()
        .map(|ef| SweepRow {
            record_index: 0,
            ef: *ef,
            returned_keys: (1..=10).collect(),
            recall: f64::from(*ef) / 160.0,
            latency_ms: f64::from(*ef) / 10.0,
            result_count: 10,
        })
        .collect::<Vec<_>>();
    let pair = write_intermediate_pair(root.path(), &fixture_metadata, &[truth], &sweeps).unwrap();
    let intermediate = read_intermediate_pair(&pair.truth_path, &pair.sweep_path).unwrap();

    let mut first_input = aggregation_input(vec![intermediate.clone()]);
    first_input.computed_at = "2026-07-08T12:10:01Z".to_owned();
    let previous = aggregate(&aggregation_input(vec![IntermediatePart {
        metadata: metadata(100),
        samples: measured_samples(100),
    }]))
    .unwrap();
    assert!(previous.effective.is_some());
    first_input.previous_round = Some(previous);
    let mut second_input = first_input.clone();
    second_input.computed_at = "2026-07-08T12:10:02Z".to_owned();
    second_input
        .listed_parts
        .push(second_input.listed_parts[0].clone());

    let mut first = aggregate(&first_input).unwrap();
    let mut second = aggregate(&second_input).unwrap();
    first.computed_at = "normalized".to_owned();
    second.computed_at = "normalized".to_owned();

    assert_eq!(
        round_json_bytes(&first).unwrap(),
        round_json_bytes(&second).unwrap()
    );
}

fn valid_yaml() -> String {
    r#"
calibration:
  interval: 10min
  ef_search: [10, 20, 40]
storage:
  root: /tmp/vectorseam
  window_seconds: 600
data_sources:
  primary:
    server: localhost:5432
    database: postgres
    user: postgres
indexes:
  fixture:
    data_source: primary
    table: docs_fixture
    key: doc_id
    column: embedding
targets:
  recall:
    k: 10
    value: 0.9
    percentile: 0.95
    window: 1h
cohorts:
  acceptance/f-agg:
    index: fixture
    target: recall
"#
    .to_owned()
}

fn aggregation_input(intermediates: Vec<IntermediatePart>) -> AggregationInput {
    AggregationInput {
        config: AggregationConfig {
            cohort: "acceptance/f-agg".to_owned(),
            target_name: "recall".to_owned(),
            index: "fixture".to_owned(),
            table: "docs_fixture".to_owned(),
            column: "embedding".to_owned(),
            key: "doc_id".to_owned(),
            k: 10,
            value: 0.9,
            percentile: 0.95,
            window_duration_seconds: 600,
            storage_window_seconds: 600,
            ef_grid: vec![10, 20, 40, 80, 160],
            train_fraction: 0.7,
            split_seed: 7,
            min_samples: 100,
        },
        round_end: DEFAULT_WINDOW_START + u64::from(DEFAULT_WINDOW_SECONDS),
        computed_at: "2026-07-08T12:10:00Z".to_owned(),
        phase_a_abort: None,
        phase_a_incompatible_parts: 0,
        previous_round: None,
        listed_parts: vec![ListedPart {
            part_ulid: DEFAULT_PART_ULID.to_owned(),
            window_start: DEFAULT_WINDOW_START,
            window_seconds: DEFAULT_WINDOW_SECONDS,
            received_frame_count: 1,
            record_count: 1,
        }],
        intermediates,
    }
}

fn metadata(measured_count: u64) -> IntermediateMetadata {
    IntermediateMetadata {
        format_version: 1,
        cohort: "acceptance/f-agg".to_owned(),
        part_ulid: DEFAULT_PART_ULID.to_owned(),
        window_start: DEFAULT_WINDOW_START,
        window_seconds: DEFAULT_WINDOW_SECONDS,
        received_frame_count: measured_count,
        record_count: measured_count,
        index: "fixture".to_owned(),
        table: "docs_fixture".to_owned(),
        column: "embedding".to_owned(),
        key: "doc_id".to_owned(),
        k: 10,
        ef_grid: vec![10, 20, 40, 80, 160],
        failed_count: 0,
        measured_count,
        computed_at_us: 1_783_512_000_000_000,
    }
}

fn measured_samples(count: usize) -> Vec<MeasuredSample> {
    (0..count)
        .map(|record_index| MeasuredSample {
            record_index: i32::try_from(record_index).unwrap(),
            vector_hash: record_index as u64,
            dup_count: 1,
            sweeps: [10, 20, 40, 80, 160]
                .into_iter()
                .map(|ef| {
                    (
                        ef,
                        SweepMeasurement {
                            recall: 1.0,
                            latency_ms: 1.0,
                        },
                    )
                })
                .collect(),
        })
        .collect()
}
