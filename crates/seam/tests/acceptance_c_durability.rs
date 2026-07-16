mod support;

use std::collections::HashSet;

use seam::aggregate::{aggregate, round_json_bytes};
use seam::config::Config;
use seam::intermediate::read_intermediate_pair;
use seam::model::{
    AggregationConfig, AggregationInput, IntermediateMetadata, IntermediatePart, ListedPart,
    RoundStatus,
};
use support::f_agg::{
    DEFAULT_PART_ULID, DEFAULT_WINDOW_SECONDS, DEFAULT_WINDOW_START, PartMetadata, SweepRow,
    TruthRow, write_intermediate_pair,
};
use tempfile::tempdir;

#[derive(Debug)]
struct PendingRoundResult {
    status: String,
    measured: u64,
    error: Option<String>,
}

#[test]
#[ignore = "Stage 3: durable part resume is not implemented"]
fn c1_resume_mid_part_rewrites_pair_and_matches_clean_run() {
    let truth_was_rewritten = support::pending::<bool>("C1");
    let sweep_was_written = support::pending::<bool>("C1");
    let statements_for_incomplete_part_only = support::pending::<bool>("C1");
    let crashed_and_clean_rounds_equal = support::pending::<bool>("C1");
    assert!(truth_was_rewritten);
    assert!(sweep_was_written);
    assert!(statements_for_incomplete_part_only);
    assert!(crashed_and_clean_rounds_equal);
}

#[test]
#[ignore = "Stage 3: database degradation and publishing are not implemented"]
fn c2_database_down_still_publishes_cached_phase_b_and_exits_zero() {
    let published = support::pending::<PendingRoundResult>("C2");
    // The expected side is intentionally independent: it must count raw truth
    // parquet rows directly, never call the tuner's intermediate reader.
    let cached_measured = support::pending::<u64>("C2");
    let shutdown_exit_code = support::pending::<i32>("C2");
    assert_eq!(published.measured, cached_measured);
    assert_eq!(shutdown_exit_code, 0);
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
#[ignore = "Stage 3: incompatible-part remeasurement is not implemented"]
fn c3_config_fingerprint_k_change_remeasures_with_k_20() {
    let part_was_remeasured_with_k_20 = support::pending::<bool>("C3");
    assert!(part_was_remeasured_with_k_20);
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
        valid_yaml().replace("    table:", "    password: secret\n    table:"),
        valid_yaml().replace("server: localhost:5432", "server: user:pass@localhost:5432"),
        valid_yaml().replace("ef_search: [10, 20, 40]", "ef_search: [9, 20, 40]"),
        valid_yaml().replace("ef_search: [10, 20, 40]", "ef_search: [10, 20, 1001]"),
        valid_yaml().replace("ef_search: [10, 20, 40]", "ef_search: [10, 40, 20]"),
        valid_yaml().replace("index: fixture", "index: absent"),
        valid_yaml().replace("target: recall", "target: absent"),
        valid_yaml().replace("percentile: 0.95", "percentile: 1.0"),
        valid_yaml().replace("window: 1h", "window: 5min"),
    ];
    let errors = cases
        .iter()
        .map(|yaml| Config::from_yaml_str(yaml).unwrap_err().to_string())
        .collect::<Vec<_>>();

    assert_eq!(errors.len(), 9);
    assert_eq!(errors.iter().collect::<HashSet<_>>().len(), 9);
    assert!(errors[0].contains("password_env"));
    assert!(errors[1].contains("password_env"));
    assert!(errors[2].contains("ef_search"));
    assert!(errors[3].contains("ef_search"));
    assert!(errors[4].contains("ef_search"));
    assert!(errors[5].contains("unknown"));
    assert!(errors[6].contains("unknown"));
    assert!(errors[7].contains("percentile"));
    assert!(errors[8].contains("window"));
}

#[test]
#[ignore = "Stage 3: table-size abort is not implemented"]
fn c6_table_smaller_than_k_aborts_one_cohort_after_exact_scan() {
    let affected = support::pending::<PendingRoundResult>("C6");
    let exact_scan_count = support::pending::<u64>("C6");
    let sweep_statement_count = support::pending::<u64>("C6");
    let other_cohort_completed = support::pending::<bool>("C6");
    assert_eq!(affected.status, "insufficient_samples");
    assert!(affected.error.is_some());
    assert!(exact_scan_count <= 1);
    assert_eq!(sweep_statement_count, 0);
    assert!(other_cohort_completed);
}

#[test]
fn c8_phase_b_reproducible_except_computed_at() {
    let root = tempdir().unwrap();
    let metadata = PartMetadata::default();
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
            recall: f64::from(*ef) / 160.0,
            latency_ms: f64::from(*ef) / 10.0,
            result_count: 10,
        })
        .collect::<Vec<_>>();
    let pair = write_intermediate_pair(root.path(), &metadata, &[truth], &sweeps).unwrap();
    let intermediate = read_intermediate_pair(&pair.truth_path, &pair.sweep_path).unwrap();

    let mut first_input = aggregation_input(vec![intermediate.clone()]);
    first_input.computed_at = "2026-07-08T12:10:01Z".to_owned();
    let mut second_input = first_input.clone();
    second_input.computed_at = "2026-07-08T12:10:02Z".to_owned();
    second_input
        .listed_parts
        .push(second_input.listed_parts[0].clone());

    let first = round_json_bytes(&aggregate(&first_input).unwrap()).unwrap();
    let second = round_json_bytes(&aggregate(&second_input).unwrap()).unwrap();
    let mut first_json: serde_json::Value = serde_json::from_slice(&first).unwrap();
    let mut second_json: serde_json::Value = serde_json::from_slice(&second).unwrap();
    first_json["computed_at"] = serde_json::Value::Null;
    second_json["computed_at"] = serde_json::Value::Null;

    assert_eq!(
        serde_json::to_vec(&first_json).unwrap(),
        serde_json::to_vec(&second_json).unwrap()
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
indexes:
  fixture:
    server: localhost:5432
    database: postgres
    user: postgres
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
        error: None,
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
