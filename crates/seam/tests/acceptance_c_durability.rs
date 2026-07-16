mod support;

use std::collections::HashSet;

use support::pending;

#[derive(Debug)]
struct RoundResult {
    status: String,
    recommended_ef: Option<i32>,
    confidence: Option<f64>,
    available: u64,
    measured: u64,
    incompatible_parts: u64,
    empty_window_fraction: f64,
    error: Option<String>,
}

#[test]
#[ignore = "Stage 3: durable part resume is not implemented"]
fn c1_resume_mid_part_rewrites_pair_and_matches_clean_run() {
    let truth_was_rewritten = pending::<bool>("C1");
    let sweep_was_written = pending::<bool>("C1");
    let statements_for_incomplete_part_only = pending::<bool>("C1");
    let crashed_and_clean_rounds_equal = pending::<bool>("C1");
    assert!(truth_was_rewritten);
    assert!(sweep_was_written);
    assert!(statements_for_incomplete_part_only);
    assert!(crashed_and_clean_rounds_equal);
}

#[test]
#[ignore = "Stage 3: database degradation and publishing are not implemented"]
fn c2_database_down_still_publishes_cached_phase_b_and_exits_zero() {
    let published = pending::<RoundResult>("C2");
    // The implemented test must obtain this expected count by independently
    // counting raw truth-parquet rows, never through the tuner's reader.
    let cached_measured = pending::<u64>("C2");
    let shutdown_exit_code = pending::<i32>("C2");
    assert_eq!(published.measured, cached_measured);
    assert_eq!(shutdown_exit_code, 0);
}

#[test]
#[ignore = "Stages 2 and 3: compatibility and remeasurement are not implemented"]
fn c3_config_fingerprint_k_change_ignores_and_remeasures() {
    let original_k = 10_u32;
    let changed_k = 20_u32;
    let observed = pending::<RoundResult>("C3");
    let part_was_remeasured_with_k_20 = pending::<bool>("C3");
    assert_ne!(original_k, changed_k);
    assert!(observed.incompatible_parts > 0);
    assert!(part_was_remeasured_with_k_20);
}

#[test]
#[ignore = "Stage 2: empty aggregation is not implemented"]
fn c4_empty_round_reports_insufficient_samples_and_full_gap() {
    let observed = pending::<RoundResult>("C4");
    assert_eq!(observed.status, "insufficient_samples");
    assert_eq!(observed.available, 0);
    assert_eq!(observed.empty_window_fraction, 1.0);
    assert_eq!(observed.recommended_ef, None);
    assert_eq!(observed.confidence, None);
}

#[test]
#[ignore = "Stage 2: configuration model is not implemented"]
fn c5_config_validation_distinct_errors_and_password_env_guidance() {
    let errors = pending::<Vec<String>>("C5");
    assert_eq!(errors.len(), 9);
    assert_eq!(
        errors
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>()
            .len(),
        9
    );
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
    let affected = pending::<RoundResult>("C6");
    let exact_scan_count = pending::<u64>("C6");
    let sweep_statement_count = pending::<u64>("C6");
    let other_cohort_completed = pending::<bool>("C6");
    assert_eq!(affected.status, "insufficient_samples");
    assert!(affected.error.is_some());
    assert!(exact_scan_count <= 1);
    assert_eq!(sweep_statement_count, 0);
    assert!(other_cohort_completed);
}

#[test]
#[ignore = "Stage 2: deterministic aggregation is not implemented"]
fn c8_phase_b_reproducible_except_computed_at() {
    let normalized_first_json = pending::<Vec<u8>>("C8");
    let normalized_second_json = pending::<Vec<u8>>("C8");
    assert_eq!(normalized_first_json, normalized_second_json);
}
