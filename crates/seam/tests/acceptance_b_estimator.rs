mod support;

use std::collections::BTreeMap;

use support::pending;

#[derive(Debug)]
struct Selection {
    recommended_ef: Option<i32>,
    status: String,
    confidence: Option<f64>,
    test_quantile_recall: Option<f64>,
}

#[derive(Debug)]
struct SampleCounts {
    available: u64,
    unique: u64,
}

#[test]
#[ignore = "Stage 2: recall estimator is not implemented"]
fn b1_recall_set_intersection_and_short_results() {
    let gt_keys: Vec<i64> = (1..=10).collect();
    let ten_returned = vec![1, 2, 3, 11, 12, 13, 14, 15, 16, 17];
    let seven_returned = vec![1, 2, 3, 4, 5, 11, 12];
    let observed = pending::<(f64, f64)>("B1");
    let _inputs = (gt_keys, ten_returned, seven_returned, 10_u32);
    assert_eq!(observed.0, 0.3);
    assert_eq!(observed.1, 0.5);
}

#[test]
#[ignore = "Stage 3: pgvector transaction is not implemented"]
fn b2_ground_truth_tie_break_prefers_key_7_over_9() {
    // B2 drives the ground-truth query path in isolation. It deliberately does
    // not run the sweep against the unindexed tie-break table.
    let repeated_ground_truth_keys = pending::<Vec<Vec<i64>>>("B2");
    assert!(
        repeated_ground_truth_keys
            .iter()
            .all(|keys| keys.contains(&7) && !keys.contains(&9))
    );
}

#[test]
#[ignore = "Stage 2: type-7 quantile is not implemented"]
fn b3_quantile_type7_linear_and_singleton() {
    let recalls = [0.5, 0.7, 0.9, 1.0];
    let singleton = [0.7];
    let observed = pending::<(f64, f64)>("B3");
    let _inputs = (recalls, singleton, 0.95_f64);
    assert_eq!(observed.0, 0.53);
    assert_eq!(observed.1, 0.7);
}

#[test]
#[ignore = "Stage 2: FNV-1a and split are not implemented"]
fn b4_fnv1a_reference_split_fraction_and_membership_stability() {
    let observed_hashes = pending::<[u64; 3]>("B4");
    assert_eq!(observed_hashes[0], 0xcbf2_9ce4_8422_2325);
    assert_eq!(observed_hashes[1], 0xaf63_dc4c_8601_ec8c);
    assert_eq!(observed_hashes[2], 0x8594_4171_f739_67e8);

    let tuner_and_independent_memberships = pending::<Vec<(bool, bool)>>("B4");
    assert!(
        tuner_and_independent_memberships
            .iter()
            .all(|(tuner, independent)| tuner == independent)
    );
    let observed_fraction = pending::<f64>("B4");
    let distinct_vector_count = pending::<usize>("B4");
    assert_eq!(distinct_vector_count, 10_000);
    assert!((observed_fraction - 0.7).abs() <= 0.03);
    let membership_before_and_after = pending::<Vec<(bool, bool, bool)>>("B4");
    assert!(
        membership_before_and_after
            .iter()
            .all(|(initial, resumed, moved)| initial == resumed && resumed == moved)
    );
}

#[test]
#[ignore = "Stage 2: ef selection is not implemented"]
fn b5_selects_smallest_clearing_ef_40() {
    let train_quantiles =
        BTreeMap::from([(10, 0.62), (20, 0.85), (40, 0.91), (80, 0.93), (160, 0.95)]);
    let observed = pending::<Selection>("B5");
    let _inputs = (train_quantiles, 0.9_f64);
    assert_eq!(observed.recommended_ef, Some(40));
    assert_eq!(observed.status, "ok");
}

#[test]
#[ignore = "Stage 2: target-unmet selection is not implemented"]
fn b6_target_unmet_uses_max_ef_and_keeps_transfer_fields() {
    let train_quantiles =
        BTreeMap::from([(10, 0.62), (20, 0.85), (40, 0.91), (80, 0.93), (160, 0.95)]);
    let observed = pending::<Selection>("B6");
    let _inputs = (train_quantiles, 0.99_f64);
    assert_eq!(observed.recommended_ef, Some(160));
    assert_eq!(observed.status, "target_unmet");
    assert!(observed.confidence.is_some());
    assert!(observed.test_quantile_recall.is_some());
}

#[test]
#[ignore = "Stage 2: minimum-sample gating is not implemented"]
fn b7_min_samples_999_refuses_and_1000_emits() {
    let min_samples = 1_000_u64;
    let below = pending::<Selection>("B7");
    let below_counts = pending::<SampleCounts>("B7");
    assert_eq!(below.status, "insufficient_samples");
    assert_eq!(below.recommended_ef, None);
    assert_eq!(below.confidence, None);
    assert_eq!(below_counts.unique, 999);
    assert!(below_counts.available >= below_counts.unique);

    let at_threshold = pending::<Selection>("B7");
    let at_threshold_counts = pending::<SampleCounts>("B7");
    assert_eq!(at_threshold_counts.unique, min_samples);
    assert!(at_threshold.recommended_ef.is_some());
}

#[test]
#[ignore = "Stage 2: rolling-window membership is not implemented"]
fn b8_window_membership_six_slots_and_one_sixth_empty() {
    let observed_round_end = pending::<u64>("B8");
    let observed_window_starts = pending::<Vec<&'static str>>("B8");
    let excluded_window_starts = pending::<Vec<&'static str>>("B8");
    let observed_empty_fraction = pending::<f64>("B8");
    let _inputs = (600_u32, 3_600_u64, "12:07");
    assert_eq!(observed_round_end, 12 * 60 * 60);
    assert_eq!(
        observed_window_starts,
        ["11:00", "11:10", "11:20", "11:30", "11:40", "11:50"]
    );
    assert_eq!(excluded_window_starts, ["10:50", "12:00"]);
    assert!((observed_empty_fraction - 1.0 / 6.0).abs() <= 1e-12);
}

#[test]
#[ignore = "Stage 3: part diffing and statement instrumentation are not implemented"]
fn b9_no_double_count_across_overlapping_rounds() {
    let in_scope_record_counts = [80_u64, 50_u64];
    let expected_available = in_scope_record_counts.iter().sum::<u64>();
    let first_counts = pending::<SampleCounts>("B9");
    let second_counts = pending::<SampleCounts>("B9");
    let second_round_new_statements = pending::<u64>("B9");
    assert_eq!(first_counts.available, expected_available);
    assert_eq!(second_counts.available, expected_available);
    assert_eq!(first_counts.unique, second_counts.unique);
    assert_eq!(second_round_new_statements, 0);
}

#[test]
#[ignore = "Stage 2: beta-posterior confidence is not implemented"]
fn b10_confidence_matches_closed_form_and_scipy_grid() {
    let all_success = pending::<f64>("B10");
    let all_failure = pending::<f64>("B10");
    let expected_all_success = 1.0 - 0.95_f64.powi(101);
    assert!((all_success - expected_all_success).abs() <= 1e-5);
    assert!(all_failure < 1e-6);

    let tuner_and_scipy = pending::<Vec<(f64, f64)>>("B10");
    assert!(
        tuner_and_scipy
            .iter()
            .all(|(tuner, scipy)| (tuner - scipy).abs() <= 1e-6)
    );
}

#[test]
#[ignore = "Stage 2: drop accounting is not implemented"]
fn b11_drop_fraction_is_two_fifteenths() {
    let headers = [(100_u64, 80_u64), (50_u64, 50_u64)];
    let observed = pending::<f64>("B11");
    let _inputs = headers;
    assert!((observed - 2.0 / 15.0).abs() <= 1e-12);
}

#[derive(Debug)]
struct DedupResult {
    truth_rows: usize,
    sweep_rows: usize,
    first_record_index: i32,
    dup_count: i32,
    unique: u64,
    survivor_part_ulid: String,
    duplicate_split_memberships: Vec<bool>,
}

#[test]
#[ignore = "Stages 2 and 3: measure and aggregate deduplication are not implemented"]
fn b12_dedup_within_part_across_parts_and_split() {
    let duplicate_record_indexes = [3, 5, 9];
    let observed = pending::<DedupResult>("B12");
    let grid_len = 5;
    assert_eq!(observed.truth_rows, 1);
    assert_eq!(observed.sweep_rows, grid_len);
    assert_eq!(observed.first_record_index, 3);
    assert_eq!(observed.first_record_index, duplicate_record_indexes[0]);
    assert_eq!(observed.dup_count, 3);
    assert_eq!(observed.unique, 1);
    assert_eq!(observed.survivor_part_ulid, "01J00000000000000000000000");
    assert!(
        observed
            .duplicate_split_memberships
            .windows(2)
            .all(|pair| pair[0] == pair[1])
    );
}
