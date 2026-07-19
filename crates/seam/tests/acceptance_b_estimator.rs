mod support;

use std::collections::BTreeMap;

use seam::aggregate::aggregate;
use seam::intermediate::read_intermediate_pair;
use seam::math::{
    fnv1a64, is_train_member, quantile_type7, recall_at_k, select_ef, transfer_confidence,
};
use seam::model::{
    AggregationConfig, AggregationInput, IntermediateMetadata, IntermediatePart, ListedPart,
    MeasuredSample, RoundStatus, SweepMeasurement,
};
use support::f_agg::{
    DEFAULT_PART_ULID, DEFAULT_WINDOW_SECONDS, DEFAULT_WINDOW_START, write_b12_cross_part_fixture,
};
use tempfile::tempdir;
use vectorseam_core::window::aligned_window_start;

const EF_GRID: [i32; 5] = [10, 20, 40, 80, 160];

#[test]
fn b1_recall_set_intersection_and_short_results() {
    let gt_keys: Vec<i64> = (1..=10).collect();
    let ten_returned = vec![1, 2, 3, 11, 12, 13, 14, 15, 16, 17];
    let seven_returned = vec![1, 2, 3, 4, 5, 11, 12];

    assert_eq!(recall_at_k(&gt_keys, &ten_returned, 10).unwrap(), 0.3);
    assert_eq!(recall_at_k(&gt_keys, &seven_returned, 10).unwrap(), 0.5);
}

#[test]
fn b3_quantile_type7_linear_and_singleton() {
    assert_eq!(quantile_type7(&[0.5, 0.7, 0.9, 1.0], 0.05).unwrap(), 0.53);
    assert_eq!(quantile_type7(&[0.7], 0.05).unwrap(), 0.7);
}

#[test]
fn b4_fnv1a_reference_split_fraction_and_membership_stability() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);

    let vector_hashes = (0_u64..10_000).collect::<Vec<_>>();
    let memberships = vector_hashes
        .iter()
        .map(|hash| is_train_member(*hash, 7, 0.7).unwrap())
        .collect::<Vec<_>>();
    for (vector_hash, tuner) in vector_hashes.iter().zip(&memberships) {
        assert_eq!(*tuner, independent_is_train(*vector_hash, 7, 0.7));
    }
    let observed_fraction = memberships.iter().filter(|member| **member).count() as f64 / 10_000.0;
    assert!((observed_fraction - 0.7).abs() <= 0.03);
}

#[test]
fn b5_selects_smallest_clearing_ef_40() {
    let train_quantiles =
        BTreeMap::from([(10, 0.62), (20, 0.85), (40, 0.91), (80, 0.93), (160, 0.95)]);

    let observed = select_ef(&train_quantiles, 0.9).unwrap();

    assert_eq!(observed.recommended_ef, 40);
    assert_eq!(observed.status, RoundStatus::Ok);
}

#[test]
fn b6_target_unmet_uses_max_ef_and_keeps_transfer_fields() {
    let recalls = BTreeMap::from([(10, 0.62), (20, 0.85), (40, 0.91), (80, 0.93), (160, 0.95)]);
    let input = populated_input(100, 100, 0.99, &recalls);

    let observed = aggregate(&input).unwrap();

    assert_eq!(observed.recommended_ef, Some(160));
    assert_eq!(observed.status, RoundStatus::TargetUnmet);
    assert!(observed.confidence.is_some());
    assert!(observed.test_quantile_recall.is_some());
}

#[test]
fn b7_min_samples_999_refuses_and_1000_emits() {
    let recalls = EF_GRID.into_iter().map(|ef| (ef, 1.0)).collect();
    let below = aggregate(&populated_input(999, 1_000, 0.9, &recalls)).unwrap();
    assert_eq!(below.status, RoundStatus::InsufficientSamples);
    assert_eq!(below.recommended_ef, None);
    assert_eq!(below.confidence, None);
    assert_eq!(below.samples.unique, 999);
    assert!(below.samples.available >= below.samples.unique);
    assert_eq!(below.per_ef.len(), 5);

    let at_threshold = aggregate(&populated_input(1_000, 1_000, 0.9, &recalls)).unwrap();
    assert_eq!(at_threshold.samples.unique, 1_000);
    assert!(at_threshold.recommended_ef.is_some());
}

#[test]
fn b7_realized_empty_split_is_insufficient_even_at_min_samples() {
    let recalls = EF_GRID.into_iter().map(|ef| (ef, 1.0)).collect();
    let mut input = populated_input(100, 100, 0.9, &recalls);
    input.config.train_fraction = 0.0001;
    assert!(
        (0_u64..100).all(|hash| !is_train_member(hash, 7, input.config.train_fraction).unwrap())
    );

    let observed = aggregate(&input).unwrap();

    assert_eq!(observed.samples.unique, 100);
    assert_eq!(observed.samples.train, 0);
    assert_eq!(observed.status, RoundStatus::InsufficientSamples);
    assert_eq!(observed.recommended_ef, None);
    assert_eq!(observed.transferred, None);
}

#[test]
fn b8_window_membership_six_slots_and_one_sixth_empty() {
    let observed_round_end = aligned_window_start(12 * 60 * 60 + 7 * 60, 600).unwrap();
    assert_eq!(observed_round_end, 12 * 60 * 60);

    let starts_with_parts = [
        11 * 60 * 60,
        11 * 60 * 60 + 10 * 60,
        11 * 60 * 60 + 30 * 60,
        11 * 60 * 60 + 40 * 60,
        11 * 60 * 60 + 50 * 60,
    ];
    let mut listed_parts = starts_with_parts
        .into_iter()
        .enumerate()
        .map(|(index, start)| listed_part(format!("part-{index}"), start, 1))
        .collect::<Vec<_>>();
    listed_parts.push(listed_part("excluded-before", 10 * 60 * 60 + 50 * 60, 1));
    listed_parts.push(listed_part("excluded-open", 12 * 60 * 60, 1));
    let input = AggregationInput {
        config: AggregationConfig {
            window_duration_seconds: 3_600,
            ..aggregation_config(100, 0.9)
        },
        round_end: observed_round_end,
        computed_at: "1970-01-01T12:07:00Z".to_owned(),
        phase_a_abort: None,
        phase_a_incompatible_parts: 0,
        previous_round: None,
        listed_parts,
        intermediates: Vec::new(),
    };

    let observed = aggregate(&input).unwrap();
    assert_eq!(observed.coverage.windows_in_scope, 6);
    assert_eq!(observed.coverage.windows_with_parts, 5);
    assert!((observed.coverage.empty_window_fraction - 1.0 / 6.0).abs() <= 1e-12);
}

#[test]
fn b9_no_double_count_across_overlapping_rounds_in_phase_b() {
    let listed_parts = vec![
        listed_part("part-a", 6_000, 80),
        listed_part("part-b", 6_600, 50),
    ];
    let make_input = |round_end| AggregationInput {
        config: AggregationConfig {
            window_duration_seconds: 3_600,
            ..aggregation_config(100, 0.9)
        },
        round_end,
        computed_at: "1970-01-01T02:00:00Z".to_owned(),
        phase_a_abort: None,
        phase_a_incompatible_parts: 0,
        previous_round: None,
        listed_parts: listed_parts.clone(),
        intermediates: Vec::new(),
    };

    let first = aggregate(&make_input(7_200)).unwrap();
    let second = aggregate(&make_input(7_800)).unwrap();
    assert_eq!(first.samples.available, 130);
    assert_eq!(second.samples.available, 130);
    assert_eq!(first.samples.unique, second.samples.unique);
}

#[test]
fn b10_confidence_matches_closed_form_and_scipy_grid() {
    let all_success = transfer_confidence(100, 100, 0.95).unwrap();
    let all_failure = transfer_confidence(100, 0, 0.95).unwrap();
    let expected_all_success = 1.0 - 0.95_f64.powi(101);
    assert!((all_success - expected_all_success).abs() <= 1e-5);
    assert!(all_failure < 1e-6);

    // Literals were generated independently with scipy.stats.beta.sf.
    let scipy_grid = [
        ((1, 0), 0.0025000000000000044),
        ((1, 1), 0.09750000000000009),
        ((10, 0), 4.882812500000048e-15),
        ((10, 5), 5.80134505859378e-06),
        ((10, 10), 0.43119990772354033),
        ((100, 90), 0.012308204272281355),
        ((100, 95), 0.3930017634073816),
    ];
    for ((n, m), scipy) in scipy_grid {
        let tuner = transfer_confidence(n, m, 0.95).unwrap();
        assert!((tuner - scipy).abs() <= 1e-6, "n={n}, m={m}");
    }
}

#[test]
fn b11_drop_fraction_is_two_fifteenths() {
    let input = AggregationInput {
        config: aggregation_config(100, 0.9),
        round_end: DEFAULT_WINDOW_START + u64::from(DEFAULT_WINDOW_SECONDS),
        computed_at: "2026-07-08T12:10:00Z".to_owned(),
        phase_a_abort: None,
        phase_a_incompatible_parts: 0,
        previous_round: None,
        listed_parts: vec![
            ListedPart {
                part_ulid: "part-a".to_owned(),
                window_start: DEFAULT_WINDOW_START,
                window_seconds: DEFAULT_WINDOW_SECONDS,
                received_frame_count: 100,
                record_count: 80,
            },
            ListedPart {
                part_ulid: "part-b".to_owned(),
                window_start: DEFAULT_WINDOW_START,
                window_seconds: DEFAULT_WINDOW_SECONDS,
                received_frame_count: 50,
                record_count: 50,
            },
        ],
        intermediates: Vec::new(),
    };
    let observed = aggregate(&input).unwrap();
    assert!((observed.dropped_frame_fraction - 2.0 / 15.0).abs() <= 1e-12);
}

#[test]
fn b4_b12_aggregate_survivor_movement_preserves_split_membership() {
    let root = tempdir().unwrap();
    let fixture = write_b12_cross_part_fixture(root.path()).unwrap();
    let first =
        read_intermediate_pair(&fixture.first.truth_path, &fixture.first.sweep_path).unwrap();
    let second =
        read_intermediate_pair(&fixture.second.truth_path, &fixture.second.sweep_path).unwrap();
    assert_eq!(first.samples.len(), 1);
    assert_eq!(first.samples[0].record_index, 3);
    assert_eq!(first.samples[0].dup_count, 3);
    assert_eq!(first.samples[0].sweeps.len(), 5);

    assert!(first.metadata.part_ulid < second.metadata.part_ulid);
    assert_eq!(first.samples[0].vector_hash, second.samples[0].vector_hash);

    let both_input = AggregationInput {
        config: aggregation_config(100, 0.9),
        round_end: DEFAULT_WINDOW_START + u64::from(DEFAULT_WINDOW_SECONDS),
        computed_at: "2026-07-08T12:10:00Z".to_owned(),
        phase_a_abort: None,
        phase_a_incompatible_parts: 0,
        previous_round: None,
        listed_parts: vec![
            listed_from_intermediate(&first),
            listed_from_intermediate(&second),
        ],
        intermediates: vec![second.clone(), first.clone()],
    };
    let both = aggregate(&both_input).unwrap();
    assert_eq!(both.samples.unique, 1);
    assert!(both.per_ef.iter().all(|summary| summary.mean_recall == 1.0));

    let mut resumed_input = both_input.clone();
    resumed_input.listed_parts.reverse();
    resumed_input.intermediates.reverse();
    let resumed = aggregate(&resumed_input).unwrap();
    assert_eq!(resumed.samples.train, both.samples.train);
    assert_eq!(resumed.samples.test, both.samples.test);

    let mut moved_input = both_input;
    moved_input.listed_parts = vec![listed_from_intermediate(&second)];
    moved_input.intermediates = vec![second];
    let moved = aggregate(&moved_input).unwrap();
    assert_eq!(moved.samples.unique, 1);
    assert!(
        moved
            .per_ef
            .iter()
            .all(|summary| summary.mean_recall == 0.5)
    );
    assert_eq!(moved.samples.train, both.samples.train);
    assert_eq!(moved.samples.test, both.samples.test);
}

fn independent_is_train(vector_hash: u64, split_seed: u64, train_fraction: f64) -> bool {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in format!("s:{split_seed}:{vector_hash}").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash % 10_000 < (train_fraction * 10_000.0).round() as u64
}

fn aggregation_config(min_samples: usize, value: f64) -> AggregationConfig {
    AggregationConfig {
        cohort: "acceptance/f-agg".to_owned(),
        target_name: "recall".to_owned(),
        index: "fixture".to_owned(),
        table: "docs_fixture".to_owned(),
        column: "embedding".to_owned(),
        key: "doc_id".to_owned(),
        k: 10,
        value,
        percentile: 0.95,
        window_duration_seconds: 600,
        storage_window_seconds: 600,
        ef_grid: EF_GRID.to_vec(),
        train_fraction: 0.7,
        split_seed: 7,
        min_samples,
    }
}

fn populated_input(
    sample_count: usize,
    min_samples: usize,
    target_value: f64,
    recalls: &BTreeMap<i32, f64>,
) -> AggregationInput {
    let samples = (0..sample_count)
        .map(|record_index| MeasuredSample {
            record_index: i32::try_from(record_index).unwrap(),
            vector_hash: record_index as u64,
            dup_count: 1,
            sweeps: recalls
                .iter()
                .map(|(ef, recall)| {
                    (
                        *ef,
                        SweepMeasurement {
                            recall: *recall,
                            latency_ms: f64::from(*ef) / 10.0,
                        },
                    )
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let metadata = IntermediateMetadata {
        format_version: 1,
        cohort: "acceptance/f-agg".to_owned(),
        part_ulid: DEFAULT_PART_ULID.to_owned(),
        window_start: DEFAULT_WINDOW_START,
        window_seconds: DEFAULT_WINDOW_SECONDS,
        received_frame_count: sample_count as u64,
        record_count: sample_count as u64,
        index: "fixture".to_owned(),
        table: "docs_fixture".to_owned(),
        column: "embedding".to_owned(),
        key: "doc_id".to_owned(),
        k: 10,
        ef_grid: EF_GRID.to_vec(),
        failed_count: 0,
        measured_count: sample_count as u64,
        computed_at_us: 1_783_512_000_000_000,
    };
    AggregationInput {
        config: aggregation_config(min_samples, target_value),
        round_end: DEFAULT_WINDOW_START + u64::from(DEFAULT_WINDOW_SECONDS),
        computed_at: "2026-07-08T12:10:00Z".to_owned(),
        phase_a_abort: None,
        phase_a_incompatible_parts: 0,
        previous_round: None,
        listed_parts: vec![ListedPart {
            part_ulid: DEFAULT_PART_ULID.to_owned(),
            window_start: DEFAULT_WINDOW_START,
            window_seconds: DEFAULT_WINDOW_SECONDS,
            received_frame_count: sample_count as u64,
            record_count: sample_count as u64,
        }],
        intermediates: vec![IntermediatePart { metadata, samples }],
    }
}

fn listed_part(part_ulid: impl Into<String>, window_start: u64, records: u64) -> ListedPart {
    ListedPart {
        part_ulid: part_ulid.into(),
        window_start,
        window_seconds: 600,
        received_frame_count: records,
        record_count: records,
    }
}

fn listed_from_intermediate(part: &IntermediatePart) -> ListedPart {
    ListedPart {
        part_ulid: part.metadata.part_ulid.clone(),
        window_start: part.metadata.window_start,
        window_seconds: part.metadata.window_seconds,
        received_frame_count: part.metadata.received_frame_count,
        record_count: part.metadata.record_count,
    }
}
