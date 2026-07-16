# Tuner acceptance map

Stage 2 is implemented. A criterion is marked `passing` only when all of its
current required behavior is machine-gated. Multi-stage criteria remain
`blocked` when their Phase B path passes but a Phase A/durability path is still
ignored. Harness self-tests are active and are not substitutes for acceptance
criteria. C7 is the owner-approved exception deferred to the explicit Gate 3
manual review in `docs/REVIEW_MAP.md`.

| Criterion | Test path and name | Status |
|---|---|---|
| A1 | `crates/seam/tests/acceptance_a_anchor.rs::a1_anchor_recall_exact_for_at_least_99_percent` | blocked — ignored until Stage 4 |
| A2 | `crates/seam/tests/acceptance_a_anchor.rs::a2_anchor_full_population_mean_recall_within_0_005` | blocked — ignored until Stage 4 |
| A3 | `crates/seam/tests/acceptance_a_anchor.rs::a3_anchor_train_p10_within_0_01` | blocked — ignored until Stage 4 |
| A4 | `crates/seam/tests/acceptance_a_anchor.rs::a4_anchor_recommended_ef_identical` | blocked — ignored until Stage 4 |
| A5 | `crates/seam/tests/acceptance_a_anchor.rs::a5_anchor_holdout_quantile_and_transfer_match` | blocked — ignored until Stage 4 |
| B1 | `crates/seam/tests/acceptance_b_estimator.rs::b1_recall_set_intersection_and_short_results` | passing |
| B2 | `crates/seam/tests/acceptance_b_estimator.rs::b2_ground_truth_tie_break_prefers_key_7_over_9` | blocked — ignored until Stage 3 |
| B3 | `crates/seam/tests/acceptance_b_estimator.rs::b3_quantile_type7_linear_and_singleton` | passing |
| B4 | `crates/seam/tests/acceptance_b_estimator.rs::b4_fnv1a_reference_split_fraction_and_membership_stability` | passing |
| B5 | `crates/seam/tests/acceptance_b_estimator.rs::b5_selects_smallest_clearing_ef_40` | passing |
| B6 | `crates/seam/tests/acceptance_b_estimator.rs::b6_target_unmet_uses_max_ef_and_keeps_transfer_fields` | passing |
| B7 | `crates/seam/tests/acceptance_b_estimator.rs::b7_min_samples_999_refuses_and_1000_emits`; `crates/seam/tests/acceptance_b_estimator.rs::b7_realized_empty_split_is_insufficient_even_at_min_samples` | passing |
| B8 | `crates/seam/tests/acceptance_b_estimator.rs::b8_window_membership_six_slots_and_one_sixth_empty` | passing |
| B9 | `crates/seam/tests/acceptance_b_estimator.rs::b9_no_double_count_across_overlapping_rounds_in_phase_b`; `crates/seam/tests/acceptance_b_estimator.rs::b9_second_round_issues_zero_new_database_statements` | blocked — Phase B no-double-count path passing; Stage 3 statement path ignored |
| B10 | `crates/seam/tests/acceptance_b_estimator.rs::b10_confidence_matches_closed_form_and_scipy_grid` | passing |
| B11 | `crates/seam/tests/acceptance_b_estimator.rs::b11_drop_fraction_is_two_fifteenths` | passing |
| B12 | `crates/seam/tests/acceptance_b_estimator.rs::b12_aggregate_dedup_keeps_lexicographically_smallest_survivor_and_split`; `crates/seam/tests/acceptance_b_estimator.rs::b12_measure_dedup_emits_one_truth_row_and_one_sweep_grid` | blocked — Phase B cross-part dedup path passing; Stage 3 measure-side path ignored |
| C1 | `crates/seam/tests/acceptance_c_durability.rs::c1_resume_mid_part_rewrites_pair_and_matches_clean_run` | blocked — ignored until Stage 3 |
| C2 | `crates/seam/tests/acceptance_c_durability.rs::c2_database_down_still_publishes_cached_phase_b_and_exits_zero` | blocked — ignored until Stage 3 |
| C3 | `crates/seam/tests/acceptance_c_durability.rs::c3_config_fingerprint_k_change_ignores_incompatible_intermediate`; `crates/seam/tests/acceptance_c_durability.rs::c3_config_fingerprint_k_change_remeasures_with_k_20` | blocked — Phase B compatibility path passing; Stage 3 remeasurement path ignored |
| C4 | `crates/seam/tests/acceptance_c_durability.rs::c4_empty_round_reports_insufficient_samples_and_full_gap` | passing |
| C5 | `crates/seam/tests/acceptance_c_durability.rs::c5_config_validation_distinct_errors_and_password_env_guidance` | passing |
| C6 | `crates/seam/tests/acceptance_c_durability.rs::c6_table_smaller_than_k_aborts_one_cohort_after_exact_scan` | blocked — ignored until Stage 3 |
| C7 | Manual Gate 3 checklist: `docs/REVIEW_MAP.md`, “C7 deferred manual review” | deferred — owner-approved deferral; transaction review sign-off pending at Gate 3 |
| C8 | `crates/seam/tests/acceptance_c_durability.rs::c8_phase_b_reproducible_except_computed_at` | passing |
| D1 | `crates/seam/tests/acceptance_d_resources.rs::d1_duty_cycle_20_percent_wall_time_bound` | blocked — ignored until Stage 3 |
| D2 | `crates/seam/tests/acceptance_d_resources.rs::d2_concurrency_never_exceeds_config_and_default_reaches_one` | blocked — ignored until Stage 3 |
| D3 | `crates/seam/tests/acceptance_d_resources.rs::d3_one_millisecond_timeout_fails_without_retries_or_leaks` | blocked — ignored until Stage 3 |

C7 is deferred with the owner's explicit approval in this task. No criterion
is dropped.

F-pg fixture precondition: the owner directed an ascending deterministic seed
search. Seed `0` is the first candidate and passes the strengthened PostgreSQL
boundary-gap check for all 500 queries; its minimum observed gap is
`1.3683911141981753e-6`. A1–A5 remain blocked until Stage 4 implementation,
not by fixture generation.
