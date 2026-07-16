# Tuner acceptance map

Stage 1 intentionally contains no tuner implementation. Machine-gated tests
below are compiled but ignored until the stage named in their status. Harness
self-tests are active and are not substitutes for the acceptance criteria.
C7 is the owner-approved exception: it is deferred to the explicit Gate 3
manual review recorded in `docs/REVIEW_MAP.md`.

| Criterion | Test path and name | Status |
|---|---|---|
| A1 | `crates/seam/tests/acceptance_a_anchor.rs::a1_anchor_recall_exact_for_at_least_99_percent` | blocked — ignored until Stage 4 |
| A2 | `crates/seam/tests/acceptance_a_anchor.rs::a2_anchor_full_population_mean_recall_within_0_005` | blocked — ignored until Stage 4 |
| A3 | `crates/seam/tests/acceptance_a_anchor.rs::a3_anchor_train_p10_within_0_01` | blocked — ignored until Stage 4 |
| A4 | `crates/seam/tests/acceptance_a_anchor.rs::a4_anchor_recommended_ef_identical` | blocked — ignored until Stage 4 |
| A5 | `crates/seam/tests/acceptance_a_anchor.rs::a5_anchor_holdout_quantile_and_transfer_match` | blocked — ignored until Stage 4 |
| B1 | `crates/seam/tests/acceptance_b_estimator.rs::b1_recall_set_intersection_and_short_results` | blocked — ignored until Stage 2 |
| B2 | `crates/seam/tests/acceptance_b_estimator.rs::b2_ground_truth_tie_break_prefers_key_7_over_9` | blocked — ignored until Stage 3 |
| B3 | `crates/seam/tests/acceptance_b_estimator.rs::b3_quantile_type7_linear_and_singleton` | blocked — ignored until Stage 2 |
| B4 | `crates/seam/tests/acceptance_b_estimator.rs::b4_fnv1a_reference_split_fraction_and_membership_stability` | blocked — ignored until Stage 2 |
| B5 | `crates/seam/tests/acceptance_b_estimator.rs::b5_selects_smallest_clearing_ef_40` | blocked — ignored until Stage 2 |
| B6 | `crates/seam/tests/acceptance_b_estimator.rs::b6_target_unmet_uses_max_ef_and_keeps_transfer_fields` | blocked — ignored until Stage 2 |
| B7 | `crates/seam/tests/acceptance_b_estimator.rs::b7_min_samples_999_refuses_and_1000_emits` | blocked — ignored until Stage 2 |
| B8 | `crates/seam/tests/acceptance_b_estimator.rs::b8_window_membership_six_slots_and_one_sixth_empty` | blocked — ignored until Stage 2 |
| B9 | `crates/seam/tests/acceptance_b_estimator.rs::b9_no_double_count_across_overlapping_rounds` | blocked — ignored until Stage 3 |
| B10 | `crates/seam/tests/acceptance_b_estimator.rs::b10_confidence_matches_closed_form_and_scipy_grid` | blocked — ignored until Stage 2 |
| B11 | `crates/seam/tests/acceptance_b_estimator.rs::b11_drop_fraction_is_two_fifteenths` | blocked — ignored until Stage 2 |
| B12 | `crates/seam/tests/acceptance_b_estimator.rs::b12_dedup_within_part_across_parts_and_split` | blocked — ignored until Stages 2–3 |
| C1 | `crates/seam/tests/acceptance_c_durability.rs::c1_resume_mid_part_rewrites_pair_and_matches_clean_run` | blocked — ignored until Stage 3 |
| C2 | `crates/seam/tests/acceptance_c_durability.rs::c2_database_down_still_publishes_cached_phase_b_and_exits_zero` | blocked — ignored until Stage 3 |
| C3 | `crates/seam/tests/acceptance_c_durability.rs::c3_config_fingerprint_k_change_ignores_and_remeasures` | blocked — ignored until Stages 2–3 |
| C4 | `crates/seam/tests/acceptance_c_durability.rs::c4_empty_round_reports_insufficient_samples_and_full_gap` | blocked — ignored until Stage 2 |
| C5 | `crates/seam/tests/acceptance_c_durability.rs::c5_config_validation_distinct_errors_and_password_env_guidance` | blocked — ignored until Stage 2 |
| C6 | `crates/seam/tests/acceptance_c_durability.rs::c6_table_smaller_than_k_aborts_one_cohort_after_exact_scan` | blocked — ignored until Stage 3 |
| C7 | Manual Gate 3 checklist: `docs/REVIEW_MAP.md`, “C7 deferred manual review” | deferred — human sign-off recorded; transaction construction must pass manual review at Gate 3 |
| C8 | `crates/seam/tests/acceptance_c_durability.rs::c8_phase_b_reproducible_except_computed_at` | blocked — ignored until Stage 2 |
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
