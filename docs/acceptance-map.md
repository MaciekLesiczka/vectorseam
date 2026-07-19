# Tuner acceptance map

Stage 4 and the effective-recommendation extension are implemented. A
criterion is marked `passing` only when all of its currently required behavior
is machine-gated. The F-agg and E suites run without a database; tests marked
F-pg are executed by `make seam-f-pg-tests` after the deterministic fixture is
loaded. Suite A is executed by
`make seam-anchor-tests` after the trusted Python anchor output is generated.
All database-backed tests are marked `#[ignore]` in ordinary `cargo test`;
their Docker targets explicitly run ignored tests with `SEAM_REQUIRE_F_PG=1`,
which the tests assert. C7 is the owner-approved manual-review exception
recorded in `docs/REVIEW_MAP.md`.

| Criterion | Test path and name | Status |
|---|---|---|
| A1 | `crates/seam/tests/acceptance_a_anchor.rs::a1_anchor_recall_exact_for_at_least_99_percent` (F-pg + trusted anchor) | passing |
| A2 | `crates/seam/tests/acceptance_a_anchor.rs::a2_anchor_full_population_mean_recall_within_0_005` (F-pg + trusted anchor) | passing |
| A3 | `crates/seam/tests/acceptance_a_anchor.rs::a3_anchor_train_p10_within_0_01` (F-pg + trusted anchor) | passing |
| A4 | `crates/seam/tests/acceptance_a_anchor.rs::a4_anchor_recommended_ef_identical` (F-pg + trusted anchor) | passing — both select the spec literal `80` |
| A5 | `crates/seam/tests/acceptance_a_anchor.rs::a5_anchor_holdout_quantile_and_transfer_match` (F-pg + trusted anchor) | passing |
| B1 | `crates/seam/tests/acceptance_b_estimator.rs::b1_recall_set_intersection_and_short_results` | passing |
| B2 | `crates/seam/src/database.rs::tests::b2_f_pg_ground_truth_tie_break_prefers_key_7_over_9` (F-pg); `crates/seam/src/database.rs::tests::b2_ground_truth_sql_quotes_identifiers_and_tie_breaks_by_key` | passing |
| B3 | `crates/seam/tests/acceptance_b_estimator.rs::b3_quantile_type7_linear_and_singleton` | passing |
| B4 | `crates/seam/tests/acceptance_b_estimator.rs::b4_fnv1a_reference_split_fraction_and_membership_stability`; `crates/seam/tests/acceptance_b_estimator.rs::b4_b12_aggregate_survivor_movement_preserves_split_membership` | passing |
| B5 | `crates/seam/tests/acceptance_b_estimator.rs::b5_selects_smallest_clearing_ef_40` | passing |
| B6 | `crates/seam/tests/acceptance_b_estimator.rs::b6_target_unmet_uses_max_ef_and_keeps_transfer_fields` | passing |
| B7 | `crates/seam/tests/acceptance_b_estimator.rs::b7_min_samples_999_refuses_and_1000_emits`; `crates/seam/tests/acceptance_b_estimator.rs::b7_realized_empty_split_is_insufficient_even_at_min_samples` | passing |
| B8 | `crates/seam/src/accounting.rs::tests::b8_window_membership_enumerates_exactly_six_slots`; `crates/seam/tests/acceptance_b_estimator.rs::b8_window_membership_six_slots_and_one_sixth_empty` | passing |
| B9 | `crates/seam/tests/acceptance_b_estimator.rs::b9_no_double_count_across_overlapping_rounds_in_phase_b`; `crates/seam/src/pipeline.rs::tests::b9_second_round_issues_zero_new_database_transactions` | passing |
| B10 | `crates/seam/tests/acceptance_b_estimator.rs::b10_confidence_matches_closed_form_and_scipy_grid` | passing |
| B11 | `crates/seam/tests/acceptance_b_estimator.rs::b11_drop_fraction_is_two_fifteenths` | passing |
| B12 | `crates/seam/tests/acceptance_b_estimator.rs::b4_b12_aggregate_survivor_movement_preserves_split_membership`; `crates/seam/src/measure.rs::tests::b12_measure_dedup_emits_one_truth_row_and_one_sweep_grid` | passing |
| C1 | `crates/seam/src/pipeline.rs::tests::c1_resume_mid_part_rewrites_pair_and_matches_clean_run` | passing |
| C2 | `crates/seam/tests/acceptance_c_durability.rs::c2_database_down_still_publishes_cached_phase_b_and_exits_zero`; `crates/seam/src/pipeline.rs::tests::c2_connection_outage_leaves_part_unmeasured_and_next_round_retries`; `crates/seam/src/tuner.rs::tests::c2_f_pg_connection_outage_reconnects_next_round_without_durable_failure` (F-pg); `crates/seam/src/tuner.rs::tests::c2_f_pg_client_timeout_discards_part_then_reconnects_next_round` (F-pg) | passing |
| C3 | `crates/seam/tests/acceptance_c_durability.rs::c3_config_fingerprint_k_change_ignores_incompatible_intermediate`; `crates/seam/src/pipeline.rs::tests::c3_config_fingerprint_remeasures_and_retains_incompatible_count` | passing |
| C4 | `crates/seam/tests/acceptance_c_durability.rs::c4_empty_round_reports_insufficient_samples_and_full_gap` | passing |
| C5 | `crates/seam/tests/acceptance_c_durability.rs::c5_config_validation_distinct_errors_and_password_env_guidance`; `crates/seam/src/config.rs::tests::c5_missing_password_env_is_rejected_only_when_configured`; `crates/seam/src/config.rs::tests::c5_duplicate_data_source_pair_is_rejected`; `crates/seam/src/config.rs::tests::rejects_zero_client_timeout` | passing |
| C6 | `crates/seam/tests/acceptance_c_durability.rs::c6_phase_a_abort_forces_insufficient_despite_cached_min_samples`; `crates/seam/src/pipeline.rs::tests::c6_table_smaller_than_k_stops_after_first_scan_despite_cached_population`; `crates/seam/src/tuner.rs::tests::c6_f_pg_table_smaller_stops_after_one_exact_and_other_cohort_continues` (F-pg) | passing |
| C7 | Manual Gate 3 checklist: `docs/REVIEW_MAP.md`, “C7 deferred manual review” | deferred-with-my-approval — owner completed and approved the manual transaction review on 2026-07-17 |
| C8 | `crates/seam/tests/acceptance_c_durability.rs::c8_phase_b_reproducible_except_computed_at` | passing |
| D1 | `crates/seam/src/pacer.rs::tests::d1_duty_cycle_20_percent_wall_time_bound` | passing |
| D2 | No test — criterion and `max_concurrent_queries` were removed by owner decision | deferred-with-my-approval — removal approved 2026-07-17; row retained as the required sign-off record |
| D3 | `crates/seam/src/tuner.rs::tests::d3_f_pg_one_millisecond_timeout_fails_without_retries_or_leaks` (F-pg) | passing |
| E1 | `crates/seam/src/pipeline.rs::tests::e1_carry_on_insufficient_persists_to_history_latest_and_republication`; `crates/seam/src/pipeline.rs::tests::e1_all_durable_sample_failures_carry_previous_effective`; `crates/seam/src/pipeline.rs::tests::c6_table_smaller_than_k_stops_after_first_scan_despite_cached_population` | passing |
| E2 | `crates/seam/src/pipeline.rs::tests::e2_newest_target_unmet_signal_wins_and_is_then_carried` | passing |
| E3 | `crates/seam/src/pipeline.rs::tests::e3_carry_survives_fresh_pipeline_invocation_using_only_storage` | passing |
| E4 | `crates/seam/src/pipeline.rs::tests::e4_fingerprint_change_resets_effective_for_all_required_fields` | passing |
| E5 | `crates/seam/src/pipeline.rs::tests::e5_bootstrap_corrupt_and_pre_effective_latest_publish_null_effective` | passing |

C7 is deferred from machine gating with the owner's explicit approval, and
its required manual transaction review was completed and approved on
2026-07-17. D2 is the sole dropped criterion: the owner explicitly removed
the concurrency configuration and acceptance requirement on 2026-07-17. Its
row remains so that the deletion cannot silently disappear from project
history.

F-pg fixture precondition: the owner directed an ascending deterministic seed
search. Seed `0` is the first candidate and passes the strengthened PostgreSQL
boundary-gap check for all 500 queries; its minimum observed gap is
`1.3683911141981753e-6`. The Stage 4 harness resets generated tuner and anchor
artifacts before each fixture run, preserving the shared query order while
preventing cached intermediates from masking an end-to-end regression. The
anchor target also removes `comparison.json` immediately before the Python
driver and asserts that a non-empty replacement exists before Rust tests run.
