mod support;

use support::pending;

#[test]
#[ignore = "Stage 3: duty-cycle pacer is not implemented"]
fn d1_duty_cycle_20_percent_wall_time_bound() {
    let statement_count = pending::<u64>("D1");
    let total_elapsed_seconds = pending::<f64>("D1");
    let total_busy_seconds = pending::<f64>("D1");
    assert!(statement_count >= 50);
    assert!(total_elapsed_seconds >= 0.95 * (total_busy_seconds / 0.20));
    assert!(total_busy_seconds / total_elapsed_seconds <= 0.21);
}

#[test]
#[ignore = "Stage 3: statement timeout and server cleanup are not implemented"]
fn d3_one_millisecond_timeout_fails_without_retries_or_leaks() {
    let sample_count = pending::<u64>("D3");
    let failed_count = pending::<u64>("D3");
    let retry_count = pending::<u64>("D3");
    let running_tuner_statements_after_round = pending::<u64>("D3");
    assert_eq!(failed_count, sample_count);
    assert_eq!(retry_count, 0);
    assert_eq!(running_tuner_statements_after_round, 0);
}
