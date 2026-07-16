mod support;

use support::anchor::read_anchor_comparison;
use support::pending;

#[derive(Debug)]
struct AnchorComparison {
    exact_recall_fraction: f64,
    mean_recall_absolute_differences: Vec<f64>,
    train_quantile_absolute_differences: Vec<f64>,
    tuner_recommended_ef: i32,
    anchor_recommended_ef: i32,
    test_quantile_absolute_difference: f64,
    tuner_transferred: bool,
    anchor_transferred: bool,
}

#[test]
#[ignore = "Stage 4: tuner and anchor comparison output are not implemented"]
fn a1_anchor_recall_exact_for_at_least_99_percent() {
    let _anchor = read_anchor_comparison().unwrap();
    let comparison = pending::<AnchorComparison>("A1");
    assert!(comparison.exact_recall_fraction >= 0.99);
}

#[test]
#[ignore = "Stage 4: tuner and anchor comparison output are not implemented"]
fn a2_anchor_full_population_mean_recall_within_0_005() {
    let _anchor = read_anchor_comparison().unwrap();
    let comparison = pending::<AnchorComparison>("A2");
    assert!(
        comparison
            .mean_recall_absolute_differences
            .iter()
            .all(|difference| *difference <= 0.005)
    );
}

#[test]
#[ignore = "Stage 4: tuner and anchor comparison output are not implemented"]
fn a3_anchor_train_p10_within_0_01() {
    let _anchor = read_anchor_comparison().unwrap();
    let comparison = pending::<AnchorComparison>("A3");
    assert!(
        comparison
            .train_quantile_absolute_differences
            .iter()
            .all(|difference| *difference <= 0.01)
    );
}

#[test]
#[ignore = "Stage 4: tuner and anchor comparison output are not implemented"]
fn a4_anchor_recommended_ef_identical() {
    let _anchor = read_anchor_comparison().unwrap();
    let comparison = pending::<AnchorComparison>("A4");
    assert_eq!(
        comparison.tuner_recommended_ef,
        comparison.anchor_recommended_ef
    );
}

#[test]
#[ignore = "Stage 4: tuner and anchor comparison output are not implemented"]
fn a5_anchor_holdout_quantile_and_transfer_match() {
    let _anchor = read_anchor_comparison().unwrap();
    let comparison = pending::<AnchorComparison>("A5");
    assert!(comparison.test_quantile_absolute_difference <= 0.01);
    assert_eq!(comparison.tuner_transferred, comparison.anchor_transferred);
}
