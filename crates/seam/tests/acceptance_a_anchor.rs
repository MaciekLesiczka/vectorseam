mod support;

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use vectorseam_core::cohort::CohortName;

use seam::config::{
    BudgetConfig, CalibrationConfig, CohortConfig, Config, DataSourceConfig, IndexConfig,
    StorageConfig, TargetConfig,
};
use seam::intermediate::read_intermediate_pair;
use seam::math::{is_train_member, quantile_type7};
use seam::tuner::Tuner;

use support::anchor::{fixture_root, read_anchor_comparison};

const WINDOW_START: u64 = 1_784_116_800;
const WINDOW_SECONDS: u32 = 600;
const PART_ULID: &str = "01K0A000000000000000000000";
const COHORT: &str = "anchor/f-pg";
const EF_GRID: [i32; 5] = [10, 20, 40, 80, 160];

static COMPARISON: OnceLock<Result<AnchorComparison, String>> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct AnchorOutput {
    value: f64,
    query_order: Vec<i64>,
    recall_rows: Vec<AnchorRecallRow>,
    per_ef: Vec<AnchorPerEf>,
    recommended_ef: i32,
    test_quantile_recall: f64,
    transferred: bool,
}

#[derive(Debug, Deserialize)]
struct AnchorRecallRow {
    query_id: i64,
    ef: i32,
    recall: f64,
}

#[derive(Debug, Deserialize)]
struct AnchorPerEf {
    ef: i32,
    mean_recall: f64,
    train_quantile_recall: f64,
}

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
#[ignore = "requires the trusted anchor and Docker F-pg fixture; run make seam-anchor-tests"]
fn a1_anchor_recall_exact_for_at_least_99_percent() {
    let comparison = required_comparison();
    assert!(
        comparison.exact_recall_fraction >= 0.99,
        "observed exact recall fraction {}",
        comparison.exact_recall_fraction
    );
}

#[test]
#[ignore = "requires the trusted anchor and Docker F-pg fixture; run make seam-anchor-tests"]
fn a2_anchor_full_population_mean_recall_within_0_005() {
    let comparison = required_comparison();
    assert!(
        comparison
            .mean_recall_absolute_differences
            .iter()
            .all(|difference| *difference <= 0.005),
        "observed per-ef mean differences {:?}",
        comparison.mean_recall_absolute_differences
    );
}

#[test]
#[ignore = "requires the trusted anchor and Docker F-pg fixture; run make seam-anchor-tests"]
fn a3_anchor_train_p10_within_0_01() {
    let comparison = required_comparison();
    assert!(
        comparison
            .train_quantile_absolute_differences
            .iter()
            .all(|difference| *difference <= 0.01),
        "observed per-ef train quantile differences {:?}",
        comparison.train_quantile_absolute_differences
    );
}

#[test]
#[ignore = "requires the trusted anchor and Docker F-pg fixture; run make seam-anchor-tests"]
fn a4_anchor_recommended_ef_identical() {
    let comparison = required_comparison();
    assert_eq!(
        comparison.tuner_recommended_ef,
        comparison.anchor_recommended_ef
    );
    assert_eq!(comparison.tuner_recommended_ef, 80);
}

#[test]
#[ignore = "requires the trusted anchor and Docker F-pg fixture; run make seam-anchor-tests"]
fn a5_anchor_holdout_quantile_and_transfer_match() {
    let comparison = required_comparison();
    assert!(
        comparison.test_quantile_absolute_difference <= 0.01,
        "observed holdout quantile difference {}",
        comparison.test_quantile_absolute_difference
    );
    assert_eq!(comparison.tuner_transferred, comparison.anchor_transferred);
}

fn required_comparison() -> &'static AnchorComparison {
    assert!(
        std::env::var_os("SEAM_REQUIRE_F_PG").is_some(),
        "ignored anchor tests must run through make seam-anchor-tests"
    );
    COMPARISON
        .get_or_init(|| build_comparison().map_err(|error| format!("{error:#}")))
        .as_ref()
        .unwrap_or_else(|error| panic!("build A-suite comparison: {error}"))
}

fn build_comparison() -> Result<AnchorComparison> {
    let anchor = read_anchor_comparison::<AnchorOutput>()?;
    ensure!(anchor.value == 0.8);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build A-suite Tokio runtime")?;
    let output = runtime.block_on(async {
        let mut tuner = Tuner::start(anchor_config()).await?;
        let report = tuner
            .run_round(
                WINDOW_START + u64::from(WINDOW_SECONDS),
                "2026-07-15T12:10:01Z".to_owned(),
                1_784_117_401_000_000,
                &CancellationToken::new(),
            )
            .await;
        tuner.shutdown().await;
        ensure!(
            report.failed_cohorts.is_empty(),
            "tuner failed A-suite cohorts: {:?}",
            report.failed_cohorts
        );
        report
            .published
            .get(COHORT)
            .cloned()
            .context("tuner did not publish the A-suite cohort")
    })?;

    let measurement_root = fixture_root()
        .join("storage")
        .join("measurements")
        .join(COHORT)
        .join("window=20260715T1200Z");
    let intermediate = read_intermediate_pair(
        &measurement_root.join(format!("part-{PART_ULID}.truth.parquet")),
        &measurement_root.join(format!("part-{PART_ULID}.sweep.parquet")),
    )?;
    ensure!(intermediate.samples.len() == 500);
    ensure!(anchor.query_order.len() == intermediate.samples.len());

    let tuner_recalls = intermediate
        .samples
        .iter()
        .flat_map(|sample| {
            let query_id = anchor.query_order[sample.record_index as usize];
            sample
                .sweeps
                .iter()
                .map(move |(ef, sweep)| ((query_id, *ef), sweep.recall))
        })
        .collect::<BTreeMap<_, _>>();
    ensure!(tuner_recalls.len() == 2_500);
    let exact_count = anchor
        .recall_rows
        .iter()
        .filter(|row| tuner_recalls.get(&(row.query_id, row.ef)) == Some(&row.recall))
        .count();
    let exact_recall_fraction = exact_count as f64 / anchor.recall_rows.len() as f64;

    let tuner_mean = output
        .per_ef
        .iter()
        .map(|summary| (summary.ef, summary.mean_recall))
        .collect::<BTreeMap<_, _>>();
    let mean_recall_absolute_differences = anchor
        .per_ef
        .iter()
        .map(|summary| {
            Ok((tuner_mean
                .get(&summary.ef)
                .context("tuner output omitted an anchor ef")?
                - summary.mean_recall)
                .abs())
        })
        .collect::<Result<Vec<_>>>()?;

    let mut train_recalls = EF_GRID
        .into_iter()
        .map(|ef| (ef, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for sample in &intermediate.samples {
        if is_train_member(sample.vector_hash, 7, 0.7)? {
            for (ef, recalls) in &mut train_recalls {
                recalls.push(
                    sample
                        .sweeps
                        .get(ef)
                        .context("A-suite intermediate omitted an ef")?
                        .recall,
                );
            }
        }
    }
    let tuner_train_quantiles = train_recalls
        .into_iter()
        .map(|(ef, recalls)| Ok((ef, quantile_type7(&recalls, 0.10)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let train_quantile_absolute_differences = anchor
        .per_ef
        .iter()
        .map(|summary| {
            Ok((tuner_train_quantiles
                .get(&summary.ef)
                .context("tuner train quantiles omitted an anchor ef")?
                - summary.train_quantile_recall)
                .abs())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(AnchorComparison {
        exact_recall_fraction,
        mean_recall_absolute_differences,
        train_quantile_absolute_differences,
        tuner_recommended_ef: output
            .recommended_ef
            .context("A-suite output was unexpectedly insufficient")?,
        anchor_recommended_ef: anchor.recommended_ef,
        test_quantile_absolute_difference: (output
            .test_quantile_recall
            .context("A-suite output omitted the holdout quantile")?
            - anchor.test_quantile_recall)
            .abs(),
        tuner_transferred: output
            .transferred
            .context("A-suite output omitted the transfer decision")?,
        anchor_transferred: anchor.transferred,
    })
}

fn anchor_config() -> Config {
    let port = std::env::var("SEAM_PG_PORT").unwrap_or_else(|_| "55432".to_owned());
    Config {
        calibration: CalibrationConfig {
            interval: Duration::from_secs(600),
            ef_search: EF_GRID.to_vec(),
            train_fraction: 0.7,
            split_seed: 7,
            min_samples: 100,
        },
        storage: StorageConfig {
            root: fixture_root().join("storage"),
            window_seconds: WINDOW_SECONDS,
        },
        budget: BudgetConfig {
            db_share: 1.0,
            statement_timeout: Duration::from_secs(5),
            client_timeout: Duration::from_secs(10),
        },
        data_sources: BTreeMap::from([(
            "primary".to_owned(),
            DataSourceConfig {
                server: format!("127.0.0.1:{port}"),
                database: "postgres".to_owned(),
                user: "postgres".to_owned(),
                password_env: Some("SEAM_TEST_PG_PASSWORD".to_owned()),
            },
        )]),
        indexes: BTreeMap::from([(
            "fixture".to_owned(),
            IndexConfig {
                data_source: "primary".to_owned(),
                table: "docs_seam_fixture".to_owned(),
                key: "doc_id".to_owned(),
                column: "embedding".to_owned(),
            },
        )]),
        targets: BTreeMap::from([(
            "recall".to_owned(),
            TargetConfig {
                k: 10,
                value: 0.8,
                percentile: 0.90,
                window: Duration::from_secs(u64::from(WINDOW_SECONDS)),
            },
        )]),
        cohorts: BTreeMap::from([(
            CohortName::try_from(COHORT).expect("static A-suite cohort must be valid"),
            CohortConfig {
                index: "fixture".to_owned(),
                target: "recall".to_owned(),
            },
        )]),
    }
}
