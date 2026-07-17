//! Durable one-cohort measure/aggregate pipeline.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use futures_util::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use tracing::warn;
use ulid::Ulid;
use vectorseam_core::window::{WindowError, format_window_timestamp};

use crate::accounting::in_scope_window_starts;
use crate::aggregate::{AggregateError, aggregate, is_compatible, round_json_bytes};
use crate::intermediate::{
    IntermediateError, encode_intermediate_pair, read_intermediate_pair_bytes,
};
use crate::measure::{
    MeasurePartOutcome, PreparePartError, SampleMeasurer, measure_prepared_part, prepare_segment,
};
use crate::model::{
    AggregationConfig, AggregationInput, IntermediatePart, ListedPart, RoundOutput,
};

/// A cohort round either published completely or stopped before publication.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CohortRoundOutcome {
    Published(Box<RoundOutput>),
    Cancelled,
}

/// A storage or deterministic pipeline failure that aborts one cohort round.
#[derive(Debug, Error)]
pub(crate) enum PipelineError {
    #[error(transparent)]
    Window(#[from] WindowError),
    #[error(transparent)]
    Aggregate(#[from] AggregateError),
    #[error("storage {operation} failed for {path}: {source}")]
    Storage {
        operation: &'static str,
        path: String,
        #[source]
        source: object_store::Error,
    },
    #[error("source object {path:?} does not have a canonical part-<ulid>.vseam name")]
    InvalidSourceName { path: String },
    #[error("part ULID {part_ulid:?} appears in more than one source object")]
    DuplicateSourcePart { part_ulid: String },
    #[error("source segment {path:?} is invalid: {source}")]
    SourceSegment {
        path: String,
        #[source]
        source: PreparePartError,
    },
    #[error("could not encode intermediate pair for part {part_ulid:?}: {source}")]
    EncodeIntermediate {
        part_ulid: String,
        #[source]
        source: IntermediateError,
    },
    #[error("newly encoded intermediate pair for part {part_ulid:?} could not be read: {source}")]
    ReadEncodedIntermediate {
        part_ulid: String,
        #[source]
        source: IntermediateError,
    },
    #[error("incompatible part counter overflow")]
    IncompatibleCounterOverflow,
}

#[derive(Clone, Debug)]
struct SourcePart {
    path: Path,
    listed: ListedPart,
}

pub(crate) async fn run_cohort_round(
    store: Arc<dyn ObjectStore>,
    config: AggregationConfig,
    round_end: u64,
    computed_at: String,
    computed_at_us: u64,
    measurer: &mut (dyn SampleMeasurer + Send),
    cancellation: &CancellationToken,
) -> Result<CohortRoundOutcome, PipelineError> {
    let window_starts = in_scope_window_starts(
        round_end,
        config.window_duration_seconds,
        config.storage_window_seconds,
    )?;
    let source_objects =
        list_source_objects(store.as_ref(), &config.cohort, &window_starts).await?;
    let measurement_objects =
        list_measurement_objects(store.as_ref(), &config.cohort, &window_starts).await?;
    let sources = load_source_headers(
        store.as_ref(),
        &config.cohort,
        config.storage_window_seconds,
        source_objects,
    )
    .await?;

    let mut compatible = BTreeMap::<String, IntermediatePart>::new();
    let mut pending = Vec::<SourcePart>::new();
    let mut phase_a_incompatible_parts = 0_u64;
    for source in &sources {
        let (truth_path, sweep_path) = intermediate_paths(
            &config.cohort,
            source.listed.window_start,
            &source.listed.part_ulid,
        )?;
        if measurement_objects.contains(&truth_path) && measurement_objects.contains(&sweep_path) {
            let truth_bytes = get_bytes(store.as_ref(), &truth_path).await?;
            let sweep_bytes = get_bytes(store.as_ref(), &sweep_path).await?;
            match read_intermediate_pair_bytes(truth_bytes, sweep_bytes) {
                Ok(intermediate)
                    if matches_source_metadata(&intermediate, source, &config.cohort)
                        && is_compatible(&config, &intermediate) =>
                {
                    compatible.insert(source.listed.part_ulid.clone(), intermediate);
                    continue;
                }
                Ok(intermediate)
                    if matches_source_metadata(&intermediate, source, &config.cohort) =>
                {
                    phase_a_incompatible_parts = phase_a_incompatible_parts
                        .checked_add(1)
                        .ok_or(PipelineError::IncompatibleCounterOverflow)?;
                }
                Ok(_structurally_mismatched) => {
                    warn!(
                        part_ulid = %source.listed.part_ulid,
                        "intermediate metadata does not match its source part; remeasuring"
                    );
                }
                Err(error) => {
                    warn!(
                        part_ulid = %source.listed.part_ulid,
                        %error,
                        "malformed intermediate pair; remeasuring"
                    );
                }
            }
        }
        pending.push(source.clone());
    }

    let mut phase_a_abort = None;
    for source in pending {
        if cancellation.is_cancelled() {
            return Ok(CohortRoundOutcome::Cancelled);
        }
        let bytes = get_bytes(store.as_ref(), &source.path).await?;
        let prepared = prepare_segment(
            &bytes,
            &config.cohort,
            source.listed.window_start,
            config.storage_window_seconds,
        )
        .map_err(|source_error| PipelineError::SourceSegment {
            path: source.path.to_string(),
            source: source_error,
        })?;
        match measure_prepared_part(
            &prepared,
            &source.listed.part_ulid,
            &config,
            computed_at_us,
            measurer,
            cancellation,
        )
        .await
        .map_err(|source_error| PipelineError::SourceSegment {
            path: source.path.to_string(),
            source: source_error,
        })? {
            MeasurePartOutcome::Complete(measured) => {
                let (truth_bytes, sweep_bytes) = encode_intermediate_pair(
                    &measured.metadata,
                    &measured.truth_rows,
                    &measured.sweep_rows,
                )
                .map_err(|source_error| PipelineError::EncodeIntermediate {
                    part_ulid: source.listed.part_ulid.clone(),
                    source: source_error,
                })?;
                let (truth_path, sweep_path) = intermediate_paths(
                    &config.cohort,
                    source.listed.window_start,
                    &source.listed.part_ulid,
                )?;
                put_bytes(store.as_ref(), &truth_path, truth_bytes.clone()).await?;
                put_bytes(store.as_ref(), &sweep_path, sweep_bytes.clone()).await?;
                let intermediate = read_intermediate_pair_bytes(truth_bytes, sweep_bytes).map_err(
                    |source_error| PipelineError::ReadEncodedIntermediate {
                        part_ulid: source.listed.part_ulid.clone(),
                        source: source_error,
                    },
                )?;
                compatible.insert(source.listed.part_ulid.clone(), intermediate);
            }
            MeasurePartOutcome::TableSmallerThanK(abort) => {
                phase_a_abort = Some(abort);
                break;
            }
            MeasurePartOutcome::Cancelled => return Ok(CohortRoundOutcome::Cancelled),
        }
    }

    if cancellation.is_cancelled() {
        return Ok(CohortRoundOutcome::Cancelled);
    }
    let input = AggregationInput {
        config,
        round_end,
        computed_at,
        phase_a_abort,
        phase_a_incompatible_parts,
        listed_parts: sources.iter().map(|source| source.listed.clone()).collect(),
        intermediates: compatible.into_values().collect(),
    };
    let output = aggregate(&input)?;
    let round_bytes = Bytes::from(round_json_bytes(&output)?);
    let (history_path, latest_path) = round_paths(&input.config.cohort, round_end)?;
    put_bytes(store.as_ref(), &history_path, round_bytes.clone()).await?;
    put_bytes(store.as_ref(), &latest_path, round_bytes).await?;
    Ok(CohortRoundOutcome::Published(Box::new(output)))
}

async fn list_source_objects(
    store: &dyn ObjectStore,
    cohort: &str,
    window_starts: &[u64],
) -> Result<Vec<(u64, Path)>, PipelineError> {
    let mut objects = Vec::new();
    for &window_start in window_starts {
        let timestamp = format_window_timestamp(window_start)?;
        let prefix = Path::from(format!("cohorts/{cohort}/window={timestamp}"));
        let listed = store
            .list(Some(&prefix))
            .map_err(|source| PipelineError::Storage {
                operation: "list",
                path: prefix.to_string(),
                source,
            })
            .try_collect::<Vec<_>>()
            .await?;
        objects.extend(
            listed
                .into_iter()
                .filter(|meta| meta.location.extension() == Some("vseam"))
                .map(|meta| (window_start, meta.location)),
        );
    }
    objects.sort_by(|left, right| left.1.cmp(&right.1));
    Ok(objects)
}

async fn list_measurement_objects(
    store: &dyn ObjectStore,
    cohort: &str,
    window_starts: &[u64],
) -> Result<BTreeSet<Path>, PipelineError> {
    let mut objects = BTreeSet::new();
    for &window_start in window_starts {
        let timestamp = format_window_timestamp(window_start)?;
        let prefix = Path::from(format!("measurements/{cohort}/window={timestamp}"));
        let listed = store
            .list(Some(&prefix))
            .map_err(|source| PipelineError::Storage {
                operation: "list",
                path: prefix.to_string(),
                source,
            })
            .try_collect::<Vec<_>>()
            .await?;
        objects.extend(listed.into_iter().map(|meta| meta.location));
    }
    Ok(objects)
}

async fn load_source_headers(
    store: &dyn ObjectStore,
    expected_cohort: &str,
    expected_window_seconds: u32,
    objects: Vec<(u64, Path)>,
) -> Result<Vec<SourcePart>, PipelineError> {
    let mut by_ulid = BTreeMap::<String, SourcePart>::new();
    for (window_start, path) in objects {
        let part_ulid = source_part_ulid(&path)?;
        let bytes = get_bytes(store, &path).await?;
        let prepared = prepare_segment(
            &bytes,
            expected_cohort,
            window_start,
            expected_window_seconds,
        )
        .map_err(|source| PipelineError::SourceSegment {
            path: path.to_string(),
            source,
        })?;
        let source = SourcePart {
            path,
            listed: ListedPart {
                part_ulid: part_ulid.clone(),
                window_start: prepared.window_start,
                window_seconds: prepared.window_seconds,
                received_frame_count: prepared.received_frame_count,
                record_count: prepared.record_count,
            },
        };
        if by_ulid.insert(part_ulid.clone(), source).is_some() {
            return Err(PipelineError::DuplicateSourcePart { part_ulid });
        }
    }
    Ok(by_ulid.into_values().collect())
}

fn source_part_ulid(path: &Path) -> Result<String, PipelineError> {
    let filename = path
        .filename()
        .ok_or_else(|| PipelineError::InvalidSourceName {
            path: path.to_string(),
        })?;
    let value = filename
        .strip_prefix("part-")
        .and_then(|value| value.strip_suffix(".vseam"))
        .ok_or_else(|| PipelineError::InvalidSourceName {
            path: path.to_string(),
        })?;
    value
        .parse::<Ulid>()
        .map(|ulid| ulid.to_string())
        .map_err(|_error| PipelineError::InvalidSourceName {
            path: path.to_string(),
        })
}

fn intermediate_paths(
    cohort: &str,
    window_start: u64,
    part_ulid: &str,
) -> Result<(Path, Path), WindowError> {
    let timestamp = format_window_timestamp(window_start)?;
    let prefix = format!("measurements/{cohort}/window={timestamp}/part-{part_ulid}");
    Ok((
        Path::from(format!("{prefix}.truth.parquet")),
        Path::from(format!("{prefix}.sweep.parquet")),
    ))
}

fn round_paths(cohort: &str, round_end: u64) -> Result<(Path, Path), WindowError> {
    let timestamp = format_window_timestamp(round_end)?;
    Ok((
        Path::from(format!("calibrations/{cohort}/round-{timestamp}.json")),
        Path::from(format!("calibrations/{cohort}/latest.json")),
    ))
}

fn matches_source_metadata(
    intermediate: &IntermediatePart,
    source: &SourcePart,
    cohort: &str,
) -> bool {
    let metadata = &intermediate.metadata;
    metadata.cohort == cohort
        && metadata.part_ulid == source.listed.part_ulid
        && metadata.window_start == source.listed.window_start
        && metadata.window_seconds == source.listed.window_seconds
        && metadata.received_frame_count == source.listed.received_frame_count
        && metadata.record_count == source.listed.record_count
}

async fn get_bytes(store: &dyn ObjectStore, path: &Path) -> Result<Bytes, PipelineError> {
    let result = store
        .get(path)
        .await
        .map_err(|source| PipelineError::Storage {
            operation: "GET",
            path: path.to_string(),
            source,
        })?;
    result
        .bytes()
        .await
        .map_err(|source| PipelineError::Storage {
            operation: "GET body",
            path: path.to_string(),
            source,
        })
}

async fn put_bytes(
    store: &dyn ObjectStore,
    path: &Path,
    bytes: Bytes,
) -> Result<(), PipelineError> {
    store
        .put(path, PutPayload::from(bytes))
        .await
        .map(|_result| ())
        .map_err(|source| PipelineError::Storage {
            operation: "PUT",
            path: path.to_string(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use futures_util::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutResult, Result as StoreResult,
    };
    use vectorseam_core::cohort::CohortName;
    use vectorseam_core::frame::{FIXED_FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_VERSION};
    use vectorseam_core::segment::{SegmentHeader, SegmentRecordRef, write_segment};

    use crate::measure::{SampleMeasureError, SampleMeasurement, SampleSweepResult};

    const COHORT: &str = "acceptance/f-agg";
    const WINDOW_START: u64 = 1_783_512_000;
    const WINDOW_SECONDS: u32 = 600;
    const FIRST_ULID: &str = "01J00000000000000000000000";
    const SECOND_ULID: &str = "01J00000000000000000000001";

    struct CountingMeasurer {
        calls: usize,
    }

    struct TableSmallerMeasurer {
        calls: usize,
    }

    struct CancellingMeasurer {
        calls: usize,
        cancellation: CancellationToken,
    }

    #[derive(Debug, Default)]
    struct RecordingStore {
        inner: InMemory,
        puts: Mutex<Vec<Path>>,
    }

    impl RecordingStore {
        fn take_puts(&self) -> Vec<Path> {
            std::mem::take(&mut *self.puts.lock().unwrap())
        }
    }

    impl fmt::Display for RecordingStore {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("RecordingStore")
        }
    }

    #[async_trait]
    impl ObjectStore for RecordingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> StoreResult<PutResult> {
            self.puts.lock().unwrap().push(location.clone());
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> StoreResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, StoreResult<Path>>,
        ) -> BoxStream<'static, StoreResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[async_trait]
    impl SampleMeasurer for CountingMeasurer {
        async fn measure_sample(
            &mut self,
            _vector: &[f32],
            config: &AggregationConfig,
        ) -> Result<SampleMeasurement, SampleMeasureError> {
            self.calls += 1;
            Ok(successful_measurement(config))
        }
    }

    #[async_trait]
    impl SampleMeasurer for CancellingMeasurer {
        async fn measure_sample(
            &mut self,
            _vector: &[f32],
            config: &AggregationConfig,
        ) -> Result<SampleMeasurement, SampleMeasureError> {
            self.calls += 1;
            self.cancellation.cancel();
            Ok(successful_measurement(config))
        }
    }

    #[async_trait]
    impl SampleMeasurer for TableSmallerMeasurer {
        async fn measure_sample(
            &mut self,
            _vector: &[f32],
            config: &AggregationConfig,
        ) -> Result<SampleMeasurement, SampleMeasureError> {
            self.calls += 1;
            Err(SampleMeasureError::TableSmallerThanK {
                returned: usize::try_from(config.k).unwrap() - 1,
                k: config.k,
            })
        }
    }

    #[tokio::test]
    async fn c1_resume_mid_part_rewrites_pair_and_matches_clean_run() {
        let recording_store = Arc::new(RecordingStore::default());
        let resumed_store: Arc<dyn ObjectStore> = recording_store.clone();
        seed_source(&resumed_store, FIRST_ULID, &[0.25, 0.5]).await;
        seed_source(&resumed_store, SECOND_ULID, &[0.75, 1.0]).await;
        let mut initial_measurer = CountingMeasurer { calls: 0 };
        let initial = run(
            Arc::clone(&resumed_store),
            aggregation_config(10),
            &mut initial_measurer,
        )
        .await;
        assert!(matches!(initial, CohortRoundOutcome::Published(_)));
        assert_eq!(initial_measurer.calls, 2);

        let (truth_path, sweep_path) =
            intermediate_paths(COHORT, WINDOW_START, FIRST_ULID).unwrap();
        resumed_store
            .put(&truth_path, PutPayload::from_static(b"interrupted truth"))
            .await
            .unwrap();
        resumed_store.delete(&sweep_path).await.unwrap();
        recording_store.take_puts();

        let mut resumed_measurer = CountingMeasurer { calls: 0 };
        let resumed = run(
            Arc::clone(&resumed_store),
            aggregation_config(10),
            &mut resumed_measurer,
        )
        .await;
        assert_eq!(resumed_measurer.calls, 1);
        let resumed_puts = recording_store.take_puts();
        let truth_position = resumed_puts
            .iter()
            .position(|path| path == &truth_path)
            .unwrap();
        let sweep_position = resumed_puts
            .iter()
            .position(|path| path == &sweep_path)
            .unwrap();
        assert!(truth_position < sweep_position);
        let CohortRoundOutcome::Published(resumed_output) = resumed else {
            panic!("resumed round must publish");
        };
        let repaired_truth = get_bytes(resumed_store.as_ref(), &truth_path)
            .await
            .unwrap();
        let repaired_sweep = get_bytes(resumed_store.as_ref(), &sweep_path)
            .await
            .unwrap();
        read_intermediate_pair_bytes(repaired_truth, repaired_sweep).unwrap();

        let clean_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source(&clean_store, FIRST_ULID, &[0.25, 0.5]).await;
        seed_source(&clean_store, SECOND_ULID, &[0.75, 1.0]).await;
        let mut clean_measurer = CountingMeasurer { calls: 0 };
        let clean = run(
            Arc::clone(&clean_store),
            aggregation_config(10),
            &mut clean_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(clean_output) = clean else {
            panic!("clean round must publish");
        };
        assert_eq!(clean_measurer.calls, 2);
        assert_eq!(
            round_json_bytes(&resumed_output).unwrap(),
            round_json_bytes(&clean_output).unwrap()
        );

        let (history, latest) =
            round_paths(COHORT, WINDOW_START + u64::from(WINDOW_SECONDS)).unwrap();
        let history_position = resumed_puts
            .iter()
            .position(|path| path == &history)
            .unwrap();
        let latest_position = resumed_puts
            .iter()
            .position(|path| path == &latest)
            .unwrap();
        assert!(history_position < latest_position);
        assert_eq!(
            get_bytes(resumed_store.as_ref(), &history).await.unwrap(),
            get_bytes(resumed_store.as_ref(), &latest).await.unwrap()
        );
    }

    #[tokio::test]
    async fn c3_config_fingerprint_remeasures_and_retains_incompatible_count() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source(&store, FIRST_ULID, &[0.25, 0.5]).await;
        let mut first_measurer = CountingMeasurer { calls: 0 };
        run(
            Arc::clone(&store),
            aggregation_config(10),
            &mut first_measurer,
        )
        .await;
        assert_eq!(first_measurer.calls, 1);

        let mut changed_measurer = CountingMeasurer { calls: 0 };
        let changed = run(
            Arc::clone(&store),
            aggregation_config(20),
            &mut changed_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(output) = changed else {
            panic!("changed-config round must publish");
        };
        assert_eq!(changed_measurer.calls, 1);
        assert_eq!(output.target.k, 20);
        assert_eq!(output.incompatible_parts, 1);
        assert_eq!(output.parts_used, 1);
    }

    #[tokio::test]
    async fn b9_second_round_issues_zero_new_database_transactions() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source(&store, FIRST_ULID, &[0.25, 0.5]).await;
        let mut first_measurer = CountingMeasurer { calls: 0 };
        run(
            Arc::clone(&store),
            aggregation_config(10),
            &mut first_measurer,
        )
        .await;
        assert_eq!(first_measurer.calls, 1);

        let mut second_measurer = CountingMeasurer { calls: 0 };
        run(
            Arc::clone(&store),
            aggregation_config(10),
            &mut second_measurer,
        )
        .await;
        assert_eq!(second_measurer.calls, 0);
    }

    #[tokio::test]
    async fn c6_table_smaller_than_k_stops_after_first_scan_despite_cached_population() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let cached_vectors = (0..100)
            .map(|value| vec![value as f32, 1.0])
            .collect::<Vec<_>>();
        seed_source_vectors(&store, SECOND_ULID, &cached_vectors).await;
        let mut initial_measurer = CountingMeasurer { calls: 0 };
        run(
            Arc::clone(&store),
            aggregation_config(10),
            &mut initial_measurer,
        )
        .await;
        assert_eq!(initial_measurer.calls, 100);

        seed_source(&store, FIRST_ULID, &[1_000.0, 1.0]).await;
        let mut smaller = TableSmallerMeasurer { calls: 0 };
        let observed = run(Arc::clone(&store), aggregation_config(10), &mut smaller).await;
        let CohortRoundOutcome::Published(output) = observed else {
            panic!("table-smaller round must publish an insufficient record");
        };
        assert_eq!(smaller.calls, 1);
        assert_eq!(output.samples.unique, 100);
        assert_eq!(
            output.status,
            crate::model::RoundStatus::InsufficientSamples
        );
        assert!(output.error.is_some());
        assert_eq!(output.recommended_ef, None);
    }

    #[tokio::test]
    async fn graceful_shutdown_finishes_in_flight_sample_and_abandons_partial_part() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source_vectors(&store, FIRST_ULID, &[vec![0.25, 0.5], vec![0.75, 1.0]]).await;
        let cancellation = CancellationToken::new();
        let mut measurer = CancellingMeasurer {
            calls: 0,
            cancellation: cancellation.clone(),
        };
        let outcome = run_cohort_round(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z".to_owned(),
            1_783_512_600_000_000,
            &mut measurer,
            &cancellation,
        )
        .await
        .unwrap();

        assert_eq!(outcome, CohortRoundOutcome::Cancelled);
        assert_eq!(measurer.calls, 1);
        assert!(
            listed_under(store.as_ref(), "measurements")
                .await
                .is_empty()
        );
        assert!(
            listed_under(store.as_ref(), "calibrations")
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn invalid_source_segment_aborts_without_publishing() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let timestamp = format_window_timestamp(WINDOW_START).unwrap();
        let path = Path::from(format!(
            "cohorts/{COHORT}/window={timestamp}/part-{FIRST_ULID}.vseam"
        ));
        store
            .put(&path, PutPayload::from_static(b"corrupt source"))
            .await
            .unwrap();
        let mut measurer = CountingMeasurer { calls: 0 };
        let result = run_cohort_round(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z".to_owned(),
            1_783_512_600_000_000,
            &mut measurer,
            &CancellationToken::new(),
        )
        .await;

        assert!(matches!(result, Err(PipelineError::SourceSegment { .. })));
        assert_eq!(measurer.calls, 0);
        assert!(
            listed_under(store.as_ref(), "calibrations")
                .await
                .is_empty()
        );
    }

    async fn run(
        store: Arc<dyn ObjectStore>,
        config: AggregationConfig,
        measurer: &mut (dyn SampleMeasurer + Send),
    ) -> CohortRoundOutcome {
        run_cohort_round(
            store,
            config,
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z".to_owned(),
            1_783_512_600_000_000,
            measurer,
            &CancellationToken::new(),
        )
        .await
        .unwrap()
    }

    async fn seed_source(store: &Arc<dyn ObjectStore>, part_ulid: &str, vector: &[f32]) {
        seed_source_vectors(store, part_ulid, &[vector.to_vec()]).await;
    }

    async fn seed_source_vectors(
        store: &Arc<dyn ObjectStore>,
        part_ulid: &str,
        vectors: &[Vec<f32>],
    ) {
        let frames = vectors
            .iter()
            .map(|vector| frame(COHORT, vector))
            .collect::<Vec<_>>();
        let records = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| SegmentRecordRef {
                receive_time: WINDOW_START * 1_000_000 + index as u64,
                frame,
            })
            .collect::<Vec<_>>();
        let bytes = write_segment(
            &SegmentHeader {
                window_start: WINDOW_START,
                window_seconds: WINDOW_SECONDS,
                first_receive: WINDOW_START * 1_000_000,
                last_receive: WINDOW_START * 1_000_000 + vectors.len() as u64 - 1,
                received_frame_count: vectors.len() as u64,
                record_count: vectors.len() as u64,
                cohort: CohortName::try_from(COHORT).unwrap(),
            },
            &records,
        )
        .unwrap();
        let timestamp = format_window_timestamp(WINDOW_START).unwrap();
        let path = Path::from(format!(
            "cohorts/{COHORT}/window={timestamp}/part-{part_ulid}.vseam"
        ));
        store.put(&path, PutPayload::from(bytes)).await.unwrap();
    }

    fn aggregation_config(k: u32) -> AggregationConfig {
        AggregationConfig {
            cohort: COHORT.to_owned(),
            target_name: "recall".to_owned(),
            index: "fixture".to_owned(),
            table: "docs_fixture".to_owned(),
            column: "embedding".to_owned(),
            key: "doc_id".to_owned(),
            k,
            value: 0.9,
            percentile: 0.95,
            window_duration_seconds: u64::from(WINDOW_SECONDS),
            storage_window_seconds: WINDOW_SECONDS,
            ef_grid: if k == 10 {
                vec![20, 40]
            } else {
                vec![20, 40, 80]
            },
            train_fraction: 0.7,
            split_seed: 7,
            min_samples: 100,
        }
    }

    fn successful_measurement(config: &AggregationConfig) -> SampleMeasurement {
        let keys = (1..=i64::from(config.k)).collect::<Vec<_>>();
        SampleMeasurement {
            gt_keys: keys.clone(),
            gt_distances: (0..config.k).map(|value| f64::from(value) / 10.0).collect(),
            sweeps: config
                .ef_grid
                .iter()
                .map(|ef| SampleSweepResult {
                    ef: *ef,
                    returned_keys: keys.clone(),
                    recall: 1.0,
                    latency_ms: 0.5,
                })
                .collect(),
        }
    }

    async fn listed_under(store: &dyn ObjectStore, prefix: &str) -> Vec<ObjectMeta> {
        store
            .list(Some(&Path::from(prefix)))
            .try_collect()
            .await
            .unwrap()
    }

    fn frame(cohort: &str, vector: &[f32]) -> Vec<u8> {
        let name = cohort.as_bytes();
        let vector_bytes = vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let total_len = FIXED_FRAME_HEADER_LEN + name.len() + vector_bytes.len();
        let mut bytes = Vec::with_capacity(total_len);
        bytes.extend_from_slice(&u32::try_from(total_len - 4).unwrap().to_le_bytes());
        bytes.extend_from_slice(&FRAME_MAGIC);
        bytes.extend_from_slice(&FRAME_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(name.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(vector.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(vector_bytes.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&vector_bytes);
        bytes
    }
}
