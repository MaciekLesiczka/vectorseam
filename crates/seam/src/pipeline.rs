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
    #[error("database connection became unavailable while measuring part {part_ulid:?}: {error}")]
    ConnectionUnavailable { part_ulid: String, error: String },
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
            MeasurePartOutcome::ConnectionUnavailable(source_error) => {
                return Err(PipelineError::ConnectionUnavailable {
                    part_ulid: source.listed.part_ulid,
                    error: source_error,
                });
            }
            MeasurePartOutcome::Cancelled => return Ok(CohortRoundOutcome::Cancelled),
        }
    }

    if cancellation.is_cancelled() {
        return Ok(CohortRoundOutcome::Cancelled);
    }
    let (history_path, latest_path) = round_paths(&config.cohort, round_end)?;
    let previous_round = load_previous_round(store.as_ref(), &latest_path).await?;
    let input = AggregationInput {
        config,
        round_end,
        computed_at,
        phase_a_abort,
        phase_a_incompatible_parts,
        previous_round,
        listed_parts: sources.iter().map(|source| source.listed.clone()).collect(),
        intermediates: compatible.into_values().collect(),
    };
    let output = aggregate(&input)?;
    let round_bytes = Bytes::from(round_json_bytes(&output)?);
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
        let Some(part_ulid) = source_part_ulid(&path) else {
            warn!(
                path = %path,
                "ignoring source object without a canonical part-<ulid>.vseam name"
            );
            continue;
        };
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

fn source_part_ulid(path: &Path) -> Option<String> {
    let filename = path.filename()?;
    let value = filename
        .strip_prefix("part-")
        .and_then(|value| value.strip_suffix(".vseam"))?;
    value.parse::<Ulid>().map(|ulid| ulid.to_string()).ok()
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

async fn load_previous_round(
    store: &dyn ObjectStore,
    path: &Path,
) -> Result<Option<RoundOutput>, PipelineError> {
    let result = match store.get(path).await {
        Ok(result) => result,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(source) => {
            return Err(PipelineError::Storage {
                operation: "GET",
                path: path.to_string(),
                source,
            });
        }
    };
    let bytes = match result.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            warn!(
                path = %path,
                %error,
                "previous latest round body is unreadable; effective recommendation will not carry"
            );
            return Ok(None);
        }
    };
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            warn!(
                path = %path,
                %error,
                "previous latest round is malformed; effective recommendation will not carry"
            );
            return Ok(None);
        }
    };
    let has_effective_field = value
        .as_object()
        .is_some_and(|object| object.contains_key("effective"));
    let round = match serde_json::from_value(value) {
        Ok(round) => round,
        Err(error) => {
            warn!(
                path = %path,
                %error,
                "previous latest round is malformed; effective recommendation will not carry"
            );
            return Ok(None);
        }
    };
    if !has_effective_field {
        warn!(
            path = %path,
            "previous latest round predates effective recommendations; effective recommendation will not carry"
        );
        return Ok(None);
    }
    Ok(Some(round))
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
    use std::io::Write;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use futures_util::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutResult, Result as StoreResult,
    };
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::fmt::MakeWriter;
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

    struct ConnectionUnavailableMeasurer {
        calls: usize,
    }

    struct TargetUnmetMeasurer {
        calls: usize,
    }

    struct FailingMeasurer {
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
        gets: Mutex<Vec<Path>>,
        fail_next_get: Mutex<Option<Path>>,
    }

    impl RecordingStore {
        fn take_puts(&self) -> Vec<Path> {
            std::mem::take(&mut *self.puts.lock().unwrap())
        }

        fn take_gets(&self) -> Vec<Path> {
            std::mem::take(&mut *self.gets.lock().unwrap())
        }

        fn fail_next_get(&self, path: Path) {
            *self.fail_next_get.lock().unwrap() = Some(path);
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
            self.gets.lock().unwrap().push(location.clone());
            let should_fail = {
                let mut fail_next_get = self.fail_next_get.lock().unwrap();
                if fail_next_get.as_ref() == Some(location) {
                    fail_next_get.take();
                    true
                } else {
                    false
                }
            };
            if should_fail {
                return Err(object_store::Error::Generic {
                    store: "RecordingStore",
                    source: Box::new(std::io::Error::other("injected GET failure")),
                });
            }
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

    #[derive(Clone, Debug, Default)]
    struct LogCapture {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl LogCapture {
        fn contents(&self) -> String {
            String::from_utf8(self.bytes.lock().unwrap().clone()).unwrap()
        }
    }

    struct LogWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for LogWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.bytes.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'writer> MakeWriter<'writer> for LogCapture {
        type Writer = LogWriter;

        fn make_writer(&'writer self) -> Self::Writer {
            LogWriter {
                bytes: Arc::clone(&self.bytes),
            }
        }
    }

    fn warning_capture() -> (LogCapture, tracing::Dispatch) {
        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_target(false)
            .with_max_level(tracing::Level::WARN)
            .with_writer(capture.clone())
            .finish();
        (capture, tracing::Dispatch::new(subscriber))
    }

    #[async_trait]
    impl SampleMeasurer for CountingMeasurer {
        async fn measure_sample(
            &mut self,
            _vector: &[f32],
            config: &AggregationConfig,
            _cancellation: &CancellationToken,
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
            _cancellation: &CancellationToken,
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
            _cancellation: &CancellationToken,
        ) -> Result<SampleMeasurement, SampleMeasureError> {
            self.calls += 1;
            Err(SampleMeasureError::TableSmallerThanK {
                returned: usize::try_from(config.k).unwrap() - 1,
                k: config.k,
            })
        }
    }

    #[async_trait]
    impl SampleMeasurer for ConnectionUnavailableMeasurer {
        async fn measure_sample(
            &mut self,
            _vector: &[f32],
            _config: &AggregationConfig,
            _cancellation: &CancellationToken,
        ) -> Result<SampleMeasurement, SampleMeasureError> {
            self.calls += 1;
            Err(SampleMeasureError::Connection(
                "fixture connection outage".to_owned(),
            ))
        }
    }

    #[async_trait]
    impl SampleMeasurer for TargetUnmetMeasurer {
        async fn measure_sample(
            &mut self,
            _vector: &[f32],
            config: &AggregationConfig,
            _cancellation: &CancellationToken,
        ) -> Result<SampleMeasurement, SampleMeasureError> {
            self.calls += 1;
            let mut measurement = successful_measurement(config);
            for sweep in &mut measurement.sweeps {
                sweep.returned_keys.clear();
                sweep.recall = 0.0;
            }
            Ok(measurement)
        }
    }

    #[async_trait]
    impl SampleMeasurer for FailingMeasurer {
        async fn measure_sample(
            &mut self,
            _vector: &[f32],
            _config: &AggregationConfig,
            _cancellation: &CancellationToken,
        ) -> Result<SampleMeasurement, SampleMeasureError> {
            self.calls += 1;
            Err(SampleMeasureError::Database(
                "fixture durable sample failure".to_owned(),
            ))
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
        let effective = output
            .effective
            .expect("the last successful recommendation must survive the table incident");
        assert_eq!(effective.recommended_ef, 20);
        assert!(effective.carried);
        assert_eq!(effective.source_round, "2026-07-08T12:10:00Z");
    }

    #[tokio::test]
    async fn e1_carry_on_insufficient_persists_to_history_latest_and_republication() {
        let recording_store = Arc::new(RecordingStore::default());
        let store: Arc<dyn ObjectStore> = recording_store.clone();
        let vectors = fixture_vectors(0.0);
        seed_source_vectors_at(&store, WINDOW_START, FIRST_ULID, &vectors).await;
        recording_store.take_gets();
        let mut first_measurer = CountingMeasurer { calls: 0 };
        let first = run_at(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z",
            &mut first_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(first) = first else {
            panic!("first round must publish");
        };
        let first_effective = first.effective.as_ref().unwrap();
        assert_eq!(first.status, crate::model::RoundStatus::Ok);
        assert_eq!(first.recommended_ef, Some(20));
        assert_eq!(first_effective.recommended_ef, 20);
        assert!(!first_effective.carried);
        assert_eq!(first_effective.source_round, "2026-07-08T12:10:00Z");
        let (_, latest_path) =
            round_paths(COHORT, WINDOW_START + u64::from(WINDOW_SECONDS)).unwrap();
        assert_eq!(
            recording_store
                .take_gets()
                .iter()
                .filter(|path| *path == &latest_path)
                .count(),
            1
        );

        let insufficient_round_end = WINDOW_START + 2 * u64::from(WINDOW_SECONDS);
        let mut second_measurer = CountingMeasurer { calls: 0 };
        let second = run_at(
            Arc::clone(&store),
            aggregation_config(10),
            insufficient_round_end,
            "2026-07-08T12:20:00Z",
            &mut second_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(second) = second else {
            panic!("insufficient round must publish");
        };
        assert_eq!(
            second.status,
            crate::model::RoundStatus::InsufficientSamples
        );
        assert_eq!(second.recommended_ef, None);
        let second_effective = second.effective.as_ref().unwrap();
        assert_eq!(second_effective.recommended_ef, 20);
        assert_eq!(second_effective.confidence, first_effective.confidence);
        assert_eq!(second_effective.source_round, "2026-07-08T12:10:00Z");
        assert!(second_effective.carried);

        let (history_path, latest_path) = round_paths(COHORT, insufficient_round_end).unwrap();
        assert_eq!(
            recording_store
                .take_gets()
                .iter()
                .filter(|path| *path == &latest_path)
                .count(),
            1
        );
        let history: RoundOutput =
            serde_json::from_slice(&get_bytes(store.as_ref(), &history_path).await.unwrap())
                .unwrap();
        let latest: RoundOutput =
            serde_json::from_slice(&get_bytes(store.as_ref(), &latest_path).await.unwrap())
                .unwrap();
        assert_eq!(history.effective, second.effective);
        assert_eq!(latest.effective, second.effective);
        recording_store.take_gets();

        let mut republish_measurer = CountingMeasurer { calls: 0 };
        let republished = run_at(
            Arc::clone(&store),
            aggregation_config(10),
            insufficient_round_end,
            "2026-07-08T12:20:01Z",
            &mut republish_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(republished) = republished else {
            panic!("republished insufficient round must publish");
        };
        assert_eq!(republished.effective, second.effective);
        assert_eq!(
            recording_store
                .take_gets()
                .iter()
                .filter(|path| *path == &latest_path)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn e1_all_durable_sample_failures_carry_previous_effective() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source_vectors_at(&store, WINDOW_START, FIRST_ULID, &fixture_vectors(0.0)).await;
        let mut first_measurer = CountingMeasurer { calls: 0 };
        run_at(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z",
            &mut first_measurer,
        )
        .await;

        seed_source_vectors_at(
            &store,
            WINDOW_START + u64::from(WINDOW_SECONDS),
            SECOND_ULID,
            &fixture_vectors(1_000.0),
        )
        .await;
        let mut failing_measurer = FailingMeasurer { calls: 0 };
        let failed = run_at(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + 2 * u64::from(WINDOW_SECONDS),
            "2026-07-08T12:20:00Z",
            &mut failing_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(failed) = failed else {
            panic!("failure-heavy round must publish");
        };
        assert_eq!(failing_measurer.calls, 100);
        assert_eq!(
            failed.status,
            crate::model::RoundStatus::InsufficientSamples
        );
        assert_eq!(failed.samples.measured, 0);
        assert_eq!(failed.samples.failed, 100);
        assert_eq!(failed.recommended_ef, None);
        let effective = failed.effective.as_ref().unwrap();
        assert_eq!(effective.recommended_ef, 20);
        assert_eq!(effective.source_round, "2026-07-08T12:10:00Z");
        assert!(effective.carried);
    }

    #[tokio::test]
    async fn e2_newest_target_unmet_signal_wins_and_is_then_carried() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source_vectors_at(&store, WINDOW_START, FIRST_ULID, &fixture_vectors(0.0)).await;
        let mut first_measurer = CountingMeasurer { calls: 0 };
        run_at(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z",
            &mut first_measurer,
        )
        .await;

        seed_source_vectors_at(
            &store,
            WINDOW_START + u64::from(WINDOW_SECONDS),
            SECOND_ULID,
            &fixture_vectors(1_000.0),
        )
        .await;
        let mut unmet_measurer = TargetUnmetMeasurer { calls: 0 };
        let unmet = run_at(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + 2 * u64::from(WINDOW_SECONDS),
            "2026-07-08T12:20:00Z",
            &mut unmet_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(unmet) = unmet else {
            panic!("target-unmet round must publish");
        };
        assert_eq!(unmet.status, crate::model::RoundStatus::TargetUnmet);
        assert_eq!(unmet.recommended_ef, Some(40));
        let unmet_effective = unmet.effective.as_ref().unwrap();
        assert_eq!(unmet_effective.recommended_ef, 40);
        assert!(!unmet_effective.carried);
        assert_eq!(unmet_effective.source_round, "2026-07-08T12:20:00Z");

        let mut insufficient_measurer = CountingMeasurer { calls: 0 };
        let insufficient = run_at(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + 3 * u64::from(WINDOW_SECONDS),
            "2026-07-08T12:30:00Z",
            &mut insufficient_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(insufficient) = insufficient else {
            panic!("insufficient round must publish");
        };
        assert_eq!(insufficient.recommended_ef, None);
        let carried = insufficient.effective.as_ref().unwrap();
        assert_eq!(carried.recommended_ef, 40);
        assert_eq!(carried.confidence, unmet_effective.confidence);
        assert_eq!(carried.source_round, "2026-07-08T12:20:00Z");
        assert!(carried.carried);
    }

    #[tokio::test]
    async fn e3_carry_survives_fresh_pipeline_invocation_using_only_storage() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source_vectors_at(&store, WINDOW_START, FIRST_ULID, &fixture_vectors(0.0)).await;
        {
            let mut first_measurer = CountingMeasurer { calls: 0 };
            let first = run_at(
                Arc::clone(&store),
                aggregation_config(10),
                WINDOW_START + u64::from(WINDOW_SECONDS),
                "2026-07-08T12:10:00Z",
                &mut first_measurer,
            )
            .await;
            assert!(matches!(first, CohortRoundOutcome::Published(_)));
        }

        // `run_cohort_round` has no retained state; only the shared object
        // store crosses this fresh invocation boundary.
        let mut fresh_measurer = CountingMeasurer { calls: 0 };
        let restarted = run_at(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + 2 * u64::from(WINDOW_SECONDS),
            "2026-07-08T12:20:00Z",
            &mut fresh_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(restarted) = restarted else {
            panic!("restarted insufficient round must publish");
        };
        let effective = restarted.effective.as_ref().unwrap();
        assert_eq!(effective.recommended_ef, 20);
        assert_eq!(effective.source_round, "2026-07-08T12:10:00Z");
        assert!(effective.carried);
    }

    #[tokio::test]
    async fn e4_fingerprint_change_resets_effective_for_all_required_fields() {
        for changed in [
            AggregationConfig {
                cohort: "acceptance/other".to_owned(),
                ..aggregation_config(10)
            },
            AggregationConfig {
                index: "other-index".to_owned(),
                ..aggregation_config(10)
            },
            AggregationConfig {
                k: 20,
                ef_grid: vec![20, 40],
                ..aggregation_config(10)
            },
            AggregationConfig {
                ef_grid: vec![20, 60],
                ..aggregation_config(10)
            },
            AggregationConfig {
                value: 0.8,
                ..aggregation_config(10)
            },
            AggregationConfig {
                percentile: 0.9,
                ..aggregation_config(10)
            },
        ] {
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            seed_source_vectors_at(&store, WINDOW_START, FIRST_ULID, &fixture_vectors(0.0)).await;
            let mut first_measurer = CountingMeasurer { calls: 0 };
            run_at(
                Arc::clone(&store),
                aggregation_config(10),
                WINDOW_START + u64::from(WINDOW_SECONDS),
                "2026-07-08T12:10:00Z",
                &mut first_measurer,
            )
            .await;

            let mut changed_measurer = CountingMeasurer { calls: 0 };
            let changed = run_at(
                Arc::clone(&store),
                changed,
                WINDOW_START + 2 * u64::from(WINDOW_SECONDS),
                "2026-07-08T12:20:00Z",
                &mut changed_measurer,
            )
            .await;
            let CohortRoundOutcome::Published(changed) = changed else {
                panic!("changed-fingerprint round must publish");
            };
            assert_eq!(changed.recommended_ef, None);
            assert_eq!(changed.effective, None);
        }
    }

    #[tokio::test]
    async fn e5_bootstrap_content_and_get_failure_policy_preserves_effective_chain() {
        let round_end = WINDOW_START + u64::from(WINDOW_SECONDS);
        let (_, latest_path) = round_paths(COHORT, round_end).unwrap();

        let missing_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let mut missing_measurer = CountingMeasurer { calls: 0 };
        let (missing_logs, missing_subscriber) = warning_capture();
        let missing = run_at(
            Arc::clone(&missing_store),
            aggregation_config(10),
            round_end,
            "2026-07-08T12:10:00Z",
            &mut missing_measurer,
        )
        .with_subscriber(missing_subscriber)
        .await;
        let CohortRoundOutcome::Published(missing) = missing else {
            panic!("bootstrap round must publish");
        };
        assert_eq!(missing.effective, None);
        assert_eq!(missing_logs.contents(), "");
        let missing_latest: serde_json::Value = serde_json::from_slice(
            &get_bytes(missing_store.as_ref(), &latest_path)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(missing_latest["effective"], serde_json::Value::Null);

        let corrupt_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        corrupt_store
            .put(&latest_path, PutPayload::from_static(b"not round JSON"))
            .await
            .unwrap();
        let mut corrupt_measurer = CountingMeasurer { calls: 0 };
        let (corrupt_logs, corrupt_subscriber) = warning_capture();
        let corrupt = run_at(
            Arc::clone(&corrupt_store),
            aggregation_config(10),
            round_end,
            "2026-07-08T12:10:00Z",
            &mut corrupt_measurer,
        )
        .with_subscriber(corrupt_subscriber)
        .await;
        let CohortRoundOutcome::Published(corrupt) = corrupt else {
            panic!("corrupt-carry round must publish");
        };
        assert_eq!(corrupt.effective, None);
        let corrupt_logs = corrupt_logs.contents();
        assert_eq!(corrupt_logs.lines().count(), 1);
        assert!(corrupt_logs.contains("previous latest round is malformed"));
        let corrupt_latest: serde_json::Value = serde_json::from_slice(
            &get_bytes(corrupt_store.as_ref(), &latest_path)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(corrupt_latest["effective"], serde_json::Value::Null);

        let legacy_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source_vectors_at(
            &legacy_store,
            WINDOW_START,
            FIRST_ULID,
            &fixture_vectors(0.0),
        )
        .await;
        let mut legacy_seed_measurer = CountingMeasurer { calls: 0 };
        let legacy_seed = run_at(
            Arc::clone(&legacy_store),
            aggregation_config(10),
            round_end,
            "2026-07-08T12:10:00Z",
            &mut legacy_seed_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(legacy_seed) = legacy_seed else {
            panic!("legacy seed round must publish");
        };
        let mut legacy_json = serde_json::to_value(&legacy_seed).unwrap();
        legacy_json.as_object_mut().unwrap().remove("effective");
        legacy_store
            .put(
                &latest_path,
                PutPayload::from(serde_json::to_vec(&legacy_json).unwrap()),
            )
            .await
            .unwrap();

        let mut legacy_measurer = CountingMeasurer { calls: 0 };
        let (legacy_logs, legacy_subscriber) = warning_capture();
        let legacy = run_at(
            Arc::clone(&legacy_store),
            aggregation_config(10),
            WINDOW_START + 2 * u64::from(WINDOW_SECONDS),
            "2026-07-08T12:20:00Z",
            &mut legacy_measurer,
        )
        .with_subscriber(legacy_subscriber)
        .await;
        let CohortRoundOutcome::Published(legacy) = legacy else {
            panic!("pre-effective carry round must publish");
        };
        assert_eq!(legacy.effective, None);
        let legacy_logs = legacy_logs.contents();
        assert_eq!(legacy_logs.lines().count(), 1);
        assert!(legacy_logs.contains("predates effective recommendations"));
        let legacy_latest: serde_json::Value = serde_json::from_slice(
            &get_bytes(legacy_store.as_ref(), &latest_path)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(legacy_latest["effective"], serde_json::Value::Null);

        let failing_store = Arc::new(RecordingStore::default());
        let failing_store_dyn: Arc<dyn ObjectStore> = failing_store.clone();
        seed_source_vectors_at(
            &failing_store_dyn,
            WINDOW_START,
            FIRST_ULID,
            &fixture_vectors(0.0),
        )
        .await;
        let mut seed_measurer = CountingMeasurer { calls: 0 };
        let seed = run_at(
            Arc::clone(&failing_store_dyn),
            aggregation_config(10),
            round_end,
            "2026-07-08T12:10:00Z",
            &mut seed_measurer,
        )
        .await;
        let CohortRoundOutcome::Published(seed) = seed else {
            panic!("carry seed round must publish");
        };
        assert_eq!(seed.effective.as_ref().unwrap().recommended_ef, 20);
        let stored_latest = get_bytes(failing_store_dyn.as_ref(), &latest_path)
            .await
            .unwrap();
        failing_store.take_puts();
        failing_store.fail_next_get(latest_path.clone());

        let failed_round_end = WINDOW_START + 2 * u64::from(WINDOW_SECONDS);
        let mut failed_measurer = CountingMeasurer { calls: 0 };
        let failed = run_cohort_round(
            Arc::clone(&failing_store_dyn),
            aggregation_config(10),
            failed_round_end,
            "2026-07-08T12:20:00Z".to_owned(),
            failed_round_end * 1_000_000,
            &mut failed_measurer,
            &CancellationToken::new(),
        )
        .await;
        assert!(matches!(
            failed,
            Err(PipelineError::Storage {
                operation: "GET",
                ref path,
                ..
            }) if path == &latest_path.to_string()
        ));
        assert!(failing_store.take_puts().is_empty());

        let stored_after_failure = get_bytes(failing_store_dyn.as_ref(), &latest_path)
            .await
            .unwrap();
        assert_eq!(stored_after_failure, stored_latest);
        let (failed_history_path, _) = round_paths(COHORT, failed_round_end).unwrap();
        assert!(matches!(
            failing_store_dyn.get(&failed_history_path).await,
            Err(object_store::Error::NotFound { .. })
        ));
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

    #[tokio::test]
    async fn c2_connection_outage_leaves_part_unmeasured_and_next_round_retries() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source(&store, FIRST_ULID, &[0.25, 0.5]).await;
        let mut unavailable = ConnectionUnavailableMeasurer { calls: 0 };

        let first = run_cohort_round(
            Arc::clone(&store),
            aggregation_config(10),
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z".to_owned(),
            1_783_512_600_000_000,
            &mut unavailable,
            &CancellationToken::new(),
        )
        .await;

        assert!(matches!(
            first,
            Err(PipelineError::ConnectionUnavailable { .. })
        ));
        assert_eq!(unavailable.calls, 1);
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

        let mut recovered = CountingMeasurer { calls: 0 };
        let second = run(Arc::clone(&store), aggregation_config(10), &mut recovered).await;
        let CohortRoundOutcome::Published(output) = second else {
            panic!("the recovered round must retry and publish the part");
        };
        assert_eq!(recovered.calls, 1);
        assert_eq!(output.samples.measured, 1);
        assert_eq!(output.samples.failed, 0);
        assert_eq!(listed_under(store.as_ref(), "measurements").await.len(), 2);
    }

    #[tokio::test]
    async fn noncanonical_vseam_name_is_skipped_without_wedging_cohort() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        seed_source(&store, FIRST_ULID, &[0.25, 0.5]).await;
        let timestamp = format_window_timestamp(WINDOW_START).unwrap();
        let stray = Path::from(format!(
            "cohorts/{COHORT}/window={timestamp}/part-not-a-ulid.vseam"
        ));
        store
            .put(&stray, PutPayload::from_static(b"stray object"))
            .await
            .unwrap();
        let mut measurer = CountingMeasurer { calls: 0 };

        let observed = run(Arc::clone(&store), aggregation_config(10), &mut measurer).await;

        let CohortRoundOutcome::Published(output) = observed else {
            panic!("a noncanonical stray object must not abort the cohort");
        };
        assert_eq!(measurer.calls, 1);
        assert_eq!(output.samples.available, 1);
        assert_eq!(output.parts_used, 1);
    }

    async fn run(
        store: Arc<dyn ObjectStore>,
        config: AggregationConfig,
        measurer: &mut (dyn SampleMeasurer + Send),
    ) -> CohortRoundOutcome {
        run_at(
            store,
            config,
            WINDOW_START + u64::from(WINDOW_SECONDS),
            "2026-07-08T12:10:00Z",
            measurer,
        )
        .await
    }

    async fn run_at(
        store: Arc<dyn ObjectStore>,
        config: AggregationConfig,
        round_end: u64,
        computed_at: &str,
        measurer: &mut (dyn SampleMeasurer + Send),
    ) -> CohortRoundOutcome {
        run_cohort_round(
            store,
            config,
            round_end,
            computed_at.to_owned(),
            round_end * 1_000_000,
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
        seed_source_vectors_at(store, WINDOW_START, part_ulid, vectors).await;
    }

    async fn seed_source_vectors_at(
        store: &Arc<dyn ObjectStore>,
        window_start: u64,
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
                receive_time: window_start * 1_000_000 + index as u64,
                frame,
            })
            .collect::<Vec<_>>();
        let bytes = write_segment(
            &SegmentHeader {
                window_start,
                window_seconds: WINDOW_SECONDS,
                first_receive: window_start * 1_000_000,
                last_receive: window_start * 1_000_000 + vectors.len() as u64 - 1,
                received_frame_count: vectors.len() as u64,
                record_count: vectors.len() as u64,
                cohort: CohortName::try_from(COHORT).unwrap(),
            },
            &records,
        )
        .unwrap();
        let timestamp = format_window_timestamp(window_start).unwrap();
        let path = Path::from(format!(
            "cohorts/{COHORT}/window={timestamp}/part-{part_ulid}.vseam"
        ));
        store.put(&path, PutPayload::from(bytes)).await.unwrap();
    }

    fn fixture_vectors(offset: f32) -> Vec<Vec<f32>> {
        (0..100)
            .map(|value| vec![offset + value as f32, 1.0])
            .collect()
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
            ground_truth_latency_ms: 400.5,
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
