//! Phase A segment preparation and per-part measurement.

use std::collections::HashMap;

use async_trait::async_trait;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use vectorseam_core::frame::{FIXED_FRAME_HEADER_LEN, FrameError, parse_frame_header};
use vectorseam_core::segment::{SegmentError, read_segment};

use crate::intermediate::{SweepMeasurementRow, TruthMeasurement};
use crate::math::fnv1a64;
use crate::model::{AggregationConfig, IntermediateMetadata, PhaseAAbort};

const F32_DTYPE: u32 = 1;
const F32_BYTES: usize = std::mem::size_of::<f32>();

/// One database sweep result for a successfully measured sample.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SampleSweepResult {
    /// Validated ef value.
    pub ef: i32,
    /// ANN keys in result order.
    pub returned_keys: Vec<i64>,
    /// Recall computed against the transaction's ground truth.
    pub recall: f64,
    /// Client-observed ANN statement duration.
    pub latency_ms: f64,
}

/// One complete, snapshot-consistent database measurement.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SampleMeasurement {
    /// Exact keys ordered by distance and key.
    pub gt_keys: Vec<i64>,
    /// Exact distances matching `gt_keys`.
    pub gt_distances: Vec<f64>,
    /// Client-observed ground-truth statement duration.
    pub ground_truth_latency_ms: f64,
    /// One ANN observation for each configured ef value.
    pub sweeps: Vec<SampleSweepResult>,
}

/// A per-sample database outcome relevant to Phase A control flow.
#[derive(Debug, Error)]
pub(crate) enum SampleMeasureError {
    /// A database operation failed; the sample is counted and the part continues.
    #[error("database sample measurement failed: {0}")]
    Database(String),
    /// The connection became unusable; the current part must remain unmeasured.
    #[error("database connection unavailable: {0}")]
    Connection(String),
    /// Cancellation arrived during the cooldown before a transaction started.
    #[error("sample measurement cancelled before transaction start")]
    CancelledBeforeTransaction,
    /// The exact query proved that fewer than `k` rows are visible.
    #[error("table returned {returned} rows, fewer than k={k}")]
    TableSmallerThanK {
        /// Exact rows returned.
        returned: usize,
        /// Required recall denominator.
        k: u32,
    },
}

/// The database seam consumed by deterministic per-part orchestration.
#[async_trait]
pub(crate) trait SampleMeasurer {
    /// Measures one decoded f32 query vector.
    async fn measure_sample(
        &mut self,
        vector: &[f32],
        config: &AggregationConfig,
        cancellation: &CancellationToken,
    ) -> Result<SampleMeasurement, SampleMeasureError>;
}

/// One segment after exact-byte within-part deduplication.
#[derive(Clone, Debug)]
pub(crate) struct PreparedPart {
    /// Segment cohort.
    pub cohort: String,
    /// Segment window start.
    pub window_start: u64,
    /// Segment window width.
    pub window_seconds: u32,
    /// Collector-received frames represented by the part.
    pub received_frame_count: u64,
    /// Stored records before deduplication.
    pub record_count: u64,
    samples: Vec<PreparedSample>,
}

impl PreparedPart {
    /// First-occurrence ordinals and duplicate counts, in record order.
    #[cfg(test)]
    pub(crate) fn sample_identities(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.samples
            .iter()
            .map(|sample| (sample.record_index, sample.dup_count))
    }
}

#[derive(Clone, Debug)]
struct PreparedSample {
    record_index: i32,
    dup_count: i32,
    receive_time_us: Result<i64, SampleFrameError>,
    vector_hash: u64,
    vector: Result<Vec<f32>, SampleFrameError>,
}

#[derive(Clone, Debug, Error)]
enum SampleFrameError {
    #[error("frame cohort does not match segment cohort")]
    CohortMismatch,
    #[error("unsupported frame dtype {0}; only F32 dtype 1 is supported")]
    UnsupportedDtype(u32),
    #[error("f32 vector length does not match declared dimension")]
    DimensionMismatch,
    #[error("receive timestamp does not fit the truth parquet schema")]
    ReceiveTimeOutOfRange,
}

/// A structurally invalid or mismatched `.vseam` part.
#[derive(Debug, Error)]
pub(crate) enum PreparePartError {
    /// The core segment parser rejected the bytes.
    #[error(transparent)]
    Segment(#[from] SegmentError),
    /// A frame header could not be reparsed while locating its vector payload.
    #[error(transparent)]
    Frame(#[from] FrameError),
    /// The segment belongs to a different cohort.
    #[error("segment cohort {observed:?} does not match expected cohort {expected:?}")]
    Cohort {
        /// Configured cohort.
        expected: String,
        /// Segment header cohort.
        observed: String,
    },
    /// The segment belongs to a different storage window.
    #[error("segment window start {observed} does not match expected {expected}")]
    WindowStart {
        /// Listed prefix window.
        expected: u64,
        /// Segment header window.
        observed: u64,
    },
    /// The segment window width differs from storage configuration.
    #[error("segment window seconds {observed} does not match expected {expected}")]
    WindowSeconds {
        /// Configured storage window.
        expected: u32,
        /// Segment header storage window.
        observed: u32,
    },
    /// Header record count did not match parsed records.
    #[error("segment record_count {declared} does not match {actual} parsed records")]
    RecordCount {
        /// Header value.
        declared: u64,
        /// Parsed record count.
        actual: usize,
    },
    /// Header accounting claimed fewer received frames than stored records.
    #[error("segment received_frame_count {received} is smaller than record_count {record_count}")]
    ReceivedFrameCount {
        /// Frames accepted before collector-side drops.
        received: u64,
        /// Frames stored in the part.
        record_count: u64,
    },
    /// A record ordinal did not fit the durable int32 identity.
    #[error("segment record index {0} does not fit int32")]
    RecordIndex(usize),
    /// A duplicate count did not fit the durable int32 field.
    #[error("segment duplicate count does not fit int32")]
    DuplicateCount,
    /// A sweep result count did not fit the durable int32 field.
    #[error("sweep result count {0} does not fit int32")]
    ResultCount(usize),
    /// The measured-row count did not fit the durable uint64 metadata field.
    #[error("measured row count {0} does not fit uint64")]
    MeasuredCount(usize),
}

/// A fully measured part ready for truth-then-sweep persistence.
#[derive(Clone, Debug)]
pub(crate) struct MeasuredPart {
    pub(crate) metadata: IntermediateMetadata,
    pub(crate) truth_rows: Vec<TruthMeasurement>,
    pub(crate) sweep_rows: Vec<SweepMeasurementRow>,
}

/// Outcome of measuring one prepared part.
#[derive(Clone, Debug)]
pub(crate) enum MeasurePartOutcome {
    Complete(Box<MeasuredPart>),
    TableSmallerThanK(PhaseAAbort),
    ConnectionUnavailable(String),
    Cancelled,
}

/// Parses, validates, and deduplicates one segment without database access.
pub(crate) fn prepare_segment(
    bytes: &[u8],
    expected_cohort: &str,
    expected_window_start: u64,
    expected_window_seconds: u32,
) -> Result<PreparedPart, PreparePartError> {
    let segment = read_segment(bytes)?;
    let observed_cohort = segment.header.cohort.as_str();
    if observed_cohort != expected_cohort {
        return Err(PreparePartError::Cohort {
            expected: expected_cohort.to_owned(),
            observed: observed_cohort.to_owned(),
        });
    }
    if segment.header.window_start != expected_window_start {
        return Err(PreparePartError::WindowStart {
            expected: expected_window_start,
            observed: segment.header.window_start,
        });
    }
    if segment.header.window_seconds != expected_window_seconds {
        return Err(PreparePartError::WindowSeconds {
            expected: expected_window_seconds,
            observed: segment.header.window_seconds,
        });
    }
    if u64::try_from(segment.records.len()) != Ok(segment.header.record_count) {
        return Err(PreparePartError::RecordCount {
            declared: segment.header.record_count,
            actual: segment.records.len(),
        });
    }
    if segment.header.received_frame_count < segment.header.record_count {
        return Err(PreparePartError::ReceivedFrameCount {
            received: segment.header.received_frame_count,
            record_count: segment.header.record_count,
        });
    }

    let mut first_by_vector = HashMap::<Vec<u8>, usize>::new();
    let mut samples = Vec::<PreparedSample>::new();
    for (record_index, record) in segment.records.iter().enumerate() {
        let header = parse_frame_header(&record.frame)?;
        let vector_start = FIXED_FRAME_HEADER_LEN
            .checked_add(
                usize::try_from(header.name_len)
                    .map_err(|_error| PreparePartError::RecordIndex(record_index))?,
            )
            .ok_or(PreparePartError::RecordIndex(record_index))?;
        let vector_bytes = &record.frame[vector_start..];
        if let Some(existing) = first_by_vector.get(vector_bytes) {
            samples[*existing].dup_count = samples[*existing]
                .dup_count
                .checked_add(1)
                .ok_or(PreparePartError::DuplicateCount)?;
            continue;
        }

        let sample_index = samples.len();
        first_by_vector.insert(vector_bytes.to_vec(), sample_index);
        let record_index = i32::try_from(record_index)
            .map_err(|_error| PreparePartError::RecordIndex(record_index))?;
        let semantic_error = if header.name != expected_cohort {
            Some(SampleFrameError::CohortMismatch)
        } else if header.dtype != F32_DTYPE {
            Some(SampleFrameError::UnsupportedDtype(header.dtype))
        } else if header.dimension == 0
            || usize::try_from(header.dimension)
                .ok()
                .and_then(|dimension| dimension.checked_mul(F32_BYTES))
                != Some(vector_bytes.len())
        {
            Some(SampleFrameError::DimensionMismatch)
        } else {
            None
        };
        let vector = semantic_error.map_or_else(
            || {
                Ok(vector_bytes
                    .chunks_exact(F32_BYTES)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            },
            Err,
        );
        let receive_time_us = i64::try_from(record.receive_time)
            .map_err(|_error| SampleFrameError::ReceiveTimeOutOfRange);
        samples.push(PreparedSample {
            record_index,
            dup_count: 1,
            receive_time_us,
            vector_hash: fnv1a64(vector_bytes),
            vector,
        });
    }

    Ok(PreparedPart {
        cohort: observed_cohort.to_owned(),
        window_start: segment.header.window_start,
        window_seconds: segment.header.window_seconds,
        received_frame_count: segment.header.received_frame_count,
        record_count: segment.header.record_count,
        samples,
    })
}

pub(crate) async fn measure_prepared_part<M: SampleMeasurer + Send + ?Sized>(
    part: &PreparedPart,
    part_ulid: &str,
    config: &AggregationConfig,
    computed_at_us: u64,
    measurer: &mut M,
    cancellation: &CancellationToken,
) -> Result<MeasurePartOutcome, PreparePartError> {
    let mut failed_count = 0_u64;
    let mut truth_rows = Vec::new();
    let mut sweep_rows = Vec::new();
    for sample in &part.samples {
        if cancellation.is_cancelled() {
            return Ok(MeasurePartOutcome::Cancelled);
        }
        let (Ok(receive_time_us), Ok(vector)) = (&sample.receive_time_us, &sample.vector) else {
            failed_count += 1;
            continue;
        };
        let measured = match measurer.measure_sample(vector, config, cancellation).await {
            Ok(measured) => measured,
            Err(SampleMeasureError::Database(_error)) => {
                failed_count += 1;
                continue;
            }
            Err(SampleMeasureError::Connection(error)) => {
                return Ok(MeasurePartOutcome::ConnectionUnavailable(error));
            }
            Err(SampleMeasureError::CancelledBeforeTransaction) => {
                return Ok(MeasurePartOutcome::Cancelled);
            }
            Err(SampleMeasureError::TableSmallerThanK { returned, k }) => {
                return Ok(MeasurePartOutcome::TableSmallerThanK(
                    PhaseAAbort::TableSmallerThanK {
                        error: format!("table returned {returned} rows, fewer than required k={k}"),
                    },
                ));
            }
        };
        if measured.gt_keys.len() != usize::try_from(config.k).unwrap_or(usize::MAX)
            || measured.gt_distances.len() != measured.gt_keys.len()
            || measured.sweeps.len() != config.ef_grid.len()
            || measured
                .sweeps
                .iter()
                .zip(&config.ef_grid)
                .any(|(sweep, ef)| sweep.ef != *ef)
        {
            failed_count += 1;
            continue;
        }
        truth_rows.push(TruthMeasurement {
            record_index: sample.record_index,
            vector_hash: sample.vector_hash,
            dup_count: sample.dup_count,
            receive_time_us: *receive_time_us,
            latency_ms: measured.ground_truth_latency_ms,
            gt_keys: measured.gt_keys,
            gt_distances: measured.gt_distances,
        });
        for sweep in measured.sweeps {
            sweep_rows.push(SweepMeasurementRow {
                record_index: sample.record_index,
                ef: sweep.ef,
                result_count: i32::try_from(sweep.returned_keys.len())
                    .map_err(|_error| PreparePartError::ResultCount(sweep.returned_keys.len()))?,
                returned_keys: sweep.returned_keys,
                recall: sweep.recall,
                latency_ms: sweep.latency_ms,
            });
        }
    }

    let measured_count = u64::try_from(truth_rows.len())
        .map_err(|_error| PreparePartError::MeasuredCount(truth_rows.len()))?;
    Ok(MeasurePartOutcome::Complete(Box::new(MeasuredPart {
        metadata: IntermediateMetadata {
            format_version: 1,
            cohort: part.cohort.clone(),
            part_ulid: part_ulid.to_owned(),
            window_start: part.window_start,
            window_seconds: part.window_seconds,
            received_frame_count: part.received_frame_count,
            record_count: part.record_count,
            index: config.index.clone(),
            table: config.table.clone(),
            column: config.column.clone(),
            key: config.key.clone(),
            k: config.k,
            ef_grid: config.ef_grid.clone(),
            failed_count,
            measured_count,
            computed_at_us,
        },
        truth_rows,
        sweep_rows,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    use vectorseam_core::cohort::CohortName;
    use vectorseam_core::frame::{FRAME_MAGIC, FRAME_VERSION};
    use vectorseam_core::segment::{SegmentHeader, SegmentRecordRef, write_segment};

    use crate::intermediate::{encode_intermediate_pair, read_intermediate_pair_bytes};

    const COHORT: &str = "acceptance/b12";
    const WINDOW_START: u64 = 1_783_512_000;
    const WINDOW_SECONDS: u32 = 600;

    struct CountingMeasurer {
        calls: usize,
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
            Ok(SampleMeasurement {
                gt_keys: (1..=i64::from(config.k)).collect(),
                gt_distances: (0..config.k).map(|value| f64::from(value) / 10.0).collect(),
                ground_truth_latency_ms: 400.5,
                sweeps: config
                    .ef_grid
                    .iter()
                    .map(|ef| SampleSweepResult {
                        ef: *ef,
                        returned_keys: (1..=i64::from(config.k)).collect(),
                        recall: 1.0,
                        latency_ms: 0.5,
                    })
                    .collect(),
            })
        }
    }

    #[tokio::test]
    async fn b12_measure_dedup_emits_one_truth_row_and_one_sweep_grid() {
        let duplicate = vec![0.25_f32, -0.75_f32];
        let frames = (0..10)
            .map(|record_index| {
                if [3, 5, 9].contains(&record_index) {
                    frame(COHORT, F32_DTYPE, &duplicate)
                } else {
                    frame(COHORT, 99, &[record_index as f32])
                }
            })
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
                last_receive: WINDOW_START * 1_000_000 + 9,
                received_frame_count: 10,
                record_count: 10,
                cohort: CohortName::try_from(COHORT).unwrap(),
            },
            &records,
        )
        .unwrap();
        let part = prepare_segment(&bytes, COHORT, WINDOW_START, WINDOW_SECONDS).unwrap();
        let duplicate_identity = part
            .sample_identities()
            .find(|(record_index, _dup_count)| *record_index == 3);
        assert_eq!(duplicate_identity, Some((3, 3)));

        let config = aggregation_config();
        let cancellation = CancellationToken::new();
        let mut measurer = CountingMeasurer { calls: 0 };
        let outcome = measure_prepared_part(
            &part,
            "01J00000000000000000000000",
            &config,
            WINDOW_START * 1_000_000,
            &mut measurer,
            &cancellation,
        )
        .await
        .unwrap();
        let MeasurePartOutcome::Complete(measured) = outcome else {
            panic!("B12 fixture must complete");
        };

        assert_eq!(measurer.calls, 1);
        assert_eq!(measured.metadata.failed_count, 7);
        assert_eq!(measured.metadata.measured_count, 1);
        assert_eq!(measured.truth_rows.len(), 1);
        assert_eq!(measured.truth_rows[0].record_index, 3);
        assert_eq!(measured.truth_rows[0].dup_count, 3);
        assert_eq!(measured.truth_rows[0].latency_ms, 400.5);
        assert_eq!(measured.sweep_rows.len(), 5);

        let (truth, sweep) = encode_intermediate_pair(
            &measured.metadata,
            &measured.truth_rows,
            &measured.sweep_rows,
        )
        .unwrap();
        let decoded = read_intermediate_pair_bytes(truth, sweep).unwrap();
        assert_eq!(decoded.samples.len(), 1);
        assert_eq!(decoded.samples[0].record_index, 3);
        assert_eq!(decoded.samples[0].dup_count, 3);
        assert_eq!(decoded.samples[0].ground_truth_latency_ms, 400.5);
        assert_eq!(decoded.samples[0].sweeps.len(), 5);
    }

    fn aggregation_config() -> AggregationConfig {
        AggregationConfig {
            cohort: COHORT.to_owned(),
            target_name: "recall".to_owned(),
            index: "fixture".to_owned(),
            table: "docs_fixture".to_owned(),
            column: "embedding".to_owned(),
            key: "doc_id".to_owned(),
            k: 10,
            value: 0.9,
            percentile: 0.95,
            window_duration_seconds: u64::from(WINDOW_SECONDS),
            storage_window_seconds: WINDOW_SECONDS,
            ef_grid: vec![10, 20, 40, 80, 160],
            train_fraction: 0.7,
            split_seed: 7,
            min_samples: 100,
        }
    }

    fn frame(cohort: &str, dtype: u32, vector: &[f32]) -> Vec<u8> {
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
        bytes.extend_from_slice(&dtype.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(name.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(vector.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(vector_bytes.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&vector_bytes);
        bytes
    }
}
