use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;
use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use tokio::sync::mpsc;
use tracing::{error, info};
use ulid::Ulid;
use vectorseam_core::cohort::CohortName;
use vectorseam_core::segment::{SegmentHeader, SegmentRecordRef, write_segment};
use vectorseam_core::window::{aligned_window_start, object_key};

use crate::config::WriterConfig;
use crate::counters::CollectorCounters;
use crate::memory::{MemoryGuard, MemoryTracker};
use crate::reader::FrameEvent;
use crate::time::{duration_until_unix_second, unix_seconds_now};

pub(crate) struct Writer {
    config: WriterConfig,
    store: Arc<dyn ObjectStore>,
    counters: Arc<CollectorCounters>,
    memory: Arc<MemoryTracker>,
    states: HashMap<CohortName, CohortState>,
    current_window_start: u64,
}

#[derive(Default)]
struct CohortState {
    records: Vec<BufferedRecord>,
    bytes: usize,
    received_frame_count: u64,
}

struct BufferedRecord {
    receive_time: u64,
    frame: Bytes,
    _memory: MemoryGuard,
}

struct FlushBatch {
    window_start: u64,
    header: SegmentHeader,
    records: Vec<BufferedRecord>,
    record_bytes: usize,
}

impl Writer {
    pub(crate) fn new(
        config: WriterConfig,
        store: Arc<dyn ObjectStore>,
        counters: Arc<CollectorCounters>,
        memory: Arc<MemoryTracker>,
    ) -> Result<Self> {
        let now_seconds = unix_seconds_now()?;
        let current_window_start = aligned_window_start(now_seconds, config.window_seconds)?;
        Ok(Self {
            config,
            store,
            counters,
            memory,
            states: HashMap::new(),
            current_window_start,
        })
    }

    pub(crate) async fn run(mut self, mut rx: mpsc::Receiver<FrameEvent>) -> Result<()> {
        loop {
            let sleep = tokio::time::sleep(duration_until_unix_second(
                self.current_window_start + u64::from(self.config.window_seconds),
            ));
            tokio::pin!(sleep);

            tokio::select! {
                maybe_event = rx.recv() => {
                    match maybe_event {
                        Some(event) => self.handle_frame(event).await,
                        None => break,
                    }
                }
                _ = &mut sleep => {
                    self.close_window().await;
                }
            }
        }

        self.fold_all_pending_drops();
        self.flush_all(self.current_window_start).await;
        Ok(())
    }

    async fn handle_frame(&mut self, event: FrameEvent) {
        self.fold_pending_drops(&event.cohort);
        let record_bytes = event.memory.bytes();

        if self.cohort_would_exceed_cap(&event.cohort, record_bytes) {
            self.flush_cohort(&event.cohort, self.current_window_start)
                .await;
        }

        if self.global_live_budget_is_full() {
            self.flush_largest_cohort().await;
        }

        self.add_record(event, record_bytes);
    }

    fn add_record(&mut self, event: FrameEvent, record_bytes: usize) {
        let state = self.states.entry(event.cohort).or_default();
        state.received_frame_count = state.received_frame_count.saturating_add(1);
        state.bytes = state.bytes.saturating_add(record_bytes);
        state.records.push(BufferedRecord {
            receive_time: event.receive_time,
            frame: event.frame,
            _memory: event.memory,
        });
        self.counters.kept_frames.fetch_add(1, Ordering::Relaxed);
    }

    fn cohort_would_exceed_cap(&self, cohort: &CohortName, record_bytes: usize) -> bool {
        self.states
            .get(cohort)
            .and_then(|state| state.bytes.checked_add(record_bytes))
            .is_some_and(|bytes| bytes > self.config.per_cohort_memory_bytes)
    }

    fn global_live_budget_is_full(&self) -> bool {
        self.memory.used_bytes() >= self.config.live_memory_bytes
    }

    async fn flush_largest_cohort(&mut self) {
        let cohort = self
            .states
            .iter()
            .filter(|(_cohort, state)| !state.records.is_empty())
            .max_by_key(|(_cohort, state)| state.bytes)
            .map(|(cohort, _state)| cohort.clone());

        if let Some(cohort) = cohort {
            self.flush_cohort(&cohort, self.current_window_start).await;
        }
    }

    async fn close_window(&mut self) {
        let closing_window = self.current_window_start;
        self.fold_all_pending_drops();
        self.flush_all(closing_window).await;
        match unix_seconds_now().and_then(|now| {
            aligned_window_start(now, self.config.window_seconds).map_err(Into::into)
        }) {
            Ok(window_start) => {
                self.current_window_start = window_start;
            }
            Err(error) => {
                error!(%error, "failed to compute next window start");
                self.current_window_start =
                    closing_window.saturating_add(u64::from(self.config.window_seconds));
            }
        }
    }

    async fn flush_all(&mut self, window_start: u64) {
        let cohorts: Vec<CohortName> = self.states.keys().cloned().collect();
        for cohort in cohorts {
            self.flush_cohort(&cohort, window_start).await;
        }
    }

    async fn flush_cohort(&mut self, cohort: &CohortName, window_start: u64) -> bool {
        self.fold_pending_drops(cohort);
        let Some(state) = self.states.get(cohort) else {
            return false;
        };
        if state.records.is_empty() {
            return false;
        }

        let Some(batch) = self.take_flush_batch(cohort, window_start) else {
            return false;
        };
        self.put_flush(batch).await;
        true
    }

    fn take_flush_batch(&mut self, cohort: &CohortName, window_start: u64) -> Option<FlushBatch> {
        let mut state = self.states.remove(cohort)?;
        let records = std::mem::take(&mut state.records);
        if records.is_empty() {
            return None;
        }
        let record_bytes = state.bytes;

        let first_receive = records.first()?.receive_time;
        let last_receive = records.last()?.receive_time;
        let record_count = u64::try_from(records.len()).unwrap_or(u64::MAX);
        let header = SegmentHeader {
            window_start,
            window_seconds: self.config.window_seconds,
            first_receive,
            last_receive,
            received_frame_count: state.received_frame_count,
            record_count,
            cohort: cohort.clone(),
        };

        Some(FlushBatch {
            window_start,
            header,
            records,
            record_bytes,
        })
    }

    async fn put_flush(&mut self, batch: FlushBatch) {
        let FlushBatch {
            window_start,
            header,
            records,
            record_bytes,
        } = batch;

        let segment_records: Vec<SegmentRecordRef<'_>> = records
            .iter()
            .map(|record| SegmentRecordRef {
                receive_time: record.receive_time,
                frame: record.frame.as_ref(),
            })
            .collect();
        let segment = match write_segment(&header, &segment_records) {
            Ok(segment) => segment,
            Err(error) => {
                error!(cohort = %header.cohort, %error, "failed to serialize segment");
                self.counters.flush_failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        drop(segment_records);
        drop(records);

        let key = match object_key(&header.cohort, window_start, Ulid::new()) {
            Ok(key) => key,
            Err(error) => {
                error!(cohort = %header.cohort, %error, "failed to build object key");
                self.counters.flush_failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        let record_count = header.record_count;
        let received_frame_count = header.received_frame_count;
        let dropped_count = received_frame_count.saturating_sub(record_count);
        let window = header.window_start;
        let segment_bytes = segment.len();

        let path = Path::from(key.clone());
        match tokio::time::timeout(
            self.config.put_timeout,
            self.store.put(&path, PutPayload::from(segment)),
        )
        .await
        {
            Ok(Ok(_result)) => {
                self.counters.flushed_parts.fetch_add(1, Ordering::Relaxed);
                info!(
                    cohort = %header.cohort,
                    window,
                    record_count,
                    received_frame_count,
                    dropped_count,
                    bytes = segment_bytes,
                    record_bytes,
                    key = %key,
                    "flushed segment"
                );
            }
            Ok(Err(error)) => {
                self.counters.flush_failures.fetch_add(1, Ordering::Relaxed);
                error!(cohort = %header.cohort, key = %key, %error, "failed to put segment");
            }
            Err(_elapsed) => {
                self.counters.flush_failures.fetch_add(1, Ordering::Relaxed);
                error!(
                    cohort = %header.cohort,
                    key = %key,
                    timeout_seconds = self.config.put_timeout.as_secs_f64(),
                    "timed out putting segment"
                );
            }
        }
    }

    fn fold_pending_drops(&mut self, cohort: &CohortName) {
        let drops = self.counters.take_pending_cohort_drops(cohort);
        if drops == 0 {
            return;
        }
        let state = self.states.entry(cohort.clone()).or_default();
        state.received_frame_count = state.received_frame_count.saturating_add(drops);
    }

    fn fold_all_pending_drops(&mut self) {
        for (cohort, drops) in self.counters.take_all_pending_cohort_drops() {
            if let Some(state) = self.states.get_mut(&cohort) {
                state.received_frame_count = state.received_frame_count.saturating_add(drops);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt;
    use std::time::Duration;

    use async_trait::async_trait;
    use futures_util::stream::BoxStream;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutResult, Result as StoreResult,
    };
    use vectorseam_core::frame::{FIXED_FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_VERSION};
    use vectorseam_core::segment::{Segment, read_segment};

    const TEST_WINDOW_START: u64 = 0;
    const TEST_WINDOW_SECONDS: u32 = 60;
    const TEST_LIVE_LIMIT: usize = 64 * 1024;

    struct WriterFixture {
        writer: Writer,
        store: Arc<InMemory>,
        counters: Arc<CollectorCounters>,
        memory: Arc<MemoryTracker>,
        live_memory_bytes: usize,
    }

    #[tokio::test]
    async fn flushes_buffered_records_with_counts_and_exact_frames() {
        let mut fixture = writer_fixture(8 * 1024, TEST_LIVE_LIMIT);
        let first = frame("prod", 1);
        let second = frame("prod", 2);

        fixture
            .writer
            .handle_frame(fixture.event("prod", 100, first.clone()))
            .await;
        fixture
            .writer
            .handle_frame(fixture.event("prod", 200, second.clone()))
            .await;
        fixture
            .counters
            .record_channel_drop(&CohortName::try_from("prod").unwrap());

        assert!(
            fixture
                .writer
                .flush_cohort(&CohortName::try_from("prod").unwrap(), TEST_WINDOW_START,)
                .await
        );

        let segments = fixture.segments_for("prod").await;
        assert_eq!(segments.len(), 1);
        let segment = &segments[0];
        assert_eq!(segment.header.first_receive, 100);
        assert_eq!(segment.header.last_receive, 200);
        assert_eq!(segment.header.record_count, 2);
        assert_eq!(segment.header.received_frame_count, 3);
        assert_eq!(segment.records[0].frame, first);
        assert_eq!(segment.records[1].frame, second);
        assert_eq!(fixture.memory.used_bytes(), 0);
        assert!(
            !fixture
                .writer
                .states
                .contains_key(&CohortName::try_from("prod").unwrap())
        );
    }

    #[tokio::test]
    async fn spills_cohort_before_it_exceeds_per_cohort_cap() {
        let record_bytes = record_bytes("prod");
        let mut fixture = writer_fixture(record_bytes * 2, record_bytes * 10);

        for index in 0..3 {
            let event = fixture.event("prod", 100 + index, frame("prod", index as u32));
            fixture.writer.handle_frame(event).await;
        }

        let segments = fixture.segments_for("prod").await;
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].header.record_count, 2);
        assert_eq!(segments[0].records.len(), 2);
        assert_eq!(
            fixture.writer.states[&CohortName::try_from("prod").unwrap()]
                .records
                .len(),
            1
        );
        assert_eq!(fixture.memory.used_bytes(), record_bytes);
    }

    #[tokio::test]
    async fn global_pressure_flushes_largest_cohort() {
        let prod_record_bytes = record_bytes("prod");
        let other_record_bytes = record_bytes("other");
        let mut fixture = writer_fixture(
            prod_record_bytes * 10,
            prod_record_bytes * 2 + other_record_bytes,
        );

        fixture
            .writer
            .handle_frame(fixture.event("prod", 100, frame("prod", 1)))
            .await;
        fixture
            .writer
            .handle_frame(fixture.event("prod", 101, frame("prod", 2)))
            .await;
        fixture
            .writer
            .handle_frame(fixture.event("other", 102, frame("other", 3)))
            .await;

        let prod_segments = fixture.segments_for("prod").await;
        assert_eq!(prod_segments.len(), 1);
        assert_eq!(prod_segments[0].header.record_count, 2);
        assert_eq!(
            fixture.writer.states[&CohortName::try_from("other").unwrap()]
                .records
                .len(),
            1
        );
        assert_eq!(fixture.memory.used_bytes(), record_bytes("other"));
    }

    #[tokio::test]
    async fn run_flushes_open_buffers_when_channel_closes() {
        let fixture = writer_fixture(8 * 1024, TEST_LIVE_LIMIT);
        let store = fixture.store.clone();
        let memory = fixture.memory.clone();
        let live_memory_bytes = fixture.live_memory_bytes;
        let (tx, rx) = mpsc::channel(4);

        tx.send(event(
            &memory,
            live_memory_bytes,
            "prod",
            100,
            frame("prod", 1),
        ))
        .await
        .unwrap();
        drop(tx);

        fixture.writer.run(rx).await.unwrap();

        let segments = segments_for(&store, "prod").await;
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].header.record_count, 1);
        assert_eq!(memory.used_bytes(), 0);
    }

    #[tokio::test]
    async fn close_window_removes_flushed_cohort_states() {
        let mut fixture = writer_fixture(8 * 1024, TEST_LIVE_LIMIT);

        fixture
            .writer
            .handle_frame(fixture.event("prod", 100, frame("prod", 1)))
            .await;
        fixture
            .writer
            .handle_frame(fixture.event("other", 101, frame("other", 2)))
            .await;

        fixture.writer.flush_all(TEST_WINDOW_START).await;

        assert!(fixture.writer.states.is_empty());
        assert_eq!(fixture.memory.used_bytes(), 0);
        assert_eq!(fixture.segments_for("prod").await.len(), 1);
        assert_eq!(fixture.segments_for("other").await.len(), 1);
    }

    #[test]
    fn fold_all_pending_drops_does_not_create_drop_only_states() {
        let mut fixture = writer_fixture(8 * 1024, TEST_LIVE_LIMIT);
        fixture
            .counters
            .record_channel_drop(&CohortName::try_from("drop-only").unwrap());

        fixture.writer.fold_all_pending_drops();

        assert!(fixture.writer.states.is_empty());
    }

    #[tokio::test]
    async fn serialization_failure_counts_failure_and_releases_memory() {
        let mut fixture = writer_fixture(8 * 1024, TEST_LIVE_LIMIT);
        let cohort = CohortName::try_from("prod").unwrap();

        fixture
            .writer
            .handle_frame(fixture.event("prod", 100, Bytes::from_static(b"bad")))
            .await;

        assert!(
            fixture
                .writer
                .flush_cohort(&cohort, TEST_WINDOW_START)
                .await
        );

        assert_eq!(fixture.counters.flush_failures.load(Ordering::Relaxed), 1);
        assert!(fixture.segments_for("prod").await.is_empty());
        assert_eq!(fixture.memory.used_bytes(), 0);
        assert!(!fixture.writer.states.contains_key(&cohort));
    }

    #[tokio::test]
    async fn put_timeout_counts_failure_and_releases_memory() {
        let counters = Arc::new(CollectorCounters::default());
        let memory = Arc::new(MemoryTracker::default());
        let store: Arc<dyn ObjectStore> = Arc::new(HangingPutStore::default());
        let mut writer = Writer {
            config: WriterConfig {
                window_seconds: TEST_WINDOW_SECONDS,
                per_cohort_memory_bytes: 8 * 1024,
                live_memory_bytes: TEST_LIVE_LIMIT,
                put_timeout: Duration::from_millis(10),
            },
            store,
            counters: counters.clone(),
            memory: memory.clone(),
            states: HashMap::new(),
            current_window_start: TEST_WINDOW_START,
        };
        let cohort = CohortName::try_from("prod").unwrap();

        writer
            .handle_frame(event(
                &memory,
                TEST_LIVE_LIMIT,
                "prod",
                100,
                frame("prod", 1),
            ))
            .await;

        let flushed = tokio::time::timeout(
            Duration::from_secs(1),
            writer.flush_cohort(&cohort, TEST_WINDOW_START),
        )
        .await
        .unwrap();

        assert!(flushed);
        assert_eq!(counters.flush_failures.load(Ordering::Relaxed), 1);
        assert_eq!(counters.flushed_parts.load(Ordering::Relaxed), 0);
        assert_eq!(memory.used_bytes(), 0);
        assert!(!writer.states.contains_key(&cohort));
    }

    impl WriterFixture {
        fn event(&self, cohort: &str, receive_time: u64, frame: Bytes) -> FrameEvent {
            event(
                &self.memory,
                self.live_memory_bytes,
                cohort,
                receive_time,
                frame,
            )
        }

        async fn segments_for(&self, cohort: &str) -> Vec<Segment> {
            segments_for(&self.store, cohort).await
        }
    }

    fn writer_fixture(per_cohort_memory_bytes: usize, live_memory_bytes: usize) -> WriterFixture {
        let store = Arc::new(InMemory::new());
        let object_store: Arc<dyn ObjectStore> = store.clone();
        let counters = Arc::new(CollectorCounters::default());
        let memory = Arc::new(MemoryTracker::default());
        let writer = Writer {
            config: WriterConfig {
                window_seconds: TEST_WINDOW_SECONDS,
                per_cohort_memory_bytes,
                live_memory_bytes,
                put_timeout: Duration::from_secs(60),
            },
            store: object_store,
            counters: counters.clone(),
            memory: memory.clone(),
            states: HashMap::new(),
            current_window_start: TEST_WINDOW_START,
        };
        WriterFixture {
            writer,
            store,
            counters,
            memory,
            live_memory_bytes,
        }
    }

    fn event(
        memory: &Arc<MemoryTracker>,
        live_memory_bytes: usize,
        cohort: &str,
        receive_time: u64,
        frame: Bytes,
    ) -> FrameEvent {
        let memory_bytes = frame.len() + 8;
        let memory = memory.try_reserve(memory_bytes, live_memory_bytes).unwrap();
        FrameEvent {
            cohort: CohortName::try_from(cohort).unwrap(),
            receive_time,
            frame,
            memory,
        }
    }

    async fn segments_for(store: &InMemory, cohort: &str) -> Vec<Segment> {
        let prefix = Path::from(format!("cohorts/{cohort}/window=19700101T0000Z"));
        let mut objects = store
            .list_with_delimiter(Some(&prefix))
            .await
            .unwrap()
            .objects;
        objects.sort_by(|left, right| left.location.cmp(&right.location));

        let mut segments = Vec::new();
        for object in objects {
            let bytes = store
                .get(&object.location)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            segments.push(read_segment(&bytes).unwrap());
        }
        segments
    }

    fn record_bytes(cohort: &str) -> usize {
        frame(cohort, 0).len() + 8
    }

    fn frame(name: &str, seed: u32) -> Bytes {
        let vector: Vec<u8> = [
            (seed as f32).to_le_bytes(),
            (seed as f32 + 1.0).to_le_bytes(),
            (seed as f32 + 2.0).to_le_bytes(),
            (seed as f32 + 3.0).to_le_bytes(),
        ]
        .concat();
        let name_bytes = name.as_bytes();
        let frame_len = (FIXED_FRAME_HEADER_LEN - 4 + name_bytes.len() + vector.len()) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&frame_len.to_le_bytes());
        out.extend_from_slice(&FRAME_MAGIC);
        out.extend_from_slice(&FRAME_VERSION.to_le_bytes());
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&4_u32.to_le_bytes());
        out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&vector);
        Bytes::from(out)
    }

    #[derive(Debug, Default)]
    struct HangingPutStore {
        inner: InMemory,
    }

    impl fmt::Display for HangingPutStore {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("HangingPutStore")
        }
    }

    #[async_trait]
    impl ObjectStore for HangingPutStore {
        async fn put_opts(
            &self,
            _location: &Path,
            _payload: PutPayload,
            _opts: PutOptions,
        ) -> StoreResult<PutResult> {
            std::future::pending().await
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
}
