use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use vectorseam_collector::{Config, run_with_store};
use vectorseam_core::frame::{
    FIXED_FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_VERSION, parse_frame_header,
};
use vectorseam_core::segment::{Segment, read_segment};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flushes_valid_frames_and_omits_invalid_frames() {
    let harness = CollectorHarness::start(2, 8 * 1024 * 1024, 64 * 1024 * 1024).await;
    let mut sent_by_cohort: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
    let mut frames = Vec::new();

    for index in 0..240 {
        let frame = frame("prod", index);
        sent_by_cohort
            .entry("prod".to_owned())
            .or_default()
            .push(frame.clone());
        frames.push(frame);
    }
    for index in 0..180 {
        let frame = frame("prod/tenant-a/products", index + 1_000);
        sent_by_cohort
            .entry("prod/tenant-a/products".to_owned())
            .or_default()
            .push(frame.clone());
        frames.push(frame);
    }
    frames.push(bad_magic_frame("prod", 9_999));
    frames.push(frame("window=prod", 10_000));

    send_frames(harness.addr(), &frames).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    let root = harness.shutdown().await;
    let segments = read_segments(root.path());

    assert!(!segments.is_empty());
    let mut kept_by_cohort: HashMap<String, u64> = HashMap::new();
    let mut received_by_cohort: HashMap<String, u64> = HashMap::new();
    let mut dropped_by_cohort: HashMap<String, u64> = HashMap::new();
    let mut frames_by_cohort: HashMap<String, Vec<Vec<u8>>> = HashMap::new();

    for (key, segment) in &segments {
        assert!(key.starts_with("cohorts/"), "{key}");
        assert!(key.ends_with(".vseam"), "{key}");
        assert_eq!(segment.header.window_seconds, 2);
        assert_eq!(segment.header.window_start % 2, 0);
        assert_eq!(segment.header.record_count, segment.records.len() as u64);
        let cohort = segment.header.cohort.as_str();

        *kept_by_cohort.entry(cohort.to_owned()).or_default() += segment.header.record_count;
        *received_by_cohort.entry(cohort.to_owned()).or_default() +=
            segment.header.received_frame_count;
        *dropped_by_cohort.entry(cohort.to_owned()).or_default() += segment
            .header
            .received_frame_count
            .saturating_sub(segment.header.record_count);

        for record in &segment.records {
            let parsed = parse_frame_header(&record.frame).unwrap();
            assert_eq!(parsed.name, cohort);
            frames_by_cohort
                .entry(cohort.to_owned())
                .or_default()
                .push(record.frame.to_vec());
        }
    }

    for (cohort, sent) in &sent_by_cohort {
        assert_eq!(
            kept_by_cohort.get(cohort).copied().unwrap_or(0),
            sent.len() as u64
        );
        assert_eq!(
            received_by_cohort.get(cohort).copied().unwrap_or(0),
            sent.len() as u64
        );
        assert_eq!(dropped_by_cohort.get(cohort).copied().unwrap_or(0), 0);
        assert!(
            frames_by_cohort
                .get(cohort)
                .unwrap()
                .iter()
                .any(|stored| stored == &sent[0]),
            "missing byte-exact sampled frame for {cohort}"
        );
    }

    assert!(!frames_by_cohort.contains_key("window=prod"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spills_early_when_per_cohort_cap_is_tiny() {
    let sample = frame("prod", 0);
    let record_bytes = sample.len() + 8;
    let harness = CollectorHarness::start_with_max_frame_size(
        60,
        record_bytes * 4,
        64 * 1024 * 1024,
        sample.len(),
    )
    .await;
    let frames: Vec<Vec<u8>> = (0..25).map(|index| frame("prod", index)).collect();

    send_frames_with_pause(harness.addr(), &frames, 5, Duration::from_millis(200)).await;
    let root = harness.shutdown().await;
    let segments = read_segments(root.path());
    let prod_segments: Vec<&Segment> = segments
        .iter()
        .map(|(_key, segment)| segment)
        .filter(|segment| segment.header.cohort.as_str() == "prod")
        .collect();

    assert!(prod_segments.len() > 1, "expected early spill parts");
    let kept = prod_segments
        .iter()
        .map(|segment| segment.header.record_count)
        .sum::<u64>();
    let received = prod_segments
        .iter()
        .map(|segment| segment.header.received_frame_count)
        .sum::<u64>();
    let dropped = received.saturating_sub(kept);
    assert_eq!(received, frames.len() as u64);
    assert_eq!(kept + dropped, received);
    assert!(kept > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graceful_shutdown_flushes_partial_window() {
    let harness = CollectorHarness::start(60, 8 * 1024 * 1024, 64 * 1024 * 1024).await;
    let frames: Vec<Vec<u8>> = (0..12).map(|index| frame("prod", index)).collect();

    send_frames(harness.addr(), &frames).await;
    let root = harness.shutdown().await;
    let segments = read_segments(root.path());

    assert_eq!(
        segments
            .iter()
            .map(|(_key, segment)| segment.header.record_count)
            .sum::<u64>(),
        frames.len() as u64
    );
}

struct CollectorHarness {
    _tmp: TempDir,
    addr: SocketAddr,
    storage_root: TempDir,
    shutdown_tx: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl CollectorHarness {
    async fn start(
        window_seconds: u32,
        per_cohort_memory_bytes: usize,
        global_memory_bytes: usize,
    ) -> Self {
        Self::start_with_max_frame_size(
            window_seconds,
            per_cohort_memory_bytes,
            global_memory_bytes,
            32 * 1024,
        )
        .await
    }

    async fn start_with_max_frame_size(
        window_seconds: u32,
        per_cohort_memory_bytes: usize,
        global_memory_bytes: usize,
        max_frame_size: usize,
    ) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let storage_root = tempfile::tempdir().unwrap();
        let addr = free_tcp_addr();
        let config = Config {
            listen: addr,
            unix_socket: None,
            storage_root: storage_root.path().to_path_buf(),
            window_seconds,
            per_cohort_memory_bytes,
            global_memory_bytes,
            max_frame_size,
            channel_capacity: 4_096,
            max_connections: 1_024,
            put_timeout_seconds: 60,
        };
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(storage_root.path()).unwrap());
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(run_with_store(config, store, async {
            let _ = shutdown_rx.await;
        }));

        wait_for_tcp(addr).await;
        Self {
            _tmp: tmp,
            addr,
            storage_root,
            shutdown_tx,
            task,
        }
    }

    fn addr(&self) -> SocketAddr {
        self.addr
    }

    async fn shutdown(self) -> TempDir {
        let _ = self.shutdown_tx.send(());
        self.task.await.unwrap().unwrap();
        self.storage_root
    }
}

fn free_tcp_addr() -> SocketAddr {
    let listener = StdTcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.local_addr().unwrap()
}

async fn wait_for_tcp(addr: SocketAddr) {
    for _ in 0..200 {
        if TcpStream::connect(addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("collector did not listen on {addr}");
}

async fn send_frames(addr: SocketAddr, frames: &[Vec<u8>]) {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    for frame in frames {
        stream.write_all(frame).await.unwrap();
    }
    stream.shutdown().await.unwrap();
}

async fn send_frames_with_pause(
    addr: SocketAddr,
    frames: &[Vec<u8>],
    pause_every: usize,
    pause: Duration,
) {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    for (index, frame) in frames.iter().enumerate() {
        stream.write_all(frame).await.unwrap();
        if (index + 1) % pause_every == 0 {
            tokio::time::sleep(pause).await;
        }
    }
    stream.shutdown().await.unwrap();
}

fn frame(name: &str, seed: u32) -> Vec<u8> {
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
    out
}

fn bad_magic_frame(name: &str, seed: u32) -> Vec<u8> {
    let mut bytes = frame(name, seed);
    bytes[4..8].copy_from_slice(b"NOPE");
    bytes
}

fn read_segments(root: &Path) -> Vec<(String, Segment)> {
    let mut paths = Vec::new();
    collect_vseam_paths(root, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(&path).unwrap();
            let segment = read_segment(&bytes).unwrap();
            (relative_key(root, &path), segment)
        })
        .collect()
}

fn collect_vseam_paths(dir: &Path, paths: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_vseam_paths(&path, paths);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "vseam")
        {
            paths.push(path);
        }
    }
}

fn relative_key(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap()
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}
