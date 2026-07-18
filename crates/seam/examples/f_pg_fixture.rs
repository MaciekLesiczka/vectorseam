//! Generates the deterministic pgvector acceptance fixture.

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, FixedSizeListArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::json;
use tokio_postgres::NoTls;
use ulid::Ulid;
use vectorseam_core::cohort::CohortName;
use vectorseam_core::frame::{
    FIXED_FRAME_HEADER_LEN, FRAME_MAGIC, FRAME_VERSION, parse_frame_header,
};
use vectorseam_core::segment::{SegmentHeader, SegmentRecordRef, read_segment, write_segment};
use vectorseam_core::window::{format_window_timestamp, object_key};

const DOC_COUNT: usize = 10_000;
const QUERY_COUNT: usize = 500;
const DIMENSION: usize = 64;
const SEED: u64 = 0;
const K: usize = 10;
const EF_GRID: [i32; 5] = [10, 20, 40, 80, 160];
const WINDOW_START: u64 = 1_784_116_800;
const WINDOW_SECONDS: u32 = 600;
const PART_ULID: &str = "01K0A000000000000000000000";
const COHORT: &str = "anchor/f-pg";
const DEFAULT_DATABASE_URL: &str = "postgresql://postgres:password@localhost:55432/postgres";

#[tokio::main]
async fn main() -> Result<()> {
    let root = std::env::var_os("SEAM_F_PG_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/seam-fixtures/f-pg"));
    let database_url =
        std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_owned());
    std::fs::create_dir_all(&root)
        .with_context(|| format!("create fixture root {}", root.display()))?;
    reset_generated_outputs(&root)?;

    let (documents, queries) = generate_vectors(SEED);
    verify_normalized(&documents)?;
    verify_normalized(&queries)?;
    verify_pairwise_distinct(&queries)?;
    verify_no_boundary_ties_in_memory(&documents, &queries)?;
    write_vector_parquet(&root.join("docs.parquet"), "doc_id", &documents)?;
    write_vector_parquet(&root.join("queries.parquet"), "query_id", &queries)?;
    let segment_path = write_query_segment(&root, &queries)?;
    verify_segment_query_order(&segment_path, &queries)?;

    let (mut client, connection) = tokio_postgres::connect(&database_url, NoTls)
        .await
        .context("connect to F-pg PostgreSQL")?;
    let connection_task = tokio::spawn(connection);
    load_main_table(&mut client, &documents).await?;
    load_tie_break_table(&mut client).await?;
    verify_tie_break_table(&client).await?;
    let minimum_postgres_boundary_gap =
        verify_no_boundary_ties_in_postgres(&client, &queries).await?;
    drop(client);
    connection_task
        .await
        .context("join F-pg PostgreSQL connection task")?
        .context("drive F-pg PostgreSQL connection")?;

    let manifest = json!({
        "format_version": 1,
        "seed": SEED,
        "dimension": DIMENSION,
        "document_count": DOC_COUNT,
        "query_count": QUERY_COUNT,
        "k": K,
        "ef_grid": EF_GRID,
        "table": "docs_seam_fixture",
        "tie_break_table": "docs_seam_tie_fixture",
        "cohort": COHORT,
        "segment": segment_path.strip_prefix(&root).unwrap_or(&segment_path),
        "pairwise_distinct_queries_verified": true,
        "normalized_vectors_verified": true,
        "no_ground_truth_boundary_ties_verified": true,
        "minimum_postgres_boundary_gap": minimum_postgres_boundary_gap
    });
    std::fs::write(
        root.join("manifest.json"),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    )?;
    println!("generated F-pg fixture at {}", root.display());
    Ok(())
}

fn reset_generated_outputs(root: &Path) -> Result<()> {
    for directory in ["storage", "anchor"] {
        let path = root.join(directory);
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("remove {}", path.display()));
            }
        }
    }
    Ok(())
}

fn generate_vectors(seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let documents = (0..DOC_COUNT)
        .map(|_| random_normalized_vector(&mut rng))
        .collect();
    let queries = (0..QUERY_COUNT)
        .map(|_| random_normalized_vector(&mut rng))
        .collect();
    (documents, queries)
}

fn random_normalized_vector(rng: &mut ChaCha8Rng) -> Vec<f32> {
    let mut vector = (0..DIMENSION)
        .map(|_| rng.random_range(-1.0_f32..1.0_f32))
        .collect::<Vec<_>>();
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn verify_pairwise_distinct(queries: &[Vec<f32>]) -> Result<()> {
    let mut seen = HashSet::with_capacity(queries.len());
    for query in queries {
        ensure!(
            seen.insert(vector_bytes(query)),
            "F-pg query vectors must be pairwise distinct"
        );
    }
    Ok(())
}

fn verify_normalized(vectors: &[Vec<f32>]) -> Result<()> {
    for (index, vector) in vectors.iter().enumerate() {
        let squared_norm = vector.iter().map(|value| value * value).sum::<f32>();
        ensure!(
            (squared_norm - 1.0).abs() <= 1e-5,
            "F-pg vector {index} is not normalized"
        );
    }
    Ok(())
}

fn verify_no_boundary_ties_in_memory(documents: &[Vec<f32>], queries: &[Vec<f32>]) -> Result<()> {
    queries
        .par_iter()
        .enumerate()
        .try_for_each(|(query_index, query)| {
            let mut scores = documents
                .iter()
                .map(|document| dot_product(query, document))
                .collect::<Vec<_>>();
            scores.sort_unstable_by(|left, right| right.total_cmp(left));
            ensure!(
                scores[K - 1] != scores[K],
                "F-pg query {query_index} has an in-memory top-k boundary tie"
            );
            Ok::<(), anyhow::Error>(())
        })
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn write_vector_parquet(path: &Path, id_column: &str, vectors: &[Vec<f32>]) -> Result<()> {
    let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vectors
            .iter()
            .map(|vector| Some(vector.iter().copied().map(Some))),
        i32::try_from(DIMENSION)?,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new(id_column, DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(DIMENSION)?,
            ),
            false,
        ),
    ]));
    let ids = Int64Array::from_iter_values(1..=i64::try_from(vectors.len())?);
    let columns: Vec<ArrayRef> = vec![Arc::new(ids), Arc::new(vector_array)];
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_query_segment(root: &Path, queries: &[Vec<f32>]) -> Result<PathBuf> {
    let cohort = CohortName::try_from(COHORT)?;
    let frames = queries
        .iter()
        .map(|query| encode_f32_frame(&cohort, query))
        .collect::<Result<Vec<_>>>()?;
    let base_receive_time = WINDOW_START * 1_000_000;
    let records = frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            Ok(SegmentRecordRef {
                receive_time: base_receive_time + u64::try_from(index)?,
                frame,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let header = SegmentHeader {
        window_start: WINDOW_START,
        window_seconds: WINDOW_SECONDS,
        first_receive: base_receive_time,
        last_receive: base_receive_time + u64::try_from(queries.len() - 1)?,
        received_frame_count: u64::try_from(queries.len())?,
        record_count: u64::try_from(queries.len())?,
        cohort: cohort.clone(),
    };
    let bytes = write_segment(&header, &records)?;
    let part = PART_ULID.parse::<Ulid>()?;
    let relative = object_key(&cohort, WINDOW_START, part)?;
    let path = root.join("storage").join(relative);
    let parent = path.parent().context("segment path must have a parent")?;
    std::fs::create_dir_all(parent)?;
    std::fs::write(&path, bytes)?;
    ensure!(
        format_window_timestamp(WINDOW_START)? == "20260715T1200Z",
        "fixture window timestamp changed unexpectedly"
    );
    Ok(path)
}

fn verify_segment_query_order(path: &Path, queries: &[Vec<f32>]) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let segment = read_segment(&bytes)?;
    ensure!(segment.records.len() == queries.len());
    for (record, query) in segment.records.iter().zip(queries) {
        let header = parse_frame_header(&record.frame)?;
        ensure!(header.name == COHORT);
        let vector_start = FIXED_FRAME_HEADER_LEN + header.name.len();
        ensure!(record.frame[vector_start..] == vector_bytes(query));
    }
    Ok(())
}

async fn load_main_table(
    client: &mut tokio_postgres::Client,
    documents: &[Vec<f32>],
) -> Result<()> {
    client
        .batch_execute(
            "CREATE EXTENSION IF NOT EXISTS vector;
             DROP TABLE IF EXISTS docs_seam_fixture;
             CREATE TABLE docs_seam_fixture (
                 doc_id bigint PRIMARY KEY,
                 embedding vector(64) NOT NULL
             );",
        )
        .await?;
    let transaction = client.transaction().await?;
    let insert = transaction
        .prepare(
            "INSERT INTO docs_seam_fixture (doc_id, embedding)
             VALUES ($1, $2::text::vector);",
        )
        .await?;
    for (index, document) in documents.iter().enumerate() {
        let key = i64::try_from(index + 1)?;
        let vector = format_vector(document);
        transaction.execute(&insert, &[&key, &vector]).await?;
    }
    transaction.commit().await?;
    client
        .batch_execute(
            "CREATE INDEX docs_seam_fixture_embedding_hnsw_idx
             ON docs_seam_fixture
             USING hnsw (embedding vector_cosine_ops)
             WITH (m = 16, ef_construction = 64);
             ANALYZE docs_seam_fixture;

             DROP TABLE IF EXISTS docs_seam_timeout_fixture;
             CREATE TABLE docs_seam_timeout_fixture (
                 doc_id bigint PRIMARY KEY,
                 embedding vector(64) NOT NULL
             );
             INSERT INTO docs_seam_timeout_fixture (doc_id, embedding)
             SELECT doc_id, embedding FROM docs_seam_fixture;
             ANALYZE docs_seam_timeout_fixture;

             DROP TABLE IF EXISTS docs_seam_client_timeout_fixture;
             CREATE TABLE docs_seam_client_timeout_fixture (
                 doc_id bigint PRIMARY KEY,
                 embedding vector(64) NOT NULL
             );
             INSERT INTO docs_seam_client_timeout_fixture (doc_id, embedding)
             SELECT doc_id, embedding FROM docs_seam_fixture;
             ANALYZE docs_seam_client_timeout_fixture;",
        )
        .await?;
    Ok(())
}

async fn load_tie_break_table(client: &mut tokio_postgres::Client) -> Result<()> {
    client
        .batch_execute(
            "DROP TABLE IF EXISTS docs_seam_tie_fixture;
             CREATE TABLE docs_seam_tie_fixture (
                 doc_id bigint PRIMARY KEY,
                 embedding vector(64) NOT NULL
             );",
        )
        .await?;
    let closer_keys = [1_i64, 2, 3, 4, 5, 6, 8, 10, 11];
    let mut rows = closer_keys
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, controlled_cosine_vector(0.99 - index as f32 * 0.01)))
        .collect::<Vec<_>>();
    let tied = controlled_cosine_vector(0.5);
    rows.push((7, tied.clone()));
    rows.push((9, tied));
    rows.sort_unstable_by_key(|(key, _vector)| *key);
    let insert = client
        .prepare(
            "INSERT INTO docs_seam_tie_fixture (doc_id, embedding)
             VALUES ($1, $2::text::vector);",
        )
        .await?;
    for (key, vector) in rows {
        client
            .execute(&insert, &[&key, &format_vector(&vector)])
            .await?;
    }
    Ok(())
}

fn controlled_cosine_vector(first: f32) -> Vec<f32> {
    let mut vector = vec![0.0; DIMENSION];
    vector[0] = first;
    vector[1] = (1.0 - first * first).sqrt();
    vector
}

async fn verify_no_boundary_ties_in_postgres(
    client: &tokio_postgres::Client,
    queries: &[Vec<f32>],
) -> Result<f64> {
    const MINIMUM_BOUNDARY_GAP: f64 = 1e-6;

    client.batch_execute("SET enable_indexscan = off;").await?;
    let statement = client
        .prepare(
            "SELECT embedding <=> $1::text::vector AS distance
             FROM docs_seam_fixture
             ORDER BY embedding <=> $1::text::vector ASC, doc_id ASC
             LIMIT 11;",
        )
        .await?;
    let mut narrow_boundaries = Vec::new();
    let mut minimum_gap = f64::INFINITY;
    for (query_index, query) in queries.iter().enumerate() {
        let rows = client.query(&statement, &[&format_vector(query)]).await?;
        ensure!(rows.len() == K + 1);
        let kth = rows[K - 1].try_get::<_, f64>(0)?;
        let next = rows[K].try_get::<_, f64>(0)?;
        let gap = next - kth;
        minimum_gap = minimum_gap.min(gap);
        if gap <= MINIMUM_BOUNDARY_GAP {
            narrow_boundaries.push(format!(
                "query={query_index}, kth={kth:.17e}, next={next:.17e}, gap={gap:.17e}"
            ));
        }
    }
    client.batch_execute("SET enable_indexscan = on;").await?;
    ensure!(
        narrow_boundaries.is_empty(),
        "F-pg PostgreSQL top-k boundary gaps must exceed {MINIMUM_BOUNDARY_GAP:.1e}: {}",
        narrow_boundaries.join("; ")
    );
    Ok(minimum_gap)
}

async fn verify_tie_break_table(client: &tokio_postgres::Client) -> Result<()> {
    let query = controlled_cosine_vector(1.0);
    let statement = client
        .prepare(
            "SELECT doc_id
             FROM docs_seam_tie_fixture
             ORDER BY embedding <=> $1::text::vector ASC, doc_id ASC
             LIMIT 10;",
        )
        .await?;
    for _ in 0..3 {
        let rows = client.query(&statement, &[&format_vector(&query)]).await?;
        let keys = rows
            .iter()
            .map(|row| row.try_get::<_, i64>(0))
            .collect::<Result<Vec<_>, _>>()?;
        ensure!(keys.contains(&7));
        ensure!(!keys.contains(&9));
    }
    Ok(())
}

fn encode_f32_frame(cohort: &CohortName, vector: &[f32]) -> Result<Vec<u8>> {
    let name = cohort.as_str().as_bytes();
    let vector_bytes = vector_bytes(vector);
    let total_len = FIXED_FRAME_HEADER_LEN
        .checked_add(name.len())
        .and_then(|value| value.checked_add(vector_bytes.len()))
        .context("fixture frame length overflow")?;
    let frame_len = u32::try_from(total_len.checked_sub(4).context("invalid frame length")?)?;
    let mut frame = Vec::with_capacity(total_len);
    frame.extend_from_slice(&frame_len.to_le_bytes());
    frame.extend_from_slice(&FRAME_MAGIC);
    frame.extend_from_slice(&FRAME_VERSION.to_le_bytes());
    frame.extend_from_slice(&1_u32.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(name.len())?.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(vector.len())?.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(vector_bytes.len())?.to_le_bytes());
    frame.extend_from_slice(name);
    frame.extend_from_slice(&vector_bytes);
    Ok(frame)
}

fn vector_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn format_vector(vector: &[f32]) -> String {
    format!(
        "[{}]",
        vector
            .iter()
            .map(f32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}
