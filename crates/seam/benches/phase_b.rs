use std::collections::BTreeMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use seam::aggregate::aggregate;
use seam::math::quantile_type7;
use seam::model::{
    AggregationConfig, AggregationInput, IntermediateMetadata, IntermediatePart, ListedPart,
    MeasuredSample, SweepMeasurement,
};

const SAMPLE_COUNT: usize = 10_000;
const EF_GRID: [i32; 5] = [10, 20, 40, 80, 160];
const WINDOW_START: u64 = 1_783_512_000;
const WINDOW_SECONDS: u32 = 600;

fn phase_b_benchmarks(criterion: &mut Criterion) {
    let input = aggregation_input();
    criterion.bench_function("phase_b/aggregate_10k_samples_5_ef", |bencher| {
        bencher.iter(|| aggregate(black_box(&input)).expect("benchmark fixture must aggregate"))
    });

    let quantile_values = (0..SAMPLE_COUNT)
        .map(|index| (index % 101) as f64 / 100.0)
        .collect::<Vec<_>>();
    criterion.bench_function("phase_b/type7_quantile_10k", |bencher| {
        bencher.iter(|| {
            quantile_type7(black_box(&quantile_values), black_box(0.05))
                .expect("benchmark quantile input must be valid")
        })
    });
}

fn aggregation_input() -> AggregationInput {
    let samples = (0..SAMPLE_COUNT)
        .map(|record_index| MeasuredSample {
            record_index: i32::try_from(record_index)
                .expect("benchmark record index must fit in i32"),
            vector_hash: record_index as u64,
            dup_count: 1,
            ground_truth_latency_ms: 400.5,
            sweeps: EF_GRID
                .into_iter()
                .map(|ef| {
                    (
                        ef,
                        SweepMeasurement {
                            recall: f64::from(ef) / 160.0,
                            latency_ms: f64::from(ef) / 10.0,
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        })
        .collect();
    let metadata = IntermediateMetadata {
        format_version: 1,
        cohort: "benchmark/phase-b".to_owned(),
        part_ulid: "01J00000000000000000000000".to_owned(),
        window_start: WINDOW_START,
        window_seconds: WINDOW_SECONDS,
        received_frame_count: SAMPLE_COUNT as u64,
        record_count: SAMPLE_COUNT as u64,
        index: "fixture".to_owned(),
        table: "docs_fixture".to_owned(),
        column: "embedding".to_owned(),
        key: "doc_id".to_owned(),
        k: 10,
        ef_grid: EF_GRID.to_vec(),
        failed_count: 0,
        measured_count: SAMPLE_COUNT as u64,
        computed_at_us: WINDOW_START * 1_000_000,
    };
    AggregationInput {
        config: AggregationConfig {
            cohort: metadata.cohort.clone(),
            target_name: "recall".to_owned(),
            index: metadata.index.clone(),
            table: metadata.table.clone(),
            column: metadata.column.clone(),
            key: metadata.key.clone(),
            k: metadata.k,
            value: 0.9,
            percentile: 0.95,
            window_duration_seconds: u64::from(WINDOW_SECONDS),
            storage_window_seconds: WINDOW_SECONDS,
            ef_grid: metadata.ef_grid.clone(),
            train_fraction: 0.7,
            split_seed: 7,
            min_samples: 1_000,
        },
        round_end: WINDOW_START + u64::from(WINDOW_SECONDS),
        computed_at: "2026-07-08T12:10:00Z".to_owned(),
        phase_a_abort: None,
        phase_a_incompatible_parts: 0,
        previous_round: None,
        listed_parts: vec![ListedPart {
            part_ulid: metadata.part_ulid.clone(),
            window_start: metadata.window_start,
            window_seconds: metadata.window_seconds,
            received_frame_count: metadata.received_frame_count,
            record_count: metadata.record_count,
        }],
        intermediates: vec![IntermediatePart { metadata, samples }],
    }
}

criterion_group!(benches, phase_b_benchmarks);
criterion_main!(benches);
