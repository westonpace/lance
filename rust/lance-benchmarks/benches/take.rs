// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use arrow_schema::Schema as ArrowSchema;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::{stream::FuturesUnordered, TryStreamExt};
use lance::{
    dataset::{ProjectionRequest, WriteParams},
    deps::lance_file::version::LanceFileVersion,
    Dataset,
};
use rand::seq::SliceRandom;

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const NUM_ROWS_TO_COPY: i64 = 6_000_000;
const NUM_INDICES: usize = 100_000;

async fn prepare_dataset(data_storage_version: LanceFileVersion) -> Arc<Dataset> {
    let source_uri = "/home/pace/lance-benchmarks-ci-datasets/tpch-2.1";

    println!("Opening source dataset from {}", source_uri);
    let source_dataset = Dataset::open(source_uri).await.unwrap();

    println!(
        "Reading first {} rows from source dataset",
        NUM_ROWS_TO_COPY
    );
    let mut scanner = source_dataset.scan();
    scanner.limit(Some(NUM_ROWS_TO_COPY), None).unwrap();

    let stream = scanner.try_into_stream().await.unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();

    println!("Collected {} batches", batches.len());

    let arrow_schema: ArrowSchema = source_dataset.schema().into();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), Arc::new(arrow_schema));

    let write_params = WriteParams {
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };

    println!("Writing in-memory dataset");
    let memory_dataset = Dataset::write(reader, "memory://benchmark_data", Some(write_params))
        .await
        .unwrap();

    println!(
        "In-memory dataset created with {} rows",
        memory_dataset.count_rows(None).await.unwrap()
    );

    Arc::new(memory_dataset)
}

fn generate_random_indices(max_value: u64, count: usize) -> Arc<[u64]> {
    let mut rng = rand::rng();
    let mut all_indices: Vec<u64> = (0..max_value).collect();
    let (selected, _) = all_indices.partial_shuffle(&mut rng, count);
    selected.to_vec().into()
}

struct TakeParams {
    rows_per_take: usize,
    data_storage_version: LanceFileVersion,
    threaded_runtime: bool,
}

impl std::fmt::Display for TakeParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rows_per_take={}&data_storage_version={}&threaded_runtime={}",
            self.rows_per_take, self.data_storage_version, self.threaded_runtime
        )
    }
}

fn bench_take(c: &mut Criterion) {
    let mut group = c.benchmark_group("take");

    for threaded_runtime in [true, false] {
        for data_storage_version in [LanceFileVersion::V2_0, LanceFileVersion::V2_1] {
            let runtime = tokio::runtime::Builder::new_multi_thread().build().unwrap();
            let dataset = runtime.block_on(prepare_dataset(data_storage_version));
            let projection = ProjectionRequest::from_columns(&["l_linenumber"], dataset.schema());

            for rows_per_take in [1, 10, 100] {
                group.throughput(Throughput::Elements(rows_per_take as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        "take_operation",
                        TakeParams {
                            rows_per_take,
                            data_storage_version,
                            threaded_runtime,
                        },
                    ),
                    &TakeParams {
                        rows_per_take,
                        data_storage_version,
                        threaded_runtime,
                    },
                    |b, take_params| {
                        let runtime = if threaded_runtime {
                            tokio::runtime::Builder::new_multi_thread().build().unwrap()
                        } else {
                            tokio::runtime::Builder::new_current_thread()
                                .build()
                                .unwrap()
                        };

                        let mut counter = 0;
                        let indices = generate_random_indices(NUM_ROWS_TO_COPY as u64, NUM_INDICES);
                        b.iter(|| {
                            let dataset = dataset.clone();
                            let indices = indices.clone();
                            let projection = projection.clone();
                            let start = counter;
                            counter += take_params.rows_per_take;
                            let end = counter;
                            if counter + take_params.rows_per_take > NUM_INDICES {
                                counter = 0;
                            }
                            let indices = &indices[start..end];
                            runtime.block_on(async {
                                dataset.take(indices, projection).await.unwrap()
                            });
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

pub const TAKES_PER_ITERATION: usize = 1000;

fn bench_parallel_take(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_take");

    for data_storage_version in [LanceFileVersion::V2_0, LanceFileVersion::V2_1] {
        let runtime = tokio::runtime::Builder::new_multi_thread().build().unwrap();
        let dataset = runtime.block_on(prepare_dataset(data_storage_version));
        let projection = ProjectionRequest::from_columns(&["l_linenumber"], dataset.schema());

        for rows_per_take in [1, 10, 100] {
            group.throughput(Throughput::Elements(
                rows_per_take as u64 * TAKES_PER_ITERATION as u64,
            ));
            group.bench_with_input(
                BenchmarkId::new(
                    "take_operation",
                    TakeParams {
                        rows_per_take,
                        data_storage_version,
                        threaded_runtime: true,
                    },
                ),
                &TakeParams {
                    rows_per_take,
                    data_storage_version,
                    threaded_runtime: true,
                },
                |b, take_params| {
                    let runtime = if take_params.threaded_runtime {
                        tokio::runtime::Builder::new_multi_thread().build().unwrap()
                    } else {
                        tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap()
                    };

                    let mut counter = 0;
                    let indices = generate_random_indices(NUM_ROWS_TO_COPY as u64, NUM_INDICES);
                    b.iter(|| {
                        let dataset = dataset.clone();
                        let indices = indices.clone();
                        let projection = projection.clone();

                        let futures_unordered = FuturesUnordered::new();
                        for _ in 0..TAKES_PER_ITERATION {
                            let start = counter;
                            counter += take_params.rows_per_take;
                            let end = counter;
                            if counter + take_params.rows_per_take > NUM_INDICES {
                                counter = 0;
                            }
                            let dataset = dataset.clone();
                            let indices = indices.clone();
                            let projection = projection.clone();
                            futures_unordered.push(runtime.spawn(async move {
                                let indices = &indices[start..end];
                                dataset.take(indices, projection).await.unwrap()
                            }));
                        }
                        runtime.block_on(async move {
                            for future in futures_unordered {
                                future.await.unwrap();
                            }
                        });
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_take, bench_parallel_take
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(10);
    targets = bench_take, bench_parallel_take
);

criterion_main!(benches);
