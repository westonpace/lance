use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use lance_encoding::encoder::encode_batch;
use pprof::criterion::{Output, PProfProfiler};

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    let data = lance_datagen::gen()
        .col(None, lance_datagen::array::rand_type(&DataType::Int32))
        .into_batch_rows(lance_datagen::RowCount::from(1024 * 1024))
        .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let encoded = rt.block_on(encode_batch(&data, 1024 * 1024)).unwrap();
    group.bench_function("decode", |b| {
        b.iter(|| {
            let batch = rt
                .block_on(lance_encoding::decoder::decode_batch(&encoded))
                .unwrap();
            assert_eq!(data.num_rows(), batch.num_rows());
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_decode);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_distance, bench_small_distance);
criterion_main!(benches);
