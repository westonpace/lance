use arrow_array::types::{Float32Type, Int16Type, Int32Type, Int64Type, Int8Type};
use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, Criterion,
    Throughput,
};

use lance_datagen::{
    generator::ArrayGenerator, BatchCount, ByteCount, Dimension, RoundingBehavior,
};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const NUM_BATCHES: u32 = 100;
const KB_PER_BATCH: u64 = 128;
const BYTES_PER_BATCH: u64 = KB_PER_BATCH * 1024;
const BYTES_PER_BENCH: u64 = BYTES_PER_BATCH * NUM_BATCHES as u64;

fn bench_gen<M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    id: &str,
    gen_factory: impl Fn() -> Box<dyn ArrayGenerator>,
) {
    let num_batches: BatchCount = BatchCount::from(NUM_BATCHES);

    group.bench_function(id, |b| {
        b.iter(|| {
            let reader = lance_datagen::gen()
                .col(gen_factory())
                .into_reader_bytes(
                    ByteCount::from(BYTES_PER_BATCH),
                    num_batches,
                    RoundingBehavior::ExactOrErr,
                )
                .unwrap();
            reader.for_each(|batch| assert!(batch.is_ok()));
        })
    });
}

fn bench_step_gen<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("step");
    group.throughput(Throughput::Bytes(BYTES_PER_BENCH));
    bench_gen(&mut group, "i8", || {
        lance_datagen::array::step::<Int8Type>()
    });
    bench_gen(&mut group, "16", || {
        lance_datagen::array::step::<Int16Type>()
    });
    bench_gen(&mut group, "i32", || {
        lance_datagen::array::step::<Int32Type>()
    });
    bench_gen(&mut group, "i64", || {
        lance_datagen::array::step::<Int64Type>()
    });
    group.finish();
}

fn bench_fill_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill");
    group.throughput(Throughput::Bytes(BYTES_PER_BENCH));
    bench_gen(&mut group, "fill_i8", || {
        lance_datagen::array::fill::<Int8Type>(42)
    });
    bench_gen(&mut group, "fill_i16", || {
        lance_datagen::array::fill::<Int16Type>(42)
    });
    bench_gen(&mut group, "fill_i32", || {
        lance_datagen::array::fill::<Int32Type>(42)
    });
    bench_gen(&mut group, "fill_i64", || {
        lance_datagen::array::fill::<Int64Type>(42)
    });
    bench_gen(&mut group, "fill_varbin", || {
        lance_datagen::array::fill_varbin(vec![
            0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB,
        ])
    });
    bench_gen(&mut group, "fill_utf8", || {
        lance_datagen::array::fill_utf8("hello wo".to_string())
    });
    group.finish();
}

fn bench_rand_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand");
    group.throughput(Throughput::Bytes(BYTES_PER_BENCH));
    let seed = lance_datagen::DEFAULT_SEED;
    bench_gen(&mut group, "rand_i8", || {
        lance_datagen::array::rand::<Int8Type>(seed)
    });
    bench_gen(&mut group, "rand_i16", || {
        lance_datagen::array::rand::<Int16Type>(seed)
    });
    bench_gen(&mut group, "rand_i32", || {
        lance_datagen::array::rand::<Int32Type>(seed)
    });
    bench_gen(&mut group, "rand_i64", || {
        lance_datagen::array::rand::<Int64Type>(seed)
    });
    bench_gen(&mut group, "rand_varbin", || {
        lance_datagen::array::rand_varbin(seed, ByteCount::from(12))
    });
    bench_gen(&mut group, "rand_utf8", || {
        lance_datagen::array::rand_utf8(seed, ByteCount::from(12))
    });
    bench_gen(&mut group, "rand_vec", || {
        lance_datagen::array::rand_vec::<Float32Type>(seed, Dimension::from(512))
    });
    bench_gen(&mut group, "rand_dict_i32_utf8", || {
        lance_datagen::array::dict::<Int32Type>(lance_datagen::array::rand_utf8(
            seed,
            ByteCount::from(8),
        ))
    });
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_step_gen, bench_fill_gen, bench_rand_gen);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_step_gen, bench_fill_gen, bench_rand_gen);

criterion_main!(benches);
