# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import lance
import numpy as np
import pyarrow as pa
from lance.indices import IndicesBuilder, IvfModel


def gen_dataset(tmpdir, datatype=np.float32):
    vectors = np.random.randn(10000, 128).astype(datatype)
    vectors.shape = -1
    vectors = pa.FixedSizeListArray.from_arrays(vectors, 128)
    table = pa.Table.from_arrays([vectors], names=["vectors"])
    ds = lance.write_dataset(table, str(tmpdir / "dataset"))

    return ds


def test_ivf_centroids(tmpdir):
    ds = gen_dataset(tmpdir)

    ivf = IndicesBuilder(ds).train_ivf("vectors", sample_rate=16)

    assert ivf.distance_type == "l2"
    assert len(ivf.centroids) == 100

    ivf.save(str(tmpdir / "ivf"))
    reloaded = IvfModel.load(str(tmpdir / "ivf"))
    assert reloaded.distance_type == "l2"
    assert ivf.centroids == reloaded.centroids


def test_ivf_centroids_column_type(tmpdir):
    def check(column_type, typename):
        ds = gen_dataset(tmpdir / typename, column_type)
        ivf = IndicesBuilder(ds).train_ivf(
            "vectors", sample_rate=16, accelerator="cuda"
        )
        assert len(ivf.centroids) == 100
        ivf.save(str(tmpdir / f"ivf_{typename}"))
        reloaded = IvfModel.load(str(tmpdir / f"ivf_{typename}"))
        assert ivf.centroids == reloaded.centroids

    check(np.float16, "f16")
    check(np.float32, "f32")
    check(np.float64, "f64")


def test_ivf_centroids_distance_type(tmpdir):
    ds = gen_dataset(tmpdir)

    def check(distance_type):
        ivf = IndicesBuilder(ds).train_ivf(
            "vectors", sample_rate=16, distance_type=distance_type
        )
        assert ivf.distance_type == distance_type
        ivf.save(str(tmpdir / "ivf"))
        reloaded = IvfModel.load(str(tmpdir / "ivf"))
        assert reloaded.distance_type == distance_type

    check("l2")
    check("cosine")
    check("dot")


def test_num_partitions(tmpdir):
    ds = gen_dataset(tmpdir)

    ivf = IndicesBuilder(ds).train_ivf("vectors", sample_rate=16, num_partitions=10)
    assert ivf.num_partitions() == 10
