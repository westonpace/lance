import pyarrow as pa

import lance._datagen as datagen


def test_rand_batches():
    schema = pa.schema([pa.field("int", pa.int64()), pa.field("text", pa.utf8())])

    batches = datagen.rand_batches(schema, batch_size_bytes=1024, num_batches=10)

    assert len(batches) == 10
    for batch in batches:
        assert batch.num_rows == math.ceil(1024 / 16)
        assert batch.schema == schema
