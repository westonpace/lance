import pyarrow as pa

def rand_batches(
    schema: pa.Schema, num_batches: int = None, batch_size_bytes: int = None
): ...
