# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
An internal module for generating Arrow data for use in testing and benchmarking.
"""

from typing import TYPE_CHECKING, Optional, Union

import pyarrow as pa

from .lance import datagen

if TYPE_CHECKING:
    import os


def is_datagen_supported():
    return datagen.is_datagen_supported()


def rand_batches(
    schema: pa.Schema,
    *,
    num_batches: Optional[int] = None,
    batch_size_bytes: Optional[int] = None,
):
    if not datagen.is_datagen_supported():
        raise NotImplementedError(
            "This version of lance was not built with the datagen feature"
        )
    batch_iter = datagen.rand_batches(schema, num_batches, batch_size_bytes)
    return pa.RecordBatchReader.from_batches(schema, batch_iter)


def from_yaml(
    yaml: str,
    *,
    num_rows: int,
    num_batches: int,
) -> pa.RecordBatchReader:
    """Create a RecordBatchReader from a YAML datagen specification.

    Parameters
    ----------
    yaml : str
        A YAML string describing the data generator configuration.
    num_rows : int
        Number of rows per batch.
    num_batches : int
        Number of batches to generate.

    Returns
    -------
    pa.RecordBatchReader
    """
    if not datagen.is_datagen_supported():
        raise NotImplementedError(
            "This version of lance was not built with the datagen feature"
        )
    return datagen.from_yaml(yaml, num_rows, num_batches)


def from_yaml_file(
    path: Union[str, "os.PathLike[str]"],
    *,
    num_rows: int,
    num_batches: int,
) -> pa.RecordBatchReader:
    """Create a RecordBatchReader from a YAML datagen specification file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the YAML file describing the data generator configuration.
    num_rows : int
        Number of rows per batch.
    num_batches : int
        Number of batches to generate.

    Returns
    -------
    pa.RecordBatchReader
    """
    if not datagen.is_datagen_supported():
        raise NotImplementedError(
            "This version of lance was not built with the datagen feature"
        )
    import os

    return datagen.from_yaml_file(os.fspath(path), num_rows, num_batches)
