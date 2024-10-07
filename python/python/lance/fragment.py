# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset Fragment"""

from __future__ import annotations

import json
import uuid
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import pyarrow as pa

from .dependencies import _check_for_pandas
from .dependencies import pandas as pd
from .file import LanceFileReader, LanceFileWriter
from .lance import _Fragment, _write_fragments
from .lance import _FragmentMetadata as _FragmentMetadata
from .progress import FragmentWriteProgress, NoopFragmentWriteProgress

if TYPE_CHECKING:
    from .dataset import LanceDataset, LanceScanner, ReaderLike
    from .schema import LanceSchema


DEFAULT_MAX_BYTES_PER_FILE = 90 * 1024 * 1024 * 1024


class FragmentMetadata:
    """Metadata of a Fragment in the dataset."""

    def __init__(self, metadata: str):
        """Construct a FragmentMetadata from a JSON representation of the metadata.

        Internal use only.
        """
        self._metadata = _FragmentMetadata.from_json(metadata)

    @classmethod
    def from_metadata(cls, metadata: _FragmentMetadata):
        instance = cls.__new__(cls)
        instance._metadata = metadata
        return instance

    def __repr__(self):
        return self._metadata.__repr__()

    def __reduce__(self):
        return (FragmentMetadata, (self._metadata.json(),))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FragmentMetadata):
            return False
        return self._metadata.__eq__(other._metadata)

    def to_json(self) -> str:
        """Serialize :class:`FragmentMetadata` to a JSON blob"""
        return json.loads(self._metadata.json())

    @staticmethod
    def from_json(json_data: str) -> FragmentMetadata:
        """Reconstruct :class:`FragmentMetadata` from a JSON blob"""
        return FragmentMetadata(json_data)

    def data_files(self) -> Iterable[str]:
        """Return the data files of the fragment"""
        return self._metadata.data_files()

    def deletion_file(self):
        """Return the deletion file, if any"""
        return self._metadata.deletion_file()

    @property
    def id(self) -> int:
        return self._metadata.id

    def with_id(self, new_id: Optional[int]) -> FragmentMetadata:
        """Replaces the id of the fragment with the given id."""
        return FragmentMetadata.from_metadata(self._metadata.with_id(new_id))


class LanceFragment(pa.dataset.Fragment):
    def __init__(
        self,
        dataset: "LanceDataset",
        fragment_id: Optional[int],
        *,
        fragment: Optional[_Fragment] = None,
    ):
        self._ds = dataset
        if fragment is None:
            if fragment_id is None:
                raise ValueError("Either fragment or fragment_id must be specified")
            fragment = dataset.get_fragment(fragment_id)._fragment
        self._fragment = fragment
        if self._fragment is None:
            raise ValueError(f"Fragment id does not exist: {fragment_id}")

    def __repr__(self):
        return self._fragment.__repr__()

    def __reduce__(self):
        from .dataset import LanceDataset

        ds = LanceDataset(self._ds.uri, self._ds.version)
        return LanceFragment, (ds, self.fragment_id)

    @staticmethod
    def create_from_file(
        filename: Union[str, Path],
        dataset: LanceDataset,
        fragment_id: int,
    ) -> FragmentMetadata:
        """Create a fragment from the given datafile uri.

        This can be used if the datafile is loss from dataset.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        filename: str
            The filename of the datafile.
        dataset: LanceDataset
            The dataset that the fragment belongs to.
        fragment_id: int
            The ID of the fragment.
        """
        fragment = _Fragment.create_from_file(filename, dataset._ds, fragment_id)
        return FragmentMetadata(fragment.json())

    @staticmethod
    def create(
        dataset_uri: Union[str, Path],
        data: Union[pa.Table, pa.RecordBatchReader],
        fragment_id: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        max_rows_per_group: int = 1024,
        progress: Optional[FragmentWriteProgress] = None,
        mode: str = "append",
        *,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
    ) -> FragmentMetadata:
        """Create a :class:`FragmentMetadata` from the given data.

        This can be used if the dataset is not yet created.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        dataset_uri: str
            The URI of the dataset.
        fragment_id: int
            The ID of the fragment.
        data: pa.Table or pa.RecordBatchReader
            The data to be written to the fragment.
        schema: pa.Schema, optional
            The schema of the data. If not specified, the schema will be inferred
            from the data.
        max_rows_per_group: int, default 1024
            The maximum number of rows per group in the data file.
        progress: FragmentWriteProgress, optional
            *Experimental API*. Progress tracking for writing the fragment. Pass
            a custom class that defines hooks to be called when each fragment is
            starting to write and finishing writing.
        mode: str, default "append"
            The write mode. If "append" is specified, the data will be checked
            against the existing dataset's schema. Otherwise, pass "create" or
            "overwrite" to assign new field ids to the schema.
        data_storage_version: optional, str, default None
            The version of the data storage format to use. Newer versions are more
            efficient but require newer versions of lance to read.  The default (None)
            will use the latest stable version.  See the user guide for more details.
        use_legacy_format: bool, default None
            Deprecated parameter.  Use data_storage_version instead.
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.

        See Also
        --------
        lance.dataset.LanceOperation.Overwrite :
            The operation used to create a new dataset or overwrite one using
            fragments created with this API. See the doc page for an example of
            using this API.
        lance.dataset.LanceOperation.Append :
            The operation used to append fragments created with this API to an
            existing dataset. See the doc page for an example of using this API.

        Returns
        -------
        FragmentMetadata
        """
        if use_legacy_format is not None:
            warnings.warn(
                "use_legacy_format is deprecated, use data_storage_version instead",
                DeprecationWarning,
            )
            if use_legacy_format:
                data_storage_version = "legacy"
            else:
                data_storage_version = "stable"

        if _check_for_pandas(data) and isinstance(data, pd.DataFrame):
            reader = pa.Table.from_pandas(data, schema=schema).to_reader()
        elif isinstance(data, pa.Table):
            reader = data.to_reader()
        elif isinstance(data, pa.dataset.Scanner):
            reader = data.to_reader()
        elif isinstance(data, pa.RecordBatchReader):
            reader = data
        else:
            raise TypeError(f"Unknown data_obj type {type(data)}")

        if isinstance(dataset_uri, Path):
            dataset_uri = str(dataset_uri)
        if progress is None:
            progress = NoopFragmentWriteProgress()

        inner_meta = _Fragment.create(
            dataset_uri,
            fragment_id,
            reader,
            max_rows_per_group=max_rows_per_group,
            progress=progress,
            mode=mode,
            data_storage_version=data_storage_version,
            storage_options=storage_options,
        )
        return FragmentMetadata(inner_meta.json())

    def with_new_data_file(
        self, filename: str, target_field_ids: List[int]
    ) -> Tuple[FragmentMetadata, LanceSchema]:
        fragment, schema = self._fragment.with_new_data_file(filename, target_field_ids)
        return FragmentMetadata.from_metadata(fragment), schema

    @property
    def fragment_id(self):
        return self._fragment.id()

    def count_rows(
        self, filter: Optional[Union[pa.compute.Expression, str]] = None
    ) -> int:
        if filter is not None:
            raise ValueError("Does not support filter at the moment")
        return self._fragment.count_rows()

    @property
    def num_deletions(self) -> int:
        """Return the number of deleted rows in this fragment."""
        return self._fragment.num_deletions

    @property
    def physical_rows(self) -> int:
        """
        Return the number of rows originally in this fragment.

        To get the number of rows after deletions, use
        :meth:`count_rows` instead.
        """
        return self._fragment.physical_rows

    @property
    def physical_schema(self) -> pa.Schema:
        # override the pyarrow super class method otherwise causes segfault
        raise NotImplementedError("Not implemented yet for LanceFragment")

    @property
    def partition_expression(self) -> pa.Schema:
        # override the pyarrow super class method otherwise causes segfault
        raise NotImplementedError("Not implemented yet for LanceFragment")

    def head(self, num_rows: int) -> pa.Table:
        return self.scanner(limit=num_rows).to_table()

    def scanner(
        self,
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
        batch_readahead: int = 16,
    ) -> "LanceScanner":
        """See Dataset::scanner for details"""
        filter_str = str(filter) if filter is not None else None

        columns_arg = {}
        if isinstance(columns, dict):
            # convert to list of tuples
            columns_arg = {"columns_with_transform": list(columns.items())}
        elif isinstance(columns, list):
            columns_arg = {"columns": columns}

        s = self._fragment.scanner(
            batch_size=batch_size,
            filter=filter_str,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            batch_readahead=batch_readahead,
            **columns_arg,
        )
        from .dataset import LanceScanner

        return LanceScanner(s, self._ds)

    def take(
        self,
        indices,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
    ) -> pa.Table:
        return pa.Table.from_batches([self._fragment.take(indices, columns=columns)])

    def to_batches(
        self,
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
        batch_readahead: int = 16,
    ) -> Iterator[pa.RecordBatch]:
        return self.scanner(
            columns=columns,
            batch_size=batch_size,
            filter=filter,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            batch_readahead=batch_readahead,
        ).to_batches()

    def to_table(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
    ) -> pa.Table:
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
        ).to_table()

    def merge_columns(
        self,
        value_func: Callable[[pa.RecordBatch], pa.RecordBatch],
        columns: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[FragmentMetadata, LanceSchema]:
        """Add columns to this Fragment.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        value_func: Callable.
            A function that takes a RecordBatch as input and returns a RecordBatch.
        columns: Optional[list[str]].
            If specified, only the columns in this list will be passed to the
            value_func. Otherwise, all columns will be passed to the value_func.

        See Also
        --------
        lance.dataset.LanceOperation.Merge :
            The operation used to commit these changes to the dataset. See the
            doc page for an example of using this API.

        Returns
        -------
        Tuple[FragmentMetadata, LanceSchema]
            A new fragment with the added column(s) and the final schema.
        """
        updater = self._fragment.updater(columns, batch_size)

        while True:
            batch = updater.next()
            if batch is None:
                break
            new_value = value_func(batch)
            if not isinstance(new_value, pa.RecordBatch):
                raise ValueError(
                    f"value_func must return a Pyarrow RecordBatch, "
                    f"got {type(new_value)}"
                )

            updater.update(new_value)
        metadata = updater.finish()
        schema = updater.schema()
        return FragmentMetadata.from_metadata(metadata), schema

    def add_columns(
        self,
        value_func: Callable[[pa.RecordBatch], pa.RecordBatch],
        columns: Optional[list[str]] = None,
    ) -> FragmentMetadata:
        """Add columns to this Fragment.

        .. deprecated:: 0.10.14
            Use :meth:`merge_columns` instead.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        value_func: Callable.
            A function that takes a RecordBatch as input and returns a RecordBatch.
        columns: Optional[list[str]].
            If specified, only the columns in this list will be passed to the
            value_func. Otherwise, all columns will be passed to the value_func.

        See Also
        --------
        lance.dataset.LanceOperation.Merge :
            The operation used to commit these changes to the dataset. See the
            doc page for an example of using this API.

        Returns
        -------
        FragmentMetadata
            A new fragment with the added column(s).
        """
        warnings.warn(
            "LanceFragment.add_columns is deprecated, use LanceFragment.merge_columns "
            "instead",
            DeprecationWarning,
        )
        return self.merge_columns(value_func, columns)[0]

    def delete(self, predicate: str) -> FragmentMetadata | None:
        """Delete rows from this Fragment.

        This will add or update the deletion file of this fragment. It does not
        modify or delete the data files of this fragment. If no rows are left after
        the deletion, this method will return None.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        predicate: str
            A SQL predicate that specifies the rows to delete.

        Returns
        -------
        FragmentMetadata or None
            A new fragment containing the new deletion file, or None if no rows left.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> tab = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> dataset = lance.write_dataset(tab, "dataset")
        >>> frag = dataset.get_fragment(0)
        >>> frag.delete("a > 1")
        Fragment { id: 0, files: ..., deletion_file: Some(...), ...}
        >>> frag.delete("a > 0") is None
        True

        See Also
        --------
        lance.dataset.LanceOperation.Delete :
            The operation used to commit these changes to a dataset. See the
            doc page for an example of using this API.
        """
        raw_fragment = self._fragment.delete(predicate)
        if raw_fragment is None:
            return None
        return FragmentMetadata.from_metadata(raw_fragment.metadata())

    @property
    def schema(self) -> pa.Schema:
        """Return the schema of this fragment."""

        return self._fragment.schema()

    def data_files(self):
        """Return the data files of this fragment."""

        return self._fragment.data_files()

    def deletion_file(self):
        """Return the deletion file, if any"""
        return self._fragment.deletion_file()

    @property
    def metadata(self) -> FragmentMetadata:
        """Return the metadata of this fragment.

        Returns
        -------
        FragmentMetadata
        """
        return FragmentMetadata.from_metadata(self._fragment.metadata())


def write_fragments(
    data: ReaderLike,
    dataset_uri: Union[str, Path],
    schema: Optional[pa.Schema] = None,
    *,
    mode: str = "append",
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
    progress: Optional[FragmentWriteProgress] = None,
    data_storage_version: Optional[str] = None,
    use_legacy_format: Optional[bool] = None,
    storage_options: Optional[Dict[str, str]] = None,
) -> List[FragmentMetadata]:
    """
    Write data into one or more fragments.

    .. warning::

        This is a low-level API intended for manually implementing distributed
        writes. For most users, :func:`lance.write_dataset` is the recommended API.

    Parameters
    ----------
    data : pa.Table or pa.RecordBatchReader
        The data to be written to the fragment.
    dataset_uri : str
        The URI of the dataset.
    schema : pa.Schema, optional
        The schema of the data. If not specified, the schema will be inferred
        from the data.
    mode : str, default "append"
        The write mode. If "append" is specified, the data will be checked
        against the existing dataset's schema. Otherwise, pass "create" or
        "overwrite" to assign new field ids to the schema.
    max_rows_per_file : int, default 1024 * 1024
        The maximum number of rows per data file.
    max_rows_per_group : int, default 1024
        The maximum number of rows per group in the data file.
    max_bytes_per_file : int, default 90 * 1024 * 1024 * 1024
        The max number of bytes to write before starting a new file. This is a
        soft limit. This limit is checked after each group is written, which
        means larger groups may cause this to be overshot meaningfully. This
        defaults to 90 GB, since we have a hard limit of 100 GB per file on
        object stores.
    progress : FragmentWriteProgress, optional
        *Experimental API*. Progress tracking for writing the fragment. Pass
        a custom class that defines hooks to be called when each fragment is
        starting to write and finishing writing.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default (None)
        will use the 2.0 version.  See the user guide for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    storage_options : Optional[Dict[str, str]]
        Extra options that make sense for a particular storage connection. This is
        used to store connection parameters like credentials, endpoint, etc.

    Returns
    -------
    List[FragmentMetadata]
        A list of :class:`FragmentMetadata` for the fragments written. The
        fragment ids are left as zero meaning they are not yet specified. They
        will be assigned when the fragments are committed to a dataset.
    """
    if _check_for_pandas(data) and isinstance(data, pd.DataFrame):
        reader = pa.Table.from_pandas(data, schema=schema).to_reader()
    elif isinstance(data, pa.Table):
        reader = data.to_reader()
    elif isinstance(data, pa.dataset.Scanner):
        reader = data.to_reader()
    elif isinstance(data, pa.RecordBatchReader):
        reader = data
    else:
        raise TypeError(f"Unknown data_obj type {type(data)}")

    if isinstance(dataset_uri, Path):
        dataset_uri = str(dataset_uri)

    if use_legacy_format is not None:
        warnings.warn(
            "use_legacy_format is deprecated, use data_storage_version instead",
            DeprecationWarning,
        )
        if use_legacy_format:
            data_storage_version = "legacy"
        else:
            data_storage_version = "stable"

    fragments = _write_fragments(
        dataset_uri,
        reader,
        mode=mode,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        progress=progress,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )
    return [FragmentMetadata.from_metadata(frag) for frag in fragments]


class CombineInputsTask:
    def __init__(
        self,
        dataset_uri: str,
        input_uris: List[str],
        rows_to_skip: int,
        num_rows: int,
        fragment_id: int,
    ):
        self._dataset_uri = dataset_uri
        self._input_uris = input_uris
        self._rows_to_skip = rows_to_skip
        self._num_rows = num_rows
        self._fragment_id = fragment_id

    def execute(
        self,
        *,
        batch_size: Optional[int] = None,
        storage_options: Optional[Dict[str, str]] = None,
    ) -> Tuple[FragmentMetadata, LanceSchema]:
        """
        Combines ``num_rows`` rows from the given input files (starting from the first
        and advancing in order) into a new fragment with the given fragment ID.

        Fragments on the end may be skipped or only partially included if ``num_rows``
        does not span all the given fragments.  If ``num_rows`` is greater than the
        total number of rows in the given fragments, an error will be raised.

        This will read all data from the given input files and write them into a new
        fragment. The new fragment will have a new path (UUID) and will have the
        given fragment ID.

        This is an internal operation.

        This operation is only supported with v2 files.
        """
        if len(self._input_uris) == 0:
            raise ValueError("No fragments to combine")

        schema = LanceFileReader(self._input_uris[0]).metadata().schema

        def scan_fragments():
            rows_remaining = self._num_rows
            skip_remaining = self._rows_to_skip
            print(
                f"Scanning files to write fragment with {rows_remaining} rows and skipping {skip_remaining} rows of input"
            )
            for input_uri in self._input_uris:
                reader = LanceFileReader(input_uri)
                print(f"Starting on file with {reader.metadata().num_rows} rows")
                try:
                    if schema != reader.metadata().schema:
                        print("Schema mismatch")
                        raise ValueError("Input schemas do not match")
                    for batch in reader.read_all(batch_size=batch_size).to_batches():
                        print(f"Batch arrived with {batch.num_rows} rows")
                        if skip_remaining > 0:
                            if batch.num_rows <= skip_remaining:
                                print(
                                    f"Fully skipping batch since skip_remaining={skip_remaining}"
                                )
                                skip_remaining -= batch.num_rows
                                continue
                            batch = batch.slice(
                                skip_remaining, batch.num_rows - skip_remaining
                            )
                            print(f"Proceeding with {batch.num_rows} rows after skip")
                            skip_remaining = 0
                        if batch.num_rows > rows_remaining:
                            batch = batch.slice(0, rows_remaining)
                            rows_remaining = 0
                        else:
                            rows_remaining -= batch.num_rows
                        yield batch
                        if rows_remaining == 0:
                            print("Bailing early since rows_remaining=0")
                            return
                    print("Bailing naturally since fully scanned input file")
                except Exception as e:
                    print(e)
                    raise e
            if rows_remaining == 0:
                raise ValueError("Not enough rows in the input files to combine")

        filename = f"{uuid.uuid4()}.lance"
        with LanceFileWriter(f"{self._dataset_uri}/data/{filename}") as writer:
            for batch in scan_fragments():
                writer.write_batch(batch)

        from lance import LanceDataset

        ds = LanceDataset(self._dataset_uri)
        frag = ds.get_fragment(self._fragment_id)

        return frag.with_new_data_file(filename, [-1 for _ in range(len(schema))])

    def to_json(self) -> str:
        return json.dumps(
            {
                "input_uris": self._input_uris,
                "dataset_uri": self._dataset_uri,
                "rows_to_skip": self._rows_to_skip,
                "num_rows": self._num_rows,
                "fragment_id": self._fragment_id,
            }
        )

    @staticmethod
    def from_json(json_data: str) -> CombineInputsTask:
        data = json.loads(json_data)
        return CombineInputsTask(
            data["dataset_uri"],
            data["input_uris"],
            data["rows_to_skip"],
            data["num_rows"],
            data["fragment_id"],
        )

    def __reduce__(self):
        return (CombineInputsTask.from_json, (self.to_json(),))


def plan_alignment(
    dataset: LanceDataset, input_uris: List[str]
) -> List[CombineInputsTask]:
    """
    Plan the alignment of input files into fragments

    Given an ordered list of input URIs this will create tasks to convert
    the Lance files into a new list of fragments.  Each fragment in the new list will
    correspond to one fragment in the dataset, with the same fragment ID and the same
    number of rows.

    This is an internal operation that will prepare for a merge operation to add a new
    column to the dataset.
    """
    target_frags = dataset.get_fragments()
    if len(target_frags) == 0:
        raise ValueError("Dataset has no fragments")

    target_frag_iter = iter(target_frags)

    def next_target():
        current_target = next(target_frag_iter, None)
        if current_target is None:
            print("Finished with targets")
            return -1, -1
        else:
            print(f"Processing target with {current_target.physical_rows} rows")
            return current_target.fragment_id, current_target.physical_rows

    target_id, rows_in_target = next_target()
    rows_remaining = rows_in_target

    def count_rows(input_uri):
        return LanceFileReader(input_uri).metadata().num_rows

    tasks = []
    input_files = []
    rows_to_skip = 0
    for input_uri in input_uris:
        rows_in_input = LanceFileReader(input_uri).metadata().num_rows
        print(f"Processing input with {rows_in_input} rows")
        remaining_in_input = rows_in_input
        while remaining_in_input > 0:
            if target_id < 0:
                raise ValueError(
                    "More rows in input files than there are in dataset fragments"
                )
            input_files.append(input_uri)
            if remaining_in_input != rows_in_input:
                rows_to_skip = rows_in_input - remaining_in_input
            if remaining_in_input >= rows_remaining:
                tasks.append(
                    CombineInputsTask(
                        dataset.uri,
                        input_files,
                        rows_to_skip,
                        rows_in_target,
                        target_id,
                    )
                )
                print(
                    f"Creating task with {len(input_files)} input files skip={rows_to_skip} num_rows={rows_in_target}"
                )
                input_files = []
                remaining_in_input -= rows_remaining

                target_id, rows_in_target = next_target()
                rows_remaining = rows_in_target
            else:
                print("Finished input without creating new task")
                rows_remaining -= rows_in_input
                remaining_in_input = 0
    try:
        next(target_frag_iter)
        raise ValueError("More rows in dataset fragments than there are in input files")
    except StopIteration:
        pass

    return tasks
