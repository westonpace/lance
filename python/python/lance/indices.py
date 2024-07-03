# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import math
import warnings
from typing import TYPE_CHECKING, Optional, Union

import pyarrow as pa

from lance.file import LanceFileReader, LanceFileWriter

from .lance import indices

if TYPE_CHECKING:
    from .dependencies import torch


class PqModel:
    """
    A class that represents a trained PQ model

    Can be saved / loaded to checkpoint progress.
    """

    def __init__(self, num_subvectors: int, codebook: pa.FixedSizeListArray):
        self.num_subvectors = num_subvectors
        """The number of subvectors to divide source vectors into"""
        self.codebook = codebook
        """The centroids of the PQ clusters"""

    @property
    def dimension(self):
        """The dimension of the vectors this model was trained on"""
        return self.codebook.type.list_size

    def save(self, uri: str):
        """
        Save the PQ model to a lance file.

        Parameters
        ----------

        uri: str
            The URI to save the model to.  The URI can be a local file path or a
            cloud storage path.
        """
        with LanceFileWriter(
            uri,
            pa.schema(
                [pa.field("codebook", self.codebook.type)],
                metadata={b"num_subvectors": str(self.num_subvectors).encode()},
            ),
        ) as writer:
            batch = pa.table([self.codebook], names=["codebook"])
            writer.write_batch(batch)

    @classmethod
    def load(cls, uri: str):
        """
        Load a PQ model from a lance file.

        Parameters
        ----------

        uri: str
            The URI to load the model from.  The URI can be a local file path or a
            cloud storage path.
        """
        reader = LanceFileReader(uri)
        num_rows = reader.metadata().num_rows
        metadata = reader.metadata().schema.metadata
        num_subvectors = int(metadata[b"num_subvectors"].decode())
        codebook = (
            reader.read_all(batch_size=num_rows).to_table().column("codebook").chunk(0)
        )
        return cls(num_subvectors, codebook)


class IvfModel:
    """
    A class that represents a trained IVF model.
    """

    def __init__(self, centroids: pa.Array, distance_type: str):
        self.centroids = centroids
        """The centroids of the IVF clusters"""
        self.distance_type = distance_type
        """The distance type used to train the IVF model"""

    @property
    def num_partitions(self) -> int:
        """
        The number of partitions / centroids in the IVF model
        """
        return len(self.centroids)

    def save(self, uri: str):
        """
        Save the IVF model to a lance file.

        Parameters
        ----------

        uri: str
            The URI to save the model to.  The URI can be a local file path or a
            cloud storage path.
        """
        with LanceFileWriter(
            uri,
            pa.schema(
                [pa.field("centroids", self.centroids.type)],
                metadata={b"distance_type": self.distance_type.encode()},
            ),
        ) as writer:
            batch = pa.table([self.centroids], names=["centroids"])
            writer.write_batch(batch)

    @classmethod
    def load(cls, uri: str):
        """
        Load an IVF model from a lance file.

        Parameters
        ----------

        uri: str
            The URI to load the model from.  The URI can be a local file path or a
            cloud storage path.
        """
        reader = LanceFileReader(uri)
        num_rows = reader.metadata().num_rows
        metadata = reader.metadata().schema.metadata
        distance_type = metadata[b"distance_type"].decode()
        centroids = (
            reader.read_all(batch_size=num_rows).to_table().column("centroids").chunk(0)
        )
        return cls(centroids, distance_type)


class IndicesBuilder:
    """
    A class with helper functions for building indices on a dataset.

    This methods in this class can break down the process of building indices into
    smaller steps.  This can be useful for debugging and checkpointing when building
    indices for extremely large datasets.

    This class is intended for advanced users that need to create vector indices at
    large scales.

    The methods in this class are **experimental** and may change in future versions.

    For datasets with 10s of millions or fewer rows it will likely be simpler to just
    use the `create_index` method on the dataset object.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def train_ivf(
        self,
        column,
        num_partitions=None,
        *,
        distance_type="l2",
        accelerator: Optional[Union[str, "torch.Device"]] = None,
        sample_rate: int = 256,
        max_iters: int = 50,
    ) -> IvfModel:
        """
        Train IVF centroids for the given vector column.

        This will run k-means clustering on the given vector column to train the IVF
        centroids. This is the first step in several vector indices.  The centroids
        will be used to partition the vectors into different clusters.

        IVF centroids are trained from a sample of the data (determined by the
        sample_rate).  While this sample is not huge it might still be quite large.

        K-means is an iterative algorithm that can be computationally expensive. The
        accelerator argument can be used to offload the computation to a hardware
        accelerator such as a GPU or TPU.

        Parameters
        ----------

        column: str
            The vector column to partition, must be a fixed size list of floats
            or 1-dimensional fixed-shape tensor column.
        num_partitions: int
            The number of partitions to train.  Large values are more expensive to
            train and can lead to longer search times.  Smaller values could lead to
            overtraining, reduced recall, and require large nprobes values.  If not
            specified the default will be the integer nearest the square root of the
            number of rows.
        distance_type: "l2" | "dot" | "cosine"
            The distance type to used.  This is defined in more detail in the LanceDB
            documentation on creating indices.
        accelerator: str | torch.Device
            An optional accelerator to use to offload computation to specialized
            hardware.  Currently supported values are "cuda" and "mps".
        sample_rate: int
            IVF is trained on a random sample of the dataset.  The sample_rate
            determines the size of this sample.  There will be sample_rate rows loaded
            for each partition for a total of sample_rate * num_partitions rows.  If
            the dataset does not contain enough rows an error will be raised.
        max_iters: int
            K-means is an iterative algorithm that is run until it converges.  In
            some cases, k-means will not converge but will cycle between various
            possible minima.  In these cases we must terminate or run forever.  The
            max_iters parameter defines a cutoff at which we terminate training.
        """
        column = self._normalize_column(column)
        num_rows = self.dataset.count_rows()
        num_partitions = self._determine_num_partitions(num_partitions, num_rows)
        self._verify_ivf_sample_rate(sample_rate, num_partitions, num_rows)
        distance_type = self._normalize_distance_type(distance_type)
        self._verify_ivf_params(num_partitions)

        if accelerator is None:
            dimension = self.dataset.schema.field(column[0]).type.list_size
            ivf_centroids = indices.train_ivf_model(
                self.dataset._ds,
                column[0],
                dimension,
                num_partitions,
                distance_type,
                sample_rate,
                max_iters,
            )
            return IvfModel(ivf_centroids, distance_type)
        else:
            # Use accelerator to train ivf centroids
            from .vector import train_ivf_centroids_on_accelerator

            ivf_centroids, _ = train_ivf_centroids_on_accelerator(
                self.dataset,
                column[0],
                num_partitions,
                distance_type,
                accelerator,
                sample_rate=sample_rate,
                max_iters=max_iters,
            )
            num_dims = ivf_centroids.shape[1]
            ivf_centroids.shape = -1
            flat_centroids_array = pa.array(ivf_centroids)
            centroids_array = pa.FixedSizeListArray.from_arrays(
                flat_centroids_array, num_dims
            )
            return IvfModel(centroids_array, distance_type)

    def train_pq(
        self,
        column,
        ivf_model: IvfModel,
        num_subvectors=None,
        *,
        sample_rate: int = 256,
        max_iters: int = 50,
    ) -> PqModel:
        """
        Train a PQ model for a given column.

        This will run k-means clustering on each subvector to determine the centroids
        that will be used to quantize the subvectors.  This step runs against a
        randomly chosen sample of the data.  The sample size is typically quite small
        and PQ training is relatively fast regardless of dataset scale.  As a result,
        accelerators are not needed here.

        Parameters
        ----------

        column: str
            The vector column to quantize, must be a fixed size list of floats
            or 1-dimensional fixed-shape tensor column.
        ivf_model: IvfModel
            The IVF model to use to partition the vectors into clusters.  This is
            needed because PQ is trained on residuals from the IVF model.
        num_subvectors: int
            The number of subvectors to divide the source vectors into.  This must be
            a divisor of the vector dimension.  If not specified the default will be
            the vector dimension divided by 16 if the dimension is divisible by 16,
            otherwise the vector dimension divided by 8 if the dimension is divisible
            by 8.

            Automatic calculation of num_subvectors will fail if the vector dimension
            is not divisible by 16 or 8.  In this case you must specify num_subvectors
            manually (though any value you choose is likely to lead to poor performance)
        sample_rate: int
            This parameter is used in the same way as in the IVF model.
        max_iters: int
            This parameter is used in the same way as in the IVF model.
        """
        column = self._normalize_column(column)
        num_rows = self.dataset.count_rows()
        dimension = self.dataset.schema.field(column[0]).type.list_size
        self.dataset.schema.field(column[0]).type.list_size
        num_subvectors = self._normalize_pq_params(num_subvectors, dimension)
        self._verify_pq_sample_rate(num_rows, sample_rate)
        distance_type = ivf_model.distance_type
        pq_codebook = indices.train_pq_model(
            self.dataset._ds,
            column[0],
            dimension,
            num_subvectors,
            distance_type,
            sample_rate,
            max_iters,
            ivf_model.centroids,
        )
        return PqModel(num_subvectors, pq_codebook)

    def _determine_num_partitions(self, num_partitions: Optional[int], num_rows: int):
        if num_partitions is None:
            return round(math.sqrt(num_rows))
        return num_partitions

    def _normalize_pq_params(self, num_subvectors: int, dimension: int):
        if num_subvectors is None:
            if dimension % 16 == 0:
                return dimension // 16
            elif dimension % 8 == 0:
                return dimension // 8
            else:
                raise ValueError(
                    f"vector dimension {dimension} is not divisible by 16 or 8."
                    " PQ performance will be poor.  Cowardly refusing to create"
                    " PQ model.  Please specify num_subvectors manually."
                )
        if not isinstance(num_subvectors, int):
            raise ValueError("num_subvectors must be an int")
        if num_subvectors < 1:
            raise ValueError("num_subvectors must be greater than 0")
        if num_subvectors > dimension:
            raise ValueError(
                "num_subvectors must be less than or equal to the dimension of"
                " the vectors"
            )
        if dimension % num_subvectors != 0:
            raise ValueError(
                "dimension ({dimension}) must be divisible by num_subvectors"
                " ({num_subvectors}) without remainder"
            )
        return num_subvectors

    def _verify_base_sample_rate(self, sample_rate: int):
        if not isinstance(sample_rate, int) or sample_rate < 2:
            raise ValueError(
                f"The sample_rate must be an int greater than 1, got {sample_rate}"
            )

    def _verify_pq_sample_rate(self, num_rows: int, sample_rate: int):
        self._verify_base_sample_rate(sample_rate)
        if 256 * sample_rate > num_rows:
            raise ValueError(
                "There are not enough rows in the dataset to create PQ"
                f" codebook with a sample rate of {sample_rate}.  {sample_rate * 256}"
                f" rows needed and there are {num_rows}"
            )

    def _verify_ivf_sample_rate(
        self, sample_rate: int, num_partitions: int, num_rows: int
    ):
        self._verify_base_sample_rate(sample_rate)
        if num_partitions * sample_rate > num_rows:
            raise ValueError(
                "There are not enough rows in the dataset to create IVF centroids with"
                f" {num_partitions} partitions and a sample rate of {sample_rate}."
                f" {sample_rate * num_partitions} rows needed and there are {num_rows}"
            )

    def _verify_ivf_params(self, num_partitions):
        if num_partitions is None:
            raise ValueError(
                "num_partitions and num_sub_vectors are required for IVF_PQ"
            )
        if isinstance(num_partitions, float):
            warnings.warn("num_partitions is float, converting to int")
            num_partitions = int(num_partitions)
        elif not isinstance(num_partitions, int):
            raise TypeError(f"num_partitions must be int, got {type(num_partitions)}")

    def _normalize_distance_type(self, distance_type):
        if not isinstance(distance_type, str) or distance_type.lower() not in [
            "l2",
            "cosine",
            "euclidean",
            "dot",
        ]:
            raise ValueError(f"Distance type {distance_type} not supported.")
        return distance_type.lower()

    def _normalize_column(self, column):
        # Only support building index for 1 column from the API aspect, however
        # the internal implementation might support building multi-column index later.
        if isinstance(column, str):
            column = [column]

        # validate args
        for c in column:
            if c not in self.dataset.schema.names:
                raise KeyError(f"{c} not found in schema")
            field = self.dataset.schema.field(c)
            if not (
                pa.types.is_fixed_size_list(field.type)
                or (
                    isinstance(field.type, pa.FixedShapeTensorType)
                    and len(field.type.shape) == 1
                )
            ):
                raise TypeError(
                    f"Vector column {c} must be FixedSizeListArray "
                    f"1-dimensional FixedShapeTensorArray, got {field.type}"
                )
            if not pa.types.is_floating(field.type.value_type):
                raise TypeError(
                    f"Vector column {c} must have floating value type, "
                    f"got {field.type.value_type}"
                )

        return column
