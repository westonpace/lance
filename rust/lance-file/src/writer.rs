// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod statistics;

use std::marker::PhantomData;

use arrow_array::builder::{ArrayBuilder, PrimitiveBuilder};
use arrow_array::cast::{as_large_list_array, as_list_array, as_struct_array};
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use async_recursion::async_recursion;
use async_trait::async_trait;
use lance_arrow::*;
use lance_core::datatypes::{Encoding, Field, Schema, SchemaCompareOptions};
use lance_core::{Error, Result};
use lance_io::encodings::{
    binary::BinaryEncoder, dictionary::DictionaryEncoder, plain::PlainEncoder, Encoder,
};
use lance_io::object_store::ObjectStore;
use lance_io::object_writer::ObjectWriter;
use lance_io::traits::{WriteExt, Writer};
use object_store::path::Path;
use snafu::{location, Location};
use tokio::io::AsyncWriteExt;

use crate::format::metadata::{Metadata, StatisticsMetadata};
use crate::format::{MAGIC, MAJOR_VERSION, MINOR_VERSION};
use crate::page_table::{PageInfo, PageTable};

/// The file format currently includes a "manifest" where it stores the schema for
/// self-describing files.  Historically this has been a table format manifest that
/// is empty except for the schema field.
///
/// Since this crate is not aware of the table format we need this to be provided
/// externally.  You should always use lance_table::io::manifest::ManifestDescribing
/// for this today.
#[async_trait]
pub trait ManifestProvider {
    /// Store the schema in the file
    ///
    /// This should just require writing the schema (or a manifest wrapper) as a proto struct
    ///
    /// Note: the dictionaries have already been written by this point and the schema should
    /// be populated with the dictionary lengths/offsets
    async fn store_schema(
        object_writer: &mut ObjectWriter,
        schema: &Schema,
    ) -> Result<Option<usize>>;
}

/// Implementation of ManifestProvider that does not store the schema
pub(crate) struct NotSelfDescribing {}

#[async_trait]
impl ManifestProvider for NotSelfDescribing {
    async fn store_schema(_: &mut ObjectWriter, _: &Schema) -> Result<Option<usize>> {
        Ok(None)
    }
}

/// [FileWriter] writes Arrow [RecordBatch] to one Lance file.
///
/// ```ignored
/// use lance::io::FileWriter;
/// use futures::stream::Stream;
///
/// let mut file_writer = FileWriter::new(object_store, &path, &schema);
/// while let Ok(batch) = stream.next().await {
///     file_writer.write(&batch).unwrap();
/// }
/// // Need to close file writer to flush buffer and footer.
/// file_writer.shutdown();
/// ```
pub struct FileWriter<M: ManifestProvider + Send + Sync> {
    object_writer: ObjectWriter,
    schema: Schema,
    batch_id: i32,
    page_table: PageTable,
    metadata: Metadata,
    stats_collector: Option<statistics::StatisticsCollector>,
    manifest_provider: PhantomData<M>,
}

#[derive(Debug, Clone, Default)]
pub struct FileWriterOptions {
    /// The field ids to collect statistics for.
    ///
    /// If None, will collect for all fields in the schema (that support stats).
    /// If an empty vector, will not collect any statistics.
    pub collect_stats_for_fields: Option<Vec<i32>>,
}

impl<M: ManifestProvider + Send + Sync> FileWriter<M> {
    pub async fn try_new(
        object_store: &ObjectStore,
        path: &Path,
        schema: Schema,
        options: &FileWriterOptions,
    ) -> Result<Self> {
        let object_writer = object_store.create(path).await?;
        Self::with_object_writer(object_writer, schema, options)
    }

    pub fn with_object_writer(
        object_writer: ObjectWriter,
        schema: Schema,
        options: &FileWriterOptions,
    ) -> Result<Self> {
        let collect_stats_for_fields = if let Some(stats_fields) = &options.collect_stats_for_fields
        {
            stats_fields.clone()
        } else {
            schema.field_ids()
        };

        let stats_collector = if !collect_stats_for_fields.is_empty() {
            let stats_schema = schema.project_by_ids(&collect_stats_for_fields);
            statistics::StatisticsCollector::try_new(&stats_schema)
        } else {
            None
        };

        Ok(Self {
            object_writer,
            schema,
            batch_id: 0,
            page_table: PageTable::default(),
            metadata: Metadata::default(),
            stats_collector,
            manifest_provider: PhantomData,
        })
    }

    /// Write a [RecordBatch] to the open file.
    /// All RecordBatch will be treated as one RecordBatch on disk
    ///
    /// Returns [Err] if the schema does not match with the batch.
    pub async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        for batch in batches {
            // Compare, ignore metadata and dictionary
            //   dictionary should have been checked earlier and could be an expensive check
            let schema = Schema::try_from(batch.schema().as_ref())?;
            schema.check_compatible(&self.schema, &SchemaCompareOptions::default())?;
        }

        // If we are collecting stats for this column, collect them.
        // Statistics need to traverse nested arrays, so it's a separate loop
        // from writing which is done on top-level arrays.
        if let Some(stats_collector) = &mut self.stats_collector {
            for (field, arrays) in fields_in_batches(batches, &self.schema) {
                if let Some(stats_builder) = stats_collector.get_builder(field.id) {
                    let stats_row = statistics::collect_statistics(&arrays);
                    stats_builder.append(stats_row);
                }
            }
        }

        // Copy a list of fields to avoid borrow checker error.
        let fields = self.schema.fields.clone();
        for field in fields.iter() {
            let arrs = batches
                .iter()
                .map(|batch| {
                    batch.column_by_name(&field.name).ok_or_else(|| Error::IO {
                        message: format!("FileWriter::write: Field '{}' not found", field.name),
                        location: location!(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            Self::write_array(
                &mut self.object_writer,
                field,
                &arrs,
                self.batch_id,
                &mut self.page_table,
            )
            .await?;
        }
        let batch_length = batches.iter().map(|b| b.num_rows() as i32).sum();
        self.metadata.push_batch_length(batch_length);

        // It's imperative we complete any in-flight requests, since we are
        // returning control to the caller. If the caller takes a long time to
        // write the next batch, the in-flight requests will not be polled and
        // may time out.
        self.object_writer.flush().await?;

        self.batch_id += 1;
        Ok(())
    }

    pub async fn finish(&mut self) -> Result<usize> {
        self.write_footer().await?;
        self.object_writer.shutdown().await?;
        let num_rows = self
            .metadata
            .batch_offsets
            .last()
            .cloned()
            .unwrap_or_default();
        Ok(num_rows as usize)
    }

    /// Total records written in this file.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Total bytes written so far
    pub async fn tell(&mut self) -> Result<usize> {
        self.object_writer.tell().await
    }

    /// Returns the in-flight multipart ID.
    pub fn multipart_id(&self) -> &str {
        &self.object_writer.multipart_id
    }

    /// Return the id of the next batch to be written.
    pub fn next_batch_id(&self) -> i32 {
        self.batch_id
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[async_recursion]
    async fn write_array(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrs: &[&ArrayRef],
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        assert!(!arrs.is_empty());
        let data_type = arrs[0].data_type();
        let arrs_ref = arrs.iter().map(|a| a.as_ref()).collect::<Vec<_>>();

        match data_type {
            DataType::Null => {
                Self::write_null_array(
                    object_writer,
                    field,
                    arrs_ref.as_slice(),
                    batch_id,
                    page_table,
                )
                .await
            }
            dt if dt.is_fixed_stride() => {
                Self::write_fixed_stride_array(
                    object_writer,
                    field,
                    arrs_ref.as_slice(),
                    batch_id,
                    page_table,
                )
                .await
            }
            dt if dt.is_binary_like() => {
                Self::write_binary_array(
                    object_writer,
                    field,
                    arrs_ref.as_slice(),
                    batch_id,
                    page_table,
                )
                .await
            }
            DataType::Dictionary(key_type, _) => {
                Self::write_dictionary_arr(
                    object_writer,
                    field,
                    arrs_ref.as_slice(),
                    key_type,
                    batch_id,
                    page_table,
                )
                .await
            }
            dt if dt.is_struct() => {
                let struct_arrays = arrs.iter().map(|a| as_struct_array(a)).collect::<Vec<_>>();
                Self::write_struct_array(
                    object_writer,
                    field,
                    struct_arrays.as_slice(),
                    batch_id,
                    page_table,
                )
                .await
            }
            DataType::FixedSizeList(_, _) | DataType::FixedSizeBinary(_) => {
                Self::write_fixed_stride_array(
                    object_writer,
                    field,
                    arrs_ref.as_slice(),
                    batch_id,
                    page_table,
                )
                .await
            }
            DataType::List(_) => {
                Self::write_list_array(
                    object_writer,
                    field,
                    arrs_ref.as_slice(),
                    batch_id,
                    page_table,
                )
                .await
            }
            DataType::LargeList(_) => {
                Self::write_large_list_array(
                    object_writer,
                    field,
                    arrs_ref.as_slice(),
                    batch_id,
                    page_table,
                )
                .await
            }
            _ => Err(Error::Schema {
                message: format!("FileWriter::write: unsupported data type: {data_type}"),
                location: location!(),
            }),
        }
    }

    async fn write_null_array(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrs: &[&dyn Array],
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(object_writer.tell().await?, arrs_length as usize);
        page_table.set(field.id, batch_id, page_info);
        Ok(())
    }

    /// Write fixed size array, including, primtiives, fixed size binary, and fixed size list.
    async fn write_fixed_stride_array(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrs: &[&dyn Array],
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::Plain));
        assert!(!arrs.is_empty());
        let data_type = arrs[0].data_type();

        let mut encoder = PlainEncoder::new(object_writer, data_type);
        let pos = encoder.encode(arrs).await?;
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(pos, arrs_length as usize);
        page_table.set(field.id, batch_id, page_info);
        Ok(())
    }

    /// Write var-length binary arrays.
    async fn write_binary_array(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrs: &[&dyn Array],
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::VarBinary));
        let mut encoder = BinaryEncoder::new(object_writer);
        let pos = encoder.encode(arrs).await?;
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(pos, arrs_length as usize);
        page_table.set(field.id, batch_id, page_info);
        Ok(())
    }

    async fn write_dictionary_arr(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrs: &[&dyn Array],
        key_type: &DataType,
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::Dictionary));

        // Write the dictionary keys.
        let mut encoder = DictionaryEncoder::new(object_writer, key_type);
        let pos = encoder.encode(arrs).await?;
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(pos, arrs_length as usize);
        page_table.set(field.id, batch_id, page_info);
        Ok(())
    }

    #[async_recursion]
    async fn write_struct_array(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrays: &[&StructArray],
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        arrays
            .iter()
            .for_each(|a| assert_eq!(a.num_columns(), field.children.len()));

        for child in &field.children {
            let mut arrs: Vec<&ArrayRef> = Vec::new();
            for struct_array in arrays {
                let arr = struct_array
                    .column_by_name(&child.name)
                    .ok_or(Error::Schema {
                        message: format!(
                            "FileWriter: schema mismatch: column {} does not exist in array: {:?}",
                            child.name,
                            struct_array.data_type()
                        ),
                        location: location!(),
                    })?;
                arrs.push(arr);
            }
            Self::write_array(object_writer, child, arrs.as_slice(), batch_id, page_table).await?;
        }
        Ok(())
    }

    async fn write_list_array(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrs: &[&dyn Array],
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        let capacity: usize = arrs.iter().map(|a| a.len()).sum();
        let mut list_arrs: Vec<ArrayRef> = Vec::new();
        let mut pos_builder: PrimitiveBuilder<Int32Type> =
            PrimitiveBuilder::with_capacity(capacity);

        let mut last_offset: usize = 0;
        pos_builder.append_value(last_offset as i32);
        for array in arrs.iter() {
            let list_arr = as_list_array(*array);
            let offsets = list_arr.value_offsets();

            assert!(!offsets.is_empty());
            let start_offset = offsets[0].as_usize();
            let end_offset = offsets[offsets.len() - 1].as_usize();

            let list_values = list_arr.values();
            let sliced_values = list_values.slice(start_offset, end_offset - start_offset);
            list_arrs.push(sliced_values);

            offsets
                .iter()
                .skip(1)
                .map(|b| b.as_usize() - start_offset + last_offset)
                .for_each(|o| pos_builder.append_value(o as i32));
            last_offset = pos_builder.values_slice()[pos_builder.len() - 1_usize] as usize;
        }

        let positions: &dyn Array = &pos_builder.finish();
        Self::write_fixed_stride_array(object_writer, field, &[positions], batch_id, page_table)
            .await?;
        let arrs = list_arrs.iter().collect::<Vec<_>>();
        Self::write_array(
            object_writer,
            &field.children[0],
            arrs.as_slice(),
            batch_id,
            page_table,
        )
        .await
    }

    async fn write_large_list_array(
        object_writer: &mut ObjectWriter,
        field: &Field,
        arrs: &[&dyn Array],
        batch_id: i32,
        page_table: &mut PageTable,
    ) -> Result<()> {
        let capacity: usize = arrs.iter().map(|a| a.len()).sum();
        let mut list_arrs: Vec<ArrayRef> = Vec::new();
        let mut pos_builder: PrimitiveBuilder<Int64Type> =
            PrimitiveBuilder::with_capacity(capacity);

        let mut last_offset: usize = 0;
        pos_builder.append_value(last_offset as i64);
        for array in arrs.iter() {
            let list_arr = as_large_list_array(*array);
            let offsets = list_arr.value_offsets();

            assert!(!offsets.is_empty());
            let start_offset = offsets[0].as_usize();
            let end_offset = offsets[offsets.len() - 1].as_usize();

            let sliced_values = list_arr
                .values()
                .slice(start_offset, end_offset - start_offset);
            list_arrs.push(sliced_values);

            offsets
                .iter()
                .skip(1)
                .map(|b| b.as_usize() - start_offset + last_offset)
                .for_each(|o| pos_builder.append_value(o as i64));
            last_offset = pos_builder.values_slice()[pos_builder.len() - 1_usize] as usize;
        }

        let positions: &dyn Array = &pos_builder.finish();
        Self::write_fixed_stride_array(object_writer, field, &[positions], batch_id, page_table)
            .await?;
        let arrs = list_arrs.iter().collect::<Vec<_>>();
        Self::write_array(
            object_writer,
            &field.children[0],
            arrs.as_slice(),
            batch_id,
            page_table,
        )
        .await
    }

    async fn write_statistics(&mut self) -> Result<Option<StatisticsMetadata>> {
        let statistics = self
            .stats_collector
            .as_mut()
            .map(|collector| collector.finish());

        match statistics {
            Some(Ok(stats_batch)) if stats_batch.num_rows() > 0 => {
                debug_assert_eq!(self.next_batch_id() as usize, stats_batch.num_rows());
                let schema = Schema::try_from(stats_batch.schema().as_ref())?;
                let leaf_field_ids = schema.field_ids();

                let mut stats_page_table = PageTable::default();
                for (i, field) in schema.fields.iter().enumerate() {
                    Self::write_array(
                        &mut self.object_writer,
                        field,
                        &[stats_batch.column(i)],
                        0, // Only one batch for statistics.
                        &mut stats_page_table,
                    )
                    .await?;
                }

                let page_table_position =
                    stats_page_table.write(&mut self.object_writer, 0).await?;

                Ok(Some(StatisticsMetadata {
                    schema,
                    leaf_field_ids,
                    page_table_position,
                }))
            }
            Some(Err(e)) => Err(e),
            _ => Ok(None),
        }
    }

    /// Writes the dictionaries (using plain/binary encoding) into the file
    ///
    /// The offsets and lengths of the written buffers are stored in the given
    /// schema so that the dictionaries can be loaded in the future.
    async fn write_dictionaries(writer: &mut ObjectWriter, schema: &mut Schema) -> Result<()> {
        // Write dictionary values.
        let max_field_id = schema.max_field_id().unwrap_or(-1);
        for field_id in 0..max_field_id + 1 {
            if let Some(field) = schema.mut_field_by_id(field_id) {
                if field.data_type().is_dictionary() {
                    let dict_info = field.dictionary.as_mut().ok_or_else(|| Error::IO {
                        message: format!("Lance field {} misses dictionary info", field.name),
                        location: location!(),
                    })?;

                    let value_arr = dict_info.values.as_ref().ok_or_else(|| Error::IO {
                        message: format!(
                        "Lance field {} is dictionary type, but misses the dictionary value array",
                        field.name
                    ),
                        location: location!(),
                    })?;

                    let data_type = value_arr.data_type();
                    let pos = match data_type {
                        dt if dt.is_numeric() => {
                            let mut encoder = PlainEncoder::new(writer, dt);
                            encoder.encode(&[value_arr]).await?
                        }
                        dt if dt.is_binary_like() => {
                            let mut encoder = BinaryEncoder::new(writer);
                            encoder.encode(&[value_arr]).await?
                        }
                        _ => {
                            return Err(Error::IO {
                                message: format!(
                                    "Does not support {} as dictionary value type",
                                    value_arr.data_type()
                                ),
                                location: location!(),
                            });
                        }
                    };
                    dict_info.offset = pos;
                    dict_info.length = value_arr.len();
                }
            }
        }
        Ok(())
    }

    async fn write_footer(&mut self) -> Result<()> {
        // Step 1. Write page table.
        let field_id_offset = *self.schema.field_ids().iter().min().unwrap();
        let pos = self
            .page_table
            .write(&mut self.object_writer, field_id_offset)
            .await?;
        self.metadata.page_table_position = pos;

        // Step 2. Write statistics.
        self.metadata.stats_metadata = self.write_statistics().await?;

        // Step 3. Write manifest and dictionary values.
        Self::write_dictionaries(&mut self.object_writer, &mut self.schema).await?;
        let pos = M::store_schema(&mut self.object_writer, &self.schema).await?;

        // Step 4. Write metadata.
        self.metadata.manifest_position = pos;
        let pos = self.object_writer.write_struct(&self.metadata).await?;

        // Step 5. Write magics.
        self.object_writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
    }
}

/// Walk through the schema and return arrays with their Lance field.
///
/// This skips over nested arrays and fields within list arrays. It does walk
/// over the children of structs.
fn fields_in_batches<'a>(
    batches: &'a [RecordBatch],
    schema: &'a Schema,
) -> impl Iterator<Item = (&'a Field, Vec<&'a ArrayRef>)> {
    let num_columns = batches[0].num_columns();
    let array_iters = (0..num_columns).map(|col_i| {
        batches
            .iter()
            .map(|batch| batch.column(col_i))
            .collect::<Vec<_>>()
    });
    let mut to_visit: Vec<(&'a Field, Vec<&'a ArrayRef>)> =
        schema.fields.iter().zip(array_iters).collect();

    std::iter::from_fn(move || {
        loop {
            let (field, arrays): (_, Vec<&'a ArrayRef>) = to_visit.pop()?;
            match field.data_type() {
                DataType::Struct(_) => {
                    for (i, child_field) in field.children.iter().enumerate() {
                        let child_arrays = arrays
                            .iter()
                            .map(|arr| as_struct_array(*arr).column(i))
                            .collect::<Vec<&'a ArrayRef>>();
                        to_visit.push((child_field, child_arrays));
                    }
                    continue;
                }
                // We only walk structs right now.
                _ if field.data_type().is_nested() => continue,
                _ => return Some((field, arrays)),
            }
        }
    })
}

pub mod v2 {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
    use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, Buffer};
    use arrow_schema::DataType;
    use futures::stream::FuturesUnordered;
    use futures::StreamExt;
    use lance_core::datatypes::Schema;
    use lance_core::utils::tokio::spawn_cpu;
    use lance_core::{Error, Result};
    use lance_io::object_writer::ObjectWriter;
    use lance_io::traits::{WriteExt, Writer};
    use prost::Message;
    use snafu::{location, Location};
    use tokio::io::AsyncWriteExt;

    use crate::datatypes::FieldsWithMeta;
    use crate::format::pb::{self, FileDescriptor};
    use crate::format::{MAGIC, MAJOR_VERSION, MINOR_VERSION};

    /// An encoded buffer
    pub struct EncodedBuffer {
        /// If true, the buffer should be stored as "data"
        /// If false, the buffer should be stored as "metadata"
        ///
        /// Metadata buffers are typically small buffers that should be cached.  For example,
        /// this might be a small dictionary when data has been dictionary encoded.  Or it might
        /// contain a skip block when data has been RLE encoded.
        pub is_data: bool,
        /// Buffers that make up the encoded buffer
        ///
        /// All of these buffers should be written to the file as one contiguous buffer
        ///
        /// This is a Vec to allow for zero-copy
        ///
        /// For example, if we are asked to write 3 primitive arrays of 1000 rows and we can write them all
        /// as one page then this will be the value buffers from the 3 primitive arrays
        pub parts: Vec<Buffer>,
    }

    /// An array that has been encoded, along with a description of the encoding
    pub struct EncodedArray {
        /// The encoded buffers
        buffers: Vec<EncodedBuffer>,
        /// The encoding that was used to encode the buffers
        encoding: pb::EncodingType,
        /// The logical length of the encoded array
        num_rows: u32,
    }

    /// Encodes data from Arrow format into some kind of on-disk format
    ///
    /// The encoder is responsible for looking at the incoming data and determining
    /// which encoding is most appropriate.  It then needs to actually encode that
    /// data according to the chosen encoding.
    pub trait ArrayEncoder: std::fmt::Debug + Send + Sync {
        /// Encode data
        ///
        /// This method may receive multiple arrays and should encode them all into
        /// a single encoded array.
        ///
        /// The result should contain the encoded buffers and a description of the
        /// encoding that was chosen.  This can be used to decode the data later.
        fn encode(&self, arrays: Vec<ArrayRef>) -> Result<EncodedArray>;
    }

    // Simple encoder for arrays where every element is null
    //
    // This is a zero-size encoder, no data is written
    #[derive(Debug, Default)]
    struct NullEncoder {}

    impl ArrayEncoder for NullEncoder {
        fn encode(&self, arrays: Vec<ArrayRef>) -> Result<EncodedArray> {
            let num_rows = arrays.iter().map(|arr| arr.len() as u32).sum::<u32>();
            Ok(EncodedArray {
                buffers: vec![],
                encoding: pb::EncodingType {
                    r#type: Some(pb::encoding_type::Type::Constant(pb::ConstantEncoding {
                        value: Vec::new(),
                    })),
                },
                num_rows,
            })
        }
    }

    // Encoder for writing boolean arrays as dense bitmaps
    #[derive(Debug, Default)]
    struct BitmapEncoder {}

    // TODO: Write unit tests to ensure that encoders handle offsets
    // TODO: The top-level "boolean encoder" should check to see if the
    //       values can be compressed efficiently in some way

    impl ArrayEncoder for BitmapEncoder {
        fn encode(&self, arrays: Vec<ArrayRef>) -> Result<EncodedArray> {
            debug_assert!(arrays
                .iter()
                .all(|arr| *arr.data_type() == DataType::Boolean));
            let num_rows: u32 = arrays.iter().map(|arr| arr.len() as u32).sum();
            // Empty pages don't make sense, this should be prevented before we
            // get here
            debug_assert_ne!(num_rows, 0);
            // We can't just write the inner value buffers one after the other because
            // bitmaps can have junk padding at the end (e.g. a boolean array with 12
            // values will be 2 bytes but the last four bits of the second byte are
            // garbage).  So we go ahead and pay the cost of a copy (we could avoid this
            // if we really needed to, at the expense of more complicated code and a slightly
            // larger encoded size but writer cost generally doesn't matter all that much)
            let mut builder = BooleanBufferBuilder::new(num_rows as usize);
            for arr in &arrays {
                let bool_arr = arr.as_boolean();
                builder.append_buffer(bool_arr.values());
            }
            let buffer = builder.finish().into_inner();
            let parts = vec![buffer];
            let buffer = EncodedBuffer {
                is_data: true,
                parts,
            };
            Ok(EncodedArray {
                buffers: vec![buffer],
                num_rows,
                encoding: pb::EncodingType {
                    r#type: Some(pb::encoding_type::Type::Value(pb::ValueEncoding {
                        item_width: 1,
                    })),
                },
            })
        }
    }

    #[derive(Debug, Default)]
    struct ValidityEncoder {}

    impl ArrayEncoder for ValidityEncoder {
        fn encode(&self, arrays: Vec<ArrayRef>) -> Result<EncodedArray> {
            let has_nulls = arrays.iter().any(|arr| arr.null_count() > 0);
            if has_nulls {
                todo!()
            } else {
                let num_rows = arrays.iter().map(|arr| arr.len() as u32).sum::<u32>();
                Ok(EncodedArray {
                    buffers: vec![],
                    encoding: pb::EncodingType {
                        r#type: Some(pb::encoding_type::Type::Constant(pb::ConstantEncoding {
                            value: Vec::new(),
                        })),
                    },
                    num_rows,
                })
            }
        }
    }

    // An encoder to use for non-nullable primitive arrays
    //
    // This creates a single contiguous buffer of values
    //
    // Todo: add support for detecting when better encodings (e.g. run length / FOR)
    //       can be used
    #[derive(Debug)]
    struct ValueEncoder {
        item_width: u32,
    }

    impl ValueEncoder {
        fn try_new(data_type: DataType) -> Result<Self> {
            // For some odd reason, boolean is not primitive
            if !data_type.is_primitive() && data_type != DataType::Boolean {
                Err(Error::InvalidInput { source: format!("attempt to use the primitive encoder for the data type {} which is not primitive", data_type).into(), location: location!() })
            } else {
                let item_width = match data_type {
                    DataType::Boolean => 1,
                    _ => (data_type.primitive_width().unwrap() * 8) as u32,
                };
                Ok(Self { item_width })
            }
        }
    }

    impl ArrayEncoder for ValueEncoder {
        fn encode(&self, arrays: Vec<ArrayRef>) -> Result<EncodedArray> {
            let num_rows: u32 = arrays.iter().map(|arr| arr.len() as u32).sum();
            // Empty pages don't make sense, this should be prevented before we
            // get here
            debug_assert_ne!(num_rows, 0);
            let parts = arrays
                .iter()
                .map(|arr| {
                    let data = arr.to_data();
                    println!(
                        "Encoding {} bytes: {:?}",
                        data.buffers()[0].len(),
                        data.buffers()[0]
                    );
                    data.buffers()[0].clone()
                })
                .collect::<Vec<_>>();
            let buffer = EncodedBuffer {
                is_data: true,
                parts,
            };
            Ok(EncodedArray {
                buffers: vec![buffer],
                num_rows,
                encoding: pb::EncodingType {
                    r#type: Some(pb::encoding_type::Type::Value(pb::ValueEncoding {
                        item_width: self.item_width,
                    })),
                },
            })
        }
    }

    // An encoder to use for (potentially nullable) types
    #[derive(Debug)]
    struct BasicEncoder {
        nulls_encoder: Box<dyn ArrayEncoder>,
        no_nulls_encoder: Box<dyn ArrayEncoder>,
        values_encoder: Box<dyn ArrayEncoder>,
    }

    impl BasicEncoder {
        fn new(
            nulls_encoder: Box<dyn ArrayEncoder>,
            values_encoder: Box<dyn ArrayEncoder>,
        ) -> Self {
            Self {
                nulls_encoder,
                values_encoder,
                // If there are no nulls we always encode the validity buffer
                // with NullEncoder which creates a zero-size constant encoding
                no_nulls_encoder: Box::new(NullEncoder {}),
            }
        }
    }

    // TODO: If we detect nulls we immediately fall back to writing a bitmap here.  We should try and
    //       see if we can use a sentinel encoding first.

    impl ArrayEncoder for BasicEncoder {
        fn encode(&self, arrays: Vec<ArrayRef>) -> Result<EncodedArray> {
            let has_nulls = arrays.iter().any(|arr| arr.nulls().is_some());
            let nulls_encoding = if has_nulls {
                // Convert the null bitmaps into boolean arrays so we can use the same encoders for both
                let null_arrays = arrays
                    .iter()
                    .map(|arr| {
                        if let Some(null_buffer) = arr.nulls() {
                            Arc::new(BooleanArray::new(null_buffer.inner().clone(), None))
                                as ArrayRef
                        } else {
                            // If some parts have nulls and others don't then we need to encode all of them
                            // and so we create an all-valid part
                            Arc::new(BooleanArray::new(BooleanBuffer::new_set(arr.len()), None))
                                as ArrayRef
                        }
                    })
                    .collect::<Vec<_>>();
                self.nulls_encoder.encode(null_arrays)?
            } else {
                self.no_nulls_encoder.encode(arrays.clone())?
            };
            let values_encoding = self.values_encoder.encode(arrays)?;

            debug_assert_eq!(nulls_encoding.num_rows, values_encoding.num_rows);
            let num_rows = nulls_encoding.num_rows;

            let mut all_buffers = nulls_encoding.buffers;
            all_buffers.extend(values_encoding.buffers);

            println!(
                "There are {} buffers with lengths [{}]",
                all_buffers.len(),
                all_buffers
                    .iter()
                    .map(|val| val
                        .parts
                        .iter()
                        .map(|part| part.len())
                        .sum::<usize>()
                        .to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            let encoding = pb::EncodingType {
                r#type: Some(pb::encoding_type::Type::Masked(Box::new(
                    pb::MaskedEncoding {
                        value_encoding: Some(Box::new(values_encoding.encoding)),
                        validity_encoding: Some(Box::new(nulls_encoding.encoding)),
                    },
                ))),
            };

            Ok(EncodedArray {
                buffers: all_buffers,
                encoding,
                num_rows,
            })
        }
    }

    pub fn create_encoder(data_type: DataType) -> Box<dyn ArrayEncoder> {
        match data_type {
            DataType::Null => Box::new(NullEncoder::default()),
            _ => {
                let validity_encoder = Box::new(BitmapEncoder {});
                let values_encoder = Box::new(ValueEncoder::try_new(data_type).unwrap());
                Box::new(BasicEncoder::new(validity_encoder, values_encoder))
            }
        }
    }

    #[derive(Debug)]
    pub struct BufferedArrayEncoder {
        cache_bytes: u64,
        buffered_arrays: Vec<ArrayRef>,
        current_bytes: u64,
        encoder: Arc<dyn ArrayEncoder>,
        column_index: u32,
    }

    impl BufferedArrayEncoder {
        fn new(cache_bytes: u64, data_type: DataType, column_index: u32) -> Self {
            Self {
                cache_bytes,
                buffered_arrays: Vec::with_capacity(8),
                current_bytes: 0,
                encoder: create_encoder(data_type).into(),
                column_index,
            }
        }

        // Creates an encode task, consuming all buffered data
        fn do_flush(
            &mut self,
        ) -> impl std::future::Future<Output = Result<(u32, EncodedArray)>> + 'static {
            let mut arrays = Vec::with_capacity(8);
            std::mem::swap(&mut arrays, &mut self.buffered_arrays);
            self.current_bytes = 0;
            let encoder = self.encoder.clone();
            let column_index = self.column_index;
            spawn_cpu(move || {
                encoder
                    .encode(arrays)
                    .map(|encoded_array| (column_index, encoded_array))
            })
        }

        // Buffers data, if there is enough to write a page then we create an encode task
        fn maybe_encode(
            &mut self,
            array: ArrayRef,
        ) -> Option<impl std::future::Future<Output = Result<(u32, EncodedArray)>> + 'static>
        {
            self.current_bytes += array.get_array_memory_size() as u64;
            self.buffered_arrays.push(array);
            if self.current_bytes > self.cache_bytes {
                Some(self.do_flush())
            } else {
                None
            }
        }

        // If there is any data left in the buffer then create an encode task from it
        fn flush(
            &mut self,
        ) -> Option<impl std::future::Future<Output = Result<(u32, EncodedArray)>> + 'static>
        {
            if self.current_bytes > 0 {
                Some(self.do_flush())
            } else {
                None
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct FileWriterOptions {
        /// The field ids to collect statistics for.
        ///
        /// If None, will collect for all fields in the schema (that support stats).
        /// If an empty vector, will not collect any statistics.
        pub collect_stats_for_fields: Option<Vec<i32>>,
        /// How many bytes to use for buffering column data
        ///
        /// When data comes in small batches the writer will buffer column data so that
        /// larger pages can be created.  This value will be divided evenly across all of the
        /// columns.  Generally you want this to be at least large enough to match your
        /// filesystem's ideal read size per column.
        ///
        /// In some cases you might want this value to be even larger if you have highly
        /// compressible data.  However, if this is too large, then the writer could require
        /// a lot of memory and write performance may suffer if the CPU-expensive encoding
        /// falls behind and can't be interleaved with the I/O expensive flushing.
        ///
        /// The default will use 8MiB per column which should be reasonable for most cases.
        pub data_cache_bytes: Option<u64>,
        /// How many bytes to use for buffering data to write
        ///
        /// After data has been encoded it is placed into an I/O cache while we wait for it
        /// to be written to disk.  If data is arriving quickly and your I/O is very jittery
        /// then you might want to increase this value.  However, the default of 128MiB should
        /// be sufficient for most cases.
        ///
        /// If data is arriving slowly or write speed is not important you can turn this down
        /// all the way to zero to eliminate I/O caching entirely and free up some RAM.  In this
        /// mode each write will block until its data has been flushed.  This prevents any
        /// interleaving of encoding and writing.
        pub io_cache_bytes: Option<u64>,
    }

    pub struct FileWriter {
        writer: ObjectWriter,
        schema: Schema,
        column_writers: Vec<BufferedArrayEncoder>,
        column_metadata: Vec<pb::ColumnMetadata>,
        // TODO
        // stats_collector: Option<statistics::StatisticsCollector>,
        rows_written: u32,
    }

    impl FileWriter {
        /// Create a new FileWriter
        ///
        /// The FileWriter will take responsibility for closing the object_writer when Self::finish
        /// is called.
        pub fn new(
            object_writer: ObjectWriter,
            schema: Schema,
            options: FileWriterOptions,
        ) -> Self {
            let cache_bytes_per_column = if let Some(data_cache_bytes) = options.data_cache_bytes {
                data_cache_bytes / schema.num_top_level_fields() as u64
            } else {
                8 * 1024 * 1024
            };

            let column_writers = schema
                .fields
                .iter()
                .enumerate()
                .map(|(idx, field)| {
                    BufferedArrayEncoder::new(cache_bytes_per_column, field.data_type(), idx as u32)
                })
                .collect();
            let column_metadata = vec![pb::ColumnMetadata::default(); schema.fields.len()];

            Self {
                writer: object_writer,
                schema,
                column_writers,
                column_metadata,
                rows_written: 0,
            }
        }

        async fn write_page(
            &mut self,
            encoded_array: EncodedArray,
            column_index: u32,
        ) -> Result<()> {
            let mut buffer_offsets = Vec::with_capacity(encoded_array.buffers.len());
            let mut buffer_sizes = Vec::with_capacity(encoded_array.buffers.len());
            for buffer in encoded_array.buffers {
                buffer_offsets.push(self.writer.tell().await? as u64);
                buffer_sizes.push(
                    buffer
                        .parts
                        .iter()
                        .map(|part| part.len() as u64)
                        .sum::<u64>(),
                );
                // Note: could potentially use write_vectored here but there is no
                // write_vectored_all and object_store doesn't support it anyways and
                // buffers won't normally be in *too* many parts so its unlikely to
                // have much benefit in most cases.
                for part in &buffer.parts {
                    self.writer.write_all(part).await?;
                }
            }
            let page = pb::column_metadata::Page {
                buffer_offsets,
                buffer_sizes,
                encoding: Some(encoded_array.encoding),
                length: encoded_array.num_rows,
            };
            self.column_metadata[column_index as usize].pages.push(page);
            Ok(())
        }

        async fn write_pages(
            &mut self,
            mut encoding_tasks: FuturesUnordered<
                impl std::future::Future<Output = Result<(u32, EncodedArray)>>,
            >,
        ) -> Result<()> {
            // As soon as an encoding task is done we write it.  There is no parallelism
            // needed here because "writing" is really just submitting the buffer to the
            // underlying write scheduler (either the OS or object_store's scheduler for
            // cloud writes).  The only time we might truly await on write_page is if the
            // scheduler's write queue is full.
            //
            // Also, there is no point in trying to make write_page parallel anyways
            // because we wouldn't want buffers getting mixed up across pages.
            while let Some(encoding_task) = encoding_tasks.next().await {
                let (column_index, encoded_array) = encoding_task?;
                self.write_page(encoded_array, column_index).await?;
            }
            Ok(())
        }

        /// Schedule a batch of data to be written to the file
        ///
        /// Note: the future returned by this method may complete before the data has been fully
        /// flushed to the file (some data may be in the data cache or the I/O cache)
        pub async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
            let num_rows = batch.num_rows() as u64;
            if num_rows == 0 {
                return Ok(());
            }
            if num_rows > u32::MAX as u64 {
                return Err(Error::InvalidInput {
                    source: "cannot write Lance files with more than 2^32 rows".into(),
                    location: location!(),
                });
            }
            self.rows_written = match self.rows_written.checked_add(batch.num_rows() as u32) {
                Some(rows_written) => rows_written,
                None => {
                    return Err(Error::InvalidInput { source: format!("cannot write batch with {} rows because {} rows have already been written and Lance files cannot contain more than 2^32 rows", num_rows, self.rows_written).into(), location: location!() });
                }
            };
            // First we push each array into its column writer.  This may or may not generate enough
            // data to trigger an encoding task.  We collect any encoding tasks into a queue.
            let encoding_tasks = self
                .schema
                .fields
                .iter()
                .zip(self.column_writers.iter_mut())
                .map(|(field, column_writer)| {
                    let array = batch
                        .column_by_name(&field.name)
                        .ok_or(Error::InvalidInput {
                            source: format!(
                                "Cannot write batch.  The batch was missing the column `{}`",
                                field.name
                            )
                            .into(),
                            location: location!(),
                        })?;
                    Ok(column_writer.maybe_encode(array.clone()))
                })
                .filter_map(|item| item.transpose())
                .collect::<Result<FuturesUnordered<_>>>()?;

            self.write_pages(encoding_tasks).await?;

            Ok(())
        }

        async fn write_column_metadata(
            &mut self,
            metadata: pb::ColumnMetadata,
        ) -> Result<(u64, u64)> {
            let metadata_bytes = metadata.encode_to_vec();
            let position = self.writer.tell().await? as u64;
            let len = metadata_bytes.len() as u64;
            self.writer.write_all(&metadata_bytes).await?;
            Ok((position, len))
        }

        async fn write_column_metadatas(&mut self) -> Result<Vec<(u64, u64)>> {
            let mut metadatas = Vec::new();
            std::mem::swap(&mut self.column_metadata, &mut metadatas);
            let mut metadata_positions = Vec::with_capacity(metadatas.len());
            for metadata in metadatas {
                metadata_positions.push(self.write_column_metadata(metadata).await?);
            }
            Ok(metadata_positions)
        }

        fn make_file_descriptor(&self) -> FileDescriptor {
            let fields_with_meta = FieldsWithMeta::from(&self.schema);
            FileDescriptor {
                schema: Some(pb::Schema {
                    fields: fields_with_meta.fields.0,
                    metadata: fields_with_meta.metadata,
                }),
                length: self.rows_written,
            }
        }

        async fn write_file_descriptor(&mut self) -> Result<u64> {
            let file_descriptor = self.make_file_descriptor();
            let file_descriptor_bytes = file_descriptor.encode_to_vec();
            let file_descriptor_position = self.writer.tell().await? as u64;
            self.writer.write_all(&file_descriptor_bytes).await?;
            Ok(file_descriptor_position)
        }

        /// Finishes writing the file
        ///
        /// This method will wait until all data has been flushed to the file.  Then it
        /// will write the file metadata and the footer.  It will not return until all
        /// data has been flushed and the file has been closed.
        pub async fn finish(&mut self) -> Result<()> {
            // 1. flush any remaining data and write out those pages
            let encoding_tasks = self
                .column_writers
                .iter_mut()
                .map(|writer| writer.flush())
                .flatten()
                .collect::<FuturesUnordered<_>>();
            self.write_pages(encoding_tasks).await?;

            // 2. write the column metadatas
            let column_metadata_start = self.writer.tell().await? as u64;
            let metadata_positions = self.write_column_metadatas().await?;

            // 3. write the column metadata position table
            for (meta_pos, meta_len) in metadata_positions {
                self.writer.write_u64_le(meta_pos).await?;
                self.writer.write_u64_le(meta_len).await?;
            }

            // 4. write the file descriptor
            let file_descriptor_position = self.write_file_descriptor().await?;

            // 5. write the file metadata
            let metadata_bytes = pb::Metadata {
                batch_offsets: Vec::new(),
                manifest_position: file_descriptor_position,
                num_columns: self.schema.num_top_level_fields(),
                page_table_position: 0,
                statistics: None,
                column_metadata_start,
            }
            .encode_to_vec();
            let metadata_position = self.writer.tell().await? as u64;
            self.writer.write_all(&metadata_bytes).await?;

            // 6. write the footer
            self.writer
                .write_magics(
                    metadata_position as usize,
                    MAJOR_VERSION,
                    MINOR_VERSION,
                    MAGIC,
                )
                .await?;

            // 7. close the writer
            self.writer.shutdown().await?;
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {

        use crate::reader::v2::FileReader;

        use super::*;

        use arrow_array::{Float32Array, Int64Array, NullArray};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_io::{object_store::ObjectStore, ReadBatchParams};
        use object_store::path::Path;

        #[tokio::test]
        async fn test_write_v2() {
            let arrow_schema = ArrowSchema::new(vec![
                ArrowField::new("null", DataType::Null, true),
                // ArrowField::new("bool", DataType::Boolean, true),
                ArrowField::new("i", DataType::Int64, true),
                ArrowField::new("f", DataType::Float32, false),
                // ArrowField::new("b", DataType::Utf8, true),
                // ArrowField::new("decimal128", DataType::Decimal128(7, 3), false),
                // ArrowField::new("decimal256", DataType::Decimal256(7, 3), false),
                // ArrowField::new("duration_sec", DataType::Duration(TimeUnit::Second), false),
                // ArrowField::new(
                //     "duration_msec",
                //     DataType::Duration(TimeUnit::Millisecond),
                //     false,
                // ),
                // ArrowField::new(
                //     "duration_usec",
                //     DataType::Duration(TimeUnit::Microsecond),
                //     false,
                // ),
                // ArrowField::new(
                //     "duration_nsec",
                //     DataType::Duration(TimeUnit::Nanosecond),
                //     false,
                // ),
                // ArrowField::new(
                //     "d",
                //     DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                //     true,
                // ),
                // ArrowField::new(
                //     "fixed_size_list",
                //     DataType::FixedSizeList(
                //         Arc::new(ArrowField::new("item", DataType::Float32, true)),
                //         16,
                //     ),
                //     true,
                // ),
                // ArrowField::new("fixed_size_binary", DataType::FixedSizeBinary(8), true),
                // ArrowField::new(
                //     "l",
                //     DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                //     true,
                // ),
                // ArrowField::new(
                //     "large_l",
                //     DataType::LargeList(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                //     true,
                // ),
                // ArrowField::new(
                //     "l_dict",
                //     DataType::List(Arc::new(ArrowField::new(
                //         "item",
                //         DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                //         true,
                //     ))),
                //     true,
                // ),
                // ArrowField::new(
                //     "large_l_dict",
                //     DataType::LargeList(Arc::new(ArrowField::new(
                //         "item",
                //         DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                //         true,
                //     ))),
                //     true,
                // ),
                // ArrowField::new(
                //     "s",
                //     DataType::Struct(ArrowFields::from(vec![
                //         ArrowField::new("si", DataType::Int64, true),
                //         ArrowField::new("sb", DataType::Utf8, true),
                //     ])),
                //     true,
                // ),
            ]);
            let mut schema = Schema::try_from(&arrow_schema).unwrap();

            // let dict_vec = (0..100).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
            // let dict_arr: DictionaryArray<UInt32Type> = dict_vec.into_iter().collect();

            // let fixed_size_list_arr = FixedSizeListArray::try_new_from_values(
            //     Float32Array::from_iter((0..1600).map(|n| n as f32).collect::<Vec<_>>()),
            //     16,
            // )
            // .unwrap();

            // let binary_data: [u8; 800] = [123; 800];
            // let fixed_size_binary_arr =
            //     FixedSizeBinaryArray::try_new_from_values(&UInt8Array::from_iter(binary_data), 8)
            //         .unwrap();

            // let list_offsets = (0..202).step_by(2).collect();
            // let list_values =
            //     StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
            // let list_arr: arrow_array::GenericListArray<i32> =
            //     try_new_generic_list_array(list_values, &list_offsets).unwrap();

            // let large_list_offsets: Int64Array = (0..202).step_by(2).collect();
            // let large_list_values =
            //     StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
            // let large_list_arr: arrow_array::GenericListArray<i64> =
            //     try_new_generic_list_array(large_list_values, &large_list_offsets).unwrap();

            // let list_dict_offsets = (0..202).step_by(2).collect();
            // let list_dict_vec = (0..200).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
            // let list_dict_arr: DictionaryArray<UInt32Type> = list_dict_vec.into_iter().collect();
            // let list_dict_arr: arrow_array::GenericListArray<i32> =
            //     try_new_generic_list_array(list_dict_arr, &list_dict_offsets).unwrap();

            // let large_list_dict_offsets: Int64Array = (0..202).step_by(2).collect();
            // let large_list_dict_vec = (0..200).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
            // let large_list_dict_arr: DictionaryArray<UInt32Type> =
            //     large_list_dict_vec.into_iter().collect();
            // let large_list_dict_arr: arrow_array::GenericListArray<i64> =
            //     try_new_generic_list_array(large_list_dict_arr, &large_list_dict_offsets).unwrap();

            let columns: Vec<ArrayRef> = vec![
                Arc::new(NullArray::new(100)),
                // Arc::new(BooleanArray::from_iter(
                //     (0..100).map(|f| Some(f % 3 == 0)).collect::<Vec<_>>(),
                // )),
                Arc::new(Int64Array::from_iter((0..100).collect::<Vec<_>>())),
                Arc::new(Float32Array::from_iter(
                    (0..100).map(|n| n as f32).collect::<Vec<_>>(),
                )),
                // Arc::new(StringArray::from(
                //     (0..100).map(|n| n.to_string()).collect::<Vec<_>>(),
                // )),
                // Arc::new(
                //     Decimal128Array::from_iter_values(0..100)
                //         .with_precision_and_scale(7, 3)
                //         .unwrap(),
                // ),
                // Arc::new(
                //     Decimal256Array::from_iter_values((0..100).map(|v| i256::from_i128(v as i128)))
                //         .with_precision_and_scale(7, 3)
                //         .unwrap(),
                // ),
                // Arc::new(DurationSecondArray::from_iter_values(0..100)),
                // Arc::new(DurationMillisecondArray::from_iter_values(0..100)),
                // Arc::new(DurationMicrosecondArray::from_iter_values(0..100)),
                // Arc::new(DurationNanosecondArray::from_iter_values(0..100)),
                // Arc::new(dict_arr),
                // Arc::new(fixed_size_list_arr),
                // Arc::new(fixed_size_binary_arr),
                // Arc::new(list_arr),
                // Arc::new(large_list_arr),
                // Arc::new(list_dict_arr),
                // Arc::new(large_list_dict_arr),
                // Arc::new(StructArray::from(vec![
                //     (
                //         Arc::new(ArrowField::new("si", DataType::Int64, true)),
                //         Arc::new(Int64Array::from_iter((100..200).collect::<Vec<_>>())) as ArrayRef,
                //     ),
                //     (
                //         Arc::new(ArrowField::new("sb", DataType::Utf8, true)),
                //         Arc::new(StringArray::from(
                //             (0..100).map(|n| n.to_string()).collect::<Vec<_>>(),
                //         )) as ArrayRef,
                //     ),
                // ])),
            ];
            let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();
            schema.set_dictionary(&batch).unwrap();

            let store = ObjectStore::memory();
            let path = Path::from("foo");
            let obj_writer = store.create(&path).await.unwrap();

            let mut file_writer =
                FileWriter::new(obj_writer, schema.clone(), FileWriterOptions::default());
            file_writer.write_batch(&batch).await.unwrap();
            file_writer.finish().await.unwrap();

            let obj_reader = store.open(&path).await.unwrap();
            let file_reader = FileReader::try_open(obj_reader.into(), schema)
                .await
                .unwrap();

            let mut data_stream = file_reader.read_stream(ReadBatchParams::RangeFull, 1024);

            let read_batch = data_stream.next().await.unwrap().unwrap();
            assert_eq!(batch, read_batch);

            let end = data_stream.next().await;
            assert!(end.is_none());
        }

        #[tokio::test]
        async fn test_write_with_nulls() {
            let arrow_schema = ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int64, true),
                ArrowField::new("f", DataType::Float32, true),
            ]);
            let mut schema = Schema::try_from(&arrow_schema).unwrap();

            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from(vec![Some(0), None, Some(1)])),
                Arc::new(Float32Array::from(vec![Some(0.0), None, Some(1.0)])),
            ];
            let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();
            schema.set_dictionary(&batch).unwrap();

            let store = ObjectStore::memory();
            let path = Path::from("foo");
            let obj_writer = store.create(&path).await.unwrap();

            let mut file_writer =
                FileWriter::new(obj_writer, schema.clone(), FileWriterOptions::default());
            file_writer.write_batch(&batch).await.unwrap();
            file_writer.finish().await.unwrap();

            let obj_reader = store.open(&path).await.unwrap();
            let file_reader = FileReader::try_open(obj_reader.into(), schema)
                .await
                .unwrap();

            let mut data_stream = file_reader.read_stream(ReadBatchParams::RangeFull, 1024);

            let read_batch = data_stream.next().await.unwrap().unwrap();
            assert_eq!(batch, read_batch);

            let end = data_stream.next().await;
            assert!(end.is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::reader::FileReader;

    use super::*;

    use std::sync::Arc;

    use arrow_array::{
        types::UInt32Type, BooleanArray, Decimal128Array, Decimal256Array, DictionaryArray,
        DurationMicrosecondArray, DurationMillisecondArray, DurationNanosecondArray,
        DurationSecondArray, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, Int32Array,
        Int64Array, ListArray, NullArray, StringArray, TimestampMicrosecondArray,
        TimestampSecondArray, UInt8Array,
    };
    use arrow_buffer::i256;
    use arrow_schema::{
        DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema, TimeUnit,
    };
    use arrow_select::concat::concat_batches;
    use object_store::path::Path;

    #[tokio::test]
    async fn test_write_file() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("null", DataType::Null, true),
            ArrowField::new("bool", DataType::Boolean, true),
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
            ArrowField::new("b", DataType::Utf8, true),
            ArrowField::new("decimal128", DataType::Decimal128(7, 3), false),
            ArrowField::new("decimal256", DataType::Decimal256(7, 3), false),
            ArrowField::new("duration_sec", DataType::Duration(TimeUnit::Second), false),
            ArrowField::new(
                "duration_msec",
                DataType::Duration(TimeUnit::Millisecond),
                false,
            ),
            ArrowField::new(
                "duration_usec",
                DataType::Duration(TimeUnit::Microsecond),
                false,
            ),
            ArrowField::new(
                "duration_nsec",
                DataType::Duration(TimeUnit::Nanosecond),
                false,
            ),
            ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            ),
            ArrowField::new(
                "fixed_size_list",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    16,
                ),
                true,
            ),
            ArrowField::new("fixed_size_binary", DataType::FixedSizeBinary(8), true),
            ArrowField::new(
                "l",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                true,
            ),
            ArrowField::new(
                "large_l",
                DataType::LargeList(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                true,
            ),
            ArrowField::new(
                "l_dict",
                DataType::List(Arc::new(ArrowField::new(
                    "item",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ))),
                true,
            ),
            ArrowField::new(
                "large_l_dict",
                DataType::LargeList(Arc::new(ArrowField::new(
                    "item",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ))),
                true,
            ),
            ArrowField::new(
                "s",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("si", DataType::Int64, true),
                    ArrowField::new("sb", DataType::Utf8, true),
                ])),
                true,
            ),
        ]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();

        let dict_vec = (0..100).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let dict_arr: DictionaryArray<UInt32Type> = dict_vec.into_iter().collect();

        let fixed_size_list_arr = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter((0..1600).map(|n| n as f32).collect::<Vec<_>>()),
            16,
        )
        .unwrap();

        let binary_data: [u8; 800] = [123; 800];
        let fixed_size_binary_arr =
            FixedSizeBinaryArray::try_new_from_values(&UInt8Array::from_iter(binary_data), 8)
                .unwrap();

        let list_offsets = (0..202).step_by(2).collect();
        let list_values =
            StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
        let list_arr: arrow_array::GenericListArray<i32> =
            try_new_generic_list_array(list_values, &list_offsets).unwrap();

        let large_list_offsets: Int64Array = (0..202).step_by(2).collect();
        let large_list_values =
            StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
        let large_list_arr: arrow_array::GenericListArray<i64> =
            try_new_generic_list_array(large_list_values, &large_list_offsets).unwrap();

        let list_dict_offsets = (0..202).step_by(2).collect();
        let list_dict_vec = (0..200).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let list_dict_arr: DictionaryArray<UInt32Type> = list_dict_vec.into_iter().collect();
        let list_dict_arr: arrow_array::GenericListArray<i32> =
            try_new_generic_list_array(list_dict_arr, &list_dict_offsets).unwrap();

        let large_list_dict_offsets: Int64Array = (0..202).step_by(2).collect();
        let large_list_dict_vec = (0..200).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let large_list_dict_arr: DictionaryArray<UInt32Type> =
            large_list_dict_vec.into_iter().collect();
        let large_list_dict_arr: arrow_array::GenericListArray<i64> =
            try_new_generic_list_array(large_list_dict_arr, &large_list_dict_offsets).unwrap();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(NullArray::new(100)),
            Arc::new(BooleanArray::from_iter(
                (0..100).map(|f| Some(f % 3 == 0)).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from_iter((0..100).collect::<Vec<_>>())),
            Arc::new(Float32Array::from_iter(
                (0..100).map(|n| n as f32).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..100).map(|n| n.to_string()).collect::<Vec<_>>(),
            )),
            Arc::new(
                Decimal128Array::from_iter_values(0..100)
                    .with_precision_and_scale(7, 3)
                    .unwrap(),
            ),
            Arc::new(
                Decimal256Array::from_iter_values((0..100).map(|v| i256::from_i128(v as i128)))
                    .with_precision_and_scale(7, 3)
                    .unwrap(),
            ),
            Arc::new(DurationSecondArray::from_iter_values(0..100)),
            Arc::new(DurationMillisecondArray::from_iter_values(0..100)),
            Arc::new(DurationMicrosecondArray::from_iter_values(0..100)),
            Arc::new(DurationNanosecondArray::from_iter_values(0..100)),
            Arc::new(dict_arr),
            Arc::new(fixed_size_list_arr),
            Arc::new(fixed_size_binary_arr),
            Arc::new(list_arr),
            Arc::new(large_list_arr),
            Arc::new(list_dict_arr),
            Arc::new(large_list_dict_arr),
            Arc::new(StructArray::from(vec![
                (
                    Arc::new(ArrowField::new("si", DataType::Int64, true)),
                    Arc::new(Int64Array::from_iter((100..200).collect::<Vec<_>>())) as ArrayRef,
                ),
                (
                    Arc::new(ArrowField::new("sb", DataType::Utf8, true)),
                    Arc::new(StringArray::from(
                        (0..100).map(|n| n.to_string()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ),
            ])),
        ];
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();
        schema.set_dictionary(&batch).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let actual = reader
            .read_batch(0, .., reader.schema(), None)
            .await
            .unwrap();
        assert_eq!(actual, batch);
    }

    #[tokio::test]
    async fn test_dictionary_first_element_file() {
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "d",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();

        let dict_vec = (0..100).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let dict_arr: DictionaryArray<UInt32Type> = dict_vec.into_iter().collect();

        let columns: Vec<ArrayRef> = vec![Arc::new(dict_arr)];
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();
        schema.set_dictionary(&batch).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let actual = reader
            .read_batch(0, .., reader.schema(), None)
            .await
            .unwrap();
        assert_eq!(actual, batch);
    }

    #[tokio::test]
    async fn test_write_temporal_types() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(
                "ts_notz",
                DataType::Timestamp(TimeUnit::Second, None),
                false,
            ),
            ArrowField::new(
                "ts_tz",
                DataType::Timestamp(TimeUnit::Microsecond, Some("America/Los_Angeles".into())),
                false,
            ),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(TimestampSecondArray::from(vec![11111111, 22222222])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![3333333, 4444444])
                    .with_timezone("America/Los_Angeles"),
            ),
        ];
        let batch = RecordBatch::try_new(arrow_schema.clone(), columns).unwrap();

        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let actual = reader
            .read_batch(0, .., reader.schema(), None)
            .await
            .unwrap();
        assert_eq!(actual, batch);
    }

    #[tokio::test]
    async fn test_collect_stats() {
        // Validate:
        // Only collects stats for requested columns
        // Can collect stats in nested structs
        // Won't collect stats for list columns (for now)

        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("i2", DataType::Int64, true),
            ArrowField::new(
                "l",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                true,
            ),
            ArrowField::new(
                "s",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("si", DataType::Int64, true),
                    ArrowField::new("sb", DataType::Utf8, true),
                ])),
                true,
            ),
        ]);

        let schema = Schema::try_from(&arrow_schema).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");

        let options = FileWriterOptions {
            collect_stats_for_fields: Some(vec![0, 1, 5, 6]),
        };
        let mut file_writer =
            FileWriter::<NotSelfDescribing>::try_new(&store, &path, schema.clone(), &options)
                .await
                .unwrap();

        let batch1 = RecordBatch::try_new(
            Arc::new(arrow_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![4, 5, 6])),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1i32), Some(2), Some(3)]),
                    Some(vec![Some(4), Some(5)]),
                    Some(vec![]),
                ])),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("si", DataType::Int64, true)),
                        Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("sb", DataType::Utf8, true)),
                        Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
                    ),
                ])),
            ],
        )
        .unwrap();
        file_writer.write(&[batch1]).await.unwrap();

        let batch2 = RecordBatch::try_new(
            Arc::new(arrow_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![5, 6])),
                Arc::new(Int64Array::from(vec![10, 11])),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1i32), Some(2), Some(3)]),
                    Some(vec![]),
                ])),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("si", DataType::Int64, true)),
                        Arc::new(Int64Array::from(vec![4, 5])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("sb", DataType::Utf8, true)),
                        Arc::new(StringArray::from(vec!["d", "e"])) as ArrayRef,
                    ),
                ])),
            ],
        )
        .unwrap();
        file_writer.write(&[batch2]).await.unwrap();

        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();

        let read_stats = reader.read_page_stats(&[0, 1, 5, 6]).await.unwrap();
        assert!(read_stats.is_some());
        let read_stats = read_stats.unwrap();

        let expected_stats_schema = stats_schema([
            (0, DataType::Int64),
            (1, DataType::Int64),
            (5, DataType::Int64),
            (6, DataType::Utf8),
        ]);

        assert_eq!(read_stats.schema().as_ref(), &expected_stats_schema);

        let expected_stats = stats_batch(&[
            Stats {
                field_id: 0,
                null_counts: vec![0, 0],
                min_values: Arc::new(Int64Array::from(vec![1, 5])),
                max_values: Arc::new(Int64Array::from(vec![3, 6])),
            },
            Stats {
                field_id: 1,
                null_counts: vec![0, 0],
                min_values: Arc::new(Int64Array::from(vec![4, 10])),
                max_values: Arc::new(Int64Array::from(vec![6, 11])),
            },
            Stats {
                field_id: 5,
                null_counts: vec![0, 0],
                min_values: Arc::new(Int64Array::from(vec![1, 4])),
                max_values: Arc::new(Int64Array::from(vec![3, 5])),
            },
            // FIXME: these max values shouldn't be incremented
            // https://github.com/lancedb/lance/issues/1517
            Stats {
                field_id: 6,
                null_counts: vec![0, 0],
                min_values: Arc::new(StringArray::from(vec!["a", "d"])),
                max_values: Arc::new(StringArray::from(vec!["c", "e"])),
            },
        ]);

        assert_eq!(read_stats, expected_stats);
    }

    fn stats_schema(data_fields: impl IntoIterator<Item = (i32, DataType)>) -> ArrowSchema {
        let fields = data_fields
            .into_iter()
            .map(|(field_id, data_type)| {
                Arc::new(ArrowField::new(
                    format!("{}", field_id),
                    DataType::Struct(
                        vec![
                            Arc::new(ArrowField::new("null_count", DataType::Int64, false)),
                            Arc::new(ArrowField::new("min_value", data_type.clone(), true)),
                            Arc::new(ArrowField::new("max_value", data_type, true)),
                        ]
                        .into(),
                    ),
                    false,
                ))
            })
            .collect::<Vec<_>>();
        ArrowSchema::new(fields)
    }

    struct Stats {
        field_id: i32,
        null_counts: Vec<i64>,
        min_values: ArrayRef,
        max_values: ArrayRef,
    }

    fn stats_batch(stats: &[Stats]) -> RecordBatch {
        let schema = stats_schema(
            stats
                .iter()
                .map(|s| (s.field_id, s.min_values.data_type().clone())),
        );

        let columns = stats
            .iter()
            .map(|s| {
                let data_type = s.min_values.data_type().clone();
                let fields = vec![
                    Arc::new(ArrowField::new("null_count", DataType::Int64, false)),
                    Arc::new(ArrowField::new("min_value", data_type.clone(), true)),
                    Arc::new(ArrowField::new("max_value", data_type, true)),
                ];
                let arrays = vec![
                    Arc::new(Int64Array::from(s.null_counts.clone())),
                    s.min_values.clone(),
                    s.max_values.clone(),
                ];
                Arc::new(StructArray::new(fields.into(), arrays, None)) as ArrayRef
            })
            .collect();

        RecordBatch::try_new(Arc::new(schema), columns).unwrap()
    }

    async fn read_file_as_one_batch(
        object_store: &ObjectStore,
        path: &Path,
        schema: Schema,
    ) -> RecordBatch {
        let reader = FileReader::try_new(object_store, path, schema)
            .await
            .unwrap();
        let mut batches = vec![];
        for i in 0..reader.num_batches() {
            batches.push(
                reader
                    .read_batch(i as i32, .., reader.schema(), None)
                    .await
                    .unwrap(),
            );
        }
        let arrow_schema = Arc::new(reader.schema().into());
        concat_batches(&arrow_schema, &batches).unwrap()
    }

    /// Test encoding arrays that share the same underneath buffer.
    #[tokio::test]
    async fn test_encode_slice() {
        let store = ObjectStore::memory();
        let path = Path::from("/shared_slice");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();

        let array = Int32Array::from_iter_values(0..1000);

        for i in (0..1000).step_by(4) {
            let data = array.slice(i, 4);
            file_writer
                .write(&[RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(data)]).unwrap()])
                .await
                .unwrap();
        }
        file_writer.finish().await.unwrap();
        assert!(store.size(&path).await.unwrap() < 2 * 8 * 1000);

        let batch = read_file_as_one_batch(&store, &path, schema).await;
        assert_eq!(batch.column_by_name("i").unwrap().as_ref(), &array);
    }
}
