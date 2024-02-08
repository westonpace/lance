use std::{ops::Range, sync::Arc};

use arrow_array::{
    new_null_array,
    types::{
        ArrowPrimitiveType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, IntervalDayTimeType, IntervalMonthDayNanoType, IntervalYearMonthType,
        Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    ArrayRef, BooleanArray, PrimitiveArray, RecordBatch,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, IntervalUnit, Schema, TimeUnit};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, Stream, StreamExt, TryFutureExt};
use snafu::{location, Location};

use lance_core::{Error, Result};
use tokio::sync::mpsc;

use crate::io::BatchRequest;

pub trait DataDecoder: Send + Sync + std::fmt::Debug {
    fn update_capacity(
        &self,
        data: &[Vec<Bytes>],
        rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
    );
    fn drain(
        &self,
        // TODO: Coalesce on read and change this to &[Bytes]
        data: &[Vec<Bytes>],
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [BytesMut],
    );
}

#[derive(Debug)]
pub struct Scheduled {
    pub batch_requests: Vec<BatchRequest>,
    pub data_decoder: Box<dyn DataDecoder>,
}

impl Scheduled {
    pub fn new(batch_requests: Vec<BatchRequest>, data_decoder: Box<dyn DataDecoder>) -> Self {
        Self {
            batch_requests,
            data_decoder,
        }
    }
}

pub trait PageDecoder: Send + Sync {
    fn schedule_range(&self, range: Range<u32>) -> Scheduled;
    fn schedule_take(&self, indices: PrimitiveArray<UInt32Type>) -> Scheduled;
    fn num_buffers(&self) -> u32;
}

pub trait FieldDecoder: Send + Sync {
    fn finalize(&self, buffers: Vec<BytesMut>, num_rows: u32) -> Result<ArrayRef>;
    fn dest_buffers_per_col(&self) -> Vec<u32>;
    fn total_num_dest_buffers(&self) -> u32;
}

struct PrimitiveFieldDecoder {
    data_type: DataType,
}

impl PrimitiveFieldDecoder {
    fn new_primitive_array<T: ArrowPrimitiveType>(
        buffers: Vec<BytesMut>,
        num_rows: u32,
    ) -> ArrayRef {
        let mut buffer_iter = buffers.into_iter();
        let null_buffer = buffer_iter.next().unwrap();
        let null_buffer = if null_buffer.is_empty() {
            None
        } else {
            let null_buffer = null_buffer.freeze().into();
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from_bytes(null_buffer),
                0,
                num_rows as usize,
            )))
        };

        let data_buffer = buffer_iter.next().unwrap().freeze();
        println!("Using data buffer: {:?}", data_buffer);
        let data_buffer = Buffer::from(data_buffer);
        let data_buffer = ScalarBuffer::<T::Native>::new(data_buffer, 0, num_rows as usize);

        Arc::new(PrimitiveArray::<T>::new(data_buffer, null_buffer))
    }

    fn primitive_array_from_buffers(
        data_type: &DataType,
        buffers: Vec<BytesMut>,
        num_rows: u32,
    ) -> Result<ArrayRef> {
        match data_type {
            DataType::Boolean => {
                let mut buffer_iter = buffers.into_iter();
                let null_buffer = buffer_iter.next().unwrap();
                let null_buffer = if null_buffer.is_empty() {
                    None
                } else {
                    let null_buffer = null_buffer.freeze().into();
                    Some(NullBuffer::new(BooleanBuffer::new(
                        Buffer::from_bytes(null_buffer),
                        0,
                        num_rows as usize,
                    )))
                };

                let data_buffer = buffer_iter.next().unwrap().freeze();
                let data_buffer = Buffer::from(data_buffer);
                let data_buffer = BooleanBuffer::new(data_buffer, 0, num_rows as usize);

                Ok(Arc::new(BooleanArray::new(data_buffer, null_buffer)))
            }
            DataType::Date32 => Ok(Self::new_primitive_array::<Date32Type>(buffers, num_rows)),
            DataType::Date64 => Ok(Self::new_primitive_array::<Date64Type>(buffers, num_rows)),
            DataType::Decimal128(_, _) => Ok(Self::new_primitive_array::<Decimal128Type>(
                buffers, num_rows,
            )),
            DataType::Decimal256(_, _) => Ok(Self::new_primitive_array::<Decimal256Type>(
                buffers, num_rows,
            )),
            DataType::Duration(units) => Ok(match units {
                TimeUnit::Second => {
                    Self::new_primitive_array::<DurationSecondType>(buffers, num_rows)
                }
                TimeUnit::Microsecond => {
                    Self::new_primitive_array::<DurationMicrosecondType>(buffers, num_rows)
                }
                TimeUnit::Millisecond => {
                    Self::new_primitive_array::<DurationMillisecondType>(buffers, num_rows)
                }
                TimeUnit::Nanosecond => {
                    Self::new_primitive_array::<DurationNanosecondType>(buffers, num_rows)
                }
            }),
            DataType::Float16 => Ok(Self::new_primitive_array::<Float16Type>(buffers, num_rows)),
            DataType::Float32 => Ok(Self::new_primitive_array::<Float32Type>(buffers, num_rows)),
            DataType::Float64 => Ok(Self::new_primitive_array::<Float64Type>(buffers, num_rows)),
            DataType::Int16 => Ok(Self::new_primitive_array::<Int16Type>(buffers, num_rows)),
            DataType::Int32 => Ok(Self::new_primitive_array::<Int32Type>(buffers, num_rows)),
            DataType::Int64 => Ok(Self::new_primitive_array::<Int64Type>(buffers, num_rows)),
            DataType::Int8 => Ok(Self::new_primitive_array::<Int8Type>(buffers, num_rows)),
            DataType::Interval(unit) => Ok(match unit {
                IntervalUnit::DayTime => {
                    Self::new_primitive_array::<IntervalDayTimeType>(buffers, num_rows)
                }
                IntervalUnit::MonthDayNano => {
                    Self::new_primitive_array::<IntervalMonthDayNanoType>(buffers, num_rows)
                }
                IntervalUnit::YearMonth => {
                    Self::new_primitive_array::<IntervalYearMonthType>(buffers, num_rows)
                }
            }),
            DataType::Null => Ok(new_null_array(data_type, num_rows as usize)),
            DataType::Time32(unit) => match unit {
                TimeUnit::Millisecond => Ok(Self::new_primitive_array::<Time32MillisecondType>(
                    buffers, num_rows,
                )),
                TimeUnit::Second => Ok(Self::new_primitive_array::<Time32SecondType>(
                    buffers, num_rows,
                )),
                _ => Err(Error::IO {
                    message: format!("invalid time unit {:?} for 32-bit time type", unit),
                    location: location!(),
                }),
            },
            DataType::Time64(unit) => match unit {
                TimeUnit::Microsecond => Ok(Self::new_primitive_array::<Time64MicrosecondType>(
                    buffers, num_rows,
                )),
                TimeUnit::Nanosecond => Ok(Self::new_primitive_array::<Time64NanosecondType>(
                    buffers, num_rows,
                )),
                _ => Err(Error::IO {
                    message: format!("invalid time unit {:?} for 64-bit time type", unit),
                    location: location!(),
                }),
            },
            DataType::Timestamp(unit, _) => Ok(match unit {
                TimeUnit::Microsecond => {
                    Self::new_primitive_array::<TimestampMicrosecondType>(buffers, num_rows)
                }
                TimeUnit::Millisecond => {
                    Self::new_primitive_array::<TimestampMillisecondType>(buffers, num_rows)
                }
                TimeUnit::Nanosecond => {
                    Self::new_primitive_array::<TimestampNanosecondType>(buffers, num_rows)
                }
                TimeUnit::Second => {
                    Self::new_primitive_array::<TimestampSecondType>(buffers, num_rows)
                }
            }),
            DataType::UInt16 => Ok(Self::new_primitive_array::<UInt16Type>(buffers, num_rows)),
            DataType::UInt32 => Ok(Self::new_primitive_array::<UInt32Type>(buffers, num_rows)),
            DataType::UInt64 => Ok(Self::new_primitive_array::<UInt64Type>(buffers, num_rows)),
            DataType::UInt8 => Ok(Self::new_primitive_array::<UInt8Type>(buffers, num_rows)),
            _ => Err(Error::IO {
                message: format!(
                    "The data type {} cannot be decoded from a primitive encoding",
                    data_type
                ),
                location: location!(),
            }),
        }
    }
}

impl FieldDecoder for PrimitiveFieldDecoder {
    fn finalize(&self, buffers: Vec<BytesMut>, num_rows: u32) -> Result<ArrayRef> {
        Self::primitive_array_from_buffers(&self.data_type, buffers, num_rows)
    }

    fn dest_buffers_per_col(&self) -> Vec<u32> {
        vec![2]
    }

    fn total_num_dest_buffers(&self) -> u32 {
        2
    }
}

fn field_decoders_from_schema(schema: &Schema) -> Vec<Box<dyn FieldDecoder>> {
    schema
        .fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Boolean
            | DataType::Date32
            | DataType::Date64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Duration(_)
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int8
            | DataType::Interval(_)
            | DataType::Null
            | DataType::RunEndEncoded(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::UInt8 => Box::new(PrimitiveFieldDecoder {
                data_type: field.data_type().clone(),
            }) as Box<dyn FieldDecoder>,
            _ => todo!(),
        })
        .collect::<Vec<_>>()
}

pub struct PageInfo {
    pub num_rows: u32,
    pub decoder: Arc<dyn PageDecoder>,
    pub buffer_offsets: Arc<Vec<u64>>,
}

pub struct ColumnInfo {
    page_infos: Vec<PageInfo>,
}

impl ColumnInfo {
    pub fn new(page_infos: Vec<PageInfo>) -> Self {
        Self { page_infos }
    }
}

pub struct DecodeScheduler {
    column_infos: Vec<ColumnInfo>,
}

#[derive(Debug)]
pub struct ScheduledPage {
    pub buffer_offsets: Arc<Vec<u64>>,
    pub io_request: Scheduled,
    pub num_rows: u32,
    pub col_idx: u32,
}

impl DecodeScheduler {
    pub fn new(column_infos: Vec<ColumnInfo>) -> Self {
        Self { column_infos }
    }

    pub async fn schedule_range(&self, range: Range<u32>, sink: mpsc::Sender<ScheduledPage>) {
        let mut rows_to_skip = range.start;
        let mut rows_to_read = range.end - range.start;
        let mut row_offsets = vec![0_u32; self.column_infos.len()];
        let mut page_offsets = vec![0_u32; self.column_infos.len()];
        let mut rows_queued = vec![0_u32; self.column_infos.len()];

        while rows_to_read > 0 {
            let mut min_rows_added = u32::MAX;
            for (col_idx, column) in self.column_infos.iter().enumerate() {
                if rows_queued[col_idx] == 0 {
                    let mut page_info = &column.page_infos[page_offsets[col_idx as usize] as usize];
                    page_offsets[col_idx] += 1;

                    while rows_to_skip > page_info.num_rows {
                        rows_to_skip -= page_info.num_rows;
                        row_offsets[col_idx as usize] += page_info.num_rows;
                        page_info = &column.page_infos[page_offsets[col_idx as usize] as usize];
                        page_offsets[col_idx] += 1;
                    }

                    let rows_to_add = rows_to_read.min(page_info.num_rows - rows_to_skip);
                    let range_start = row_offsets[col_idx as usize] + rows_to_skip;
                    let range_end = range_start + rows_to_add;
                    min_rows_added = min_rows_added.min(rows_to_add);
                    rows_queued[col_idx] = rows_to_add;
                    rows_to_skip = 0;

                    let scheduled = page_info.decoder.schedule_range(range_start..range_end);

                    sink.send(ScheduledPage {
                        buffer_offsets: page_info.buffer_offsets.clone(),
                        col_idx: col_idx as u32,
                        io_request: scheduled,
                        num_rows: rows_to_add,
                    })
                    .await
                    .unwrap();
                }
            }
            if min_rows_added == 0 {
                panic!("Error in scheduling logic, panic to avoid infinite loop");
            }
            rows_to_read -= min_rows_added;
            for col_idx in 0..self.column_infos.len() {
                rows_queued[col_idx] -= min_rows_added;
            }
        }
    }
}

#[derive(Debug)]
pub struct ReceivedPage {
    pub data: Vec<Vec<Bytes>>,
    pub decoder: Arc<dyn DataDecoder>,
    pub col_index: u32,
    pub num_rows: u32,
}

struct PartiallyDecodedPage {
    page: Arc<ReceivedPage>,
    rows_remaining: u32,
    rows_taken: u32,
}

struct DecodeStep {
    page: Arc<ReceivedPage>,
    rows_to_take: u32,
    rows_to_skip: u32,
}

struct DecodeTask {
    field_decoders: Arc<Vec<Box<dyn FieldDecoder>>>,
    columns: Vec<Vec<DecodeStep>>,
    schema: Arc<Schema>,
    num_rows: u32,
}

impl DecodeTask {
    fn run(self) -> Result<RecordBatch> {
        // TODO: to .map instead of for-each
        let mut arrays = Vec::new();
        let mut cols_iter = self.columns.into_iter();
        for field_decoder in self.field_decoders.iter() {
            let buffers_per_col = field_decoder.dest_buffers_per_col();
            let mut field_dest_buffers =
                Vec::with_capacity(field_decoder.total_num_dest_buffers() as usize);

            for num_buffers in buffers_per_col {
                let mut capacities = (0..num_buffers).map(|_| (0_u64, false)).collect::<Vec<_>>();
                let next_col_steps = cols_iter.next().unwrap();
                for step in &next_col_steps {
                    step.page.decoder.update_capacity(
                        &step.page.data,
                        step.rows_to_skip,
                        step.rows_to_take,
                        &mut capacities,
                    );
                }
                let mut buffers = capacities
                    .into_iter()
                    .map(|(num_bytes, needed)| {
                        if !needed {
                            BytesMut::new()
                        } else {
                            BytesMut::with_capacity(num_bytes as usize)
                        }
                    })
                    .collect::<Vec<_>>();
                let mut row_offset = 0;
                for step in &next_col_steps {
                    step.page.decoder.drain(
                        &step.page.data,
                        step.rows_to_skip,
                        step.rows_to_take,
                        row_offset,
                        &mut buffers,
                    );
                    row_offset += step.rows_to_take;
                }
                field_dest_buffers.extend(buffers);
            }

            arrays.push(field_decoder.finalize(field_dest_buffers, self.num_rows)?);
        }
        Ok(RecordBatch::try_new(self.schema.clone(), arrays)?)
    }
    fn submit(self) -> impl std::future::Future<Output = Result<RecordBatch>> {
        tokio::task::spawn_blocking(move || self.run()).unwrap_or_else(|err| panic!("{}", err))
    }
}

pub struct DecodeStream {
    source: mpsc::Receiver<ReceivedPage>,
    partial_pages: Vec<PartiallyDecodedPage>,
    schema: Arc<Schema>,
    field_decoders: Arc<Vec<Box<dyn FieldDecoder>>>,
    num_columns: u32,
    rows_remaining: u32,
    rows_per_batch: u32,
}

impl DecodeStream {
    pub fn new(
        rx: mpsc::Receiver<ReceivedPage>,
        schema: Arc<Schema>,
        num_columns: u32,
        num_rows: u32,
        rows_per_batch: u32,
    ) -> Self {
        let field_decoders = Arc::new(field_decoders_from_schema(schema.as_ref()));
        Self {
            source: rx,
            partial_pages: Vec::new(),
            schema,
            num_columns,
            rows_remaining: num_rows,
            rows_per_batch,
            field_decoders,
        }
    }

    async fn next_batch_task(&mut self) -> Option<DecodeTask> {
        if self.rows_remaining == 0 {
            return None;
        }
        let mut pages = Vec::new();
        std::mem::swap(&mut pages, &mut self.partial_pages);
        let mut page_iter = pages.into_iter().peekable();
        let mut batch_steps = Vec::new();
        let rows_in_batch = self.rows_per_batch.min(self.rows_remaining);
        self.rows_remaining -= rows_in_batch;
        for col_idx in 0..self.num_columns {
            let mut col_steps = Vec::new();
            let mut rows_remaining = rows_in_batch;
            while rows_remaining > 0 {
                let mut next_page_for_field = if page_iter
                    .peek()
                    .map(|partial_page| partial_page.page.col_index)
                    .unwrap_or(u32::MAX)
                    == col_idx
                {
                    page_iter.next().unwrap()
                } else {
                    let page = self.source.recv().await.unwrap();
                    PartiallyDecodedPage {
                        rows_remaining: page.num_rows,
                        rows_taken: 0,
                        page: Arc::new(page),
                    }
                };
                let rows_to_take = rows_remaining.min(next_page_for_field.rows_remaining);
                col_steps.push(DecodeStep {
                    page: next_page_for_field.page.clone(),
                    rows_to_take,
                    rows_to_skip: next_page_for_field.rows_taken,
                });
                next_page_for_field.rows_remaining -= rows_to_take;
                next_page_for_field.rows_taken += rows_to_take;
                rows_remaining -= rows_to_take;
                if next_page_for_field.rows_remaining > 0 {
                    self.partial_pages.push(next_page_for_field);
                }
            }
            batch_steps.push(col_steps);
        }
        Some(DecodeTask {
            columns: batch_steps,
            schema: self.schema.clone(),
            field_decoders: self.field_decoders.clone(),
            num_rows: rows_in_batch,
        })
    }

    pub fn into_stream(
        self,
    ) -> BoxStream<'static, impl std::future::Future<Output = Result<RecordBatch>>> {
        futures::stream::unfold(self, |mut slf| async move {
            let next_task = slf.next_batch_task().await;
            next_task.map(|task| (task.submit(), slf))
        })
        .boxed()
    }
}
