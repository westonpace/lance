use std::{ops::Range, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::BytesMut;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;

use lance_core::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::{encodings::logical, io::FileScheduler2};

pub trait DataDecoder: Send {
    fn load<'a>(&'a mut self) -> BoxFuture<'a, Result<()>>;
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]);
    fn drain(
        &self,
        // TODO: Coalesce on read and change this to &[Bytes]
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [BytesMut],
    );
}

pub struct PageInfo2 {
    pub num_rows: u32,
    pub decoder: Arc<dyn PhysicalPageScheduler>,
    pub buffer_offsets: Arc<Vec<u64>>,
}

pub struct ColumnInfo2 {
    pub page_infos: Vec<PageInfo2>,
}

impl ColumnInfo2 {
    pub fn new(page_infos: Vec<PageInfo2>) -> Self {
        Self { page_infos }
    }
}

pub struct BatchScheduler {
    field_schedulers: Vec<Box<dyn LogicalPageScheduler>>,
}

impl BatchScheduler {
    fn create_field_scheduler<'a>(
        field: &Field,
        column_infos: &mut impl Iterator<Item = &'a Arc<ColumnInfo2>>,
    ) -> Box<dyn LogicalPageScheduler> {
        match field.data_type() {
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
            | DataType::UInt8 => Box::new(logical::primitive::PrimitiveFieldScheduler::new(
                field.data_type().clone(),
                column_infos.next().unwrap().clone(),
            )),
            _ => todo!(),
        }
    }

    pub fn new(schema: &Schema, column_infos: Vec<Arc<ColumnInfo2>>) -> Self {
        let mut col_info_iter = column_infos.iter();
        let field_schedulers = schema
            .fields
            .iter()
            .map(|field| Self::create_field_scheduler(field, &mut col_info_iter))
            .collect::<Vec<_>>();
        Self { field_schedulers }
    }

    pub async fn schedule_range(
        &mut self,
        range: Range<u64>,
        sink: mpsc::Sender<Box<dyn LogicalPageDecoder>>,
        scheduler: Arc<dyn FileScheduler2>,
    ) -> Result<()> {
        let mut rows_to_read = range.end - range.start;

        let mut field_ranges = vec![range.clone(); self.field_schedulers.len()];
        let mut rows_queued = vec![0_u32; self.field_schedulers.len()];

        while rows_to_read > 0 {
            let mut min_rows_added = u32::MAX;
            for (col_idx, field_scheduler) in self.field_schedulers.iter_mut().enumerate() {
                if rows_queued[col_idx] == 0 {
                    let remaining_range = field_ranges[col_idx].clone();
                    let scheduled = field_scheduler.schedule_next(remaining_range, &scheduler)?;

                    let updated_range = (range.start + (scheduled.rows_taken as u64))..range.end;
                    field_ranges[col_idx] = updated_range;

                    sink.send(scheduled.decoder).await.unwrap();

                    min_rows_added = min_rows_added.min(scheduled.rows_taken);
                    rows_queued[col_idx] = scheduled.rows_taken;
                }
            }
            if min_rows_added == 0 {
                panic!("Error in scheduling logic, panic to avoid infinite loop");
            }
            rows_to_read -= min_rows_added as u64;
            for col_idx in 0..self.field_schedulers.len() {
                rows_queued[col_idx] -= min_rows_added;
            }
        }
        Ok(())
    }
}

struct DecodeBatchTask {
    columns: Vec<Vec<Box<dyn DecodeArrayTask>>>,
    schema: Schema,
}

impl DecodeBatchTask {
    fn run(self) -> Result<RecordBatch> {
        let columns = self
            .columns
            .into_iter()
            .map(|col_tasks| {
                let arrays = col_tasks
                    .into_iter()
                    .map(|col_task| col_task.decode())
                    .collect::<Result<Vec<_>>>()?;
                let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
                // TODO: If this is a primtiive column we should be able to avoid this
                // allocation + copy with "page bridging" which could save us a few CPU
                // cycles.
                Ok(arrow_select::concat::concat(&array_refs)?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new(Arc::new(self.schema), columns)?)
    }
}

struct PartiallyDecodedPage {
    decoder: Box<dyn LogicalPageDecoder>,
    col_index: u32,
}

pub struct BatchDecodeStream {
    scheduled: mpsc::Receiver<Box<dyn LogicalPageDecoder>>,
    partial_pages: Vec<PartiallyDecodedPage>,
    schema: Schema,
    rows_remaining: u64,
    rows_per_batch: u32,
    num_columns: u32,
}

impl BatchDecodeStream {
    async fn next_batch_task(&mut self) -> Result<Option<DecodeBatchTask>> {
        if self.rows_remaining == 0 {
            return Ok(None);
        }

        let mut pages = Vec::new();
        std::mem::swap(&mut pages, &mut self.partial_pages);
        let mut page_iter = pages.into_iter().peekable();
        let mut batch_steps = Vec::new();
        let rows_in_batch = (self.rows_per_batch as u64).min(self.rows_remaining) as u32;
        self.rows_remaining -= rows_in_batch as u64;

        for col_idx in 0..self.num_columns {
            let mut col_steps = Vec::new();
            let mut rows_remaining = rows_in_batch;
            while rows_remaining > 0 {
                let mut next_page_for_field = if page_iter
                    .peek()
                    .map(|partial_page| partial_page.col_index)
                    .unwrap_or(u32::MAX)
                    == col_idx
                {
                    page_iter.next().unwrap()
                } else {
                    let decoder = self.scheduled.recv().await.unwrap();
                    PartiallyDecodedPage {
                        col_index: col_idx,
                        decoder,
                    }
                };
                let next_step = next_page_for_field.decoder.drain(rows_remaining)?;
                rows_remaining -= next_step.num_rows;
                col_steps.push(next_step.task);
                if next_step.has_more {
                    self.partial_pages.push(next_page_for_field);
                }
            }
            batch_steps.push(col_steps);
        }
        Ok(Some(DecodeBatchTask {
            columns: batch_steps,
            schema: self.schema.clone(),
        }))
    }

    pub fn into_stream(self) -> BoxStream<'static, JoinHandle<Result<RecordBatch>>> {
        let stream = futures::stream::unfold(self, |mut slf| async move {
            let next_task = slf.next_batch_task().await;
            let next_task = next_task.transpose().map(|next_task| {
                tokio::spawn(async move {
                    let next_task = next_task?;
                    next_task.run()
                })
            });
            next_task.map(|next_task| (next_task, slf))
        });
        stream.boxed()
    }
}

pub trait PhysicalPageDecoder: Send + Sync {
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]);
    fn drain(
        &self,
        // TODO: Coalesce on read and change this to &[Bytes]
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [BytesMut],
    );
}

pub trait PhysicalPageScheduler {
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &dyn FileScheduler2,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>;
}

pub struct Scheduled2 {
    pub decoder: Box<dyn LogicalPageDecoder>,
    pub rows_taken: u32,
}

pub trait LogicalPageScheduler {
    fn schedule_next(
        &mut self,
        range: Range<u64>,
        scheduler: &Arc<dyn FileScheduler2>,
    ) -> Result<Scheduled2>;
    fn schedule_all(
        &mut self,
        range: Range<u64>,
        scheduler: Arc<dyn FileScheduler2>,
    ) -> Result<Box<dyn LogicalPageDecoder>>;
}

pub trait DecodeArrayTask: Send {
    fn decode(self: Box<Self>) -> Result<ArrayRef>;
}

pub struct NextDecodeTask {
    pub task: Box<dyn DecodeArrayTask>,
    pub num_rows: u32,
    pub has_more: bool,
}

pub trait LogicalPageDecoder: Send {
    fn wait<'a>(&'a mut self) -> BoxFuture<'a, Result<()>>;
    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask>;
}
