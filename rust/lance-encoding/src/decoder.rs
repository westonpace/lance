use std::{collections::VecDeque, ops::Range, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::BytesMut;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;

use lance_core::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::encodings::logical::primitive::PrimitivePageScheduler;
use crate::io::FileScheduler2;

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
    pub page_infos: Vec<Arc<PageInfo2>>,
}

impl ColumnInfo2 {
    pub fn new(page_infos: Vec<Arc<PageInfo2>>) -> Self {
        Self { page_infos }
    }
}

pub struct BatchScheduler {
    field_schedulers: Vec<Vec<Box<dyn LogicalPageScheduler>>>,
}

#[derive(Debug, Clone, Copy)]
struct FieldWalkStatus {
    rows_to_skip: u64,
    rows_to_take: u64,
    page_offset: u32,
    rows_queued: u64,
}

impl FieldWalkStatus {
    fn new_from_range(range: Range<u64>) -> Self {
        Self {
            rows_to_skip: range.start,
            rows_to_take: range.end - range.start,
            page_offset: 0,
            rows_queued: 0,
        }
    }
}

impl BatchScheduler {
    fn create_field_scheduler<'a>(
        field: &Field,
        column_infos: &mut impl Iterator<Item = &'a Arc<ColumnInfo2>>,
    ) -> Vec<Box<dyn LogicalPageScheduler>> {
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
            | DataType::UInt8 => {
                // Primitive fields map to a single column
                let column = column_infos.next().unwrap();
                column
                    .page_infos
                    .iter()
                    .cloned()
                    .map(|page_info| {
                        Box::new(PrimitivePageScheduler::new(
                            field.data_type().clone(),
                            page_info,
                        )) as Box<dyn LogicalPageScheduler>
                    })
                    .collect::<Vec<_>>()
            }
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
        scheduler: &Arc<dyn FileScheduler2>,
    ) -> Result<()> {
        let mut rows_to_read = range.end - range.start;

        let mut field_status =
            vec![FieldWalkStatus::new_from_range(range); self.field_schedulers.len()];

        while rows_to_read > 0 {
            let mut min_rows_added = u32::MAX;
            for (col_idx, field_scheduler) in self.field_schedulers.iter().enumerate() {
                let status = &mut field_status[col_idx];
                if status.rows_queued == 0 {
                    let mut next_page = &field_scheduler[status.page_offset as usize];

                    while status.rows_to_skip > next_page.num_rows() as u64 {
                        status.rows_to_skip -= next_page.num_rows() as u64;
                        status.page_offset += 1;
                        next_page = &field_scheduler[status.page_offset as usize];
                    }

                    let page_range_start = status.rows_to_skip as u32;
                    let page_rows_remaining = next_page.num_rows() - page_range_start;
                    let rows_to_take = status.rows_to_take.min(page_rows_remaining as u64) as u32;
                    let page_range = page_range_start..(page_range_start + rows_to_take);

                    let scheduled = next_page.schedule_range(page_range, &scheduler)?;

                    status.rows_queued += rows_to_take as u64;
                    status.rows_to_take -= rows_to_take as u64;
                    status.page_offset += 1;
                    status.rows_to_skip = 0;

                    sink.send(scheduled).await.unwrap();

                    min_rows_added = min_rows_added.min(rows_to_take);
                }
            }
            if min_rows_added == 0 {
                panic!("Error in scheduling logic, panic to avoid infinite loop");
            }
            rows_to_read -= min_rows_added as u64;
            for col_idx in 0..self.field_schedulers.len() {
                field_status[col_idx].rows_queued -= min_rows_added as u64;
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
    partial_pages: VecDeque<PartiallyDecodedPage>,
    schema: Schema,
    rows_remaining: u64,
    rows_per_batch: u32,
    num_columns: u32,
}

impl BatchDecodeStream {
    pub fn new(
        scheduled: mpsc::Receiver<Box<dyn LogicalPageDecoder>>,
        schema: Schema,
        rows_per_batch: u32,
        num_rows: u64,
        num_columns: u32,
    ) -> Self {
        Self {
            scheduled,
            partial_pages: VecDeque::new(),
            schema,
            rows_remaining: num_rows,
            rows_per_batch,
            num_columns,
        }
    }

    async fn next_batch_task(&mut self) -> Result<Option<DecodeBatchTask>> {
        if self.rows_remaining == 0 {
            return Ok(None);
        }

        let mut batch_steps = Vec::new();
        let rows_in_batch = (self.rows_per_batch as u64).min(self.rows_remaining) as u32;
        println!(
            "next_batch_task self.rows_remaining={} rows_in_batch={}",
            self.rows_remaining, rows_in_batch
        );
        self.rows_remaining -= rows_in_batch as u64;

        for col_idx in 0..self.num_columns {
            let mut col_steps = Vec::new();
            let mut rows_remaining = rows_in_batch;
            while rows_remaining > 0 {
                println!("Looking for col_idx {}", col_idx);
                println!(
                    "Next partial is {}",
                    self.partial_pages
                        .front()
                        .map(|partial_page| partial_page.col_index)
                        .unwrap_or(u32::MAX)
                );
                let mut next_page_for_field = if self
                    .partial_pages
                    .front()
                    .map(|partial_page| partial_page.col_index)
                    .unwrap_or(u32::MAX)
                    == col_idx
                {
                    println!("Take partially decoded");
                    self.partial_pages.pop_front().unwrap()
                } else {
                    println!("Take new item from stream");
                    let mut decoder = self.scheduled.recv().await.unwrap();
                    decoder.wait().await?;
                    PartiallyDecodedPage {
                        col_index: col_idx,
                        decoder,
                    }
                };
                let next_step = next_page_for_field.decoder.drain(rows_remaining)?;
                rows_remaining -= next_step.num_rows;
                col_steps.push(next_step.task);
                println!(
                    "next_step has_more={} rows_remaining={}",
                    next_step.has_more, rows_remaining
                );
                if next_step.has_more {
                    println!("push front");
                    self.partial_pages.push_front(next_page_for_field);
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
    fn decode_into(
        &self,
        // TODO: Coalesce on read and change this to &[Bytes]
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [BytesMut],
    );
}

pub trait PhysicalPageScheduler: Send + Sync {
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &dyn FileScheduler2,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>;
}

pub trait LogicalPageScheduler: Send + Sync {
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &Arc<dyn FileScheduler2>,
    ) -> Result<Box<dyn LogicalPageDecoder>>;
    fn num_rows(&self) -> u32;
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
    fn avail(&self) -> u32;
}
