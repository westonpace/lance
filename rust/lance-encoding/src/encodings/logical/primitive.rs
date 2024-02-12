use std::sync::Arc;

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
    ArrayRef, BooleanArray, PrimitiveArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use bytes::BytesMut;
use futures::{future::BoxFuture, FutureExt};
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::decoder::{
    ColumnInfo2, DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask,
    PhysicalPageDecoder, Scheduled2,
};

pub struct PrimitiveFieldScheduler {
    data_type: DataType,
    column: Arc<ColumnInfo2>,
    scheduled_pages: u32,
}

impl PrimitiveFieldScheduler {
    pub fn new(data_type: DataType, column: Arc<ColumnInfo2>) -> Self {
        Self {
            data_type,
            column,
            scheduled_pages: 0,
        }
    }
}

impl LogicalPageScheduler for PrimitiveFieldScheduler {
    fn schedule_next(
        &mut self,
        range: std::ops::Range<u64>,
        scheduler: &Arc<dyn crate::io::FileScheduler2>,
    ) -> Result<Scheduled2> {
        let mut next_page = &self.column.page_infos[self.scheduled_pages as usize];
        let num_rows_desired = range.end - range.start;

        // First, skip any entirely skipped pages
        let mut rows_to_skip = range.start;
        while (next_page.num_rows as u64) < rows_to_skip {
            self.scheduled_pages += 1;
            rows_to_skip -= next_page.num_rows as u64;
            next_page = &self.column.page_infos[self.scheduled_pages as usize];
        }

        // Now we have page that overlaps our range somewhat.  Figure out how many
        // rows we can take
        let rows_available = next_page.num_rows as u64 - rows_to_skip;
        let rows_to_take = rows_available.min(num_rows_desired) as u32;

        let page_start = rows_to_skip as u32;
        let page_end = page_start + rows_to_take;

        let physical_decoder = next_page
            .decoder
            .schedule_range(page_start..page_end, scheduler.as_ref());

        self.scheduled_pages += 1;

        let logical_decoder = PrimitiveFieldDecoder {
            data_type: self.data_type.clone(),
            unloaded_physical_decoder: Some(physical_decoder),
            physical_decoder: None,
            rows_drained: 0,
            num_rows: rows_to_take,
        };

        Ok(Scheduled2 {
            decoder: Box::new(logical_decoder),
            rows_taken: rows_to_take,
        })
    }

    fn schedule_all(
        &mut self,
        _range: std::ops::Range<u64>,
        _scheduler: Arc<dyn crate::io::FileScheduler2>,
    ) -> Result<Box<dyn crate::decoder::LogicalPageDecoder>> {
        todo!()
    }
}

struct PrimitiveFieldDecoder {
    data_type: DataType,
    unloaded_physical_decoder: Option<BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>>,
    physical_decoder: Option<Arc<dyn PhysicalPageDecoder>>,
    num_rows: u32,
    rows_drained: u32,
}

struct PrimitiveFieldDecodeTask {
    rows_to_skip: u32,
    rows_to_take: u32,
    physical_decoder: Arc<dyn PhysicalPageDecoder>,
    data_type: DataType,
}

impl DecodeArrayTask for PrimitiveFieldDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let mut capacities = [(0, false), (0, true)];
        self.physical_decoder.update_capacity(
            self.rows_to_skip,
            self.rows_to_take,
            &mut capacities,
        );
        let mut bufs = capacities
            .into_iter()
            .map(|(num_bytes, is_needed)| {
                if is_needed {
                    BytesMut::with_capacity(num_bytes as usize)
                } else {
                    BytesMut::default()
                }
            })
            .collect::<Vec<_>>();

        self.physical_decoder
            .drain(self.rows_to_skip, self.rows_to_take, 0, &mut bufs);

        Self::primitive_array_from_buffers(&self.data_type, bufs, self.rows_to_take)
    }
}

impl PrimitiveFieldDecodeTask {
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

impl LogicalPageDecoder for PrimitiveFieldDecoder {
    fn wait<'a>(&'a mut self) -> BoxFuture<'a, Result<()>> {
        async move {
            let physical_decoder = self.unloaded_physical_decoder.take().unwrap().await?;
            self.physical_decoder = Some(Arc::from(physical_decoder));
            Ok(())
        }
        .boxed()
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        let rows_to_skip = self.rows_drained;
        let rows_to_take = num_rows - rows_to_skip;

        self.rows_drained += rows_to_take;

        let task = Box::new(PrimitiveFieldDecodeTask {
            rows_to_skip,
            rows_to_take,
            physical_decoder: self.physical_decoder.as_ref().unwrap().clone(),
            data_type: self.data_type.clone(),
        });

        Ok(NextDecodeTask {
            task,
            num_rows: rows_to_take,
            has_more: self.rows_drained != self.num_rows,
        })
    }
}

// impl<S: FileScheduler> FieldDecoder<S> for PrimitiveFieldScheduler<S> {
//     fn schedule_next_range(&mut self, range: Range<u64>, scheduler: &S) -> Result<u64> {
//         let mut next_page = &self.column.page_infos[self.scheduled_pages as usize];
//         let num_rows_desired = range.end - range.start;

//         // First, skip any entirely skipped pages
//         let mut rows_to_skip = range.start;
//         while (next_page.num_rows as u64) < rows_to_skip {
//             self.scheduled_pages += 1;
//             rows_to_skip -= next_page.num_rows as u64;
//             next_page = &self.column.page_infos[self.scheduled_pages as usize];
//         }

//         // Now we have page that overlaps our range somewhat.  Figure out how many
//         // rows we can take
//         let rows_available = next_page.num_rows as u64 - rows_to_skip;
//         let rows_to_take = rows_available.min(num_rows_desired) as u32;

//         let page_start = rows_to_skip as u32;
//         let page_end = page_start + rows_to_take;

//         let data_decoder = next_page
//             .decoder
//             .schedule_range(page_start..page_end, scheduler);
//         self.drainable_chunks.push_back(DrainableChunk {
//             decoder: data_decoder,
//             num_rows: rows_to_take,
//             rows_drained: 0,
//             loaded: false,
//         });

//         self.scheduled_pages += 1;

//         Ok(range.start + rows_to_take as u64)
//     }

//     async fn wait_for_rows(&mut self, num_rows: u32) -> Result<()> {
//         let mut rows_to_wait = num_rows;
//         let mut drainable_chunks_iter = self.drainable_chunks.iter_mut();
//         while rows_to_wait > 0 {
//             let next = drainable_chunks_iter.next().unwrap();
//             if !next.loaded {
//                 next.decoder.load().await?;
//             }
//             rows_to_wait -= next.rows_available();
//         }
//         Ok(())
//     }

//     fn drain(&mut self, num_rows: u32) -> Result<ArrayRef> {
//         let mut chunks_to_drain = Vec::new();
//         let mut drained_chunks = Vec::new();

//         let mut rows_to_drain = num_rows;

//         while self.drainable_chunks.front().unwrap().rows_available() < rows_to_drain {
//             let next_chunk = self.drainable_chunks.pop_front().unwrap();
//             rows_to_drain -= next_chunk.rows_available();
//             drained_chunks.push(next_chunk);
//         }

//         chunks_to_drain.extend(
//             drained_chunks
//                 .iter()
//                 .map(|chunk| (chunk, chunk.rows_drained, chunk.rows_available())),
//         );

//         if rows_to_drain > 0 {
//             let last_chunk = self.drainable_chunks.front_mut().unwrap();
//             let rows_to_skip = last_chunk.rows_drained;
//             last_chunk.rows_drained += rows_to_drain;
//             chunks_to_drain.push((last_chunk, rows_to_skip, rows_to_drain));
//         }

//         let mut capacities = [(0, false), (0, true)];
//         for (chunk, rows_to_skip, num_rows) in chunks_to_drain.iter() {
//             chunk
//                 .decoder
//                 .update_capacity(*rows_to_skip, *num_rows, &mut capacities)
//         }
//         let mut bufs = capacities
//             .into_iter()
//             .map(|(num_bytes, is_needed)| {
//                 if is_needed {
//                     BytesMut::with_capacity(num_bytes as usize)
//                 } else {
//                     BytesMut::default()
//                 }
//             })
//             .collect::<Vec<_>>();

//         let mut dest_offset = 0;
//         for (chunk, rows_to_skip, num_rows) in chunks_to_drain {
//             chunk
//                 .decoder
//                 .drain(rows_to_skip, num_rows, dest_offset, &mut bufs);
//             dest_offset += num_rows;
//         }

//         Self::primitive_array_from_buffers(&self.data_type, bufs, num_rows)
//     }
// }
