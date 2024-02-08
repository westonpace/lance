use arrow_array::{cast::AsArray, ArrayRef};
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{ArrayEncoder, EncodedPage},
    format::pb,
    EncodingsIo,
};

// use super::{bitmap::BitmapEncoder, value::ValueEncoder};

#[derive(Debug)]
pub struct FixedListScheduler {
    items_scheduler: Box<dyn PhysicalPageScheduler>,
    dimension: u32,
}

impl FixedListScheduler {
    pub fn new(items_scheduler: Box<dyn PhysicalPageScheduler>, dimension: u32) -> Self {
        Self {
            items_scheduler,
            dimension,
        }
    }
}

impl PhysicalPageScheduler for FixedListScheduler {
    fn schedule_range(
        &self,
        range: std::ops::Range<u32>,
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let expanded_range = (range.start * self.dimension)..(range.end * self.dimension);
        let inner_page_decoder = self
            .items_scheduler
            .schedule_range(expanded_range, scheduler);
        let dimension = self.dimension;
        async move {
            let items_decoder = inner_page_decoder.await?;
            Ok(Box::new(FixedListDecoder {
                items_decoder,
                dimension,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

pub struct FixedListDecoder {
    items_decoder: Box<dyn PhysicalPageDecoder>,
    dimension: u32,
}

impl PhysicalPageDecoder for FixedListDecoder {
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        let rows_to_skip = rows_to_skip * self.dimension;
        let num_rows = num_rows * self.dimension;
        self.items_decoder
            .update_capacity(rows_to_skip, num_rows, buffers);
    }

    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [bytes::BytesMut]) {
        let rows_to_skip = rows_to_skip * self.dimension;
        let num_rows = num_rows * self.dimension;
        self.items_decoder
            .decode_into(rows_to_skip, num_rows, dest_buffers);
    }
}

#[derive(Debug)]
pub struct FslEncoder {
    items_encoder: Box<dyn ArrayEncoder>,
    dimension: u32,
}

impl FslEncoder {
    pub fn new(items_encoder: Box<dyn ArrayEncoder>, dimension: u32) -> Self {
        Self {
            items_encoder,
            dimension,
        }
    }
}

impl ArrayEncoder for FslEncoder {
    fn encode(&self, arrays: &[ArrayRef]) -> Result<EncodedPage> {
        let inner_arrays = arrays
            .iter()
            .map(|arr| arr.as_fixed_size_list().values().clone())
            .collect::<Vec<_>>();
        let items_page = self.items_encoder.encode(&inner_arrays)?;
        Ok(EncodedPage {
            buffers: items_page.buffers,
            num_rows: items_page.num_rows / self.dimension,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::FixedSizeList(Box::new(
                    pb::FixedSizeList {
                        dimension: self.dimension,
                        items: Some(Box::new(items_page.encoding)),
                    },
                ))),
            },
            column_idx: items_page.column_idx,
        })
    }
}
