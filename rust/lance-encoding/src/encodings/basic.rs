use futures::{future::BoxFuture, FutureExt};

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    io::FileScheduler2,
};

use lance_core::Result;

enum DataValidity {
    NoNull,
    SomeNull(Box<dyn PhysicalPageDecoder>),
}

struct BasicDataDecoder {
    validity: DataValidity,
    values: Box<dyn PhysicalPageDecoder>,
}

impl PhysicalPageDecoder for BasicDataDecoder {
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        // No need to look at the validity decoder to know the dest buffer size since it is boolean
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        buffers[0].1 = match self.validity {
            DataValidity::NoNull => false,
            DataValidity::SomeNull(_) => true,
        };
        self.values
            .update_capacity(rows_to_skip, num_rows, &mut buffers[1..]);
    }

    fn drain(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) {
        if let DataValidity::SomeNull(validity_decoder) = &self.validity {
            validity_decoder.drain(rows_to_skip, num_rows, dest_offset, &mut dest_buffers[..1]);
        }
        self.values
            .drain(rows_to_skip, num_rows, dest_offset, &mut dest_buffers[1..]);
    }
}

enum PageValidity {
    NoNull,
    SomeNull(Box<dyn PhysicalPageScheduler>),
}

pub struct BasicDecoder {
    validity_decoder: PageValidity,
    values_decoder: Box<dyn PhysicalPageScheduler>,
}

impl BasicDecoder {
    pub fn new_nullable(
        validity_decoder: Box<dyn PhysicalPageScheduler>,
        values_decoder: Box<dyn PhysicalPageScheduler>,
    ) -> Self {
        Self {
            validity_decoder: PageValidity::SomeNull(validity_decoder),
            values_decoder,
        }
    }

    pub fn new_non_nullable(values_decoder: Box<dyn PhysicalPageScheduler>) -> Self {
        Self {
            validity_decoder: PageValidity::NoNull,
            values_decoder,
        }
    }
}

impl PhysicalPageScheduler for BasicDecoder {
    fn schedule_range(
        &self,
        range: std::ops::Range<u32>,
        scheduler: &dyn FileScheduler2,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let validity_future = match &self.validity_decoder {
            PageValidity::NoNull => None,
            PageValidity::SomeNull(validity_decoder) => {
                Some(validity_decoder.schedule_range(range.clone(), scheduler))
            }
        };

        let values_future = self.values_decoder.schedule_range(range, scheduler);

        async move {
            let validity = match validity_future {
                None => DataValidity::NoNull,
                Some(fut) => DataValidity::SomeNull(fut.await?),
            };
            let values = values_future.await?;
            Ok(Box::new(BasicDataDecoder { validity, values }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }

    // fn schedule_take(
    //     &self,
    //     indices: arrow_array::PrimitiveArray<arrow_array::types::UInt32Type>,
    //     scheduler: &S,
    // ) -> Box<dyn DataDecoder> {
    //     let validity = match &self.validity_decoder {
    //         PageValidity::NoNull => DataValidity::NoNull,
    //         PageValidity::SomeNull(validity_decoder) => {
    //             DataValidity::SomeNull(validity_decoder.schedule_take(indices.clone(), scheduler))
    //         }
    //     };
    //     let values = self.values_decoder.schedule_take(indices, scheduler);
    //     Box::new(BasicDataDecoder { validity, values })
    // }
}
