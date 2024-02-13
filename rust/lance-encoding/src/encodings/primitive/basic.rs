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

/// A physical scheduler for "basic" fields.  These are fields that have an optional
/// validity bitmap and some kind of values buffer.
///
/// No actual decoding happens here, we are simply aggregating the two buffers.
///
// TODO: A validity bitmap is also not needed if everything is null.  Refactor
// DataValidity to be
//
// NoNull(values decoder)
// SomeNull(validity decoder, values decoder)
// AllNull
pub struct BasicPageScheduler {
    validity_decoder: PageValidity,
    values_decoder: Box<dyn PhysicalPageScheduler>,
}

impl BasicPageScheduler {
    /// Creates a new instance that expect a validity bitmap
    pub fn new_nullable(
        validity_decoder: Box<dyn PhysicalPageScheduler>,
        values_decoder: Box<dyn PhysicalPageScheduler>,
    ) -> Self {
        Self {
            validity_decoder: PageValidity::SomeNull(validity_decoder),
            values_decoder,
        }
    }

    /// Create a new instance that does not need a validity bitmap because no item is null
    pub fn new_non_nullable(values_decoder: Box<dyn PhysicalPageScheduler>) -> Self {
        Self {
            validity_decoder: PageValidity::NoNull,
            values_decoder,
        }
    }
}

impl PhysicalPageScheduler for BasicPageScheduler {
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
            Ok(Box::new(BasicPageDecoder { validity, values }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct BasicPageDecoder {
    validity: DataValidity,
    values: Box<dyn PhysicalPageDecoder>,
}

impl PhysicalPageDecoder for BasicPageDecoder {
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        // No need to look at the validity decoder to know the dest buffer size since it is boolean
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        // The validity buffer is only required if we have some nulls
        buffers[0].1 = match self.validity {
            DataValidity::NoNull => false,
            DataValidity::SomeNull(_) => true,
        };
        self.values
            .update_capacity(rows_to_skip, num_rows, &mut buffers[1..]);
    }

    fn decode_into(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) {
        if let DataValidity::SomeNull(validity_decoder) = &self.validity {
            validity_decoder.decode_into(
                rows_to_skip,
                num_rows,
                dest_offset,
                &mut dest_buffers[..1],
            );
        }
        self.values
            .decode_into(rows_to_skip, num_rows, dest_offset, &mut dest_buffers[1..]);
    }
}

enum PageValidity {
    NoNull,
    SomeNull(Box<dyn PhysicalPageScheduler>),
}
