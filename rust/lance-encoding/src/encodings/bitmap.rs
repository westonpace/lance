use std::ops::Range;

use arrow_array::{cast::AsArray, types::UInt32Type, ArrayRef, PrimitiveArray};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::DataType;
use bytes::{Bytes, BytesMut};

use lance_core::Result;

use crate::{
    decoder::{DataDecoder, PageDecoder, Scheduled},
    encoder::{ArrayEncoder, EncodedArray, EncodedBuffer},
    io::BatchRequest,
};

#[derive(Debug, Clone, Copy)]
struct BitmapDecoder {}

impl DataDecoder for BitmapDecoder {
    fn update_capacity(
        &self,
        _data: &[Vec<Bytes>],
        _rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
    ) {
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        // This could be a validity buffer, if so, then it is needed since the writer
        // went through the hassle of encoding it
        buffers[0].1 = true;
    }

    fn drain(
        &self,
        data: &[Vec<Bytes>],
        rows_to_skip: u32,
        // TODO: Review and remove these unused args
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [BytesMut],
    ) {
        let mut bytes_to_fully_skip = rows_to_skip as u64 / 8;
        let mut bits_to_skip = rows_to_skip % 8;
        let data_buf = &data[0];

        let mut dest_builder = BooleanBufferBuilder::new(num_rows as usize);

        let mut rows_remaining = num_rows;
        for buf in data_buf {
            let buf_len = buf.len() as u64;
            if bytes_to_fully_skip > buf_len {
                println!("Skipping entire page of {} bytes", buf_len);
                bytes_to_fully_skip -= buf_len;
            } else {
                println!(
                    "Copying bytes {}..{} of page (skipping {} bits in first byte)",
                    bytes_to_fully_skip,
                    buf.len(),
                    bits_to_skip
                );
                let num_vals = (buf_len * 8) as u32;
                let num_vals_to_take = rows_remaining.min(num_vals);
                let end = (num_vals_to_take + bits_to_skip) as usize;
                // I love that this method exists
                dest_builder.append_packed_range(bits_to_skip as usize..end, &buf);
                bytes_to_fully_skip = 0;
                bits_to_skip = 0;
                rows_remaining -= num_vals_to_take;
            }
        }

        let bool_buffer = dest_builder.finish().into_inner();
        unsafe { dest_buffers[0].set_len(bool_buffer.len()) }
        // TODO: This requires an extra copy.  First we copy the data from the read buffer(s)
        // into dest_builder (one copy is inevitable).  Then we copy the data from dest_builder
        // into dest_buffers.  This second copy could be avoided (e.g. BooleanBufferBuilder
        // has a new_from_buffer but that requires MutableBuffer and we can't easily get there
        // from BytesMut [or can we?])
        //
        // Worst case, we vendor our own copy of BooleanBufferBuilder based on BytesMut.  We could
        // also use MutableBuffer ourselves instead of BytesMut but arrow-rs claims MutableBuffer may
        // be deprecated in the future (though that discussion seems to have died)

        // TODO: Will this work at the boundaries?  If we have to skip 3 bits for example then the first
        // bytes of bool_buffer.as_slice will be 000XXXXX and if we copy it on top of YYY00000 then the YYY
        // will be clobbered.
        dest_buffers[0][dest_offset as usize..].copy_from_slice(bool_buffer.as_slice());
    }
}

impl PageDecoder for BitmapDecoder {
    fn num_buffers(&self) -> u32 {
        1
    }

    fn schedule_range(&self, range: Range<u32>) -> Scheduled {
        debug_assert_ne!(range.start, range.end);
        let start = range.start as u64 / 8;
        let end = (range.end as u64 / 8) + 1;
        let byte_range = start..end;

        let mut io_req = BatchRequest::new();
        io_req.direct_read(byte_range);

        Scheduled {
            batch_requests: vec![io_req],
            data_decoder: Box::new(*self),
        }
    }

    fn schedule_take(&self, indices: PrimitiveArray<UInt32Type>) -> Scheduled {
        let mut io_request = BatchRequest::new();
        io_request.reserve_direct(indices.len() as u32);
        for idx in indices.values().iter() {
            let byte_idx = (*idx as u64) / 8;
            io_request.direct_read(byte_idx..byte_idx + 1)
        }

        // TODO: We will definitely want coalescing in the I/O scheduler, some of these
        // requests will be for the same byte

        Scheduled {
            batch_requests: vec![io_request],
            data_decoder: Box::new(*self),
        }
    }
}

// Encoder for writing boolean arrays as dense bitmaps
#[derive(Debug, Default)]
struct BitmapEncoder {}

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
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use crate::{encodings::basic::BasicDecoder, testing::check_round_trip_encoding};

    use super::{BitmapDecoder, BitmapEncoder};

    #[tokio::test]
    async fn test_bitmap_boolean() {
        let encoder = BitmapEncoder {};
        let decoder = Arc::new(BasicDecoder::new_non_nullable(Box::new(BitmapDecoder {})));
        let field = Field::new("", DataType::Boolean, false);

        check_round_trip_encoding(&encoder, decoder, field).await;
    }
}
