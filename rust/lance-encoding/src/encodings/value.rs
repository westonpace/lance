use bytes::BufMut;

use crate::{
    decoder::{DataDecoder, PageDecoder, Scheduled},
    io::BatchRequest,
};

#[derive(Debug, Clone, Copy)]
struct ValueDecoder {
    bytes_per_value: u64,
}

impl DataDecoder for ValueDecoder {
    fn update_capacity(
        &self,
        _data: &[Vec<bytes::Bytes>],
        _rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
    ) {
        buffers[0].0 = self.bytes_per_value * num_rows as u64;
        buffers[0].1 = true;
    }

    fn drain(
        &self,
        data: &[Vec<bytes::Bytes>],
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) {
        let mut bytes_to_skip = rows_to_skip as u64 * self.bytes_per_value;
        let mut bytes_to_take = num_rows as u64 * self.bytes_per_value;
        let data_buf = &data[0];

        let dest_offset_bytes = dest_offset as u64 * self.bytes_per_value;
        let dest = &mut dest_buffers[0].split_at_mut(dest_offset_bytes as usize).1;

        debug_assert!(dest.len() as u64 >= bytes_to_take);

        for buf in data_buf {
            let buf_len = buf.len() as u64;
            if bytes_to_skip > buf_len {
                bytes_to_skip -= buf_len;
            } else {
                let bytes_to_take_here = buf_len.min(bytes_to_take);
                bytes_to_take -= bytes_to_take_here;
                let start = bytes_to_skip as usize;
                let end = start + bytes_to_take_here as usize;
                dest.put_slice(&buf.slice(start..end));
                bytes_to_skip = 0;
            }
        }
    }
}

impl PageDecoder for ValueDecoder {
    fn schedule_range(&self, range: std::ops::Range<u32>) -> Scheduled {
        let start = range.start as u64 * self.bytes_per_value;
        let end = range.end as u64 * self.bytes_per_value;
        let byte_range = start..end;

        let mut batch_request = BatchRequest::new();
        batch_request.direct_read(byte_range);

        Scheduled::new(vec![batch_request], Box::new(*self))
    }

    fn schedule_take(
        &self,
        indices: arrow_array::PrimitiveArray<arrow_array::types::UInt32Type>,
    ) -> crate::decoder::Scheduled {
        let mut batch_request = BatchRequest::new();
        batch_request.reserve_direct(indices.len() as u32);

        for idx in indices.values().iter() {
            let start = *idx as u64 * self.bytes_per_value;
            let end = start + self.bytes_per_value as u64;
            batch_request.direct_read(start..end);
        }

        Scheduled::new(vec![batch_request], Box::new(*self))
    }

    fn num_buffers(&self) -> u32 {
        1
    }
}

// Plain decoder does not really make sense as a
