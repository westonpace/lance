use bytes::{BufMut, Bytes};
use futures::{future::BoxFuture, FutureExt};

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    io::FileScheduler2,
};

use lance_core::Result;

#[derive(Debug, Clone, Copy)]
struct ValueDecoder {
    bytes_per_value: u64,
}

struct ValueDataDecoder2 {
    bytes_per_value: u64,
    data: Vec<Bytes>,
}

impl PhysicalPageDecoder for ValueDataDecoder2 {
    fn update_capacity(&self, _rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        buffers[0].0 = self.bytes_per_value * num_rows as u64;
        buffers[0].1 = true;
    }

    fn drain(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) {
        let mut bytes_to_skip = rows_to_skip as u64 * self.bytes_per_value;
        let mut bytes_to_take = num_rows as u64 * self.bytes_per_value;

        let dest_offset_bytes = dest_offset as u64 * self.bytes_per_value;
        let dest = &mut dest_buffers[0].split_at_mut(dest_offset_bytes as usize).1;

        debug_assert!(dest.len() as u64 >= bytes_to_take);

        for buf in &self.data {
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

impl PhysicalPageScheduler for ValueDecoder {
    fn schedule_range(
        &self,
        range: std::ops::Range<u32>,
        scheduler: &dyn FileScheduler2,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let start = range.start as u64 * self.bytes_per_value;
        let end = range.end as u64 * self.bytes_per_value;
        let byte_range = start..end;

        let bytes = scheduler.submit_request(vec![byte_range]);
        let bytes_per_value = self.bytes_per_value;

        async move {
            let bytes = bytes.await?;
            Ok(Box::new(ValueDataDecoder2 {
                bytes_per_value: bytes_per_value,
                data: bytes,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}
