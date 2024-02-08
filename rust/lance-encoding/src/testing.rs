use std::sync::Arc;

use arrow_schema::{Field, Schema};
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use lance_datagen::{array, gen, RowCount};
use tokio::sync::mpsc;

use crate::{
    decoder::{
        ColumnInfo, DecodeScheduler, DecodeStream, PageDecoder, PageInfo, ReceivedPage,
        ScheduledPage,
    },
    encoder::{ArrayEncoder, EncodedArray},
    io::DirectIoRequest,
};

pub(crate) struct SimulatedScheduler {
    data: Bytes,
}

impl SimulatedScheduler {
    pub fn new(data: Vec<EncodedArray>) -> Self {
        let mut bytes = BytesMut::new();
        for arr in data.into_iter() {
            for buf in arr.buffers.into_iter() {
                for part in buf.parts.into_iter() {
                    bytes.extend(part.into_iter())
                }
            }
        }
        Self {
            data: bytes.freeze(),
        }
    }

    fn satisfy_request(&self, req: DirectIoRequest) -> Bytes {
        self.data
            .slice(req.range.start as usize..req.range.end as usize)
    }

    pub async fn schedule(
        &self,
        mut rx: mpsc::Receiver<ScheduledPage>,
        num_pages: u32,
    ) -> mpsc::Receiver<ReceivedPage> {
        let (tx, dest_rx) = mpsc::channel::<ReceivedPage>(num_pages as usize);
        while let Some(io) = rx.recv().await {
            let decoder = io.io_request.data_decoder;
            let data = io
                .io_request
                .batch_requests
                .into_iter()
                .map(|batch_req| {
                    let bytes = batch_req
                        .direct_requests
                        .into_iter()
                        .map(|direct_request| self.satisfy_request(direct_request))
                        .collect::<Vec<_>>();
                    bytes
                })
                .collect::<Vec<_>>();
            let received_page = ReceivedPage {
                data,
                decoder: decoder.into(),
                col_index: io.col_idx,
                num_rows: io.num_rows,
            };
            tx.send(received_page).await.unwrap();
        }
        dest_rx
    }
}

pub async fn check_round_trip_encoding(
    encoder: &dyn ArrayEncoder,
    decoder: Arc<dyn PageDecoder>,
    field: Field,
) {
    let data = gen()
        .col(None, array::rand_type(field.data_type()))
        .into_batch_rows(RowCount::from(10000))
        .unwrap()
        .column(0)
        .clone();

    let num_rows = data.len();

    for num_pages in [1, 5, 10] {
        println!("RUNNING TEST WITH {} PAGES", num_pages);
        let rows_per_page = num_rows / num_pages;

        let mut offset = 0;
        let mut encoded_arrays = Vec::new();
        let mut page_infos = Vec::new();
        let mut buffer_offset = 0;

        for _ in 0..num_pages {
            let data = data.slice(offset, rows_per_page);

            let encoded_array = encoder.encode(vec![data]).unwrap();
            let buffer_offsets = encoded_array
                .buffers
                .iter()
                .map(|buf| {
                    let offset = buffer_offset;
                    buffer_offset += buf.parts.iter().map(|part| part.len() as u64).sum::<u64>();
                    offset
                })
                .collect::<Vec<_>>();
            encoded_arrays.push(encoded_array);

            let page_info = PageInfo {
                num_rows: rows_per_page as u32,
                decoder: decoder.clone(),
                buffer_offsets: Arc::new(buffer_offsets),
            };
            page_infos.push(page_info);

            offset += rows_per_page;
        }

        let scheduler = SimulatedScheduler::new(encoded_arrays);

        let column_infos = vec![ColumnInfo::new(page_infos)];
        let decode_scheduler = DecodeScheduler::new(column_infos);

        let (tx, rx) = mpsc::channel(1024);

        decode_scheduler.schedule_range(100..1100, tx).await;

        let rx = scheduler.schedule(rx, 1024).await;

        let schema = Schema::new(vec![field.clone()]);

        let mut decode_stream =
            DecodeStream::new(rx, Arc::new(schema), 1, num_rows as u32, 100).into_stream();

        while let Some(batch) = decode_stream.next().await {
            let batch = batch.await.unwrap();
            dbg!(batch);
        }
    }
}
