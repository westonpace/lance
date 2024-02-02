// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use arrow_array::UInt32Array;
use arrow_buffer::{Buffer, ScalarBuffer};
use bytes::Bytes;
use futures::channel::oneshot;
use futures::{FutureExt, StreamExt};
use object_store::path::Path;
use snafu::{location, Location};
use std::future::Future;
use std::mem::size_of;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::Stream;

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::traits::Reader;

// A request for a single range of data
struct DirectIoRequest {
    range: Range<u64>,
}

// A request to fetch a single range of data indirectly
struct IndirectIoRequest {
    offsets_range: Range<u64>,
    data_offset: u64,
}

/// A collection of requested I/O operations
pub struct BatchRequest {
    direct_requests: Vec<DirectIoRequest>,
    indirect_requests: Vec<IndirectIoRequest>,
}

impl BatchRequest {
    /// Create an empty request
    pub fn new() -> Self {
        Self {
            direct_requests: Vec::new(),
            indirect_requests: Vec::new(),
        }
    }

    /// Add a request to read a specified range of bytes from the disk
    pub fn direct_read(&mut self, range: Range<u64>) {
        self.direct_requests.push(DirectIoRequest { range })
    }

    /// Add a request to read a range of bytes from disk indirectly
    ///
    /// This assumes that the offsets of the the desired data are
    /// written somewhere.  It first reads those offsets and then reads
    /// the range specified by those offsets.
    ///
    /// For example, if the offsets are written to the start of the file
    ///
    /// [0]: 100
    /// [4]: 120
    /// [8]: 300
    /// [12]: 330
    ///
    /// And the request has offsets_range: 4..16 then it will
    /// first read those offsets (120, 300, and 330) and then read that range
    /// (120..330) from the disk.
    ///
    /// All offsets are assumed to be little endian u32 values.
    ///
    /// A `data_offset` can be supplied which will be added to the range
    /// used to access the data.  This allows offset buffers to store
    /// "buffer offsets" instead of "file offsets".
    ///
    /// The offset values themselves will be saved as well.  This
    /// can be used to recover the length of the individual items.
    pub fn indirect_read(&mut self, offsets_range: Range<u64>, data_offset: u64) {
        self.indirect_requests.push(IndirectIoRequest {
            offsets_range,
            data_offset,
        })
    }
}

// Every I/O task spawned will have a reference to this so that it can
// store its results.  When this goes out of scope the data is delivered.
struct MutableBatch {
    when_done: Option<oneshot::Sender<Result<LoadedBatch>>>,
    data_buffers: Vec<Bytes>,
    offset_buffers: Vec<UInt32Array>,
    err: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

struct DeliveredDirect {
    idx: u32,
    data: Bytes,
}

struct DeliveredIndirect {
    data_idx: u32,
    data: Bytes,
    offset_idx: u32,
    offsets: UInt32Array,
}

impl MutableBatch {
    // Converts this into a LoadedBatch for delivery
    fn into_loaded_batch(&mut self) -> LoadedBatch {
        let mut data_buffers = Vec::new();
        std::mem::swap(&mut self.data_buffers, &mut data_buffers);
        let mut offset_buffers = Vec::new();
        std::mem::swap(&mut self.offset_buffers, &mut offset_buffers);
        LoadedBatch {
            data_buffers,
            offset_buffers,
        }
    }

    // Called by worker tasks to add data to the MutableBatch
    fn deliver_direct(&mut self, data: Result<DeliveredDirect>) {
        match data {
            Ok(data) => {
                self.data_buffers[data.idx as usize] = data.data;
            }
            Err(err) => {
                // This keeps the original error, if present
                self.err.get_or_insert(Box::new(err));
            }
        }
    }

    fn deliver_indirect(&mut self, data: Result<DeliveredIndirect>) {
        match data {
            Ok(data) => {
                self.data_buffers[data.data_idx as usize] = data.data;
                self.offset_buffers[data.offset_idx as usize] = data.offsets;
            }
            Err(err) => {
                self.err.get_or_insert(Box::new(err));
            }
        }
    }
}

// Rather than keep track of when all the I/O requests are finished so that we
// can deliver the batch of data we let Rust do that for us.  When all I/O's are
// done then the MutableBatch will go out of scope and we know we have all the
// data.
impl Drop for MutableBatch {
    fn drop(&mut self) {
        // If we have an error, return that.  Otherwise return the data
        let result = if self.err.is_some() {
            Err(Error::Wrapped {
                error: self.err.take().unwrap().into(),
                location: location!(),
            })
        } else {
            Ok(self.into_loaded_batch())
        };
        // We don't really care if no one is around to receive it, just let
        // the result go out of scope and get cleaned up
        let _ = self.when_done.take().unwrap().send(result);
    }
}

type MutableBatchRef = Arc<Mutex<MutableBatch>>;

// An IoTask represents a stream of futures that will be used to load
// the data requested by a single BatchRequest
struct IoTask {
    // The original request
    request: BatchRequest,
    // The file we are reading from
    reader: Arc<dyn Reader>,
    // Where to put the bytes.  Initially blank until the task is allocated.
    dest: Option<MutableBatchRef>,
    // Where to deliver the bytes when all is done.  When this task is allocated
    // the sender will be moved out of here and into `dest`.
    when_done: Option<oneshot::Sender<Result<LoadedBatch>>>,
    // The current position of the stream in request.indirect_requests
    indirect_index: usize,
    // The current position of the stream inrequest.direct_requests
    direct_index: usize,
}

impl IoTask {
    fn new(
        request: BatchRequest,
        reader: Arc<dyn Reader>,
        when_done: oneshot::Sender<Result<LoadedBatch>>,
    ) -> Self {
        Self {
            request,
            reader,
            dest: None,
            when_done: Some(when_done),
            indirect_index: 0,
            direct_index: 0,
        }
    }

    fn spawn_direct(&mut self, task_idx: usize) -> tokio::task::JoinHandle<()> {
        let task = &self.request.direct_requests[task_idx];

        let reader = self.reader.clone();
        let dest = self.dest.as_ref().unwrap().clone();
        let range = (task.range.start as usize)..(task.range.end as usize);

        tokio::spawn(async move {
            let bytes = reader.get_range(range).await;
            let mut dest = dest.lock().unwrap();
            dest.deliver_direct(bytes.map(|bytes| DeliveredDirect {
                data: bytes,
                idx: task_idx as u32,
            }));
        })
    }

    async fn do_indirect(
        reader: Arc<dyn Reader>,
        offsets_range: Range<u64>,
        data_offset: u64,
        data_idx: u32,
        offset_idx: u32,
    ) -> Result<DeliveredIndirect> {
        let bytes = reader
            .get_range(offsets_range.start as usize..offsets_range.end as usize)
            .await?;
        let length = bytes.len() / size_of::<u32>();
        let values = ScalarBuffer::new(Buffer::from_bytes(bytes.into()), 0, length);
        let offsets = UInt32Array::new(values, None);

        let start = offsets.value(0);
        let end = offsets.value(offsets.len() - 1);
        let start = start as usize + data_offset as usize;
        let end = end as usize + data_offset as usize;
        let data_bytes = reader.get_range(start..end).await?;

        Ok(DeliveredIndirect {
            data_idx,
            data: data_bytes,
            offset_idx,
            offsets,
        })
    }

    fn spawn_indirect(&self, offset_idx: usize) -> JoinHandle<()> {
        let data_idx = self.request.direct_requests.len() + offset_idx;
        let task = &self.request.indirect_requests[offset_idx];

        let reader = self.reader.clone();
        let dest = self.dest.as_ref().unwrap().clone();
        let offsets_range = task.offsets_range.clone();
        let data_offset = task.data_offset;

        tokio::spawn(async move {
            let to_deliver = Self::do_indirect(
                reader,
                offsets_range,
                data_offset,
                data_idx as u32,
                offset_idx as u32,
            )
            .await;
            let mut dest = dest.lock().unwrap();
            dest.deliver_indirect(to_deliver);
        })
    }

    // We don't want to allocate a task as soon as it is created so this is
    // deferred.  This should be called once a task is ready to start working
    // and before it is consumed as a stream.
    fn allocate(&mut self) {
        let num_data_buffers =
            self.request.direct_requests.len() + self.request.indirect_requests.len();
        let data_buffers = vec![Bytes::default(); num_data_buffers];

        let num_offset_buffers = self.request.indirect_requests.len();
        let default_arr = UInt32Array::new_null(0);
        let offset_buffers = vec![default_arr; num_offset_buffers];

        self.dest = Some(Arc::new(Mutex::new(MutableBatch {
            data_buffers,
            offset_buffers,
            when_done: self.when_done.take(),
            err: None,
        })));
    }
}

impl Stream for IoTask {
    type Item = JoinHandle<()>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // This task_index is wrong because it doesn't take into account multiple buffers
        if self.indirect_index < self.request.indirect_requests.len() {
            let task_index = self.indirect_index;
            self.indirect_index += 1;
            std::task::Poll::Ready(Some(self.spawn_indirect(task_index)))
        } else if self.direct_index < self.request.direct_requests.len() {
            let task_index = self.request.indirect_requests.len() + self.direct_index;
            self.direct_index += 1;
            std::task::Poll::Ready(Some(self.spawn_direct(task_index)))
        } else {
            std::task::Poll::Ready(None)
        }
    }
}

// A batch of data loaded in response to a BatchRequest
pub struct LoadedBatch {
    /// The loaded data, grouped into one or more data buffers
    ///
    /// Each call to [`BatchRequest::direct_read`] or [`BatchRequest::indirect_read`]
    /// will result in exactly one `Bytes` object in this result.
    pub data_buffers: Vec<Bytes>,
    /// If there are indirect reads then the file offsets will be placed into this
    /// buffer.  This can be used to retrieve the lengths of the individual items.
    pub offset_buffers: Vec<UInt32Array>,
}

// Every time a scheduler starts up it launches a task to run the I/O loop.  This loop
// repeats endlessly until the scheduler is destroyed.
async fn run_io_loop(tasks: mpsc::UnboundedReceiver<IoTask>, io_capacity: u32) {
    let task_stream = UnboundedReceiverStream::new(tasks);
    let mut task_stream = task_stream
        .flat_map(|mut task| {
            task.allocate();
            task
        })
        .buffer_unordered(io_capacity as usize);
    while let Some(_) = task_stream.next().await {
        // We don't actually do anything with the results here, they are sent
        // via the io tasks's when_done.  Instead we just keep chugging away
        // indefinitely until the tasks receiver returns none (scheduler has
        // been shut down)
    }
}

/// Wraps an ObjectStore and throttles the amount of parallel I/O that can
/// be run.
///
/// TODO: This will also add coalescing
pub struct StoreScheduler {
    object_store: Arc<ObjectStore>,
    task_submitter: mpsc::UnboundedSender<IoTask>,
}

impl StoreScheduler {
    /// Create a new scheduler with the given I/O capacity
    ///
    /// # Arguments
    ///
    /// - object_store: the store to wrap
    /// - io_capacity: the maximum number of parallel requests that will be allowed
    pub fn new(object_store: Arc<ObjectStore>, io_capacity: u32) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let scheduler = Self {
            object_store,
            task_submitter: tx,
        };
        tokio::task::spawn(async move { run_io_loop(rx, io_capacity).await });
        scheduler
    }

    /// Open a file for reading
    pub async fn open_file(&self, path: &Path) -> Result<FileScheduler> {
        let reader = self.object_store.open(path).await?;
        Ok(FileScheduler {
            reader: reader.into(),
            root: self,
        })
    }

    fn submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: BatchRequest,
    ) -> impl Future<Output = Result<LoadedBatch>> + Send {
        let (tx, rx) = oneshot::channel::<Result<LoadedBatch>>();
        let io_task = IoTask::new(request, reader, tx);
        // We can unwrap here because the only possible error could be that the receiver
        // has shut down but that should only happen if the receiver had a panic.
        self.task_submitter.send(io_task).unwrap();
        // Right now, it isn't possible for I/O to be cancelled so a cancel error should
        // not occur
        rx.map(|wrapped_err| wrapped_err.unwrap())
    }
}

/// A throttled file reader
pub struct FileScheduler<'a> {
    reader: Arc<dyn Reader>,
    root: &'a StoreScheduler,
}

impl<'a> FileScheduler<'a> {
    /// Submit a batch of I/O requests to the reader
    ///
    /// The requests will be queued in a FIFO manner and, when all requests
    /// have been fulfilled, the returned future will be completed.
    pub fn submit_request(
        &self,
        request: BatchRequest,
    ) -> impl Future<Output = Result<LoadedBatch>> + Send {
        self.root.submit_request(self.reader.clone(), request)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use byteorder::{LittleEndian, WriteBytesExt};
    use bytes::BufMut;
    use object_store::path::Path;
    use rand::RngCore;
    use tempfile::tempdir;
    use tracing::subscriber;
    use tracing_chrome::{ChromeLayerBuilder, TraceStyle};
    use tracing_subscriber::{filter, prelude::*, Layer, Registry};

    use super::*;

    #[tokio::test]
    async fn test_full_seq_read() {
        let tmpdir = tempdir().unwrap();
        let tmp_path = tmpdir.path().to_str().unwrap();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_file = tmp_path.child("foo.file");

        let obj_store = Arc::new(ObjectStore::local());

        // Write 1MiB of data
        const DATA_SIZE: u64 = 1024 * 1024;
        let mut some_data = vec![0; DATA_SIZE as usize];
        rand::thread_rng().fill_bytes(&mut some_data);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let scheduler = StoreScheduler::new(obj_store, 16);

        let file_scheduler = scheduler.open_file(&tmp_file).await.unwrap();

        // Read it back 4KiB at a time
        const READ_SIZE: u64 = 4 * 1024;
        let mut reqs = VecDeque::new();
        let mut offset = 0;
        while offset < DATA_SIZE {
            reqs.push_back(
                file_scheduler
                    .submit_request(BatchRequest {
                        direct_requests: vec![DirectIoRequest {
                            range: offset..offset + READ_SIZE,
                        }],
                        indirect_requests: vec![],
                    })
                    .await
                    .unwrap(),
            );
            offset += READ_SIZE;
        }

        offset = 0;
        // Note: we should get parallel I/O even though we are consuming serially
        while offset < DATA_SIZE {
            let data = reqs.pop_front().unwrap();
            let actual = &data.data_buffers[0];
            let expected = &some_data[offset as usize..(offset + READ_SIZE) as usize];
            assert_eq!(expected, actual);
            offset += READ_SIZE;
        }
    }

    #[tokio::test]
    async fn test_full_indirect_read() {
        let builder = ChromeLayerBuilder::new()
            .trace_style(TraceStyle::Async)
            .include_args(true);
        let (chrome_layer, _guard) = builder.build();
        // Narrow down to just our targets, otherwise we get a lot of spam from
        // our dependencies. The target check is based on a prefix, so `lance` is
        // sufficient to match `lance_*`.
        let filter = filter::Targets::new().with_target("lance", filter::LevelFilter::DEBUG);
        let subscriber = Registry::default().with(chrome_layer.with_filter(filter));
        subscriber::set_global_default(subscriber).unwrap();

        let tmpdir = tempdir().unwrap();
        let tmp_path = tmpdir.path().to_str().unwrap();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_file = tmp_path.child("foo.file");

        let obj_store = Arc::new(ObjectStore::local());

        // We will pretend this is binary data and write 32Ki strings consisting
        // of 32 characters each (1MiB of data + (256KiB + 8B) of offsets)
        const STRING_WIDTH: u64 = 32;
        const NUM_STRINGS: u64 = 32 * 1024;
        const DATA_SIZE: u64 = STRING_WIDTH * NUM_STRINGS;
        const OFFSET_SIZE: u64 = size_of::<u32>() as u64 * (NUM_STRINGS + 1);
        let mut some_data = Vec::with_capacity((DATA_SIZE + OFFSET_SIZE) as usize);
        // Initialize the offsets section with offsets
        some_data.write_u32::<LittleEndian>(0).unwrap();
        for idx in 0..NUM_STRINGS {
            some_data
                .write_u32::<LittleEndian>((idx as u32 + 1) * 32)
                .unwrap();
        }
        // Initialize the data section with random data
        some_data.put_bytes(0, DATA_SIZE as usize);
        rand::thread_rng().fill_bytes(&mut some_data[OFFSET_SIZE as usize..]);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let scheduler = StoreScheduler::new(obj_store, 16);

        let file_scheduler = scheduler.open_file(&tmp_file).await.unwrap();

        // Read it back in one big read
        let mut req = BatchRequest::new();
        req.indirect_read(0..OFFSET_SIZE, OFFSET_SIZE);
        let data = file_scheduler.submit_request(req).await.unwrap();

        assert_eq!(data.offset_buffers.len(), 1);
        assert_eq!(data.data_buffers.len(), 1);
        assert_eq!(data.data_buffers[0], some_data[OFFSET_SIZE as usize..]);
        assert!(data.offset_buffers[0]
            .values()
            .iter()
            .enumerate()
            .all(|(idx, len)| *len == (idx as u32 * STRING_WIDTH as u32)));

        // Read it back in batches
        let mut reqs = VecDeque::new();
        let mut offset = 0;
        const BATCH_SIZE: u64 = 1024 * size_of::<u32>() as u64;
        while offset < DATA_SIZE {
            let mut req = BatchRequest::new();
            req.indirect_read(
                offset..offset + BATCH_SIZE + size_of::<u32>() as u64,
                OFFSET_SIZE,
            );
            reqs.push_back(file_scheduler.submit_request(req));
            offset += BATCH_SIZE;
        }

        let mut data_offset = OFFSET_SIZE;
        const DATA_BATCH_SIZE: u64 = 1024 * STRING_WIDTH;
        // Note: we should get parallel I/O even though we are consuming serially
        while data_offset < DATA_SIZE {
            let data = reqs.pop_front().unwrap().await.unwrap();
            let actual = &data.data_buffers[0];
            let expected =
                &some_data[data_offset as usize..(data_offset + DATA_BATCH_SIZE) as usize];
            assert_eq!(expected, actual);
            data_offset += DATA_BATCH_SIZE;
        }
    }
}
