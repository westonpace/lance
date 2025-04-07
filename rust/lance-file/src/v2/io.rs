// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use lance_encoding::EncodingsIo;
use lance_io::scheduler::FileScheduler;

use super::reader::FileReaderIo;

#[derive(Debug)]
pub struct LanceEncodingsIo(pub FileScheduler);

impl EncodingsIo for LanceEncodingsIo {
    fn submit_request(
        &self,
        range: Vec<std::ops::Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, lance_core::Result<Vec<bytes::Bytes>>> {
        self.0.submit_request(range, priority).boxed()
    }
}

#[async_trait]
impl FileReaderIo for LanceEncodingsIo {
    async fn file_size(&self) -> Result<u64> {
        Ok(self.0.reader().size().await? as u64)
    }

    fn metadata_read_size(&self) -> u64 {
        self.0.reader().block_size() as u64
    }
}

#[derive(Debug)]
pub struct LanceEncodingsIoRef<'a>(pub &'a FileScheduler);

impl<'a> EncodingsIo for LanceEncodingsIoRef<'a> {
    fn submit_request(
        &self,
        range: Vec<std::ops::Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, lance_core::Result<Vec<bytes::Bytes>>> {
        self.0.submit_request(range, priority).boxed()
    }
}

#[async_trait]
impl<'a> FileReaderIo for LanceEncodingsIoRef<'a> {
    async fn file_size(&self) -> Result<u64> {
        Ok(self.0.reader().size().await? as u64)
    }

    fn metadata_read_size(&self) -> u64 {
        self.0.reader().block_size() as u64
    }
}
