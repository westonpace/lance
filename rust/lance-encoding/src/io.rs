use std::ops::Range;

use bytes::Bytes;
use futures::future::BoxFuture;

use lance_core::Result;

pub trait FileScheduler2: Send + Sync {
    fn submit_request(&self, range: Vec<Range<u64>>) -> BoxFuture<'static, Result<Vec<Bytes>>>;
}
