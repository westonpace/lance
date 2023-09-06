// Copyright 2023 Lance Developers.
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

//! Secondary Index pre-filter
//!
//! Based on the query, we might have information about which fragment ids and
//! row ids can be excluded from the search.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::intrinsics::unlikely;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};

use crate::error::Result;
use crate::io::deletion::DeletionVector;
use crate::{Dataset, Error};

/// Filter out row ids that we know are not relevant to the query. This currently
/// is just deleted rows.
#[derive(Debug)]
pub struct PreFilter {
    dataset: Arc<Dataset>,
    deletion_vecs: Vec<(usize, Option<Arc<DeletionVector>>)>,
}

pub struct PreFilt<'a, I: Iterator, F: Fn(&I::Item) -> (usize, usize)> {
    iter: I,
    map_fn: F,
    pre_filter: &'a PreFilter,
    cur_frag_idx: usize,
}

impl<'a, I: Iterator, F: Fn(&I::Item) -> (usize, usize)> Iterator for PreFilt<'a, I, F> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.iter.next() {
            let (frag_id, row_id) = (self.map_fn)(&item);
            while self.cur_frag_idx < self.pre_filter.deletion_vecs.len()
                && self.pre_filter.deletion_vecs[self.cur_frag_idx].0 < frag_id
            {
                self.cur_frag_idx += 1;
            }
            let (cur_frag_id, deletion_vec) = &self.pre_filter.deletion_vecs[self.cur_frag_idx];
            if unlikely(*cur_frag_id != frag_id) {
                continue;
            }
            if let Some(deletion_vec) = deletion_vec {
                if unlikely(deletion_vec.contains(row_id as u32)) {
                    continue;
                }
            }
            return Some(item);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.iter.size_hint().1)
    }
}

impl PreFilter {
    pub async fn try_new(dataset: Arc<Dataset>) -> Result<Self> {
        let fragments = dataset.get_fragments();
        // let deletion_vecs = futures::stream::iter(fragments.iter().map(|f| async move {
        //     let id = f.id();
        //     let dv = f.get_deletion_vector().await?;
        //     Ok::<_, Error>((id, dv))
        // }))
        // .buffered(num_cpus::get())
        // .try_collect::<Vec<_>>()
        // .await?;
        let deletion_vecs = futures::stream::iter(fragments.iter())
            .then(|f| async move {
                let id = f.id();
                let dv = f.get_deletion_vector().await?;
                Ok::<_, Error>((id, dv))
            })
            .try_collect::<Vec<_>>()
            .await?;
        Ok(Self {
            dataset,
            deletion_vecs,
        })
    }

    pub fn apply<I: Iterator, F: Fn(&I::Item) -> (usize, usize)>(
        &self,
        iter: I,
        map_fn: F,
    ) -> PreFilt<I, F> {
        PreFilt {
            iter,
            map_fn,
            pre_filter: self,
            cur_frag_idx: 0,
        }
    }

    /// Check whether a single row id should be included in the query.
    pub async fn check_one(&self, row_id: u64) -> Result<bool> {
        let fragment_id = (row_id >> 32) as u32;
        // If the fragment isn't found, then it must have been deleted.
        let Some(fragment) = self.dataset.get_fragment(fragment_id as usize) else {
            return Ok(false);
        };
        // If the fragment has no deletion vector, then the row must be there.
        let Some(deletion_vector) = fragment.get_deletion_vector().await? else {
            return Ok(true);
        };
        let local_row_id = row_id as u32;
        Ok(!deletion_vector.contains(local_row_id))
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    pub async fn filter_row_ids(&self, row_ids: &[u64]) -> Result<Vec<u64>> {
        let dataset = self.dataset.as_ref();
        let mut relevant_fragments: HashMap<u32, _> = HashMap::new();
        for row_id in row_ids {
            let fragment_id = (row_id >> 32) as u32;
            if let Entry::Vacant(entry) = relevant_fragments.entry(fragment_id) {
                if let Some(fragment) = dataset.get_fragment(fragment_id as usize) {
                    entry.insert(fragment);
                }
            }
        }
        let stream: BoxStream<_> = futures::stream::iter(relevant_fragments.drain())
            .map(|(fragment_id, fragment)| async move {
                let deletion_vector = match fragment.get_deletion_vector().await {
                    Ok(Some(deletion_vector)) => deletion_vector,
                    Ok(None) => return Ok((fragment_id, None)),
                    Err(err) => return Err(err),
                };
                Ok((fragment_id, Some(deletion_vector)))
            })
            .buffer_unordered(num_cpus::get())
            .boxed();
        let deletion_vector_map: HashMap<u32, Option<Arc<DeletionVector>>> =
            stream.try_collect::<HashMap<_, _>>().await?;

        let selection_vector = row_ids
            .iter()
            .enumerate()
            .filter_map(|(i, row_id)| {
                let fragment_id = (row_id >> 32) as u32;
                let local_row_id = *row_id as u32;
                match deletion_vector_map.get(&fragment_id) {
                    Some(Some(deletion_vector)) => {
                        if deletion_vector.contains(local_row_id) {
                            None
                        } else {
                            Some(i as u64)
                        }
                    }
                    // If the fragment has no deletion vector, then the row must be there.
                    Some(None) => Some(i as u64),
                    // If the fragment isn't found, then it must have been deleted.
                    None => None,
                }
            })
            .collect::<Vec<u64>>();

        Ok(selection_vector)
    }
}
