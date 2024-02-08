use std::ops::Range;

// A request for a single range of data
#[derive(Debug)]
pub struct DirectIoRequest {
    pub range: Range<u64>,
}

// A request to fetch a single range of data indirectly
#[derive(Debug)]
pub struct IndirectIoRequest {
    pub offsets_range: Range<u64>,
    pub data_offset: u64,
}

/// A collection of requested I/O operations
#[derive(Debug)]
pub struct BatchRequest {
    pub direct_requests: Vec<DirectIoRequest>,
    pub indirect_requests: Vec<IndirectIoRequest>,
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

    /// Reserves capacity for `additional` additional direct requests
    ///
    /// This does not have to be called but can be to help save on allocations
    /// if you are going to be making many direct requests
    pub fn reserve_direct(&mut self, additional: u32) {
        self.direct_requests.reserve(additional as usize);
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

    /// Reserves capacity for `additional` additional indirect requests
    ///
    /// This does not have to be called but can be to help save on allocations
    /// if you are going to be making many indirect requests
    pub fn reserve_indirect(&mut self, additional: u32) {
        self.indirect_requests.reserve(additional as usize);
    }
}

impl Default for BatchRequest {
    fn default() -> Self {
        Self::new()
    }
}
