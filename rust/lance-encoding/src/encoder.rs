use arrow_array::ArrayRef;
use arrow_buffer::Buffer;

use lance_core::Result;

/// An encoded buffer
pub struct EncodedBuffer {
    /// If true, the buffer should be stored as "data"
    /// If false, the buffer should be stored as "metadata"
    ///
    /// Metadata buffers are typically small buffers that should be cached.  For example,
    /// this might be a small dictionary when data has been dictionary encoded.  Or it might
    /// contain a skip block when data has been RLE encoded.
    pub is_data: bool,
    /// Buffers that make up the encoded buffer
    ///
    /// All of these buffers should be written to the file as one contiguous buffer
    ///
    /// This is a Vec to allow for zero-copy
    ///
    /// For example, if we are asked to write 3 primitive arrays of 1000 rows and we can write them all
    /// as one page then this will be the value buffers from the 3 primitive arrays
    pub parts: Vec<Buffer>,
}

/// An array that has been encoded, along with a description of the encoding
pub struct EncodedArray {
    /// The encoded buffers
    pub buffers: Vec<EncodedBuffer>,
    /// The logical length of the encoded array
    pub num_rows: u32,
}

/// Encodes data from Arrow format into some kind of on-disk format
///
/// The encoder is responsible for looking at the incoming data and determining
/// which encoding is most appropriate.  It then needs to actually encode that
/// data according to the chosen encoding.
pub trait ArrayEncoder: std::fmt::Debug + Send + Sync {
    /// Encode data
    ///
    /// This method may receive multiple arrays and should encode them all into
    /// a single encoded array.
    ///
    /// The result should contain the encoded buffers and a description of the
    /// encoding that was chosen.  This can be used to decode the data later.
    fn encode(&self, arrays: Vec<ArrayRef>) -> Result<EncodedArray>;
}
