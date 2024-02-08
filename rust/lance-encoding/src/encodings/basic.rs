use crate::decoder::{DataDecoder, PageDecoder, Scheduled};

#[derive(Debug)]
enum DataValidity {
    NoNull,
    SomeNull(Box<dyn DataDecoder>),
}

#[derive(Debug)]
struct BasicDataDecoder {
    validity: DataValidity,
    values: Box<dyn DataDecoder>,
    num_validity_buffers: usize,
}

impl DataDecoder for BasicDataDecoder {
    fn update_capacity(
        &self,
        data: &[Vec<bytes::Bytes>],
        rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
    ) {
        // No need to look at the validity decoder to know the dest buffer size since it is boolean
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        buffers[0].1 = match self.validity {
            DataValidity::NoNull => false,
            DataValidity::SomeNull(_) => true,
        };
        self.values.update_capacity(
            &data[self.num_validity_buffers..],
            rows_to_skip,
            num_rows,
            &mut buffers[1..],
        );
    }

    fn drain(
        &self,
        // TODO: Coalesce on read and change this to &[Bytes]
        data: &[Vec<bytes::Bytes>],
        rows_to_skip: u32,
        num_rows: u32,
        dest_offset: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) {
        if let DataValidity::SomeNull(validity_decoder) = &self.validity {
            validity_decoder.drain(
                &data[..self.num_validity_buffers],
                rows_to_skip,
                num_rows,
                dest_offset,
                &mut dest_buffers[..1],
            );
        }
        self.values.drain(
            &data[self.num_validity_buffers..],
            rows_to_skip,
            num_rows,
            dest_offset,
            &mut dest_buffers[1..],
        );
    }
}

enum PageValidity {
    NoNull,
    SomeNull(Box<dyn PageDecoder>),
}

pub struct BasicDecoder {
    validity_decoder: PageValidity,
    values_decoder: Box<dyn PageDecoder>,
    num_validity_buffers: usize,
}

impl BasicDecoder {
    pub fn new_nullable(
        validity_decoder: Box<dyn PageDecoder>,
        values_decoder: Box<dyn PageDecoder>,
    ) -> Self {
        let num_validity_buffers = validity_decoder.num_buffers() as usize;
        Self {
            validity_decoder: PageValidity::SomeNull(validity_decoder),
            values_decoder,
            num_validity_buffers,
        }
    }

    pub fn new_non_nullable(values_decoder: Box<dyn PageDecoder>) -> Self {
        Self {
            validity_decoder: PageValidity::NoNull,
            values_decoder,
            num_validity_buffers: 0,
        }
    }

    fn combine(&self, validity: Option<Scheduled>, values: Scheduled) -> Scheduled {
        let mut all_reqs = values.batch_requests;
        if let Some(validity) = validity {
            all_reqs.extend(validity.batch_requests);
            let decoder = Box::new(BasicDataDecoder {
                validity: DataValidity::SomeNull(validity.data_decoder),
                values: values.data_decoder,
                num_validity_buffers: self.num_validity_buffers as usize,
            });
            Scheduled::new(all_reqs, decoder)
        } else {
            let decoder = Box::new(BasicDataDecoder {
                validity: DataValidity::NoNull,
                values: values.data_decoder,
                num_validity_buffers: 0,
            });
            Scheduled::new(all_reqs, decoder)
        }
    }
}

impl PageDecoder for BasicDecoder {
    fn schedule_range(&self, range: std::ops::Range<u32>) -> Scheduled {
        let validity = match &self.validity_decoder {
            PageValidity::NoNull => None,
            PageValidity::SomeNull(validity_decoder) => {
                Some(validity_decoder.schedule_range(range.clone()))
            }
        };
        let values = self.values_decoder.schedule_range(range);
        self.combine(validity, values)
    }

    fn schedule_take(
        &self,
        indices: arrow_array::PrimitiveArray<arrow_array::types::UInt32Type>,
    ) -> crate::decoder::Scheduled {
        let validity = match &self.validity_decoder {
            PageValidity::NoNull => None,
            PageValidity::SomeNull(validity_decoder) => {
                Some(validity_decoder.schedule_take(indices.clone()))
            }
        };
        let values = self.values_decoder.schedule_take(indices);
        self.combine(validity, values)
    }

    fn num_buffers(&self) -> u32 {
        self.num_validity_buffers as u32 + self.values_decoder.num_buffers()
    }
}
