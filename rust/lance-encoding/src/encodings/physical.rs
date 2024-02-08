use crate::{decoder::PhysicalPageScheduler, format::pb};

use self::{
    basic::BasicPageScheduler, bitmap::DenseBitmapScheduler, fixed_size_list::FixedListScheduler,
    value::ValuePageScheduler,
};

pub mod basic;
pub mod bitmap;
pub mod fixed_size_list;
pub mod value;

#[derive(Clone, Copy, Debug)]
pub struct FileBuffers<'a> {
    pub positions: &'a Vec<u64>,
}
#[derive(Clone, Copy, Debug)]
pub struct ColumnBuffers<'a, 'b> {
    pub file_buffers: FileBuffers<'a>,
    pub positions: &'b Vec<u64>,
}
#[derive(Clone, Copy, Debug)]
pub struct PageBuffers<'a, 'b, 'c> {
    pub column_buffers: ColumnBuffers<'a, 'b>,
    pub positions: &'c Vec<u64>,
}

fn get_buffer(buffer_desc: &pb::Buffer, buffers: &PageBuffers) -> u64 {
    match pb::buffer::BufferType::try_from(buffer_desc.buffer_type).unwrap() {
        pb::buffer::BufferType::Page => buffers.positions[buffer_desc.buffer_index as usize],
        pb::buffer::BufferType::Column => {
            buffers.column_buffers.positions[buffer_desc.buffer_index as usize]
        }
        pb::buffer::BufferType::File => {
            buffers.column_buffers.file_buffers.positions[buffer_desc.buffer_index as usize]
        }
    }
}

fn decoder_from_buffer_encoding(
    encoding: &pb::BufferEncoding,
    buffers: &PageBuffers,
) -> Box<dyn PhysicalPageScheduler> {
    match encoding.buffer_encoding.as_ref().unwrap() {
        pb::buffer_encoding::BufferEncoding::Value(value) => Box::new(ValuePageScheduler::new(
            value.bytes_per_value,
            get_buffer(value.buffer.as_ref().unwrap(), buffers),
        )),
        pb::buffer_encoding::BufferEncoding::Bitmap(bitmap) => Box::new(DenseBitmapScheduler::new(
            get_buffer(bitmap.buffer.as_ref().unwrap(), buffers),
        )),
    }
}

pub fn decoder_from_array_encoding(
    encoding: &pb::ArrayEncoding,
    buffers: &PageBuffers,
) -> Box<dyn PhysicalPageScheduler> {
    match encoding.array_encoding.as_ref().unwrap() {
        pb::array_encoding::ArrayEncoding::Basic(basic) => {
            match basic.nullability.as_ref().unwrap() {
                pb::basic::Nullability::NoNulls(no_nulls) => {
                    Box::new(BasicPageScheduler::new_non_nullable(
                        decoder_from_buffer_encoding(no_nulls.values.as_ref().unwrap(), buffers),
                    ))
                }
                pb::basic::Nullability::SomeNulls(some_nulls) => {
                    Box::new(BasicPageScheduler::new_nullable(
                        decoder_from_buffer_encoding(
                            some_nulls.validity.as_ref().unwrap(),
                            buffers,
                        ),
                        decoder_from_buffer_encoding(some_nulls.values.as_ref().unwrap(), buffers),
                    ))
                }
                pb::basic::Nullability::AllNulls(_) => todo!(),
            }
        }
        pb::array_encoding::ArrayEncoding::FixedSizeList(fixed_size_list) => {
            let item_encoding = fixed_size_list.items.as_ref().unwrap();
            let item_scheduler = decoder_from_array_encoding(&item_encoding, buffers);
            Box::new(FixedListScheduler::new(
                item_scheduler,
                fixed_size_list.dimension,
            ))
        }
        pb::array_encoding::ArrayEncoding::List(list) => todo!(),
        pb::array_encoding::ArrayEncoding::Struct(strct) => todo!(),
    }
}
