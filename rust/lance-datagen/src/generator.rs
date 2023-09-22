use std::{iter::{TrustedLen, self}, marker::PhantomData, sync::Arc};

use arrow::buffer::{Buffer, OffsetBuffer};
use arrow_array::{RecordBatch, RecordBatchReader, types::{ByteArrayType, BinaryType, Utf8Type, ArrowDictionaryKeyType}};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use lance_arrow::FixedSizeListArrayExt;
use rand::{SeedableRng, RngCore};

#[derive(Copy, Clone, Debug, Default)]
pub struct RowCount(u64);
#[derive(Copy, Clone, Debug, Default)]
pub struct BatchCount(u32);
#[derive(Copy, Clone, Debug, Default)]
pub struct ByteCount(u64);
#[derive(Copy, Clone, Debug, Default)]
pub struct Dimension(u32);

impl From<u32> for BatchCount {
    fn from(n: u32) -> Self {
        Self(n)
    }
}

impl From<u64> for RowCount {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

impl From<u64> for ByteCount {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

impl From<u32> for Dimension {
    fn from(n: u32) -> Self {
        Self(n)
    }
}

pub trait ArrayGenerator {
    fn generate(&mut self, length: RowCount) -> Result<Arc<dyn arrow_array::Array>, ArrowError>;
    fn data_type(&self) -> &DataType;
    fn element_size_bytes(&self) -> Option<ByteCount>;
}

pub struct NTimesIter<I: Iterator>
where
    I::Item: Copy,
{
    iter: I,
    n: u32,
    cur: I::Item,
    count: u32,
}

// Note: if this is used then there is a performance hit as the
// inner loop cannot experience vectorization
//
// TODO: maybe faster to build the vec and then repeat it into
// the destination array?
impl<I: Iterator> Iterator for NTimesIter<I>
where
    I::Item: Copy,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            self.count = self.n - 1;
            self.cur = self.iter.next()?;
        } else {
            self.count -= 1;
        }
        Some(self.cur)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let lower = lower * self.n as usize;
        let upper = upper.map(|u| u * self.n as usize);
        (lower, upper)
    }
}

unsafe impl<I: Iterator> TrustedLen for NTimesIter<I>
where
    I: TrustedLen,
    I::Item: Copy,
{
}

pub struct FnGen<T, ArrayType, F: FnMut() -> T>
where
    T: Copy + Default,
    ArrayType: arrow_array::Array + From<Vec<T>>,
{
    data_type: DataType,
    generator: F,
    array_type: PhantomData<ArrayType>,
    repeat: u32,
    leftover: T,
    leftover_count: u32,
    element_size_bytes: Option<ByteCount>,
}

impl<T, ArrayType, F: FnMut() -> T> FnGen<T, ArrayType, F>
where
    T: Copy + Default,
    ArrayType: arrow_array::Array + From<Vec<T>>,
{
    fn new_known_size(
        data_type: DataType,
        generator: F,
        repeat: u32,
        element_size_bytes: ByteCount,
    ) -> Self {
        Self {
            data_type: data_type,
            generator,
            array_type: PhantomData,
            repeat,
            leftover: T::default(),
            leftover_count: 0,
            element_size_bytes: Some(element_size_bytes),
        }
    }
}

impl<T, ArrayType, F: FnMut() -> T> ArrayGenerator for FnGen<T, ArrayType, F>
where
    T: Copy + Default,
    ArrayType: arrow_array::Array + From<Vec<T>> + 'static,
{
    fn generate(&mut self, length: RowCount) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let iter = (0..length.0).map(|_| (self.generator)());
        let values = if self.repeat > 1 {
            Vec::from_iter(
                NTimesIter {
                    iter,
                    n: self.repeat as u32,
                    cur: self.leftover,
                    count: self.leftover_count,
                }
                .take(length.0 as usize),
            )
        } else {
            Vec::from_iter(iter)
        };
        self.leftover_count = ((self.leftover_count as u64 + length.0) % self.repeat as u64) as u32;
        self.leftover = values.last().copied().unwrap_or(T::default());
        Ok(Arc::new(ArrayType::from(values)))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.element_size_bytes
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Seed(u64);
pub const DEFAULT_SEED: Seed = Seed(42);

impl From<u64> for Seed {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

pub struct CycleVectorGenerator {
    underlying_gen: Box<dyn ArrayGenerator>,
    dimension: Dimension,
    data_type: DataType,
}

impl CycleVectorGenerator {
    pub fn new(underlying_gen: Box<dyn ArrayGenerator>, dimension: Dimension) -> Self {
        let data_type = DataType::FixedSizeList(Arc::new(Field::new("item", underlying_gen.data_type().clone(), true)), dimension.0 as i32);
        Self {
            underlying_gen,
            dimension,
            data_type,
        }
    }
}

impl ArrayGenerator for CycleVectorGenerator {

    fn generate(&mut self, length: RowCount) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let values = self.underlying_gen.generate(RowCount::from(length.0 * self.dimension.0 as u64))?;
        let array = <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
            values,
            self.dimension.0 as i32,
        )?;
        Ok(Arc::new(array))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.underlying_gen.element_size_bytes().map(|byte_count| ByteCount::from(byte_count.0 * self.dimension.0 as u64))
    }
}

pub struct RandomBinaryGenerator {
    rng: rand_xoshiro::Xoshiro256PlusPlus,
    bytes_per_element: ByteCount,
    scale_to_utf8: bool,
    data_type: DataType,
}

impl RandomBinaryGenerator {
    pub fn new(seed: Seed, bytes_per_element: ByteCount, scale_to_utf8: bool) -> Self {
        let rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed.0);
        Self {
            rng,
            bytes_per_element,
            scale_to_utf8,
            data_type: if scale_to_utf8 { Utf8Type::DATA_TYPE.clone() } else { BinaryType::DATA_TYPE.clone() },
        }
    }
}

impl ArrayGenerator for RandomBinaryGenerator {

    fn generate(&mut self, length: RowCount) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let mut bytes = Vec::new();
        bytes.resize((self.bytes_per_element.0 * length.0) as usize, 0);
        self.rng.fill_bytes(&mut bytes);
        if self.scale_to_utf8 {
            // This doesn't give us the full UTF-8 range and it isn't statistically correct but
            // it's fast and probably good enough for most cases
            bytes = bytes.into_iter().map(|val| (val % 95) + 32).collect();
        }
        let bytes = Buffer::from(bytes);
        let offsets = OffsetBuffer::from_lengths(iter::repeat(self.bytes_per_element.0 as usize).take(length.0 as usize));
        if self.scale_to_utf8 {
            Ok(Arc::new(arrow_array::StringArray::new_unchecked(offsets, bytes, None)))
        } else {
            Ok(Arc::new(arrow_array::BinaryArray::new(offsets, bytes, None)))
        }
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        // Not exactly correct since there are N + 1 4-byte offsets and this only counts N
        Some(ByteCount::from(self.bytes_per_element.0 + std::mem::size_of::<i32>() as u64))
    }

}

pub struct FixedBinaryGenerator<T: ByteArrayType> {
    value: Vec<u8>,
    data_type: DataType,
    array_type: PhantomData<T>,
}

impl<T: ByteArrayType> FixedBinaryGenerator<T> {
    pub fn new(value: Vec<u8>) -> Self {
        Self {
            value,
            data_type: T::DATA_TYPE.clone(),
            array_type: PhantomData,
        }
    }
}

impl<T: ByteArrayType> ArrayGenerator for FixedBinaryGenerator<T> {

    fn generate(&mut self, length: RowCount) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let bytes = Buffer::from(Vec::from_iter(self.value.iter().cycle().take((length.0 * self.value.len() as u64) as usize).copied()));
        let offsets = OffsetBuffer::from_lengths(iter::repeat(self.value.len()).take(length.0 as usize));
        Ok(Arc::new(arrow_array::GenericByteArray::<T>::new(offsets, bytes, None)))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        // Not exactly correct since there are N + 1 4-byte offsets and this only counts N
        Some(ByteCount::from(self.value.len() as u64 + std::mem::size_of::<i32>() as u64))
    }
}

pub struct DictionaryGenerator<K: ArrowDictionaryKeyType> {
    generator: Box<dyn ArrayGenerator>,
    data_type: DataType,
    key_type: PhantomData<K>,
    key_width: u64,
}

impl<K: ArrowDictionaryKeyType> DictionaryGenerator<K> {
    fn new(generator: Box<dyn ArrayGenerator>) -> Self {
        let key_type = Box::new(K::DATA_TYPE.clone());
        let key_width = key_type.primitive_width().expect("dictionary key types should have a known width") as u64;
        let val_type = Box::new(generator.data_type().clone());
        let dict_type = DataType::Dictionary(key_type, val_type);
        Self {
            generator,
            data_type: dict_type,
            key_type: PhantomData,
            key_width,
        }
    }
}

impl<K: ArrowDictionaryKeyType> ArrayGenerator for DictionaryGenerator<K> {

    fn generate(&mut self, length: RowCount) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let underlying = self.generator.generate(length)?;
        arrow_cast::cast::cast(&underlying, &self.data_type)
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.generator.element_size_bytes().map(|size_bytes| ByteCount::from(size_bytes.0 + self.key_width))
    }

}

pub struct FixedSizeBatchGenerator {
    generators: Vec<(Option<String>, Box<dyn ArrayGenerator>)>,
    batch_size: RowCount,
    num_batches: BatchCount,
    schema: SchemaRef,
}

impl FixedSizeBatchGenerator {
    fn new(
        generators: Vec<(Option<String>, Box<dyn ArrayGenerator>)>,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> Self {
        let mut fields = Vec::with_capacity(generators.len());
        for (field_index, field_gen) in generators.iter().enumerate() {
            let (name, gen) = field_gen;
            let default_name = format!("field_{}", field_index);
            let name = name.clone().unwrap_or(default_name);
            fields.push(Field::new(name, gen.data_type().clone(), true));
        }
        let schema = Arc::new(Schema::new(fields));
        Self {
            generators,
            batch_size,
            num_batches,
            schema,
        }
    }

    fn gen_next(&mut self) -> Result<RecordBatch, ArrowError> {
        let mut arrays = Vec::with_capacity(self.generators.len());
        for field_gen in self.generators.iter_mut() {
            let (_, gen) = field_gen;
            let arr = gen.generate(self.batch_size)?;
            arrays.push(arr);
        }
        self.num_batches.0 -= 1;
        Ok(
            RecordBatch::try_new(self.schema.clone(), arrays).unwrap()
        )
    }
}

impl Iterator for FixedSizeBatchGenerator {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_batches.0 == 0 {
            return None;
        }
        Some(self.gen_next())
    }
}

impl RecordBatchReader for FixedSizeBatchGenerator {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[derive(Default)]
pub struct BatchGeneratorBuilder {
    generators: Vec<(Option<String>, Box<dyn ArrayGenerator>)>,
}

pub enum RoundingBehavior {
    ExactOrErr,
    RoundUp,
    RoundDown,
}

impl BatchGeneratorBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn col(mut self, name: Option<String>, gen: Box<dyn ArrayGenerator>) -> Self {
        self.generators.push((name, gen));
        self
    }

    /// Create a RecordBatchReader that generates batches of the given size
    pub fn into_reader_rows(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> impl RecordBatchReader {
        FixedSizeBatchGenerator::new(self.generators, batch_size, num_batches)
    }

    pub fn into_reader_bytes(
        self,
        batch_size_bytes: ByteCount,
        num_batches: BatchCount,
        rounding: RoundingBehavior,
    ) -> Result<impl RecordBatchReader, ArrowError> {
        let bytes_per_row = self
            .generators
            .iter()
            .map(|gen| 
                gen.1.element_size_bytes().map(|byte_count| byte_count.0).ok_or(
                        ArrowError::NotYetImplemented("The function into_reader_bytes currently requires each array generator to have a fixed element size".to_string())
                )
            )
            .sum::<Result<u64, ArrowError>>()?;
        let mut num_rows = RowCount::from(batch_size_bytes.0 / bytes_per_row);
        if batch_size_bytes.0 % bytes_per_row != 0 {
            match rounding {
                RoundingBehavior::ExactOrErr => {
                    return Err(ArrowError::NotYetImplemented(
                        format!("Exact rounding requested but not possible.  Batch size requested {}, row size: {}", batch_size_bytes.0, bytes_per_row))
                    );
                }
                RoundingBehavior::RoundUp => {
                    num_rows = RowCount::from(num_rows.0 + 1);
                }
                RoundingBehavior::RoundDown => (),
            }
        }
        Ok(self.into_reader_rows(num_rows, num_batches))
    }
}

pub mod array {
    use arrow_array::{ArrowNativeTypeOp, PrimitiveArray};
    use arrow_array::types::{ArrowPrimitiveType, Utf8Type, Int8Type, Int16Type, Int32Type, Int64Type, Float32Type, Float64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type};
    use rand::Rng;

    use super::*;

    pub fn cycle_vec(generator: Box<dyn ArrayGenerator>, dimension: Dimension) -> Box<dyn ArrayGenerator> {
        Box::new(CycleVectorGenerator::new(generator, dimension))
    }

    pub fn step<DataType>() -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + Default + std::ops::AddAssign<DataType::Native> + 'static,
        DataType: ArrowPrimitiveType,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
    {
        let mut x = DataType::Native::default();
        Box::new(FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
            DataType::DATA_TYPE.clone(),
            move || {
                let y = x;
                x += DataType::Native::from(DataType::Native::ONE);
                y
            },
            1,
            DataType::DATA_TYPE.primitive_width().map(|width| ByteCount::from(width as u64)).expect("Primitive types should have a fixed width"),
        ))
    }

    pub fn step_custom<DataType>(
        start: DataType::Native,
        step: DataType::Native,
    ) -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + Default + std::ops::AddAssign<DataType::Native> + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
    {
        let mut x = start;
        Box::new(FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
            DataType::DATA_TYPE.clone(),
            move || {
                let y = x;
                x += step;
                y
            },
            1,
            DataType::DATA_TYPE.primitive_width().map(|width| ByteCount::from(width as u64)).expect("Primitive types should have a fixed width"),
        ))
    }

    pub fn fill<DataType>(value: DataType::Native) -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + 'static,
        DataType: ArrowPrimitiveType,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static
    {
        Box::new(FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
            DataType::DATA_TYPE.clone(),
            move || value,
            1,
            DataType::DATA_TYPE.primitive_width().map(|width| ByteCount::from(width as u64)).expect("Primitive types should have a fixed width"),
        ))
    }

    pub fn fill_varbin(value: Vec<u8>) -> Box<dyn ArrayGenerator> {
        Box::new(FixedBinaryGenerator::<BinaryType>::new(value))
    }

    pub fn fill_utf8(value: String) -> Box<dyn ArrayGenerator> {
        Box::new(FixedBinaryGenerator::<Utf8Type>::new(value.into_bytes()))
    }

    pub fn rand<DataType>(seed: Seed) -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
        rand::distributions::Standard: rand::distributions::Distribution<DataType::Native>,
    {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed.0);
        Box::new(FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
            DataType::DATA_TYPE.clone(),
            move || rng.gen(),
            1,
            DataType::DATA_TYPE.primitive_width().map(|width| ByteCount::from(width as u64)).expect("Primitive types should have a fixed width"),
        ))
    }

    pub fn rand_vec<DataType>(seed: Seed, dimension: Dimension) -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
        rand::distributions::Standard: rand::distributions::Distribution<DataType::Native>
        {
            let underlying = rand::<DataType>(seed);
            cycle_vec(underlying, dimension)
        }

    pub fn rand_varbin(seed: Seed, bytes_per_element: ByteCount) -> Box<dyn ArrayGenerator> {
        Box::new(RandomBinaryGenerator::new(seed, bytes_per_element, false))
    }

    pub fn rand_utf8(seed: Seed, bytes_per_element: ByteCount) -> Box<dyn ArrayGenerator> {
        Box::new(RandomBinaryGenerator::new(seed, bytes_per_element, true))
    }

    pub fn rand_type(seed: Seed, data_type: &DataType) -> Box<dyn ArrayGenerator> {
        match data_type {
            DataType::Int8 => rand::<Int8Type>(seed),
            DataType::Int16 => rand::<Int16Type>(seed),
            DataType::Int32 => rand::<Int32Type>(seed),
            DataType::Int64 => rand::<Int64Type>(seed),
            DataType::UInt8 => rand::<UInt8Type>(seed),
            DataType::UInt16 => rand::<UInt16Type>(seed),
            DataType::UInt32 => rand::<UInt32Type>(seed),
            DataType::UInt64 => rand::<UInt64Type>(seed),
            DataType::Float32 => rand::<Float32Type>(seed),
            DataType::Float64 => rand::<Float64Type>(seed),
            DataType::Utf8 => rand_utf8(seed, ByteCount::from(12)),
            DataType::Binary => rand_varbin(seed, ByteCount::from(12)),
            DataType::Dictionary(key_type, value_type) => dict_type(rand_type(seed, value_type), key_type),
            _ => unimplemented!(),
        }
    }

    pub fn dict<K: ArrowDictionaryKeyType>(generator: Box<dyn ArrayGenerator>) -> Box<dyn ArrayGenerator> {
        Box::new(DictionaryGenerator::<K>::new(generator))
    }

    pub fn dict_type(generator: Box<dyn ArrayGenerator>, key_type: &DataType) -> Box<dyn ArrayGenerator> {
        match key_type {
            DataType::Int8 => dict::<Int8Type>(generator),
            DataType::Int16 => dict::<Int16Type>(generator),
            DataType::Int32 => dict::<Int32Type>(generator),
            DataType::Int64 => dict::<Int64Type>(generator),
            DataType::UInt8 => dict::<UInt8Type>(generator),
            DataType::UInt16 => dict::<UInt16Type>(generator),
            DataType::UInt32 => dict::<UInt32Type>(generator),
            DataType::UInt64 => dict::<UInt64Type>(generator),
            _ => unimplemented!(),
        }
    }
}

pub fn gen() -> BatchGeneratorBuilder {
    BatchGeneratorBuilder::default()
}

pub fn rand(schema: &Schema) -> BatchGeneratorBuilder {
    let mut builder = BatchGeneratorBuilder::default();
    for field in schema.fields() {
        builder = builder.col(Some(field.name().clone()), array::rand_type(DEFAULT_SEED, field.data_type()));
    }
    builder
}

#[cfg(test)]
mod tests {

    use arrow_array::{
        types::{Float32Type, Int16Type, Int32Type, Int8Type},
        Float32Array, Int16Array, Int32Array, Int8Array,
    };

    use super::*;

    #[test]
    fn test_step() {
        let mut gen = array::step::<Int32Type>();
        assert_eq!(
            *gen.generate(RowCount::from(5)).unwrap(),
            Int32Array::from_iter([0, 1, 2, 3, 4])
        );
        assert_eq!(
            *gen.generate(RowCount::from(5)).unwrap(),
            Int32Array::from_iter([5, 6, 7, 8, 9])
        );

        let mut gen = array::step::<Int8Type>();
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            Int8Array::from_iter([0, 1, 2])
        );

        let mut gen = array::step::<Float32Type>();
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            Float32Array::from_iter([0.0, 1.0, 2.0])
        );

        let mut gen = array::step_custom::<Int16Type>(4, 8);
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            Int16Array::from_iter([4, 12, 20])
        );
        assert_eq!(
            *gen.generate(RowCount::from(2)).unwrap(),
            Int16Array::from_iter([28, 36])
        );
    }

    #[test]
    fn test_fill() {
        let mut gen = array::fill::<Int32Type>(42);
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            Int32Array::from_iter([42, 42, 42])
        );
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            Int32Array::from_iter([42, 42, 42])
        );

        let mut gen = array::fill_varbin(vec![0, 1, 2]);
        assert_eq!(*gen.generate(RowCount::from(3)).unwrap(),
            arrow_array::BinaryArray::from_iter_values(["\x00\x01\x02", "\x00\x01\x02", "\x00\x01\x02"])
        );

        let mut gen = array::fill_utf8("xyz".to_string());
        assert_eq!(*gen.generate(RowCount::from(3)).unwrap(),
            arrow_array::StringArray::from_iter_values(["xyz", "xyz", "xyz"])
        );
    }

    #[test]
    fn test_rng() {
        let mut gen = array::rand::<Int32Type>(DEFAULT_SEED);
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            Int32Array::from_iter([-797553329, 1369325940, -69174021])
        );

        let mut gen = array::rand_varbin(DEFAULT_SEED, ByteCount::from(3));
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            arrow_array::BinaryArray::from_iter_values([[159, 104, 118], [68, 79, 77], [118, 208, 116]]));

        let mut gen = array::rand_utf8(DEFAULT_SEED, ByteCount::from(3));
        assert_eq!(
            *gen.generate(RowCount::from(3)).unwrap(),
            arrow_array::StringArray::from_iter_values(["`)7", "dom", "725"]));
    }

    #[test]
    fn test_rand_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Float32, true),
            Field::new("d", DataType::Int32, true),
            Field::new("e", DataType::Int32, true),
        ]);
        let rbr = rand(&schema).into_reader_bytes(ByteCount::from(1024*1024), BatchCount::from(8), RoundingBehavior::ExactOrErr).unwrap();
        assert_eq!(*rbr.schema(), schema);

        let batches = rbr.map(|val| val.unwrap()).collect::<Vec<_>>();
        assert_eq!(batches.len(), 8);

        for batch in batches {
            assert_eq!(batch.num_rows(), 1024*1024/32);
            assert_eq!(batch.num_columns(), 5);
        }
    }
}
