// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! YAML-based configuration parser for data generation schemas.
//!
//! This module provides a declarative way to define data generation schemas using YAML files.
//! It supports all generator types available in the [`crate::generator::array`] module.
//!
//! # Example
//!
//! ```
//! # use lance_datagen::yaml::from_yaml;
//! # use lance_datagen::RowCount;
//! let yaml = r#"
//! seed: 42
//! columns:
//!   - name: id
//!     generator: step
//!     type: int64
//!   - name: score
//!     generator: rand
//!     type: float32
//!     null_probability: 0.1
//! "#;
//! let builder = from_yaml(yaml).unwrap();
//! let batch = builder.into_batch_rows(RowCount::from(100)).unwrap();
//! assert_eq!(batch.num_rows(), 100);
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, IntervalUnit, TimeUnit};
use serde::Deserialize;

use crate::generator::{array, ArrayGenerator, ArrayGeneratorExt, BatchGeneratorBuilder, Seed};

/// Error type for YAML parsing and configuration validation
#[derive(Debug)]
pub enum YamlError {
    /// Error parsing YAML syntax
    Parse(serde_yaml::Error),
    /// Invalid Arrow data type string
    InvalidType(String),
    /// Invalid generator configuration
    InvalidGenerator(String),
    /// Invalid value for a specific field
    InvalidValue { field: String, message: String },
    /// IO error reading YAML file
    Io(std::io::Error),
}

impl std::fmt::Display for YamlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "YAML parse error: {}", e),
            Self::InvalidType(t) => write!(f, "Invalid data type: {}", t),
            Self::InvalidGenerator(g) => write!(f, "Invalid generator: {}", g),
            Self::InvalidValue { field, message } => {
                write!(f, "Invalid value for '{}': {}", field, message)
            }
            Self::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for YamlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_yaml::Error> for YamlError {
    fn from(e: serde_yaml::Error) -> Self {
        Self::Parse(e)
    }
}

impl From<std::io::Error> for YamlError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Top-level configuration for data generation
#[derive(Debug, Deserialize)]
pub struct DatagenConfig {
    /// Optional global RNG seed
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optional default null probability applied to all columns
    #[serde(default)]
    pub default_null_probability: Option<f64>,
    /// Column definitions
    pub columns: Vec<ColumnConfig>,
}

/// Configuration for a single column
#[derive(Debug, Deserialize)]
pub struct ColumnConfig {
    /// Column name
    pub name: String,
    /// Generator configuration
    #[serde(flatten)]
    pub generator: GeneratorConfig,
    /// Optional per-column null probability (overrides default)
    #[serde(default)]
    pub null_probability: Option<f64>,
    /// Optional modifiers for the generated data
    #[serde(default)]
    pub modifiers: Option<ModifiersConfig>,
}

/// Configuration for a struct field (used in rand_struct)
#[derive(Debug, Deserialize)]
pub struct StructFieldConfig {
    /// Field name
    pub name: String,
    /// Field data type
    #[serde(rename = "type")]
    pub data_type: String,
    /// Whether the field is nullable
    #[serde(default = "default_true")]
    pub nullable: bool,
}

fn default_true() -> bool {
    true
}

/// Generator configuration variants
#[derive(Debug, Deserialize)]
#[serde(tag = "generator", rename_all = "snake_case")]
pub enum GeneratorConfig {
    // Primitives
    /// Random values of the specified type
    Rand {
        #[serde(rename = "type")]
        data_type: String,
    },
    /// Fill with a constant value
    Fill {
        #[serde(rename = "type")]
        data_type: String,
        value: serde_yaml::Value,
    },
    /// Cycle through a list of values
    Cycle {
        #[serde(rename = "type")]
        data_type: String,
        values: Vec<serde_yaml::Value>,
    },
    /// Step through values starting from 0
    Step {
        #[serde(rename = "type")]
        data_type: String,
    },
    /// Step through values with custom start and step
    StepCustom {
        #[serde(rename = "type")]
        data_type: String,
        start: serde_yaml::Value,
        step: serde_yaml::Value,
    },

    // Strings
    /// Fill with a constant UTF-8 string
    FillUtf8 { value: String },
    /// Cycle through a list of UTF-8 strings
    CycleUtf8 { values: Vec<String> },
    /// Random UTF-8 strings with fixed byte length
    RandUtf8 {
        bytes_per_element: u64,
        #[serde(default)]
        is_large: bool,
    },
    /// Random sentences with word count in range
    RandomSentence {
        min_words: usize,
        max_words: usize,
        #[serde(default)]
        is_large: bool,
    },
    /// Random single words
    RandomWord {
        #[serde(default)]
        is_large: bool,
    },

    // Vectors
    /// Random fixed-size vectors
    RandVec {
        #[serde(rename = "type")]
        element_type: String,
        dimension: u32,
    },
    /// Cycle vectors from an inner generator
    CycleVec {
        inner: Box<GeneratorConfig>,
        dimension: u32,
    },
    /// Variable-length lists from an inner generator
    CycleVecVar {
        inner: Box<GeneratorConfig>,
        min_size: u32,
        max_size: u32,
    },

    // Lists/Maps/Structs
    /// Random lists with items of the specified type
    RandList {
        item_type: String,
        #[serde(default)]
        is_large: bool,
    },
    /// Random maps with keys and values of specified types
    RandMap {
        key_type: String,
        value_type: String,
    },
    /// Random structs with specified fields
    RandStruct { fields: Vec<StructFieldConfig> },

    // Wrappers
    /// Dictionary-encode the inner generator
    Dict {
        key_type: String,
        inner: Box<GeneratorConfig>,
    },
    /// Limit unique values from the inner generator
    LowCardinality {
        cardinality: usize,
        inner: Box<GeneratorConfig>,
    },

    // Temporal
    /// Random timestamps with specified unit and timezone
    RandTimestamp {
        unit: String,
        #[serde(default)]
        timezone: Option<String>,
    },
    /// Random date32 values
    RandDate32,
    /// Random date64 values
    RandDate64,
    /// Random time32 values with specified unit
    RandTime32 { unit: String },
    /// Random time64 values with specified unit
    RandTime64 { unit: String },

    // Special
    /// Random boolean values
    RandBoolean,
    /// Random pseudo-UUID (16-byte binary)
    RandPseudoUuid,
    /// Random pseudo-UUID as hex string
    RandPseudoUuidHex,
    /// Blob data (4MB fixed-size binary with metadata)
    Blob,

    // Type-aware
    /// Random values based on Arrow data type (convenience wrapper)
    RandType {
        #[serde(rename = "type")]
        data_type: String,
    },
}

/// Optional modifiers for generated data
#[derive(Debug, Default, Deserialize)]
pub struct ModifiersConfig {
    /// Cyclic NaN pattern for float types
    #[serde(default)]
    pub nans: Option<Vec<bool>>,
    /// Cyclic null pattern (alternative to null_probability)
    #[serde(default)]
    pub nulls: Option<Vec<bool>>,
    /// Field metadata
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

/// Parse a YAML string into a DatagenConfig
pub fn parse_yaml(yaml: &str) -> Result<DatagenConfig, YamlError> {
    Ok(serde_yaml::from_str(yaml)?)
}

/// Parse a YAML file into a DatagenConfig
pub fn parse_yaml_file(path: &Path) -> Result<DatagenConfig, YamlError> {
    let content = std::fs::read_to_string(path)?;
    parse_yaml(&content)
}

/// Convert a DatagenConfig into a BatchGeneratorBuilder
pub fn config_to_builder(config: DatagenConfig) -> Result<BatchGeneratorBuilder, YamlError> {
    let mut builder = if let Some(seed) = config.seed {
        BatchGeneratorBuilder::new_with_seed(Seed(seed))
    } else {
        BatchGeneratorBuilder::new()
    };

    if let Some(null_prob) = config.default_null_probability {
        builder.with_random_nulls(null_prob);
    }

    for col in config.columns {
        let mut datagen = build_generator(&col.generator)?;

        // Apply modifiers
        if let Some(modifiers) = &col.modifiers {
            if let Some(nans) = &modifiers.nans {
                datagen = datagen.with_nans(nans);
            }
            if let Some(nulls) = &modifiers.nulls {
                datagen = datagen.with_nulls(nulls);
            }
            if let Some(metadata) = &modifiers.metadata {
                datagen = datagen.with_metadata(metadata.clone());
            }
        }

        // Apply per-column null probability
        if let Some(null_prob) = col.null_probability {
            datagen = datagen.with_random_nulls(null_prob);
        }

        builder = builder.col(col.name, datagen);
    }

    Ok(builder)
}

/// Convenience function to parse YAML and build a BatchGeneratorBuilder
pub fn from_yaml(yaml: &str) -> Result<BatchGeneratorBuilder, YamlError> {
    let config = parse_yaml(yaml)?;
    config_to_builder(config)
}

/// Convenience function to parse a YAML file and build a BatchGeneratorBuilder
pub fn from_yaml_file(path: &Path) -> Result<BatchGeneratorBuilder, YamlError> {
    let config = parse_yaml_file(path)?;
    config_to_builder(config)
}

/// Parse a data type string into an Arrow DataType
fn parse_data_type(type_str: &str) -> Result<DataType, YamlError> {
    let type_str = type_str.trim();

    // Simple types
    match type_str.to_lowercase().as_str() {
        "boolean" | "bool" => return Ok(DataType::Boolean),
        "int8" => return Ok(DataType::Int8),
        "int16" => return Ok(DataType::Int16),
        "int32" => return Ok(DataType::Int32),
        "int64" => return Ok(DataType::Int64),
        "uint8" => return Ok(DataType::UInt8),
        "uint16" => return Ok(DataType::UInt16),
        "uint32" => return Ok(DataType::UInt32),
        "uint64" => return Ok(DataType::UInt64),
        "float16" => return Ok(DataType::Float16),
        "float32" => return Ok(DataType::Float32),
        "float64" => return Ok(DataType::Float64),
        "utf8" | "string" => return Ok(DataType::Utf8),
        "large_utf8" | "large_string" => return Ok(DataType::LargeUtf8),
        "binary" => return Ok(DataType::Binary),
        "large_binary" => return Ok(DataType::LargeBinary),
        "date32" => return Ok(DataType::Date32),
        "date64" => return Ok(DataType::Date64),
        "null" => return Ok(DataType::Null),
        _ => {}
    }

    // Complex types with parameters
    // Decimal128(precision, scale)
    if let Some(params) = type_str
        .strip_prefix("Decimal128(")
        .or_else(|| type_str.strip_prefix("decimal128("))
    {
        let params = params
            .strip_suffix(')')
            .ok_or_else(|| YamlError::InvalidType(type_str.to_string()))?;
        let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(YamlError::InvalidType(type_str.to_string()));
        }
        let precision: u8 = parts[0]
            .parse()
            .map_err(|_| YamlError::InvalidType(type_str.to_string()))?;
        let scale: i8 = parts[1]
            .parse()
            .map_err(|_| YamlError::InvalidType(type_str.to_string()))?;
        return Ok(DataType::Decimal128(precision, scale));
    }

    // Decimal256(precision, scale)
    if let Some(params) = type_str
        .strip_prefix("Decimal256(")
        .or_else(|| type_str.strip_prefix("decimal256("))
    {
        let params = params
            .strip_suffix(')')
            .ok_or_else(|| YamlError::InvalidType(type_str.to_string()))?;
        let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(YamlError::InvalidType(type_str.to_string()));
        }
        let precision: u8 = parts[0]
            .parse()
            .map_err(|_| YamlError::InvalidType(type_str.to_string()))?;
        let scale: i8 = parts[1]
            .parse()
            .map_err(|_| YamlError::InvalidType(type_str.to_string()))?;
        return Ok(DataType::Decimal256(precision, scale));
    }

    // FixedSizeBinary(size)
    if let Some(params) = type_str
        .strip_prefix("FixedSizeBinary(")
        .or_else(|| type_str.strip_prefix("fixed_size_binary("))
    {
        let params = params
            .strip_suffix(')')
            .ok_or_else(|| YamlError::InvalidType(type_str.to_string()))?;
        let size: i32 = params
            .trim()
            .parse()
            .map_err(|_| YamlError::InvalidType(type_str.to_string()))?;
        return Ok(DataType::FixedSizeBinary(size));
    }

    // Timestamp(unit, timezone) or Timestamp(unit)
    if let Some(params) = type_str
        .strip_prefix("Timestamp(")
        .or_else(|| type_str.strip_prefix("timestamp("))
    {
        let params = params
            .strip_suffix(')')
            .ok_or_else(|| YamlError::InvalidType(type_str.to_string()))?;
        let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();
        if parts.is_empty() || parts.len() > 2 {
            return Err(YamlError::InvalidType(type_str.to_string()));
        }
        let unit = parse_time_unit(parts[0])?;
        let tz = if parts.len() == 2 {
            let tz_str = parts[1].trim_matches('"').trim_matches('\'');
            if tz_str.is_empty() {
                None
            } else {
                Some(Arc::from(tz_str))
            }
        } else {
            None
        };
        return Ok(DataType::Timestamp(unit, tz));
    }

    // Duration(unit)
    if let Some(params) = type_str
        .strip_prefix("Duration(")
        .or_else(|| type_str.strip_prefix("duration("))
    {
        let params = params
            .strip_suffix(')')
            .ok_or_else(|| YamlError::InvalidType(type_str.to_string()))?;
        let unit = parse_time_unit(params.trim())?;
        return Ok(DataType::Duration(unit));
    }

    // Time32(unit)
    if let Some(params) = type_str
        .strip_prefix("Time32(")
        .or_else(|| type_str.strip_prefix("time32("))
    {
        let params = params
            .strip_suffix(')')
            .ok_or_else(|| YamlError::InvalidType(type_str.to_string()))?;
        let unit = parse_time_unit(params.trim())?;
        return Ok(DataType::Time32(unit));
    }

    // Time64(unit)
    if let Some(params) = type_str
        .strip_prefix("Time64(")
        .or_else(|| type_str.strip_prefix("time64("))
    {
        let params = params
            .strip_suffix(')')
            .ok_or_else(|| YamlError::InvalidType(type_str.to_string()))?;
        let unit = parse_time_unit(params.trim())?;
        return Ok(DataType::Time64(unit));
    }

    Err(YamlError::InvalidType(type_str.to_string()))
}

/// Parse a time unit string
fn parse_time_unit(unit_str: &str) -> Result<TimeUnit, YamlError> {
    match unit_str.to_lowercase().as_str() {
        "second" | "s" => Ok(TimeUnit::Second),
        "millisecond" | "ms" => Ok(TimeUnit::Millisecond),
        "microsecond" | "us" => Ok(TimeUnit::Microsecond),
        "nanosecond" | "ns" => Ok(TimeUnit::Nanosecond),
        _ => Err(YamlError::InvalidValue {
            field: "unit".to_string(),
            message: format!("Unknown time unit: {}", unit_str),
        }),
    }
}

/// Parse an interval unit string
#[allow(dead_code)]
fn parse_interval_unit(unit_str: &str) -> Result<IntervalUnit, YamlError> {
    match unit_str.to_lowercase().as_str() {
        "year_month" | "yearmonth" => Ok(IntervalUnit::YearMonth),
        "day_time" | "daytime" => Ok(IntervalUnit::DayTime),
        "month_day_nano" | "monthdaynano" => Ok(IntervalUnit::MonthDayNano),
        _ => Err(YamlError::InvalidValue {
            field: "unit".to_string(),
            message: format!("Unknown interval unit: {}", unit_str),
        }),
    }
}

/// Build a generator from a GeneratorConfig
fn build_generator(config: &GeneratorConfig) -> Result<Box<dyn ArrayGenerator>, YamlError> {
    match config {
        GeneratorConfig::Rand { data_type } => {
            let arrow_type = parse_data_type(data_type)?;
            Ok(array::rand_type(&arrow_type))
        }

        GeneratorConfig::Fill { data_type, value } => build_fill_generator(data_type, value),

        GeneratorConfig::Cycle { data_type, values } => build_cycle_generator(data_type, values),

        GeneratorConfig::Step { data_type } => build_step_generator(data_type),

        GeneratorConfig::StepCustom {
            data_type,
            start,
            step,
        } => build_step_custom_generator(data_type, start, step),

        GeneratorConfig::FillUtf8 { value } => Ok(array::fill_utf8(value.clone())),

        GeneratorConfig::CycleUtf8 { values } => {
            let static_strs: Vec<&'static str> = values
                .iter()
                .map(|s| Box::leak(s.clone().into_boxed_str()) as &'static str)
                .collect();
            Ok(array::cycle_utf8_literals(&static_strs))
        }

        GeneratorConfig::RandUtf8 {
            bytes_per_element,
            is_large,
        } => Ok(array::rand_utf8(
            crate::ByteCount::from(*bytes_per_element),
            *is_large,
        )),

        GeneratorConfig::RandomSentence {
            min_words,
            max_words,
            is_large,
        } => Ok(array::random_sentence(*min_words, *max_words, *is_large)),

        GeneratorConfig::RandomWord { is_large } => Ok(array::random_word(*is_large)),

        GeneratorConfig::RandVec {
            element_type,
            dimension,
        } => build_rand_vec_generator(element_type, *dimension),

        GeneratorConfig::CycleVec { inner, dimension } => {
            let inner_gen = build_generator(inner)?;
            Ok(array::cycle_vec(
                inner_gen,
                crate::Dimension::from(*dimension),
            ))
        }

        GeneratorConfig::CycleVecVar {
            inner,
            min_size,
            max_size,
        } => {
            let inner_gen = build_generator(inner)?;
            Ok(array::cycle_vec_var(
                inner_gen,
                crate::Dimension::from(*min_size),
                crate::Dimension::from(*max_size),
            ))
        }

        GeneratorConfig::RandList {
            item_type,
            is_large,
        } => {
            let arrow_type = parse_data_type(item_type)?;
            Ok(array::rand_list(&arrow_type, *is_large))
        }

        GeneratorConfig::RandMap {
            key_type,
            value_type,
        } => {
            let key_arrow = parse_data_type(key_type)?;
            let value_arrow = parse_data_type(value_type)?;
            Ok(array::rand_map(&key_arrow, &value_arrow))
        }

        GeneratorConfig::RandStruct { fields } => {
            let arrow_fields: Vec<Field> = fields
                .iter()
                .map(|f| {
                    let dt = parse_data_type(&f.data_type)?;
                    Ok(Field::new(&f.name, dt, f.nullable))
                })
                .collect::<Result<Vec<_>, YamlError>>()?;
            Ok(array::rand_struct(Fields::from(arrow_fields)))
        }

        GeneratorConfig::Dict { key_type, inner } => {
            let inner_gen = build_generator(inner)?;
            let key_arrow = parse_data_type(key_type)?;
            Ok(array::dict_type(inner_gen, &key_arrow))
        }

        GeneratorConfig::LowCardinality { cardinality, inner } => {
            let inner_gen = build_generator(inner)?;
            Ok(array::low_cardinality(inner_gen, *cardinality))
        }

        GeneratorConfig::RandTimestamp { unit, timezone } => {
            let time_unit = parse_time_unit(unit)?;
            let tz = timezone.as_ref().map(|s| Arc::from(s.as_str()));
            let data_type = DataType::Timestamp(time_unit, tz);
            Ok(array::rand_timestamp(&data_type))
        }

        GeneratorConfig::RandDate32 => Ok(array::rand_date32()),

        GeneratorConfig::RandDate64 => Ok(array::rand_date64()),

        GeneratorConfig::RandTime32 { unit } => {
            let time_unit = parse_time_unit(unit)?;
            Ok(array::rand_time32(&time_unit))
        }

        GeneratorConfig::RandTime64 { unit } => {
            let time_unit = parse_time_unit(unit)?;
            Ok(array::rand_time64(&time_unit))
        }

        GeneratorConfig::RandBoolean => Ok(array::rand_boolean()),

        GeneratorConfig::RandPseudoUuid => Ok(array::rand_pseudo_uuid()),

        GeneratorConfig::RandPseudoUuidHex => Ok(array::rand_pseudo_uuid_hex()),

        GeneratorConfig::Blob => Ok(array::blob()),

        GeneratorConfig::RandType { data_type } => {
            let arrow_type = parse_data_type(data_type)?;
            Ok(array::rand_type(&arrow_type))
        }
    }
}

/// Build a fill generator for a specific type
fn build_fill_generator(
    data_type: &str,
    value: &serde_yaml::Value,
) -> Result<Box<dyn ArrayGenerator>, YamlError> {
    use arrow_array::types::*;

    let arrow_type = parse_data_type(data_type)?;

    macro_rules! fill_primitive {
        ($native:ty, $arrow_type:ty) => {{
            let v: $native = value_to_native(value, stringify!($native))?;
            Ok(array::fill::<$arrow_type>(v))
        }};
    }

    match arrow_type {
        DataType::Int8 => fill_primitive!(i8, Int8Type),
        DataType::Int16 => fill_primitive!(i16, Int16Type),
        DataType::Int32 => fill_primitive!(i32, Int32Type),
        DataType::Int64 => fill_primitive!(i64, Int64Type),
        DataType::UInt8 => fill_primitive!(u8, UInt8Type),
        DataType::UInt16 => fill_primitive!(u16, UInt16Type),
        DataType::UInt32 => fill_primitive!(u32, UInt32Type),
        DataType::UInt64 => fill_primitive!(u64, UInt64Type),
        DataType::Float32 => fill_primitive!(f32, Float32Type),
        DataType::Float64 => fill_primitive!(f64, Float64Type),
        DataType::Utf8 | DataType::LargeUtf8 => {
            let s = value
                .as_str()
                .ok_or_else(|| YamlError::InvalidValue {
                    field: "value".to_string(),
                    message: "Expected string value".to_string(),
                })?
                .to_string();
            Ok(array::fill_utf8(s))
        }
        DataType::Binary | DataType::LargeBinary => {
            let bytes = value_to_bytes(value)?;
            Ok(array::fill_varbin(bytes))
        }
        _ => Err(YamlError::InvalidGenerator(format!(
            "fill generator not supported for type: {}",
            data_type
        ))),
    }
}

/// Build a cycle generator for a specific type
fn build_cycle_generator(
    data_type: &str,
    values: &[serde_yaml::Value],
) -> Result<Box<dyn ArrayGenerator>, YamlError> {
    use arrow_array::types::*;

    let arrow_type = parse_data_type(data_type)?;

    macro_rules! cycle_primitive {
        ($native:ty, $arrow_type:ty) => {{
            let vals: Vec<$native> = values
                .iter()
                .map(|v| value_to_native::<$native>(v, stringify!($native)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(array::cycle::<$arrow_type>(vals))
        }};
    }

    match arrow_type {
        DataType::Int8 => cycle_primitive!(i8, Int8Type),
        DataType::Int16 => cycle_primitive!(i16, Int16Type),
        DataType::Int32 => cycle_primitive!(i32, Int32Type),
        DataType::Int64 => cycle_primitive!(i64, Int64Type),
        DataType::UInt8 => cycle_primitive!(u8, UInt8Type),
        DataType::UInt16 => cycle_primitive!(u16, UInt16Type),
        DataType::UInt32 => cycle_primitive!(u32, UInt32Type),
        DataType::UInt64 => cycle_primitive!(u64, UInt64Type),
        DataType::Float32 => cycle_primitive!(f32, Float32Type),
        DataType::Float64 => cycle_primitive!(f64, Float64Type),
        DataType::Utf8 | DataType::LargeUtf8 => {
            let strs: Vec<String> = values
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| YamlError::InvalidValue {
                            field: "values".to_string(),
                            message: "Expected string values".to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let static_strs: Vec<&'static str> = strs
                .into_iter()
                .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
                .collect();
            Ok(array::cycle_utf8_literals(&static_strs))
        }
        _ => Err(YamlError::InvalidGenerator(format!(
            "cycle generator not supported for type: {}",
            data_type
        ))),
    }
}

/// Build a step generator for a specific type
fn build_step_generator(data_type: &str) -> Result<Box<dyn ArrayGenerator>, YamlError> {
    use arrow_array::types::*;

    let arrow_type = parse_data_type(data_type)?;

    match arrow_type {
        DataType::Int8 => Ok(array::step::<Int8Type>()),
        DataType::Int16 => Ok(array::step::<Int16Type>()),
        DataType::Int32 => Ok(array::step::<Int32Type>()),
        DataType::Int64 => Ok(array::step::<Int64Type>()),
        DataType::UInt8 => Ok(array::step::<UInt8Type>()),
        DataType::UInt16 => Ok(array::step::<UInt16Type>()),
        DataType::UInt32 => Ok(array::step::<UInt32Type>()),
        DataType::UInt64 => Ok(array::step::<UInt64Type>()),
        DataType::Float32 => Ok(array::step::<Float32Type>()),
        DataType::Float64 => Ok(array::step::<Float64Type>()),
        _ => Err(YamlError::InvalidGenerator(format!(
            "step generator not supported for type: {}",
            data_type
        ))),
    }
}

/// Build a step_custom generator for a specific type
fn build_step_custom_generator(
    data_type: &str,
    start: &serde_yaml::Value,
    step: &serde_yaml::Value,
) -> Result<Box<dyn ArrayGenerator>, YamlError> {
    use arrow_array::types::*;

    let arrow_type = parse_data_type(data_type)?;

    macro_rules! step_custom_primitive {
        ($native:ty, $arrow_type:ty) => {{
            let start_val: $native = value_to_native(start, stringify!($native))?;
            let step_val: $native = value_to_native(step, stringify!($native))?;
            Ok(array::step_custom::<$arrow_type>(start_val, step_val))
        }};
    }

    match arrow_type {
        DataType::Int8 => step_custom_primitive!(i8, Int8Type),
        DataType::Int16 => step_custom_primitive!(i16, Int16Type),
        DataType::Int32 => step_custom_primitive!(i32, Int32Type),
        DataType::Int64 => step_custom_primitive!(i64, Int64Type),
        DataType::UInt8 => step_custom_primitive!(u8, UInt8Type),
        DataType::UInt16 => step_custom_primitive!(u16, UInt16Type),
        DataType::UInt32 => step_custom_primitive!(u32, UInt32Type),
        DataType::UInt64 => step_custom_primitive!(u64, UInt64Type),
        DataType::Float32 => step_custom_primitive!(f32, Float32Type),
        DataType::Float64 => step_custom_primitive!(f64, Float64Type),
        _ => Err(YamlError::InvalidGenerator(format!(
            "step_custom generator not supported for type: {}",
            data_type
        ))),
    }
}

/// Build a rand_vec generator for a specific element type
fn build_rand_vec_generator(
    element_type: &str,
    dimension: u32,
) -> Result<Box<dyn ArrayGenerator>, YamlError> {
    use arrow_array::types::*;

    let arrow_type = parse_data_type(element_type)?;
    let dim = crate::Dimension::from(dimension);

    match arrow_type {
        DataType::Int8 => Ok(array::rand_vec::<Int8Type>(dim)),
        DataType::Int16 => Ok(array::rand_vec::<Int16Type>(dim)),
        DataType::Int32 => Ok(array::rand_vec::<Int32Type>(dim)),
        DataType::Int64 => Ok(array::rand_vec::<Int64Type>(dim)),
        DataType::UInt8 => Ok(array::rand_vec::<UInt8Type>(dim)),
        DataType::UInt16 => Ok(array::rand_vec::<UInt16Type>(dim)),
        DataType::UInt32 => Ok(array::rand_vec::<UInt32Type>(dim)),
        DataType::UInt64 => Ok(array::rand_vec::<UInt64Type>(dim)),
        DataType::Float32 => Ok(array::rand_vec::<Float32Type>(dim)),
        DataType::Float64 => Ok(array::rand_vec::<Float64Type>(dim)),
        _ => Err(YamlError::InvalidGenerator(format!(
            "rand_vec generator not supported for element type: {}",
            element_type
        ))),
    }
}

/// Convert a YAML value to a native Rust type
fn value_to_native<T>(value: &serde_yaml::Value, type_name: &str) -> Result<T, YamlError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    // Try to get as the appropriate type
    let str_val = match value {
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        _ => {
            return Err(YamlError::InvalidValue {
                field: "value".to_string(),
                message: format!("Cannot convert {:?} to {}", value, type_name),
            })
        }
    };

    str_val.parse::<T>().map_err(|e| YamlError::InvalidValue {
        field: "value".to_string(),
        message: format!("Cannot parse '{}' as {}: {}", str_val, type_name, e),
    })
}

/// Convert a YAML value to bytes
fn value_to_bytes(value: &serde_yaml::Value) -> Result<Vec<u8>, YamlError> {
    match value {
        serde_yaml::Value::String(s) => Ok(s.as_bytes().to_vec()),
        serde_yaml::Value::Sequence(seq) => {
            let bytes: Vec<u8> = seq
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| YamlError::InvalidValue {
                            field: "value".to_string(),
                            message: "Expected byte value (0-255)".to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(bytes)
        }
        _ => Err(YamlError::InvalidValue {
            field: "value".to_string(),
            message: "Expected string or byte array".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RowCount;

    #[test]
    fn test_parse_simple_config() {
        let yaml = r#"
seed: 42
columns:
  - name: id
    generator: step
    type: int64
  - name: score
    generator: rand
    type: float32
"#;
        let config = parse_yaml(yaml).unwrap();
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.columns.len(), 2);
        assert_eq!(config.columns[0].name, "id");
        assert_eq!(config.columns[1].name, "score");
    }

    #[test]
    fn test_parse_with_null_probability() {
        let yaml = r#"
default_null_probability: 0.1
columns:
  - name: data
    generator: rand
    type: int32
    null_probability: 0.2
"#;
        let config = parse_yaml(yaml).unwrap();
        assert_eq!(config.default_null_probability, Some(0.1));
        assert_eq!(config.columns[0].null_probability, Some(0.2));
    }

    #[test]
    fn test_parse_fill_generator() {
        let yaml = r#"
columns:
  - name: status
    generator: fill
    type: utf8
    value: "active"
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(5)).unwrap();
        assert_eq!(batch.num_rows(), 5);
        let arr = batch.column(0);
        let str_arr = arr
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        for i in 0..5 {
            assert_eq!(str_arr.value(i), "active");
        }
    }

    #[test]
    fn test_parse_cycle_generator() {
        let yaml = r#"
columns:
  - name: category
    generator: cycle
    type: int32
    values: [1, 2, 3]
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(6)).unwrap();
        assert_eq!(batch.num_rows(), 6);
        let arr = batch.column(0);
        let int_arr = arr
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(int_arr.value(0), 1);
        assert_eq!(int_arr.value(1), 2);
        assert_eq!(int_arr.value(2), 3);
        assert_eq!(int_arr.value(3), 1);
        assert_eq!(int_arr.value(4), 2);
        assert_eq!(int_arr.value(5), 3);
    }

    #[test]
    fn test_parse_rand_vec_generator() {
        let yaml = r#"
columns:
  - name: embedding
    generator: rand_vec
    type: float32
    dimension: 4
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(3)).unwrap();
        assert_eq!(batch.num_rows(), 3);
        let arr = batch.column(0);
        assert!(matches!(arr.data_type(), DataType::FixedSizeList(_, 4)));
    }

    #[test]
    fn test_parse_dict_generator() {
        let yaml = r#"
columns:
  - name: region
    generator: dict
    key_type: int32
    inner:
      generator: cycle_utf8
      values: ["north", "south", "east", "west"]
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(8)).unwrap();
        assert_eq!(batch.num_rows(), 8);
        let arr = batch.column(0);
        assert!(matches!(arr.data_type(), DataType::Dictionary(_, _)));
    }

    #[test]
    fn test_parse_low_cardinality_generator() {
        let yaml = r#"
columns:
  - name: status_code
    generator: low_cardinality
    cardinality: 5
    inner:
      generator: rand
      type: int32
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(100)).unwrap();
        assert_eq!(batch.num_rows(), 100);
    }

    #[test]
    fn test_parse_timestamp_generator() {
        // Note: timezone is not used here because the underlying rand_timestamp
        // generator creates arrays that don't preserve timezone information
        let yaml = r#"
columns:
  - name: created_at
    generator: rand_timestamp
    unit: millisecond
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(5)).unwrap();
        assert_eq!(batch.num_rows(), 5);
    }

    #[test]
    fn test_parse_modifiers() {
        let yaml = r#"
columns:
  - name: data
    generator: rand
    type: float32
    modifiers:
      nans: [false, false, true]
      metadata:
        custom_key: "custom_value"
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(6)).unwrap();
        assert_eq!(batch.num_rows(), 6);
        let schema = batch.schema();
        let field = schema.field(0);
        let metadata = field.metadata();
        assert_eq!(
            metadata.get("custom_key"),
            Some(&"custom_value".to_string())
        );
    }

    #[test]
    fn test_parse_data_types() {
        assert_eq!(parse_data_type("int32").unwrap(), DataType::Int32);
        assert_eq!(parse_data_type("float64").unwrap(), DataType::Float64);
        assert_eq!(parse_data_type("utf8").unwrap(), DataType::Utf8);
        assert_eq!(parse_data_type("string").unwrap(), DataType::Utf8);
        assert_eq!(parse_data_type("boolean").unwrap(), DataType::Boolean);
        assert_eq!(parse_data_type("bool").unwrap(), DataType::Boolean);

        assert_eq!(
            parse_data_type("Decimal128(10, 2)").unwrap(),
            DataType::Decimal128(10, 2)
        );
        assert_eq!(
            parse_data_type("FixedSizeBinary(16)").unwrap(),
            DataType::FixedSizeBinary(16)
        );
        assert_eq!(
            parse_data_type("Timestamp(millisecond, UTC)").unwrap(),
            DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("UTC")))
        );
        assert_eq!(
            parse_data_type("Duration(second)").unwrap(),
            DataType::Duration(TimeUnit::Second)
        );
    }

    #[test]
    fn test_from_yaml_convenience() {
        let yaml = r#"
seed: 42
columns:
  - name: id
    generator: step
    type: int64
"#;
        let builder = from_yaml(yaml).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(100)).unwrap();
        assert_eq!(batch.num_rows(), 100);
    }

    #[test]
    fn test_invalid_type_error() {
        let result = parse_data_type("invalid_type");
        assert!(matches!(result, Err(YamlError::InvalidType(_))));
    }

    #[test]
    fn test_rand_list_generator() {
        let yaml = r#"
columns:
  - name: tags
    generator: rand_list
    item_type: utf8
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(5)).unwrap();
        assert_eq!(batch.num_rows(), 5);
    }

    #[test]
    fn test_rand_struct_generator() {
        let yaml = r#"
columns:
  - name: person
    generator: rand_struct
    fields:
      - name: age
        type: int32
      - name: name
        type: utf8
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(5)).unwrap();
        assert_eq!(batch.num_rows(), 5);
        let arr = batch.column(0);
        assert!(matches!(arr.data_type(), DataType::Struct(_)));
    }

    #[test]
    fn test_random_sentence_generator() {
        let yaml = r#"
columns:
  - name: bio
    generator: random_sentence
    min_words: 3
    max_words: 10
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(5)).unwrap();
        assert_eq!(batch.num_rows(), 5);
    }

    #[test]
    fn test_uuid_generators() {
        let yaml = r#"
columns:
  - name: uuid_bin
    generator: rand_pseudo_uuid
  - name: uuid_hex
    generator: rand_pseudo_uuid_hex
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(5)).unwrap();
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_step_custom_generator() {
        let yaml = r#"
columns:
  - name: seq
    generator: step_custom
    type: int32
    start: 100
    step: 10
"#;
        let config = parse_yaml(yaml).unwrap();
        let builder = config_to_builder(config).unwrap();
        let batch = builder.into_batch_rows(RowCount::from(5)).unwrap();
        let arr = batch.column(0);
        let int_arr = arr
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(int_arr.value(0), 100);
        assert_eq!(int_arr.value(1), 110);
        assert_eq!(int_arr.value(2), 120);
    }
}
