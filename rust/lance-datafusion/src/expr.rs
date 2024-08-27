// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with datafusion expressions

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{cast::AsArray, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use datafusion_common::ScalarValue;

#[cfg(feature = "substrait")]
use {
    arrow_schema::Schema,
    datafusion::{
        datasource::empty::EmptyTable, execution::context::SessionContext, logical_expr::Expr,
    },
    datafusion_common::{
        tree_node::{Transformed, TreeNode},
        Column, DataFusionError, TableReference,
    },
    datafusion_substrait::substrait::proto::{
        expression::field_reference::{ReferenceType, RootType},
        expression::reference_segment,
        expression::RexType,
        expression_reference::ExprType,
        extensions::{simple_extension_declaration::MappingType, SimpleExtensionDeclaration},
        function_argument::ArgType,
        plan_rel::RelType,
        r#type::{Kind, Struct},
        read_rel::{NamedTable, ReadType},
        rel, Expression, ExtendedExpression, NamedStruct, Plan, PlanRel, ProjectRel, ReadRel, Rel,
        RelRoot,
    },
    lance_core::{Error, Result},
    prost::Message,
    snafu::{location, Location},
    std::collections::HashMap,
};

const MS_PER_DAY: i64 = 86400000;

// This is slightly tedious but when we convert expressions from SQL strings to logical
// datafusion expressions there is no type coercion that happens.  In other words "x = 7"
// will always yield "x = 7_u64" regardless of the type of the column "x".  As a result, we
// need to do that literal coercion ourselves.
pub fn safe_coerce_scalar(value: &ScalarValue, ty: &DataType) -> Option<ScalarValue> {
    match value {
        ScalarValue::Int8(val) => match ty {
            DataType::Int8 => Some(value.clone()),
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(i16::from(v)))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => Some(value.clone()),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => Some(value.clone()),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // These conversions are inherently lossy as the full range of i32 cannot
            // be represented in f32.  However, there is no f32::TryFrom(i32) and its not
            // clear users would want that anyways
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Int64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => Some(value.clone()),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::UInt8(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(v.into()))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => Some(value.clone()),
            DataType::UInt16 => val.map(|v| ScalarValue::UInt16(Some(u16::from(v)))),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => Some(value.clone()),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => Some(value.clone()),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::UInt64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => {
                val.and_then(|v| i64::try_from(v).map(|v| ScalarValue::Int64(Some(v))).ok())
            }
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => Some(value.clone()),
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Float32(val) => match ty {
            DataType::Float32 => Some(value.clone()),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Float64(val) => match ty {
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Utf8(val) => match ty {
            DataType::Utf8 => Some(value.clone()),
            DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(val.clone())),
            _ => None,
        },
        ScalarValue::Boolean(_) => match ty {
            DataType::Boolean => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Null => Some(value.clone()),
        ScalarValue::List(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::TimestampSecond(seconds, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Millisecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampMillisecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Microsecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000000))
                .map(|val| ScalarValue::TimestampMicrosecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000000000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampMillisecond(millis, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, tz) => {
                millis.map(|val| ScalarValue::TimestampSecond(Some(val / 1000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Microsecond, tz) => millis
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampMicrosecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => millis
                .and_then(|v| v.checked_mul(1000000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampMicrosecond(micros, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, tz) => {
                micros.map(|val| ScalarValue::TimestampSecond(Some(val / 1000000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                micros.map(|val| ScalarValue::TimestampMillisecond(Some(val / 1000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => micros
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampNanosecond(nanos, _) => {
            match ty {
                DataType::Timestamp(TimeUnit::Second, tz) => nanos
                    .map(|val| ScalarValue::TimestampSecond(Some(val / 1000000000), tz.clone())),
                DataType::Timestamp(TimeUnit::Millisecond, tz) => nanos
                    .map(|val| ScalarValue::TimestampMillisecond(Some(val / 1000000), tz.clone())),
                DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                    nanos.map(|val| ScalarValue::TimestampMicrosecond(Some(val / 1000), tz.clone()))
                }
                DataType::Timestamp(TimeUnit::Nanosecond, _) => Some(value.clone()),
                _ => None,
            }
        }
        ScalarValue::Date32(ticks) => match ty {
            DataType::Date32 => Some(value.clone()),
            DataType::Date64 => Some(ScalarValue::Date64(
                ticks.map(|v| i64::from(v) * MS_PER_DAY),
            )),
            _ => None,
        },
        ScalarValue::Date64(ticks) => match ty {
            DataType::Date32 => Some(ScalarValue::Date32(ticks.map(|v| (v / MS_PER_DAY) as i32))),
            DataType::Date64 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Time32Second(seconds) => {
            match ty {
                DataType::Time32(TimeUnit::Second) => Some(value.clone()),
                DataType::Time32(TimeUnit::Millisecond) => {
                    seconds.map(|val| ScalarValue::Time32Millisecond(Some(val * 1000)))
                }
                DataType::Time64(TimeUnit::Microsecond) => seconds
                    .map(|val| ScalarValue::Time64Microsecond(Some(i64::from(val) * 1000000))),
                DataType::Time64(TimeUnit::Nanosecond) => seconds
                    .map(|val| ScalarValue::Time64Nanosecond(Some(i64::from(val) * 1000000000))),
                _ => None,
            }
        }
        ScalarValue::Time32Millisecond(millis) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                millis.map(|val| ScalarValue::Time32Second(Some(val / 1000)))
            }
            DataType::Time32(TimeUnit::Millisecond) => Some(value.clone()),
            DataType::Time64(TimeUnit::Microsecond) => {
                millis.map(|val| ScalarValue::Time64Microsecond(Some(i64::from(val) * 1000)))
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                millis.map(|val| ScalarValue::Time64Nanosecond(Some(i64::from(val) * 1000000)))
            }
            _ => None,
        },
        ScalarValue::Time64Microsecond(micros) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                micros.map(|val| ScalarValue::Time32Second(Some((val / 1000000) as i32)))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                micros.map(|val| ScalarValue::Time32Millisecond(Some((val / 1000) as i32)))
            }
            DataType::Time64(TimeUnit::Microsecond) => Some(value.clone()),
            DataType::Time64(TimeUnit::Nanosecond) => {
                micros.map(|val| ScalarValue::Time64Nanosecond(Some(val * 1000)))
            }
            _ => None,
        },
        ScalarValue::Time64Nanosecond(nanos) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                nanos.map(|val| ScalarValue::Time32Second(Some((val / 1000000000) as i32)))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                nanos.map(|val| ScalarValue::Time32Millisecond(Some((val / 1000000) as i32)))
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                nanos.map(|val| ScalarValue::Time64Microsecond(Some(val / 1000)))
            }
            DataType::Time64(TimeUnit::Nanosecond) => Some(value.clone()),
            _ => None,
        },
        ScalarValue::LargeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::FixedSizeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Convert a DF Expr into a Substrait ExtendedExpressions message
#[cfg(feature = "substrait")]
pub fn encode_substrait(expr: Expr, schema: Arc<Schema>) -> Result<Vec<u8>> {
    use datafusion::logical_expr::{builder::LogicalTableSource, logical_plan, LogicalPlan};
    use datafusion_substrait::substrait::proto::{plan_rel, ExpressionReference, NamedStruct};

    let table_source = Arc::new(LogicalTableSource::new(schema.clone()));

    // DF doesn't handled ExtendedExpressions and so we need to create
    // a dummy plan with a single filter node
    let plan = LogicalPlan::Filter(logical_plan::Filter::try_new(
        expr,
        Arc::new(LogicalPlan::TableScan(logical_plan::TableScan::try_new(
            "dummy",
            table_source,
            None,
            vec![],
            None,
        )?)),
    )?);

    let session_context = SessionContext::new();

    let substrait_plan =
        datafusion_substrait::logical_plan::producer::to_substrait_plan(&plan, &session_context)?;

    if let Some(plan_rel::RelType::Root(root)) = &substrait_plan.relations[0].rel_type {
        if let Some(rel::RelType::Filter(filt)) = &root.input.as_ref().unwrap().rel_type {
            let expr = filt.condition.as_ref().unwrap().clone();
            let schema = NamedStruct {
                names: schema.fields().iter().map(|f| f.name().clone()).collect(),
                r#struct: None,
            };
            let envelope = ExtendedExpression {
                advanced_extensions: substrait_plan.advanced_extensions.clone(),
                base_schema: Some(schema),
                expected_type_urls: substrait_plan.expected_type_urls.clone(),
                extension_uris: substrait_plan.extension_uris.clone(),
                extensions: substrait_plan.extensions.clone(),
                referred_expr: vec![ExpressionReference {
                    output_names: vec![],
                    expr_type: Some(ExprType::Expression(*expr)),
                }],
                version: substrait_plan.version.clone(),
            };
            Ok(envelope.encode_to_vec())
        } else {
            unreachable!()
        }
    } else {
        unreachable!()
    }
}

#[cfg(feature = "substrait")]
fn remove_extension_types(
    substrait_schema: &NamedStruct,
    arrow_schema: Arc<Schema>,
) -> Result<(NamedStruct, Arc<Schema>, HashMap<usize, usize>)> {
    let fields = substrait_schema.r#struct.as_ref().unwrap();
    if fields.types.len() != arrow_schema.fields.len() {
        return Err(Error::InvalidInput {
            source: "the number of fields in the provided substrait schema did not match the number of fields in the input schema.".into(),
            location: location!(),
        });
    }
    let mut kept_substrait_fields = Vec::with_capacity(fields.types.len());
    let mut kept_arrow_fields = Vec::with_capacity(arrow_schema.fields.len());
    let mut index_mapping = HashMap::with_capacity(arrow_schema.fields.len());
    let mut field_counter = 0;
    for (field_index, (substrait_field, arrow_field)) in fields
        .types
        .iter()
        .zip(arrow_schema.fields.iter())
        .enumerate()
    {
        if !matches!(
            substrait_field.kind.as_ref().unwrap(),
            Kind::UserDefined(_) | Kind::UserDefinedTypeReference(_)
        ) {
            kept_substrait_fields.push(substrait_field.clone());
            kept_arrow_fields.push(arrow_field.clone());
            index_mapping.insert(field_index, field_counter);
            field_counter += 1;
        }
    }
    let new_arrow_schema = Arc::new(Schema::new(kept_arrow_fields));
    let new_substrait_schema = NamedStruct {
        names: vec![],
        r#struct: Some(Struct {
            nullability: fields.nullability,
            type_variation_reference: fields.type_variation_reference,
            types: kept_substrait_fields,
        }),
    };
    Ok((new_substrait_schema, new_arrow_schema, index_mapping))
}

#[cfg(feature = "substrait")]
fn remove_type_extensions(
    declarations: &[SimpleExtensionDeclaration],
) -> Vec<SimpleExtensionDeclaration> {
    declarations
        .iter()
        .filter(|d| matches!(d.mapping_type, Some(MappingType::ExtensionFunction(_))))
        .cloned()
        .collect()
}

#[cfg(feature = "substrait")]
fn remap_expr_references(expr: &mut Expression, mapping: &HashMap<usize, usize>) -> Result<()> {
    match expr.rex_type.as_mut().unwrap() {
        // Simple, no field references possible
        RexType::Literal(_) | RexType::Nested(_) | RexType::Enum(_) => Ok(()),
        // Complex operators not supported in filters
        RexType::WindowFunction(_) | RexType::Subquery(_) => Err(Error::invalid_input(
            "Window functions or subqueries not allowed in filter expression",
            location!(),
        )),
        // Pass through operators, nested children may have field references
        RexType::ScalarFunction(ref mut func) => {
            #[allow(deprecated)]
            for arg in &mut func.args {
                remap_expr_references(arg, mapping)?;
            }
            for arg in &mut func.arguments {
                match arg.arg_type.as_mut().unwrap() {
                    ArgType::Value(expr) => remap_expr_references(expr, mapping)?,
                    ArgType::Enum(_) | ArgType::Type(_) => {}
                }
            }
            Ok(())
        }
        RexType::IfThen(ref mut ifthen) => {
            for clause in ifthen.ifs.iter_mut() {
                remap_expr_references(clause.r#if.as_mut().unwrap(), mapping)?;
                remap_expr_references(clause.then.as_mut().unwrap(), mapping)?;
            }
            remap_expr_references(ifthen.r#else.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::SwitchExpression(ref mut switch) => {
            for clause in switch.ifs.iter_mut() {
                remap_expr_references(clause.then.as_mut().unwrap(), mapping)?;
            }
            remap_expr_references(switch.r#else.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::SingularOrList(ref mut orlist) => {
            for opt in orlist.options.iter_mut() {
                remap_expr_references(opt, mapping)?;
            }
            remap_expr_references(orlist.value.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::MultiOrList(ref mut orlist) => {
            for opt in orlist.options.iter_mut() {
                for field in opt.fields.iter_mut() {
                    remap_expr_references(field, mapping)?;
                }
            }
            for val in orlist.value.iter_mut() {
                remap_expr_references(val, mapping)?;
            }
            Ok(())
        }
        RexType::Cast(ref mut cast) => {
            remap_expr_references(cast.input.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::Selection(ref mut sel) => {
            // Finally, the selection, which might actually have field references
            let root_type = sel.root_type.as_mut().unwrap();
            // These types of references do not reference input fields so no remap needed
            if matches!(
                root_type,
                RootType::Expression(_) | RootType::OuterReference(_)
            ) {
                return Ok(());
            }
            match sel.reference_type.as_mut().unwrap() {
                ReferenceType::DirectReference(direct) => {
                    match direct.reference_type.as_mut().unwrap() {
                        reference_segment::ReferenceType::ListElement(_)
                        | reference_segment::ReferenceType::MapKey(_) => Err(Error::invalid_input(
                            "map/list nested references not supported in pushdown filters",
                            location!(),
                        )),
                        reference_segment::ReferenceType::StructField(field) => {
                            if field.child.is_some() {
                                Err(Error::invalid_input(
                                    "nested references in pushdown filters not yet supported",
                                    location!(),
                                ))
                            } else {
                                if let Some(new_index) = mapping.get(&(field.field as usize)) {
                                    field.field = *new_index as i32;
                                } else {
                                    return Err(Error::invalid_input("pushdown filter referenced a field that is not yet supported by Substrait conversion", location!()));
                                }
                                Ok(())
                            }
                        }
                    }
                }
                ReferenceType::MaskedReference(_) => Err(Error::invalid_input(
                    "masked references not yet supported in filter expressions",
                    location!(),
                )),
            }
        }
    }
}

/// Convert a Substrait ExtendedExpressions message into a DF Expr
///
/// The ExtendedExpressions message must contain a single scalar expression
#[cfg(feature = "substrait")]
pub async fn parse_substrait(expr: &[u8], input_schema: Arc<Schema>) -> Result<Expr> {
    let envelope = ExtendedExpression::decode(expr)?;
    if envelope.referred_expr.is_empty() {
        return Err(Error::InvalidInput {
            source: "the provided substrait expression is empty (contains no expressions)".into(),
            location: location!(),
        });
    }
    if envelope.referred_expr.len() > 1 {
        return Err(Error::InvalidInput {
            source: format!(
                "the provided substrait expression had {} expressions when only 1 was expected",
                envelope.referred_expr.len()
            )
            .into(),
            location: location!(),
        });
    }
    let mut expr = match &envelope.referred_expr[0].expr_type {
        None => Err(Error::InvalidInput {
            source: "the provided substrait had an expression but was missing an expr_type".into(),
            location: location!(),
        }),
        Some(ExprType::Expression(expr)) => Ok(expr.clone()),
        _ => Err(Error::InvalidInput {
            source: "the provided substrait was not a scalar expression".into(),
            location: location!(),
        }),
    }?;

    let (substrait_schema, input_schema, index_mapping) =
        remove_extension_types(envelope.base_schema.as_ref().unwrap(), input_schema.clone())?;

    if substrait_schema.r#struct.as_ref().unwrap().types.len()
        != envelope
            .base_schema
            .as_ref()
            .unwrap()
            .r#struct
            .as_ref()
            .unwrap()
            .types
            .len()
    {
        remap_expr_references(&mut expr, &index_mapping)?;
    }

    // Datafusion's substrait consumer only supports Plan (not ExtendedExpression) and so
    // we need to create a dummy plan with a single project node
    let plan = Plan {
        version: None,
        extensions: remove_type_extensions(&envelope.extensions),
        advanced_extensions: envelope.advanced_extensions.clone(),
        expected_type_urls: vec![],
        extension_uris: vec![],
        relations: vec![PlanRel {
            rel_type: Some(RelType::Root(RelRoot {
                input: Some(Rel {
                    rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                        common: None,
                        input: Some(Box::new(Rel {
                            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                                common: None,
                                base_schema: Some(substrait_schema),
                                filter: None,
                                best_effort_filter: None,
                                projection: None,
                                advanced_extension: None,
                                read_type: Some(ReadType::NamedTable(NamedTable {
                                    names: vec!["dummy".to_string()],
                                    advanced_extension: None,
                                })),
                            }))),
                        })),
                        expressions: vec![expr],
                        advanced_extension: None,
                    }))),
                }),
                // Not technically accurate but pretty sure DF ignores this
                names: vec![],
            })),
        }],
    };

    let session_context = SessionContext::new();
    let dummy_table = Arc::new(EmptyTable::new(input_schema));
    session_context.register_table(
        TableReference::Bare {
            table: "dummy".into(),
        },
        dummy_table,
    )?;
    let df_plan =
        datafusion_substrait::logical_plan::consumer::from_substrait_plan(&session_context, &plan)
            .await?;

    let expr = df_plan.expressions().pop().unwrap();

    // When DF parses the above plan it turns column references into qualified references
    // into `dummy` (e.g. we get `WHERE dummy.x < 0` instead of `WHERE x < 0`)  We want
    // these to be unqualified references instead and so we need a quick trasnformation pass

    let expr = expr.transform(&|node| match node {
        Expr::Column(column) => {
            if let Some(relation) = column.relation {
                match relation {
                    TableReference::Bare { table } => {
                        if table.as_ref() == "dummy" {
                            Ok(Transformed::yes(Expr::Column(Column {
                                relation: None,
                                name: column.name,
                            })))
                        } else {
                            // This should not be possible
                            Err(DataFusionError::Substrait(format!(
                                "Unexpected reference to table {} found when parsing filter",
                                table
                            )))
                        }
                    }
                            // This should not be possible
                            _ => Err(DataFusionError::Substrait("Unexpected partially or fully qualified table reference encountered when parsing filter".into()))
                }
            } else {
                Ok(Transformed::no(Expr::Column(column)))
            }
        }
        _ => Ok(Transformed::no(node)),
    })?;
    Ok(expr.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "substrait")]
    use {
        arrow_schema::Field,
        datafusion::logical_expr::{BinaryExpr, Operator},
        substrait_expr::{
            builder::{schema::SchemaBuildersExt, BuilderParams, ExpressionsBuilder},
            functions::functions_comparison::FunctionsComparisonExt,
            helpers::{literals::literal, schema::SchemaInfo},
        },
    };

    #[test]
    fn test_temporal_coerce() {
        // Conversion from timestamps in one resolution to timestamps in another resolution is allowed
        // s->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // s->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // s->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // s->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // ms->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // ms->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // ms->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // ms->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // us->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // us->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // us->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // us->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // ns->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // ns->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // ns->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // ns->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // Precision loss on coercion is allowed (truncation)
        // ns->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5987654321), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // Conversions from date-32 to date-64 is allowed
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date32(Some(5)), &DataType::Date32,),
            Some(ScalarValue::Date32(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date32(Some(5)), &DataType::Date64,),
            Some(ScalarValue::Date64(Some(5 * MS_PER_DAY)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Date64(Some(5 * MS_PER_DAY)),
                &DataType::Date32,
            ),
            Some(ScalarValue::Date32(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date64(Some(5)), &DataType::Date64,),
            Some(ScalarValue::Date64(Some(5)))
        );
        // Time-32 to time-64 (and within time-32 and time-64) is allowed
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
    }

    #[cfg(feature = "substrait")]
    #[tokio::test]
    async fn test_substrait_conversion() {
        let schema = SchemaInfo::new_full()
            .field("x", substrait_expr::helpers::types::i32(true))
            .build();
        let expr_builder = ExpressionsBuilder::new(schema, BuilderParams::default());
        expr_builder
            .add_expression(
                "filter_mask",
                expr_builder
                    .functions()
                    .lt(
                        expr_builder.fields().resolve_by_name("x").unwrap(),
                        literal(0_i32),
                    )
                    .build()
                    .unwrap(),
            )
            .unwrap();
        let expr = expr_builder.build();
        let expr_bytes = expr.encode_to_vec();

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));

        let df_expr = parse_substrait(expr_bytes.as_slice(), schema)
            .await
            .unwrap();

        let expected = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column(Column {
                relation: None,
                name: "x".to_string(),
            })),
            op: Operator::Lt,
            right: Box::new(Expr::Literal(ScalarValue::Int32(Some(0)))),
        });
        assert_eq!(df_expr, expected);
    }
}
