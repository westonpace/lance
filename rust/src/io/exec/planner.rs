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

//! Exec plan planner

use std::sync::Arc;

use arrow_cast::can_cast_types;
use arrow_schema::{DataType as ArrowDataType, SchemaRef};
use datafusion::common::{Result as DatafusionResult, ToDFSchema};
use datafusion::config::ConfigOptions;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{AggregateUDF, ScalarUDF, TableSource, WindowUDF};
use datafusion::physical_plan::expressions::GetIndexedFieldExpr;
use datafusion::sql::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion::sql::TableReference;
use datafusion::{
    logical_expr::{
        expr::{InList, ScalarFunction},
        BuiltinScalarFunction,
    },
    physical_expr::execution_props::ExecutionProps,
    physical_plan::{
        expressions::{
            CastExpr, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal, NotExpr,
        },
        functions, PhysicalExpr,
    },
    prelude::Expr,
};

use crate::datafusion::logical_expr::coerce_filter_type_to_boolean;
use crate::{
    datafusion::logical_expr::resolve_expr, datatypes::Schema, utils::sql::parse_sql_filter, Error,
    Result,
};

pub struct Planner {
    schema: SchemaRef,
    df_config_opts: ConfigOptions,
}

impl ContextProvider for Planner {
    fn get_table_provider(&self, name: TableReference) -> DatafusionResult<Arc<dyn TableSource>> {
        Err(DataFusionError::Internal(
            format!("Lance ContextProvider only supports scalar expressions but got reference to table {name:?}")
        ))
    }

    fn get_function_meta(&self, _name: &str) -> Option<Arc<ScalarUDF>> {
        None
    }

    fn get_aggregate_meta(&self, _name: &str) -> Option<Arc<AggregateUDF>> {
        None
    }

    fn get_window_meta(&self, _name: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_variable_type(&self, variable_names: &[String]) -> Option<ArrowDataType> {
        if variable_names.len() != 1 {
            return None;
        }
        self.schema
            .column_with_name(variable_names[0].as_str())
            .map(|c| c.1.data_type().clone())
    }

    fn options(&self) -> &ConfigOptions {
        &self.df_config_opts
    }
}

impl Planner {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            df_config_opts: ConfigOptions::default(),
        }
    }

    /// Create Logical [Expr] from a SQL filter clause.
    pub fn parse_filter(&self, filter: &str) -> Result<Expr> {
        // Allow sqlparser to parse filter as part of ONE SQL statement.

        let ast_expr = parse_sql_filter(filter)?;
        let df_schema = self.schema.clone().to_dfschema()?;
        let mut unused_context = PlannerContext::new();
        let expr = SqlToRel::new(self).sql_to_expr(ast_expr, &df_schema, &mut unused_context)?;
        // let expr = self.parse_sql_expr(&ast_expr)?;
        let schema = Schema::try_from(self.schema.as_ref())?;
        let resolved = resolve_expr(&expr, &schema)?;
        coerce_filter_type_to_boolean(resolved)
    }

    /// Create the [`PhysicalExpr`] from a logical [`Expr`]
    pub fn create_physical_expr(&self, expr: &Expr) -> Result<Arc<dyn PhysicalExpr>> {
        use crate::datafusion::physical_expr::Column;
        use datafusion::physical_expr::expressions::{BinaryExpr, NegativeExpr};

        Ok(match expr {
            Expr::Column(c) => Arc::new(Column::new(c.flat_name())),
            Expr::Literal(v) => Arc::new(Literal::new(v.clone())),
            Expr::BinaryExpr(expr) => {
                let left = self.create_physical_expr(expr.left.as_ref())?;
                let right = self.create_physical_expr(expr.right.as_ref())?;
                let left_data_type = left.data_type(&self.schema)?;
                let right_data_type = right.data_type(&self.schema)?;
                // Make sure RHS matches the LHS
                let right = if right_data_type != left_data_type {
                    if can_cast_types(&right_data_type, &left_data_type) {
                        Arc::new(CastExpr::new(right, left_data_type, None))
                    } else {
                        return Err(Error::invalid_input(format!(
                            "Cannot compare {} and {}",
                            left_data_type, right_data_type
                        )));
                    }
                } else {
                    right
                };
                Arc::new(BinaryExpr::new(left, expr.op, right))
            }
            Expr::GetIndexedField(indexed_field) => {
                let expr = self.create_physical_expr(&*indexed_field.expr)?;
                Arc::new(GetIndexedFieldExpr::new(expr, indexed_field.key.clone()))
            }
            Expr::Negative(expr) => {
                Arc::new(NegativeExpr::new(self.create_physical_expr(expr.as_ref())?))
            }
            Expr::IsNotNull(expr) => Arc::new(IsNotNullExpr::new(self.create_physical_expr(expr)?)),
            Expr::IsNull(expr) => Arc::new(IsNullExpr::new(self.create_physical_expr(expr)?)),
            Expr::IsTrue(expr) => self.create_physical_expr(expr)?,
            Expr::IsFalse(expr) => Arc::new(NotExpr::new(self.create_physical_expr(expr)?)),
            Expr::InList(InList {
                expr,
                list,
                negated,
            }) => {
                // It's important that all the values in the list are casted to match
                // the datatype of the column.
                let expr = self.create_physical_expr(expr)?;
                let datatype = expr.data_type(self.schema.as_ref())?;

                let list = list
                    .iter()
                    .map(|e| {
                        let e = self.create_physical_expr(e)?;
                        if e.data_type(self.schema.as_ref())? == datatype {
                            Ok(e)
                        } else {
                            // Cast the value to the column's datatype
                            let e: Arc<dyn PhysicalExpr> =
                                Arc::new(CastExpr::new(e, datatype.clone(), None));
                            Ok(e)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                Arc::new(InListExpr::new(expr, list, *negated, None))
            }
            Expr::Like(expr) => Arc::new(LikeExpr::new(
                expr.negated,
                true,
                self.create_physical_expr(expr.expr.as_ref())?,
                self.create_physical_expr(expr.pattern.as_ref())?,
            )),
            Expr::Not(expr) => Arc::new(NotExpr::new(self.create_physical_expr(expr)?)),
            Expr::Cast(datafusion::logical_expr::Cast { expr, data_type }) => {
                let expr = self.create_physical_expr(expr.as_ref())?;
                Arc::new(CastExpr::new(expr, data_type.clone(), None))
            }
            Expr::ScalarFunction(ScalarFunction { fun, args }) => {
                if fun != &BuiltinScalarFunction::RegexpMatch
                    && fun != &BuiltinScalarFunction::ArrayContains
                {
                    return Err(Error::IO {
                        message: format!("Scalar function '{:?}' is not supported", fun),
                    });
                }
                let execution_props = ExecutionProps::new();
                let args_vec = args
                    .iter()
                    .map(|e| self.create_physical_expr(e).unwrap())
                    .collect::<Vec<_>>();
                if args_vec.len() != 2 {
                    return Err(Error::IO {
                        message: format!(
                            "Scalar function '{:?}' only supports 2 args, got {}",
                            fun,
                            args_vec.len()
                        ),
                    });
                }

                let args_array: [Arc<dyn PhysicalExpr>; 2] =
                    [args_vec[0].clone(), args_vec[1].clone()];

                let physical_expr = functions::create_physical_expr(
                    fun,
                    &args_array,
                    self.schema.as_ref(),
                    &execution_props,
                );
                physical_expr?
            }
            _ => {
                return Err(Error::IO {
                    message: format!("Expression '{expr:?}' is not supported as filter in lance"),
                })
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
        StructArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    };
    use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit};
    use datafusion::{
        logical_expr::{col, lit, BinaryExpr, Cast, GetIndexedField},
        scalar::ScalarValue,
    };

    #[test]
    fn test_parse_filter_simple() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, true),
            Field::new(
                "st",
                DataType::Struct(Fields::from(vec![
                    Field::new("x", DataType::Float32, false),
                    Field::new("y", DataType::Float32, false),
                ])),
                true,
            ),
        ]));

        let planner = Planner::new(schema.clone());

        let st_x = Expr::GetIndexedField(GetIndexedField {
            expr: Box::new(col("st")),
            key: ScalarValue::Utf8(Some("x".to_string())),
        });

        let expected = col("i").gt(lit(3_i32)).and(st_x.lt_eq(lit(5.0_f64))).and(
            col("s")
                .eq(lit("str-4"))
                .or(col("s").in_list(vec![lit("str-4"), lit("str-5")], false)),
        );

        // double quotes
        let expr = planner
            .parse_filter("i > 3 AND st.x <= 5.0 AND (s == 'str-4' OR s in ('str-4', 'str-5'))")
            .unwrap();
        assert_eq!(expr, expected);

        // single quote
        let expr = planner
            .parse_filter("i > 3 AND st.x <= 5.0 AND (s = 'str-4' OR s in ('str-4', 'str-5'))")
            .unwrap();

        let physical_expr = planner.create_physical_expr(&expr).unwrap();
        println!("Physical expr: {:#?}", physical_expr);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(0..10)) as ArrayRef,
                Arc::new(StringArray::from_iter_values(
                    (0..10).map(|v| format!("str-{}", v)),
                )),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(Field::new("x", DataType::Float32, false)),
                        Arc::new(Float32Array::from_iter_values((0..10).map(|v| v as f32)))
                            as ArrayRef,
                    ),
                    (
                        Arc::new(Field::new("y", DataType::Float32, false)),
                        Arc::new(Float32Array::from_iter_values(
                            (0..10).map(|v| (v * 10) as f32),
                        )),
                    ),
                ])),
            ],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, true, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_negative_expressions() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));

        let planner = Planner::new(schema.clone());

        let expected = col("x")
            .gt(lit(-3_i64))
            .and(col("x").lt(-(lit(-5_i64) + lit(3_i64))));

        let expr = planner.parse_filter("x > -3 AND x < -(-5 + 3)").unwrap();

        assert_eq!(expr, expected);

        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from_iter_values(-5..5)) as ArrayRef],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, false, false, true, true, true, true, false, false, false
            ])
        );
    }

    #[test]
    fn test_sql_like() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").like(lit("str-4"));
        // single quote
        let expr = planner.parse_filter("s LIKE 'str-4'").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, false, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_not_like() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").not_like(lit("str-4"));
        // single quote
        let expr = planner.parse_filter("s NOT LIKE 'str-4'").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                true, true, true, true, false, true, true, true, true, true
            ])
        );
    }

    #[test]
    fn test_sql_is_in() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").in_list(vec![lit("str-4"), lit("str-5")], false);
        // single quote
        let expr = planner.parse_filter("s IN ('str-4', 'str-5')").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, true, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_sql_is_null() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").is_null();
        let expr = planner.parse_filter("s IS NULL").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter((0..10).map(|v| {
                if v % 3 == 0 {
                    Some(format!("str-{}", v))
                } else {
                    None
                }
            })))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, true, true, false, true, true, false, true, true, false
            ])
        );

        let expr = planner.parse_filter("s IS NOT NULL").unwrap();
        let physical_expr = planner.create_physical_expr(&expr).unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                true, false, false, true, false, false, true, false, false, true,
            ])
        );
    }

    #[test]
    fn test_sql_invert() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Boolean, true)]));

        let planner = Planner::new(schema.clone());

        let expr = planner.parse_filter("NOT s").unwrap();
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BooleanArray::from_iter(
                (0..10).map(|v| Some(v % 3 == 0)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, true, true, false, true, true, false, true, true, false
            ])
        );
    }

    #[test]
    fn test_sql_cast() {
        let cases = &[
            (
                "x = arrow_cast('2021-01-01 00:00:00', 'Timestamp(Microsecond, None)')",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                "x = arrow_cast('2021-01-01 00:00:00', 'Timestamp(Second, None)')",
                ArrowDataType::Timestamp(TimeUnit::Second, None),
            ),
            (
                "x = arrow_cast('2021-01-01 00:00:00.123', 'Timestamp(Nanosecond, None)')",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            (
                "x = arrow_cast('2021-01-01 00:00:00.123', 'Timestamp(Nanosecond, None)')",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            ("x = cast('2021-01-01' as date)", ArrowDataType::Date32),
            (
                "x = cast('1.238' as decimal(9,3))",
                ArrowDataType::Decimal128(9, 3),
            ),
            ("x = cast(1 as float)", ArrowDataType::Float32),
            ("x = cast(1 as double)", ArrowDataType::Float64),
            ("x = cast(1 as tinyint)", ArrowDataType::Int8),
            ("x = cast(1 as smallint)", ArrowDataType::Int16),
            ("x = cast(1 as int)", ArrowDataType::Int32),
            ("x = cast(1 as integer)", ArrowDataType::Int32),
            ("x = cast(1 as bigint)", ArrowDataType::Int64),
            ("x = cast(1 as tinyint unsigned)", ArrowDataType::UInt8),
            ("x = cast(1 as smallint unsigned)", ArrowDataType::UInt16),
            ("x = cast(1 as int unsigned)", ArrowDataType::UInt32),
            ("x = cast(1 as integer unsigned)", ArrowDataType::UInt32),
            ("x = cast(1 as bigint unsigned)", ArrowDataType::UInt64),
            ("x = cast(1 as boolean)", ArrowDataType::Boolean),
            ("x = cast(1 as string)", ArrowDataType::Utf8),
        ];

        for (sql, expected_data_type) in cases {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "x",
                expected_data_type.clone(),
                true,
            )]));
            let planner = Planner::new(schema.clone());
            let expr = planner.parse_filter(sql).unwrap();

            // Get the thing after 'cast(` but before ' as'.
            let expected_value_str = sql
                .split("cast(")
                .nth(1)
                .unwrap()
                .split(" as")
                .next()
                .unwrap();
            // Remove any quote marks
            let expected_value_str = expected_value_str
                .split('\'')
                .nth(1)
                .unwrap_or(expected_value_str);

            match expr {
                Expr::BinaryExpr(BinaryExpr { right, .. }) => match right.as_ref() {
                    Expr::Cast(Cast { expr, data_type }) => {
                        match expr.as_ref() {
                            Expr::Literal(ScalarValue::Utf8(Some(value_str))) => {
                                assert_eq!(value_str, expected_value_str);
                            }
                            Expr::Literal(ScalarValue::Int64(Some(value))) => {
                                assert_eq!(*value, 1);
                            }
                            _ => panic!("Expected cast to be applied to literal"),
                        }
                        assert_eq!(data_type, expected_data_type);
                    }
                    _ => panic!("Expected right to be a cast"),
                },
                _ => panic!("Expected binary expression"),
            }
        }
    }

    #[test]
    fn test_sql_literals() {
        let cases = &[
            (
                "x = arrow_cast(timestamp '2021-01-01 00:00:00', 'Timestamp(Microsecond, None)')",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                "x = arrow_cast(timestamp '2021-01-01 00:00:00', 'Timestamp(Second, None)')",
                ArrowDataType::Timestamp(TimeUnit::Second, None),
            ),
            (
                "x = timestamp '2021-01-01 00:00:00.123'",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            ("x = date '2021-01-01'", ArrowDataType::Date32),
            ("x = decimal(9,3) '1.238'", ArrowDataType::Decimal128(9, 3)),
        ];

        for (sql, expected_data_type) in cases {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "x",
                expected_data_type.clone(),
                true,
            )]));
            let planner = Planner::new(schema.clone());
            let expr = planner.parse_filter(sql).unwrap();

            let expected_value_str = sql.split('\'').nth(1).unwrap();

            match expr {
                Expr::BinaryExpr(BinaryExpr { right, .. }) => match right.as_ref() {
                    Expr::Cast(Cast { expr, data_type }) => {
                        match expr.as_ref() {
                            Expr::Literal(ScalarValue::Utf8(Some(value_str))) => {
                                assert_eq!(value_str, expected_value_str);
                            }
                            // For timestamps we get an outer arrow_cast cast and an inner literal parsing cast
                            Expr::Cast(inner_cast) => match inner_cast.expr.as_ref() {
                                Expr::Literal(ScalarValue::Utf8(Some(value_str))) => {
                                    assert_eq!(value_str, expected_value_str);
                                }
                                _ => {
                                    panic!("Expected inner timestamp cast to be applied to literal")
                                }
                            },
                            _ => panic!("Expected cast to be applied to literal"),
                        }
                        assert_eq!(data_type, expected_data_type);
                    }
                    _ => panic!("Expected right to be a cast"),
                },
                _ => panic!("Expected binary expression"),
            }
        }
    }

    #[test]
    fn test_sql_comparison() {
        // Create a batch with all data types
        let batch: Vec<(&str, ArrayRef)> = vec![
            (
                "timestamp_s",
                Arc::new(TimestampSecondArray::from_iter_values(0..10)),
            ),
            (
                "timestamp_ms",
                Arc::new(TimestampMillisecondArray::from_iter_values(0..10)),
            ),
            (
                "timestamp_us",
                Arc::new(TimestampMicrosecondArray::from_iter_values(0..10)),
            ),
            (
                "timestamp_ns",
                Arc::new(TimestampNanosecondArray::from_iter_values(4995..5005)),
            ),
        ];
        let batch = RecordBatch::try_from_iter(batch).unwrap();

        let planner = Planner::new(batch.schema());

        // Each expression is meant to select the final 5 rows
        let expressions = &[
            "timestamp_s >= TIMESTAMP '1970-01-01 00:00:05'",
            "timestamp_ms >= TIMESTAMP '1970-01-01 00:00:00.005'",
            "timestamp_us >= TIMESTAMP '1970-01-01 00:00:00.000005'",
            "timestamp_ns >= TIMESTAMP '1970-01-01 00:00:00.000005'",
        ];

        let expected: ArrayRef = Arc::new(BooleanArray::from_iter(
            std::iter::repeat(Some(false))
                .take(5)
                .chain(std::iter::repeat(Some(true)).take(5)),
        ));
        for expression in expressions {
            // convert to physical expression
            let logical_expr = planner.parse_filter(expression).unwrap();
            let physical_expr = planner.create_physical_expr(&logical_expr).unwrap();

            // Evaluate and assert they have correct results
            let result = physical_expr.evaluate(&batch).unwrap();
            let result = result.into_array(batch.num_rows());
            assert_eq!(&expected, &result, "unexpected result for {}", expression);
        }
    }
}
