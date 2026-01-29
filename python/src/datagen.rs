use std::path::Path;

use arrow::pyarrow::PyArrowType;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use lance_datagen::{BatchCount, ByteCount, RowCount};
use pyo3::{
    pyfunction,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};

const DEFAULT_BATCH_SIZE_BYTES: u64 = 32 * 1024;
const DEFAULT_BATCH_COUNT: u32 = 4;

#[pyfunction]
pub fn is_datagen_supported() -> bool {
    true
}

#[pyfunction]
#[pyo3(signature=(schema, batch_count=None, bytes_in_batch=None))]
pub fn rand_batches(
    schema: PyArrowType<Schema>,
    batch_count: Option<u32>,
    bytes_in_batch: Option<u64>,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    lance_datagen::rand(&schema.0)
        .into_reader_bytes(
            ByteCount::from(bytes_in_batch.unwrap_or(DEFAULT_BATCH_SIZE_BYTES)),
            BatchCount::from(batch_count.unwrap_or(DEFAULT_BATCH_COUNT)),
            lance_datagen::RoundingBehavior::RoundUp,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to generate batches: {}", e))
        })?
        .map(|item| {
            item.map(PyArrowType::from).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to generate batch: {}", e))
            })
        })
        .collect::<PyResult<Vec<PyArrowType<RecordBatch>>>>()
}

#[pyfunction]
#[pyo3(signature=(yaml, num_rows, num_batches))]
pub fn from_yaml(
    yaml: &str,
    num_rows: u64,
    num_batches: u32,
) -> PyResult<PyArrowType<Box<dyn arrow_array::RecordBatchReader + Send>>> {
    let gen = lance_datagen::generator_from_yaml(yaml).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to parse YAML: {}", e))
    })?;
    let reader = lance_datagen::DataGeneratorReader::new(
        gen,
        RowCount::from(num_rows),
        BatchCount::from(num_batches),
    );
    Ok(PyArrowType(Box::new(reader)))
}

#[pyfunction]
#[pyo3(signature=(path, num_rows, num_batches))]
pub fn from_yaml_file(
    path: &str,
    num_rows: u64,
    num_batches: u32,
) -> PyResult<PyArrowType<Box<dyn arrow_array::RecordBatchReader + Send>>> {
    let gen = lance_datagen::generator_from_yaml_file(Path::new(path)).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to parse YAML file: {}", e))
    })?;
    let reader = lance_datagen::DataGeneratorReader::new(
        gen,
        RowCount::from(num_rows),
        BatchCount::from(num_batches),
    );
    Ok(PyArrowType(Box::new(reader)))
}

pub fn register_datagen(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let datagen = PyModule::new(py, "datagen")?;
    datagen.add_wrapped(wrap_pyfunction!(is_datagen_supported))?;
    datagen.add_wrapped(wrap_pyfunction!(rand_batches))?;
    datagen.add_wrapped(wrap_pyfunction!(from_yaml))?;
    datagen.add_wrapped(wrap_pyfunction!(from_yaml_file))?;
    m.add_submodule(&datagen)?;
    Ok(())
}
