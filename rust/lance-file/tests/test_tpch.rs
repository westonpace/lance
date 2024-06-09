use std::{any::Any, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::{
    datasource::{TableProvider, TableType},
    execution::context::{SessionContext, SessionState},
    logical_expr::{Expr, TableProviderFilterPushDown},
};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::{
    stream::RecordBatchStreamAdapter, DisplayAs, ExecutionMode, ExecutionPlan, Partitioning,
    PlanProperties,
};
use futures::{stream::BoxStream, Stream, StreamExt};
use lance_encoding::decoder::ReadBatchTask;
use lance_file::v2::reader::{FileReader, ReaderProjection};
use lance_io::{object_store::ObjectStore, scheduler::ScanScheduler, ReadBatchParams};
use object_store::path::Path;

type LockedTaskStream = Arc<tokio::sync::Mutex<BoxStream<'static, ReadBatchTask>>>;

trait LanceResultExt<T> {
    fn to_df_err(self) -> Result<T>;
}

impl<T> LanceResultExt<T> for lance_core::Result<T> {
    fn to_df_err(self) -> Result<T> {
        self.map_err(|err| datafusion_common::DataFusionError::External(err.into()))
    }
}

struct LanceScanExec {
    properties: PlanProperties,
    output_schema: SchemaRef,
    tasks: LockedTaskStream,
}

impl std::fmt::Debug for LanceScanExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceScanExec")
            .field("properties", &self.properties)
            .finish()
    }
}

impl DisplayAs for LanceScanExec {
    fn fmt_as(
        &self,
        _t: datafusion_physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "LanceScanExec")
    }
}

const IO_PARALLELISM: u32 = 16;

async fn next_task(
    tasks: LockedTaskStream,
    _partition: usize,
) -> Option<(Result<RecordBatch>, LockedTaskStream)> {
    let mut tasks_ref = tasks.lock().await;
    let task = tasks_ref.next().await;
    drop(tasks_ref);
    if let Some(task) = task {
        Some((task.task.await.to_df_err(), tasks))
    } else {
        None
    }
}

fn scan_stream(
    tasks: LockedTaskStream,
    partition: usize,
) -> impl Stream<Item = Result<RecordBatch>> {
    futures::stream::unfold(tasks, move |tasks| next_task(tasks, partition)).fuse()
}

impl LanceScanExec {
    pub async fn try_new(
        num_partitions: u32,
        reader: &FileReader,
        projection: &Vec<usize>,
    ) -> Result<Self> {
        let schema = reader.schema();

        let base_projection = FileReader::default_projection(schema);
        let refined_projection = ReaderProjection {
            column_indices: projection
                .iter()
                .map(|&i| base_projection.column_indices[i])
                .collect(),
            schema: Arc::new(base_projection.schema.project(&projection)?),
        };
        let arrow_schema = refined_projection.schema.clone();

        let properties = PlanProperties::new(
            EquivalenceProperties::new(arrow_schema.clone()),
            Partitioning::RoundRobinBatch(num_partitions as usize),
            ExecutionMode::Bounded,
        );
        let tasks = reader
            .read_tasks(ReadBatchParams::RangeFull, 8192, &refined_projection)
            .to_df_err()?
            .fuse()
            .boxed();
        let tasks = Arc::new(tokio::sync::Mutex::new(tasks));
        println!("output_schema; {:#?}", arrow_schema);
        Ok(Self {
            properties,
            tasks,
            output_schema: arrow_schema,
        })
    }
}

impl ExecutionPlan for LanceScanExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &datafusion_physical_plan::PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self.clone())
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let stream = scan_stream(self.tasks.clone(), partition);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }
}

const CPU_PARALLELISM: u32 = 16;

struct LineItemTable {
    reader: FileReader,
    schema: SchemaRef,
}

impl LineItemTable {
    async fn try_new() -> Result<Self> {
        let object_store = Arc::new(ObjectStore::local());
        let scheduler = ScanScheduler::new(object_store, IO_PARALLELISM);
        let path = Path::parse("/tmp/lineitem_10.lance").unwrap();
        let reader = FileReader::try_open(scheduler.open_file(&path).await.to_df_err()?, None)
            .await
            .to_df_err()?;

        let schema = Arc::new(arrow_schema::Schema::from(reader.schema().as_ref()));
        Ok(Self { reader, schema })
    }
}

#[async_trait::async_trait]
impl TableProvider for LineItemTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &SessionState,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(
            LanceScanExec::try_new(CPU_PARALLELISM, &self.reader, projection.unwrap()).await?,
        ))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Unsupported)
            .collect())
    }
}

#[test]
fn test_tpch_fused() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
        .trace_style(tracing_chrome::TraceStyle::Async)
        .build();
    let subscriber = ::tracing_subscriber::registry::Registry::default();
    let subscriber = tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt::with(
        subscriber,
        chrome_layer,
    );
    let _sub_guard = ::tracing::subscriber::set_global_default(subscriber);

    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async move {
        let ctx = SessionContext::new();

        let line_item_table = LineItemTable::try_new().await.unwrap();
        ctx.register_table("lineitem", Arc::new(line_item_table))
            .unwrap();

        let start = std::time::Instant::now();
        ctx.sql("SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_shipdate >= date '1994-01-01' AND l_shipdate < date '1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24")
        .await
        .unwrap().collect().await.unwrap();

        println!("Finished query in {} seconds", start.elapsed().as_secs_f64());    
    });
}
