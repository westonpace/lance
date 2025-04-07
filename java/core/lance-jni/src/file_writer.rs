use std::{
    future::Future,
    io::{Error as IoError, ErrorKind},
    pin::Pin,
    sync::{Arc, Mutex},
    task::Poll,
};

use crate::{
    error::{Error, Result},
    traits::IntoJava,
    utils::JvmRef,
    RT,
};
use arrow::{
    array::{RecordBatch, StructArray},
    ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema},
};
use arrow_schema::DataType;
use async_trait::async_trait;
use jni::{
    errors::Error as JniError,
    objects::{GlobalRef, JObject, JString, JValueGen},
    sys::jlong,
    JNIEnv, JavaVM,
};
use lance::io::ObjectStore;
use lance::{Error as LanceError, Result as LanceResult};
use lance_file::{
    v2::writer::{FileWriter, FileWriterOptions},
    version::LanceFileVersion,
};
use lance_io::traits::Writer;
use snafu::location;
use tokio::{io::AsyncWrite, task::JoinHandle};

pub const NATIVE_WRITER: &str = "nativeFileWriterHandle";

pub struct CallbackWriter {
    jvm: JvmRef,
    output_stream: GlobalRef,
    write_task: Option<JoinHandle<std::result::Result<(), JniError>>>,
    shutdown: bool,
}

impl CallbackWriter {
    pub fn new(env: &mut JNIEnv<'_>, output_stream: JObject) -> Result<Self> {
        let jvm = JvmRef::new(env.get_java_vm()?.get_java_vm_pointer());
        let output_stream = env.new_global_ref(output_stream)?;
        Ok(Self {
            jvm,
            output_stream,
            write_task: None,
            shutdown: false,
        })
    }
}

impl AsyncWrite for CallbackWriter {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::result::Result<usize, IoError>> {
        let this = self.get_mut();
        // A write may already be in progress, if so, block until it completes
        if let Some(mut write_task) = this.write_task.take() {
            let write_task_pin = Pin::new(&mut write_task);
            match write_task_pin.poll(cx) {
                Poll::Ready(Ok(Ok(_))) => {}
                Poll::Ready(Ok(Err(e))) => {
                    return std::task::Poll::Ready(Err(IoError::new(ErrorKind::Other, e)));
                }
                Poll::Ready(Err(e)) => {
                    return std::task::Poll::Ready(Err(IoError::new(ErrorKind::Other, e)));
                }
                Poll::Pending => {
                    this.write_task = Some(write_task);
                    return std::task::Poll::Pending;
                }
            }
        }
        if this.shutdown {
            return Poll::Ready(Err(IoError::new(ErrorKind::Other, "Writer is closed")));
        }
        let jvm = this.jvm;
        let output_stream = this.output_stream.clone();
        let num_bytes = buf.len();
        let buf = buf.to_vec();
        let write_task = tokio::task::spawn_blocking(move || {
            // Tell rust we are capturing jvm and not jvm.val
            let _ = &jvm;
            let jvm = unsafe { JavaVM::from_raw(jvm.val) }?;
            let mut jvm_env = jvm.attach_current_thread()?;
            let java_bytes = jvm_env.byte_array_from_slice(&buf)?;
            jvm_env.call_method(
                output_stream.as_obj(),
                "write",
                "([B)V",
                &[JValueGen::from(&java_bytes)],
            )?;
            Ok(())
        });
        this.write_task = Some(write_task);
        Poll::Ready(Ok(num_bytes))
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        let this = self.get_mut();
        if let Some(mut write_task) = this.write_task.take() {
            let write_task_pin = Pin::new(&mut write_task);
            match write_task_pin.poll(cx) {
                Poll::Ready(Ok(Ok(_))) => {}
                Poll::Ready(Ok(Err(e))) => {
                    return std::task::Poll::Ready(Err(IoError::new(ErrorKind::Other, e)));
                }
                Poll::Ready(Err(e)) => {
                    return std::task::Poll::Ready(Err(IoError::new(ErrorKind::Other, e)));
                }
                Poll::Pending => {
                    this.write_task = Some(write_task);
                    return std::task::Poll::Pending;
                }
            }
        }
        if this.shutdown {
            return Poll::Ready(Err(IoError::new(ErrorKind::Other, "Writer is closed")));
        }
        let jvm = this.jvm;
        let output_stream = this.output_stream.clone();
        let write_task = tokio::task::spawn_blocking(move || {
            let _ = &jvm;
            let jvm = unsafe { JavaVM::from_raw(jvm.val) }?;
            let mut jvm_env = jvm.attach_current_thread()?;
            jvm_env.call_method(output_stream.as_obj(), "flush", "()V", &[])?;
            Ok(())
        });
        this.write_task = Some(write_task);
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        let this = self.get_mut();
        if let Some(mut write_task) = this.write_task.take() {
            let write_task_pin = Pin::new(&mut write_task);
            match write_task_pin.poll(cx) {
                Poll::Ready(Ok(_)) => {}
                Poll::Ready(Err(e)) => {
                    return std::task::Poll::Ready(Err(IoError::new(ErrorKind::Other, e)));
                }
                Poll::Pending => {
                    this.write_task = Some(write_task);
                    return std::task::Poll::Pending;
                }
            }
        }
        if this.shutdown {
            return std::task::Poll::Ready(Ok(()));
        }
        this.shutdown = true;
        let jvm = this.jvm;
        let output_stream = this.output_stream.clone();
        let mut write_task = tokio::task::spawn_blocking(move || {
            let _ = &jvm;
            let jvm = unsafe { JavaVM::from_raw(jvm.val) }?;
            let mut jvm_env = jvm.attach_current_thread()?;
            jvm_env.call_method(output_stream.as_obj(), "close", "()V", &[])?;
            Ok(())
        });
        let tmp_write_task = Pin::new(&mut write_task);
        match tmp_write_task.poll(cx) {
            Poll::Ready(Ok(_)) => {
                return std::task::Poll::Ready(Ok(()));
            }
            Poll::Ready(Err(e)) => {
                return std::task::Poll::Ready(Err(IoError::new(ErrorKind::Other, e)));
            }
            Poll::Pending => {
                this.write_task = Some(write_task);
                return std::task::Poll::Pending;
            }
        }
    }
}

#[async_trait]
impl Writer for CallbackWriter {
    async fn tell(&mut self) -> LanceResult<usize> {
        if let Some(write_task) = self.write_task.take() {
            write_task.await?.map_err(|e| LanceError::Wrapped {
                error: Box::new(e),
                location: location!(),
            })?;
        }
        let jvm = self.jvm;
        let output_stream = self.output_stream.clone();
        tokio::task::spawn_blocking(move || {
            let _ = &jvm;
            let jvm = unsafe { JavaVM::from_raw(jvm.val) }
                .map_err(|e| IoError::new(ErrorKind::Other, e))?;
            let mut jvm_env = jvm
                .attach_current_thread()
                .map_err(|e| IoError::new(ErrorKind::Other, e))?;
            let res = jvm_env
                .call_method(output_stream.as_obj(), "tell", "()J", &[])
                .map_err(|e| IoError::new(ErrorKind::Other, e))?;
            Ok(res.j().map_err(|e| IoError::new(ErrorKind::Other, e))? as usize)
        })
        .await
        .unwrap()
    }
}

#[derive(Clone)]
pub struct BlockingFileWriter {
    pub(crate) inner: Arc<Mutex<FileWriter>>,
}

impl BlockingFileWriter {
    pub fn create(file_writer: FileWriter) -> Self {
        Self {
            inner: Arc::new(Mutex::new(file_writer)),
        }
    }
}

impl IntoJava for BlockingFileWriter {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        attach_native_writer(env, self)
    }
}

fn attach_native_writer<'local>(
    env: &mut JNIEnv<'local>,
    writer: BlockingFileWriter,
) -> Result<JObject<'local>> {
    let j_writer = create_java_writer_object(env)?;
    unsafe { env.set_rust_field(&j_writer, NATIVE_WRITER, writer) }?;
    Ok(j_writer)
}

fn create_java_writer_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let res = env.new_object("com/lancedb/lance/file/LanceFileWriter", "()V", &[])?;
    Ok(res)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileWriter_openNative<'local>(
    mut env: JNIEnv<'local>,
    _writer_class: JObject,
    file_uri: JString,
) -> JObject<'local> {
    ok_or_throw!(env, inner_open(&mut env, file_uri,))
}

fn inner_open<'local>(env: &mut JNIEnv<'local>, file_uri: JString) -> Result<JObject<'local>> {
    let file_uri_str: String = env.get_string(&file_uri)?.into();

    let writer = RT.block_on(async move {
        let (obj_store, path) = ObjectStore::from_uri(&file_uri_str).await?;
        let obj_store = Arc::new(obj_store);
        let obj_writer = obj_store.create(&path).await?;

        Result::Ok(FileWriter::new_lazy(
            obj_writer,
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            },
        ))
    })?;

    let writer = BlockingFileWriter::create(writer);

    writer.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileWriter_openNativeWithJavaIo<'local>(
    mut env: JNIEnv<'local>,
    _writer_class: JObject,
    output_stream: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_open_with_java_io(&mut env, output_stream))
}

fn inner_open_with_java_io<'local>(
    env: &mut JNIEnv<'local>,
    output_stream: JObject,
) -> Result<JObject<'local>> {
    let callback_writer = Box::new(CallbackWriter::new(env, output_stream)?);
    let writer = RT.block_on(async move {
        Result::Ok(FileWriter::new_custom_io(
            callback_writer,
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            },
        ))
    })?;

    let writer = BlockingFileWriter::create(writer);

    writer.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileWriter_closeNative<'local>(
    mut env: JNIEnv<'local>,
    writer: JObject,
) -> JObject<'local> {
    let maybe_err =
        unsafe { env.take_rust_field::<_, _, BlockingFileWriter>(writer, NATIVE_WRITER) };
    let writer = match maybe_err {
        Ok(writer) => Some(writer),
        // We were already closed, do nothing
        Err(jni::errors::Error::NullPtr(_)) => None,
        Err(err) => {
            Error::from(err).throw(&mut env);
            None
        }
    };
    if let Some(writer) = writer {
        match RT.block_on(writer.inner.lock().unwrap().finish()) {
            Ok(_) => {}
            Err(e) => {
                Error::from(e).throw(&mut env);
            }
        }
    }
    JObject::null()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileWriter_writeNative<'local>(
    mut env: JNIEnv<'local>,
    writer: JObject,
    batch_address: jlong,
    schema_address: jlong,
) -> JObject<'local> {
    if let Err(e) = inner_write_batch(&mut env, writer, batch_address, schema_address) {
        e.throw(&mut env);
        return JObject::null();
    }
    JObject::null()
}

fn inner_write_batch(
    env: &mut JNIEnv<'_>,
    writer: JObject,
    batch_address: jlong,
    schema_address: jlong,
) -> Result<()> {
    let c_array_ptr = batch_address as *mut FFI_ArrowArray;
    let c_schema_ptr = schema_address as *mut FFI_ArrowSchema;

    let c_array = unsafe { FFI_ArrowArray::from_raw(c_array_ptr) };
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };

    let data_type = DataType::try_from(&c_schema)?;
    let array_data = unsafe { from_ffi_and_data_type(c_array, data_type) }?;
    let record_batch = RecordBatch::from(StructArray::from(array_data));

    let writer = unsafe { env.get_rust_field::<_, _, BlockingFileWriter>(writer, NATIVE_WRITER) }?;

    let mut writer = writer.inner.lock().unwrap();
    RT.block_on(writer.write_batch(&record_batch))?;
    Ok(())
}
