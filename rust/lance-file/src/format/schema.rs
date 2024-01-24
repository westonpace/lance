use lance_core::datatypes::Schema;

use crate::datatypes::{Fields, FieldsWithMeta};

use super::pb;

pub struct FileDescriptor {
    pub schema: Schema,
    pub length: u32,
}

impl From<pb::FileDescriptor> for FileDescriptor {
    fn from(value: pb::FileDescriptor) -> Self {
        let pb_schema = value.schema.unwrap();
        let fields = Fields(pb_schema.fields);
        let metadata = pb_schema.metadata;
        let schema = Schema::from(FieldsWithMeta { fields, metadata });
        Self {
            length: value.length,
            schema,
        }
    }
}
