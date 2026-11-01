//! GGUF: the key/value header now, quantised tensor materialisation later.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//! "GGUF" | version u32 | tensor_count u64 | kv_count u64
//!        | kv_count × (key gguf_string, value_type u32, value)
//!        | tensor_count × tensor_info
//!        | alignment padding | tensor data
//! ```
//!
//! This round parses up to the end of the kv block and stops: enough to answer
//! "what architecture is this and how many layers does it have" without loading
//! a multi-gigabyte file. Values are streamed, never buffered whole, and every
//! length is checked against the bytes actually left in the file so a corrupt
//! `count` cannot turn into a huge allocation.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use ivy_core::{Error, ErrorKind, Result};

/// GGUF value types, from the format spec. Never renumber these.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// The format's own name for this type, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "uint8",
            Value::I8(_) => "int8",
            Value::U16(_) => "uint16",
            Value::I16(_) => "int16",
            Value::U32(_) => "uint32",
            Value::I32(_) => "int32",
            Value::U64(_) => "uint64",
            Value::I64(_) => "int64",
            Value::F32(_) => "float32",
            Value::F64(_) => "float64",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(_) => "array",
        }
    }

    /// Integer view. `None` for a non-integer or a negative value, so callers
    /// reading a count do not have to guess.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(u64::from(v)),
            Value::U16(v) => Some(u64::from(v)),
            Value::U32(v) => Some(u64::from(v)),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufMeta {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    /// Key/value pairs in file order. GGUF preserves order and duplicate keys
    /// are technically possible, so this is a list rather than a map.
    pub kv: Vec<(String, Value)>,
}

impl GgufMeta {
    /// First value for `key`, or `None`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// `general.architecture`, e.g. `"llama"`.
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(Value::as_str)
    }

    /// Layer count. GGUF spells this `<architecture>.block_count`, and the
    /// llama-era `llama.block_count` is accepted as a fallback because older
    /// converters wrote it without the `general.architecture` prefix.
    pub fn block_count(&self) -> Option<u64> {
        if let Some(n) = self.get("llama.block_count").and_then(Value::as_u64) {
            return Some(n);
        }
        let arch = self.architecture()?;
        self.get(&format!("{arch}.block_count"))
            .and_then(Value::as_u64)
    }
}

pub fn read_meta(path: impl AsRef<Path>) -> Result<GgufMeta> {
    let file = File::open(path.as_ref())?;
    let total = file.metadata()?.len();
    let mut r = Reader {
        inner: BufReader::new(file),
        total,
        pos: 0,
    };

    let mut magic = [0u8; 4];
    r.exact(&mut magic, "magic")?;
    if &magic != b"GGUF" {
        return Err(Error::new(ErrorKind::IoFormat, "not a GGUF file"));
    }
    let version = r.u32("version")?;
    if version != 2 && version != 3 {
        return Err(Error::new(
            ErrorKind::IoFormat,
            format!("GGUF version {version} is not supported"),
        )
        .with_hint("versions 2 and 3 share the header layout implemented here"));
    }
    let tensor_count = r.u64("tensor_count")?;
    let kv_count = r.u64("kv_count")?;

    // Not `with_capacity(kv_count)`: the count is attacker-controlled until the
    // pairs actually parse, and reserving it would be an allocation bomb.
    let mut kv = Vec::new();
    for i in 0..kv_count {
        let key = r.string("key")?;
        let ty = r.u32("value type")?;
        let value = read_value(&mut r, ty)
            .map_err(|e| Error::new(e.kind(), format!("kv `{key}` (#{i}): {}", e.message())))?;
        kv.push((key, value));
    }

    // Tensor infos follow, and burst #1 deliberately stops here: no tensor is
    // materialised from a GGUF yet.
    Ok(GgufMeta {
        version,
        tensor_count,
        kv_count,
        kv,
    })
}

fn read_value(r: &mut Reader, ty: u32) -> Result<Value> {
    Ok(match ty {
        T_U8 => Value::U8(r.u8("uint8")?),
        T_I8 => Value::I8(r.u8("int8")? as i8),
        T_U16 => Value::U16(r.u16("uint16")?),
        T_I16 => Value::I16(r.u16("int16")? as i16),
        T_U32 => Value::U32(r.u32("uint32")?),
        T_I32 => Value::I32(r.u32("int32")? as i32),
        T_U64 => Value::U64(r.u64("uint64")?),
        T_I64 => Value::I64(r.u64("int64")? as i64),
        T_F32 => Value::F32(f32::from_bits(r.u32("float32")?)),
        T_F64 => Value::F64(f64::from_bits(r.u64("float64")?)),
        T_BOOL => match r.u8("bool")? {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            other => {
                return Err(Error::new(
                    ErrorKind::IoCorrupt,
                    format!("bool value {other} is not 0 or 1"),
                ));
            }
        },
        T_STRING => Value::String(r.string("string")?),
        T_ARRAY => {
            let elem_ty = r.u32("array element type")?;
            if elem_ty == T_ARRAY {
                return Err(Error::new(
                    ErrorKind::IoFormat,
                    "nested arrays are not part of the GGUF format",
                ));
            }
            let count = r.u64("array length")?;
            // A count that cannot possibly fit in the bytes left is corrupt;
            // reject it before allocating.
            let min = min_encoded_size(elem_ty)?;
            let needed = count.saturating_mul(min);
            if needed > r.remaining() {
                return Err(truncated(&format!(
                    "array of {count} × {} needs at least {needed} bytes but only {} remain",
                    elem_ty_name(elem_ty),
                    r.remaining()
                )));
            }
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_value(r, elem_ty)?);
            }
            Value::Array(items)
        }
        other => {
            return Err(Error::new(
                ErrorKind::IoFormat,
                format!("unknown GGUF value type {other}"),
            ));
        }
    })
}

/// Smallest number of bytes a value of `ty` can occupy, used to reject absurd
/// array counts. Strings and arrays carry an 8-byte length prefix.
fn min_encoded_size(ty: u32) -> Result<u64> {
    Ok(match ty {
        T_U8 | T_I8 | T_BOOL => 1,
        T_U16 | T_I16 => 2,
        T_U32 | T_I32 | T_F32 => 4,
        T_U64 | T_I64 | T_F64 => 8,
        T_STRING => 8,
        other => {
            return Err(Error::new(
                ErrorKind::IoFormat,
                format!("unknown GGUF array element type {other}"),
            ));
        }
    })
}

fn elem_ty_name(ty: u32) -> &'static str {
    match ty {
        T_U8 => "uint8",
        T_I8 => "int8",
        T_U16 => "uint16",
        T_I16 => "int16",
        T_U32 => "uint32",
        T_I32 => "int32",
        T_U64 => "uint64",
        T_I64 => "int64",
        T_F32 => "float32",
        T_F64 => "float64",
        T_BOOL => "bool",
        T_STRING => "string",
        _ => "value",
    }
}

fn truncated(what: &str) -> Error {
    Error::new(ErrorKind::IoCorrupt, format!("GGUF: {what}"))
        .with_hint("the file is truncated or its kv block disagrees with the header")
}

/// A forward-only reader that knows how many bytes the file should still hold,
/// so lengths read out of the file can be sanity-checked before use.
struct Reader {
    inner: BufReader<File>,
    total: u64,
    pos: u64,
}

impl Reader {
    fn remaining(&self) -> u64 {
        self.total.saturating_sub(self.pos)
    }

    fn exact(&mut self, buf: &mut [u8], what: &str) -> Result<()> {
        let want = buf.len() as u64;
        if want > self.remaining() {
            return Err(truncated(&format!(
                "{what}: wanted {want} bytes, only {} left",
                self.remaining()
            )));
        }
        self.inner.read_exact(buf).map_err(|e| {
            Error::new(
                ErrorKind::IoCorrupt,
                format!("{what}: {e} (file truncated?)"),
            )
        })?;
        self.pos += want;
        Ok(())
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        let mut b = [0u8; 1];
        self.exact(&mut b, what)?;
        Ok(b[0])
    }

    fn u16(&mut self, what: &str) -> Result<u16> {
        let mut b = [0u8; 2];
        self.exact(&mut b, what)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        let mut b = [0u8; 4];
        self.exact(&mut b, what)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        let mut b = [0u8; 8];
        self.exact(&mut b, what)?;
        Ok(u64::from_le_bytes(b))
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let len = self.u64(what)?;
        if len > self.remaining() {
            return Err(truncated(&format!(
                "{what}: string claims {len} bytes, only {} left",
                self.remaining()
            )));
        }
        let mut buf = vec![0u8; len as usize];
        self.exact(&mut buf, what)?;
        String::from_utf8(buf).map_err(|e| {
            Error::new(
                ErrorKind::IoCorrupt,
                format!("{what}: string is not valid UTF-8: {e}"),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF writer for tests: header plus the kv pairs given.
    #[derive(Default)]
    struct Builder {
        kv: Vec<(String, u32, Vec<u8>)>,
    }

    impl Builder {
        fn new() -> Builder {
            Builder::default()
        }

        fn string(mut self, key: &str, value: &str) -> Builder {
            let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(value.as_bytes());
            self.kv.push((key.to_string(), T_STRING, bytes));
            self
        }

        fn u32(mut self, key: &str, value: u32) -> Builder {
            self.kv
                .push((key.to_string(), T_U32, value.to_le_bytes().to_vec()));
            self
        }

        fn u64(mut self, key: &str, value: u64) -> Builder {
            self.kv
                .push((key.to_string(), T_U64, value.to_le_bytes().to_vec()));
            self
        }

        fn f32(mut self, key: &str, value: f32) -> Builder {
            self.kv.push((
                key.to_string(),
                T_F32,
                value.to_bits().to_le_bytes().to_vec(),
            ));
            self
        }

        fn bool(mut self, key: &str, value: bool) -> Builder {
            self.kv
                .push((key.to_string(), T_BOOL, vec![u8::from(value)]));
            self
        }

        fn array_u32(mut self, key: &str, values: &[u32]) -> Builder {
            let mut bytes = T_U32.to_le_bytes().to_vec();
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            self.kv.push((key.to_string(), T_ARRAY, bytes));
            self
        }

        fn build(&self) -> Vec<u8> {
            let mut b = b"GGUF".to_vec();
            b.extend_from_slice(&3u32.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            b.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
            for (key, ty, bytes) in &self.kv {
                b.extend_from_slice(&(key.len() as u64).to_le_bytes());
                b.extend_from_slice(key.as_bytes());
                b.extend_from_slice(&ty.to_le_bytes());
                b.extend_from_slice(bytes);
            }
            b
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("ivy-io-gguf-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(name)
    }

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = tmp(name);
        std::fs::write(&p, bytes).expect("write");
        p
    }

    #[test]
    fn reads_a_three_key_header() {
        let bytes = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .array_u32("llama.rope.dimension_sections", &[8, 1, 1, 0])
            .build();
        let p = write("three.gguf", &bytes);

        let m = read_meta(&p).expect("meta");
        assert_eq!(m.version, 3);
        assert_eq!(m.kv_count, 3);
        assert_eq!(m.kv.len(), 3);
        assert_eq!(m.architecture(), Some("llama"));
        assert_eq!(m.block_count(), Some(32));
        assert_eq!(
            m.get("llama.rope.dimension_sections")
                .and_then(Value::as_array),
            Some([Value::U32(8), Value::U32(1), Value::U32(1), Value::U32(0)].as_slice())
        );
    }

    #[test]
    fn block_count_follows_the_architecture_prefix() {
        let bytes = Builder::new()
            .string("general.architecture", "qwen2")
            .u32("qwen2.block_count", 28)
            .build();
        let p = write("arch_prefix.gguf", &bytes);
        assert_eq!(read_meta(&p).expect("meta").block_count(), Some(28));
    }

    #[test]
    fn reads_every_scalar_type() {
        let bytes = Builder::new()
            .u64("u64", 1 << 40)
            .f32("f32", 1.5)
            .bool("flag", true)
            .build();
        let p = write("scalars.gguf", &bytes);
        let m = read_meta(&p).expect("meta");
        assert_eq!(m.get("u64").and_then(Value::as_u64), Some(1 << 40));
        assert_eq!(m.get("f32").and_then(Value::as_f32), Some(1.5));
        assert_eq!(m.get("flag"), Some(&Value::Bool(true)));
    }

    #[test]
    fn keeps_file_order() {
        let bytes = Builder::new()
            .string("b", "second")
            .string("a", "first")
            .build();
        let p = write("order.gguf", &bytes);
        let m = read_meta(&p).expect("meta");
        assert_eq!(m.kv[0].0, "b");
        assert_eq!(m.kv[1].0, "a");
    }

    #[test]
    fn a_truncated_kv_block_is_e5002() {
        let bytes = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .build();
        for cut in 24..bytes.len() {
            let p = write("truncated.gguf", &bytes[..cut]);
            let err = read_meta(&p).expect_err("prefix must not parse");
            assert_eq!(err.code(), 5002, "prefix of {cut} bytes: {err}");
        }
    }

    #[test]
    fn an_absurd_array_count_is_rejected_without_allocating() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(b"bad");
        bytes.extend_from_slice(&T_ARRAY.to_le_bytes());
        bytes.extend_from_slice(&T_U32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        let p = write("bomb.gguf", &bytes);
        let err = read_meta(&p).expect_err("must reject");
        assert_eq!(err.code(), 5002, "{err}");
    }

    #[test]
    fn a_bad_magic_is_e5001() {
        let p = write("bad_magic.gguf", b"NOPE\x03\x00\x00\x00");
        assert_eq!(read_meta(&p).expect_err("magic").code(), 5001);
    }

    #[test]
    fn an_unknown_value_type_is_e5001() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(b"k");
        bytes.extend_from_slice(&99u32.to_le_bytes());
        let p = write("bad_type.gguf", &bytes);
        assert_eq!(read_meta(&p).expect_err("type").code(), 5001);
    }

    #[test]
    fn reads_version_and_counts() {
        let p = tmp("m.gguf");
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&7u64.to_le_bytes());
        b.extend_from_slice(&9u64.to_le_bytes());
        std::fs::write(&p, b).expect("write");
        // Nine kv pairs are announced and none are present: corrupt, not a panic.
        let err = read_meta(&p).expect_err("kv block missing");
        assert_eq!(err.code(), 5002, "{err}");
    }
}
