//! safetensors: header parsing and tensor materialisation.
//!
//! On disk a safetensors file is `u64` little-endian header length, that many
//! bytes of JSON, then the tensor payload. Each entry's `data_offsets` are
//! relative to the start of the payload, not the start of the file, and the
//! format does not require the ranges to be sorted or contiguous — so every
//! offset is checked against the real file length before it is dereferenced.
//!
//! Reading goes through a memory map: one `mmap` per file, no copy of the whole
//! checkpoint, and the payload slice for the requested tensor is copied out
//! exactly once into the `Tensor`.

use std::path::Path;

use ivy_core::{DType, Error, ErrorKind, Result, Shape};
use ivy_tensor::Tensor;
use serde::Deserialize;

/// A header larger than this is not a safetensors file, it is a corrupt length.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

impl TensorEntry {
    /// The entry's dtype, restricted to the ones this build can materialise.
    ///
    /// An entry naming `F64` or `I8` is a well-formed file that this build
    /// cannot hold, so it is `E2002` (dtype capability), not `E5001` (format).
    pub fn dtype(&self) -> Result<DType> {
        match self.dtype.as_str() {
            "F32" => Ok(DType::F32),
            "BF16" => Ok(DType::Bf16),
            "F16" => Ok(DType::F16),
            other => Err(Error::new(
                ErrorKind::UnsupportedDType,
                format!("safetensors dtype `{other}` is not supported by this build"),
            )
            .with_hint("burst #1 materialises F32, BF16 and F16 only")),
        }
    }

    /// Element count, checked: a hostile `shape` must not overflow into a
    /// too-small byte count and hand back a short buffer.
    pub fn numel(&self) -> Result<usize> {
        let mut n = 1usize;
        for &d in &self.shape {
            n = n.checked_mul(d).ok_or_else(|| {
                Error::new(
                    ErrorKind::IoCorrupt,
                    format!("tensor shape {:?} overflows usize", self.shape),
                )
            })?;
        }
        Ok(n)
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub tensors: Vec<(String, TensorEntry)>,
    pub metadata_bytes: u64,
}

impl Header {
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, e)| e)
    }
}

/// Parse the JSON header of a safetensors file without touching tensor data.
pub fn read_header(path: impl AsRef<Path>) -> Result<Header> {
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    read_exact(&mut f, &mut len, "header length")?;
    let len = u64::from_le_bytes(len);
    check_header_len(len)?;
    let mut buf = vec![0u8; len as usize];
    read_exact(&mut f, &mut buf, "header JSON")?;
    parse_header_json(&buf, len)
}

/// A file that ends mid-read is truncated (`E5002`); anything else is an I/O or
/// format problem (`E5001`). `read_tensor` uses the same rule, so both entry
/// points classify the same file the same way.
fn read_exact(f: &mut impl std::io::Read, buf: &mut [u8], what: &str) -> Result<()> {
    f.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            truncated(&format!("{what} runs past the end of the file"))
        } else {
            Error::new(ErrorKind::IoFormat, format!("{what}: {e}"))
        }
    })
}

fn check_header_len(len: u64) -> Result<()> {
    if len == 0 || len > MAX_HEADER_BYTES {
        return Err(Error::new(
            ErrorKind::IoFormat,
            format!("implausible safetensors header length {len}"),
        ));
    }
    Ok(())
}

fn parse_header_json(buf: &[u8], len: u64) -> Result<Header> {
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(buf).map_err(|e| {
            Error::new(
                ErrorKind::IoFormat,
                format!("safetensors header is not valid JSON: {e}"),
            )
        })?;

    let mut tensors = Vec::new();
    for (name, value) in map {
        if name == "__metadata__" {
            continue;
        }
        let entry: TensorEntry = serde_json::from_value(value).map_err(|e| {
            Error::new(
                ErrorKind::IoFormat,
                format!("tensor `{name}` has a malformed entry: {e}"),
            )
        })?;
        tensors.push((name, entry));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Header {
        tensors,
        metadata_bytes: len,
    })
}

/// Materialise one tensor.
///
/// Works for `F32` today. `BF16` and `F16` entries are parsed and bounds-checked
/// and then refused with `E9002`, because `ivy-tensor` has no public constructor
/// for a `Tensor` whose storage is not f32; use [`read_tensor_as_f32`] for those
/// (an exact widening), or teach `ivy-tensor` a `Tensor::from_storage`.
pub fn read_tensor(path: impl AsRef<Path>, name: &str) -> Result<Tensor> {
    let (entry, bytes) = read_tensor_bytes(path.as_ref(), name)?;
    let dtype = entry.dtype()?;
    let shape = Shape::new(entry.shape.clone());
    match dtype {
        DType::F32 => Tensor::from_f32(&decode_f32(&bytes), shape),
        other => Err(Error::new(
            ErrorKind::Unsupported,
            format!(
                "tensor `{name}` is {} and ivy-tensor cannot build non-f32 storage from bytes",
                other.name()
            ),
        )
        .with_hint(
            "use safetensors::read_tensor_as_f32, which widens bf16/f16 to f32 exactly; \
             the native-dtype path needs a Tensor::from_storage constructor in ivy-tensor",
        )),
    }
}

/// Materialise one tensor as f32, widening `BF16`/`F16` exactly.
///
/// Both formats are a strict subset of f32's range and precision, so the values
/// are bit-exact: this is a representation change, not a rounding.
pub fn read_tensor_as_f32(path: impl AsRef<Path>, name: &str) -> Result<Tensor> {
    let (entry, bytes) = read_tensor_bytes(path.as_ref(), name)?;
    let values = match entry.dtype()? {
        DType::F32 => decode_f32(&bytes),
        DType::Bf16 => decode_bf16(&bytes),
        DType::F16 => decode_f16(&bytes),
        other => {
            return Err(Error::new(
                ErrorKind::UnsupportedDType,
                format!("cannot widen {} to f32", other.name()),
            ));
        }
    };
    Tensor::from_f32(&values, Shape::new(entry.shape.clone()))
}

/// Shared path: map the file, find `name`, validate every offset, copy the bytes.
fn read_tensor_bytes(path: &Path, name: &str) -> Result<(TensorEntry, Vec<u8>)> {
    let map = map_file(path)?;
    let file_len = map.len() as u64;

    let len_bytes: [u8; 8] = map
        .get(..8)
        .ok_or_else(|| truncated("file is shorter than the 8-byte header length"))?
        .try_into()
        .map_err(|_| truncated("file is shorter than the 8-byte header length"))?;
    let header_len = u64::from_le_bytes(len_bytes);
    check_header_len(header_len)?;

    let header_end = 8u64
        .checked_add(header_len)
        .ok_or_else(|| truncated("header length overflows the file offset space"))?;
    if header_end > file_len {
        return Err(truncated(&format!(
            "header claims {header_len} bytes but only {} remain",
            file_len.saturating_sub(8)
        )));
    }
    let header = parse_header_json(
        map.get(8..header_end as usize)
            .ok_or_else(|| truncated("header range falls outside the mapping"))?,
        header_len,
    )?;

    let entry = header.get(name).ok_or_else(|| {
        Error::new(
            ErrorKind::IoFormat,
            format!("no tensor named `{name}` in {}", path.display()),
        )
        .with_hint("read_header() lists every name in the file")
    })?;

    let dtype = entry.dtype()?;
    let numel = entry.numel()?;
    let expected = (numel as u64)
        .checked_mul(dtype.size_bytes() as u64)
        .ok_or_else(|| truncated("tensor byte size overflows u64"))?;

    let [start, end] = entry.data_offsets;
    if end < start {
        return Err(truncated(&format!(
            "tensor `{name}` has data_offsets [{start}, {end}] with end < start"
        )));
    }
    let declared = end - start;
    if declared != expected {
        return Err(truncated(&format!(
            "tensor `{name}` declares {declared} bytes but shape {:?} of {} needs {expected}",
            entry.shape,
            dtype.name()
        )));
    }

    // Payload-relative offsets are only meaningful once added to the header end.
    let abs_start = header_end
        .checked_add(start)
        .ok_or_else(|| truncated("tensor start offset overflows the file offset space"))?;
    let abs_end = header_end
        .checked_add(end)
        .ok_or_else(|| truncated("tensor end offset overflows the file offset space"))?;
    if abs_end > file_len {
        return Err(truncated(&format!(
            "tensor `{name}` needs bytes {abs_start}..{abs_end} but the file is {file_len} bytes"
        )));
    }

    let bytes = map
        .get(abs_start as usize..abs_end as usize)
        .ok_or_else(|| truncated("tensor range falls outside the mapping"))?
        .to_vec();
    Ok((entry.clone(), bytes))
}

fn truncated(what: &str) -> Error {
    Error::new(ErrorKind::IoCorrupt, format!("safetensors: {what}"))
        .with_hint("the file is truncated or its header disagrees with its payload; re-download it")
}

/// Map a file read-only.
///
/// `memmap2::Mmap::map` is `unsafe` because the caller promises the file is not
/// mutated while the map lives. A checkpoint is written once and never edited in
/// place, and the map is confined to this function — the payload is copied out
/// and the map dropped before `read_tensor_bytes` returns — so no caller can
/// observe a torn read.
#[allow(unsafe_code)]
fn map_file(path: &Path) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len == 0 {
        return Err(Error::new(
            ErrorKind::IoFormat,
            format!("{} is empty", path.display()),
        ));
    }
    // SAFETY: we hold the only handle used to map this file, we never write
    // through it, and the mapping is dropped inside this function. Every access
    // is bounds-checked against `map.len()` above, so a file that shrinks under
    // us cannot produce an out-of-bounds read here.
    unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
        Error::new(
            ErrorKind::IoFormat,
            format!("cannot mmap {}: {e}", path.display()),
        )
    })
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    // Length is validated as a multiple of 4 by `read_tensor_bytes`, so the
    // remainder `as_chunks` reports is always empty.
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// bf16 is the top 16 bits of an f32, so widening is a shift and is exact.
fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
        .collect()
}

/// IEEE binary16 to f32. Exact: every f16 value is representable in f32.
fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x03ff);
    let out = match exp {
        // Zero or subnormal: frac * 2^-24 is exact in f32.
        0 => {
            let v = frac as f32 * 2f32.powi(-24);
            return if bits & 0x8000 != 0 { -v } else { v };
        }
        // Inf and NaN keep their payload in the mantissa's high bits.
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        // Normal: rebias the exponent from 15 to 127.
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tensor to put in a test file: name, dtype string, shape, payload.
    struct Fixture<'a> {
        name: &'a str,
        dtype: &'a str,
        shape: Vec<usize>,
        payload: Vec<u8>,
    }

    /// Build a valid safetensors file the way a writer would, so the reader is
    /// tested against the real layout rather than a hand-copied one.
    fn write_file(path: &Path, tensors: &[Fixture<'_>]) -> Vec<u8> {
        let mut entries = serde_json::Map::new();
        let mut payload = Vec::new();
        for t in tensors {
            let start = payload.len() as u64;
            payload.extend_from_slice(&t.payload);
            let end = payload.len() as u64;
            entries.insert(
                t.name.to_string(),
                serde_json::json!({
                    "dtype": t.dtype,
                    "shape": t.shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        let json = serde_json::to_vec(&serde_json::Value::Object(entries)).expect("json");
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&json);
        bytes.extend_from_slice(&payload);
        std::fs::write(path, &bytes).expect("write fixture");
        bytes
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("ivy-io-safetensors-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(name)
    }

    /// `Tensor` has no `Debug`, so `Result::expect_err` is unavailable; this is
    /// the same assertion without the bound.
    fn expect_error(result: Result<Tensor>, what: &str) -> Error {
        match result {
            Ok(_) => panic!("{what}: expected an error, got a tensor"),
            Err(e) => e,
        }
    }

    #[test]
    fn parses_a_minimal_header() {
        let p = tmp("h.safetensors");
        let json = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        std::fs::write(&p, bytes).expect("write");
        let h = read_header(&p).expect("header");
        assert_eq!(h.tensors.len(), 1);
        assert_eq!(h.tensors[0].0, "w");
        assert_eq!(h.tensors[0].1.shape, vec![2, 2]);
    }

    #[test]
    fn round_trips_a_2x3_f32_tensor() {
        let p = tmp("round_trip.safetensors");
        let values = [1.0f32, -2.5, 3.25, 0.0, 1e-3, 42.0];
        write_file(
            &p,
            &[Fixture {
                name: "w",
                dtype: "F32",
                shape: vec![2, 3],
                payload: f32_bytes(&values),
            }],
        );

        let t = read_tensor(&p, "w").expect("tensor");
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.as_f32().expect("f32"), &values);
    }

    #[test]
    fn reads_the_second_tensor_in_a_file() {
        let p = tmp("two.safetensors");
        write_file(
            &p,
            &[
                Fixture {
                    name: "a",
                    dtype: "F32",
                    shape: vec![2],
                    payload: f32_bytes(&[1.0, 2.0]),
                },
                Fixture {
                    name: "b",
                    dtype: "F32",
                    shape: vec![3],
                    payload: f32_bytes(&[3.0, 4.0, 5.0]),
                },
            ],
        );
        let t = read_tensor(&p, "b").expect("tensor");
        assert_eq!(t.as_f32().expect("f32"), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn a_truncated_payload_is_e5002() {
        let p = tmp("truncated.safetensors");
        let full = write_file(
            &p,
            &[Fixture {
                name: "w",
                dtype: "F32",
                shape: vec![2, 3],
                payload: f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            }],
        );
        // Drop the last 8 payload bytes: the header still claims all 24.
        std::fs::write(&p, &full[..full.len() - 8]).expect("write");

        let err = expect_error(read_tensor(&p, "w"), "must not read a short file");
        assert_eq!(err.code(), 5002, "{err}");
    }

    #[test]
    fn a_truncated_header_is_e5002() {
        let p = tmp("short_header.safetensors");
        let json = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        std::fs::write(&p, &bytes[..bytes.len() - 4]).expect("write");

        let err = expect_error(read_tensor(&p, "w"), "must not read a short header");
        assert_eq!(err.code(), 5002, "{err}");
        // Both entry points agree on how to classify the same file.
        assert_eq!(read_header(&p).expect_err("short header").code(), 5002);
    }

    #[test]
    fn offsets_that_disagree_with_the_shape_are_e5002() {
        let p = tmp("bad_offsets.safetensors");
        // 2x3 f32 needs 24 bytes; claim 20.
        let json = br#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,20]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        bytes.extend_from_slice(&[0u8; 20]);
        std::fs::write(&p, bytes).expect("write");

        let err = expect_error(read_tensor(&p, "w"), "offset/shape mismatch");
        assert_eq!(err.code(), 5002, "{err}");
    }

    #[test]
    fn offsets_past_the_end_of_file_are_e5002() {
        let p = tmp("past_eof.safetensors");
        // Consistent shape and length, but the payload was never written.
        let json = br#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        std::fs::write(&p, bytes).expect("write");

        let err = expect_error(read_tensor(&p, "w"), "payload missing");
        assert_eq!(err.code(), 5002, "{err}");
    }

    #[test]
    fn an_unknown_dtype_is_e2002() {
        let p = tmp("f64.safetensors");
        write_file(
            &p,
            &[Fixture {
                name: "w",
                dtype: "F64",
                shape: vec![1],
                payload: 1.0f64.to_le_bytes().to_vec(),
            }],
        );
        let err = expect_error(read_tensor(&p, "w"), "f64 unsupported");
        assert_eq!(err.code(), 2002, "{err}");
    }

    #[test]
    fn a_missing_name_is_e5001() {
        let p = tmp("missing.safetensors");
        write_file(
            &p,
            &[Fixture {
                name: "w",
                dtype: "F32",
                shape: vec![1],
                payload: f32_bytes(&[1.0]),
            }],
        );
        let err = expect_error(read_tensor(&p, "nope"), "no such tensor");
        assert_eq!(err.code(), 5001, "{err}");
    }

    #[test]
    fn bf16_widens_exactly_and_native_read_reports_the_gap() {
        let p = tmp("bf16.safetensors");
        // 1.0, -2.0, 0.5 in bf16 = the top halves of the f32 bit patterns.
        let values = [1.0f32, -2.0, 0.5];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        write_file(
            &p,
            &[Fixture {
                name: "w",
                dtype: "BF16",
                shape: vec![3],
                payload,
            }],
        );

        let t = read_tensor_as_f32(&p, "w").expect("widened");
        assert_eq!(t.as_f32().expect("f32"), &values);

        let err = expect_error(read_tensor(&p, "w"), "no bf16 storage constructor");
        assert_eq!(err.code(), 9002, "{err}");
        assert!(err.hint().contains("read_tensor_as_f32"), "{err}");
    }

    #[test]
    fn f16_widens_exactly() {
        // 1.0, -0.5, 65504 (f16 max), and a subnormal 2^-24.
        let cases: [(u16, f32); 4] = [
            (0x3c00, 1.0),
            (0xb800, -0.5),
            (0x7bff, 65504.0),
            (0x0001, 2f32.powi(-24)),
        ];
        let p = tmp("f16.safetensors");
        let payload: Vec<u8> = cases.iter().flat_map(|(b, _)| b.to_le_bytes()).collect();
        write_file(
            &p,
            &[Fixture {
                name: "w",
                dtype: "F16",
                shape: vec![4],
                payload,
            }],
        );
        let t = read_tensor_as_f32(&p, "w").expect("widened");
        let got = t.as_f32().expect("f32");
        for (i, (_, want)) in cases.iter().enumerate() {
            assert_eq!(got[i], *want, "case {i}");
        }
    }

    #[test]
    fn f16_specials_survive_the_widening() {
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    /// Every prefix of a valid file must be an error, never a panic and never a
    /// read past the end of the mapping.
    #[test]
    fn every_truncation_point_fails_cleanly() {
        let p = tmp("truncation_sweep.safetensors");
        let full = write_file(
            &p,
            &[Fixture {
                name: "w",
                dtype: "F32",
                shape: vec![3, 2],
                payload: f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            }],
        );
        for cut in 0..full.len() {
            std::fs::write(&p, &full[..cut]).expect("write");
            let err = expect_error(read_tensor(&p, "w"), "prefix must not read");
            assert!(
                err.code() == 5001 || err.code() == 5002,
                "prefix of {cut} bytes gave E{:04}: {err}",
                err.code()
            );
        }
    }
}
