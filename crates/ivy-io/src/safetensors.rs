//! safetensors header parsing (tensor bytes arrive in burst #2).

use std::path::Path;

use ivy_core::{Error, ErrorKind, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
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
}

/// Parse the JSON header of a safetensors file without touching tensor data.
pub fn read_header(path: impl AsRef<Path>) -> Result<Header> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    if len == 0 || len > 64 * 1024 * 1024 {
        return Err(Error::new(
            ErrorKind::IoFormat,
            format!("implausible safetensors header length {len}"),
        ));
    }
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf)?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&buf).map_err(|e| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_header() {
        let dir = std::env::temp_dir().join("ivy-io-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("h.safetensors");
        let json = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        std::fs::write(&p, bytes).unwrap();
        let h = read_header(&p).unwrap();
        assert_eq!(h.tensors.len(), 1);
        assert_eq!(h.tensors[0].0, "w");
        assert_eq!(h.tensors[0].1.shape, vec![2, 2]);
    }
}
