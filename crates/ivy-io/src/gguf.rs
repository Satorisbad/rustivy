//! GGUF: read the header now, quantised tensor materialisation later.

use std::path::Path;

use ivy_core::{Error, ErrorKind, Result};

#[derive(Debug, Clone)]
pub struct GgufMeta {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

pub fn read_meta(path: impl AsRef<Path>) -> Result<GgufMeta> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hdr = [0u8; 24];
    f.read_exact(&mut hdr)?;
    if &hdr[..4] != b"GGUF" {
        return Err(Error::new(ErrorKind::IoFormat, "not a GGUF file"));
    }
    let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    let tensor_count = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
    let kv_count = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
    Ok(GgufMeta {
        version,
        tensor_count,
        kv_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_version_and_counts() {
        let dir = std::env::temp_dir().join("ivy-io-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.gguf");
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&7u64.to_le_bytes());
        b.extend_from_slice(&9u64.to_le_bytes());
        std::fs::write(&p, b).unwrap();
        let m = read_meta(&p).unwrap();
        assert_eq!((m.version, m.tensor_count, m.kv_count), (3, 7, 9));
    }
}
