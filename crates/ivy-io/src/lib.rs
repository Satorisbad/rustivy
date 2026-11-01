//! `ivy-io` — read what the ecosystem already wrote.
//!
//! Compatibility is a hard requirement: safetensors and GGUF must load without
//! Python. Burst #1 materialises f32 tensors from safetensors, parses the GGUF
//! key/value header, and carries the run-state sidecar a resume needs.
//!
//! `unsafe` is denied rather than forbidden because `memmap2`'s map constructors
//! are `unsafe fn`: the crate needs exactly one audited `mmap` call, in
//! `safetensors::map_file`, and nothing else.
#![deny(unsafe_code)]

pub mod gguf;
pub mod run_state;
pub mod safetensors;

use ivy_core::Result;
use std::path::Path;

/// Format detected from the first bytes of a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Gguf,
    Onnx,
    Unknown,
}

pub fn detect(path: impl AsRef<Path>) -> Result<Format> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 16];
    let n = f.read(&mut magic)?;
    if n >= 9 && looks_like_safetensors(&magic[..9]) {
        return Ok(Format::Safetensors);
    }
    if n >= 4 && &magic[..4] == b"GGUF" {
        return Ok(Format::Gguf);
    }
    if n >= 4 && magic[..4] == [0x08, 0x01, 0x12, 0x0a] {
        return Ok(Format::Onnx);
    }
    Ok(Format::Unknown)
}

fn looks_like_safetensors(first9: &[u8]) -> bool {
    let len = u64::from_le_bytes(first9[..8].try_into().unwrap_or_default());
    // A real header is a few bytes to a few MB, and the first payload byte is `{`.
    (1..=64 * 1024 * 1024).contains(&len) && first9.get(8) == Some(&b'{')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_magic_is_recognised() {
        let dir = std::env::temp_dir().join("ivy-io-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.gguf");
        std::fs::write(&p, b"GGUF\x03\x00\x00\x00").unwrap();
        assert_eq!(detect(&p).unwrap(), Format::Gguf);
    }

    #[test]
    fn safetensors_header_is_sniffed() {
        let dir = std::env::temp_dir().join("ivy-io-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let payload = b"{}";
        let mut bytes = (payload.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(payload);
        std::fs::write(&p, bytes).unwrap();
        assert_eq!(detect(&p).unwrap(), Format::Safetensors);
    }
}
