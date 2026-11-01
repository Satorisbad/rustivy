//! Where a tensor lives.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Device {
    Cpu,
    Cuda { index: u8 },
    Metal,
    Vulkan,
    Tpu { index: u8 },
}

impl Device {
    pub const CPU: Device = Device::Cpu;

    /// True for anything that is not host memory. Used by the op layer to decide
    /// whether a transfer is needed, never to change results silently.
    pub const fn is_accelerator(&self) -> bool {
        !matches!(self, Device::Cpu)
    }

    pub fn index(&self) -> Option<u8> {
        match self {
            Device::Cuda { index } | Device::Tpu { index } => Some(*index),
            _ => None,
        }
    }

    /// Parse `cpu`, `cuda`, `cuda:1`, `metal`, `vulkan`, `tpu:3`.
    pub fn parse(s: &str) -> Result<Device> {
        let (name, index) = match s.split_once(':') {
            Some((n, i)) => (
                n,
                Some(i.parse::<u8>().map_err(|_| {
                    Error::new(
                        crate::error::ErrorKind::DeviceParse,
                        format!("bad device index `{i}`"),
                    )
                })?),
            ),
            None => (s, None),
        };
        let idx = index.unwrap_or(0);
        match name {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda { index: idx }),
            "metal" => Ok(Device::Metal),
            "vulkan" => Ok(Device::Vulkan),
            "tpu" => Ok(Device::Tpu { index: idx }),
            other => Err(Error::new(
                crate::error::ErrorKind::DeviceParse,
                format!("unknown device `{other}`"),
            )
            .with_hint("expected one of: cpu, cuda[:N], metal, vulkan, tpu[:N]")),
        }
    }
}

impl core::fmt::Display for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Cuda { index } => write!(f, "cuda:{index}"),
            Device::Metal => f.write_str("metal"),
            Device::Vulkan => f.write_str("vulkan"),
            Device::Tpu { index } => write!(f, "tpu:{index}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Device;

    #[test]
    fn parses_and_round_trips() {
        for s in ["cpu", "cuda:0", "cuda:3", "metal", "vulkan", "tpu:1"] {
            let d = Device::parse(s).expect("parse");
            assert_eq!(d.to_string(), s);
        }
        // A bare `cuda` means device 0, and normalises to `cuda:0` when printed.
        assert_eq!(
            Device::parse("cuda").expect("parse"),
            Device::Cuda { index: 0 }
        );
        assert!(Device::parse("gpu").is_err());
    }
}
