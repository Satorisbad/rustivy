//! `ivy` — the run-control surface agents use.
//!
//! Every subcommand prints one JSON object on stdout and exits non-zero on failure
//! with `{"code":"Exxxx",...}`. No prose, no spinners, no progress bars: something
//! has to be parseable.

use clap::{Parser, Subcommand};
use ivy_kernel::{BinaryOp, Emitter, Program, Scalar};
use ivy_tensor::{DType, Tensor};

#[derive(Parser)]
#[command(name = "ivy", about = "RustIVY run control", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report op coverage against the aten surface.
    Coverage {
        /// Path to a JSON list of aten ops with a `needs_kernel` flag.
        #[arg(long)]
        ops: Option<String>,
    },
    /// Emit a kernel program and print the backend source.
    Kernel {
        /// Program name, e.g. ivy_add_f32.
        #[arg(long)]
        name: String,
        /// PTX target SM (75, 80, 89, 90).
        #[arg(long, default_value_t = 75)]
        sm: u16,
        #[arg(long, default_value = "add")]
        op: String,
        /// Print the emitted source instead of a summary.
        #[arg(long)]
        dump: bool,
    },
    /// Benchmark one kernel or one model step.
    Bench {
        #[arg(long, default_value = "matmul")]
        what: String,
        #[arg(long, default_value_t = 4096)]
        size: usize,
    },
    /// Run a training configuration.
    Run {
        #[arg(long)]
        config: String,
        #[arg(long, default_value_t = false)]
        resume: bool,
    },
    /// Inspect a checkpoint directory.
    Checkpoint {
        #[arg(long)]
        dir: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let out = match run(cli) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e.to_json());
            std::process::exit(1);
        }
    };
    println!("{out}");
}

fn run(cli: Cli) -> ivy_core::Result<String> {
    match cli.command {
        Command::Coverage { ops } => {
            let total: usize = 2590;
            let implemented: usize = 0;
            let body = serde_json::json!({
                "aten_ops": total,
                "implemented": implemented,
                "remaining": total - implemented,
                "percent": 0.0,
                "source": ops.unwrap_or_else(|| "builtin".into()),
            });
            Ok(body.to_string())
        }
        Command::Kernel { name, sm, op, dump } => {
            let binop = match op.as_str() {
                "add" => BinaryOp::Add,
                "sub" => BinaryOp::Sub,
                "mul" => BinaryOp::Mul,
                "div" => BinaryOp::Div,
                "max" => BinaryOp::Max,
                "min" => BinaryOp::Min,
                other => {
                    return Err(ivy_core::Error::new(
                        ivy_core::ErrorKind::Unsupported,
                        format!("unknown binary op `{other}`"),
                    )
                    .with_hint("add | sub | mul | div | max | min"));
                }
            };
            let p = Program::elementwise_binary(name, Scalar::F32, binop);
            let emitter = ivy_cuda::PtxEmitter { sm };
            let src = emitter.emit(&p)?;
            if dump {
                Ok(src)
            } else {
                Ok(serde_json::json!({
                    "program": p.name,
                    "target": emitter.target().name(),
                    "bytes": src.len(),
                    "lines": src.lines().count(),
                })
                .to_string())
            }
        }
        Command::Bench { what, size } => {
            // Burst #1 has no kernels to time; report the scaffold honestly.
            let t = Tensor::zeros([size.min(8), size.min(8)], DType::F32)?;
            Ok(serde_json::json!({
                "what": what,
                "size": size,
                "elements": t.numel(),
                "gflops": null,
                "status": "no kernel yet",
            })
            .to_string())
        }
        Command::Run { config, resume } => Ok(serde_json::json!({
            "config": config,
            "resume": resume,
            "status": "training loop not implemented",
            "next": "burst #3",
        })
        .to_string()),
        Command::Checkpoint { dir } => {
            let state = ivy_io::run_state::RunState::read(&dir)?;
            Ok(serde_json::to_string(&state).unwrap_or_default())
        }
    }
}
