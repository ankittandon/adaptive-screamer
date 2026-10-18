//! Merge a Screamer telemetry directory into training-ready JSON lines.
//!
//!     screamer-telemetry-export <telemetry-dir> [--all] > labeled.jsonl
//!
//! By default only dictations that were corrected (have `settled_text`) are
//! written. `--all` includes the uncorrected ones too. `audio_path` stays
//! relative to the telemetry directory.

use std::io::{self, BufWriter};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut dir: Option<PathBuf> = None;
    let mut all = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--all" => all = true,
            "-h" | "--help" => {
                eprintln!("usage: screamer-telemetry-export <telemetry-dir> [--all] > out.jsonl");
                return ExitCode::SUCCESS;
            }
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => {
                eprintln!("unexpected argument: {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(dir) = dir else {
        eprintln!("usage: screamer-telemetry-export <telemetry-dir> [--all] > out.jsonl");
        return ExitCode::from(2);
    };

    let rows = match screamer_telemetry::export_rows(&dir, !all) {
        Ok(rows) => rows,
        Err(err) => {
            eprintln!("could not read {}: {err}", dir.display());
            return ExitCode::FAILURE;
        }
    };

    let mut out = BufWriter::new(io::stdout().lock());
    if let Err(err) = screamer_telemetry::write_export_jsonl(&rows, &mut out) {
        eprintln!("write failed: {err}");
        return ExitCode::FAILURE;
    }
    eprintln!("exported {} row(s)", rows.len());
    ExitCode::SUCCESS
}
