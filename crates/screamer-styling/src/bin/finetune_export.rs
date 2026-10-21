//! Turn a Screamer telemetry directory into a Fireworks SFT dataset.
//!
//!     finetune-export <telemetry-dir> [--no-system-prompt] [--max-edit-ratio F] > sft.jsonl
//!
//! Emits one JSON line per labeled dictation in Fireworks' supervised
//! fine-tuning chat format: a `messages` array of system (the current styling
//! `SYSTEM_PROMPT`, so training matches inference), user (the raw transcript),
//! and assistant (the final text you settled on). Only the assistant turn is
//! trained; Fireworks masks system/user from the loss.
//!
//! The target is `settled_text` — what you actually kept — which is the
//! strongest "what I wanted" signal the telemetry captures. Rows without a
//! label, with an empty raw transcript, or whose correction rewrote more than
//! `--max-edit-ratio` of the words (likely a bad label) are dropped, and
//! identical (raw, settled) pairs are de-duplicated.
//!
//! DPO is not emitted yet: when Fireworks' preference line format is confirmed,
//! `settled_text` vs `pasted_text` (where they differ) is the natural
//! chosen/rejected source.

use std::collections::HashSet;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use screamer_styling::SYSTEM_PROMPT;
use screamer_telemetry::export_rows;

const DEFAULT_MAX_EDIT_RATIO: f32 = 0.8;

fn usage() {
    eprintln!(
        "usage: finetune-export <telemetry-dir> [--no-system-prompt] [--max-edit-ratio F] > sft.jsonl"
    );
}

fn main() -> ExitCode {
    let mut dir: Option<PathBuf> = None;
    let mut system = true;
    let mut max_edit_ratio = DEFAULT_MAX_EDIT_RATIO;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-system-prompt" => system = false,
            "--max-edit-ratio" => {
                let Some(v) = args.next() else {
                    eprintln!("--max-edit-ratio needs a value");
                    return ExitCode::from(2);
                };
                match v.parse::<f32>() {
                    Ok(f) if (0.0..=1.0).contains(&f) => max_edit_ratio = f,
                    _ => {
                        eprintln!("--max-edit-ratio must be a number in 0.0..=1.0");
                        return ExitCode::from(2);
                    }
                }
            }
            "-h" | "--help" => {
                usage();
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
        usage();
        return ExitCode::from(2);
    };

    let rows = match export_rows(&dir, /* labeled_only = */ true) {
        Ok(rows) => rows,
        Err(err) => {
            eprintln!("could not read {}: {err}", dir.display());
            return ExitCode::FAILURE;
        }
    };

    let out = io::stdout();
    let mut out = BufWriter::new(out.lock());
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let (mut written, mut skipped) = (0usize, 0usize);

    for row in &rows {
        let raw = row.raw_text.trim();
        let settled = row.settled_text.as_deref().unwrap_or("").trim();
        if raw.is_empty() || settled.is_empty() {
            skipped += 1;
            continue;
        }
        if row.edit_ratio.unwrap_or(0.0) > max_edit_ratio {
            skipped += 1;
            continue;
        }
        if !seen.insert((raw.to_string(), settled.to_string())) {
            skipped += 1;
            continue;
        }

        let mut messages = Vec::new();
        if system {
            messages.push(serde_json::json!({"role": "system", "content": SYSTEM_PROMPT}));
        }
        messages.push(serde_json::json!({"role": "user", "content": raw}));
        messages.push(serde_json::json!({"role": "assistant", "content": settled}));

        let line = serde_json::json!({ "messages": messages });
        let mut bytes = match serde_json::to_vec(&line) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("serialize failed: {err}");
                return ExitCode::FAILURE;
            }
        };
        bytes.push(b'\n');
        if let Err(err) = out.write_all(&bytes) {
            eprintln!("write failed: {err}");
            return ExitCode::FAILURE;
        }
        written += 1;
    }

    if let Err(err) = out.flush() {
        eprintln!("flush failed: {err}");
        return ExitCode::FAILURE;
    }

    eprintln!(
        "exported {written} example(s) ({skipped} skipped, system prompt: {})",
        if system { "included" } else { "omitted" }
    );
    if written < 3 {
        eprintln!("warning: Fireworks SFT needs at least 3 examples; this dataset has {written}.");
    }

    ExitCode::SUCCESS
}
