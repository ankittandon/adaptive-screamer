//! Try the styling generator from the terminal.
//!
//!     export FIREWORKS_API_KEY=fw_...
//!     style-try "yeah so i was thinking maybe we could look at the eval set"
//!     echo "um so like can you check the tone dimension" | style-try
//!
//! Prints the styled rewrite to stdout; model + latency go to stderr.
//! Override the model with FIREWORKS_MODEL.

use std::io::{self, Read};
use std::process::ExitCode;

use screamer_styling::{FireworksGenerator, StyleContext, StyleGenerator};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: style-try \"<dictated text>\"   (or pipe text on stdin)");
        eprintln!("env: FIREWORKS_API_KEY (required), FIREWORKS_MODEL (optional)");
        return ExitCode::SUCCESS;
    }

    let raw = if args.is_empty() {
        let mut buf = String::new();
        if io::stdin().read_to_string(&mut buf).is_err() {
            eprintln!("failed to read stdin");
            return ExitCode::FAILURE;
        }
        buf
    } else {
        args.join(" ")
    };

    if raw.trim().is_empty() {
        eprintln!("no input text (pass an argument or pipe text on stdin)");
        return ExitCode::from(2);
    }

    let generator = match FireworksGenerator::from_env() {
        Ok(g) => g,
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!("set it with: export FIREWORKS_API_KEY=fw_...");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("[style-try] model: {}", generator.model());
    eprintln!("[style-try] raw:   {}", raw.trim());

    match generator.style(&raw, &StyleContext::default()) {
        Ok(Some(result)) => {
            eprintln!(
                "[style-try] styled in {}ms:",
                result.latency_ms
            );
            println!("{}", result.styled);
            ExitCode::SUCCESS
        }
        Ok(None) => {
            eprintln!("[style-try] generator declined (no rewrite returned)");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[style-try] error: {err}");
            ExitCode::FAILURE
        }
    }
}
