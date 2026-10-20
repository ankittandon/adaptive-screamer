//! Score a Screamer telemetry log for AI-writing "slop" (per 1,000 words).
//!
//!     slop-report "$HOME/Library/Application Support/Screamer/telemetry"
//!     slop-report path/to/records.jsonl
//!
//! Prints, per dictation, the slop score of the raw transcript, the styled
//! rewrite that was pasted (if any), and your final edited text (if captured),
//! plus averages — so you can see whether styling lowers slop and where your own
//! edits land.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use screamer_styling::slop_score;

struct Row {
    id: String,
    raw: String,
    styled: Option<String>,
    final_text: Option<String>,
}

fn main() -> ExitCode {
    let Some(arg) = std::env::args().nth(1) else {
        eprintln!("usage: slop-report <telemetry-dir | records.jsonl>");
        return ExitCode::from(2);
    };
    let mut path = PathBuf::from(&arg);
    if path.is_dir() {
        path = path.join("records.jsonl");
    }
    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not open {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let mut order: Vec<String> = Vec::new();
    let mut rows: HashMap<String, Row> = HashMap::new();

    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let kind = v["kind"].as_str().unwrap_or("");
        let id = v["id"].as_str().unwrap_or("").to_string();
        if id.is_empty() {
            continue;
        }
        match kind {
            "dictation" => {
                let raw = v["raw_text"].as_str().unwrap_or("").to_string();
                let pasted = v["pasted_text"].as_str().unwrap_or("").to_string();
                let styled = if !pasted.is_empty() && pasted != raw {
                    Some(pasted)
                } else {
                    None
                };
                order.push(id.clone());
                rows.insert(
                    id.clone(),
                    Row {
                        id,
                        raw,
                        styled,
                        final_text: None,
                    },
                );
            }
            "label" => {
                if let Some(row) = rows.get_mut(&id) {
                    row.final_text = v["settled_text"].as_str().map(|s| s.to_string());
                }
            }
            _ => {}
        }
    }

    if order.is_empty() {
        eprintln!("no dictations found in {}", path.display());
        return ExitCode::SUCCESS;
    }

    println!(
        "{:<20} {:>8} {:>8} {:>8}   excerpt",
        "id", "raw", "styled", "final"
    );
    println!("{}", "-".repeat(70));

    let mut sum_raw = 0.0f32;
    let mut sum_styled = 0.0f32;
    let mut sum_final = 0.0f32;
    let (mut n_styled, mut n_final, mut styled_improved) = (0usize, 0usize, 0usize);

    for id in &order {
        let row = &rows[id];
        let s_raw = slop_score(&row.raw);
        sum_raw += s_raw;
        let s_styled = row.styled.as_ref().map(|s| slop_score(s));
        let s_final = row.final_text.as_ref().map(|s| slop_score(s));
        if let Some(ss) = s_styled {
            sum_styled += ss;
            n_styled += 1;
            if ss <= s_raw {
                styled_improved += 1;
            }
        }
        if let Some(sf) = s_final {
            sum_final += sf;
            n_final += 1;
        }
        let cell = |v: Option<f32>| v.map(|x| format!("{x:>8.1}")).unwrap_or_else(|| format!("{:>8}", "-"));
        let excerpt: String = row.raw.chars().take(40).collect();
        println!(
            "{:<20} {:>8.1} {} {}   {}",
            short(&row.id),
            s_raw,
            cell(s_styled),
            cell(s_final),
            excerpt
        );
    }

    println!("{}", "-".repeat(70));
    println!("dictations: {}", order.len());
    println!("avg slop  raw: {:.1}", sum_raw / order.len() as f32);
    if n_styled > 0 {
        println!(
            "avg slop  styled: {:.1}   ({} rewrites, {} at or below raw)",
            sum_styled / n_styled as f32,
            n_styled,
            styled_improved
        );
    }
    if n_final > 0 {
        println!("avg slop  final (your edits): {:.1}   ({} labeled)", sum_final / n_final as f32, n_final);
    }
    println!("(lower is more human; human prose is typically well under ~15)");

    ExitCode::SUCCESS
}

fn short(id: &str) -> String {
    id.chars().take(20).collect()
}
