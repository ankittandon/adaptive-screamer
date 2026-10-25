//! Transcribe a TORGO manifest split with one GGML model, in Screamer's own
//! whisper stack, for a base-vs-tuned WER comparison across severity strata.
//!
//!     asr-eval <model-path> <manifest.jsonl> <torgo-root> [split=eval]
//!
//! Loads the model ONCE and streams every matching utterance through it,
//! printing one JSON object per line: {"speaker","ref","hyp"}. WER is computed
//! downstream (jiwer) from this output, so the numbers match the training eval.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use screamer_whisper::Transcriber;

fn read_wav(path: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    r.samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32768.0)
        .collect()
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: asr-eval <model-path> <manifest.jsonl> <torgo-root> [split=eval]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let manifest = &args[2];
    let root = Path::new(&args[3]);
    let split = args.get(4).map(String::as_str).unwrap_or("eval");

    let transcriber = Transcriber::new(Path::new(model_path)).expect("load model");
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut n = 0;

    let file = std::fs::File::open(manifest).expect("open manifest");
    for line in BufReader::new(file).lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line).expect("parse json");
        if row["split"].as_str() != Some(split) {
            continue;
        }
        let audio = row["audio"].as_str().expect("audio");
        let reference = row["text"].as_str().unwrap_or("");
        let speaker = row["speaker"].as_str().unwrap_or("?");

        let wav = root.join(audio);
        let samples = read_wav(&wav);
        let hyp = transcriber.transcribe(&samples).unwrap_or_default();

        writeln!(
            out,
            "{{\"speaker\":{},\"ref\":{},\"hyp\":{}}}",
            json_str(speaker),
            json_str(reference),
            json_str(hyp.trim())
        )
        .unwrap();
        n += 1;
        if n % 25 == 0 {
            eprintln!("  {n} transcribed");
        }
    }
    eprintln!("done: {n} utterances");
}
