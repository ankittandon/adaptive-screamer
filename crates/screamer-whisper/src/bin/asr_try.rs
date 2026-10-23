//! Transcribe one WAV with a given GGML model, for spot-checking a model in
//! Screamer's own whisper stack.
//!
//!     asr-try <model-path> <file.wav>

use std::path::Path;

use screamer_whisper::Transcriber;

fn read_wav(path: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    r.samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32768.0)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: asr-try <model-path> <file.wav>");
        std::process::exit(2);
    }
    let transcriber = Transcriber::new(Path::new(&args[1])).expect("load model");
    let samples = read_wav(Path::new(&args[2]));
    let text = transcriber.transcribe(&samples).expect("transcribe");
    println!("{}", text.trim());
}
