//! Benchmark on-device transcription against a cloud Whisper API over the
//! dictation clips already in telemetry — the "local vs cloud" tradeoff.
//!
//!     bench-asr <telemetry-dir> [--model base] [--model-path FILE] [--limit N] [--labeled-only]
//!
//! For each clip it runs the local Whisper (profiled) and, when an Azure Whisper
//! deployment is configured, the cloud one, then reports latency (p50/p95),
//! throughput vs real time, WER against the human-settled reference (labeled
//! clips only), and cloud cost. Local cost is $0.
//!
//! Cloud (Azure AI Foundry / Azure OpenAI Whisper) is enabled when these are set:
//!   AZURE_OPENAI_ENDPOINT   e.g. https://my-res.openai.azure.com
//!   AZURE_OPENAI_DEPLOYMENT whisper deployment name (default "whisper")
//!   AZURE_OPENAI_API_VERSION (default "2024-06-01")
//!   key: AZURE_OPENAI_API_KEY, else ~/Library/Application Support/Screamer/azure_openai.key
//! With no endpoint/key the cloud columns are skipped (local-only run).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use screamer_telemetry::{export_rows, word_edit_distance};
use screamer_whisper::Transcriber;

/// Azure OpenAI Whisper list price (USD per audio minute) for the cost estimate.
const CLOUD_USD_PER_MIN: f64 = 0.006;

struct Args {
    dir: PathBuf,
    model: String,
    model_path: Option<PathBuf>,
    limit: Option<usize>,
    labeled_only: bool,
}

fn parse_args() -> Result<Args, ExitCode> {
    let mut dir = None;
    let mut model = "base".to_string();
    let mut model_path = None;
    let mut limit = None;
    let mut labeled_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--labeled-only" => labeled_only = true,
            "--model" => model = args.next().ok_or_else(|| usage())?,
            "--model-path" => model_path = Some(PathBuf::from(args.next().ok_or_else(|| usage())?)),
            "--limit" => {
                limit = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| usage())?,
                )
            }
            "-h" | "--help" => return Err(usage_ok()),
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => {
                eprintln!("unexpected argument: {other}");
                return Err(ExitCode::from(2));
            }
        }
    }
    let Some(dir) = dir else { return Err(usage()) };
    Ok(Args { dir, model, model_path, limit, labeled_only })
}

fn usage() -> ExitCode {
    eprintln!("usage: bench-asr <telemetry-dir> [--model base] [--model-path FILE] [--limit N] [--labeled-only]");
    ExitCode::from(2)
}
fn usage_ok() -> ExitCode {
    eprintln!("usage: bench-asr <telemetry-dir> [--model base] [--model-path FILE] [--limit N] [--labeled-only]");
    ExitCode::SUCCESS
}

struct AzureCfg {
    endpoint: String,
    deployment: String,
    api_version: String,
    key: String,
}

fn azure_cfg() -> Option<AzureCfg> {
    let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())?;
    let key = std::env::var("AZURE_OPENAI_API_KEY")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let p = dirs_config()?.join("azure_openai.key");
            std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
        })?;
    let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "whisper".to_string());
    let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "2024-06-01".to_string());
    Some(AzureCfg { endpoint, deployment, api_version, key })
}

fn dirs_config() -> Option<PathBuf> {
    // ~/Library/Application Support/Screamer
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join("Library/Application Support/Screamer"))
}

/// Read a 16-bit PCM mono WAV into f32 samples in [-1, 1].
fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let samples: Result<Vec<f32>, _> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect();
    samples.map_err(|e| e.to_string())
}

/// Minimal multipart/form-data body for the Azure audio upload.
fn multipart(boundary: &str, file: &[u8], filename: &str, fields: &[(&str, &str)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes(),
        );
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

/// Transcribe one clip via Azure Whisper; returns (text, wall-clock latency).
fn cloud_transcribe(agent: &ureq::Agent, cfg: &AzureCfg, wav: &[u8]) -> Result<(String, Duration), String> {
    let url = format!(
        "{}/openai/deployments/{}/audio/transcriptions?api-version={}",
        cfg.endpoint, cfg.deployment, cfg.api_version
    );
    let boundary = "----screamerbench0xADAPTIVE";
    let body = multipart(boundary, wav, "clip.wav", &[("response_format", "json"), ("language", "en")]);
    let ct = format!("multipart/form-data; boundary={boundary}");
    // Fresh Azure Whisper deployments have a low requests-per-minute quota, so
    // retry on 429, honoring Retry-After. The reported latency is the
    // successful call's own round trip, not the waiting.
    let mut backoff = Duration::from_secs(5);
    for attempt in 1..=6u32 {
        let t0 = Instant::now();
        match agent.post(&url).set("api-key", &cfg.key).set("Content-Type", &ct).send_bytes(&body) {
            Ok(resp) => {
                let latency = t0.elapsed();
                let v: serde_json::Value = resp.into_json().map_err(|e| e.to_string())?;
                return Ok((v["text"].as_str().unwrap_or("").to_string(), latency));
            }
            Err(ureq::Error::Status(429, resp)) => {
                if attempt == 6 {
                    return Err("rate limited (429) after 6 attempts".to_string());
                }
                let wait = resp
                    .header("retry-after")
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(Duration::from_secs)
                    .unwrap_or(backoff);
                std::thread::sleep(wait);
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
            Err(ureq::Error::Status(code, resp)) => {
                let detail = resp.into_string().unwrap_or_default().chars().take(150).collect::<String>();
                return Err(format!("status {code}: {detail}"));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    Err("rate limited (429)".to_string())
}

struct Row {
    audio_secs: f64,
    local_ms: f64,
    cloud_ms: Option<f64>,
    wer_local: Option<f64>,
    wer_cloud: Option<f64>,
}

fn wer(reference: &str, hypothesis: &str) -> Option<f64> {
    let ref_words = reference.split_whitespace().count();
    if ref_words == 0 {
        return None;
    }
    Some(word_edit_distance(reference, hypothesis) as f64 / ref_words as f64)
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(code) => return code,
    };

    let model_path = match args.model_path.clone().or_else(|| Transcriber::find_model(&args.model)) {
        Some(p) => p,
        None => {
            eprintln!(
                "could not find model '{}'. Pass --model-path, or put ggml-{}.en.bin in ./models/",
                args.model, args.model
            );
            return ExitCode::FAILURE;
        }
    };
    let transcriber = match Transcriber::new(&model_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load model {}: {e}", model_path.display());
            return ExitCode::FAILURE;
        }
    };
    eprintln!("local model: {} | {}", model_path.display(), transcriber.runtime_summary());
    let _ = transcriber.warm_up(false); // exclude cold-start from the measured runs

    let cloud = azure_cfg();
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).build();
    eprintln!(
        "cloud: {}",
        cloud.as_ref().map(|c| format!("Azure {} @ {}", c.deployment, c.endpoint)).unwrap_or_else(|| "off (no AZURE_OPENAI_ENDPOINT/key)".into())
    );

    let rows = match export_rows(&args.dir, args.labeled_only) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("could not read telemetry {}: {e}", args.dir.display());
            return ExitCode::FAILURE;
        }
    };

    let mut out: Vec<Row> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if let Some(lim) = args.limit {
            if i >= lim {
                break;
            }
        }
        let clip = args.dir.join(&row.audio_path);
        let samples = match read_wav(&clip) {
            Ok(s) if !s.is_empty() => s,
            _ => continue, // clip missing or unreadable
        };
        let audio_secs = samples.len() as f64 / 16_000.0;
        let reference = row.settled_text.as_deref(); // ground truth, labeled clips only

        let local = match transcriber.transcribe_profiled(&samples) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("local transcribe failed for {}: {e}", row.id);
                continue;
            }
        };
        let local_ms = local.profile.total.as_secs_f64() * 1000.0;
        let wer_local = reference.and_then(|r| wer(r, &local.text));

        let (cloud_ms, wer_cloud) = if let Some(cfg) = &cloud {
            let wav_bytes = std::fs::read(&clip).unwrap_or_default();
            match cloud_transcribe(&agent, cfg, &wav_bytes) {
                Ok((text, lat)) => (
                    Some(lat.as_secs_f64() * 1000.0),
                    reference.and_then(|r| wer(r, &text)),
                ),
                Err(e) => {
                    eprintln!("cloud transcribe failed for {}: {e}", row.id);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        out.push(Row { audio_secs, local_ms, cloud_ms, wer_local, wer_cloud });
    }

    if out.is_empty() {
        eprintln!("no clips benchmarked (is the telemetry dir right, with clips/?)");
        return ExitCode::FAILURE;
    }

    report(&out);
    ExitCode::SUCCESS
}

fn report(rows: &[Row]) {
    let n = rows.len();
    let total_audio: f64 = rows.iter().map(|r| r.audio_secs).sum();

    let mut local: Vec<f64> = rows.iter().map(|r| r.local_ms).collect();
    local.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let local_mean = local.iter().sum::<f64>() / n as f64;
    // Throughput vs real time: audio seconds processed per second of compute.
    let local_xrt = total_audio / (local.iter().sum::<f64>() / 1000.0);

    let cloud: Vec<f64> = rows.iter().filter_map(|r| r.cloud_ms).collect();
    let mut cloud_sorted = cloud.clone();
    cloud_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = |v: &[Option<f64>]| {
        let vals: Vec<f64> = v.iter().filter_map(|x| *x).collect();
        if vals.is_empty() { None } else { Some(vals.iter().sum::<f64>() / vals.len() as f64) }
    };
    let wer_local = mean(&rows.iter().map(|r| r.wer_local).collect::<Vec<_>>());
    let wer_cloud = mean(&rows.iter().map(|r| r.wer_cloud).collect::<Vec<_>>());

    let audio_min = total_audio / 60.0;
    let cloud_cost = audio_min * CLOUD_USD_PER_MIN;

    println!("\n=== ASR benchmark: on-device vs cloud ===");
    println!("clips: {n} | audio: {:.1}s total\n", total_audio);
    println!("{:<10} {:>10} {:>10} {:>10} {:>9} {:>12}", "path", "p50 (ms)", "p95 (ms)", "mean (ms)", "WER", "cost /1k");
    println!("{}", "-".repeat(66));
    println!(
        "{:<10} {:>10.0} {:>10.0} {:>10.0} {:>9} {:>12}",
        "on-device",
        pct(&local, 50.0),
        pct(&local, 95.0),
        local_mean,
        wer_local.map(|w| format!("{:.1}%", w * 100.0)).unwrap_or_else(|| "-".into()),
        "$0.00",
    );
    if !cloud.is_empty() {
        let cloud_mean = cloud.iter().sum::<f64>() / cloud.len() as f64;
        println!(
            "{:<10} {:>10.0} {:>10.0} {:>10.0} {:>9} {:>12}",
            "cloud",
            pct(&cloud_sorted, 50.0),
            pct(&cloud_sorted, 95.0),
            cloud_mean,
            wer_cloud.map(|w| format!("{:.1}%", w * 100.0)).unwrap_or_else(|| "-".into()),
            format!("${:.2}", (cloud_cost / n as f64) * 1000.0),
        );
        println!("(cloud latency/WER from {}/{} clips; the rest were rate-limited)", cloud.len(), n);
    } else {
        println!("cloud      (not run — set AZURE_OPENAI_ENDPOINT + key to enable)");
    }
    println!("\non-device throughput: {:.1}x real time, $0 marginal cost, no network / fully private.", local_xrt);
    println!("cloud list price assumed: ${:.4}/audio-min.", CLOUD_USD_PER_MIN);
}
