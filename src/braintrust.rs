//! Background Braintrust trace sink — fully decoupled from dictation.
//!
//! The local telemetry log (`records.jsonl`) is the source of truth and the
//! durable queue. This module tails that file on its own background thread and
//! ships each event to Braintrust's REST log-insert endpoint. Nothing here runs
//! on the dictation hot path: the dictation code only writes the local log, as
//! it already did. If Braintrust is slow, down, or unreachable, dictation is
//! unaffected and the backlog simply ships later.
//!
//! Idempotent by construction: every event is keyed by the stable telemetry
//! record id and sent with `_is_merge: true`, so re-shipping (after a lost
//! cursor, say) merges into the same row rather than duplicating it. A
//! byte-offset cursor (`braintrust.cursor`) is kept only to avoid re-POSTing the
//! whole log every cycle.
//!
//! When `braintrust_upload_audio` is set, each dictation's WAV clip is uploaded
//! as a Braintrust attachment and referenced on the event. The upload is
//! best-effort (text always ships regardless) and uses the same signed-URL flow
//! as the Braintrust SDK: POST /attachment -> PUT the bytes to the signed URL ->
//! POST /attachment/status.

use crate::config::Config;
use crate::logging;
use crate::telemetry::telemetry_dir;
use screamer_styling::slop_score;
use screamer_telemetry::{DictationRecord, Event, LabelSource};
use serde_json::{json, Value};
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

const RECORDS_FILE: &str = "records.jsonl";
const CURSOR_FILE: &str = "braintrust.cursor";
/// How often the reconciler checks the log for new lines.
const POLL_INTERVAL: Duration = Duration::from_secs(5);
const HTTP_TIMEOUT: Duration = Duration::from_secs(20);
/// Longer timeout for the object-store PUT (audio bytes).
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(60);
/// Max events shipped per cycle (and per insert request). Bounds a large first
/// backfill; the rest ships on the next cycles.
const BATCH_MAX: usize = 100;
/// Upper bound on the exponential backoff after transient failures.
const MAX_BACKOFF: Duration = Duration::from_secs(300);

/// `~/Library/Application Support/Screamer/braintrust.key`
pub fn key_file_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join("Screamer")
        .join("braintrust.key")
}

/// The Braintrust key: `BRAINTRUST_API_KEY` if set, else the key file. A
/// double-clicked app does not inherit the shell env, so the file is the
/// reliable path (same pattern as the Fireworks key in `styling.rs`).
fn resolve_api_key() -> Option<String> {
    if let Ok(k) = std::env::var("BRAINTRUST_API_KEY") {
        let k = k.trim().to_string();
        if !k.is_empty() {
            return Some(k);
        }
    }
    std::fs::read_to_string(key_file_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Launch the background sink when `braintrust_enabled` is set, telemetry is on,
/// and a key is available. Returns immediately; a no-op otherwise. The dictation
/// path never calls into this module.
pub fn start(config: &Config) {
    if !config.braintrust_enabled {
        return;
    }
    if !config.telemetry_enabled {
        eprintln!(
            "[screamer] Braintrust sink off: telemetry_enabled is false, so there is no log to ship."
        );
        return;
    }
    let api_key = match resolve_api_key() {
        Some(k) => k,
        None => {
            eprintln!(
                "[screamer] Braintrust sink off: no API key. Set BRAINTRUST_API_KEY or write it to {}",
                key_file_path().display()
            );
            return;
        }
    };
    let project = config.braintrust_project.trim().to_string();
    if project.is_empty() {
        eprintln!("[screamer] Braintrust sink off: braintrust_project is empty.");
        return;
    }
    let base = config.braintrust_endpoint.trim_end_matches('/');
    let endpoint = format!("{base}/project_logs/{project}/insert");
    // Attachment endpoints live at the host root (e.g. .../attachment), not under /v1.
    let api_root = base.strip_suffix("/v1").unwrap_or(base).trim_end_matches('/').to_string();

    let upload_audio = config.braintrust_upload_audio;
    let org = config.braintrust_org.trim().to_string();
    if upload_audio && org.is_empty() {
        eprintln!("[screamer] Braintrust: audio upload needs braintrust_org; shipping text only.");
    }

    let sink = Sink {
        endpoint,
        api_root,
        api_key,
        org,
        upload_audio,
        dir: telemetry_dir(),
    };
    eprintln!(
        "[screamer] Braintrust sink on: shipping telemetry to project {project} (audio: {})",
        if upload_audio { "on" } else { "off" }
    );
    std::thread::Builder::new()
        .name("braintrust-sink".into())
        .spawn(move || sink.run())
        .ok();
}

struct Sink {
    /// Full insert URL: `{endpoint}/project_logs/{project}/insert`.
    endpoint: String,
    /// Host root for attachment endpoints (`/attachment`, `/attachment/status`).
    api_root: String,
    api_key: String,
    org: String,
    upload_audio: bool,
    /// Telemetry directory holding `records.jsonl`, `clips/`, and the cursor.
    dir: PathBuf,
}

enum Ship {
    Ok,
    /// Transient failure (network, 5xx, 429): keep the cursor and back off.
    Retry(String),
    /// Permanent misconfiguration (bad key/project): stop the thread.
    Fatal(String),
}

impl Sink {
    fn run(self) {
        let agent = ureq::AgentBuilder::new().timeout(HTTP_TIMEOUT).build();
        let uploader = ureq::AgentBuilder::new().timeout(UPLOAD_TIMEOUT).build();
        let records = self.dir.join(RECORDS_FILE);
        let cursor_path = self.dir.join(CURSOR_FILE);
        let mut cursor = read_cursor(&cursor_path);
        let mut backoff = POLL_INTERVAL;

        loop {
            std::thread::sleep(backoff);

            let (events, next_cursor) = match collect_since(&records, cursor) {
                Ok(v) => v,
                Err(_) => {
                    // The log may not exist yet (telemetry on but nothing dictated).
                    backoff = POLL_INTERVAL;
                    continue;
                }
            };
            if events.is_empty() {
                cursor = next_cursor; // skips past any corrupt/blank lines
                backoff = POLL_INTERVAL;
                continue;
            }

            // Build payload (uploads audio attachments as a side effect when enabled).
            let payload: Vec<Value> = events
                .iter()
                .filter_map(|ev| self.build_event(&uploader, ev))
                .collect();

            match self.ship(&agent, &payload) {
                Ship::Ok => {
                    cursor = next_cursor;
                    write_cursor(&cursor_path, cursor);
                    logging::log_flow_event(
                        "braintrust",
                        "shipped",
                        &format!("{} events (offset {cursor})", payload.len()),
                    );
                    backoff = POLL_INTERVAL;
                }
                Ship::Retry(msg) => {
                    backoff = (backoff * 2).min(MAX_BACKOFF);
                    logging::log_flow_event(
                        "braintrust",
                        "retry",
                        &format!("{msg} (next in {}s)", backoff.as_secs()),
                    );
                }
                Ship::Fatal(msg) => {
                    eprintln!("[screamer] Braintrust sink stopping: {msg}");
                    return;
                }
            }
        }
    }

    /// Map one event to a Braintrust log event. For dictations with audio upload
    /// enabled, best-effort upload the clip and attach a reference; text always
    /// ships regardless of whether the audio upload succeeds.
    fn build_event(&self, uploader: &ureq::Agent, ev: &Event) -> Option<Value> {
        let mut v = to_braintrust(ev)?;
        if self.upload_audio && !self.org.is_empty() {
            if let Event::Dictation(d) = ev {
                if let Some(att) = self.upload_clip(uploader, d) {
                    if let Some(meta) = v.get_mut("metadata").and_then(|m| m.as_object_mut()) {
                        meta.insert("audio".to_string(), att);
                    }
                }
            }
        }
        Some(v)
    }

    /// Upload `clips/<id>.wav` as a Braintrust attachment and return the
    /// reference object to embed on the event. Best-effort: `None` on any
    /// failure (logged), so text still ships. Up to two attempts to ride out a
    /// transient blip; a persistent failure drops only that clip's audio.
    fn upload_clip(&self, uploader: &ureq::Agent, d: &DictationRecord) -> Option<Value> {
        let path = self.dir.join(&d.audio_path);
        let data = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => return None, // clip missing (e.g. pruned) — nothing to upload
        };
        // Deterministic key -> idempotent re-upload overwrites the same object.
        let key = format!("screamer/{}.wav", d.id);
        let filename = format!("{}.wav", d.id);

        for attempt in 1..=2 {
            match self.try_upload(uploader, &key, &filename, &data) {
                Ok(()) => {
                    return Some(json!({
                        "type": "braintrust_attachment",
                        "filename": filename,
                        "content_type": "audio/wav",
                        "key": key,
                    }));
                }
                Err(e) => {
                    logging::log_flow_event(
                        "braintrust",
                        "audio",
                        &format!("clip {} upload failed (attempt {attempt}): {e}", d.id),
                    );
                }
            }
        }
        None
    }

    /// The signed-URL attachment flow (mirrors the Braintrust SDK):
    /// POST /attachment -> PUT bytes to the signed URL -> POST /attachment/status.
    fn try_upload(
        &self,
        uploader: &ureq::Agent,
        key: &str,
        filename: &str,
        data: &[u8],
    ) -> Result<(), String> {
        let meta: Value = uploader
            .post(&format!("{}/attachment", self.api_root))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(json!({
                "key": key,
                "filename": filename,
                "content_type": "audio/wav",
                "org_id": self.org,
            }))
            .map_err(stringify_err)?
            .into_json()
            .map_err(|e| e.to_string())?;

        let signed = meta
            .get("signedUrl")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("no signedUrl in response: {meta}"))?;

        let mut put = uploader.put(signed);
        if let Some(headers) = meta.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in headers {
                if let Some(s) = v.as_str() {
                    put = put.set(k, s);
                }
            }
        }
        if signed.contains("blob.core.windows.net") {
            put = put.set("x-ms-blob-type", "BlockBlob");
        }
        put.send_bytes(data).map_err(stringify_err)?;

        // Best-effort status update so the attachment shows as complete in the UI.
        let _ = uploader
            .post(&format!("{}/attachment/status", self.api_root))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(json!({
                "key": key,
                "org_id": self.org,
                "status": { "upload_status": "done" },
            }));

        Ok(())
    }

    fn ship(&self, agent: &ureq::Agent, events: &[Value]) -> Ship {
        if events.is_empty() {
            return Ship::Ok;
        }
        let body = json!({ "events": events });
        match agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body)
        {
            Ok(_) => Ship::Ok,
            Err(ureq::Error::Status(code, resp)) => {
                let detail = resp
                    .into_string()
                    .unwrap_or_default()
                    .chars()
                    .take(300)
                    .collect::<String>();
                // 429 is a 4xx but is retryable; other 4xx (bad key/project) are not.
                if code == 429 || code >= 500 {
                    Ship::Retry(format!("HTTP {code}: {detail}"))
                } else {
                    Ship::Fatal(format!("HTTP {code}: {detail}"))
                }
            }
            Err(e) => Ship::Retry(e.to_string()),
        }
    }
}

fn stringify_err(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, resp) => {
            let detail = resp
                .into_string()
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect::<String>();
            format!("HTTP {code}: {detail}")
        }
        other => other.to_string(),
    }
}

/// Map one telemetry event to a Braintrust log event, or `None` to skip it.
/// Text only; audio is attached by `Sink::build_event`.
fn to_braintrust(ev: &Event) -> Option<Value> {
    match ev {
        Event::Dictation(d) => Some(json!({
            "id": d.id,
            "_is_merge": true,
            "span_attributes": { "name": "dictation", "type": "task" },
            "input": d.raw_text,
            "output": d.pasted_text,
            "metadata": {
                "app_context": d.app_context,
                "model": d.model,
                "styled": d.pasted_text != d.raw_text,
                "duration_ms": d.duration_ms,
                "timestamp": d.timestamp,
                "audio_path": d.audio_path,
                "slop_raw": slop_score(&d.raw_text),
                "slop_pasted": slop_score(&d.pasted_text),
            },
        })),
        Event::Label(l) => {
            let source = match l.label_source {
                LabelSource::Fix => "fix",
                LabelSource::Implicit => "implicit",
            };
            Some(json!({
                "id": l.id,
                "_is_merge": true,
                "expected": l.settled_text,
                "scores": { "edit_ratio": l.edit_ratio },
                "metadata": {
                    "settled_text": l.settled_text,
                    "label_source": source,
                    "labeled_at": l.labeled_at,
                    "slop_final": slop_score(&l.settled_text),
                },
            }))
        }
    }
}

/// Read complete lines of `records.jsonl` past `cursor`, parse them to events,
/// and return them with the byte offset to persist once they ship.
///
/// Only whole lines (terminated by `\n`) are consumed, so a torn final write is
/// never shipped; it is picked up once complete. A line that is non-empty but
/// fails to parse is skipped yet still consumed, so a single corrupt line cannot
/// wedge the cursor. Capped at `BATCH_MAX` events per call.
fn collect_since(records: &Path, cursor: u64) -> io::Result<(Vec<Event>, u64)> {
    let mut file = fs::File::open(records)?;
    let len = file.metadata()?.len();
    // If the file shrank (shouldn't — the log is append-only), start over; the
    // idempotent merge makes re-shipping harmless.
    let start = if cursor > len { 0 } else { cursor };
    if start >= len {
        return Ok((Vec::new(), start));
    }
    file.seek(SeekFrom::Start(start))?;

    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let mut consumed: u64 = 0;
    let mut events = Vec::new();
    let mut line_start = 0usize;
    while let Some(nl) = buf[line_start..].iter().position(|&b| b == b'\n') {
        let line_end = line_start + nl; // index of '\n'
        let line = &buf[line_start..line_end];
        let line_len = (line_end - line_start + 1) as u64; // include '\n'

        if let Ok(text) = std::str::from_utf8(line) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                if let Ok(ev) = serde_json::from_str::<Event>(trimmed) {
                    events.push(ev);
                }
            }
        }

        consumed += line_len;
        line_start = line_end + 1;
        if events.len() >= BATCH_MAX {
            break;
        }
    }

    Ok((events, start + consumed))
}

fn read_cursor(path: &Path) -> u64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn write_cursor(path: &Path, offset: u64) {
    if let Err(err) = fs::write(path, offset.to_string()) {
        eprintln!("[screamer] Braintrust: could not persist cursor: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use screamer_telemetry::{DictationRecord, LabelRecord};

    /// Self-contained temp-dir guard (no external deps).
    struct TempPath(PathBuf);
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_log(tag: &str, body: &str) -> (TempPath, PathBuf) {
        let dir = std::env::temp_dir().join(format!("bt-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("records.jsonl");
        fs::write(&path, body).unwrap();
        (TempPath(dir), path)
    }

    #[test]
    fn maps_a_dictation_to_a_merge_event() {
        let ev = Event::Dictation(DictationRecord {
            id: "abc".into(),
            timestamp: "2026-10-20T00:00:00Z".into(),
            audio_path: "clips/abc.wav".into(),
            duration_ms: 1000,
            model: "base".into(),
            raw_text: "um hello there".into(),
            pasted_text: "Hello there.".into(),
            app_context: Some("com.apple.Notes".into()),
        });
        let v = to_braintrust(&ev).unwrap();
        assert_eq!(v["id"], "abc");
        assert_eq!(v["_is_merge"], true);
        assert_eq!(v["input"], "um hello there");
        assert_eq!(v["output"], "Hello there.");
        assert_eq!(v["metadata"]["styled"], true);
        assert_eq!(v["metadata"]["app_context"], "com.apple.Notes");
        assert!(v["metadata"]["slop_raw"].is_number());
    }

    #[test]
    fn maps_a_label_to_scores_and_expected() {
        let ev = Event::Label(LabelRecord {
            id: "abc".into(),
            labeled_at: "2026-10-20T00:01:00Z".into(),
            settled_text: "Hello there!".into(),
            label_source: LabelSource::Implicit,
            edit_ratio: 0.25,
        });
        let v = to_braintrust(&ev).unwrap();
        assert_eq!(v["id"], "abc");
        assert_eq!(v["_is_merge"], true);
        assert_eq!(v["expected"], "Hello there!");
        assert_eq!(v["scores"]["edit_ratio"], 0.25);
        assert_eq!(v["metadata"]["label_source"], "implicit");
    }

    #[test]
    fn collect_reads_whole_lines_and_advances_cursor() {
        let dict = r#"{"kind":"dictation","id":"a","timestamp":"t","audio_path":"clips/a.wav","duration_ms":1,"model":"base","raw_text":"one","pasted_text":"One."}"#;
        let label = r#"{"kind":"label","id":"a","labeled_at":"t","settled_text":"One!","label_source":"fix","edit_ratio":0.1}"#;
        let (_tmp, path) = write_log("whole", &format!("{dict}\n{label}\n"));

        let (events, cursor) = collect_since(&path, 0).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(cursor, fs::metadata(&path).unwrap().len());

        // From the end: nothing new.
        let (events2, cursor2) = collect_since(&path, cursor).unwrap();
        assert!(events2.is_empty());
        assert_eq!(cursor2, cursor);
    }

    #[test]
    fn collect_ignores_a_torn_trailing_line() {
        let dict = r#"{"kind":"dictation","id":"a","timestamp":"t","audio_path":"clips/a.wav","duration_ms":1,"model":"base","raw_text":"one","pasted_text":"One."}"#;
        let torn = r#"{"kind":"dictation","id":"b","timesta"#; // no newline
        let (_tmp, path) = write_log("torn", &format!("{dict}\n{torn}"));

        let (events, cursor) = collect_since(&path, 0).unwrap();
        assert_eq!(events.len(), 1, "only the complete line ships");
        assert_eq!(cursor, (dict.len() + 1) as u64, "cursor stops before the torn line");
    }

    #[test]
    fn collect_skips_a_corrupt_line_without_wedging() {
        let dict = r#"{"kind":"dictation","id":"a","timestamp":"t","audio_path":"clips/a.wav","duration_ms":1,"model":"base","raw_text":"one","pasted_text":"One."}"#;
        let (_tmp, path) = write_log("corrupt", &format!("not json at all\n{dict}\n"));
        let (events, cursor) = collect_since(&path, 0).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(cursor, fs::metadata(&path).unwrap().len());
    }
}
