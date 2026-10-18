use crate::edit::edit_ratio;
use crate::record::{DictationRecord, Event, ExportRow, LabelRecord, LabelSource};
use crate::timefmt::{format_rfc3339_utc, unix_now};
use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const RECORDS_FILE: &str = "records.jsonl";
const CLIPS_DIR: &str = "clips";
const SAMPLE_RATE: u32 = 16_000;

/// A fix is accepted only for the most recent dictation, and only this long
/// after it was pasted.
pub const DEFAULT_FIX_MAX_AGE: Duration = Duration::from_secs(10 * 60);

/// If the "corrected" text differs from the transcript by more than this share
/// of words, the user almost certainly selected the wrong text (a whole
/// paragraph, say), so the label is rejected instead of poisoning the data.
/// Kept loose on purpose: short dictations legitimately change a lot when one
/// jargon term is fixed ("cooper netties rocks" -> "Kubernetes rocks" is 0.67).
pub const MAX_FIX_EDIT_RATIO: f32 = 0.8;

/// Everything needed to log one dictation.
pub struct NewDictation<'a> {
    /// 16 kHz mono samples in [-1.0, 1.0], exactly what was sent to the model.
    pub samples: &'a [f32],
    pub raw_text: &'a str,
    pub pasted_text: &'a str,
    pub app_context: Option<&'a str>,
    pub model: &'a str,
}

#[derive(Debug)]
pub enum FixError {
    /// Nothing was selected (or the selection was only whitespace).
    EmptySelection,
    /// No dictation has been logged yet in this session, or it was already fixed.
    NothingToFix,
    /// The last dictation is older than the allowed window.
    Expired,
    /// The selection looks unrelated to the last dictation.
    TooDifferent(f32),
    Io(io::Error),
}

impl fmt::Display for FixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixError::EmptySelection => write!(f, "no text was selected"),
            FixError::NothingToFix => write!(f, "no recent dictation to attach a correction to"),
            FixError::Expired => write!(f, "the last dictation is too old to correct"),
            FixError::TooDifferent(ratio) => write!(
                f,
                "selected text differs from the last dictation by {:.0}% of words; \
                 select just the corrected passage",
                ratio * 100.0
            ),
            FixError::Io(err) => write!(f, "could not write the label: {err}"),
        }
    }
}

impl std::error::Error for FixError {}

impl From<io::Error> for FixError {
    fn from(err: io::Error) -> Self {
        FixError::Io(err)
    }
}

struct PendingFix {
    id: String,
    raw_text: String,
    logged_at: Instant,
}

pub struct TelemetryStore {
    dir: PathBuf,
    write_lock: Mutex<()>,
    pending: Mutex<Option<PendingFix>>,
    seq: AtomicU32,
    fix_max_age: Duration,
}

impl TelemetryStore {
    /// Open (creating if needed) a telemetry directory. The directory and every
    /// file in it are accessible to the current user only.
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder.create(dir.join(CLIPS_DIR))?;
        // `recursive` only applies the mode to directories it creates, so make
        // sure a pre-existing directory is locked down too.
        set_private_dir(&dir)?;
        set_private_dir(&dir.join(CLIPS_DIR))?;

        Ok(Self {
            dir,
            write_lock: Mutex::new(()),
            pending: Mutex::new(None),
            seq: AtomicU32::new(0),
            fix_max_age: DEFAULT_FIX_MAX_AGE,
        })
    }

    pub fn with_fix_max_age(mut self, max_age: Duration) -> Self {
        self.fix_max_age = max_age;
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Save the clip and append the dictation record. Returns the record id.
    /// Also arms the fix hotkey for this dictation.
    pub fn log_dictation(&self, new: NewDictation<'_>) -> io::Result<String> {
        let (secs, millis) = unix_now();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) % 1000;
        let id = format!("{secs}{millis:03}-{seq:03}");
        let audio_path = format!("{CLIPS_DIR}/{id}.wav");

        write_wav(&self.dir.join(&audio_path), new.samples)?;

        let record = DictationRecord {
            id: id.clone(),
            timestamp: format_rfc3339_utc(secs),
            audio_path,
            duration_ms: new.samples.len() as u64 * 1000 / u64::from(SAMPLE_RATE),
            model: new.model.to_string(),
            raw_text: new.raw_text.to_string(),
            pasted_text: new.pasted_text.to_string(),
            app_context: new.app_context.map(str::to_string),
        };
        self.append(&Event::Dictation(record))?;

        if let Ok(mut pending) = self.pending.lock() {
            *pending = Some(PendingFix {
                id: id.clone(),
                raw_text: new.raw_text.to_string(),
                logged_at: Instant::now(),
            });
        }
        Ok(id)
    }

    /// Attach corrected text to the most recent dictation. On success the
    /// dictation is no longer pending; on any error it stays pending so the
    /// user can select again and retry.
    pub fn apply_fix(&self, selected_text: &str) -> Result<LabelRecord, FixError> {
        let settled = selected_text.trim();
        if settled.is_empty() {
            return Err(FixError::EmptySelection);
        }

        let mut pending = self.pending.lock().map_err(|_| FixError::NothingToFix)?;
        let Some(last) = pending.as_ref() else {
            return Err(FixError::NothingToFix);
        };
        if last.logged_at.elapsed() >= self.fix_max_age {
            return Err(FixError::Expired);
        }

        let ratio = edit_ratio(&last.raw_text, settled);
        if ratio > MAX_FIX_EDIT_RATIO {
            return Err(FixError::TooDifferent(ratio));
        }

        let (secs, _) = unix_now();
        let label = LabelRecord {
            id: last.id.clone(),
            labeled_at: format_rfc3339_utc(secs),
            settled_text: settled.to_string(),
            label_source: LabelSource::Fix,
            edit_ratio: ratio,
        };
        self.append(&Event::Label(label.clone()))?;
        *pending = None;
        Ok(label)
    }

    fn append(&self, event: &Event) -> io::Result<()> {
        let mut line = serde_json::to_vec(event).map_err(io::Error::other)?;
        line.push(b'\n');

        let _guard = self.write_lock.lock().map_err(|_| io::Error::other("lock poisoned"))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(self.dir.join(RECORDS_FILE))?;
        file.write_all(&line)
    }
}

fn set_private_dir(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

fn write_wav(path: &Path, samples: &[f32]) -> io::Result<()> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(BufWriter::new(file), spec).map_err(io::Error::other)?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        writer.write_sample(value).map_err(io::Error::other)?;
    }
    writer.finalize().map_err(io::Error::other)
}

/// Merge `records.jsonl` into one row per dictation, newest label winning.
/// Lines that fail to parse are skipped (a crash mid-write must not lose the
/// rest of the file). Rows keep the order dictations were logged in.
pub fn export_rows(dir: &Path, labeled_only: bool) -> io::Result<Vec<ExportRow>> {
    let file = File::open(dir.join(RECORDS_FILE))?;

    let mut rows: Vec<ExportRow> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Event>(&line) {
            Ok(Event::Dictation(d)) => {
                index.insert(d.id.clone(), rows.len());
                rows.push(ExportRow {
                    id: d.id,
                    timestamp: d.timestamp,
                    audio_path: d.audio_path,
                    duration_ms: d.duration_ms,
                    model: d.model,
                    raw_text: d.raw_text,
                    pasted_text: d.pasted_text,
                    settled_text: None,
                    label_source: None,
                    edit_ratio: None,
                    app_context: d.app_context,
                });
            }
            Ok(Event::Label(l)) => {
                if let Some(&i) = index.get(&l.id) {
                    rows[i].settled_text = Some(l.settled_text);
                    rows[i].label_source = Some(l.label_source);
                    rows[i].edit_ratio = Some(l.edit_ratio);
                }
            }
            Err(_) => continue,
        }
    }

    if labeled_only {
        rows.retain(|row| row.settled_text.is_some());
    }
    Ok(rows)
}

/// Write rows as JSON lines.
pub fn write_export_jsonl(rows: &[ExportRow], out: &mut impl Write) -> io::Result<()> {
    for row in rows {
        serde_json::to_writer(&mut *out, row).map_err(io::Error::other)?;
        out.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicU32;

    static NEXT_DIR: AtomicU32 = AtomicU32::new(0);

    /// A fresh directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let n = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "screamer-telemetry-test-{}-{n}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn dictation<'a>(samples: &'a [f32], raw: &'a str) -> NewDictation<'a> {
        NewDictation {
            samples,
            raw_text: raw,
            pasted_text: raw,
            app_context: Some("com.apple.Notes"),
            model: "base",
        }
    }

    fn read_events(dir: &Path) -> Vec<Event> {
        fs::read_to_string(dir.join(RECORDS_FILE))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn logs_a_clip_and_a_record() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        let samples = vec![0.25f32; 16_000]; // one second
        let id = store
            .log_dictation(dictation(&samples, "deploy to cooper netties"))
            .unwrap();

        let events = read_events(&tmp.0);
        assert_eq!(events.len(), 1);
        let Event::Dictation(rec) = &events[0] else {
            panic!("expected a dictation event");
        };
        assert_eq!(rec.id, id);
        assert_eq!(rec.audio_path, format!("clips/{id}.wav"));
        assert_eq!(rec.duration_ms, 1000);
        assert_eq!(rec.raw_text, "deploy to cooper netties");
        assert_eq!(rec.pasted_text, "deploy to cooper netties");
        assert_eq!(rec.app_context.as_deref(), Some("com.apple.Notes"));
        assert_eq!(rec.model, "base");
        assert!(rec.timestamp.ends_with('Z'));

        let reader = hound::WavReader::open(tmp.0.join(&rec.audio_path)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(reader.len(), 16_000);
    }

    #[test]
    fn wav_samples_are_scaled_and_clamped() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        let samples = [0.0f32, 0.5, -0.5, 2.0, -2.0];
        let id = store.log_dictation(dictation(&samples, "x")).unwrap();

        let mut reader = hound::WavReader::open(tmp.0.join(format!("clips/{id}.wav"))).unwrap();
        let values: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        assert_eq!(values, vec![0, 16384, -16384, 32767, -32767]);
    }

    #[test]
    fn ids_are_unique_for_back_to_back_dictations() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        let a = store.log_dictation(dictation(&[0.1], "a")).unwrap();
        let b = store.log_dictation(dictation(&[0.1], "b")).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn files_are_private_to_the_user() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        let id = store.log_dictation(dictation(&[0.1], "a")).unwrap();

        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&tmp.0), 0o700);
        assert_eq!(mode(&tmp.0.join("clips")), 0o700);
        assert_eq!(mode(&tmp.0.join(RECORDS_FILE)), 0o600);
        assert_eq!(mode(&tmp.0.join(format!("clips/{id}.wav"))), 0o600);
    }

    #[test]
    fn reopening_a_loose_directory_locks_it_down() {
        let tmp = TempDir::new();
        fs::create_dir_all(tmp.0.join("clips")).unwrap();
        fs::set_permissions(&tmp.0, fs::Permissions::from_mode(0o755)).unwrap();

        TelemetryStore::open(&tmp.0).unwrap();
        assert_eq!(fs::metadata(&tmp.0).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn fix_attaches_the_corrected_text_to_the_last_dictation() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        let id = store
            .log_dictation(dictation(&[0.1; 160], "we use cooper netties for the cluster"))
            .unwrap();

        let label = store
            .apply_fix("  we use Kubernetes for the cluster \n")
            .unwrap();
        assert_eq!(label.id, id);
        assert_eq!(label.settled_text, "we use Kubernetes for the cluster");
        assert_eq!(label.label_source, LabelSource::Fix);
        assert_eq!(label.edit_ratio, 0.2857);

        let events = read_events(&tmp.0);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1], Event::Label(label));
    }

    #[test]
    fn an_unchanged_selection_is_a_valid_zero_edit_label() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        store.log_dictation(dictation(&[0.1], "ship it")).unwrap();
        let label = store.apply_fix("ship it").unwrap();
        assert_eq!(label.edit_ratio, 0.0);
    }

    #[test]
    fn fix_without_a_dictation_is_rejected() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        assert!(matches!(store.apply_fix("hello"), Err(FixError::NothingToFix)));
    }

    #[test]
    fn a_dictation_can_only_be_fixed_once() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        store.log_dictation(dictation(&[0.1], "ship it")).unwrap();
        store.apply_fix("ship it now").unwrap();
        assert!(matches!(store.apply_fix("ship it now"), Err(FixError::NothingToFix)));
    }

    #[test]
    fn empty_selection_is_rejected_and_keeps_the_dictation_pending() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        store.log_dictation(dictation(&[0.1], "ship it")).unwrap();
        assert!(matches!(store.apply_fix("  \n "), Err(FixError::EmptySelection)));
        assert!(store.apply_fix("ship it").is_ok());
    }

    #[test]
    fn an_unrelated_selection_is_rejected_and_keeps_the_dictation_pending() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        store
            .log_dictation(dictation(&[0.1], "schedule the review for friday"))
            .unwrap();

        let err = store
            .apply_fix("Quarterly planning notes: revenue is up and churn is down across segments")
            .unwrap_err();
        assert!(matches!(err, FixError::TooDifferent(r) if r > MAX_FIX_EDIT_RATIO));
        assert_eq!(read_events(&tmp.0).len(), 1, "nothing may be written for a rejected fix");

        assert!(store.apply_fix("schedule the review for Friday").is_ok());
    }

    #[test]
    fn a_stale_dictation_cannot_be_fixed() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0)
            .unwrap()
            .with_fix_max_age(Duration::ZERO);
        store.log_dictation(dictation(&[0.1], "ship it")).unwrap();
        assert!(matches!(store.apply_fix("ship it"), Err(FixError::Expired)));
    }

    #[test]
    fn only_the_newest_dictation_is_fixable() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        store.log_dictation(dictation(&[0.1], "first one")).unwrap();
        let second = store.log_dictation(dictation(&[0.1], "second one")).unwrap();
        let label = store.apply_fix("second one.").unwrap();
        assert_eq!(label.id, second);
    }

    #[test]
    fn export_merges_labels_and_filters_unlabeled() {
        let tmp = TempDir::new();
        let store = TelemetryStore::open(&tmp.0).unwrap();
        store.log_dictation(dictation(&[0.1], "left alone")).unwrap();
        let fixed = store
            .log_dictation(dictation(&[0.1], "cooper netties rocks"))
            .unwrap();
        store.apply_fix("Kubernetes rocks").unwrap();

        let all = export_rows(&tmp.0, false).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].settled_text, None);
        assert_eq!(all[1].id, fixed);
        assert_eq!(all[1].settled_text.as_deref(), Some("Kubernetes rocks"));
        assert_eq!(all[1].label_source, Some(LabelSource::Fix));

        let labeled = export_rows(&tmp.0, true).unwrap();
        assert_eq!(labeled.len(), 1);
        assert_eq!(labeled[0].id, fixed);

        let mut out = Vec::new();
        write_export_jsonl(&labeled, &mut out).unwrap();
        let line: serde_json::Value = serde_json::from_slice(out.strip_suffix(b"\n").unwrap()).unwrap();
        assert_eq!(line["raw_text"], "cooper netties rocks");
        assert_eq!(line["settled_text"], "Kubernetes rocks");
        assert_eq!(line["label_source"], "fix");
        assert_eq!(line["audio_path"], format!("clips/{fixed}.wav"));
        assert_eq!(line["app_context"], "com.apple.Notes");
    }

    #[test]
    fn export_uses_the_newest_label_and_skips_corrupt_lines() {
        let tmp = TempDir::new();
        fs::create_dir_all(&tmp.0).unwrap();
        let lines = [
            r#"{"kind":"dictation","id":"a","timestamp":"2026-11-02T10:00:00Z","audio_path":"clips/a.wav","duration_ms":1000,"model":"base","raw_text":"one","pasted_text":"one"}"#,
            r#"{"kind":"label","id":"a","labeled_at":"2026-11-02T10:01:00Z","settled_text":"One","label_source":"fix","edit_ratio":1.0}"#,
            r#"{"kind":"dictation","id":"b","timesta"#, // torn write
            r#"{"kind":"label","id":"a","labeled_at":"2026-11-02T10:02:00Z","settled_text":"One.","label_source":"implicit","edit_ratio":1.0}"#,
            r#"{"kind":"label","id":"ghost","labeled_at":"2026-11-02T10:03:00Z","settled_text":"x","label_source":"fix","edit_ratio":0.0}"#,
        ];
        fs::write(tmp.0.join(RECORDS_FILE), lines.join("\n") + "\n").unwrap();

        let rows = export_rows(&tmp.0, false).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].settled_text.as_deref(), Some("One."));
        assert_eq!(rows[0].label_source, Some(LabelSource::Implicit));
        assert_eq!(rows[0].app_context, None);
    }

    #[test]
    fn export_of_a_missing_log_is_an_error_not_an_empty_success() {
        let tmp = TempDir::new();
        fs::create_dir_all(&tmp.0).unwrap();
        assert!(export_rows(&tmp.0, true).is_err());
    }
}
