use serde::{Deserialize, Serialize};

/// Where a corrected transcript came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelSource {
    /// The user selected the corrected text and tapped the fix hotkey.
    Fix,
    /// Detected automatically from an edit in the target app (not implemented yet).
    Implicit,
}

/// One line of `records.jsonl`. The file is append-only: a dictation line is
/// written when the text is pasted, and a label line is appended later if the
/// user corrects it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Dictation(DictationRecord),
    Label(LabelRecord),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DictationRecord {
    pub id: String,
    /// RFC 3339 UTC.
    pub timestamp: String,
    /// Relative to the telemetry directory, e.g. `clips/<id>.wav`.
    pub audio_path: String,
    pub duration_ms: u64,
    /// Model setting in effect (the config name, e.g. `base`).
    pub model: String,
    /// What the model produced.
    pub raw_text: String,
    /// What was pasted into the app.
    pub pasted_text: String,
    /// Bundle id of the frontmost app at paste time. Never a window title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_context: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelRecord {
    pub id: String,
    pub labeled_at: String,
    pub settled_text: String,
    pub label_source: LabelSource,
    pub edit_ratio: f32,
}

/// A dictation merged with its latest label, one per exported line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportRow {
    pub id: String,
    pub timestamp: String,
    pub audio_path: String,
    pub duration_ms: u64,
    pub model: String,
    pub raw_text: String,
    pub pasted_text: String,
    pub settled_text: Option<String>,
    pub label_source: Option<LabelSource>,
    pub edit_ratio: Option<f32>,
    pub app_context: Option<String>,
}
