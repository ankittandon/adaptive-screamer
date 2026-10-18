//! Opt-in, local-only dictation telemetry.
//!
//! Every dictation can be saved as a 16 kHz mono WAV clip plus a JSONL record
//! (`raw_text`, `pasted_text`, `app_context`, ...). When the user later corrects
//! the pasted text and taps the "fix" key, the corrected text is attached to the
//! most recent dictation as `settled_text`, together with an `edit_ratio`. The
//! labeled pairs are the training data for fine-tuning Whisper on the user's own
//! vocabulary and voice.
//!
//! Nothing here touches the network. All files live under one directory and are
//! created readable by the current user only.
//!
//! The crate is plain Rust with no macOS dependencies, so its tests run anywhere.

mod edit;
mod record;
mod store;
mod timefmt;

pub use edit::{edit_ratio, word_edit_distance};
pub use record::{DictationRecord, Event, ExportRow, LabelRecord, LabelSource};
pub use store::{
    export_rows, write_export_jsonl, FixError, NewDictation, TelemetryStore, DEFAULT_FIX_MAX_AGE,
    MAX_FIX_EDIT_RATIO,
};
pub use timefmt::format_rfc3339_utc;
