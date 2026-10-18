//! App-side glue for opt-in dictation telemetry.
//!
//! The storage and labeling logic lives in `crates/screamer-telemetry`; this file
//! only adds the macOS pieces: where the data lives, which app is in front, and
//! reading the user's selection for the fix hotkey.

use crate::config::Config;
use crate::logging;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;
use screamer_telemetry::{NewDictation, TelemetryStore};
use std::path::PathBuf;
use std::sync::Arc;

/// `~/Library/Application Support/Screamer/telemetry`
pub fn telemetry_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join("Screamer")
        .join("telemetry")
}

/// Open the store when `telemetry_enabled` is set. Returns `None` when telemetry
/// is off or the directory cannot be created, so dictation never depends on it.
pub fn init(config: &Config) -> Option<Arc<TelemetryStore>> {
    if !config.telemetry_enabled {
        return None;
    }

    let dir = telemetry_dir();
    match TelemetryStore::open(&dir) {
        Ok(store) => {
            let fix_key = if config.fix_hotkey.is_empty() {
                "off".to_string()
            } else {
                config.fix_hotkey.clone()
            };
            eprintln!(
                "[screamer] Telemetry on: saving dictations to {} (fix key: {})",
                dir.display(),
                fix_key
            );
            Some(Arc::new(store))
        }
        Err(err) => {
            eprintln!(
                "[screamer] Telemetry disabled: cannot open {}: {err}",
                dir.display()
            );
            None
        }
    }
}

/// Save one pasted dictation. `samples` must be the 16 kHz audio that was sent to
/// the model. Failures are logged and otherwise ignored.
pub fn log_dictation(
    store: &TelemetryStore,
    samples: &[f32],
    text: &str,
    app_context: Option<String>,
) {
    let model = Config::load().model;
    let result = store.log_dictation(NewDictation {
        samples,
        raw_text: text,
        pasted_text: text,
        app_context: app_context.as_deref(),
        model: &model,
    });

    match result {
        Ok(id) => logging::log_flow_event(
            "telemetry",
            "logged",
            &format!("id={id} samples={}", samples.len()),
        ),
        Err(err) => eprintln!("[screamer] Telemetry: could not save dictation: {err}"),
    }
}

/// Read the current selection and attach it as the corrected transcript of the
/// last dictation. Returns true if a label was saved.
pub fn apply_fix_from_selection(store: &TelemetryStore) -> bool {
    let selected = match crate::paster::capture_selection() {
        Ok(text) => text,
        Err(err) => {
            eprintln!("[screamer] Fix: could not read the selection: {err}");
            return false;
        }
    };

    match store.apply_fix(&selected) {
        Ok(label) => {
            logging::log_flow_event(
                "telemetry",
                "fix",
                &format!("id={} edit_ratio={}", label.id, label.edit_ratio),
            );
            true
        }
        Err(err) => {
            eprintln!("[screamer] Fix rejected: {err}");
            false
        }
    }
}

/// Bundle id (e.g. `com.apple.Notes`) of the frontmost app. Never a window title.
pub fn frontmost_app_bundle_id() -> Option<String> {
    unsafe {
        let workspace: *mut AnyObject = msg_send![objc2::class!(NSWorkspace), sharedWorkspace];
        if workspace.is_null() {
            return None;
        }
        let app: *mut AnyObject = msg_send![workspace, frontmostApplication];
        if app.is_null() {
            return None;
        }
        let bundle_id: *mut NSString = msg_send![app, bundleIdentifier];
        if bundle_id.is_null() {
            return None;
        }
        Some((*bundle_id).to_string())
    }
}
