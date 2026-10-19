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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    raw_text: &str,
    pasted_text: &str,
    app_context: Option<String>,
) {
    let model = Config::load().model;
    let result = store.log_dictation(NewDictation {
        samples,
        raw_text,
        pasted_text,
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

/// How often the implicit-capture watcher reads the focused text field.
const IMPLICIT_POLL_INTERVAL: Duration = Duration::from_millis(400);
/// Stop watching a dictation this long after it was logged.
const IMPLICIT_MAX_WATCH: Duration = Duration::from_secs(10 * 60);
/// Consecutive empty/unreadable reads (after the field held text) that count as
/// "the message was sent / the field was cleared".
const IMPLICIT_CLEAR_DEBOUNCE: u32 = 2;

/// Generation counter so only the newest dictation is watched; each new
/// dictation supersedes any still-running watcher.
static WATCH_GEN: AtomicU64 = AtomicU64::new(0);

/// After a dictation is logged, watch the frontmost focused text field. When it
/// clears (you sent or submitted the message), record the final user-edited text
/// as an implicit correction of that dictation — no hotkey, no selection needed.
///
/// Only the most recent dictation is watched at a time, and the store's own
/// guardrails (not empty, within the fix window, not wildly different from the
/// raw transcript) still apply, so an unrelated field value is rejected.
pub fn start_implicit_capture(store: Arc<TelemetryStore>) {
    let my_gen = WATCH_GEN.fetch_add(1, Ordering::SeqCst) + 1;

    // Nudge Electron/Chromium apps (Claude, Slack, …) to expose their text to
    // the Accessibility API; otherwise AXValue reads come back empty.
    if let Some(pid) = frontmost_app_pid() {
        crate::axtext::enable_accessibility_for_app(pid);
    }

    std::thread::spawn(move || {
        let start = Instant::now();
        let mut last_nonempty: Option<String> = None;
        let mut saw_text = false;
        let mut empty_reads: u32 = 0;

        loop {
            if WATCH_GEN.load(Ordering::SeqCst) != my_gen {
                return;
            }
            if start.elapsed() >= IMPLICIT_MAX_WATCH {
                if !saw_text {
                    logging::log_flow_event(
                        "telemetry",
                        "implicit",
                        "gave up: focused text field was never readable via Accessibility",
                    );
                }
                return;
            }
            std::thread::sleep(IMPLICIT_POLL_INTERVAL);
            if WATCH_GEN.load(Ordering::SeqCst) != my_gen {
                return;
            }

            match crate::axtext::focused_text() {
                Some(text) if !text.trim().is_empty() => {
                    if !saw_text {
                        logging::log_flow_event(
                            "telemetry",
                            "implicit",
                            "watching focused field (readable via Accessibility)",
                        );
                    }
                    last_nonempty = Some(text);
                    saw_text = true;
                    empty_reads = 0;
                }
                _ => {
                    if !saw_text {
                        continue; // field not focused/readable yet — keep waiting
                    }
                    empty_reads += 1;
                    if empty_reads < IMPLICIT_CLEAR_DEBOUNCE {
                        continue;
                    }
                    // The field held text and is now empty: treat the last
                    // non-empty value as the final, user-edited message.
                    if let Some(final_text) = last_nonempty.take() {
                        match store.apply_implicit_fix(&final_text) {
                            Ok(label) => logging::log_flow_event(
                                "telemetry",
                                "implicit",
                                &format!("id={} edit_ratio={}", label.id, label.edit_ratio),
                            ),
                            Err(err) => eprintln!("[screamer] Implicit label skipped: {err}"),
                        }
                    }
                    return;
                }
            }
        }
    });
}

/// Process id of the frontmost app, used to enable its Accessibility tree.
fn frontmost_app_pid() -> Option<i32> {
    unsafe {
        let workspace: *mut AnyObject = msg_send![objc2::class!(NSWorkspace), sharedWorkspace];
        if workspace.is_null() {
            return None;
        }
        let app: *mut AnyObject = msg_send![workspace, frontmostApplication];
        if app.is_null() {
            return None;
        }
        let pid: i32 = msg_send![app, processIdentifier];
        Some(pid)
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
