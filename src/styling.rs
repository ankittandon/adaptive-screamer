//! App-side orchestration for the opt-in styling layer.
//!
//! After a dictation is pasted, [`request`] runs the triviality gate and, if the
//! text is worth styling, calls the cloud generator on a background thread. A
//! usable rewrite is handed to the main thread through a shared slot; the main
//! thread (the waveform timer in `app.rs`) presents the floating card and manages
//! apply / dismiss.

use crate::config::Config;
use crate::logging;
use screamer_styling::{
    assess, FireworksGenerator, GateConfig, GateDecision, StyleContext, StyleGenerator,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Don't show a suggestion that changed almost nothing.
const MIN_SHOW_EDIT_RATIO: f32 = 0.02;

/// Whether the card offers to apply a rewrite (card mode) or to revert one that
/// was already pasted (inline mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionKind {
    Apply,
    Revert,
}

/// A rewrite ready to present, handed from the worker thread to the main thread.
pub struct ReadySuggestion {
    pub raw: String,
    pub styled: String,
    pub edit_ratio: f32,
    pub kind: SuggestionKind,
}

/// Push a revert card (inline mode): the styled text already landed; offer to go
/// back to raw.
pub fn offer_revert(
    raw: String,
    styled: String,
    edit_ratio: f32,
    ready: &Mutex<Option<ReadySuggestion>>,
) {
    if let Ok(mut slot) = ready.lock() {
        *slot = Some(ReadySuggestion {
            raw,
            styled,
            edit_ratio,
            kind: SuggestionKind::Revert,
        });
    }
}

pub struct StylingEngine {
    generator: Arc<FireworksGenerator>,
    gate: GateConfig,
}

/// `~/Library/Application Support/Screamer/fireworks.key`
pub fn key_file_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join("Screamer")
        .join("fireworks.key")
}

/// The Fireworks key: `FIREWORKS_API_KEY` if set, else the key file. A double-
/// clicked app does not inherit the shell env, so the file is the reliable path.
fn resolve_api_key() -> Option<String> {
    if let Ok(k) = std::env::var("FIREWORKS_API_KEY") {
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

/// Build the engine when styling is enabled and a key is available.
pub fn init(config: &Config) -> Option<Arc<StylingEngine>> {
    if !config.styling_enabled {
        return None;
    }
    let key = match resolve_api_key() {
        Some(k) => k,
        None => {
            eprintln!(
                "[screamer] Styling disabled: no Fireworks API key. Set FIREWORKS_API_KEY or write it to {}",
                key_file_path().display()
            );
            return None;
        }
    };
    let generator = Arc::new(FireworksGenerator::new(key));
    eprintln!(
        "[screamer] Styling on: model {} (card {}s)",
        generator.model(),
        config.styling_card_timeout_secs
    );
    Some(Arc::new(StylingEngine {
        generator,
        gate: GateConfig::default(),
    }))
}

/// Gate `raw` and, if worthwhile, generate a styled rewrite on a worker thread.
/// A usable result is written to `ready` for the main thread to pick up.
pub fn request(engine: &Arc<StylingEngine>, raw: String, ready: Arc<Mutex<Option<ReadySuggestion>>>) {
    let engine = engine.clone();
    std::thread::spawn(move || {
        if assess(&raw, &engine.gate) == GateDecision::Skip {
            logging::log_flow_event("styling", "gate", "skip (trivial)");
            return;
        }
        match engine.generator.style(&raw, &StyleContext::default()) {
            Ok(Some(result)) => {
                let ratio = screamer_telemetry::edit_ratio(&raw, &result.styled);
                if ratio < MIN_SHOW_EDIT_RATIO {
                    logging::log_flow_event(
                        "styling",
                        "suppress",
                        &format!("edit_ratio={ratio} (~unchanged) latency_ms={}", result.latency_ms),
                    );
                    return;
                }
                logging::log_flow_event(
                    "styling",
                    "ready",
                    &format!("edit_ratio={ratio} latency_ms={}", result.latency_ms),
                );
                if let Ok(mut slot) = ready.lock() {
                    *slot = Some(ReadySuggestion {
                        raw,
                        styled: result.styled,
                        edit_ratio: ratio,
                        kind: SuggestionKind::Apply,
                    });
                }
            }
            Ok(None) => logging::log_flow_event("styling", "declined", "no rewrite"),
            Err(err) => eprintln!("[screamer] Styling error: {err}"),
        }
    });
}

/// Fuzzy "same message" test (whitespace/punctuation-level differences are fine),
/// used to decide whether overwriting is safe and whether a suggestion was applied.
pub fn close_enough(a: &str, b: &str) -> bool {
    a.trim() == b.trim() || screamer_telemetry::edit_ratio(a, b) < 0.15
}

/// Synchronously gate and style `raw`, blocking the caller (used by inline mode,
/// which runs on the dictation worker thread). Returns the styled text plus its
/// edit ratio and latency, or `None` to paste the raw text instead (trivial,
/// unchanged, declined, or errored).
pub fn style_now(engine: &StylingEngine, raw: &str) -> Option<(String, f32, u128)> {
    if assess(raw, &engine.gate) == GateDecision::Skip {
        logging::log_flow_event("styling", "gate", "skip (trivial)");
        return None;
    }
    match engine.generator.style(raw, &StyleContext::default()) {
        Ok(Some(result)) => {
            let ratio = screamer_telemetry::edit_ratio(raw, &result.styled);
            if ratio < MIN_SHOW_EDIT_RATIO {
                logging::log_flow_event(
                    "styling",
                    "suppress",
                    &format!("edit_ratio={ratio} (~unchanged) latency_ms={}", result.latency_ms),
                );
                return None;
            }
            Some((result.styled, ratio, result.latency_ms))
        }
        Ok(None) => {
            logging::log_flow_event("styling", "declined", "no rewrite");
            None
        }
        Err(err) => {
            eprintln!("[screamer] Styling error: {err}");
            None
        }
    }
}

/// Whether styling should be skipped for the frontmost app (dev/chat apps where
/// you want raw text instantly, not a polished rewrite).
pub fn app_is_skipped(bundle_id: Option<&str>, skip: &[String]) -> bool {
    match bundle_id {
        Some(id) => skip.iter().any(|s| s == id),
        None => false,
    }
}
