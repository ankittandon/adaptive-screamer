use core_graphics::event::{CGEvent, CGEventFlags, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use std::ffi::c_void;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const VK_ANSI_A: CGKeyCode = 0;
const VK_ANSI_C: CGKeyCode = 8;
const VK_ANSI_V: CGKeyCode = 9;

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

pub fn paste(text: &str) -> Result<(), String> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("Failed to open clipboard: {e}"))?;
    clipboard
        .set_text(text)
        .map_err(|e| format!("Failed to set clipboard: {e}"))?;

    if !has_accessibility_permission() {
        return Err(
            "Copied to clipboard, but automatic paste is blocked because Accessibility access is not enabled for Screamer.".to_string(),
        );
    }

    // Minimal delay for clipboard sync
    thread::sleep(Duration::from_millis(5));

    // Simulate Cmd+V
    post_command_key(VK_ANSI_V)
}

/// Copy whatever is selected in the frontmost app and return it.
///
/// Used by the opt-in telemetry "fix" hotkey. The clipboard's previous text is
/// put back afterwards; non-text clipboard contents (an image, say) are not
/// preserved.
pub fn capture_selection() -> Result<String, String> {
    if !has_accessibility_permission() {
        return Err("Accessibility access is not enabled for Screamer.".to_string());
    }

    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("Failed to open clipboard: {e}"))?;
    let previous = clipboard.get_text().ok();

    // A unique marker tells us whether Cmd+C really replaced the clipboard. If
    // nothing is selected the copy does nothing and the marker stays put.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let marker = format!("screamer-selection-marker-{}-{nanos}", std::process::id());
    clipboard
        .set_text(marker.clone())
        .map_err(|e| format!("Failed to set clipboard: {e}"))?;
    thread::sleep(Duration::from_millis(5));

    let posted = post_command_key(VK_ANSI_C);

    let mut captured = None;
    if posted.is_ok() {
        for _ in 0..40 {
            thread::sleep(Duration::from_millis(10));
            if let Ok(text) = clipboard.get_text() {
                if text != marker {
                    captured = Some(text);
                    break;
                }
            }
        }
    }

    match previous {
        Some(text) => {
            let _ = clipboard.set_text(text);
        }
        None => {
            let _ = clipboard.clear();
        }
    }

    posted?;
    captured.ok_or_else(|| "nothing was selected".to_string())
}

/// Read the clipboard's current text, if any. Used to save it before the
/// styling suggestion borrows it, so it can be restored afterwards.
pub fn get_clipboard() -> Option<String> {
    arboard::Clipboard::new().ok().and_then(|mut c| c.get_text().ok())
}

/// Put text on the clipboard.
pub fn set_clipboard(text: &str) -> Result<(), String> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("Failed to open clipboard: {e}"))?;
    clipboard
        .set_text(text)
        .map_err(|e| format!("Failed to set clipboard: {e}"))
}

/// Restore a clipboard value captured earlier (clearing it when there was none).
pub fn restore_clipboard(previous: Option<String>) {
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        match previous {
            Some(text) => {
                let _ = clipboard.set_text(text);
            }
            None => {
                let _ = clipboard.clear();
            }
        }
    }
}

/// Select all text in the focused field (Cmd+A). Used so the styling suggestion
/// can be applied with a single Cmd+V. No-op (error) without Accessibility.
pub fn select_all() -> Result<(), String> {
    if !has_accessibility_permission() {
        return Err("Accessibility access is not enabled for Screamer.".to_string());
    }
    post_command_key(VK_ANSI_A)
}

/// Post Cmd+V to the focused field (the clipboard already holds the styled text).
/// Used by the card's Apply button.
pub fn send_paste() -> Result<(), String> {
    if !has_accessibility_permission() {
        return Err("Accessibility access is not enabled for Screamer.".to_string());
    }
    post_command_key(VK_ANSI_V)
}

fn post_command_key(keycode: CGKeyCode) -> Result<(), String> {
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| "Failed to create CoreGraphics event source".to_string())?;

    let key_down = CGEvent::new_keyboard_event(source.clone(), keycode, true)
        .map_err(|_| format!("Failed to create Cmd+key-down event (keycode {keycode})"))?;
    key_down.set_flags(CGEventFlags::CGEventFlagCommand);

    let key_up = CGEvent::new_keyboard_event(source, keycode, false)
        .map_err(|_| format!("Failed to create Cmd+key-up event (keycode {keycode})"))?;
    key_up.set_flags(CGEventFlags::CGEventFlagCommand);

    key_down.post(core_graphics::event::CGEventTapLocation::HID);
    key_up.post(core_graphics::event::CGEventTapLocation::HID);

    Ok(())
}

fn has_accessibility_permission() -> bool {
    let _ = std::ptr::null::<c_void>();
    unsafe { AXIsProcessTrusted() }
}
