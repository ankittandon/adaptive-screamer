use crate::config::{Config, HOTKEYS};
use objc2::msg_send;
use objc2_foundation::MainThreadMarker;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// `fix_index` value meaning "no fix hotkey configured".
const FIX_DISABLED: usize = usize::MAX;

pub struct Hotkey {
    dictation_pressed: Arc<AtomicBool>,
    fix_pressed: Arc<AtomicBool>,
    dictation_index: Arc<AtomicUsize>,
    fix_index: Arc<AtomicUsize>,
}

impl Hotkey {
    pub fn new(config: &Config) -> Self {
        Self {
            dictation_pressed: Arc::new(AtomicBool::new(false)),
            fix_pressed: Arc::new(AtomicBool::new(false)),
            dictation_index: Arc::new(AtomicUsize::new(hotkey_index_for_id(&config.hotkey))),
            fix_index: Arc::new(AtomicUsize::new(fix_index_for_id(&config.fix_hotkey))),
        }
    }

    pub fn set_hotkey(&self, hotkey_id: &str) {
        let index = hotkey_index_for_id(hotkey_id);
        let hotkey_info = HOTKEYS.get(index).unwrap_or(&HOTKEYS[0]);
        self.dictation_index.store(index, Ordering::Relaxed);
        eprintln!(
            "[screamer] Updated dictation hotkey to: {} (modifier=0x{:x}, device=0x{:x})",
            hotkey_info.label, hotkey_info.modifier_flag, hotkey_info.device_flag
        );
    }

    /// Start listening using an NSEvent global monitor on the main thread.
    ///
    /// - Dictation: `on_dictation_press` / `on_dictation_release` (configurable push-to-talk key)
    /// - Fix (telemetry): `on_fix_release`, an optional tap key. Only the release is
    ///   reported, and only if the key went down while dictation was not held.
    pub fn start_on_main_thread(
        &self,
        _mtm: MainThreadMarker,
        on_dictation_press: impl Fn() + 'static,
        on_dictation_release: impl Fn() + 'static,
        on_fix_release: impl Fn() + 'static,
    ) {
        let dictation_pressed = self.dictation_pressed.clone();
        let dictation_index = self.dictation_index.clone();
        let fix_pressed = self.fix_pressed.clone();
        let fix_index = self.fix_index.clone();

        let dict_info = HOTKEYS
            .get(dictation_index.load(Ordering::Relaxed))
            .unwrap_or(&HOTKEYS[0]);

        eprintln!(
            "[screamer] Dictation hotkey: {} (modifier=0x{:x}, device=0x{:x})",
            dict_info.label, dict_info.modifier_flag, dict_info.device_flag
        );

        // NSEventMaskFlagsChanged = 1 << 12 = 4096
        let mask: u64 = 1 << 12;

        let block = block2::RcBlock::new(move |event: *mut objc2::runtime::AnyObject| {
            if event.is_null() {
                return;
            }
            let flags: u64 = unsafe { msg_send![event, modifierFlags] };

            // ── Dictation hotkey ──
            let dict_info = HOTKEYS
                .get(dictation_index.load(Ordering::Relaxed))
                .unwrap_or(&HOTKEYS[0]);

            let dict_modifier_down = (flags & dict_info.modifier_flag) != 0;
            let dict_key_down = if dict_info.device_flag != 0 {
                dict_modifier_down && (flags & dict_info.device_flag) != 0
            } else {
                dict_modifier_down
            };

            let dict_was = dictation_pressed.load(Ordering::SeqCst);

            if dict_key_down && !dict_was {
                eprintln!(
                    "[screamer] Dictation {} PRESSED (flags=0x{:x})",
                    dict_info.label, flags
                );
                dictation_pressed.store(true, Ordering::SeqCst);
                on_dictation_press();
            } else if !dict_key_down && dict_was {
                eprintln!(
                    "[screamer] Dictation {} RELEASED (flags=0x{:x})",
                    dict_info.label, flags
                );
                dictation_pressed.store(false, Ordering::SeqCst);
                on_dictation_release();
            }

            // ── Fix hotkey (telemetry): tap to label the last dictation ──
            let fix_idx = fix_index.load(Ordering::Relaxed);
            if let Some(fix_info) = HOTKEYS.get(fix_idx) {
                let taken = fix_idx == dictation_index.load(Ordering::Relaxed);

                let fix_modifier_down = (flags & fix_info.modifier_flag) != 0;
                let fix_key_down = if fix_info.device_flag != 0 {
                    fix_modifier_down && (flags & fix_info.device_flag) != 0
                } else {
                    fix_modifier_down
                };

                let fix_was = fix_pressed.load(Ordering::SeqCst);

                if fix_key_down && !fix_was {
                    if !taken && !dictation_pressed.load(Ordering::SeqCst) {
                        fix_pressed.store(true, Ordering::SeqCst);
                    }
                } else if !fix_key_down && fix_was {
                    eprintln!(
                        "[screamer] Fix {} tapped (flags=0x{:x})",
                        fix_info.label, flags
                    );
                    fix_pressed.store(false, Ordering::SeqCst);
                    on_fix_release();
                }
            } else {
                fix_pressed.store(false, Ordering::SeqCst);
            }
        });

        unsafe {
            let _monitor: *mut objc2::runtime::AnyObject = msg_send![
                objc2::class!(NSEvent),
                addGlobalMonitorForEventsMatchingMask: mask,
                handler: &*block
            ];

            if _monitor.is_null() {
                eprintln!("[screamer] Failed to create NSEvent global monitor");
            } else {
                eprintln!("[screamer] NSEvent global monitor installed for FlagsChanged");
            }
        }
    }
}

fn fix_index_for_id(hotkey_id: &str) -> usize {
    HOTKEYS
        .iter()
        .position(|hotkey| hotkey.id == hotkey_id)
        .unwrap_or(FIX_DISABLED)
}

fn hotkey_index_for_id(hotkey_id: &str) -> usize {
    HOTKEYS
        .iter()
        .position(|hotkey| hotkey.id == hotkey_id)
        .unwrap_or(0)
}
