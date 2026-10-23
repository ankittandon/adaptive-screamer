use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OverlayPosition {
    #[default]
    Center,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppAppearance {
    #[default]
    Dark,
    Light,
}

pub struct PositionInfo {
    pub id: OverlayPosition,
    pub label: &'static str,
}

pub const POSITIONS: &[PositionInfo] = &[
    PositionInfo {
        id: OverlayPosition::Center,
        label: "Center",
    },
    PositionInfo {
        id: OverlayPosition::Top,
        label: "Top",
    },
    PositionInfo {
        id: OverlayPosition::Bottom,
        label: "Bottom",
    },
];

pub struct AppearanceInfo {
    pub id: AppAppearance,
    pub label: &'static str,
}

pub const APPEARANCES: &[AppearanceInfo] = &[
    AppearanceInfo {
        id: AppAppearance::Dark,
        label: "Dark",
    },
    AppearanceInfo {
        id: AppAppearance::Light,
        label: "Light",
    },
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: String,
    pub hotkey: String,
    #[serde(default)]
    pub overlay_position: OverlayPosition,
    #[serde(default)]
    pub appearance: AppAppearance,
    #[serde(default = "default_live_transcription")]
    pub live_transcription: bool,
    #[serde(default = "default_sound_effects")]
    pub sound_effects: bool,
    #[serde(default = "default_show_accessibility_helper_on_launch")]
    pub show_accessibility_helper_on_launch: bool,
    #[serde(default)]
    pub accessibility_helper_dismissed: bool,
    /// Opt-in: save each dictation's audio and text under
    /// `~/Library/Application Support/Screamer/telemetry` (never uploaded).
    #[serde(default)]
    pub telemetry_enabled: bool,
    /// Tap this key to attach the currently selected text as the corrected
    /// transcript of the last dictation. Empty means disabled.
    #[serde(default)]
    pub fix_hotkey: String,
    /// Opt-in: after dictation, offer a styled rewrite via the cloud styling
    /// model (requires a Fireworks API key). Default off.
    #[serde(default)]
    pub styling_enabled: bool,
    /// How long the styling suggestion card stays up before it is dismissed.
    #[serde(default = "default_styling_card_timeout_secs")]
    pub styling_card_timeout_secs: u64,
    /// Where the styling card anchors: `top_right` (default) or `bottom_center`.
    #[serde(default = "default_styling_card_position")]
    pub styling_card_position: String,
    /// How styling is applied: `inline` (default — paste the styled rewrite
    /// directly, waiting on the model) or `card` (paste raw instantly, then
    /// offer a suggestion card).
    #[serde(default = "default_styling_mode")]
    pub styling_mode: String,
    /// Frontmost-app bundle ids where styling never runs (raw pastes instantly).
    #[serde(default = "default_styling_skip_apps")]
    pub styling_skip_apps: Vec<String>,
    /// Opt-in: mirror the local telemetry log to Braintrust (cloud) on a
    /// background thread. Requires `telemetry_enabled` (there must be a log to
    /// ship) and a Braintrust API key. Never touches the dictation hot path.
    /// Default off.
    #[serde(default)]
    pub braintrust_enabled: bool,
    /// Braintrust project id (UUID) that traces are written to.
    #[serde(default = "default_braintrust_project")]
    pub braintrust_project: String,
    /// Braintrust org id (UUID). Required only for audio attachment uploads.
    #[serde(default = "default_braintrust_org")]
    pub braintrust_org: String,
    /// Braintrust API base, overridable for self-hosted / EU regions.
    #[serde(default = "default_braintrust_endpoint")]
    pub braintrust_endpoint: String,
    /// Also upload each dictation's audio clip to Braintrust as an attachment.
    /// Only has effect when `braintrust_enabled` is set. Default off.
    #[serde(default)]
    pub braintrust_upload_audio: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "base".to_string(),
            hotkey: "left_control".to_string(),
            overlay_position: OverlayPosition::default(),
            appearance: AppAppearance::default(),
            live_transcription: default_live_transcription(),
            sound_effects: default_sound_effects(),
            show_accessibility_helper_on_launch: default_show_accessibility_helper_on_launch(),
            accessibility_helper_dismissed: false,
            telemetry_enabled: false,
            fix_hotkey: String::new(),
            styling_enabled: false,
            styling_card_timeout_secs: default_styling_card_timeout_secs(),
            styling_card_position: default_styling_card_position(),
            styling_mode: default_styling_mode(),
            styling_skip_apps: default_styling_skip_apps(),
            braintrust_enabled: false,
            braintrust_project: default_braintrust_project(),
            braintrust_org: default_braintrust_org(),
            braintrust_endpoint: default_braintrust_endpoint(),
            braintrust_upload_audio: false,
        }
    }
}

fn default_braintrust_project() -> String {
    "92bf6a63-16e3-450c-ba02-2f3247e7ef92".to_string()
}

fn default_braintrust_org() -> String {
    "45a04b97-8f56-4ad9-b5be-bc9eed436c06".to_string()
}

fn default_braintrust_endpoint() -> String {
    "https://api.braintrust.dev/v1".to_string()
}

fn default_styling_card_timeout_secs() -> u64 {
    12
}

fn default_styling_card_position() -> String {
    "top_right".to_string()
}

fn default_styling_mode() -> String {
    "inline".to_string()
}

fn default_styling_skip_apps() -> Vec<String> {
    // Dev / chat apps where you want raw text instantly, not a polished rewrite.
    [
        "com.anthropic.claudefordesktop",
        "com.apple.Terminal",
        "com.googlecode.iterm2",
        "com.microsoft.VSCode",
        "com.apple.dt.Xcode",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn default_live_transcription() -> bool {
    true
}

fn default_sound_effects() -> bool {
    true
}

fn default_show_accessibility_helper_on_launch() -> bool {
    true
}

pub struct ModelInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub size: &'static str,
}

pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "tiny",
        label: "Tiny",
        size: "75 MB",
    },
    ModelInfo {
        id: "base",
        label: "Base",
        size: "142 MB",
    },
    ModelInfo {
        id: "small",
        label: "Small",
        size: "466 MB",
    },
    ModelInfo {
        id: "accessibility",
        label: "Accessibility (dysarthric, fine-tuned)",
        size: "142 MB",
    },
];

pub struct HotkeyInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub modifier_flag: u64,
    pub device_flag: u64,
}

pub const HOTKEYS: &[HotkeyInfo] = &[
    HotkeyInfo {
        id: "left_control",
        label: "Left Control ⌃",
        modifier_flag: 0x00040000,
        device_flag: 0x00000001,
    },
    HotkeyInfo {
        id: "right_control",
        label: "Right Control ⌃",
        modifier_flag: 0x00040000,
        device_flag: 0x00002000,
    },
    HotkeyInfo {
        id: "left_option",
        label: "Left Option ⌥",
        modifier_flag: 0x00080000,
        device_flag: 0x00000020,
    },
    HotkeyInfo {
        id: "right_option",
        label: "Right Option ⌥",
        modifier_flag: 0x00080000,
        device_flag: 0x00000040,
    },
    HotkeyInfo {
        id: "left_command",
        label: "Left Command ⌘",
        modifier_flag: 0x00100000,
        device_flag: 0x00000008,
    },
    HotkeyInfo {
        id: "right_command",
        label: "Right Command ⌘",
        modifier_flag: 0x00100000,
        device_flag: 0x00000010,
    },
    HotkeyInfo {
        id: "fn",
        label: "Fn (Globe) 🌐",
        modifier_flag: 0x00800000,
        device_flag: 0,
    },
];

impl Config {
    fn config_dir() -> PathBuf {
        let base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("~"));
        base.join("Screamer")
    }

    fn config_path() -> PathBuf {
        Self::config_dir().join("config.json")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        fs::read_to_string(&path)
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
            .map(Self::normalized)
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let dir = Self::config_dir();
        if let Err(err) = fs::create_dir_all(&dir) {
            eprintln!("[screamer] Failed to create config directory: {err}");
            return;
        }

        let normalized = self.clone().normalized();
        let mut json = match serde_json::to_vec_pretty(&normalized) {
            Ok(json) => json,
            Err(err) => {
                eprintln!("[screamer] Failed to serialize config: {err}");
                return;
            }
        };
        json.push(b'\n');

        let path = Self::config_path();
        let tmp = path.with_extension("json.tmp");
        let mut file = match File::create(&tmp) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("[screamer] Failed to create temporary config file: {err}");
                return;
            }
        };

        if let Err(err) = file.write_all(&json).and_then(|_| file.sync_all()) {
            eprintln!("[screamer] Failed to write config: {err}");
            let _ = fs::remove_file(&tmp);
            return;
        }

        if let Err(err) = fs::rename(&tmp, &path) {
            eprintln!("[screamer] Failed to replace config file: {err}");
            let _ = fs::remove_file(&tmp);
        }
    }

    pub fn hotkey_info(&self) -> &'static HotkeyInfo {
        HOTKEYS
            .iter()
            .find(|h| h.id == self.hotkey)
            .unwrap_or(&HOTKEYS[0])
    }

    pub fn hotkey_label(&self) -> &'static str {
        self.hotkey_info().label
    }

    pub fn position_label(&self) -> &'static str {
        POSITIONS
            .iter()
            .find(|p| p.id == self.overlay_position)
            .map(|p| p.label)
            .unwrap_or("Center")
    }

    pub fn appearance_label(&self) -> &'static str {
        APPEARANCES
            .iter()
            .find(|appearance| appearance.id == self.appearance)
            .map(|appearance| appearance.label)
            .unwrap_or("Dark")
    }

    fn normalized(mut self) -> Self {
        let default = Self::default();

        if !MODELS.iter().any(|model| model.id == self.model) {
            self.model = default.model;
        }

        if !HOTKEYS.iter().any(|hotkey| hotkey.id == self.hotkey) {
            self.hotkey = default.hotkey;
        }

        if !APPEARANCES
            .iter()
            .any(|appearance| appearance.id == self.appearance)
        {
            self.appearance = default.appearance;
        }

        // The fix key must be a real key that is not already the dictation key.
        if !HOTKEYS.iter().any(|hotkey| hotkey.id == self.fix_hotkey)
            || self.fix_hotkey == self.hotkey
        {
            self.fix_hotkey = default.fix_hotkey;
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = Config::default();
        assert_eq!(config.model, "base");
        assert_eq!(config.hotkey, "left_control");
        assert_eq!(config.overlay_position, OverlayPosition::Center);
        assert_eq!(config.appearance, AppAppearance::Dark);
        assert!(config.live_transcription);
        assert!(config.sound_effects);
        assert!(config.show_accessibility_helper_on_launch);
        assert!(!config.accessibility_helper_dismissed);
        assert!(!config.telemetry_enabled);
        assert_eq!(config.fix_hotkey, "");
    }

    #[test]
    fn config_roundtrip() {
        let config = Config {
            model: "tiny".to_string(),
            hotkey: "fn".to_string(),
            overlay_position: OverlayPosition::Bottom,
            appearance: AppAppearance::Light,
            live_transcription: false,
            sound_effects: false,
            show_accessibility_helper_on_launch: false,
            accessibility_helper_dismissed: true,
            telemetry_enabled: true,
            fix_hotkey: "right_command".to_string(),
            styling_enabled: true,
            styling_card_timeout_secs: 20,
            styling_card_position: "bottom_center".to_string(),
            styling_mode: "card".to_string(),
            styling_skip_apps: vec!["com.apple.Terminal".to_string()],
            braintrust_enabled: true,
            braintrust_project: "proj-123".to_string(),
            braintrust_org: "org-123".to_string(),
            braintrust_endpoint: "https://api.braintrust.dev/v1".to_string(),
            braintrust_upload_audio: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "tiny");
        assert_eq!(parsed.hotkey, "fn");
        assert_eq!(parsed.overlay_position, OverlayPosition::Bottom);
        assert_eq!(parsed.appearance, AppAppearance::Light);
        assert!(!parsed.live_transcription);
        assert!(!parsed.sound_effects);
        assert!(!parsed.show_accessibility_helper_on_launch);
        assert!(parsed.accessibility_helper_dismissed);
        assert!(parsed.telemetry_enabled);
        assert_eq!(parsed.fix_hotkey, "right_command");
        assert!(parsed.braintrust_enabled);
        assert_eq!(parsed.braintrust_project, "proj-123");
        assert!(parsed.braintrust_upload_audio);
    }

    #[test]
    fn config_backward_compat() {
        // Old config, including fields from removed features, loads with the
        // unknown fields ignored and missing ones defaulted.
        let json = r#"{"model":"base","hotkey":"left_control","summary_backend":"bundled","vision_hotkey":"left_option","ambient_microphone":true}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.overlay_position, OverlayPosition::Center);
        assert_eq!(config.appearance, AppAppearance::Dark);
        assert!(config.live_transcription);
        assert!(config.sound_effects);
        assert!(config.show_accessibility_helper_on_launch);
        assert!(!config.accessibility_helper_dismissed);
        assert!(!config.telemetry_enabled);
        assert_eq!(config.fix_hotkey, "");
    }

    #[test]
    fn hotkey_info_lookup() {
        let config = Config {
            hotkey: "right_option".to_string(),
            ..Config::default()
        };
        let info = config.hotkey_info();
        assert_eq!(info.id, "right_option");
        assert_eq!(info.modifier_flag, 0x00080000);
    }

    #[test]
    fn hotkey_info_fallback() {
        let config = Config {
            hotkey: "nonexistent".to_string(),
            ..Config::default()
        };
        let info = config.hotkey_info();
        assert_eq!(info.id, "left_control");
    }

    #[test]
    fn position_label() {
        let config = Config {
            overlay_position: OverlayPosition::Top,
            ..Config::default()
        };
        assert_eq!(config.position_label(), "Top");
    }

    #[test]
    fn appearance_label() {
        let config = Config {
            appearance: AppAppearance::Light,
            ..Config::default()
        };
        assert_eq!(config.appearance_label(), "Light");
    }

    #[test]
    fn config_from_invalid_json() {
        let result: Result<Config, _> = serde_json::from_str("not json");
        assert!(result.is_err());
    }

    #[test]
    fn normalize_invalid_config_values() {
        let config = Config {
            model: "unknown".to_string(),
            hotkey: "unknown".to_string(),
            ..Config::default()
        }
        .normalized();

        assert_eq!(config.model, "base");
        assert_eq!(config.hotkey, "left_control");
    }

    #[test]
    fn normalize_keeps_a_free_fix_hotkey() {
        let config = Config {
            fix_hotkey: "right_command".to_string(),
            ..Config::default()
        }
        .normalized();

        assert_eq!(config.fix_hotkey, "right_command");
    }

    #[test]
    fn normalize_disables_a_fix_hotkey_that_is_unknown_or_the_dictation_key() {
        for taken in ["nonexistent", "left_control"] {
            let config = Config {
                fix_hotkey: taken.to_string(),
                ..Config::default()
            }
            .normalized();

            assert_eq!(config.fix_hotkey, "", "fix_hotkey {taken:?} should be disabled");
        }
    }
}
