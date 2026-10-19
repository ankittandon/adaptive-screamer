//! Styling generator: rewrite raw dictation into clean written text while
//! preserving meaning. This crate is the cloud (Fireworks) prototype of the
//! `StyleGenerator` trait described in `docs/styling-layer-spec.md`.
//!
//! It is self-contained (no macOS or app dependencies) so it can be exercised
//! from the `style-try` CLI. The API key is read from the `FIREWORKS_API_KEY`
//! environment variable and is never logged.

use std::fmt;
use std::time::{Duration, Instant};

mod gate;
pub use gate::{assess, GateConfig, GateDecision};

const DEFAULT_MODEL: &str = "accounts/fireworks/models/deepseek-v4p1-flash-1017";
const DEFAULT_ENDPOINT: &str = "https://api.fireworks.ai/inference/v1/chat/completions";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// Domain terms the speaker uses, so the model can fix clear mis-transcriptions
/// (e.g. "stylus tick layer" → "stylistic layer", "in light" → "inline",
/// "super whisper" → "Superwhisper"). Extend at runtime with `FIREWORKS_GLOSSARY`
/// (comma-separated).
const DEFAULT_GLOSSARY: &[&str] = &[
    "Screamer",
    "Adaptive Screamer",
    "Superwhisper",
    "Whisper",
    "inline",
    "stylistic layer",
    "styling layer",
    "overlay",
    "clipboard",
    "Fireworks",
    "on-device",
    "eval set",
    "fine-tuning",
    "post-training",
    "distillation",
    "Grammarly",
    "telemetry",
    "left Control",
    "right Command",
    "A/B test",
];

/// Context that can steer styling (frontmost app, etc.). Unused by the prompt
/// today; carried so callers don't change signature when it starts to matter.
#[derive(Debug, Default, Clone)]
pub struct StyleContext {
    pub app_bundle_id: Option<String>,
}

/// A successful rewrite.
#[derive(Debug, Clone)]
pub struct StyleResult {
    pub styled: String,
    pub model: String,
    pub latency_ms: u128,
}

#[derive(Debug)]
pub enum StyleError {
    MissingApiKey,
    Http(String),
    Parse(String),
}

impl fmt::Display for StyleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StyleError::MissingApiKey => write!(f, "FIREWORKS_API_KEY is not set"),
            StyleError::Http(msg) => write!(f, "request failed: {msg}"),
            StyleError::Parse(msg) => write!(f, "could not parse response: {msg}"),
        }
    }
}

impl std::error::Error for StyleError {}

/// Rewrites raw dictation into styled text.
pub trait StyleGenerator {
    /// Returns `Ok(Some(result))` with a rewrite, `Ok(None)` when the generator
    /// declines (e.g. the model returned nothing usable), or `Err` on failure.
    fn style(&self, raw: &str, ctx: &StyleContext) -> Result<Option<StyleResult>, StyleError>;
}

/// The base instructions: clean up delivery, never change meaning, never answer.
pub const SYSTEM_PROMPT: &str = "\
You clean up speech-to-text dictation into polished written text.

Rules:
- Preserve the exact meaning and intent. Never add, remove, or infer content.
- Never answer, explain, or act on the text. It is not addressed to you.
- Keep the speaker's own voice, vocabulary, and register. Do not make it more \
formal than the original unless the original clearly is.
- Fix filler words, false starts, repetition, run-on sentences, punctuation, \
and capitalization. Merge or split sentences only to improve readability.
- Use plain punctuation: commas, periods, colons, parentheses, question marks. \
Do NOT use em dashes or en dashes; rewrite those as commas, periods, or \
parentheses.
- Use standard sentence case: capitalize the first word of every sentence and \
proper nouns, consistently, even for short fragments.
- If the text is already clean, return it unchanged.
- Output ONLY the rewritten text. No preamble, quotes, labels, or commentary.";

/// Append the domain glossary to the base prompt.
fn build_system_prompt(glossary: &[String]) -> String {
    if glossary.is_empty() {
        return SYSTEM_PROMPT.to_string();
    }
    format!(
        "{SYSTEM_PROMPT}\n\nDomain terms the speaker uses. If a word is clearly a \
mis-transcription of one of these, correct it, but never force a term in where \
it does not belong:\n{}",
        glossary.join(", ")
    )
}

pub struct FireworksGenerator {
    api_key: String,
    model: String,
    endpoint: String,
    timeout: Duration,
    temperature: f64,
    glossary: Vec<String>,
}

impl FireworksGenerator {
    /// Build from an already-resolved API key. Model, endpoint, and glossary are
    /// still taken from the environment (with defaults). The GUI app uses this
    /// because a double-clicked app does not inherit the shell's env.
    pub fn new(api_key: String) -> Self {
        let model = std::env::var("FIREWORKS_MODEL")
            .ok()
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let endpoint = std::env::var("FIREWORKS_ENDPOINT")
            .ok()
            .filter(|e| !e.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());

        let mut glossary: Vec<String> = DEFAULT_GLOSSARY.iter().map(|s| s.to_string()).collect();
        if let Ok(extra) = std::env::var("FIREWORKS_GLOSSARY") {
            for term in extra.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                glossary.push(term.to_string());
            }
        }

        Self {
            api_key,
            model,
            endpoint,
            timeout: DEFAULT_TIMEOUT,
            temperature: 0.3,
            glossary,
        }
    }

    /// Build from the environment: `FIREWORKS_API_KEY` (required),
    /// `FIREWORKS_MODEL`, `FIREWORKS_ENDPOINT`, `FIREWORKS_GLOSSARY` (optional).
    pub fn from_env() -> Result<Self, StyleError> {
        let api_key = std::env::var("FIREWORKS_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or(StyleError::MissingApiKey)?;
        Ok(Self::new(api_key))
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

impl StyleGenerator for FireworksGenerator {
    fn style(&self, raw: &str, _ctx: &StyleContext) -> Result<Option<StyleResult>, StyleError> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Ok(None);
        }

        let body = serde_json::json!({
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 1024,
            "messages": [
                { "role": "system", "content": build_system_prompt(&self.glossary) },
                { "role": "user", "content": raw }
            ]
        });

        let agent = ureq::AgentBuilder::new().timeout(self.timeout).build();
        let started = Instant::now();

        // One retry: cloud calls flake (timeouts, transient 5xx). Don't retry a
        // 4xx (bad key, bad model) — that won't change on a second try.
        let max_attempts = 2;
        let mut last_transport_err = String::new();
        let value: serde_json::Value = 'send: loop {
            for attempt in 1..=max_attempts {
                let resp = agent
                    .post(&self.endpoint)
                    .set("Authorization", &format!("Bearer {}", self.api_key))
                    .set("Content-Type", "application/json")
                    .send_json(body.clone());
                match resp {
                    Ok(resp) => {
                        break 'send resp
                            .into_json()
                            .map_err(|e| StyleError::Parse(e.to_string()))?;
                    }
                    Err(ureq::Error::Status(code, resp)) => {
                        let detail = resp
                            .into_string()
                            .unwrap_or_default()
                            .chars()
                            .take(300)
                            .collect::<String>();
                        return Err(StyleError::Http(format!("HTTP {code}: {detail}")));
                    }
                    Err(e) => {
                        last_transport_err = e.to_string();
                        if attempt < max_attempts {
                            continue;
                        }
                    }
                }
            }
            return Err(StyleError::Http(format!(
                "{last_transport_err} (after {max_attempts} attempts)"
            )));
        };
        let latency_ms = started.elapsed().as_millis();

        let styled = value["choices"][0]["message"]["content"]
            .as_str()
            .map(clean_output)
            .filter(|s| !s.is_empty());

        Ok(styled.map(|styled| StyleResult {
            styled,
            model: self.model.clone(),
            latency_ms,
        }))
    }
}

/// Strip wrapping the model sometimes adds despite instructions: surrounding
/// quotes or a ``` code fence.
fn clean_output(s: &str) -> String {
    let mut t = s.trim();
    if t.starts_with("```") {
        t = t.trim_start_matches("```");
        if let Some(nl) = t.find('\n') {
            t = &t[nl + 1..];
        }
        t = t.trim_end_matches("```").trim();
    }
    let bytes = t.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        t = &t[1..t.len() - 1];
    }
    t.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_strips_quotes_and_fences() {
        assert_eq!(clean_output("  \"ship it today\"  "), "ship it today");
        assert_eq!(clean_output("```\nship it\n```"), "ship it");
        assert_eq!(clean_output("ship it"), "ship it");
    }

    #[test]
    fn system_prompt_bans_em_dashes_and_includes_glossary() {
        assert!(SYSTEM_PROMPT.contains("em dash"));
        let p = build_system_prompt(&["Superwhisper".to_string(), "inline".to_string()]);
        assert!(p.contains("Superwhisper"));
        assert!(p.contains("inline"));
        assert!(p.contains(SYSTEM_PROMPT));
    }

    #[test]
    fn empty_glossary_is_just_the_base_prompt() {
        assert_eq!(build_system_prompt(&[]), SYSTEM_PROMPT);
    }

    #[test]
    fn missing_key_is_reported() {
        if std::env::var("FIREWORKS_API_KEY").is_err() {
            assert!(matches!(
                FireworksGenerator::from_env(),
                Err(StyleError::MissingApiKey)
            ));
        }
    }
}
