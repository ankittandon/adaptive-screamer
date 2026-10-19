//! Styling generator: rewrite raw dictation into clean written text while
//! preserving meaning. This crate is the cloud (Fireworks) prototype of the
//! `StyleGenerator` trait described in `docs/styling-layer-spec.md`.
//!
//! It is self-contained (no macOS or app dependencies) so it can be exercised
//! from the `style-try` CLI. The API key is read from the `FIREWORKS_API_KEY`
//! environment variable and is never logged.

use std::fmt;
use std::time::{Duration, Instant};

const DEFAULT_MODEL: &str = "accounts/fireworks/models/llama-v3p1-8b-instruct";
const DEFAULT_ENDPOINT: &str = "https://api.fireworks.ai/inference/v1/chat/completions";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

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
            StyleError::MissingApiKey => {
                write!(f, "FIREWORKS_API_KEY is not set")
            }
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

/// The instruction that constrains the rewrite: clean up delivery, never change
/// meaning, never answer or act on the text.
pub const SYSTEM_PROMPT: &str = "\
You clean up speech-to-text dictation into polished written text.

Rules:
- Preserve the exact meaning and intent. Never add, remove, or infer content.
- Never answer, explain, or act on the text. It is not addressed to you.
- Keep the speaker's own voice, vocabulary, and register. Do not make it more \
formal than the original unless the original clearly is.
- Fix filler words, false starts, repetition, run-on sentences, punctuation, \
and capitalization. Merge or split sentences only to improve readability.
- If the text is already clean, return it unchanged.
- Output ONLY the rewritten text. No preamble, quotes, labels, or commentary.";

pub struct FireworksGenerator {
    api_key: String,
    model: String,
    endpoint: String,
    timeout: Duration,
    temperature: f64,
}

impl FireworksGenerator {
    /// Build from the environment: `FIREWORKS_API_KEY` (required),
    /// `FIREWORKS_MODEL` and `FIREWORKS_ENDPOINT` (optional overrides).
    pub fn from_env() -> Result<Self, StyleError> {
        let api_key = std::env::var("FIREWORKS_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or(StyleError::MissingApiKey)?;
        let model = std::env::var("FIREWORKS_MODEL")
            .ok()
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let endpoint = std::env::var("FIREWORKS_ENDPOINT")
            .ok()
            .filter(|e| !e.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        Ok(Self {
            api_key,
            model,
            endpoint,
            timeout: DEFAULT_TIMEOUT,
            temperature: 0.3,
        })
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
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": raw }
            ]
        });

        let agent = ureq::AgentBuilder::new()
            .timeout(self.timeout)
            .build();

        let started = Instant::now();
        let response = agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body);

        let value: serde_json::Value = match response {
            Ok(resp) => resp
                .into_json()
                .map_err(|e| StyleError::Parse(e.to_string()))?,
            Err(ureq::Error::Status(code, resp)) => {
                // Surface the API's error message, but never the key.
                let detail = resp
                    .into_string()
                    .unwrap_or_default()
                    .chars()
                    .take(300)
                    .collect::<String>();
                return Err(StyleError::Http(format!("HTTP {code}: {detail}")));
            }
            Err(e) => return Err(StyleError::Http(e.to_string())),
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
    fn missing_key_is_reported() {
        // Only meaningful when the var is unset in the environment.
        if std::env::var("FIREWORKS_API_KEY").is_err() {
            assert!(matches!(
                FireworksGenerator::from_env(),
                Err(StyleError::MissingApiKey)
            ));
        }
    }
}
