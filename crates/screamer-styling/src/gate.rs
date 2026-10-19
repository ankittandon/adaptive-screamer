//! Triviality gate: a cheap, rules-based decision on whether a raw dictation is
//! worth sending to the styling model at all. No network, no model.
//!
//! It errs toward *calling* the model (a false "worth it" just costs one request
//! that the post-generation `edit_ratio` check then suppresses), and only skips
//! input that is clearly short and already well-formed.

/// Words that signal spoken filler worth cleaning up.
const FILLERS: &[&str] = &[
    "um", "uh", "erm", "like", "kinda", "sorta", "basically", "literally",
    "actually", "honestly", "anyway", "gonna", "wanna",
];
/// Multi-word filler phrases (matched on the lowercased text).
const FILLER_PHRASES: &[&str] = &[
    "yeah so", "so yeah", "i guess", "i mean", "you know", "kind of",
    "sort of", "i think", "or whatever",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Already clean enough — don't bother the styling model.
    Skip,
    /// Worth a rewrite.
    Style,
}

#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Below this word count, never style (too short to matter).
    pub min_words: usize,
    /// At or below this word count, skip when the text is already well-formed.
    pub short_clean_max_words: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            min_words: 3,
            short_clean_max_words: 8,
        }
    }
}

/// Decide whether `raw` is worth styling.
pub fn assess(raw: &str, cfg: &GateConfig) -> GateDecision {
    let trimmed = raw.trim();
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let wc = words.len();

    if wc < cfg.min_words {
        return GateDecision::Skip;
    }

    let lower = trimmed.to_lowercase();
    let has_filler_phrase = FILLER_PHRASES.iter().any(|p| lower.contains(p));
    let has_filler_word = words
        .iter()
        .any(|w| FILLERS.contains(&strip_word(w).as_str()));
    let has_terminal_punct = matches!(trimmed.chars().last(), Some('.') | Some('?') | Some('!'));
    let starts_upper = trimmed
        .chars()
        .find(|c| c.is_alphabetic())
        .map(|c| c.is_uppercase())
        .unwrap_or(false);
    let has_repeat = words
        .windows(2)
        .any(|w| strip_word(w[0]).eq_ignore_ascii_case(&strip_word(w[1])) && !strip_word(w[0]).is_empty());

    let clean = has_terminal_punct
        && starts_upper
        && !has_filler_word
        && !has_filler_phrase
        && !has_repeat;

    // Short and already well-formed → not worth a round trip.
    if clean && wc <= cfg.short_clean_max_words {
        return GateDecision::Skip;
    }

    GateDecision::Style
}

/// Lowercase a token with surrounding punctuation stripped, for matching.
fn strip_word(w: &str) -> String {
    w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> GateDecision {
        assess(s, &GateConfig::default())
    }

    #[test]
    fn short_clean_sentences_are_skipped() {
        assert_eq!(d("I agree with point one."), GateDecision::Skip);
        assert_eq!(d("Yes, you can do all three."), GateDecision::Skip);
        assert_eq!(d("What's a seed exactly?"), GateDecision::Skip);
        assert_eq!(d("Yes, this exact model works really well."), GateDecision::Skip);
    }

    #[test]
    fn very_short_is_skipped() {
        assert_eq!(d("ok thanks"), GateDecision::Skip);
        assert_eq!(d("sounds good"), GateDecision::Skip);
    }

    #[test]
    fn lowercase_or_unpunctuated_is_styled() {
        assert_eq!(d("okay let's hold off on this for now"), GateDecision::Style);
        assert_eq!(
            d("and by the way my dictation button is left control"),
            GateDecision::Style
        );
    }

    #[test]
    fn filler_triggers_styling() {
        assert_eq!(
            d("Yeah so I was thinking maybe we could look at the eval set."),
            GateDecision::Style
        );
        assert_eq!(
            d("Um, can you basically check the tone dimension?"),
            GateDecision::Style
        );
    }

    #[test]
    fn long_runons_are_styled_even_if_capitalized() {
        assert_eq!(
            d("Do you think the stylistic layer benefits from fine tuning or do you \
               think prompting glossary indexing harness improvements would be best there"),
            GateDecision::Style
        );
    }

    #[test]
    fn repetition_triggers_styling() {
        assert_eq!(d("I think the the architecture should change."), GateDecision::Style);
    }
}
