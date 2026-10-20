//! A cheap, rules-based "slop detector": a weighted count of AI-writing habits,
//! scored per 1,000 words. Lower is better (more human). No model, no regex crate.
//!
//! Inspired by the de-slop detector in https://100toolkit.com/blog/deslop-model/
//! (human prose typically scores well under ~15). This is a pragmatic subset of
//! the habits that are reliably detectable without parsing: em/en dashes,
//! buzzwords, signpost phrases, over-joined clauses, and "not X, but Y". Harder
//! ones (rule-of-three, rhetorical-question-then-answer) are left for later.
//!
//! The buzzword list and the "excess vocabulary" framing draw on Kobak et al.,
//! "Delving into LLM-assisted writing in biomedical publications through excess
//! vocabulary," Science Advances 11(27), 2025 (arXiv:2406.07016): the LLM tells
//! are predominantly style words (verbs/adjectives like "delve", "intricate",
//! "meticulous"), found by their abrupt post-ChatGPT frequency jump. See
//! `docs/styling-guide.md`. Flagged words fade once publicized, so treat this
//! list as living, not fixed.

/// Buzzwords that read as AI/corporate filler. Matched as whole words.
const BUZZWORDS: &[&str] = &[
    "delve", "seamless", "seamlessly", "leverage", "leveraging", "utilize",
    "utilizing", "robust", "tapestry", "realm", "landscape", "testament",
    "underscore", "underscores", "showcasing", "navigating", "elevate",
    "unlock", "unlocking", "furthermore", "moreover", "notably",
];

/// Signpost phrases.
const SIGNPOSTS: &[&str] = &[
    "here's the thing", "the thing is", "at the end of the day",
    "needless to say", "it's worth noting", "that said", "when it comes to",
    "in today's world", "in the world of", "let's dive in", "let's unpack",
];

#[derive(Debug, Default, Clone)]
pub struct SlopBreakdown {
    pub words: usize,
    pub em_dashes: usize,
    pub buzzwords: usize,
    pub signposts: usize,
    pub over_joins: usize,
    pub not_but: usize,
}

impl SlopBreakdown {
    /// Weighted habits per 1,000 words.
    pub fn score(&self) -> f32 {
        let weighted = self.em_dashes as f32 * 2.0
            + self.buzzwords as f32 * 1.5
            + self.signposts as f32 * 2.0
            + self.over_joins as f32 * 1.0
            + self.not_but as f32 * 2.0;
        if self.words == 0 {
            0.0
        } else {
            let per = weighted / (self.words as f32 / 1000.0);
            (per * 10.0).round() / 10.0
        }
    }
}

/// Count AI-writing habits in `text`.
pub fn slop_analyze(text: &str) -> SlopBreakdown {
    let lower = text.to_lowercase();
    let words = text.split_whitespace().count();

    let em_dashes = text.matches('\u{2014}').count() + text.matches('\u{2013}').count();

    let word_tokens: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();
    let buzzwords = word_tokens
        .iter()
        .filter(|w| BUZZWORDS.contains(&w.as_str()))
        .count();

    let signposts = SIGNPOSTS.iter().map(|p| count_occurrences(&lower, p)).sum();

    let over_joins = count_occurrences(&lower, ", which ") + count_occurrences(&lower, ", so ");

    // "not X, but Y" / "it's not X, it's Y": a contrast connector whose preceding
    // clause recently contained "not"/"n't".
    let mut not_but = 0usize;
    for needle in [", but ", ", it's ", ", it is "] {
        let mut start = 0;
        while let Some(pos) = lower[start..].find(needle) {
            let at = start + pos;
            let window = &lower[at.saturating_sub(40)..at];
            if window.contains("not ") || window.contains("n't") {
                not_but += 1;
            }
            start = at + needle.len();
            if start >= lower.len() {
                break;
            }
        }
    }

    SlopBreakdown {
        words,
        em_dashes,
        buzzwords,
        signposts,
        over_joins,
        not_but,
    }
}

/// Slop habits per 1,000 words (lower is more human).
pub fn slop_score(text: &str) -> f32 {
    slop_analyze(text).score()
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        count += 1;
        start += pos + needle.len();
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_prose_scores_low() {
        let s = "Let's look at how Grammarly handles onboarding and mirror that for our styling layer.";
        assert!(slop_score(s) < 15.0, "score was {}", slop_score(s));
    }

    #[test]
    fn slop_scores_high() {
        let s = "Let's delve into this — it's not just a feature, it's a seamless tapestry, \
                 which showcases a robust landscape, so here's the thing: we must leverage it.";
        let b = slop_analyze(s);
        assert!(b.em_dashes >= 1);
        assert!(b.buzzwords >= 3);
        assert!(b.not_but >= 1);
        assert!(b.over_joins >= 1);
        assert!(b.signposts >= 1);
        assert!(slop_score(s) > 40.0, "score was {}", slop_score(s));
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(slop_score(""), 0.0);
    }
}
