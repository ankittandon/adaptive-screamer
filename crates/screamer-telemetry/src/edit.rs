/// Word-level Levenshtein distance between two strings.
///
/// Words are whitespace-separated and compared exactly: case and punctuation
/// count, because "kubernetes" vs "Kubernetes" is exactly the kind of error the
/// fine-tune is meant to fix.
pub fn word_edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<&str> = a.split_whitespace().collect();
    let b: Vec<&str> = b.split_whitespace().collect();

    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];

    for (i, wa) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, wb) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(wa != wb);
            let deletion = prev[j + 1] + 1;
            let insertion = cur[j] + 1;
            cur[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    prev[b.len()]
}

/// How much of the transcript the user changed, from 0.0 (untouched) to 1.0
/// (everything different): word edit distance divided by the longer word count.
/// Rounded to four decimals so records stay readable.
pub fn edit_ratio(raw: &str, settled: &str) -> f32 {
    let longest = raw
        .split_whitespace()
        .count()
        .max(settled.split_whitespace().count());
    if longest == 0 {
        return 0.0;
    }
    let ratio = word_edit_distance(raw, settled) as f32 / longest as f32;
    (ratio * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_has_no_edits() {
        assert_eq!(word_edit_distance("ship it today", "ship it today"), 0);
        assert_eq!(edit_ratio("ship it today", "ship it today"), 0.0);
    }

    #[test]
    fn whitespace_differences_do_not_count() {
        assert_eq!(word_edit_distance("ship  it\ttoday ", " ship it today"), 0);
    }

    #[test]
    fn counts_substitution_insertion_and_deletion() {
        assert_eq!(word_edit_distance("the cat sat", "the dog sat"), 1);
        assert_eq!(word_edit_distance("the cat sat", "the cat sat down"), 1);
        assert_eq!(word_edit_distance("the cat sat down", "the cat sat"), 1);
        assert_eq!(word_edit_distance("a b c d", "a x c"), 2);
    }

    #[test]
    fn case_and_punctuation_are_edits() {
        assert_eq!(word_edit_distance("deploy to kubernetes", "deploy to Kubernetes"), 1);
        assert_eq!(word_edit_distance("hello world", "hello world."), 1);
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(word_edit_distance("", ""), 0);
        assert_eq!(word_edit_distance("", "a b"), 2);
        assert_eq!(word_edit_distance("a b", ""), 2);
        assert_eq!(edit_ratio("", ""), 0.0);
        assert_eq!(edit_ratio("", "a b"), 1.0);
    }

    #[test]
    fn ratio_is_edits_over_longer_side_and_never_above_one() {
        // "cooper netties" -> "Kubernetes" is one substitution plus one deletion
        assert_eq!(
            edit_ratio("we use cooper netties for the cluster", "we use Kubernetes for the cluster"),
            0.2857 // 2 edits / 7 words
        );
        assert_eq!(edit_ratio("a b c d e f", "a b c d e g"), 0.1667);
        assert_eq!(edit_ratio("one two", "three four five six"), 1.0);
        assert!(edit_ratio("x", "y z w") <= 1.0);
    }
}
