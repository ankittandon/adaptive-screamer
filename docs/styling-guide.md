# Styling guide

The writing principles the styling layer enforces (in the generator's system
prompt) and measures (in the slop detector, `crates/screamer-styling/src/slop.rs`).
The goal is to clean up dictation into natural written text **without** making it
sound like AI.

## Principles

1. **Preserve meaning.** Never add, remove, infer, or answer. A rewrite that
   changes intent is a failure, not a trade-off.
2. **Keep the speaker's voice.** Fix delivery (filler, false starts, run-ons,
   punctuation, casing), not personality. Don't formalize beyond the original.
3. **Don't sound like AI** — the anti-slop rules below.
4. **Don't flatten.** Vary sentence length and rhythm; de-slopping that turns
   prose into even, generic sentences has over-corrected.

## Anti-slop rules

Avoid introducing any of these (they are the "tells" of LLM writing):

- Em and en dashes — use commas, periods, or parentheses.
- "not X, but Y" / "it's not X, it's Y" constructions.
- Rule-of-three lists.
- A rhetorical question the writer then answers.
- Signpost phrases — "here's the thing," "at the end of the day," "it's worth
  noting," etc.
- Sentence fragments used for emphasis.
- Buzzwords — "delve," "intricate," "meticulous," "seamless," "leverage,"
  "robust," "tapestry," "realm," "underscore," "showcasing," and similar.
- Stitching clauses with ", which" and ", so."

The slop detector scores a weighted count of the detectable subset of these per
1,000 words (lower is more human).

## Basis: excess-vocabulary research

The anti-slop vocabulary list and the excess-frequency framing come from:

> Kobak, D., González-Márquez, R., Horvát, E.-Á., & Lause, J. (2025).
> *Delving into LLM-assisted writing in biomedical publications through excess
> vocabulary.* Science Advances, 11(27). arXiv:2406.07016.
> doi:10.1126/sciadv.adt3813.

What it did and what we take from it:

- **Method — "excess vocabulary."** They tracked word frequencies across **>15
  million PubMed abstracts (2010–2024)**, fit the pre-LLM trend, and flagged
  words whose frequency jumped abruptly after ChatGPT appeared — the "excess,"
  by direct analogy to excess-mortality studies. This is a *data-driven,
  model-free* way to find AI tells, and it's the same shape as our slop
  detector (count known markers) — and as the pre-LLM baseline calibration we
  may build later.
- **Finding — the tells are style words, not content words.** The excess words
  were overwhelmingly **style words (verbs and adjectives)** — "delve,"
  "intricate," "meticulous," and the like — rather than topical nouns. *Our
  takeaway:* target vocabulary/register markers, not subject matter. Our
  buzzword list is seeded from these.
- **Finding — the effect is large.** At least **13.5%** of 2024 abstracts showed
  LLM processing (a lower bound), reaching **~40%** in some subcorpora — an
  influence the authors note **exceeds** that of major events like the COVID
  pandemic on vocabulary. *Our takeaway:* the "polished writing" default these
  models reach for is pervasive and strong, which is exactly why styling needs
  explicit guardrails against it (and why our own measurements showed styling
  *adding* slop before the rules).
- **Caveat — the list is a moving target.** Follow-up work (e.g. human–LLM
  coevolution studies) found some flagged words — "delve" among them — dropped
  in frequency soon after being publicly called out. *Our takeaway:* treat the
  buzzword list as **living**, and prefer a **data-driven excess-vocabulary pass
  over our own corpus** (raw vs. styled vs. your edits) over a static hand list.

## Supporting references

- Muñoz-Ortiz et al. (2024), *Contrasting Linguistic Patterns in Human and
  LLM-Generated News Text*, arXiv:2308.09067 — stylometric differences (lexical
  diversity, sentence-length uniformity) beyond vocabulary.
- Mitchell et al. (2023), *DetectGPT*, arXiv:2301.11305; and GPTZero's
  **perplexity / burstiness** signals (survey: arXiv:2406.18259) — burstiness
  (sentence-length variance) and type-token ratio are the strongest, cheapest
  features to add to the slop detector next.
- Juzek & Ward (2024), *Why Does ChatGPT "Delve" So Much?*, arXiv:2412.11385 —
  on the RLHF/annotator origins of the overused words.
