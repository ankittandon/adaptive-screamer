# Styling layer — build spec

Status: draft · Owner: Ankit · Last updated: 2026-10-18

A post-dictation styling layer for Adaptive Screamer: after you dictate, an
optional, non-blocking suggestion offers a cleaned-up ("styled") rewrite of
what you said. Raw text always lands first; styling is additive and never slows
or degrades dictation.

This builds directly on the existing dictation + telemetry code. It does **not**
reintroduce the ambient notetaker, summaries, vision, or TTS.

---

## 1. Goals and principles

- **Raw is sacred.** The ~55 ms release-to-paste path is untouched. Styling is
  always async and additive; if it fails or is slow, you still have your raw
  text.
- **Opt-in and local-first**, consistent with the telemetry work.
- **Meaning fidelity is a hard constraint**, not a tradeoff. A rewrite that
  changes intent is a bug and must be suppressed, never shown.
- **The interaction is also the data.** Every accept / dismiss / edit is a
  preference label for training and evaluating the styling model (see §9).

Non-goals (for v1): true in-app inline underlines, an input-method/IME
integration, per-app plugins, multi-message/document styling. These are noted in
§11 as later options.

---

## 2. End-to-end flow

```
Hold Left Control → speak → release
  → raw transcript pastes instantly        (unchanged dictation path)
  → triviality pre-gate on raw text        (rules, microseconds)
      ├─ trivial  → stop (no suggestion)
      └─ non-trivial → styling generator    (async; local or cloud — §6)
            → edit_ratio post-gate
                ├─ change below threshold → stop
                └─ meaningful + meaning-preserving
                      → floating card appears (persists; §7)
                      → dictated text pre-selected, styled on clipboard (§8)
                          ├─ Cmd+V → overwrite (accept)   → log "styled preferred"
                          ├─ edit after accept            → log "gold"
                          └─ Esc / timeout → dismiss, restore clipboard, deselect
                                                            → log "raw preferred"
```

---

## 3. Components

New / changed, with the existing files they build on:

| Area | File | Change |
|---|---|---|
| Triviality gate | `src/styling/gate.rs` (new) | Rules-based classifier on raw text |
| Styling generator | `src/styling/generator.rs` (new) | Trait + impl (local or cloud — §6) |
| Suggestion overlay | `src/styling_overlay.rs` (new) | Floating `NSPanel`, own lifecycle (patterns from `src/overlay.rs`) |
| Apply mechanism | `src/paster.rs` (extend) | Select-all / select-back + clipboard set/restore + Cmd+V |
| Field check | `src/axtext.rs` (reuse) | Confirm field ≈ raw before overwrite |
| Orchestration | `src/app.rs` (extend) | Kick off gate→generate→overlay after the dictation paste |
| Hotkeys | `src/hotkey.rs` (extend) | Observe Cmd+V (accept) and Esc (dismiss) while a card is open |
| Config | `src/config.rs` (extend) | New opt-in settings (§10) |
| Telemetry | `crates/screamer-telemetry` (extend) | New style events (§9) |

---

## 4. Triviality gate (rules, no model)

Runs on the raw transcript to decide whether styling is worth invoking at all.
Cheap heuristics, tuned to suppress noise:

- **Filler density** — count of "um, uh, like, kinda, sort of, i guess, you
  know, basically, yeah so" over total words, above a threshold.
- **Run-on** — long span (> N words) with no sentence-ending punctuation.
- **Casing** — leading-lowercase / no capitalization where a sentence is implied.
- **Disfluency** — repeated adjacent words, false starts.
- **Length floor** — skip very short utterances ("ok", "thanks").

If none trip, skip entirely. A second gate (`edit_ratio`, already in
`screamer-telemetry`) runs *after* generation to suppress rewrites that barely
changed anything. Thresholds live in config so they're tunable without a rebuild.

---

## 5. Floating overlay (lifecycle)

Reuses the `NSPanel` approach in `src/overlay.rs` but as a **separate panel with
its own lifecycle**, independent of `is_recording` (the recording overlay hides
on key release; this one must not).

- **Appears** when a suggestion is ready (after release + async generation).
- **Persists** until accept, dismiss, or timeout (default ~12 s, configurable).
- **Auto-dismisses** on a new dictation.
- **Contents**: the styled text (changed spans highlighted), an "Apply ⌘V" hint,
  a "Keep raw · Esc" hint. Shows the result, not a modal.
- **Focus**: the panel is non-activating so it never steals focus from the
  target composer — critical, because apply depends on focus staying in the
  field. Primary interaction is the keyboard (Cmd+V / Esc), not a click.

---

## 6. Styling generator — OPEN DECISION

The one unresolved architectural fork. Interface is settled; the backend is not.

```rust
trait StyleGenerator {
    /// Rewrite `raw` in the user's style, preserving meaning. Returns None to
    /// decline (e.g., low confidence, meaning at risk).
    fn style(&self, raw: &str, ctx: &StyleContext) -> Option<StyleResult>;
}
```

Options:

- **Local small model** — re-add a slim `llama.cpp` (one quantized 1–3B,
  opt-in). Keeps the offline/private posture, and is where the eventually
  fine-tuned model lives. Heaviest dependency; adds build + binary weight back.
- **Cloud API** — fastest to prototype, best quality now, but sends your text
  off-device (breaks the offline guarantee).

Latency is async in both cases, so neither affects the raw-paste path. The
fine-tuning target is a local model regardless, so cloud (if chosen) is a
prototype stepping stone, not the destination.

**Meaning-fidelity guard** belongs here: the generator (or a cheap check after
it) must decline when the rewrite risks changing intent.

---

## 7. Apply mechanism

The confirmed model: **pre-select + clipboard + Cmd+V overwrite**, with a safety
check and clipboard restore.

1. When the card appears, read the composer via `axtext::focused_text()` and
   confirm it still ≈ the raw dictation (nothing else typed, focus unmoved).
   - If it doesn't match → **clipboard-only** mode (styled on clipboard, no
     pre-select, no auto-overwrite). Safe everywhere.
2. If it matches → **pre-select the dictated text**:
   - field ≈ dictation → `Cmd+A` (clean, whole-composer),
   - else → `Shift+Left ×N` to select just the dictated run (N known; note the
     grapheme caveat for emoji — prefer `Cmd+A` when possible).
   - Put the styled text on the clipboard (saving the user's current clipboard).
3. **Cmd+V** overwrites the selection. It's a real paste event, so Electron /
   Claude update their internal state correctly (this is why we do **not** use
   `AXUIElementSetAttributeValue` — that leaves React state stale).
4. **Esc / timeout** → dismiss, deselect, and **restore the original clipboard**
   (same save/restore pattern already in `paster::capture_selection`).

Reuses `paster.rs`'s `CGEvent` key-posting. New keycodes needed: A (0), and
arrow/shift combos for the select-back fallback.

---

## 8. Config additions (`src/config.rs`)

All opt-in, defaults off/conservative:

- `styling_enabled: bool` (default `false`)
- `styling_backend: "local" | "cloud"` (once §6 is decided)
- `styling_apply: "overwrite" | "clipboard"` (default `overwrite`; falls back to
  clipboard when the AX check fails)
- `styling_card_timeout_secs: u64` (default `12`)
- triviality thresholds (filler ratio, min words, run-on length)

Old configs ignore unknown keys (serde), as with the telemetry fields.

---

## 9. Telemetry / eval integration

This interaction generates the preference data the styling model trains and is
evaluated on. Extend `crates/screamer-telemetry` with style events, keyed to the
dictation id:

- `style_shown` — candidate generated (styled text, edit_ratio)
- `style_applied` — accepted (Cmd+V) → styled preferred over raw
- `style_dismissed` — kept raw → raw preferred (the suggestion missed)
- `style_edited` — user edited after applying → **gold** target
- (optional) `style_ab` — when two variants are shown, which was chosen
  (per-dimension calibration)

The occasional A/B popup discussed earlier is **subsumed** by this: the default
single-suggestion accept/dismiss is continuous preference signal; A/B becomes an
occasional calibration mode layered on top, not a separate feature.

Eval dimensions (meaning fidelity as a hard gate; voice/style match, concision,
tone fit as scored) are defined separately — see the eval spec (TODO).

---

## 10. Open questions / risks

- **Styling backend (§6)** — local vs cloud. Blocks generator implementation.
- **Apply reliability (the spike)** — does `Cmd+A`+`Cmd+V` replace *just* the
  composer in Claude, and does focus hold with a non-activating panel? Cheap to
  validate before investing.
- **Non-composer contexts** — in docs/editors the field ≫ the dictation, so the
  AX check fails and we stay clipboard-only. Capturing in-document corrections
  (span tracking) is out of scope for v1.
- **Electron AX quirks** — we already enable `AXManualAccessibility`; confirm it
  holds for the composer read used by the apply safety check.

---

## 11. Phased plan

- **Phase 0 — apply spike (½–1 day).** Throwaway: dictate → on a hotkey, select
  composer + `Cmd+V` a canned "styled" string. Validate overwrite + focus in
  Claude and 1–2 other apps. De-risks §7 before anything else is built.
- **Phase 1 — triviality gate + overlay shell.** Rules gate (§4) and the
  floating card (§5) wired after the dictation paste, showing a *stubbed* styled
  string. No real model yet. Proves the end-to-end UX.
- **Phase 2 — apply mechanism.** Implement §7 for real (pre-select, clipboard
  set/restore, Cmd+V, Esc, AX safety check).
- **Phase 3 — styling generator.** Implement the chosen backend (§6) behind the
  trait. Meaning-fidelity guard.
- **Phase 4 — style telemetry.** The style events (§9) feeding the training/eval
  loop.
- **Phase 5 — polish.** Highlight diff in the card, timeouts, config surface,
  A/B calibration mode.

Later / out of v1: input-method (IME) inline everywhere, per-app plugins,
hotkey-cycle and voice-directed styling, in-document span tracking.
