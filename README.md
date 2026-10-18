<div align="center">

<img src="resources/image.png" width="220" alt="Screamer">

# Adaptive Screamer

Adaptive Screamer is a fork of [suvamsh/screamer](https://github.com/suvamsh/screamer) (MIT). It adds opt-in dictation telemetry so a Whisper model can be fine-tuned on my own voice and vocabulary.

**Offline push-to-talk dictation for macOS.**

Hold a key, speak, release, and your text is pasted instantly.

[![Built with Rust](https://img.shields.io/badge/Built_with-Rust-B7410E?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Metal GPU](https://img.shields.io/badge/Metal-GPU_Accelerated-0071E3?style=for-the-badge&logo=apple&logoColor=white)](#performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![100% Offline](https://img.shields.io/badge/100%25-Offline-8B5CF6?style=for-the-badge&logo=shieldsdotio&logoColor=white)](#)

</div>

## What is it?

Adaptive Screamer is a deliberately minimal fork of Screamer: just **offline push-to-talk dictation**, plus **opt-in local telemetry** for fine-tuning Whisper on your own voice.

The upstream project's ambient notetaker, Gemma summaries, screen-vision, and text-to-speech have all been removed, along with their models and dependencies. What remains is the dictation core and the telemetry hooks.

## How it works

```text
Hold Left Control -> Speak -> See waveform + live text -> Release -> Text pastes instantly
```

- Local Whisper transcription with no cloud round-trip
- Live overlay with waveform and rolling preview while the hotkey is held
- Final transcription and paste on release, so unstable partials are never typed into the target app
- A menu-bar icon for choosing the model, hotkey, overlay position, and toggling live preview / sound

## Install

Requirements:

- macOS 13+ on Apple Silicon or Intel
- [Rust toolchain](https://rustup.rs/) 1.94+
- `cmake` via `brew install cmake`

Build from source:

```bash
git clone https://github.com/ankittandon/adaptive-screamer.git
cd adaptive-screamer
./download_model.sh base          # or: tiny | small | large
./setup_signing.sh                # one-time: see "Signing" below
GGML_NATIVE=OFF cargo build --release
./bundle.sh
open Screamer.app
```

After first launch, grant **Accessibility** in `System Settings → Privacy & Security → Accessibility → Screamer` (required for the global hotkey and paste). macOS also prompts for **Microphone** the first time you record. On the very first launch Gatekeeper may block the app because it is self-signed — right-click it in Finder → **Open**, or use **Open Anyway** in Privacy & Security.

### Signing

`./setup_signing.sh` creates a stable, self-signed code-signing identity in your login keychain (once). `bundle.sh` then signs with it automatically. The stable identity is what lets macOS **remember the Accessibility grant across rebuilds** — with plain ad-hoc signing the app's identity changes every build and macOS re-prompts each time. The identity is local and not notarized, so Gatekeeper still needs a one-time "Open Anyway". If an actual `Developer ID Application` certificate is installed, `bundle.sh` prefers that instead.

If you ever change signing identity and macOS still shows the app as untrusted, reset its stale permission entry:

```bash
tccutil reset Accessibility com.screamer.app
```

## Configuration

Config lives at `~/Library/Application Support/Screamer/config.json`:

```json
{
  "model": "base",
  "hotkey": "left_control",
  "overlay_position": "center",
  "appearance": "dark",
  "live_transcription": true,
  "sound_effects": true,
  "telemetry_enabled": false,
  "fix_hotkey": ""
}
```

Key settings:

- `model`: Whisper model to use (`tiny`, `base`, `small`)
- `hotkey`: push-to-talk key (`left_control`, `right_control`, `left_option`, `right_option`, `left_command`, `right_command`, `fn`)
- `overlay_position`: overlay placement (`center`, `top`, `bottom`)
- `appearance`: app theme (`dark`, `light`)
- `live_transcription`: live preview in the overlay
- `sound_effects`: start and finish cue sounds
- `telemetry_enabled`: opt-in, default `false`. Save each dictation locally (see [Dictation telemetry](#dictation-telemetry-opt-in))
- `fix_hotkey`: optional tap key for attaching a correction to the last dictation. Empty (default) disables it; otherwise any key id other than your dictation key, e.g. `right_command`

## Performance

Accuracy mainly follows the underlying Whisper model, so treat these as relative tradeoffs:

| Model | Tradeoff | Best for |
|---|---|---|
| `tiny.en` | Fastest, lowest accuracy | Maximum speed |
| `base.en` | Balanced default | **Best default for most people** |
| `small.en` | Slower, more accurate | Harder vocabulary |

Under the hood: two pre-warmed Whisper states (final + live preview) stay in memory; the decode path is tuned for short push-to-talk utterances (no timestamps, single segment, greedy decode); silence is trimmed before inference; and only the final transcript is ever pasted.

## Dictation telemetry (opt-in)

Off by default. Set `"telemetry_enabled": true` in `config.json` and restart Screamer, and every dictation is saved to `~/Library/Application Support/Screamer/telemetry/`:

- `clips/<id>.wav`: the 16 kHz mono audio that was sent to the model
- `records.jsonl`: an append-only log, one JSON object per line

Nothing is uploaded. The folder and files are readable by your user only. It is a recording of everything you dictate, so treat it as sensitive.

Each dictation record has `id`, `timestamp`, `audio_path` (relative to the telemetry folder), `duration_ms`, `model`, `raw_text` (what the model produced), `pasted_text` and `app_context` (the frontmost app's bundle id, never a window title).

**Recording corrections.** Set `"fix_hotkey"` to a modifier key you don't use for dictation, for example `"right_command"`. After you fix a pasted dictation by hand, select the corrected passage and tap that key. Screamer copies the selection, puts your previous clipboard text back (clipboard images are not preserved), and attaches the selection to your most recent dictation as `settled_text`, with `label_source: "fix"` and an `edit_ratio` (share of words changed). You hear the completion sound when it is saved. A correction is refused if nothing is selected, if the dictation is more than 10 minutes old, or if the selection differs too much from the transcript to be the same sentence.

**Export** the corrected pairs as JSON lines for fine-tuning:

```bash
cargo run --release -p screamer-telemetry --bin screamer-telemetry-export -- \
  "$HOME/Library/Application Support/Screamer/telemetry" > labeled.jsonl
```

Add `--all` to include dictations you never corrected.

## Privacy and logging

- Transcription runs locally. Screamer does not send audio or text to a cloud service.
- Runtime logs are written to `~/Library/Logs/Screamer/screamer.log` by default.
- Dictation transcript contents are not logged by default. Set `SCREAMER_LOG_TRANSCRIPTS=1` to include them when debugging.
- Set `SCREAMER_LOG_FILE=/custom/path.log` to override the log file location.
- The opt-in telemetry directory (above) is the one place dictation audio/text is persisted, and only when you enable it.

## Stack

- Rust app, single native binary
- `whisper.cpp` via `whisper-rs`
- CoreAudio capture
- Metal GPU acceleration on Apple Silicon

## License

MIT. Fork of [suvamsh/screamer](https://github.com/suvamsh/screamer); see [LICENSE](LICENSE).
