#!/bin/bash
set -euo pipefail

# Adaptive Screamer: minimal push-to-talk dictation build.
# Bundles only the Whisper model(s). No summary/vision/TTS models or helpers.

APP="${APP:-Screamer.app}"
CONTENTS="$APP/Contents"
INFO_PLIST_TEMPLATE="${INFO_PLIST_TEMPLATE:-resources/Info.plist}"
INFO_PLIST="$CONTENTS/Info.plist"
MODELS_DIR="${MODELS_DIR:-models}"
BIN_PATH="${BIN_PATH:-target/release/screamer}"
PLIST_BUDDY="${PLIST_BUDDY:-/usr/libexec/PlistBuddy}"
SYSTEM_CODESIGN="${SYSTEM_CODESIGN:-/usr/bin/codesign}"
SYSTEM_SECURITY="${SYSTEM_SECURITY:-/usr/bin/security}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:-}"
# A stable self-signed identity makes macOS remember the Accessibility grant
# across rebuilds. Create it once with ./setup_signing.sh.
SELF_SIGNED_IDENTITY="${SELF_SIGNED_IDENTITY:-Screamer Local Signing}"
DEFAULT_ENTITLEMENTS="resources/Screamer.entitlements"
CODESIGN_ENTITLEMENTS="${CODESIGN_ENTITLEMENTS:-$DEFAULT_ENTITLEMENTS}"
APP_VERSION="${APP_VERSION:-$(awk -F '\"' '/^version = / { print $2; exit }' Cargo.toml)}"
SKIP_BUILD="${SKIP_BUILD:-0}"

# The default Whisper model is required; the others are bundled only if present.
REQUIRED_MODELS=(
    "ggml-base.en.bin"
)
OPTIONAL_MODELS=(
    "ggml-tiny.en.bin"
    "ggml-small.en.bin"
    "ggml-accessibility.en.bin"
)

IS_DEVELOPER_ID=0

detect_codesign_identity() {
    if [ -n "$CODESIGN_IDENTITY" ]; then
        echo "$CODESIGN_IDENTITY"
        return
    fi

    if [ ! -x "$SYSTEM_SECURITY" ]; then
        echo "-"
        return
    fi

    local dev_id
    dev_id="$("$SYSTEM_SECURITY" find-identity -v -p codesigning 2>/dev/null \
        | awk -F '"' '/Developer ID Application:/ { print $2; exit }')"
    if [ -n "$dev_id" ]; then
        echo "$dev_id"
        return
    fi

    # Fall back to the stable self-signed identity if it exists. No -v here: a
    # self-signed cert isn't trusted, so it won't appear in the "valid" list,
    # but codesign can still sign with it. Match by SHA-1 hash rather than name
    # (the name can be ambiguous if more than one such cert exists), and pick the
    # lowest hash deterministically so the SAME cert is used on every rebuild —
    # otherwise the app's code identity would flip between builds and macOS would
    # keep re-asking for Accessibility.
    local self_hash
    self_hash="$("$SYSTEM_SECURITY" find-identity -p codesigning 2>/dev/null \
        | awk -v name="$SELF_SIGNED_IDENTITY" 'index($0, name) { print $2 }' \
        | sort | head -n 1)"
    if [ -n "$self_hash" ]; then
        echo "$self_hash"
        return
    fi

    echo "-"
}

CODESIGN_IDENTITY="$(detect_codesign_identity)"
case "$CODESIGN_IDENTITY" in
    "Developer ID Application:"*) IS_DEVELOPER_ID=1 ;;
esac

sign_target() {
    local target="$1"
    local sign_args=(--force --sign "$CODESIGN_IDENTITY")

    if [ "$CODESIGN_IDENTITY" != "-" ]; then
        if [ -n "$CODESIGN_ENTITLEMENTS" ] && [ -f "$CODESIGN_ENTITLEMENTS" ]; then
            sign_args+=(--entitlements "$CODESIGN_ENTITLEMENTS")
        fi
        # Hardened runtime + secure timestamp only matter for notarized
        # Developer ID builds; a local self-signed identity signs plainly.
        if [ "$IS_DEVELOPER_ID" = "1" ]; then
            sign_args+=(--timestamp --options runtime)
        fi
    fi

    if "$SYSTEM_CODESIGN" "${sign_args[@]}" "$target"; then
        return 0
    fi

    if [ "$CODESIGN_IDENTITY" = "-" ]; then
        return 1
    fi

    echo "Warning: signing with '$CODESIGN_IDENTITY' failed; falling back to ad-hoc signing." >&2
    CODESIGN_IDENTITY="-"
    "$SYSTEM_CODESIGN" --force --sign - "$target"
}

echo "=== Building Adaptive Screamer (minimal dictation) ==="
echo "App version: $APP_VERSION"
if [ "$CODESIGN_IDENTITY" = "-" ]; then
    echo "Signing identity: ad-hoc (Accessibility grant will NOT persist across rebuilds —"
    echo "                  run ./setup_signing.sh once to fix this)"
else
    echo "Signing identity: $CODESIGN_IDENTITY"
fi

# Step 1: Build release binary unless a prebuilt one was provided.
if [ "$SKIP_BUILD" = "1" ]; then
    echo "Skipping cargo build and using prebuilt binary at $BIN_PATH"
else
    echo "Compiling release binary..."
    source "$HOME/.cargo/env" 2>/dev/null || true
    cargo build --release
fi

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: binary not found at $BIN_PATH"
    exit 1
fi

# Step 2: Assemble .app bundle.
echo "Assembling app bundle..."
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS"
mkdir -p "$CONTENTS/Resources/models"

cp "$BIN_PATH" "$CONTENTS/MacOS/Screamer"
cp "$INFO_PLIST_TEMPLATE" "$INFO_PLIST"
cp resources/icon.icns "$CONTENTS/Resources/"
cp resources/image.png "$CONTENTS/Resources/"
cp resources/menubarTemplate.png "$CONTENTS/Resources/"
cp resources/menubarTemplate@2x.png "$CONTENTS/Resources/"

if [ -x "$PLIST_BUDDY" ]; then
    "$PLIST_BUDDY" -c "Set :CFBundleShortVersionString $APP_VERSION" "$INFO_PLIST"
    "$PLIST_BUDDY" -c "Set :CFBundleVersion $APP_VERSION" "$INFO_PLIST"
fi

# Step 3: Copy the Whisper model(s).
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: models directory not found at $MODELS_DIR"
    echo "Run ./download_model.sh base first."
    exit 1
fi

missing_models=()
for model_name in "${REQUIRED_MODELS[@]}"; do
    model_path="$MODELS_DIR/$model_name"
    if [ ! -f "$model_path" ] || [ ! -s "$model_path" ]; then
        missing_models+=("$model_name")
        continue
    fi
    echo "Bundling model: $model_name"
    cp "$model_path" "$CONTENTS/Resources/models/"
done

if [ "${#missing_models[@]}" -ne 0 ]; then
    echo "Error: missing required Whisper model(s):"
    printf '  - %s\n' "${missing_models[@]}"
    echo "Run ./download_model.sh base and try again."
    exit 1
fi

for model_name in "${OPTIONAL_MODELS[@]}"; do
    model_path="$MODELS_DIR/$model_name"
    if [ -f "$model_path" ] && [ -s "$model_path" ]; then
        echo "Bundling model: $model_name"
        cp "$model_path" "$CONTENTS/Resources/models/"
    fi
done

if [ ! -x "$SYSTEM_CODESIGN" ]; then
    echo "Error: Apple codesign tool not found at $SYSTEM_CODESIGN"
    exit 1
fi

echo "Signing app bundle with Apple codesign..."
sign_target "$CONTENTS/MacOS/Screamer"
sign_target "$APP"
"$SYSTEM_CODESIGN" --verify --deep --strict --verbose=2 "$APP"

echo "=== Done: $APP ==="
echo "To run: open $APP"
