#!/bin/bash
set -euo pipefail

# Create a stable, self-signed code-signing identity in your login keychain so
# that macOS remembers Screamer's Accessibility (and Microphone) permission
# across rebuilds. Ad-hoc signing changes the app's code identity every build,
# which is why macOS keeps re-prompting; a stable identity fixes that.
#
# Run this ONCE:   ./setup_signing.sh
# Then build:      ./bundle.sh        (it auto-detects this identity)
#
# This creates a LOCAL, self-signed certificate. It is not from Apple and is not
# notarized — Gatekeeper will still ask you to "Open Anyway" the first time, but
# the Accessibility/Microphone grants will persist.

IDENTITY_NAME="${SELF_SIGNED_IDENTITY:-Screamer Local Signing}"
KEYCHAIN="${KEYCHAIN:-$HOME/Library/Keychains/login.keychain-db}"
SECURITY="/usr/bin/security"
OPENSSL="/usr/bin/openssl"

if "$SECURITY" find-identity -v -p codesigning 2>/dev/null | grep -qF "$IDENTITY_NAME"; then
    echo "Signing identity already present: '$IDENTITY_NAME'"
    echo "Nothing to do. Build with ./bundle.sh"
    exit 0
fi

echo "Creating self-signed code-signing identity: '$IDENTITY_NAME'"

WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

CONFIG="$WORKDIR/req.cnf"
cat > "$CONFIG" <<EOF
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = $IDENTITY_NAME
[v3]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
EOF

"$OPENSSL" req -x509 -newkey rsa:2048 -nodes \
    -keyout "$WORKDIR/key.pem" \
    -out "$WORKDIR/cert.pem" \
    -days 3650 \
    -config "$CONFIG" >/dev/null 2>&1

"$OPENSSL" pkcs12 -export \
    -inkey "$WORKDIR/key.pem" \
    -in "$WORKDIR/cert.pem" \
    -name "$IDENTITY_NAME" \
    -out "$WORKDIR/identity.p12" \
    -passout pass: >/dev/null 2>&1

# Import the key+cert, granting /usr/bin/codesign access to the private key.
"$SECURITY" import "$WORKDIR/identity.p12" \
    -k "$KEYCHAIN" \
    -P "" \
    -T /usr/bin/codesign \
    -A

# Try to pre-authorize codesign so it won't pop a keychain dialog mid-build.
# This needs your login (keychain) password; pass it as KEYCHAIN_PASSWORD to
# avoid the prompt, otherwise you'll just click "Always Allow" the first build.
if [ -n "${KEYCHAIN_PASSWORD:-}" ]; then
    "$SECURITY" set-key-partition-list \
        -S apple-tool:,apple: \
        -s \
        -k "$KEYCHAIN_PASSWORD" \
        "$KEYCHAIN" >/dev/null 2>&1 \
        && echo "Pre-authorized codesign to use the key (no dialog expected)." \
        || echo "Note: could not pre-authorize; codesign may ask once (click Always Allow)."
else
    echo "Note: on the first ./bundle.sh, macOS may ask to let codesign use the"
    echo "      new key — click \"Always Allow\". (Or re-run with KEYCHAIN_PASSWORD set"
    echo "      to skip that prompt.)"
fi

echo ""
if "$SECURITY" find-identity -v -p codesigning 2>/dev/null | grep -qF "$IDENTITY_NAME"; then
    echo "Done. Identity '$IDENTITY_NAME' is ready."
    echo "Now run ./bundle.sh — it will sign with this identity automatically."
else
    echo "Warning: identity not found after import. Check Keychain Access for '$IDENTITY_NAME'."
    exit 1
fi
