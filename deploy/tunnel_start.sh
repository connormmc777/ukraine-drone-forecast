#!/usr/bin/env bash
# Foreground the Cloudflare tunnel. Runs as long as your session is up.
# Reads the token from deploy/my-secrets.yaml so it's never in the shell
# history or the script itself.

set -euo pipefail

# Find the token in my-secrets.yaml
SECRETS_FILE="$(dirname "$0")/my-secrets.yaml"
if [ ! -f "$SECRETS_FILE" ]; then
    echo "ERROR: $SECRETS_FILE not found" >&2
    exit 1
fi

TOKEN=$(grep -E '^\s*token:' "$SECRETS_FILE" \
        | sed -E 's/^\s*token:\s*"?([^"]+)"?\s*$/\1/')

if [ -z "$TOKEN" ] || [ "$TOKEN" = "PASTE_CLOUDFLARE_TUNNEL_TOKEN_HERE" ]; then
    echo "ERROR: token not set in $SECRETS_FILE" >&2
    exit 1
fi

exec "${HOME}/.local/bin/cloudflared" tunnel --no-autoupdate run --token "$TOKEN"
