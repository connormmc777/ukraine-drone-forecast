#!/usr/bin/env bash
# One-shot installer for dronespredictions on Bazzite/k3s.
#
# Does the whole build + import + helm install + host-process cleanup.
# Idempotent — re-running upgrades the release and re-imports a fresh image.
#
# Prereqs (all already present on this box):
#   - podman   (host: /usr/bin/podman)
#   - k3s      (host: /usr/local/bin/k3s + systemd unit)
#   - helm     (host: /usr/bin/helm)
#   - kubectl  (host: /usr/bin/kubectl)
#
# Reads the Cloudflare tunnel token from deploy/my-secrets.yaml so it never
# ends up in shell history, argv, or this file.

set -euo pipefail

# --- resolve repo root regardless of where this script was invoked from ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SECRETS_FILE="$ROOT/deploy/my-secrets.yaml"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="dronespredictions:${IMAGE_TAG}"
TAR="/tmp/dronespredictions-${IMAGE_TAG}.tar"

# --- token extraction (never echoed) ---
if [ ! -f "$SECRETS_FILE" ]; then
    echo "ERROR: $SECRETS_FILE not found." >&2
    echo "Create it from deploy/my-secrets.example.yaml first." >&2
    exit 1
fi
CF_TUNNEL_TOKEN=$(grep -E '^\s*token:' "$SECRETS_FILE" \
    | sed -E 's/^\s*token:\s*"?([^"]+)"?\s*$/\1/')
if [ -z "$CF_TUNNEL_TOKEN" ] || [ "$CF_TUNNEL_TOKEN" = "PASTE_CLOUDFLARE_TUNNEL_TOKEN_HERE" ]; then
    echo "ERROR: no valid Cloudflare tunnel token in $SECRETS_FILE" >&2
    exit 1
fi

echo "==> [1/6] Building $IMAGE with podman"
podman build -t "$IMAGE" "$ROOT"

echo "==> [2/6] Saving image tarball to $TAR"
rm -f "$TAR"
podman save -o "$TAR" "$IMAGE"

echo "==> [3/6] Importing image into k3s containerd k8s.io namespace (needs sudo)"
# Must be -n k8s.io: the kubelet looks in that containerd namespace. Importing
# to the default namespace silently succeeds but leaves the pod ErrImageNeverPull.
sudo k3s ctr -n k8s.io images import "$TAR"
rm -f "$TAR"

echo "==> [4/6] Killing host Streamlit + host cloudflared (moving into pod)"
# The pre-k3s bare-process deployment is now redundant — the k3s pod runs
# the same app + cloudflared inside. If we leave a host connector alive,
# Cloudflare load-balances between BOTH connectors and half of requests
# hit the empty host localhost:8501 → intermittent 502s.
#
# Match on the user-local install path, not the tunnel ID: the tunnel ID
# is embedded inside a base64 token so grep on the raw cmdline misses it.
pkill -u "$USER" -f 'streamlit run app.py'         2>/dev/null || true
pkill -u "$USER" -f '\.local/bin/cloudflared'      2>/dev/null || true
pkill -u "$USER" -f 'cloudflared.*tunnel_start'    2>/dev/null || true

# Remove KDE autostart entries — k3s manages the app lifecycle now.
rm -f "$HOME/.config/autostart/drone-streamlit.desktop"
rm -f "$HOME/.config/autostart/drone-tunnel.desktop"
# Note: drone-sync.desktop stays disabled by the same removal only if you
# want the k3s sync sidecar to be authoritative. We keep it here so both
# are running is not a problem (SQLite writes are idempotent), but if you
# see doubled Telegram fetches, remove drone-sync.desktop too.

echo "==> [5/6] helm upgrade --install dronespredictions"
helm upgrade --install dronespredictions "$ROOT/charts/dronespredictions" \
    -f "$ROOT/deploy/bazzite/values.yaml" \
    --set-string cloudflared.token="$CF_TUNNEL_TOKEN" \
    --wait --timeout 5m

echo "==> [6/6] Verify"
kubectl get pods -l app.kubernetes.io/name=dronespredictions
echo
echo "Waiting 10s then curling https://dronespredictions.net ..."
sleep 10
curl -sI --max-time 15 https://dronespredictions.net | head -3 || true

echo
echo "Done. If you see HTTP/2 200, the pod is live and the tunnel is up."
echo "To watch logs:"
echo "  kubectl logs -l app.kubernetes.io/name=dronespredictions -c app -f"
echo "  kubectl logs -l app.kubernetes.io/name=dronespredictions -c cloudflared -f"
