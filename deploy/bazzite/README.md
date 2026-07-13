# dronespredictions on Bazzite (k3s + cloudflared)

Same shape as MarketBrawl: single pod with **app + sync-loop + cloudflared**
containers, ingress via Cloudflare Tunnel, no open ports on the host.

## Prereqs (already present on this box)

- Bazzite (immutable Fedora atomic)
- `k3s` at `/usr/local/bin/k3s` with its systemd service
- `helm`, `kubectl`, `podman` layered via `rpm-ostree`
- `deploy/my-secrets.yaml` with a valid Cloudflare tunnel token

## Install

```sh
bash deploy/bazzite/install.sh
```

That single command:

1. `podman build` — builds `dronespredictions:latest` from the Dockerfile
2. `podman save` — dumps it to `/tmp/dronespredictions-latest.tar`
3. `sudo k3s ctr images import` — makes k3s aware of the local image (no registry)
4. Kills the host-native Streamlit + cloudflared processes and removes the
   KDE autostart entries — k3s owns the lifecycle from here
5. `helm upgrade --install dronespredictions charts/dronespredictions` with
   `-f deploy/bazzite/values.yaml` and the tunnel token from `my-secrets.yaml`
6. Curls the site to confirm HTTP/2 200

## Update the app after code changes

Same one command — the install script is idempotent:

```sh
bash deploy/bazzite/install.sh
```

Each run rebuilds the image, re-imports, and does `helm upgrade`. k3s
rolls the pod (Recreate strategy, single replica).

## Survives

| Event | Behavior |
|---|---|
| Reboot | k3s systemd unit brings the pod back automatically |
| Laptop sleep | Tunnel disconnects; Cloudflare shows offline page. On wake, cloudflared reconnects within seconds |
| App crash | Pod restarts (default Deployment behavior) |
| cloudflared crash | Sidecar restarts, tunnel comes back |

## Common operations

```sh
# Watch pods
kubectl get pods -w

# Tail app logs
kubectl logs -l app.kubernetes.io/name=dronespredictions -c app -f

# Tail tunnel logs
kubectl logs -l app.kubernetes.io/name=dronespredictions -c cloudflared -f

# Shell into the app container
kubectl exec -it deploy/dronespredictions -c app -- /bin/sh

# Rotate the tunnel token: update deploy/my-secrets.yaml, then:
bash deploy/bazzite/install.sh

# Wipe everything and start over
helm uninstall dronespredictions
kubectl delete pvc -l app.kubernetes.io/name=dronespredictions
```

## Caveats

- **NetworkPolicy off** — k3s' default flannel CNI doesn't enforce policies.
  Same tradeoff as MarketBrawl. Install Calico if you want real enforcement.
- **Local image only** — nothing pushes to a registry. If you deploy this to
  a second machine you'll need to either scp the tarball or add a GHCR flow
  (change `image.repository`/`tag`/`pullPolicy` in `deploy/bazzite/values.yaml`).
