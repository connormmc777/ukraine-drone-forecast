# Deploy the drone-predictions dashboard

Same pattern as MarketBrawl and FraudTracker: local Docker image → `kind`
Kubernetes cluster → Helm chart → Cloudflare Tunnel for public access.
No open ports on your machine, no cloud VMs, no AWS.

Two paths depending on how far you want to go:

- **Path A**: `docker compose up` — laptop-only, no domain, ~5 min
- **Path B**: kind + Helm + Cloudflare Tunnel — public URL, ~30 min

Both use the same Docker image.

---

## Prereqs

Install once (adapt for your OS — commands below are PowerShell for Windows,
same package names via `brew` on macOS or `dnf`/`apt` on Linux):

```powershell
winget install Docker.DockerDesktop
winget install Kubernetes.kind
winget install Kubernetes.kubectl
winget install Helm.Helm
```

Open Docker Desktop once so its VM starts. Verify:

```powershell
docker info
kind version
kubectl version --client
helm version
```

---

## Path A — Docker Compose (fastest local test)

```bash
# From the project root:
docker compose up --build
```

First build takes ~3 min (installs Python + geopandas + friends). Then:

- Dashboard at http://localhost:8501
- Data persists in a named Docker volume `drone-data`
- Sync loop runs every 2 hours automatically

To stop and wipe:

```bash
docker compose down -v    # -v deletes the data volume too
```

**When to use Path A**: you just want the dashboard running on your machine
for personal use, no public URL.

---

## Path B — Kubernetes + Cloudflare Tunnel (public URL, matches MarketBrawl)

### 1. Build the image

```powershell
docker build -t dronespredictions:dev .
```

### 2. Create the local cluster

```powershell
kind create cluster --config deploy/kind-config.yaml
kubectl cluster-info --context kind-dronespredictions
```

### 3. Load the image into the cluster

`kind` can't see Docker Desktop images until you push them in:

```powershell
kind load docker-image dronespredictions:dev --name dronespredictions
```

### 4. Set up Cloudflare Tunnel (one time)

Follow the walkthrough in your setup notes:
1. Add domain to Cloudflare DNS (Free plan).
2. Zero Trust → Networks → Tunnels → **Create a tunnel** → name it.
3. Copy the tunnel token (starts with `eyJhIjoi...`).
4. Under **Public Hostnames**: point `drones.yourdomain.com` → HTTP `drones-dronespredictions:8501`
   (adjust to match your Helm release name; here we're using release `drones`).

### 5. Fill in your secrets

```bash
cp deploy/my-secrets.example.yaml deploy/my-secrets.yaml
# Edit deploy/my-secrets.yaml — paste the tunnel token and your domain.
```

`deploy/my-secrets.yaml` is in `.gitignore` — never commit it.

### 6. Install the chart

```powershell
helm upgrade --install drones charts/dronespredictions -f deploy/my-secrets.yaml
```

Watch pods come up:

```powershell
kubectl get pods -w
```

Wait for `drones-dronespredictions-*` to go `Running` with 3 of 3 containers
ready (app + sync-loop + cloudflared).

Check the cloudflared connection:

```powershell
kubectl logs deploy/drones-dronespredictions -c cloudflared -f
```

Look for `Connection ... registered`. Cloudflare dashboard should show the
connector as **Healthy**.

### 7. Reach the app

Browse to whatever hostname you configured (e.g., `https://drones.yourdomain.com`).
TLS is provided by Cloudflare automatically.

---

## Upgrades

Rebuild the image, reload into kind, upgrade the chart:

```powershell
docker build -t dronespredictions:dev .
kind load docker-image dronespredictions:dev --name dronespredictions
helm upgrade drones charts/dronespredictions -f deploy/my-secrets.yaml
kubectl rollout restart deploy/drones-dronespredictions
```

## Uninstall

```powershell
helm uninstall drones
kind delete cluster --name dronespredictions
```

## Security posture

Same layer cake as MarketBrawl:

| Layer | What it blocks | Free? |
|---|---|---|
| Cloudflare WAF (managed ruleset) | SQL injection, XSS, known CVE patterns | ✅ |
| Cloudflare Bot Fight Mode | Drive-by bots, scrapers | ✅ |
| Cloudflare rate limiting | Brute-force patterns | ✅ (basic) |
| Cloudflare Access (optional) | Non-approved emails at the edge | ✅ (50 seats) |
| Tunnel itself | Zero inbound ports on your machine | ✅ |
| NetworkPolicy | Pod-to-pod traffic outside allowlist | ✅ |
| Nonroot Python container | Privilege escalation if compromised | ✅ |
| K8s resource limits | Runaway CPU/memory from a pod | ✅ |

The drone app is **read-only public data** — no auth needed. If you want to
gate access (e.g., only you can view it), turn on **Cloudflare Access** in
Zero Trust → Access → Applications and set an email allow-list.
