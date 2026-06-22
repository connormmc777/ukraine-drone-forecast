#!/usr/bin/env bash
# Autostart loop — keeps the data fresh whenever the user is logged in.
# Runs cron_sync.sh every 2 hours in the user's interactive Python
# environment (avoiding Bazzite's systemd-vs-shell Python version split).
#
# Install: KDE Autostart → add ~/Desktop/dronespredictions/drone_app/autostart_sync.sh
# Or run manually in a terminal that stays open.

set -uo pipefail
cd "$(dirname "$0")"

LOGFILE=/tmp/drone_sync.log
SLEEP_SECONDS=7200   # 2 hours

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] autostart_sync.sh starting — every ${SLEEP_SECONDS}s"

# Run immediately on launch, then every 2 hours
while true; do
    ./cron_sync.sh >> "$LOGFILE" 2>&1 \
        && echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] sync OK" \
        || echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] sync FAILED — will retry in ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
done
