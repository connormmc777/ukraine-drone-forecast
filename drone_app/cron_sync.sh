#!/usr/bin/env bash
# Cron-driven sync of UA Air Force Telegram channel.
#
# Runs every 2 hours from crontab. Fetches ~20 pages of the public preview
# (= ~400 messages, well within the channel's published volume) and merges
# any new sightings/summaries/launch-site mentions into the local CSVs.
# This prevents the channel preview from rotating past unscraped messages,
# which is what cost us Jun 8-14 data.

set -euo pipefail

# Always run from the project root so relative paths in telegram_ingest work
cd "$(dirname "$0")"

# Use the venv's python explicitly. `source venv/bin/activate` doesn't work
# reliably in systemd's non-interactive shell context.
PYTHON="$(pwd)/venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found or not executable" >&2
    exit 1
fi

# Timestamp prefix for log lines
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
echo "[$(ts)] cron_sync.sh starting (python=$PYTHON)"

# Invoke the inner Python script as a real file (not stdin). This is the
# only invocation form that works reliably under systemd's user service
# on Bazzite — heredoc-piped Python sometimes loses sys.path to the venv.
"$PYTHON" "$(pwd)/cron_sync_inner.py"

# Bump file mtimes so any open Streamlit picks up the fresh data
touch data/observations.csv data/daily_totals.csv data/launch_site_log.csv 2>/dev/null || true

echo "[$(ts)] cron_sync.sh finished OK"
