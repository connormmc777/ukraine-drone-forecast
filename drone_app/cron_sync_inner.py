"""Inner Python payload for cron_sync.sh — kept in its own file so the
venv's import resolution gets a real script path (not stdin) and works
identically under systemd as it does interactively."""
import json
import os
import sys
import time
from datetime import datetime

# Ensure CWD = the project directory so relative paths work in
# telegram_ingest (data/observations.csv etc.)
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

import telegram_ingest as ti


def main() -> int:
    last_err = None
    for attempt in range(1, 4):
        try:
            result = ti.sync('data/observations.csv',
                             'data/daily_totals.csv', pages=20)
            result['at'] = datetime.now().isoformat(timespec='seconds')
            result['attempts'] = attempt
            with open('data/.tg_sync_log.json', 'w') as f:
                json.dump(result, f, default=str)
            print(f"  attempt {attempt} OK")
            print(f"  sightings:    +{result['rows_added']} new, "
                  f"{result['rows_updated']} updated")
            print(f"  summaries:    +{result['summary_rows_added']} new, "
                  f"{result['summary_rows_updated']} updated")
            print(f"  launch sites: +{result.get('launch_site_rows_added', 0)} new, "
                  f"{result.get('launch_site_rows_updated', 0)} updated")
            print(f"  date range:   {result['date_range']}")
            return 0
        except Exception as e:
            last_err = e
            wait = 15 * attempt
            print(f"  attempt {attempt} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            if attempt < 3:
                print(f"  retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
    print(f"FAILED after 3 attempts: {last_err}", file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
