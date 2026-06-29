"""
Lock a weekly forecast snapshot using data-driven parameters.

Usage:
    python lock_weekly_snapshot.py [--week-start YYYY-MM-DD] [--force]

Defaults to next Monday (or today, if today is Monday). The script:
  1. Loads the daily-totals history.
  2. Computes a budget = 14-day rolling mean × 7 × 1.10 buffer.
  3. Builds a forecast with the recalibration step trained on the last
     21 days of observations.
  4. Inserts a row into snapshots + snapshot_rows for the target week.
  5. Skips if a snapshot already exists for that week (unless --force).

Designed to be run from cron once a week (e.g. Mondays at 06:00 Kyiv):
    0 6 * * 1  cd /path/to/drone_app && venv/bin/python lock_weekly_snapshot.py
"""
from __future__ import annotations
import argparse
import sqlite3
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import telegram_ingest

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "forecast_history.db"


def next_monday(today: date | None = None) -> date:
    today = today or date.today()
    # If today is Monday, lock for THIS week starting today; otherwise next Monday
    return today + timedelta(days=(7 - today.weekday()) % 7 or 0)


def data_driven_capacity(daily_totals_df: pd.DataFrame, window: int = 14,
                          buffer: float = 1.10) -> int:
    """Average daily launches over the last `window` days, with a buffer."""
    if daily_totals_df.empty:
        return 300  # sensible fallback
    daily_totals_df = daily_totals_df.copy()
    daily_totals_df['date'] = pd.to_datetime(daily_totals_df['date'])
    by_d = (daily_totals_df.groupby('date', as_index=False)['launched']
            .sum().sort_values('date'))
    tail = by_d.tail(window)
    if tail.empty:
        return 300
    return int(round(float(tail['launched'].mean()) * buffer))


def compute_targeting_weights(oblasts):
    oblasts = oblasts.copy()
    oblasts['weight'] = (
        0.35 * oblasts['energy']
        + 1.5 * np.exp(-oblasts['border_dist'] / 400)
        + 0.25 * oblasts['pop']
    )
    oblasts.loc[oblasts['border_dist'] > 700, 'weight'] *= 0.4
    oblasts['share'] = oblasts['weight'] / oblasts['weight'].sum()
    return oblasts


def recalibrate(oblasts, obs_df, K: int = 150):
    oblasts = oblasts.copy()
    if obs_df is None or obs_df.empty:
        oblasts['obs_share'] = 0.0
        oblasts['learned_share'] = oblasts['share']
        oblasts['alpha'] = 0.0
        oblasts['n_obs'] = 0
        return oblasts
    g = obs_df.groupby('oblast')['observed_drones'].sum().reset_index()
    g.columns = ['oblast', 'n_obs']
    oblasts = oblasts.merge(g, on='oblast', how='left')
    oblasts['n_obs'] = oblasts['n_obs'].fillna(0)
    total = oblasts['n_obs'].sum()
    oblasts['obs_share'] = oblasts['n_obs'] / total if total else 0.0
    alpha = total / (total + K)
    oblasts['alpha'] = alpha
    oblasts['learned_share'] = (1 - alpha) * oblasts['share'] + alpha * oblasts['obs_share']
    oblasts['learned_share'] /= oblasts['learned_share'].sum()
    return oblasts


def snapshot_exists(week_start: date) -> bool:
    if not DB_PATH.exists():
        return False
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM snapshots WHERE week_start = ? LIMIT 1",
            (week_start.isoformat(),),
        ).fetchone()
    return row is not None


def lock(week_start: date, capacity: int, tempo: float,
         oblasts_df: pd.DataFrame, observations_df: pd.DataFrame,
         note: str = "") -> int:
    weekly_budget = int(capacity * 7 * tempo)
    # Train on the last 21 days of observations (covers ~3 weeks of pattern)
    cutoff = pd.Timestamp(week_start) - pd.Timedelta(days=21)
    obs_train = observations_df[
        pd.to_datetime(observations_df['observation_date']) >= cutoff
    ]
    w = compute_targeting_weights(oblasts_df)
    r = recalibrate(w, obs_train)
    r['adj_share'] = r['learned_share']
    r['adj_share'] /= r['adj_share'].sum()
    r['predicted_week'] = (r['adj_share'] * weekly_budget).round(0)
    forecast = r.sort_values('predicted_week', ascending=False)

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO snapshots(
                created_at, week_start, russian_daily_capacity, tempo_factor,
                weekly_budget, remaining_budget, low_tempo, learning_alpha, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec='seconds'),
             week_start.isoformat(),
             int(capacity), float(tempo),
             weekly_budget, weekly_budget,  # nothing used yet for next week
             0, float(forecast['alpha'].iloc[0]),
             note or f"Auto-lock: data-driven {capacity}/d × 7 × {tempo} = {weekly_budget}"),
        )
        snap_id = cur.lastrowid
        rows = [(snap_id, row['oblast'],
                 float(row['share']), float(row['obs_share']),
                 float(row['learned_share']), float(row['predicted_week']))
                for _, row in forecast.iterrows()]
        conn.executemany(
            "INSERT INTO snapshot_rows(snapshot_id, oblast, prior_share, "
            "obs_share, learned_share, predicted_week) VALUES (?,?,?,?,?,?)",
            rows,
        )

    # Defense-in-depth: also write a JSON archive per snapshot so the
    # prediction survives any DB corruption / accidental wipe.
    # AND compute a SHA256 hash for tamper-evidence.
    import json
    import snapshot_integrity as _si
    per_oblast = [{
        'oblast': row['oblast'],
        'prior_share': float(row['share']),
        'obs_share': float(row['obs_share']),
        'learned_share': float(row['learned_share']),
        'predicted_week': float(row['predicted_week']),
    } for _, row in forecast.iterrows()]

    canonical = _si.canonical_snapshot_payload(
        week_start=week_start.isoformat(),
        russian_daily_capacity=int(capacity),
        tempo_factor=float(tempo),
        weekly_budget=weekly_budget,
        learning_alpha=float(forecast['alpha'].iloc[0]),
        per_oblast=per_oblast,
    )
    prediction_hash = _si.compute_hash(canonical)
    hash_computed_at = datetime.now().isoformat(timespec='seconds')

    # Persist hash to DB
    _si.ensure_hash_column(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE snapshots SET prediction_hash=?, hash_computed_at=? "
                     "WHERE id=?",
                     (prediction_hash, hash_computed_at, snap_id))

    archive_dir = DATA_DIR / 'snapshot_archive'
    archive_dir.mkdir(exist_ok=True)
    archive_payload = {
        'snapshot_id': snap_id,
        'created_at': hash_computed_at,
        'week_start': week_start.isoformat(),
        'russian_daily_capacity': int(capacity),
        'tempo_factor': float(tempo),
        'weekly_budget': weekly_budget,
        'learning_alpha': float(forecast['alpha'].iloc[0]),
        'note': note or '',
        'prediction_hash': prediction_hash,
        'hash_algo': 'sha256',
        'hash_computed_at': hash_computed_at,
        'per_oblast': per_oblast,
    }
    archive_path = archive_dir / f"snapshot_{snap_id:03d}_week_{week_start.isoformat()}.json"
    archive_path.write_text(json.dumps(archive_payload, indent=2))
    return snap_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--week-start', help='Monday date YYYY-MM-DD')
    ap.add_argument('--force', action='store_true',
                    help='Lock even if a snapshot exists for this week')
    ap.add_argument('--capacity', type=int, default=None,
                    help='Override daily capacity (default: data-driven)')
    ap.add_argument('--tempo', type=float, default=1.0)
    args = ap.parse_args()

    ws = (date.fromisoformat(args.week_start)
          if args.week_start else next_monday())
    if not args.force and snapshot_exists(ws):
        print(f"Snapshot already exists for week of {ws}. Use --force to override.")
        return 0

    oblasts = pd.read_csv(DATA_DIR / 'oblast_features.csv')
    raw_obs = pd.read_csv(DATA_DIR / 'observations.csv')
    daily_totals = (pd.read_csv(DATA_DIR / 'daily_totals.csv')
                    if (DATA_DIR / 'daily_totals.csv').exists()
                    else pd.DataFrame(columns=['date', 'launched']))
    observations = telegram_ingest.scale_observations_to_totals(raw_obs, daily_totals)

    capacity = args.capacity or data_driven_capacity(daily_totals)
    snap_id = lock(ws, capacity, args.tempo, oblasts, observations)
    print(f"Locked snapshot #{snap_id} for week of {ws} — "
          f"{capacity}/d × 7 × {args.tempo} = {int(capacity * 7 * args.tempo):,} drones")
    return 0


if __name__ == '__main__':
    sys.exit(main())
