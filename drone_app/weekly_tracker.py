"""
Weekly actual-launches ledger with explicit week-close.

Architecture (per the user's spec):
- Predicted: locked at week start, immutable (already in `snapshots` table).
- Actual: grows day-by-day Mon→Sun, each day's row appended.
- Week closure: explicit action that freezes the final weekly total and
  pairs it with the locked snapshot for permanent scoring.

Tables:
  weekly_actuals       — per-day cumulative tracking (mutable until frozen)
  weekly_scores        — final paired predicted-vs-actual (immutable)

Archives:
  data/weekly_actuals_archive/week_YYYY-MM-DD.json
"""
from __future__ import annotations
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np


def init_tables(db_path) -> None:
    """Create the two ledger tables if they don't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS weekly_actuals (
            week_start TEXT NOT NULL,           -- ISO date of Monday
            observation_date TEXT NOT NULL,     -- ISO date of the actual day
            day_of_week INTEGER NOT NULL,       -- 0=Mon, 6=Sun
            daily_launched INTEGER NOT NULL,
            daily_intercepted INTEGER NOT NULL,
            daily_hits INTEGER NOT NULL,
            cumulative_launched INTEGER NOT NULL,
            cumulative_intercepted INTEGER NOT NULL,
            cumulative_hits INTEGER NOT NULL,
            per_oblast_json TEXT,
            recorded_at TEXT NOT NULL,
            is_frozen INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (week_start, observation_date)
        );
        CREATE INDEX IF NOT EXISTS idx_weekly_actuals_week
            ON weekly_actuals(week_start);

        CREATE TABLE IF NOT EXISTS weekly_scores (
            week_start TEXT PRIMARY KEY,
            closed_at TEXT NOT NULL,
            snapshot_id INTEGER,
            predicted_total INTEGER NOT NULL,
            actual_total INTEGER NOT NULL,
            total_error INTEGER,
            pct_off REAL,
            spatial_r REAL,
            mae_per_oblast REAL,
            is_data_gap INTEGER NOT NULL DEFAULT 0,
            per_oblast_json TEXT,
            archive_path TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
        );
        """)


def week_start_of(d: date) -> date:
    """Monday of the ISO week containing d."""
    return d - timedelta(days=d.weekday())


def update_today_actual(
    daily_totals_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    db_path,
    today: date | None = None,
) -> dict:
    """Recompute today's row (and any non-frozen prior days in this week)
    from the latest daily_totals + observations data. Idempotent.

    Frozen rows (is_frozen=1) are never overwritten — only the active week
    gets refreshed."""
    today = today or date.today()
    ws = week_start_of(today)

    daily = daily_totals_df.copy()
    daily['date'] = pd.to_datetime(daily['date'])

    obs = observations_df.copy()
    obs['observation_date'] = pd.to_datetime(obs['observation_date'])

    week_data = (daily[daily['date'] >= pd.Timestamp(ws)]
                 .groupby('date', as_index=False)
                 .agg(launched=('launched', 'sum'),
                      intercepted=('intercepted', 'sum'),
                      hits=('hits', 'sum'))
                 .sort_values('date'))
    week_data = week_data[
        week_data['date'] <= pd.Timestamp(ws + timedelta(days=6))
    ]

    rows_written = 0
    cum_launched = cum_intercepted = cum_hits = 0
    now_iso = datetime.now().isoformat(timespec='seconds')

    with sqlite3.connect(db_path) as conn:
        for _, row in week_data.iterrows():
            d = row['date'].date()
            dow = (d - ws).days
            daily_launched = int(row['launched']) if pd.notna(row['launched']) else 0
            daily_intercepted = int(row['intercepted']) if pd.notna(row['intercepted']) else 0
            daily_hits = int(row['hits']) if pd.notna(row['hits']) else 0
            cum_launched += daily_launched
            cum_intercepted += daily_intercepted
            cum_hits += daily_hits

            # Per-oblast breakdown from observations for this day
            day_obs = obs[obs['observation_date'].dt.date == d]
            per_ob = (day_obs.groupby('oblast')['observed_drones']
                            .sum().astype(int).to_dict())

            # Don't overwrite frozen rows
            existing = conn.execute(
                "SELECT is_frozen FROM weekly_actuals WHERE week_start=? "
                "AND observation_date=?",
                (ws.isoformat(), d.isoformat()),
            ).fetchone()
            if existing and existing[0] == 1:
                continue

            conn.execute("""
                INSERT INTO weekly_actuals(
                    week_start, observation_date, day_of_week,
                    daily_launched, daily_intercepted, daily_hits,
                    cumulative_launched, cumulative_intercepted, cumulative_hits,
                    per_oblast_json, recorded_at, is_frozen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(week_start, observation_date) DO UPDATE SET
                    day_of_week = excluded.day_of_week,
                    daily_launched = excluded.daily_launched,
                    daily_intercepted = excluded.daily_intercepted,
                    daily_hits = excluded.daily_hits,
                    cumulative_launched = excluded.cumulative_launched,
                    cumulative_intercepted = excluded.cumulative_intercepted,
                    cumulative_hits = excluded.cumulative_hits,
                    per_oblast_json = excluded.per_oblast_json,
                    recorded_at = excluded.recorded_at
            """, (
                ws.isoformat(), d.isoformat(), dow,
                daily_launched, daily_intercepted, daily_hits,
                cum_launched, cum_intercepted, cum_hits,
                json.dumps(per_ob), now_iso,
            ))
            rows_written += 1

    return {
        'week_start': ws.isoformat(),
        'rows_written': rows_written,
        'week_to_date_launched': cum_launched,
        'week_to_date_intercepted': cum_intercepted,
        'week_to_date_hits': cum_hits,
    }


def freeze_week(week_start: date, db_path) -> int:
    """Mark all rows for `week_start` as frozen (no longer updated by syncs)."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE weekly_actuals SET is_frozen=1 WHERE week_start=?",
            (week_start.isoformat(),),
        )
        return cur.rowcount


def close_week(week_start: date, db_path, data_dir: Path,
               oblasts_df: pd.DataFrame | None = None) -> dict:
    """Close a week: freeze its actuals + write the final scored row in
    weekly_scores + archive a JSON snapshot. Idempotent — re-running just
    refreshes the JSON archive and the score row (use this to recompute
    after corrections).
    """
    data_dir = Path(data_dir)
    archive_dir = data_dir / 'weekly_actuals_archive'
    archive_dir.mkdir(exist_ok=True)
    ws_iso = week_start.isoformat()

    with sqlite3.connect(db_path) as conn:
        # Pull all days for the week
        actuals = pd.read_sql_query(
            "SELECT * FROM weekly_actuals WHERE week_start=? "
            "ORDER BY observation_date",
            conn, params=(ws_iso,))

        if actuals.empty:
            return {'status': 'no_data', 'week_start': ws_iso,
                    'is_data_gap': True}

        # Pair with the locked snapshot for this week (latest data-driven)
        snap = conn.execute(
            "SELECT id, weekly_budget FROM snapshots WHERE week_start=? "
            "ORDER BY id DESC LIMIT 1",
            (ws_iso,),
        ).fetchone()
        snap_id, predicted_total = (snap[0], int(snap[1])) if snap else (None, 0)

        # Per-oblast aggregation across the week
        per_oblast_week = {}
        for _, r in actuals.iterrows():
            day_per = json.loads(r['per_oblast_json'] or '{}')
            for ob, c in day_per.items():
                per_oblast_week[ob] = per_oblast_week.get(ob, 0) + int(c)

        actual_total = int(actuals['daily_launched'].sum())
        total_error = predicted_total - actual_total
        pct_off = (total_error / actual_total * 100) if actual_total else None

        # Spatial r vs snapshot predictions
        spatial_r = None
        mae = None
        if snap_id is not None and per_oblast_week:
            snap_rows = pd.read_sql_query(
                "SELECT oblast, predicted_week FROM snapshot_rows "
                "WHERE snapshot_id=?", conn, params=(snap_id,))
            merged = snap_rows.merge(
                pd.DataFrame([{'oblast': k, 'actual': v}
                              for k, v in per_oblast_week.items()]),
                on='oblast', how='left'
            ).fillna({'actual': 0})
            if merged['predicted_week'].std() and merged['actual'].std():
                pred_share = merged['predicted_week'] / max(merged['predicted_week'].sum(), 1)
                actual_share = merged['actual'] / max(merged['actual'].sum(), 1)
                spatial_r = float(np.corrcoef(pred_share, actual_share)[0, 1])
            mae = float((merged['predicted_week'] - merged['actual']).abs().mean())

        is_data_gap = actual_total == 0
        archive_path = archive_dir / f"week_{ws_iso}.json"

        # Build the archive payload
        archive = {
            'week_start': ws_iso,
            'closed_at': datetime.now().isoformat(timespec='seconds'),
            'snapshot_id': snap_id,
            'predicted_total': predicted_total,
            'actual_total': actual_total,
            'total_error': total_error,
            'pct_off': pct_off,
            'spatial_r': spatial_r,
            'mae_per_oblast': mae,
            'is_data_gap': is_data_gap,
            'per_oblast_actual': per_oblast_week,
            'daily_progression': actuals[[
                'observation_date', 'day_of_week', 'daily_launched',
                'daily_intercepted', 'daily_hits',
                'cumulative_launched', 'cumulative_intercepted',
                'cumulative_hits',
            ]].to_dict(orient='records'),
        }
        archive_path.write_text(json.dumps(archive, indent=2, default=str))

        # Upsert into weekly_scores
        conn.execute("""
            INSERT INTO weekly_scores(
                week_start, closed_at, snapshot_id, predicted_total,
                actual_total, total_error, pct_off, spatial_r,
                mae_per_oblast, is_data_gap, per_oblast_json, archive_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(week_start) DO UPDATE SET
                closed_at = excluded.closed_at,
                snapshot_id = excluded.snapshot_id,
                predicted_total = excluded.predicted_total,
                actual_total = excluded.actual_total,
                total_error = excluded.total_error,
                pct_off = excluded.pct_off,
                spatial_r = excluded.spatial_r,
                mae_per_oblast = excluded.mae_per_oblast,
                is_data_gap = excluded.is_data_gap,
                per_oblast_json = excluded.per_oblast_json,
                archive_path = excluded.archive_path
        """, (ws_iso, archive['closed_at'], snap_id,
              predicted_total, actual_total, total_error, pct_off,
              spatial_r, mae, int(is_data_gap),
              json.dumps(per_oblast_week), str(archive_path)))

        # Freeze the actuals rows
        conn.execute(
            "UPDATE weekly_actuals SET is_frozen=1 WHERE week_start=?",
            (ws_iso,),
        )

    return {
        'status': 'closed',
        'week_start': ws_iso,
        'predicted_total': predicted_total,
        'actual_total': actual_total,
        'pct_off': pct_off,
        'spatial_r': spatial_r,
        'is_data_gap': is_data_gap,
        'archive_path': str(archive_path),
    }


def auto_close_pending_weeks(db_path, data_dir: Path,
                               today: date | None = None,
                               oblasts_df: pd.DataFrame | None = None) -> list[dict]:
    """For every week whose Sunday has passed, ensure it's closed.
    Called from the autostart sync loop so weeks close automatically every
    Monday morning without manual intervention.

    Also closes data-gap weeks: any locked snapshot whose week has zero
    observations gets a weekly_scores row marked is_data_gap=True so the
    prediction is permanently paired with 'actual = UNKNOWN'."""
    today = today or date.today()
    current_ws = week_start_of(today)
    results = []

    with sqlite3.connect(db_path) as conn:
        # Weeks with any actuals data
        weeks_with_data = set(r[0] for r in conn.execute(
            "SELECT DISTINCT week_start FROM weekly_actuals"
        ).fetchall())
        # Weeks with locked snapshots
        weeks_with_snap = set(r[0] for r in conn.execute(
            "SELECT DISTINCT week_start FROM snapshots"
        ).fetchall())
        closed_weeks = set(r[0] for r in conn.execute(
            "SELECT week_start FROM weekly_scores"
        ).fetchall())

    all_relevant = (weeks_with_data | weeks_with_snap)
    for ws_iso in sorted(all_relevant):
        ws = date.fromisoformat(ws_iso)
        if ws >= current_ws or ws_iso in closed_weeks:
            continue
        if ws_iso in weeks_with_data:
            result = close_week(ws, db_path, data_dir, oblasts_df)
        else:
            # Snapshot exists but zero observations — DATA GAP
            result = close_data_gap_week(ws, db_path, data_dir)
        results.append(result)

    return results


def close_data_gap_week(week_start: date, db_path, data_dir: Path) -> dict:
    """Mark a week as a data gap: snapshot exists but no actuals captured.
    Writes a weekly_scores row with is_data_gap=1 and a JSON archive
    documenting the gap. The PREDICTION is still preserved (via the
    snapshots table); this just records that we know the actual is unknown."""
    data_dir = Path(data_dir)
    archive_dir = data_dir / 'weekly_actuals_archive'
    archive_dir.mkdir(exist_ok=True)
    ws_iso = week_start.isoformat()

    with sqlite3.connect(db_path) as conn:
        snap = conn.execute(
            "SELECT id, weekly_budget FROM snapshots WHERE week_start=? "
            "ORDER BY id DESC LIMIT 1",
            (ws_iso,),
        ).fetchone()
        snap_id, predicted_total = (snap[0], int(snap[1])) if snap else (None, 0)

        archive_path = archive_dir / f"week_{ws_iso}_GAP.json"
        archive = {
            'week_start': ws_iso,
            'closed_at': datetime.now().isoformat(timespec='seconds'),
            'snapshot_id': snap_id,
            'predicted_total': predicted_total,
            'actual_total': None,
            'is_data_gap': True,
            'reason': 'No observations captured during this week. '
                      'Telegram channel preview rotated past these dates '
                      'before a sync ran. Backfill via Telethon if needed.',
            'per_oblast_actual': None,
            'daily_progression': [],
        }
        archive_path.write_text(json.dumps(archive, indent=2, default=str))

        conn.execute("""
            INSERT INTO weekly_scores(
                week_start, closed_at, snapshot_id, predicted_total,
                actual_total, total_error, pct_off, spatial_r,
                mae_per_oblast, is_data_gap, per_oblast_json, archive_path
            ) VALUES (?, ?, ?, ?, 0, 0, NULL, NULL, NULL, 1, NULL, ?)
            ON CONFLICT(week_start) DO UPDATE SET
                closed_at = excluded.closed_at,
                is_data_gap = 1,
                archive_path = excluded.archive_path
        """, (ws_iso, archive['closed_at'], snap_id, predicted_total,
              str(archive_path)))

    return {
        'status': 'closed_data_gap',
        'week_start': ws_iso,
        'predicted_total': predicted_total,
        'actual_total': None,
        'is_data_gap': True,
        'archive_path': str(archive_path),
    }


def get_week_progress(week_start: date, db_path) -> pd.DataFrame:
    """Return the day-by-day growth DataFrame for a week (for display)."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM weekly_actuals WHERE week_start=? "
            "ORDER BY observation_date",
            conn, params=(week_start.isoformat(),))


def get_all_closed_weeks(db_path) -> pd.DataFrame:
    """All weeks that have been formally closed (final scores)."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM weekly_scores ORDER BY week_start", conn)
