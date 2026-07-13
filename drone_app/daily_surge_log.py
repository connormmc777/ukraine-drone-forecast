"""
Daily surge probability logger.

Runs once a day (via k8s CronJob). Two jobs:

  1. Compute today's P(surge) for TOMORROW using the SurgeModel fit on all
     available history. Persist the row.

  2. Look back at YESTERDAY's prediction and stamp it with the actual outcome
     (did yesterday's launches exceed the surge threshold?). This creates
     a rolling calibration record that lets us evaluate the model over time
     without any manual bookkeeping.

Table schema (created if missing):

  CREATE TABLE surge_daily_log (
      log_date            TEXT PRIMARY KEY,      -- date the log entry was written (today)
      target_date         TEXT NOT NULL,         -- date the prediction is FOR (tomorrow)
      p_surge             REAL NOT NULL,         -- probability [0,1]
      z_score             REAL NOT NULL,         -- log-odds
      threshold           INTEGER NOT NULL,      -- surge threshold in drones/night
      features_json       TEXT NOT NULL,         -- serialized SurgeFeatures
      contributions_json  TEXT NOT NULL,         -- per-feature contribution to z
      coeffs_json         TEXT NOT NULL,         -- coefficients used
      fitted              INTEGER NOT NULL,      -- 1 if MLE fit worked, 0 if defaults
      n_train_days        INTEGER NOT NULL,      -- training sample size at prediction time
      n_surges            INTEGER NOT NULL,      -- # of surge events seen so far
      actual_launched     INTEGER,               -- filled in AFTER the target date
      was_surge           INTEGER,               -- 1/0, filled after the target date
      created_at          TEXT NOT NULL DEFAULT (datetime('now'))
  );

Idempotent: if today's log entry already exists, we UPDATE it (fresh model
re-fit on newer data). The retroactive fill only writes if actual_launched is
still NULL for that date.
"""
from __future__ import annotations
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

project_dir = Path(__file__).parent
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

from surge_probability import SurgeModel, DEFAULT_SURGE_THRESHOLD


DATA_DIR = project_dir / "data"
DB_PATH = DATA_DIR / "forecast_history.db"
DAILY_TOTALS_CSV = DATA_DIR / "daily_totals.csv"


SCHEMA = """
CREATE TABLE IF NOT EXISTS surge_daily_log (
    log_date            TEXT PRIMARY KEY,
    target_date         TEXT NOT NULL,
    p_surge             REAL NOT NULL,
    z_score             REAL NOT NULL,
    threshold           INTEGER NOT NULL,
    features_json       TEXT NOT NULL,
    contributions_json  TEXT NOT NULL,
    coeffs_json         TEXT NOT NULL,
    fitted              INTEGER NOT NULL,
    n_train_days        INTEGER NOT NULL,
    n_surges            INTEGER NOT NULL,
    actual_launched     INTEGER,
    was_surge           INTEGER,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sdl_target ON surge_daily_log(target_date);
"""


def load_daily_series() -> pd.DataFrame:
    """The full history the model trains on."""
    if not DAILY_TOTALS_CSV.exists():
        return pd.DataFrame(columns=["date", "launched"])
    df = pd.read_csv(DAILY_TOTALS_CSV).dropna(subset=["launched"])
    df = df.rename(columns={"observation_date": "date"})[["date", "launched"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def predict_and_log(today: date | None = None,
                     threshold: int = DEFAULT_SURGE_THRESHOLD) -> dict:
    today = today or date.today()
    target = today + timedelta(days=1)
    daily = load_daily_series()

    model = SurgeModel(threshold=threshold)
    stats = model.fit(daily)
    result = model.predict_next(daily, target_date=target)

    row = {
        "log_date":           today.isoformat(),
        "target_date":        target.isoformat(),
        "p_surge":            float(result["p_surge"]),
        "z_score":            float(result["z"]),
        "threshold":          threshold,
        "features_json":      json.dumps(result["features"]),
        "contributions_json": json.dumps(result["contributions_to_z"]),
        "coeffs_json":        json.dumps(model.coeffs),
        "fitted":             int(bool(model.fitted)),
        "n_train_days":       int(stats.get("n_days_trained_on",
                                             stats.get("n_days", 0))),
        "n_surges":           int(stats.get("n_surges", 0)),
    }

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        cols = ",".join(row.keys())
        placeholders = ",".join([f":{k}" for k in row.keys()])
        conn.execute(
            f"INSERT INTO surge_daily_log ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(log_date) DO UPDATE SET "
            f"  target_date        = excluded.target_date,"
            f"  p_surge            = excluded.p_surge,"
            f"  z_score            = excluded.z_score,"
            f"  threshold          = excluded.threshold,"
            f"  features_json      = excluded.features_json,"
            f"  contributions_json = excluded.contributions_json,"
            f"  coeffs_json        = excluded.coeffs_json,"
            f"  fitted             = excluded.fitted,"
            f"  n_train_days       = excluded.n_train_days,"
            f"  n_surges           = excluded.n_surges",
            row,
        )
        conn.commit()

    return row


def backfill_actuals(threshold: int = DEFAULT_SURGE_THRESHOLD) -> int:
    """For every log row whose target_date is now in the past AND whose
    actual_launched is still NULL, look up the observed launch count from
    daily_totals and stamp actual_launched + was_surge. Returns rows updated."""
    daily = load_daily_series()
    if daily.empty:
        return 0

    by_date = daily.set_index(daily["date"].dt.date)["launched"].to_dict()
    today = date.today()
    updated = 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        rows = conn.execute(
            "SELECT rowid, target_date FROM surge_daily_log "
            "WHERE actual_launched IS NULL AND target_date < ?",
            (today.isoformat(),),
        ).fetchall()
        for rowid, target_str in rows:
            target = date.fromisoformat(target_str)
            actual = by_date.get(target)
            if actual is None:
                continue
            was_surge = 1 if actual >= threshold else 0
            conn.execute(
                "UPDATE surge_daily_log "
                "SET actual_launched = ?, was_surge = ? WHERE rowid = ?",
                (int(actual), was_surge, rowid),
            )
            updated += 1
        conn.commit()
    return updated


def main() -> int:
    row = predict_and_log()
    filled = backfill_actuals()
    print(f"Logged prediction for {row['target_date']}: "
          f"P(surge) = {row['p_surge']:.4f}  z = {row['z_score']:+.3f}  "
          f"(n_train={row['n_train_days']}, n_surges={row['n_surges']}, "
          f"fitted={bool(row['fitted'])})")
    print(f"Backfilled actuals for {filled} past prediction rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
