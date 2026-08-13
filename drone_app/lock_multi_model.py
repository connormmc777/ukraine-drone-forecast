"""Multi-model weekly lock — extends the rolling-only snapshot flow to
also lock a CAGR-based prediction per week, so H₁ (rolling vs CAGR)
has a public per-week scoreboard.

Schema:
    CREATE TABLE weekly_model_predictions (
        week_start        TEXT,    -- Monday ISO date
        model_name        TEXT,    -- 'rolling_14d_x110' or 'cagr_geometric'
        predicted_total   INTEGER, -- forecast for that week
        method_note       TEXT,    -- how it was computed
        locked_at         TEXT     -- when we wrote it
    )

Idempotent on (week_start, model_name). Backfills last 12 weeks of
CAGR counterfactuals on first run so the scoreboard has history.
"""
from __future__ import annotations
import sqlite3, sys, os
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DB = DATA_DIR / "forecast_history.db"

# 153% recent annual CAGR (2024→2026) → weekly compound rate
CAGR_ANNUAL   = 1.53
CAGR_WEEKLY   = (1 + CAGR_ANNUAL) ** (1 / 52) - 1   # ≈ 0.018 per week

# Regime detector — imported so we can lock the state + a third prediction
# per week alongside rolling and CAGR.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
import regime_detector as rd

SCHEMA = """
CREATE TABLE IF NOT EXISTS weekly_model_predictions (
    week_start       TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    predicted_total  INTEGER NOT NULL,
    method_note      TEXT,
    locked_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (week_start, model_name)
);
CREATE INDEX IF NOT EXISTS idx_wmp_week ON weekly_model_predictions(week_start);

CREATE TABLE IF NOT EXISTS weekly_regime_state (
    week_start   TEXT PRIMARY KEY,
    state        TEXT NOT NULL,      -- onset / sustaining / collapsing / steady
    d1           REAL,
    d2           REAL,
    reason       TEXT,
    locked_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _upsert(conn, week_start: date, model: str, pred: float, note: str):
    conn.execute(
        """INSERT INTO weekly_model_predictions
           (week_start, model_name, predicted_total, method_note)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(week_start, model_name) DO UPDATE SET
             predicted_total = excluded.predicted_total,
             method_note     = excluded.method_note,
             locked_at       = datetime('now')""",
        (week_start.isoformat(), model, int(round(pred)), note),
    )


def _rolling_prediction_from_snapshot(conn, week_start: date) -> tuple[int, str] | None:
    """Reuse whatever lock_weekly_snapshot.py already stored — single
    source of truth for the rolling budget."""
    row = conn.execute(
        "SELECT weekly_budget, note FROM snapshots "
        "WHERE week_start = ? ORDER BY id DESC LIMIT 1",
        (week_start.isoformat(),),
    ).fetchone()
    return (row[0], row[1] or "") if row else None


def _cagr_prediction(week_start: date, daily: pd.DataFrame) -> tuple[int, str] | None:
    """CAGR-based: previous week's actual × weekly compound growth."""
    prev_start = week_start - timedelta(days=7)
    prev_end   = week_start - timedelta(days=1)
    prev = daily[
        (daily["date"] >= pd.Timestamp(prev_start)) &
        (daily["date"] <= pd.Timestamp(prev_end))
    ]
    if len(prev) < 5:
        return None
    prev_actual = int(prev["launched"].sum())
    pred = prev_actual * (1 + CAGR_WEEKLY)
    note = (
        f"prev_week_actual={prev_actual} × "
        f"({1 + CAGR_WEEKLY:.4f} weekly compound from 153% annual CAGR)"
    )
    return int(round(pred)), note


def load_daily() -> pd.DataFrame:
    if not (DATA_DIR / "daily_totals.csv").exists():
        return pd.DataFrame(columns=["date", "launched"])
    d = pd.read_csv(DATA_DIR / "daily_totals.csv").dropna(subset=["launched"])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _upsert_regime(conn, week_start: date, r: dict):
    conn.execute(
        """INSERT INTO weekly_regime_state
           (week_start, state, d1, d2, reason)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(week_start) DO UPDATE SET
             state = excluded.state, d1 = excluded.d1,
             d2 = excluded.d2, reason = excluded.reason,
             locked_at = datetime('now')""",
        (week_start.isoformat(), r["state"], r["d1"], r["d2"], r["reason"]),
    )


def lock_week(week_start: date, conn, daily: pd.DataFrame) -> dict:
    """Lock every model's prediction for `week_start` in a single
    atomic Job. New models can be added to the loop below without
    schema changes — each writes one row to weekly_model_predictions.
    """
    result = {"week_start": week_start.isoformat()}

    # ---- 1. Rolling (canonical from snapshots table) ----
    r = _rolling_prediction_from_snapshot(conn, week_start)
    if r:
        _upsert(conn, week_start, "rolling_14d_x110", r[0],
                f"14d mean × 7 × 1.10 (mirror of snapshots table): {r[1][:80]}")
        result["rolling"] = r[0]

    # ---- 2. CAGR (kept for backward compat + comparison) ----
    c = _cagr_prediction(week_start, daily)
    if c:
        _upsert(conn, week_start, "cagr_geometric", c[0], c[1])
        result["cagr"] = c[0]

    # ---- 3. Regime state ----
    regime = rd.detect_regime(daily, as_of_date=week_start)
    _upsert_regime(conn, week_start, regime)
    result["regime"] = regime["state"]

    # ---- 4. Regime-aware combined prediction ----
    if "rolling" in result and "cagr" in result:
        ra_pred, ra_note = rd.apply_bias_correction(
            result["rolling"], result["cagr"], regime["state"]
        )
        _upsert(conn, week_start, "regime_aware", ra_pred, ra_note)
        result["regime_aware"] = ra_pred

    # ---- 5. Alternative forecasters (proper models per reader critique) ----
    try:
        import forecasters as fc
        for model_name, fn in fc.FORECASTERS.items():
            # rolling_baseline in forecasters.py duplicates rolling_14d_x110 —
            # skip to avoid double-writing the same number under two names
            if model_name == "rolling_baseline":
                continue
            try:
                pred = fn(daily, week_start)
                _upsert(conn, week_start, model_name,
                        pred["predicted_total"], pred["method_note"])
                # Also store the 90% PI when we have one
                if pred.get("lower_90") is not None:
                    _upsert(conn, week_start, f"{model_name}_lo90",
                            pred["lower_90"], f"PI lower for {model_name}")
                    _upsert(conn, week_start, f"{model_name}_hi90",
                            pred["upper_90"], f"PI upper for {model_name}")
                result[model_name] = pred["predicted_total"]
            except Exception as _e:
                # A single model failing must not block the others
                result[f"{model_name}_error"] = f"{type(_e).__name__}: {_e}"
    except ImportError:
        result["forecasters_error"] = "forecasters.py not importable"

    return result


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA)

    daily = load_daily()
    if daily.empty:
        print("No daily_totals.csv; cannot compute CAGR baseline.")
        conn.close()
        return 1

    # 1. Lock this week
    this_monday = _monday_of(date.today())
    result = lock_week(this_monday, conn, daily)
    print(f"THIS WEEK  {result}")

    # 2. Backfill last 12 weeks — one-time or idempotent depending on flag
    backfill = "--no-backfill" not in sys.argv
    if backfill:
        print()
        print("Backfilling last 12 weeks (idempotent — updates existing rows):")
        for wks_back in range(1, 13):
            wk = this_monday - timedelta(days=7 * wks_back)
            r = lock_week(wk, conn, daily)
            print(f"  {r}")

    conn.commit()

    # 3. Print scoreboard so the caller can see it
    print()
    print("=" * 74)
    print("SCOREBOARD — weekly_model_predictions")
    print("=" * 74)
    rows = conn.execute("""
        SELECT week_start,
               MAX(CASE WHEN model_name='rolling_14d_x110' THEN predicted_total END) AS rolling,
               MAX(CASE WHEN model_name='cagr_geometric'  THEN predicted_total END) AS cagr
        FROM weekly_model_predictions
        GROUP BY week_start
        ORDER BY week_start
    """).fetchall()
    print(f"{'week_start':<12}{'rolling':>12}{'cagr':>12}{'delta':>12}")
    for wk, rl, cg in rows:
        if rl and cg:
            print(f"{wk:<12}{rl:>12,}{cg:>12,}{cg - rl:>+12,}")
        else:
            print(f"{wk:<12}{(rl or '—'):>12}{(cg or '—'):>12}{'—':>12}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
