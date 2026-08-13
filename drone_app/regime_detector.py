"""
Regime detector — the missing third component from the LinkedIn writeup.

Neither the rolling nor the CAGR forecast handles regime TRANSITIONS
(surge onset, surge collapse).  Both handle regime STATES fine.
This module classifies the current regime from the daily-launches
history so a downstream predictor can bias its projection
appropriately.

Rules (published in advance, no post-hoc tuning):

    d1 = latest night's launched  ÷  7-day trailing mean
    d2 = 7-day trailing mean       ÷  14-day trailing mean

    d1 > 1.5   →  onset        (bias projection UP)
    d2 > 1.3   →  sustaining   (bias projection UP, weaker)
    d2 < 0.7   →  collapsing   (bias projection DOWN)
    else       →  steady       (rolling is marginally best here)

Regime-aware combined prediction:

    onset       →  0.5 * (rolling + cagr) * 1.15    # both under; bump up
    sustaining  →  cagr * 1.10                       # trend-model + boost
    collapsing  →  cagr * 0.75                       # cagr wins, bias down
    steady      →  rolling                           # keep it simple

These multipliers are hypotheses — not tuned to the backfill.
Whether they IMPROVE MAPE against actuals will be scored publicly.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd


RegimeState = Literal["onset", "sustaining", "collapsing", "steady"]

# Detection thresholds (locked constants)
D1_ONSET_THRESHOLD       = 1.50
D2_SUSTAINING_THRESHOLD  = 1.30
D2_COLLAPSE_THRESHOLD    = 0.70

# Bias-adjustment multipliers per state (locked hypotheses)
ONSET_BIAS_UP        = 1.15
SUSTAINING_BIAS_UP   = 1.10
COLLAPSE_BIAS_DOWN   = 0.75


def detect_regime(daily_df: pd.DataFrame, as_of_date: date | None = None) -> dict:
    """Classify the regime state as of `as_of_date` (defaults to today).

    Requires at least 14 nights of history before as_of_date to compute
    both moving averages.  Returns a dict with the state + raw ratios
    so the caller can audit the decision.
    """
    if daily_df.empty:
        return {"state": "steady", "reason": "no data",
                "d1": None, "d2": None, "as_of": None}

    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    as_of_ts = pd.Timestamp(as_of_date) if as_of_date else daily_df["date"].max()
    prior = daily_df[daily_df["date"] < as_of_ts]

    if len(prior) < 14:
        return {"state": "steady", "reason": f"only {len(prior)} nights prior",
                "d1": None, "d2": None, "as_of": as_of_ts.date().isoformat()}

    latest_night = prior.iloc[-1]["launched"]
    mean_7  = prior.tail(7)["launched"].mean()
    mean_14 = prior.tail(14)["launched"].mean()

    d1 = latest_night / mean_7  if mean_7  > 0 else 0
    d2 = mean_7       / mean_14 if mean_14 > 0 else 0

    state: RegimeState = "steady"
    reason = f"d1={d1:.2f}, d2={d2:.2f}"
    if d1 > D1_ONSET_THRESHOLD:
        state = "onset"
        reason = f"d1={d1:.2f} > {D1_ONSET_THRESHOLD} (latest night vs 7d mean)"
    elif d2 > D2_SUSTAINING_THRESHOLD:
        state = "sustaining"
        reason = f"d2={d2:.2f} > {D2_SUSTAINING_THRESHOLD} (7d vs 14d mean)"
    elif d2 < D2_COLLAPSE_THRESHOLD:
        state = "collapsing"
        reason = f"d2={d2:.2f} < {D2_COLLAPSE_THRESHOLD} (7d vs 14d mean)"

    return {
        "state": state,
        "reason": reason,
        "d1": round(float(d1), 3),
        "d2": round(float(d2), 3),
        "latest_night": int(latest_night),
        "mean_7":  round(float(mean_7),  1),
        "mean_14": round(float(mean_14), 1),
        "as_of": as_of_ts.date().isoformat(),
    }


def apply_bias_correction(rolling_pred: float, cagr_pred: float,
                            regime_state: RegimeState) -> tuple[int, str]:
    """Return the regime-aware combined prediction + a plain-English
    method note explaining how it was derived."""
    if regime_state == "onset":
        pred = 0.5 * (rolling_pred + cagr_pred) * ONSET_BIAS_UP
        note = (f"onset: avg({rolling_pred:.0f}, {cagr_pred:.0f}) "
                f"× {ONSET_BIAS_UP} = both models likely under, bump up")
    elif regime_state == "sustaining":
        pred = cagr_pred * SUSTAINING_BIAS_UP
        note = (f"sustaining: cagr ({cagr_pred:.0f}) × {SUSTAINING_BIAS_UP} "
                f"= trend-following + boost")
    elif regime_state == "collapsing":
        pred = cagr_pred * COLLAPSE_BIAS_DOWN
        note = (f"collapsing: cagr ({cagr_pred:.0f}) × {COLLAPSE_BIAS_DOWN} "
                f"= both models likely over, bias down")
    else:  # steady
        pred = rolling_pred
        note = f"steady: rolling ({rolling_pred:.0f}) — marginally better in flat regimes"
    return int(round(pred)), note


def backfill_states(daily_df: pd.DataFrame, mondays: list[date]) -> pd.DataFrame:
    """Compute the regime state that WOULD HAVE been detected at each
    Monday's lock time, given only data available before that Monday.
    Enables honest counterfactual scoring against past actuals."""
    rows = []
    for m in mondays:
        r = detect_regime(daily_df, as_of_date=m)
        rows.append({
            "week_start": m.isoformat(),
            "state":      r["state"],
            "d1":         r["d1"],
            "d2":         r["d2"],
            "reason":     r["reason"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = Path(__file__).parent / "data"
    daily = pd.read_csv(D / "daily_totals.csv").dropna(subset=["launched"])
    daily["date"] = pd.to_datetime(daily["date"])
    print("Today's regime:", detect_regime(daily))
