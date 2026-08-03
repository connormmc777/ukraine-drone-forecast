"""
H₁ Scoreboard — rolling vs CAGR, head-to-head, every week.

This is the public record of the horizon-crossover hypothesis test:

  H₀:  Δ MAPE(h) = 0 for all h.  Two models are equally accurate.
  H₁:  ∃ h* such that sign(Δ MAPE) flips.  Point estimate h* ≈ 3 weeks.

Every Monday, two predictions get locked in the DB before the week
plays out — no cherry-picking, no back-fit. When Sunday closes,
both get scored against the realised UA Air Force total. The
running record lives in weekly_model_predictions and is scored
here in real time.

If CAGR keeps winning at these ~1-week horizons, either
  (a) our estimate of h* is too high, or
  (b) rolling's 14-day memory is genuinely too short for tempo shifts.

Either way, the null gets weaker with every closed week.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DB = DATA_DIR / "forecast_history.db"

st.set_page_config(
    page_title="H₁ Scoreboard — Rolling vs CAGR",
    page_icon="🥊",
    layout="wide",
)


# ---------- data ----------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(ttl=60)
def load_scoreboard(mtime_key_db: float, mtime_key_totals: float) -> pd.DataFrame:
    """Join locked predictions with realised weekly actuals; compute APE per model."""
    if not DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB) as c:
        preds = pd.read_sql(
            """SELECT week_start,
                      MAX(CASE WHEN model_name='rolling_14d_x110' THEN predicted_total END) AS rolling,
                      MAX(CASE WHEN model_name='cagr_geometric'   THEN predicted_total END) AS cagr,
                      MAX(CASE WHEN model_name='rolling_14d_x110' THEN locked_at END) AS locked_at_r,
                      MAX(CASE WHEN model_name='cagr_geometric'   THEN locked_at END) AS locked_at_c
               FROM weekly_model_predictions
               GROUP BY week_start
               ORDER BY week_start""",
            c,
        )
    if preds.empty:
        return preds
    preds["week_start"] = pd.to_datetime(preds["week_start"])

    daily_csv = DATA_DIR / "daily_totals.csv"
    if daily_csv.exists():
        d = pd.read_csv(daily_csv).dropna(subset=["launched"])
        d["date"] = pd.to_datetime(d["date"])

        def week_actual(ws):
            wk = d[(d["date"] >= ws) & (d["date"] < ws + pd.Timedelta(days=7))]
            return int(wk["launched"].sum()) if len(wk) else 0

        def week_nights(ws):
            wk = d[(d["date"] >= ws) & (d["date"] < ws + pd.Timedelta(days=7))]
            return len(wk)

        preds["actual"] = preds["week_start"].apply(week_actual)
        preds["nights_reported"] = preds["week_start"].apply(week_nights)
        preds["status"] = preds["nights_reported"].apply(
            lambda n: "closed" if n >= 7 else ("in_progress" if n > 0 else "future")
        )

        # Only score CLOSED weeks (all 7 nights in)
        closed = preds["status"] == "closed"
        preds["ape_rolling"] = np.where(
            closed & (preds["actual"] > 0),
            (preds["rolling"] - preds["actual"]).abs() / preds["actual"] * 100,
            np.nan,
        )
        preds["ape_cagr"] = np.where(
            closed & (preds["actual"] > 0),
            (preds["cagr"] - preds["actual"]).abs() / preds["actual"] * 100,
            np.nan,
        )
        preds["winner"] = np.where(
            preds[["ape_rolling", "ape_cagr"]].notna().all(axis=1),
            np.where(preds["ape_cagr"] < preds["ape_rolling"], "CAGR", "rolling"),
            "",
        )
    return preds


sb = load_scoreboard(_mtime(DB), _mtime(DATA_DIR / "daily_totals.csv"))


st.title("🥊 H₁ Scoreboard — Rolling vs CAGR")
st.markdown(
    "**Public per-week test of the horizon-crossover hypothesis.**  "
    "Every Monday two predictions are locked in the DB before the week "
    "plays out.  Every Sunday both get scored against the realised UA Air "
    "Force total.  This page is that scoring, in real time."
)

if sb.empty:
    st.warning(
        "No locked model predictions on file yet. Run "
        "`python lock_multi_model.py` in the pod, or wait for the next "
        "Monday CronJob to fire."
    )
    st.stop()


# ---------- H₀ / H₁ box ----------
with st.container(border=True):
    st.markdown(
        "### The hypothesis, in measurement terms\n"
        "Let **MAPE_R(h)** and **MAPE_C(h)** be the expected mean absolute "
        "percentage error of the rolling and CAGR models at forecast horizon "
        "*h* weeks. Define **Δ(h) ≔ MAPE_R(h) − MAPE_C(h)**."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**H₀** (null): Δ(h) = 0 for all h ∈ [1, 52].\n\n"
            "*The two models have equal expected MAPE at every horizon; "
            "any observed difference is sampling noise.*"
        )
    with c2:
        st.markdown(
            "**H₁** (alternative): ∃ h\\* such that sign(Δ) flips at h\\*.\n\n"
            "*Point estimate h\\* ≈ 3 weeks. Reject H₀ if signed-rank test "
            "on APE differences is significant across ≥ 40 closed weeks.*"
        )


# ---------- Current KPIs ----------
st.markdown("---")
closed = sb[sb["status"] == "closed"].copy()
in_progress = sb[sb["status"] == "in_progress"]
future = sb[sb["status"] == "future"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Closed weeks on record", f"{len(closed)}")
k2.metric("Weeks needed for signed-rank test", "≥ 40",
          delta=f"{40 - len(closed)} to go" if len(closed) < 40 else None,
          delta_color="off")

if len(closed) > 0:
    wins = closed["winner"].value_counts()
    r_wins = int(wins.get("rolling", 0))
    c_wins = int(wins.get("CAGR", 0))
    k3.metric("Wins — Rolling", f"{r_wins}",
              delta=f"{r_wins/len(closed)*100:.0f}%", delta_color="off")
    k4.metric("Wins — CAGR", f"{c_wins}",
              delta=f"{c_wins/len(closed)*100:.0f}%", delta_color="off")


# ---------- This week (in-progress) ----------
if len(in_progress) > 0:
    st.markdown("---")
    st.subheader("This week — predictions locked, scoring pending")
    row = in_progress.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Week of", row["week_start"].strftime("%Y-%m-%d"))
    c2.metric("Rolling prediction", f"{int(row['rolling']):,}")
    c3.metric("CAGR prediction", f"{int(row['cagr']):,}",
              delta=f"{int(row['cagr'] - row['rolling']):+,}")
    c4.metric(f"Nights reported so far", f"{int(row['nights_reported'])} / 7",
              delta=f"actual so far: {int(row['actual']):,}",
              delta_color="off")


# ---------- Head-to-head table ----------
st.markdown("---")
st.subheader("Per-week head-to-head")

disp = sb.copy()
disp["week_start"] = disp["week_start"].dt.strftime("%Y-%m-%d")
disp["rolling"] = disp["rolling"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
disp["cagr"]    = disp["cagr"].apply(   lambda v: f"{int(v):,}" if pd.notna(v) else "—")
disp["actual"]  = disp["actual"].apply( lambda v: f"{int(v):,}" if v > 0 else "—")
disp["ape_rolling"] = disp["ape_rolling"].apply(
    lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
)
disp["ape_cagr"] = disp["ape_cagr"].apply(
    lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
)
disp = disp[[
    "week_start", "rolling", "cagr", "actual",
    "ape_rolling", "ape_cagr", "winner", "status",
]].rename(columns={
    "week_start": "Week", "rolling": "Rolling", "cagr": "CAGR",
    "actual": "Actual", "ape_rolling": "APE (Rolling)",
    "ape_cagr": "APE (CAGR)", "winner": "Winner", "status": "Status",
})
st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------- Running median APE chart ----------
if len(closed) >= 3:
    st.markdown("---")
    st.subheader("Running median APE — is one model pulling ahead?")

    closed_sorted = closed.sort_values("week_start").reset_index(drop=True)
    closed_sorted["cum_median_r"] = [
        closed_sorted["ape_rolling"].iloc[: i + 1].median() for i in range(len(closed_sorted))
    ]
    closed_sorted["cum_median_c"] = [
        closed_sorted["ape_cagr"].iloc[: i + 1].median() for i in range(len(closed_sorted))
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(closed_sorted["week_start"], closed_sorted["cum_median_r"],
            "o-", color="#003d7a", lw=2.5, label="Rolling (running median APE)")
    ax.plot(closed_sorted["week_start"], closed_sorted["cum_median_c"],
            "o-", color="#0a6d3a", lw=2.5, label="CAGR (running median APE)")
    ax.set_ylabel("Median APE (% error)", fontsize=11)
    ax.set_ylim(0, closed_sorted[["cum_median_r", "cum_median_c"]].max().max() * 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    st.pyplot(fig); plt.close(fig)


# ---------- Wilcoxon signed-rank preview ----------
if len(closed) >= 6:
    st.markdown("---")
    st.subheader("H₀ test — Wilcoxon signed-rank on APE differences")

    diffs = (closed["ape_rolling"] - closed["ape_cagr"]).dropna()
    try:
        stat, p = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = np.nan, np.nan

    st.markdown(
        f"Signed-rank statistic on **{len(diffs)}** paired weeks:  "
        f"**W = {stat:.1f}, p = {p:.4f}**"
    )
    if p < 0.05:
        st.success(
            "**Reject H₀** at α = 0.05. The two models have statistically "
            "different expected APE. Direction of difference: "
            f"**{'CAGR is more accurate' if diffs.median() > 0 else 'Rolling is more accurate'}**."
        )
    else:
        st.info(
            f"**Cannot yet reject H₀** at α = 0.05.  "
            f"Direction of the median difference favors "
            f"**{'CAGR' if diffs.median() > 0 else 'rolling'}** "
            f"({diffs.median():+.1f} pp), but sample size is still small.  "
            f"Need ~{40 - len(closed)} more closed weeks for a reliable test."
        )


# ---------- Method / methodology note ----------
st.markdown("---")
st.caption(
    "**Method.**  The rolling forecast is 14-day mean of nightly launches × 7 × "
    "1.10 buffer, locked via `lock_weekly_snapshot.py`.  The CAGR forecast is "
    "previous week's realised actual × 1.018 weekly compound (derived from 153% "
    "measured annual growth 2024→2026).  Both are written to `weekly_model_predictions` "
    "BEFORE the week plays out — no cherry-picking, no retroactive edits.  "
    "APE = |predicted − actual| / actual × 100.  Wilcoxon signed-rank is the "
    "standard non-parametric test for paired continuous data.  "
    "Repo: [github.com/connormmc777/ukraine-drone-forecast]"
    "(https://github.com/connormmc777/ukraine-drone-forecast)."
)
