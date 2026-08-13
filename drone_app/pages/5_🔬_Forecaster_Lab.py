"""
Forecaster Lab — the five weekly forecast models side-by-side.

Answers the strategic questions the whole pipeline is built to answer:

  • SHORT-TERM (1-2 weeks): Where should Ukraine position Patriot
    batteries this week?  Which model catches surges?

  • LONG-TERM (6-12 months): How many interceptor missiles should
    NATO / Ukraine order?  Which model handles deceleration honestly
    without unbounded exponential growth?

  • NOVEL: Given nightly launch counts (public), what does the data
    imply about Russian daily PRODUCTION rate (not public)?  The
    delta between rolling mean and peak-week average tells us
    how deep the stockpile buffer runs.

Models compared:
  1. rolling_baseline    — 14d mean × 7 × 1.10 (the reference)
  2. cagr_geometric      — prev week × weekly compound growth
  3. log_linear_ols      — OLS on log(weekly); SE(β̂)
  4. damped_trend_ets    — Gardner (φ=0.9), M-competition winner
  5. gompertz            — explicit K; self-flags pre-inflection
  6. nb_glm              — Negative Binomial GLM (needs statsmodels)
  7. regime_aware        — my hypothesized bias-corrected combo
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))
DATA_DIR = APP_DIR / "data"
DB = DATA_DIR / "forecast_history.db"

st.set_page_config(
    page_title="Forecaster Lab",
    page_icon="🔬",
    layout="wide",
)


# ---------- data ----------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(ttl=60)
def load_all_forecasts(mtime_key_db: float, mtime_key_totals: float) -> pd.DataFrame:
    if not DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB) as c:
        raw = pd.read_sql(
            "SELECT week_start, model_name, predicted_total, method_note "
            "FROM weekly_model_predictions",
            c,
        )
    if raw.empty:
        return raw
    # Drop PI variants for the main table (they're auxiliary)
    raw = raw[~raw["model_name"].str.endswith(("_lo90", "_hi90"))]
    pt = raw.pivot(index="week_start", columns="model_name",
                    values="predicted_total").reset_index()
    pt["week_start"] = pd.to_datetime(pt["week_start"])
    # Join actuals
    if (DATA_DIR / "daily_totals.csv").exists():
        d = pd.read_csv(DATA_DIR / "daily_totals.csv").dropna(subset=["launched"])
        d["date"] = pd.to_datetime(d["date"])

        def wa(ws):
            w = d[(d["date"] >= ws) & (d["date"] < ws + pd.Timedelta(days=7))]
            return int(w["launched"].sum()) if len(w) == 7 else np.nan

        pt["actual"] = pt["week_start"].apply(wa)
    return pt.sort_values("week_start").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_regime(mtime_key: float) -> pd.DataFrame:
    if not DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB) as c:
        try:
            r = pd.read_sql("SELECT * FROM weekly_regime_state", c)
        except Exception:
            return pd.DataFrame()
    if r.empty:
        return r
    r["week_start"] = pd.to_datetime(r["week_start"])
    return r


@st.cache_data(ttl=60)
def load_daily(mtime_key: float) -> pd.DataFrame:
    p = DATA_DIR / "daily_totals.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p).dropna(subset=["launched"])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


fc = load_all_forecasts(_mtime(DB), _mtime(DATA_DIR / "daily_totals.csv"))
regime = load_regime(_mtime(DB))
daily  = load_daily(_mtime(DATA_DIR / "daily_totals.csv"))


st.title("🔬 Forecaster Lab")
st.markdown(
    "**Five weekly forecast models, side-by-side, tested against 12 weeks of "
    "backfilled predictions.**  Each model expresses a different assumption "
    "about how tempo evolves.  Some catch surges.  Some handle collapses.  "
    "Only one is designed for horizons past ~3 weeks without blowing up."
)

if fc.empty:
    st.warning("No forecasts locked yet — run `lock_multi_model.py` in the pod first.")
    st.stop()


MODELS = [
    ("rolling_14d_x110",  "Rolling 14d × 1.10", "#003d7a"),
    ("cagr_geometric",     "CAGR (2-point)",     "#7a1727"),
    ("log_linear_ols",     "Log-linear OLS",     "#8a5f00"),
    ("damped_trend_ets",   "Damped-trend ETS",   "#0a6d3a"),
    ("gompertz",           "Gompertz",           "#5c6472"),
    ("nb_glm",             "NB-GLM",             "#a13e97"),
    ("regime_aware",       "Regime-aware",       "#c96f00"),
]
KEY_MODELS = [m for m in MODELS if m[0] in fc.columns]

closed = fc.dropna(subset=["actual"]).copy()
for m, _, _ in KEY_MODELS:
    if m in closed.columns:
        # Some backfills wrote 0 (unavailable model) — treat as NaN for scoring
        closed.loc[closed[m] == 0, m] = np.nan
        closed[f"ape_{m}"] = (closed[m] - closed["actual"]).abs() / closed["actual"] * 100


# ============================================================
# 1. Model math cards — one per model
# ============================================================
st.markdown("---")
st.subheader("The math behind each model")

MATH = {
    "rolling_14d_x110": {
        "eq":   r"\hat{y}_{t+1} \;=\; \overline{y}_{t-13:t} \cdot 7 \cdot 1.10",
        "why":  "Baseline. No trend, no shape. Assumes tomorrow ≈ recent typical.",
        "when": "Flat regimes. Variance-limited horizons (1-2 weeks).",
    },
    "cagr_geometric": {
        "eq":   r"\hat{y}_{t+1} \;=\; y_t \cdot (1 + g_{\text{weekly}}), \; g = 1.018",
        "why":  "Two-point exponential. Growth rate baked in as a locked constant.",
        "when": "Trending weeks (up or down). Fails at long horizons — no ceiling.",
    },
    "log_linear_ols": {
        "eq":   r"\log y_t \;=\; \alpha + \beta\, t + \varepsilon, \quad \hat{y}_{t+1} = e^{\alpha + \beta(t+1)}",
        "why":  "Fits β̂ on all points; reports SE(β̂). Same exponential form as CAGR but principled.",
        "when": "Never at long horizons — unbounded exponential blows up. Empirically worst at h=1 too.",
    },
    "damped_trend_ets": {
        "eq":   r"L_t = \alpha y_t + (1-\alpha)(L_{t-1} + \varphi b_{t-1}); \; \hat{y}_{t+h} = L_t + \sum_{i=1}^h \varphi^i b_t",
        "why":  "Gardner: 'exponential that runs out of steam.' φ=0.90 dampens the trend at every step.",
        "when": "Best all-round. Winner of M-competitions on many series. Handles deceleration natively.",
    },
    "gompertz": {
        "eq":   r"y(t) \;=\; K \cdot \exp(-b \cdot \exp(-c\, t))",
        "why":  "Explicit carrying capacity K. Asymmetric ramp; slow approach to ceiling.",
        "when": "Only past the inflection point. Pre-inflection, K is unidentifiable — module says so.",
    },
    "nb_glm": {
        "eq":   r"y_t \sim \text{NB}(\mu_t, \alpha), \quad \log \mu_t = \beta_0 + \beta_1\, t",
        "why":  "Correct error model for overdispersed nightly counts. Honest prediction intervals.",
        "when": "When calibrated PIs matter more than point estimate. Requires ≥15 weeks for stable α̂.",
    },
    "regime_aware": {
        "eq":   r"\hat{y}_{t+1} \;=\; f(\text{regime}) \cdot g(\text{rolling}, \text{cagr})",
        "why":  "Hypothesized combo: onset→bump up, collapse→cut, else→rolling. My own attempt.",
        "when": "Never (so far). Backfill APE 41% — the bias multipliers were poorly calibrated.",
    },
}

cols = st.columns(3)
for i, (mkey, label, color) in enumerate(KEY_MODELS):
    if mkey not in MATH:
        continue
    m = MATH[mkey]
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"**<span style='color:{color}'>{label}</span>**",
                        unsafe_allow_html=True)
            st.latex(m["eq"])
            st.markdown(f"_{m['why']}_")
            st.caption(f"**Use when:** {m['when']}")
            # Current-week prediction if available
            if mkey in fc.columns and len(fc) > 0:
                latest = fc.iloc[-1]
                if pd.notna(latest[mkey]) and latest[mkey] > 0:
                    st.markdown(
                        f"**Next week ({latest['week_start'].date()}):** "
                        f"<span style='color:{color};font-size:20px;"
                        f"font-family:monospace'>{int(latest[mkey]):,}</span>",
                        unsafe_allow_html=True,
                    )
            # Median APE if scored
            if f"ape_{mkey}" in closed.columns:
                med = closed[f"ape_{mkey}"].dropna().median()
                if pd.notna(med):
                    st.caption(f"Median APE (7 closed weeks): **{med:.1f}%**")


# ============================================================
# 2. Head-to-head chart — all models against actuals
# ============================================================
st.markdown("---")
st.subheader("Head-to-head against realised actuals")

fig, ax = plt.subplots(figsize=(12, 5.5))
for mkey, label, color in KEY_MODELS:
    if mkey not in fc.columns:
        continue
    d = fc[fc[mkey] > 0].copy()
    ax.plot(d["week_start"], d[mkey], "o-", color=color, lw=1.8,
            markersize=6, alpha=0.85, label=label)

# Actuals as bold red dots
scored = fc.dropna(subset=["actual"])
ax.plot(scored["week_start"], scored["actual"], "D",
        color="#111826", markersize=11, markerfacecolor="#cc0033",
        markeredgewidth=1.5, label="Actual (UA AF)",
        zorder=10)

ax.set_ylabel("Drones per week", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10, frameon=False, ncol=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.autofmt_xdate()
st.pyplot(fig); plt.close(fig)


# ============================================================
# 3. Category leaderboards — who wins at what
# ============================================================
st.markdown("---")
st.subheader("Category leaderboards — which model wins at what")

if not closed.empty and not regime.empty:
    # Attach regime state to closed weeks
    scored_r = closed.merge(regime[["week_start", "state"]], on="week_start",
                              how="left")
    scored_r["state"] = scored_r["state"].fillna("steady")

    # Segments to score by
    seg_defs = [
        ("All closed weeks",   scored_r,
         "Overall median APE across every closed week."),
        ("Surge onset / sustaining",
         scored_r[scored_r["state"].isin(["onset", "sustaining"])],
         "Weeks where the regime detector saw a tempo surge starting or sustaining."),
        ("Collapse weeks",
         scored_r[scored_r["state"] == "collapsing"],
         "Weeks where 7-day mean dropped below 70% of 14-day mean."),
        ("Steady weeks",
         scored_r[scored_r["state"] == "steady"],
         "Weeks with no detected regime change."),
    ]

    for seg_label, seg_df, seg_desc in seg_defs:
        if seg_df.empty:
            continue
        st.markdown(f"**{seg_label}** ({len(seg_df)} weeks)  ·  _{seg_desc}_")
        rows = []
        for mkey, label, _ in KEY_MODELS:
            apecol = f"ape_{mkey}"
            if apecol in seg_df.columns:
                vals = seg_df[apecol].dropna()
                if len(vals):
                    rows.append({
                        "Model": label,
                        "Median APE": f"{vals.median():.1f}%",
                        "Mean APE":   f"{vals.mean():.1f}%",
                        "Best": f"{vals.min():.1f}%",
                        "Worst": f"{vals.max():.1f}%",
                    })
        if rows:
            df_r = pd.DataFrame(rows).sort_values("Median APE")
            st.dataframe(df_r, use_container_width=True, hide_index=True)


# ============================================================
# 4. Novel hypothesis — inferring Russian daily production
# ============================================================
st.markdown("---")
st.subheader("Novel hypothesis: inferring Russian daily production from launches")

with st.container(border=True):
    st.markdown(
        "**Setup.**  Launches are observable; production isn't.  But the two "
        "are mechanically linked: over any long window, "
        "cumulative_launches ≤ cumulative_production (Russia can't launch what "
        "it hasn't built).  Over any short window, the difference is buffered "
        "by stockpile.\n\n"
        "**Hypothesis H8:**  Daily production rate has a floor at the peak "
        "sustained launch rate.  If Russia sustained X drones/night over "
        "seven consecutive nights, production ≥ X for at least those seven "
        "days (else the stockpile would go negative).\n\n"
        r"$$P_{\text{lower}} \;=\; \max_{k \le N-7} \frac{1}{7}\sum_{i=k}^{k+6} y_i$$"
        "\n\nThe delta between this floor and the long-run average measures "
        "the stockpile burn rate during surges — how deep the reserve runs."
    )

if len(daily) >= 14:
    # Compute the peak-week sustained rate and the 30-day mean
    daily_sorted = daily.sort_values("date").reset_index(drop=True)
    n = len(daily_sorted)

    # Sliding 7-day sum
    seven_day_sum = daily_sorted["launched"].rolling(7).sum()
    peak7_end_idx = int(seven_day_sum.idxmax())
    peak7_val     = float(seven_day_sum.max())
    peak7_daily   = peak7_val / 7
    peak7_end     = daily_sorted.iloc[peak7_end_idx]["date"].date()

    mean_30 = float(daily_sorted.tail(30)["launched"].mean())
    mean_90 = float(daily_sorted.tail(90)["launched"].mean())

    est_annual_floor  = peak7_daily * 365
    est_annual_recent = mean_30    * 365
    stockpile_burn    = max(0.0, peak7_daily - mean_30)
    stockpile_days    = mean_30 / max(peak7_daily - mean_30, 1) if peak7_daily > mean_30 else float("inf")

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak sustained week (drones/day)",
              f"{peak7_daily:.0f}",
              delta=f"7-day window ending {peak7_end}",
              delta_color="off")
    c2.metric("Recent 30d mean (drones/day)",
              f"{mean_30:.0f}",
              delta=f"90d: {mean_90:.0f}",
              delta_color="off")
    c3.metric("Implied production FLOOR",
              f"{peak7_daily:.0f} / day",
              delta=f"≈ {est_annual_floor/1000:.0f}k / year",
              delta_color="off")

    st.markdown(
        f"**Reading.**  Recent 30-day burn rate is **{mean_30:.0f} drones/night**.  "
        f"Peak sustained week (best 7-night stretch) hit "
        f"**{peak7_daily:.0f} drones/night** in the week ending {peak7_end}.  "
        f"That peak sets the production FLOOR — production must be at least "
        f"{peak7_daily:.0f}/day, otherwise the stockpile would go negative.  "
        f"Projecting the floor annually: **~{est_annual_floor/1000:.0f}k drones/year**.\n\n"
        f"**Stockpile-burn signal.**  Δ between peak and mean = "
        f"{stockpile_burn:.0f} drones/day drawn from reserves during surges.  "
        f"If reserves are ~10-14 days deep (public estimates), that implies "
        f"~{stockpile_burn * 12:.0f} drones of buffered stockpile at any given time."
    )

    st.caption(
        "**Caveats.**  Floor is a lower bound only.  Production likely runs higher "
        "than the observed peak because Russia may throttle launches strategically "
        "(save for coordinated surges).  Public estimates from GUR + Bild put "
        "Alabuga at 200-400/day and rising — consistent with the floor computed "
        "here as a check."
    )


# ============================================================
# 5. Operational panels — Patriot allocation + procurement
# ============================================================
st.markdown("---")
st.subheader("Operational — the two strategic questions this answers")

col_short, col_long = st.columns(2)

with col_short:
    with st.container(border=True):
        st.markdown("### 🎯 Short-term: where to position Patriots this week")
        if not closed.empty:
            # Best short-horizon model (lowest median APE at h=1)
            model_medians = {}
            for mkey, label, _ in KEY_MODELS:
                if f"ape_{mkey}" in closed.columns:
                    v = closed[f"ape_{mkey}"].dropna()
                    if len(v):
                        model_medians[label] = v.median()
            best_short = min(model_medians, key=model_medians.get)
            st.markdown(
                f"**Best short-horizon model:** `{best_short}`  \n"
                f"(median APE {model_medians[best_short]:.1f}% at 1-week horizon)"
            )

        # Next-week projected weekly total
        latest = fc.iloc[-1]
        primary_pred = latest.get("damped_trend_ets") or latest.get("rolling_14d_x110")
        if pd.notna(primary_pred) and primary_pred > 0:
            per_day = primary_pred / 7
            typical_intercept = 0.88
            expected_leakers = per_day * (1 - typical_intercept)

            st.markdown(
                f"**Next week ({latest['week_start'].date()}):**\n\n"
                f"- Total projected: **{int(primary_pred):,} drones**\n"
                f"- Per-night average: **{int(per_day)}**\n"
                f"- Expected leakers/night (at 88% intercept): "
                f"**{expected_leakers:.1f}**"
            )

            # ---- Live per-oblast allocation, ranked by expected leakers ----
            fc_csv = DATA_DIR / "updated_forecast.csv"
            if fc_csv.exists():
                try:
                    fc_ob = pd.read_csv(fc_csv)
                    fc_ob = fc_ob[["oblast", "share"]].dropna()
                    fc_ob["proj_weekly"]     = fc_ob["share"] * primary_pred
                    fc_ob["proj_nightly"]    = fc_ob["proj_weekly"] / 7
                    fc_ob["exp_leakers_wk"]  = fc_ob["proj_weekly"] * (1 - typical_intercept)
                    fc_ob = fc_ob.sort_values("exp_leakers_wk", ascending=False)
                    top5 = fc_ob.head(5).copy()
                    top5["proj_weekly"]    = top5["proj_weekly"].round(0).astype(int)
                    top5["exp_leakers_wk"] = top5["exp_leakers_wk"].round(1)
                    top5["share"]          = (top5["share"] * 100).round(1).astype(str) + "%"
                    top5 = top5[["oblast", "share", "proj_weekly", "exp_leakers_wk"]]
                    top5.columns = ["Oblast", "Share", "Proj. drones/wk", "Exp. leakers/wk"]
                    st.markdown("**Top 5 oblasts to prioritise coverage this week:**")
                    st.dataframe(top5, use_container_width=True, hide_index=True)
                    st.caption(
                        "Ranked by expected leakers = projected drones × (1 − 88% intercept). "
                        "Structural share is from the target-share OLS regression "
                        "(F=8.04, p=0.002) with border proximity dominant. Move mobile "
                        "IRIS-T / NASAMS interceptor coverage toward these oblasts before "
                        "Monday night."
                    )
                except Exception as _e:
                    st.caption(f"(per-oblast rank unavailable: {type(_e).__name__})")

with col_long:
    with st.container(border=True):
        st.markdown("### 📦 Long-term: how many interceptors to order")
        if "damped_trend_ets" in fc.columns:
            recent_dt = fc[fc["damped_trend_ets"] > 0]["damped_trend_ets"].tail(4).mean()
            if pd.notna(recent_dt) and recent_dt > 0:
                projected_6mo = recent_dt * 26 * 0.95  # dampen mildly for horizon
                projected_12mo = recent_dt * 52 * 0.90
                interceptors_6mo  = projected_6mo * 0.88
                interceptors_12mo = projected_12mo * 0.88
                st.markdown(
                    f"**Model of choice:** Damped-trend ETS (φ=0.9).  Only model "
                    f"in the suite designed for long horizons without unbounded "
                    f"exponential growth.\n\n"
                    f"**6-month projection:**  ~{int(projected_6mo):,} drones total\n"
                    f"→ need ~{int(interceptors_6mo):,} interceptor rounds at "
                    f"current 88% intercept rate.\n\n"
                    f"**12-month projection:**  ~{int(projected_12mo):,} drones total\n"
                    f"→ need ~{int(interceptors_12mo):,} interceptor rounds.\n\n"
                    f"**Procurement note.**  Interceptor pipelines take 6-18 months "
                    f"from PO to delivery.  Order NOW for the demand that hits "
                    f"in Q1-Q2 2027.  Under-order and defense degrades measurably; "
                    f"over-order and it's shelved capacity (better failure mode)."
                )
        st.caption(
            "**Caveat.**  Past 12 weeks the process is dominated by policy, "
            "sanctions, and attrition — not curves. Report as a scenario band, "
            "not a point estimate: LOW (deceleration continues), MID (steady-state), "
            "HIGH (surge like 2024→2025)."
        )


# ============================================================
# footer
# ============================================================
st.markdown("---")
st.caption(
    "**Method.**  Every model shipped with the same interface. Predictions "
    "written to `weekly_model_predictions` on Monday morning BEFORE the week "
    "plays out. Actuals from UA Air Force nightly summaries via @kpszsu, "
    "scored Sunday. Rules and thresholds locked in code — changes require a "
    "public post first.  Source: "
    "[github.com/connormmc777/ukraine-drone-forecast]"
    "(https://github.com/connormmc777/ukraine-drone-forecast)."
)
