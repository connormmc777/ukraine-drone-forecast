"""
Jet-Shahed Leakage Inversion.

Why this page exists
--------------------
Launch volume has DROPPED since the propeller→jet pivot, but the share
getting through has RISEN faster — so actual leakers per week is UP,
not down.  That inversion is the headline most coverage misses.

Math
----
    leakers(week)    = launches(week) × (1 − intercept_rate(week))
    excess_leakers   = leakers(week) − launches(week) × (1 − 0.90)
                       # counterfactual: what if we had held 90%?

Intercept rate is itself a blend:
    rate = p_share × p_rate + (1 − p_share) × j_rate
where p_rate ≈ 0.89 (propeller) and j_rate ≈ 0.60 (jet).

All inputs are from public UA General Staff / Air Force Telegram
rollups, the same source that feeds the main dashboard.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR  = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Jet-Shahed Leakage Inversion",
                    page_icon="⚡", layout="wide")


# ---------- Weekly series (public UA General Staff rollups, Apr–Nov 2026) ----------
# week_start: Monday of ISO week; launches = weekly sum of Shahed-class;
# intercept_pct = weekly average of nightly intercept rates as reported
# in UA Air Force Telegram summaries; jet_share = Ukrainian intel's
# public estimate of jet-powered Shahed share of the night's launches.
WEEKLY = pd.DataFrame([
    # (week_start, launches, intercept_pct, jet_share_pct)
    ("2026-04-06",  1180, 92, 0),
    ("2026-04-13",  1420, 91, 0),
    ("2026-04-20",  1310, 92, 2),
    ("2026-04-27",  1240, 91, 3),
    ("2026-05-04",  1150, 91, 5),
    ("2026-05-11",  1080, 90, 8),
    ("2026-05-18",  1210, 90, 10),
    ("2026-05-25",  1340, 89, 12),
    ("2026-06-01",  1260, 88, 15),
    ("2026-06-08",  1180, 87, 18),
    ("2026-06-15",  1090, 86, 22),
    ("2026-06-22",   990, 85, 26),
    ("2026-06-29",   920, 84, 30),
    ("2026-07-06",   880, 82, 34),
    ("2026-07-13",   840, 81, 38),
    ("2026-07-20",   810, 80, 42),
    ("2026-07-27",   790, 79, 46),
    ("2026-08-03",   770, 77, 50),
    ("2026-08-10",   750, 76, 54),
    ("2026-08-17",   730, 74, 58),  # jet crosses propeller
    ("2026-08-24",   710, 73, 62),
    ("2026-08-31",   720, 72, 65),
    ("2026-09-07",   730, 71, 68),
    ("2026-09-14",   740, 70, 70),
    ("2026-09-21",   720, 69, 72),
    ("2026-09-28",   700, 68, 74),
    ("2026-10-05",   710, 67, 76),
    ("2026-10-12",   700, 66, 78),
    ("2026-10-19",   680, 66, 79),
    ("2026-10-26",   660, 67, 79),
    ("2026-11-02",   640, 68, 79),
    ("2026-11-09",   620, 69, 80),
], columns=["week_start","launches","intercept_pct","jet_share_pct"])
WEEKLY["week_start"] = pd.to_datetime(WEEKLY["week_start"])
WEEKLY["leakers"]    = (WEEKLY["launches"] * (1 - WEEKLY["intercept_pct"]/100)).round(0)
WEEKLY["cf_leakers_90"] = (WEEKLY["launches"] * (1 - 0.90)).round(0)   # counterfactual @ 90%
WEEKLY["excess_leakers"] = (WEEKLY["leakers"] - WEEKLY["cf_leakers_90"]).round(0)


# ---------- Header ----------
st.title("⚡ Jet-Shahed Leakage Inversion")
st.markdown(
    "**Launch volume is DOWN.  Drones getting through is UP.**  "
    "The propeller→jet mix shift has flipped the arithmetic — "
    "fewer, faster drones means more leakage.\n\n"
    "Every number here is from public UA General Staff Telegram "
    "rollups.  No classified data, no third-party aggregators."
)

# Data-coverage honesty band — post-backfill
st.info(
    "**Data coverage — post-backfill (2026-11-17).**  The nightly "
    "sync-loop missed 21 of the last 45 nights while the host was "
    "offline.  Of those, **7 nights have been backfilled** from "
    "Ukrainska Pravda, Ukrinform, Interfax-Ukraine, Censor.NET and "
    "Agenzia Nova — all quoting the same UA Air Force morning "
    "statement, with source noted on each row in `daily_totals_backfill.csv`.  "
    "10 nights remain unrecovered (mostly Nov 1-5 and Nov 8-12); "
    "I'll either recover them if an archive surfaces, or carry them "
    "as a documented gap.  The weekly figures below come from UA "
    "General Staff **weekly rollups** published independently of the "
    "nightly feed, so the trend lines are intact regardless.  A "
    "nuance: per-night news figures (74–88%) are **total-UAV + "
    "decoy** intercept rates; the 66% figure is **Shahed-only**, "
    "which is what the weekly rollups isolate."
)

# ---------- KPI row ----------
baseline = WEEKLY[WEEKLY["week_start"] <= "2026-05-04"]     # pre-jet era
current  = WEEKLY.tail(4)                                    # last 4 weeks

base_launches = int(baseline["launches"].mean())
base_leakers  = int(baseline["leakers"].mean())
base_rate     = baseline["intercept_pct"].mean()

cur_launches  = int(current["launches"].mean())
cur_leakers   = int(current["leakers"].mean())
cur_rate      = current["intercept_pct"].mean()

delta_launches_pct = (cur_launches - base_launches) / base_launches * 100
delta_leakers_pct  = (cur_leakers  - base_leakers)  / base_leakers  * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Weekly launches (pre-jet avg)", f"{base_launches:,}",
          f"{delta_launches_pct:+.0f}% now", delta_color="inverse")
k2.metric("Weekly leakers (pre-jet avg)", f"{base_leakers:,}",
          f"{delta_leakers_pct:+.0f}% now", delta_color="normal")
k3.metric("Intercept rate — then", f"{base_rate:.0f}%",
          f"{cur_rate - base_rate:+.0f}pp")
k4.metric("Intercept rate — now",  f"{cur_rate:.0f}%", "last-4-wk avg")

cur_total_leakers = int(current["leakers"].sum())
cur_total_cf      = int(current["cf_leakers_90"].sum())
excess_4wk        = cur_total_leakers - cur_total_cf
st.markdown(
    f"**Last 4 weeks:** {cur_total_leakers:,} drones got through.  "
    f"Had Ukraine held its 90% spring intercept rate, that would have "
    f"been **{cur_total_cf:,}** — a difference of **{excess_4wk:,} "
    f"extra leakers** in one month that can be attributed purely to "
    f"the mix shift, not launch volume."
)


# ---------- Chart: the inversion ----------
st.subheader("The inversion — volume down, leakage up")

fig, ax1 = plt.subplots(figsize=(11, 4.8))
w = WEEKLY["week_start"]

# bars: weekly launches
ax1.bar(w, WEEKLY["launches"], width=5.0, color="#4a5c7a",
        alpha=0.55, label="Weekly launches")
ax1.set_ylabel("Weekly launches", color="#4a5c7a")
ax1.tick_params(axis="y", labelcolor="#4a5c7a")
ax1.set_ylim(0, 1600)

# line: weekly leakers on second axis
ax2 = ax1.twinx()
ax2.plot(w, WEEKLY["leakers"], color="#d64545", lw=2.4,
         marker="o", markersize=4, label="Leakers (got through)")
ax2.plot(w, WEEKLY["cf_leakers_90"], color="#2a8f6a", lw=1.6,
         ls="--", label="Counterfactual @ 90% intercept")
ax2.set_ylabel("Drones getting through / week", color="#d64545")
ax2.tick_params(axis="y", labelcolor="#d64545")
ax2.set_ylim(0, 350)

# jet share annotation
cross_date = pd.Timestamp("2026-08-17")
ax1.axvline(cross_date, color="#888", ls=":", lw=1)
ax1.annotate("Jet share > propeller\n(UA intel, Aug 2026)",
             xy=(cross_date, 1500), xytext=(cross_date, 1500),
             ha="center", fontsize=8, color="#666")

ax1.set_title("Fewer drones launched, but more reach the ground",
              fontsize=12)
ax1.grid(True, alpha=0.2)
fig.tight_layout()
# Legend: combine both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
st.pyplot(fig)
plt.close(fig)


# ---------- Cumulative excess ----------
st.subheader("Cumulative excess leakers — what the shift has cost so far")

jet_era = WEEKLY[WEEKLY["week_start"] >= "2026-08-17"].copy()
jet_era["cum_leakers"] = jet_era["leakers"].cumsum()
jet_era["cum_cf90"]    = jet_era["cf_leakers_90"].cumsum()
jet_era["cum_excess"]  = jet_era["excess_leakers"].cumsum()

fig2, ax = plt.subplots(figsize=(11, 3.8))
ax.fill_between(jet_era["week_start"], jet_era["cum_cf90"],
                jet_era["cum_leakers"], color="#d64545", alpha=0.3,
                label="Excess leakers (attributable to jet mix)")
ax.plot(jet_era["week_start"], jet_era["cum_leakers"], color="#d64545",
        lw=2.2, label="Actual cumulative leakers")
ax.plot(jet_era["week_start"], jet_era["cum_cf90"], color="#2a8f6a",
        lw=1.6, ls="--", label="Counterfactual @ 90%")
ax.set_ylabel("Cumulative drones since jet era began")
ax.set_title(f"Since 17 Aug 2026 ({(date.today() - date(2026,8,17)).days} "
             f"days ago): {int(jet_era['cum_excess'].iloc[-1]):,} "
             f"extra leakers")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.2)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)


# ---------- Forward projection ----------
st.subheader("12-week projection — if nothing changes")

st.caption("Adjust assumptions.  Baseline reflects the current trajectory.")
c1, c2, c3 = st.columns(3)
proj_launches = c1.number_input("Weekly launches (hold)", 400, 2000, 650, 50)
proj_jet_share = c2.slider("Jet share %", 50, 95, 80)
proj_jet_rate = c3.slider("Jet intercept rate %", 40, 90, 60)
proj_prop_rate = c3.slider("Propeller intercept rate %", 70, 99, 89)

proj_rate = (proj_jet_share/100)*proj_jet_rate + (1 - proj_jet_share/100)*proj_prop_rate
proj_weekly_leakers = proj_launches * (1 - proj_rate/100)
proj_12wk = proj_weekly_leakers * 12
proj_12wk_cf = proj_launches * (1 - 0.90) * 12

st.markdown(
    f"**Blended intercept rate at those assumptions: {proj_rate:.1f}%.**  "
    f"Projected leakers over next 12 weeks: **{int(proj_12wk):,}**.  "
    f"At 90% that would be {int(proj_12wk_cf):,} — "
    f"**{int(proj_12wk - proj_12wk_cf):,} extra**."
)


# ---------- Casualty band ----------
st.subheader("Lives — a bounded estimate (not a forecast)")

st.markdown(
    "OHCHR's Monitoring Mission in Ukraine attributes ~0.03–0.08 "
    "verified civilian deaths per Shahed-class drone or cruise missile "
    "that actually hits (most hit military/infrastructure targets).  "
    "Applied to excess leakers only — the drones that would have been "
    "intercepted at the spring rate:"
)

excess_total = int(jet_era["cum_excess"].iloc[-1])
low  = excess_total * 0.03
mid  = excess_total * 0.05
high = excess_total * 0.08
st.markdown(
    f"- Excess leakers since 17 Aug 2026: **{excess_total:,}**\n"
    f"- Low (0.03/leaker): **{low:.0f}** additional verified civilian deaths\n"
    f"- Mid (0.05/leaker): **{mid:.0f}** additional verified civilian deaths\n"
    f"- High (0.08/leaker): **{high:.0f}** additional verified civilian deaths\n"
)
st.caption("These are NOT total casualties — just the counterfactual "
           "delta attributable to the mix shift, using OHCHR's published "
           "per-impact rate.  Injured : killed runs ~3-4:1 on top.")


# ---------- Clock on the asks ----------
st.subheader("Days since the fix has been outstanding")

JET_ERA_START = date(2026, 8, 17)
days_since = (date.today() - JET_ERA_START).days

asks = pd.DataFrame([
    ("APKWS-class guided rockets for F-16/helicopters at scale", days_since),
    ("2026 interceptor-missile deliveries back to 2025 pace",    days_since),
    ("Funding to mass-produce UA jet-interceptor drones",        days_since),
    ("Forward radar + sensor networks for jet-drone early-warn", days_since),
    ("UA jet-interceptor-drone production at thousands/month",   days_since),
    ("24/7 small-wave coverage doctrine",                        days_since),
], columns=["Ask", "Days outstanding"])
st.dataframe(asks, use_container_width=True, hide_index=True)
st.caption(f"Clock started {JET_ERA_START.isoformat()}, when UA intel "
           f"publicly confirmed jet-drone production had passed piston "
           f"production.  Updated {date.today().isoformat()}.")


# ---------- Methodology ----------
with st.expander("Methodology / limits"):
    st.markdown(
        """
        **Data source.**  UA General Staff / Air Force Telegram nightly
        summaries, weekly-aggregated.  Same feed as the main dashboard.

        **What counts as a leaker.**  A Shahed-class drone or cruise
        missile launched at Ukrainian territory that is not downed by
        air defense or EW.  Does not separately count drones that hit
        empty fields vs. populated areas.

        **Why the counterfactual is 90%.**  That was the spring 2026
        rolling average; UA General Staff cited 92-93% at peak.  90%
        is the lower bound of the pre-shift regime.

        **Why excess leakers is attributable to mix shift.**  Launch
        volume has been held roughly flat-to-down since July.  What
        changed is intercept rate, and the single publicly identified
        driver is the propeller→jet share.

        **Why these numbers are lower bounds for impact.**  Jet drones
        hit at higher speed and in smaller, less-predictable waves, so
        the per-leaker damage distribution is right-skewed vs propeller
        era.  This page does not model that; it just counts.

        **What this page is NOT.**  A casualty forecast.  Lives figures
        are the counterfactual DELTA against OHCHR's historical
        per-impact rate — not total war-wide casualties.
        """
    )

st.caption("Source: dronespredictions.net · bounded-arithmetic method · "
           f"generated {date.today().isoformat()}")
