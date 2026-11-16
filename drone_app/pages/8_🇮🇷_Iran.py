"""
Iran & proxy launch analytics.

Scope:  Iran direct + Houthi + Iraqi militias + (pre-ceasefire) Hezbollah.
What this page answers from PUBLIC data only:
  1. What has Iran + proxies launched, from where, when, against whom?
  2. What's the munition-mix trajectory?  (UAV vs cruise vs ballistic)
  3. What intercept rate are we seeing per munition class?
  4. US interceptor burn — cumulative + per-attack.
  5. Time-to-depletion at current burn — scenario band.

Sources: CRS, GAO, USNI News, CSIS, Reuters, MDA, Rafael/IAI public
statements, CENTCOM press releases.  Every number is a PUBLIC ESTIMATE.
Real DoD inventory is classified; this is the bounded-arithmetic
version — same spirit as the drone-production floor on the main site.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR  = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Iran & Proxy Burn Rate",
                    page_icon="🇮🇷", layout="wide")


# ---------- Data ----------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(ttl=120)
def load_events(mtime: float) -> pd.DataFrame:
    p = DATA_DIR / "iran_launch_events.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df["total_launched"]   = df["uav_count"] + df["cruise_count"] + df["ballistic_count"]
    df["total_leakers"]    = (df["total_launched"] * (1 - df["intercept_pct"]/100)).round(0)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=120)
def load_stockpile(mtime: float) -> pd.DataFrame:
    p = DATA_DIR / "interceptor_stockpile.csv"
    if not p.exists(): return pd.DataFrame()
    return pd.read_csv(p)


events = load_events(_mtime(DATA_DIR / "iran_launch_events.csv"))
stock  = load_stockpile(_mtime(DATA_DIR / "interceptor_stockpile.csv"))


# ---------- Header ----------
st.title("🇮🇷 Iran & Proxy Launch Burn Rate")
st.markdown(
    "**How much has Iran + its proxies launched, how much have we "
    "intercepted, and at current rates — when do we run out?**\n\n"
    "All numbers are from public reporting (CRS, GAO, USNI, CSIS, "
    "Reuters, MDA, CENTCOM).  Real DoD inventory is classified; this "
    "is the bounded-arithmetic version — same approach as the Russian "
    "drone production floor on the main site."
)

if events.empty or stock.empty:
    st.warning("Data files missing — check drone_app/data/ for iran_launch_events.csv and interceptor_stockpile.csv")
    st.stop()


# ---------- KPIs ----------
total_launched_all = int(events["total_launched"].sum())
total_uav          = int(events["uav_count"].sum())
total_cruise       = int(events["cruise_count"].sum())
total_ballistic    = int(events["ballistic_count"].sum())
total_leakers      = int(events["total_leakers"].sum())
us_cumulative      = int(events["us_interceptors_fired"].sum())
wavg_intercept = (
    (events["total_launched"] * events["intercept_pct"]).sum()
    / max(events["total_launched"].sum(), 1)
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total launched (all events)", f"{total_launched_all:,}")
c2.metric("UAV", f"{total_uav:,}")
c3.metric("Cruise", f"{total_cruise:,}")
c4.metric("Ballistic", f"{total_ballistic:,}")
c5.metric("Weighted intercept rate", f"{wavg_intercept:.1f}%",
          delta=f"{total_leakers:,} leakers")


# ---------- Launch events timeline ----------
st.markdown("---")
st.subheader("Launch events timeline")
ev_show = events[[
    "date","actor","campaign","target_zone",
    "uav_count","cruise_count","ballistic_count","total_launched",
    "intercept_pct","total_leakers","us_interceptors_fired","sources",
]].copy()
ev_show["date"] = ev_show["date"].dt.date
ev_show.columns = [
    "Date","Actor","Campaign","Target zone",
    "UAV","Cruise","Ballistic","Total",
    "Intercept %","Leakers","US interceptors fired","Sources",
]
st.dataframe(ev_show, use_container_width=True, hide_index=True)


# ---------- Flow map: origins → targets ----------
st.markdown("---")
st.subheader("Launch flow — origins to targets")

flow = events.copy()
flow["weight"] = flow["total_launched"]

arc_df = flow.rename(columns={
    "origin_lat":"src_lat","origin_lon":"src_lon",
    "target_lat":"dst_lat","target_lon":"dst_lon",
})
arc_df["width"] = np.clip(np.log1p(arc_df["weight"]) * 2, 1, 10)

arcs = pdk.Layer(
    "ArcLayer",
    data=arc_df,
    get_source_position="[src_lon, src_lat]",
    get_target_position="[dst_lon, dst_lat]",
    get_source_color="[200, 30, 30, 160]",
    get_target_color="[30, 50, 180, 200]",
    get_width="width",
    pickable=True,
    auto_highlight=True,
)

origin_pts = flow.groupby(
    ["origin_country","origin_lat","origin_lon"], as_index=False
)["total_launched"].sum()
origin_pts["radius"] = np.clip(origin_pts["total_launched"] * 50, 15000, 150000)
origin_layer = pdk.Layer(
    "ScatterplotLayer",
    data=origin_pts.rename(columns={"origin_lat":"latitude","origin_lon":"longitude"}),
    get_position="[longitude, latitude]",
    get_radius="radius",
    get_fill_color="[200, 30, 30, 100]",
    get_line_color="[200, 30, 30, 220]",
    line_width_min_pixels=1,
    pickable=True,
)

target_pts = flow.groupby(
    ["target_zone","target_lat","target_lon"], as_index=False
)["total_launched"].sum()
target_pts["radius"] = np.clip(target_pts["total_launched"] * 50, 15000, 150000)
target_layer = pdk.Layer(
    "ScatterplotLayer",
    data=target_pts.rename(columns={"target_lat":"latitude","target_lon":"longitude"}),
    get_position="[longitude, latitude]",
    get_radius="radius",
    get_fill_color="[30, 50, 180, 100]",
    get_line_color="[30, 50, 180, 220]",
    line_width_min_pixels=1,
    pickable=True,
)

st.pydeck_chart(pdk.Deck(
    layers=[arcs, origin_layer, target_layer],
    initial_view_state=pdk.ViewState(latitude=28.0, longitude=42.0, zoom=3.6, pitch=25),
    tooltip={"html":
        "<b>{origin_country}</b> → <b>{target_zone}</b><br>"
        "Total: {total_launched}<br>"
        "UAV: {uav_count}  Cruise: {cruise_count}  Ballistic: {ballistic_count}<br>"
        "Intercept: {intercept_pct}%"
    },
    map_style=None,
))
st.caption("Red circles = launch origins.  Blue circles = target zones.  "
             "Arc width ∝ log(total munitions).  Hover for per-event detail.")


# ---------- Munition-class intercept rates ----------
st.markdown("---")
st.subheader("Intercept rate by munition class")

# Weighted intercept per class
def _wavg_intercept(df, col):
    denom = df[col].sum()
    if denom == 0: return 0
    # Approximation: same event-level intercept rate applied proportionally
    return (df[col] * df["intercept_pct"]).sum() / denom

mclass = pd.DataFrame({
    "Munition class": ["UAV (Shahed-class)", "Cruise (land-attack)", "Ballistic"],
    "Total launched": [total_uav, total_cruise, total_ballistic],
    "Weighted intercept %": [
        round(_wavg_intercept(events, "uav_count"), 1),
        round(_wavg_intercept(events, "cruise_count"), 1),
        round(_wavg_intercept(events, "ballistic_count"), 1),
    ],
})
mclass["Leakers"] = (mclass["Total launched"] *
                      (1 - mclass["Weighted intercept %"]/100)).round(0).astype(int)
st.dataframe(mclass, use_container_width=True, hide_index=True)
st.caption(
    "Weighted by event: a 200-missile attack intercepted at 80% contributes "
    "more than a 20-missile attack intercepted at 95%.  Ballistic intercept "
    "is the hardest problem and the costliest interceptor shot."
)


# ---------- Interceptor stockpile ----------
st.markdown("---")
st.subheader("US + Israeli interceptor inventory (public estimates)")
st.caption("From CRS RL33745, GAO-24 reports, USNI News, MDA/Lockheed/Rafael "
            "public statements.  Real DoD inventory is classified; numbers below "
            "are bounded estimates from unclassified reporting.")

stk = stock.copy()
stk["Value of inventory (USD M)"] = (stk["inventory_2024_est"] * stk["unit_cost_musd"]).round(0)
stk_show = stk[[
    "interceptor","role","unit_cost_musd","inventory_2024_est",
    "production_per_yr_est","us_fired_cumulative_est","Value of inventory (USD M)","sources",
]].copy()
stk_show.columns = [
    "Interceptor","Role","Unit cost (USD M)","Inventory (est. 2024)",
    "Production /yr","Fired to date","Inventory value (USD M)","Sources",
]
st.dataframe(stk_show, use_container_width=True, hide_index=True)


# ---------- Burn rate + time-to-depletion ----------
st.markdown("---")
st.subheader("Burn rate & time-to-depletion")

# Annualise burn rate from events window
window_days = (events["date"].max() - events["date"].min()).days or 1
burn_per_yr_total = events["us_interceptors_fired"].sum() * 365 / window_days

st.markdown(
    f"**Cumulative US interceptors fired across all events in log:** "
    f"{us_cumulative}  (across {window_days} days = "
    f"~{burn_per_yr_total:.0f}/yr annualised at current pace)"
)

# Per-interceptor scenarios: current, surge (2×), peace (0.3×)
scenarios = {
    "Peace scenario (0.3× current)": 0.3,
    "Current pace":                   1.0,
    "Surge scenario (2× current)":    2.0,
}
rows = []
for _, s in stock.iterrows():
    curr_inv       = max(s["inventory_2024_est"] - s["us_fired_cumulative_est"], 0)
    annual_burn    = s["us_fired_cumulative_est"] * 365 / window_days
    annual_prod    = s["production_per_yr_est"]
    for scen_name, mult in scenarios.items():
        burn = annual_burn * mult
        net  = burn - annual_prod
        if net <= 0:
            months_to_zero = "∞ (production ≥ burn)"
        else:
            months_to_zero = f"{curr_inv / net * 12:.0f}"
        rows.append({
            "Interceptor":  s["interceptor"],
            "Scenario":     scen_name,
            "Current inv":  int(curr_inv),
            "Burn/yr":      int(round(burn)),
            "Prod/yr":      int(annual_prod),
            "Net drain/yr": int(round(net)) if net > 0 else 0,
            "Months to zero": months_to_zero,
        })
proj = pd.DataFrame(rows)
# Highlight the current pace column
st.dataframe(proj, use_container_width=True, hide_index=True)

st.caption(
    "**Method.**  Annual burn = (events in log × US interceptors fired) × 365 / window days.  "
    "Scenarios scale that forward.  Net drain = burn − production.  "
    "Months-to-zero = current inventory / net drain × 12.  "
    "Scenario interpretation: peace ≈ Houthi tempo only; current ≈ 2024-2026 running pace; "
    "surge ≈ sustained direct Iran exchange like June 2025."
)


# ---------- Depletion curve chart ----------
st.markdown("---")
st.subheader("Depletion curve — 60-month projection at current pace")

fig, ax = plt.subplots(figsize=(11, 5.5))
months = np.arange(0, 61)
colours = {"SM-3 Block IB":"#003d7a","SM-3 Block IIA":"#7a1727",
           "SM-6":"#0a6d3a","THAAD":"#8a5f00",
           "Patriot PAC-3 MSE":"#c96f00","Arrow 3 (IL)":"#5c6472",
           "David's Sling Stunner (IL)":"#a13e97","Iron Dome Tamir (IL)":"#999999"}
for _, s in stock.iterrows():
    inv0      = max(s["inventory_2024_est"] - s["us_fired_cumulative_est"], 0)
    annual_burn = s["us_fired_cumulative_est"] * 365 / window_days
    net       = annual_burn - s["production_per_yr_est"]
    series    = inv0 - (net / 12.0) * months
    series    = np.clip(series, 0, None)
    col       = colours.get(s["interceptor"], "#444")
    ax.plot(months, series, "-", color=col, lw=2, label=s["interceptor"])
    # Mark month-to-zero if it happens within window
    if net > 0:
        mz = inv0 / net * 12
        if mz <= 60:
            ax.axvline(mz, color=col, ls=":", alpha=0.4)

ax.set_xlabel("Months from now", fontsize=11)
ax.set_ylabel("Inventory remaining", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xlim(0, 60)
st.pyplot(fig); plt.close(fig)


# ---------- Projection: what might Iran launch next year? ----------
st.markdown("---")
st.subheader("Projected Iran + proxy launches — next 12 months")

# Simple base rate × scenario forecast
window_days_hist = window_days
annual_rate = total_launched_all * 365 / window_days_hist
proj_12mo = pd.DataFrame({
    "Scenario": ["Peace (0.3× hist pace)",
                   "Current pace",
                   "Surge (2× hist pace)",
                   "Full regional war (3× hist pace)"],
    "Mult":       [0.3, 1.0, 2.0, 3.0],
})
proj_12mo["Projected launches (12mo)"] = (proj_12mo["Mult"] * annual_rate).round(0).astype(int)
proj_12mo["Expected leakers (at current intercept%)"] = (
    proj_12mo["Projected launches (12mo)"] * (1 - wavg_intercept/100)
).round(0).astype(int)
proj_12mo["Est. US interceptors needed"] = (
    proj_12mo["Projected launches (12mo)"] *
    (us_cumulative / max(total_launched_all, 1))
).round(0).astype(int)
st.dataframe(proj_12mo.drop(columns=["Mult"]),
              use_container_width=True, hide_index=True)

st.caption(
    "**Projection method.**  Annualise the historical launch total over the log "
    "window, multiply by scenario.  Leakers use the current weighted intercept rate.  "
    "US interceptor estimate = projected launches × historical interceptor/launch ratio.  "
    "These are ballpark estimates — real planning should use classified inventory + "
    "probabilistic engagement models (Bayes w/ Poisson arrival + raid-size distribution)."
)


# ---------- Caveats ----------
st.markdown("---")
st.subheader("What this analysis can and cannot show")
st.markdown("""
**Can show (from public data):**
- Launch events timeline with munition breakdown.
- Weighted intercept rates by class.
- First-order burn-rate extrapolation.
- First-order time-to-depletion under scenario assumptions.

**Cannot show:**
- Real DoD inventory — it is classified.  Inventory estimates are
  from CRS/GAO/USNI reporting bands.
- Production ramp effects — PAC-3 MSE production is ramping toward
  650/yr by 2027; our constant-rate model is conservative.
- Covert transfers — Arrow 3 / SM-3 inventory shifts between
  theatres aren't visible to OSINT.
- Classified intercept-rate telemetry — Pentagon's PK (probability-
  of-kill) numbers are classified.  Public rates average over
  attack-level reports.

**The useful part.**  Even with wide uncertainty, the direction of
travel is unambiguous: SM-3 Block IIA (18 USD M/shot) and THAAD burn
far faster than Lockheed can produce, in any scenario where regional
tensions sustain the 2024-2026 pace.  Patriot PAC-3 MSE is the only
interceptor where production meaningfully exceeds burn in the
peace-and-current scenarios — but it is also the one where Ukraine
demand competes with Israel/CENTCOM demand.

**How to use this on the ops side:**
- **Reserve ratio alarm** — track inventory / annual-burn.  When it
  drops below ~2 years, lead times from order to delivery exceed
  stock depletion.  That's the real red line.
- **Munition-class triage** — ballistic interceptors are the
  scarcest asset.  Cheap-kill alternatives (laser, SHORAD, Iron Dome
  for short-range) preserve exquisite stock for actual strategic
  threats.
- **Production-ramp lead** — any OFAC action or production-line
  disruption on the adversary side propagates to launch counts
  with a lag (~2 weeks for Shahed, longer for ballistics).  The
  same lag applies in reverse to our production.
""")

st.markdown("---")
st.caption(
    "**Sources.**  Launch events: USNI News, CENTCOM press, IDF Spokesperson, "
    "Reuters, ISW.  Interceptors: CRS RL33745, GAO-24 reports, MDA public, "
    "Lockheed Martin + Rafael + IAI public filings.  "
    "Update data in `drone_app/data/iran_launch_events.csv` and "
    "`drone_app/data/interceptor_stockpile.csv` as new events are reported.  "
    "Source: [github.com/connormmc777/ukraine-drone-forecast]"
    "(https://github.com/connormmc777/ukraine-drone-forecast)."
)
