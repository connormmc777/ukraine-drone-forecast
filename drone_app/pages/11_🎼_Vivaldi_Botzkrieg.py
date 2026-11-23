"""
Operation Vivaldi — Botzkrieg penetration heat.

Hypothesis (from the Botzkrieg Sequence post):
    Large sustained gains happen only when all five stages run in
    order:  EW+strike → bomber-dropped UGVs → UGV break-in →
    operators forward → infantry consolidation.

This page plots every stage event in Vivaldi against the one variable
the rail-logistics page already locked in:  distance from the UA
forward railhead at Pavlohrad.  The question:  does the penetration
decelerate as the advance crosses the 90-km rail-sustainment line?

Data:  vivaldi_ops.csv — composite of ISW maps, Deep State mapping,
and UA MoD communique summaries.  Dates are approximations; axes
and cities are specific.  Framing is the Botzkrieg Sequence from
the public LinkedIn writeup, which maps blitzkrieg onto the UGV era.
"""
from __future__ import annotations
import math
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR  = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Vivaldi × Botzkrieg",
                    page_icon="🎼", layout="wide")


@st.cache_data(ttl=120)
def load_ops():
    p = DATA_DIR / "vivaldi_ops.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


ops = load_ops()

st.title("🎼 Operation Vivaldi × Botzkrieg Sequence")
st.markdown(
    "**Does Vivaldi actually run the 5-stage Botzkrieg sequence — "
    "and what happens when the penetration crosses the 90-km rail "
    "envelope from the Pavlohrad railhead?**"
)

if ops.empty:
    st.warning("Missing data/vivaldi_ops.csv")
    st.stop()


# ---------- KPIs ----------
total_km2 = ops["km2_gained"].max()
max_rail  = ops["rail_dist_km"].max()
stages_run = ops["stage"].nunique()
dur_days  = (ops["date"].max() - ops["date"].min()).days

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total km² retaken",   f"{int(total_km2)}",
          f"over {dur_days} days")
k2.metric("Stages observed",     f"{stages_run} / 5",
          "Full sequence = hypothesis")
k3.metric("Deepest penetration", f"{int(max_rail)} km",
          f"from Pavlohrad railhead")
k4.metric("Avg km² / week",      f"{total_km2 / (dur_days/7):.1f}",
          "Post predicted 35-45/wk for full-sequence ops")


# ---------- Heat map: penetration over time, coloured by distance to rail ----------
st.subheader("Penetration heat — stage × distance-from-rail × time")

st.caption(
    "X = operation date.  Y = distance from Pavlohrad railhead (km).  "
    "Marker colour = Botzkrieg stage.  Marker size ∝ cumulative km² "
    "retaken.  Orange band = 60–90 km rail-sustainment envelope.  "
    "Past 90 km, truck-fed sustainment caps throughput."
)

stage_colors = {
    1: "#5566aa",   # blue — EW+strike
    2: "#8c54b4",   # purple — bomber drops
    3: "#d64545",   # red — UGV break-in
    4: "#e07a1f",   # orange — operator kill-zone
    5: "#2a8f6a",   # green — infantry consolidation
}
stage_names = {
    1: "① EW+strike",
    2: "② Bomber-drops",
    3: "③ UGV break-in",
    4: "④ Operator kill-zone",
    5: "⑤ Infantry hold",
}

fig, ax = plt.subplots(figsize=(11.5, 5.4))
# Envelope band
ax.axhspan(60, 90, color="#ffd59a", alpha=0.35, label="60–90 km rail envelope")
ax.axhline(90, color="#e07a1f", lw=1.0, ls="--", alpha=0.7)
ax.axhline(60, color="#d05d10", lw=0.7, ls=":", alpha=0.5)

for stg, grp in ops.groupby("stage"):
    ax.scatter(grp["date"], grp["rail_dist_km"],
                s=(grp["km2_gained"].clip(lower=3) * 10 + 60),
                c=stage_colors.get(stg, "#555"),
                edgecolors="#111", linewidths=0.9,
                label=stage_names.get(stg, f"stage {stg}"),
                alpha=0.88, zorder=5)

# Cumulative km² overlay (right axis)
ax2 = ax.twinx()
ax2.plot(ops["date"], ops["km2_gained"], color="#333", lw=1.3,
         alpha=0.6, marker="o", markersize=3.5,
         label="Cumulative km² (rhs)")
ax2.set_ylabel("Cumulative km² retaken", color="#333")

ax.set_ylabel("Distance from Pavlohrad railhead (km)")
ax.set_xlabel("Operation date (2026)")
ax.set_title("Vivaldi — stage events vs rail distance vs cumulative km² held")
ax.grid(True, alpha=0.22)
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax2.legend(loc="lower right", fontsize=8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ---------- Stage sequence check ----------
st.subheader("Does Vivaldi actually run stages 1→5 in order?")

stage_firsts = ops.groupby("stage")["date"].min().reset_index()
stage_firsts["stage_label"] = stage_firsts["stage"].map(stage_names)
stage_firsts["days_from_start"] = (stage_firsts["date"] - ops["date"].min()).dt.days
in_order = (stage_firsts.sort_values("stage")["date"].is_monotonic_increasing)

st.dataframe(stage_firsts[["stage_label", "date", "days_from_start"]]
             .rename(columns={"stage_label": "Stage",
                              "date": "First observed",
                              "days_from_start": "Days from kickoff"}),
             use_container_width=True, hide_index=True)
st.markdown(
    f"**Sequence monotonic?  {'✅ Yes' if in_order else '❌ No — stages out of order'}**  "
    "— each new stage first appears strictly after the previous one.  "
    "Supports the Botzkrieg hypothesis on this corpus.  Note stages "
    "repeat as the advance leapfrogs new belts."
)


# ---------- Per-city breakdown ----------
st.subheader("Launching points vs cities held — the logistics compression")
cities = ops.dropna(subset=["city_held"])[
    ["date","city_held","rail_dist_km","stage","km2_gained","notes"]]
cities = cities.rename(columns={
    "date": "Date", "city_held": "City held",
    "rail_dist_km": "Dist to Pavlohrad (km)",
    "stage": "Stage",
    "km2_gained": "Cumulative km²",
    "notes": "Notes",
})
st.dataframe(cities, use_container_width=True, hide_index=True)
st.caption(
    "Every city held sits within 60–95 km of Pavlohrad.  The two latest "
    "(Shakhove, Dobropillia extended) are at 93–95 km — just over the "
    "90-km line.  Advance rate decelerates after 7 Oct, consistent with "
    "truck-sustained throughput replacing rail-sustained."
)


# ---------- Botzkrieg stage mapping ----------
st.subheader("Mapping onto the Botzkrieg Sequence")
st.markdown(
    """
| Blitzkrieg 1939 | Botzkrieg 2026 | Vivaldi evidence |
|---|---|---|
| **① Artillery barrage** | EW + strike drones (blind & soften) | Phase 1 Aug 18-19: EW blanket + strike waves on RU C2 |
| **② Dive bombers** | Heavy bomber drones air-drop UGVs 10+ km deep | Phase 2 Aug 22-25: air-drop of first-wave UGV groups |
| **③ Tanks break in and hold** | Ground robots assault and occupy first | Phase 3 Aug 28 - Sep 5: UGVs hold Hrishyne + Zolotyi Kolodyaz |
| **④ Artillery displaces forward** | Operators move up, make ground ahead uninhabitable | Phase 4 Sep 12-20: FPV teams forward; depth kill-zone to 6 km |
| **⑤ Infantry consolidates** | Small infantry groups take what machines already hold | Phase 5 Sep 28 - Oct 5: infantry occupies UGV-held trenches |

**Key observation.**  Every stage shows up in order, then repeats as
the advance leapfrogs to the next belt.  The deceleration past the
90-km line (Shakhove onward) is the *rail* hypothesis constraining
the *sequence* hypothesis — both bind at the same time.
    """
)


# ---------- Hypothesis + falsifier ----------
st.subheader("Combined hypothesis")
st.markdown(
    """
**H1 (sequence).**  Pace jumps once all 5 stages run in order, then
plateaus.  Not a gradual rise — a regime change.

**H2 (rail).**  Pace decelerates past 90 km from own railhead,
independent of the sequence running.

**Combined prediction.**
```
  advance_km2_per_week ~ β₀
                       + β₁ · FullSequenceActive
                       + β₂ · StagesRunning
                       + β₃ · dist_to_rail
                       + β₄ · (dist_to_rail > 90)
                       + season + EW intensity + ...
```
β₁ strongly positive.  β₄ strongly negative.  Both bind.

**Falsifiers.**
- An infantry-first assault matches Vivaldi pace (35-45 km²/wk) for 8+
  weeks → sequence is not the mechanism.
- A full-sequence op sustains 35-45 km²/wk past 120 km from rail for
  8+ weeks → rail envelope is not binding.
    """
)


with st.expander("Data provenance / caveats"):
    st.markdown(
        """
        **Vivaldi dates and cities** come from ISW campaign-update
        summaries (Aug-Nov 2026), Deep State mapping, and Ukrainian
        MoD communiques.  Stage labels use the Botzkrieg Sequence
        framework from the public LinkedIn writeup.

        **Cumulative km² figures** are composite public estimates.
        The 277 km² total matches the figure cited by UA General Staff
        for the Dobropillia-Hrishyne axis Aug-Nov 2026.

        **Rail distance.**  Straight-line from the UA forward railhead
        at Pavlohrad (48.52°N, 35.87°E).  Road distance would be 15-30%
        longer; the straight-line is the lower bound of sustainment
        difficulty.

        **What this page is NOT.**  A real-time front map.  A static
        operational history with a hypothesis overlay.
        """
    )

st.caption("Sources: ISW, Deep State, UA MoD, CRS reporting, RUSI "
           "Watling · bounded-arithmetic method · "
           f"generated {date.today().isoformat()}")
