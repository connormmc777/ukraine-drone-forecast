"""
Salients + Rail-Logistics Limits.

The question
------------
WW2 had the Kiev Bulge (Soviet, 1943) and the Battle of the Bulge
(German, 1944) — pronounced wedges driven into opposing lines, whose
fate was decided largely by logistics distance.

Does the Russia-Ukraine line have the same?  Where are the salients
now, and how far has each tip pushed past its own supply railhead?

The hypothesis
--------------
H1. **Russian 90-km rail-dependency thesis** (Watling / Kofman).
    Russian ground forces are rail-sustained.  Beyond ~90 km from an
    operational railhead, truck sustainment caps throughput and the
    advance rate drops.  Prediction: salient depth tracks railhead
    distance.

H2. **Symmetric Ukrainian dependency**.  Ukrainian defensive
    strongpoints within ~20 km of a Russian railhead fall faster
    than equivalent positions >60 km away (artillery sustainment).

H3. **Rail activation accelerates advance**.  When Russia captures
    and re-gauges a rail line, advance rate in that sector increases
    2-3 months later.

Sources: ISW campaign maps (Nov 2026), RUSI (Watling, 2023-25),
CSIS Russia Military Reports, OpenStreetMap rail layer, public
CRS reporting on Russian military logistics doctrine.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

APP_DIR  = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Salients & Rail Logistics",
                    page_icon="⚔️", layout="wide")


# ---------- Geometry ----------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


# ---------- Data ----------
@st.cache_data(ttl=120)
def load_salients():
    p = DATA_DIR / "salients.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=120)
def load_railheads():
    p = DATA_DIR / "railheads.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_oblasts():
    p = DATA_DIR / "ukraine_oblasts.geojson"
    if not p.exists(): return None
    return json.loads(p.read_text())


sal     = load_salients()
rails   = load_railheads()
oblasts = load_oblasts()


st.title("⚔️ Salients & Rail Logistics")
st.markdown(
    "**Does the Russia-Ukraine line have salients like the Kiev Bulge "
    "or the Battle of the Bulge?  How far has each one pushed past "
    "its own supply railhead?**  Russia's doctrine is to sustain "
    "ground forces within ~90 km of a railhead (Watling/RUSI).  "
    "Past that, trucks take over and throughput collapses."
)

if sal.empty or rails.empty:
    st.warning("Missing data files — check drone_app/data/salients.csv "
               "and railheads.csv")
    st.stop()


# ---------- Compute distances for each salient ----------
def nearest_rail(tip_lat, tip_lon, operator):
    cand = rails[rails["operator"] == operator]
    if cand.empty: return None, None, None
    d = cand.apply(lambda r: haversine_km(tip_lat, tip_lon, r["lat"], r["lon"]),
                   axis=1)
    i = d.idxmin()
    return cand.loc[i, "name"], float(d.loc[i]), cand.loc[i]


def nearest_opposing_rail(tip_lat, tip_lon, actor):
    opp = "UA" if actor == "RU" else "RU"
    return nearest_rail(tip_lat, tip_lon, opp)


rows = []
for _, s in sal.iterrows():
    nr_name, nr_km, _ = nearest_rail(s["tip_lat"], s["tip_lon"], s["actor"])
    op_name, op_km, _ = nearest_opposing_rail(s["tip_lat"], s["tip_lon"],
                                                s["actor"])
    rows.append({
        "Salient":               s["name"],
        "Actor":                 s["actor"],
        "Depth (km)":            s["depth_km"],
        "Nearest own railhead":  nr_name,
        "Dist to own rail (km)": round(nr_km, 0) if nr_km else None,
        "Rail-sustained?":       "yes" if (nr_km and nr_km <= 90) else "stretched",
        "Nearest opp railhead":  op_name,
        "Dist to opp rail (km)": round(op_km, 0) if op_km else None,
        "Status":                s["status"],
        "Target":                s["target"],
    })
tbl = pd.DataFrame(rows)


# ---------- KPIs ----------
russian = tbl[tbl["Actor"] == "RU"]
ukr     = tbl[tbl["Actor"] == "UA"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Salients tracked", f"{len(tbl)}", f"RU: {len(russian)} · UA: {len(ukr)}")
k2.metric("Deepest RU bulge",
          f"{russian['Depth (km)'].max():.0f} km" if not russian.empty else "—",
          russian.loc[russian['Depth (km)'].idxmax(),'Salient'] if not russian.empty else None)
k3.metric("Deepest UA bulge",
          f"{ukr['Depth (km)'].max():.0f} km" if not ukr.empty else "—",
          ukr.loc[ukr['Depth (km)'].idxmax(),'Salient'] if not ukr.empty else None)
over_limit = tbl[tbl["Rail-sustained?"] == "stretched"]
k4.metric("Salients past 90-km rail limit",
          f"{len(over_limit)} / {len(tbl)}",
          "Watling thesis")


# ---------- Map ----------
st.subheader("Where the bulges are, and where the rails end")

# Build layers
rail_df = rails.copy()
rail_df["color"] = rail_df["operator"].map(
    {"RU": [214, 69, 69, 220], "UA": [42, 143, 106, 220]})
rail_df["radius"] = rail_df["capacity"].map(
    {"high": 12000, "med": 8500, "low": 5500}).fillna(6000)

# Salient arrows: base → tip
sal["color"] = sal["actor"].map(
    {"RU": [214, 69, 69, 220], "UA": [42, 143, 106, 220]})
sal_arrows = sal.rename(columns={
    "base_lat": "start_lat", "base_lon": "start_lon",
    "tip_lat":  "end_lat",   "tip_lon":  "end_lon",
})

layers = []

# 90-km outer sustainment rings (cranked alpha so they read on any basemap)
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=rail_df, get_position="[lon, lat]",
    get_radius=90000, radius_units="meters",
    get_fill_color=[255, 180, 80, 50],
    stroked=True, get_line_color=[255, 150, 50, 180],
    line_width_min_pixels=1, pickable=False,
))
# 60-km inner rings
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=rail_df, get_position="[lon, lat]",
    get_radius=60000, radius_units="meters",
    get_fill_color=[255, 220, 120, 70],
    stroked=True, get_line_color=[255, 180, 50, 220],
    line_width_min_pixels=1, pickable=False,
))

# Salient arcs (base → tip)
layers.append(pdk.Layer(
    "ArcLayer", data=sal_arrows,
    get_source_position="[start_lon, start_lat]",
    get_target_position="[end_lon, end_lat]",
    get_source_color="color", get_target_color="color",
    get_width=8, pickable=True,
))

# Railhead markers on top of rings
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=rail_df, get_position="[lon, lat]",
    get_radius="radius", get_fill_color="color",
    get_line_color=[20, 20, 20, 255], line_width_min_pixels=2,
    stroked=True, pickable=True, auto_highlight=True,
))

# Salient tip bullets (large, on top of everything)
layers.append(pdk.Layer(
    "ScatterplotLayer", data=sal,
    get_position="[tip_lon, tip_lat]",
    get_radius=14000, get_fill_color="color",
    get_line_color=[20, 20, 20, 255], line_width_min_pixels=2,
    stroked=True, pickable=True, auto_highlight=True,
))

view = pdk.ViewState(latitude=48.6, longitude=36.8, zoom=5.6, pitch=0)
tooltip = {"html": "<b>{name}</b><br/>{notes}",
           "style": {"backgroundColor": "#222", "color": "white"}}
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    tooltip=tooltip,
    map_provider="carto",
    map_style="light",  # free Carto Positron — no Mapbox token needed
)
st.pydeck_chart(deck, use_container_width=True)
st.caption(
    "🔴 red = Russian.  🟢 green = Ukrainian.  "
    "Dots = railheads (size ∝ capacity).  "
    "Orange rings = 60 km (inner) / 90 km (outer) rail sustainment "
    "envelope.  Arcs = salient axis from base to tip."
)

# ---------- Matplotlib fallback — always-visible static map ----------
st.markdown("**Static fallback (always renders):**")
import matplotlib.pyplot as plt
fig_map, axm = plt.subplots(figsize=(11, 6.5))

# Draw sustainment rings as circles (approx: 1° lat ~ 111 km; 1° lon ~ 111*cos(lat))
for _, r in rail_df.iterrows():
    c90_lon_r = 90.0 / (111.0 * math.cos(math.radians(r["lat"])))
    c90_lat_r = 90.0 / 111.0
    c60_lon_r = 60.0 / (111.0 * math.cos(math.radians(r["lat"])))
    c60_lat_r = 60.0 / 111.0
    axm.add_patch(plt.matplotlib.patches.Ellipse(
        (r["lon"], r["lat"]), 2*c90_lon_r, 2*c90_lat_r,
        facecolor="#ffd59a", edgecolor="#e07a1f", alpha=0.18, lw=0.6))
    axm.add_patch(plt.matplotlib.patches.Ellipse(
        (r["lon"], r["lat"]), 2*c60_lon_r, 2*c60_lat_r,
        facecolor="#ffc26e", edgecolor="#d05d10", alpha=0.3, lw=0.8))

# Ukraine oblast outlines (if geojson loaded)
if oblasts is not None:
    for feat in oblasts.get("features", []):
        geom = feat.get("geometry", {})
        gt = geom.get("type")
        coords_list = (geom.get("coordinates", []) if gt == "Polygon"
                       else [p for mp in geom.get("coordinates", []) for p in mp]
                       if gt == "MultiPolygon" else [])
        if gt == "Polygon": coords_list = geom["coordinates"]
        for ring in coords_list:
            try:
                xs, ys = zip(*ring)
                axm.plot(xs, ys, color="#8a8a92", lw=0.5, alpha=0.7)
            except (TypeError, ValueError):
                continue

# Railheads
for _, r in rail_df.iterrows():
    color = "#d64545" if r["operator"] == "RU" else "#2a8f6a"
    sz = {"high": 90, "med": 55, "low": 30}.get(r["capacity"], 40)
    axm.scatter(r["lon"], r["lat"], s=sz, c=color,
                edgecolors="#111", linewidths=0.6, zorder=5)
    axm.annotate(r["name"], (r["lon"], r["lat"]),
                 xytext=(4, 4), textcoords="offset points",
                 fontsize=6.5, color="#333", zorder=6)

# Salient arrows + tips
for _, s in sal.iterrows():
    color = "#d64545" if s["actor"] == "RU" else "#2a8f6a"
    axm.annotate("", xy=(s["tip_lon"], s["tip_lat"]),
                 xytext=(s["base_lon"], s["base_lat"]),
                 arrowprops=dict(arrowstyle="->", color=color, lw=2.2),
                 zorder=7)
    axm.scatter(s["tip_lon"], s["tip_lat"], s=140, c=color,
                edgecolors="#111", linewidths=1.2, zorder=8)
    axm.annotate(s["name"], (s["tip_lon"], s["tip_lat"]),
                 xytext=(6, -10), textcoords="offset points",
                 fontsize=8, color="#111", weight="bold", zorder=9)

axm.set_xlim(30.5, 41.5)
axm.set_ylim(46.0, 52.0)
axm.set_aspect(1.0 / math.cos(math.radians(48.6)))
axm.set_xlabel("Longitude")
axm.set_ylabel("Latitude")
axm.set_title("Salients & Rail Logistics — Eastern Ukraine theater "
              "(orange rings = 60 / 90 km rail-sustainment envelope)")
axm.grid(True, alpha=0.2)
fig_map.tight_layout()
st.pyplot(fig_map)
plt.close(fig_map)


# ---------- Per-salient table ----------
st.subheader("Per-salient depth vs railhead distance")
st.dataframe(tbl, use_container_width=True, hide_index=True)


# ---------- The 90-km test ----------
st.subheader("The 90-km test")

ru_tbl = tbl[tbl["Actor"] == "RU"].copy()
if not ru_tbl.empty:
    stretched = ru_tbl[ru_tbl["Rail-sustained?"] == "stretched"]
    within    = ru_tbl[ru_tbl["Rail-sustained?"] == "yes"]
    st.markdown(
        f"- **{len(within)}** of Russia's {len(ru_tbl)} active salients "
        f"are within the 90-km rail-sustainment envelope.\n"
        f"- **{len(stretched)}** are past the envelope — truck-sustained, "
        f"which Watling's analysis (RUSI 2023-25) argues caps combat "
        f"intensity at ~1/3 of rail-fed sustainment.\n"
        f"- **Deepest within-envelope RU bulge:** "
        f"{within.loc[within['Depth (km)'].idxmax(), 'Salient'] if not within.empty else '—'} "
        f"({within['Depth (km)'].max() if not within.empty else 0:.0f} km).\n"
        f"- **Pokrovsk Wedge** is the test case: its tip is "
        f"~{int(ru_tbl[ru_tbl['Salient'].str.startswith('Pokrovsk')]['Dist to own rail (km)'].iloc[0])} "
        f"km from Volnovakha — just inside the 90-km line.  That is "
        f"consistent with the slow but sustained advance observed in "
        f"the ISW campaign maps, not a stall."
    )


# ---------- Historical analogs ----------
st.subheader("Historical analogs")
st.markdown(
    """
| Campaign | Depth | Logistics outcome |
|---|---|---|
| **Soviet Kiev Bulge** (Oct–Nov 1943) | ~150 km | Soviet railheads kept up; bulge survived and widened. |
| **German Battle of the Bulge** (Dec 1944) | ~100 km | Fuel column ran out ~50 km short of Meuse; bulge collapsed. |
| **Pokrovsk Wedge** (Nov 2026) | ~65 km | Within 90 km rail envelope → advance continues, slowly. |
| **Kursk Residual** (UA, 2024-26) | ~30 km | UA railhead Sumy is ~70 km back; past 60-km inner limit, outside 90 — UA held but did not advance further. |

The pattern across 80 years is the same: when the tip pushes past
rail reach, the advance stops.  The tooling changes (rail vs truck
vs drone-suppressed convoys), the arithmetic does not.
    """
)


# ---------- Hypothesis + what would falsify it ----------
st.subheader("Hypothesis — testable")
st.markdown(
    """
**H1.** Weekly front-line movement per sector is a decreasing
function of distance-to-nearest-own-railhead, with a knee near 90 km.

- Fit: `advance_km_per_week ~ β₀ + β₁ · dist_to_rail + β₂ · (dist_to_rail > 90) + controls`
- Prediction: β₂ is strongly negative.
- Controls: season (winter frozen-ground bonus), weather, EW intensity,
  artillery density.

**H2.** Ukrainian defensive strongpoints within 20 km of a Russian
railhead have a higher weekly fall rate than those >60 km away.

**H3.** Russian capture + re-gauge of a rail line correlates with a
2-3-month-lagged acceleration of advance in that sector.

**What would falsify it.**  If the Pokrovsk axis stalls despite
being inside the 90-km envelope, OR if a Russian salient >120 km
from any railhead sustains advance for 8+ consecutive weeks, the
rail-dependency thesis would need revision.  I'll check back
weekly against the live ISW maps.
    """
)


with st.expander("Methodology / caveats"):
    st.markdown(
        """
        **Salient coordinates** are approximations from ISW campaign
        maps (November 2026).  Depth is measured from the salient base
        line (static for 6+ weeks) to the forwardmost reported tip.

        **Railheads.**  Positions from OSM + CRS reporting + public
        Russian military logistics assessments.  "Active" = has moved
        trains within the last 90 days, per public reporting.

        **Rail gauge note.**  Both sides use 1520 mm Soviet broad
        gauge.  Russia can move trains directly from Rostov → Mariupol
        → Volnovakha without re-gauging — a structural advantage the
        western-gauge militaries (Poland, NATO) would not have.

        **What this page is NOT.**  A real-time front map.  It is a
        strategic-geometry explainer.  Pixel-perfect front positions
        are on the ISW live map.

        **The 90-km number.**  Watling (RUSI) derives it from Russian
        Combined Arms Army (CAA) logistics TO&E.  Not a magic
        threshold — the knee is probably 70-110 km.  I use 90 as the
        doctrinal midpoint.
        """
    )

st.caption("Sources: ISW, RUSI, CSIS, OSM, CRS reporting · "
           f"generated {date.today().isoformat()}")
