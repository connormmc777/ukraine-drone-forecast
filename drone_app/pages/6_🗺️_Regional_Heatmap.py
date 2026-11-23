"""
Regional Heatmap — where drones are being sighted, by oblast, weighted
by recency, with movement traces between successive oblast mentions
inside multi-oblast sightings.

Data joins:
  drone_sightings.csv (posted_at, oblast, all_oblasts, text)
     ↔ updated_forecast.csv (oblast, lat, lon, share)
     ↔ snapshots table (weekly_budget for this week)

Layers rendered via Streamlit's built-in pydeck integration:
  1. HeatmapLayer   — sighting density at oblast centroids,
                      weighted by exp(-age_hours / 24)
  2. ArcLayer       — origin→destination pairs from multi-oblast
                      sightings, coloured by recency
  3. ScatterplotLayer — one dot per oblast sized by this week's
                      predicted share of the weekly budget
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

APP_DIR  = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DB       = DATA_DIR / "forecast_history.db"


st.set_page_config(
    page_title="Regional Heatmap",
    page_icon="🗺️",
    layout="wide",
)


# ---------- data ----------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(ttl=60)
def load_sightings(mtime_key: float) -> pd.DataFrame:
    p = DATA_DIR / "drone_sightings.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True, errors="coerce")
    return df.dropna(subset=["posted_at"]).sort_values("posted_at", ascending=False)


@st.cache_data(ttl=300)
def load_oblast_centroids(mtime_key: float) -> pd.DataFrame:
    fc = pd.read_csv(DATA_DIR / "updated_forecast.csv")
    return fc[["oblast", "lat", "lon", "share"]].dropna().reset_index(drop=True)


@st.cache_data(ttl=60)
def load_this_week_budget(mtime_key: float) -> tuple[int, str]:
    if not DB.exists():
        return 1000, "no db"
    with sqlite3.connect(DB) as c:
        row = c.execute(
            "SELECT weekly_budget, week_start FROM snapshots "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return (int(row[0]), row[1]) if row else (1000, "no snapshot")


sights    = load_sightings(_mtime(DATA_DIR / "drone_sightings.csv"))
centroids = load_oblast_centroids(_mtime(DATA_DIR / "updated_forecast.csv"))
this_week_budget, this_week_start = load_this_week_budget(_mtime(DB))


st.title("🗺️ Regional Heatmap")
st.markdown(
    "**Where drones are being sighted, right now, by oblast — with movement "
    "traces from multi-oblast sightings.**  Each sighting is weighted by "
    "recency (24-hour half-life).  Arcs are drawn between successive oblast "
    "mentions inside one Telegram post (e.g. \"BpLA from Kyiv Obl → Zhytomyr\")."
)

if sights.empty:
    st.warning("No sightings on file yet — check drone_sightings.csv sync.")
    st.stop()
if centroids.empty:
    st.warning("No oblast centroids in updated_forecast.csv.")
    st.stop()


# ---------- Controls ----------
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
with col_ctrl1:
    window_label = st.select_slider(
        "Time window",
        options=["last 1h", "last 6h", "last 24h", "last 7d", "last 30d", "all time"],
        value="last 24h",
    )
window_hours = {
    "last 1h": 1, "last 6h": 6, "last 24h": 24,
    "last 7d": 24 * 7, "last 30d": 24 * 30, "all time": None,
}[window_label]

with col_ctrl2:
    show_arcs = st.checkbox("Show movement arcs", value=True,
                             help="Draw arcs between oblasts co-mentioned in the same sighting.")
with col_ctrl3:
    show_predicted = st.checkbox("Show predicted markers", value=True,
                                  help="Dot size = predicted per-oblast share of this week's budget.")


# ---------- Filter sightings by time window ----------
now_utc = pd.Timestamp.utcnow()
if window_hours is not None:
    cutoff = now_utc - pd.Timedelta(hours=window_hours)
    win = sights[sights["posted_at"] >= cutoff].copy()
else:
    win = sights.copy()

win = win.merge(centroids, on="oblast", how="left")
win = win.dropna(subset=["lat", "lon"])
win["age_hours"] = (now_utc - win["posted_at"]).dt.total_seconds() / 3600.0
win["recency_weight"] = np.exp(-win["age_hours"] / 24.0)


# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sightings in window", f"{len(win):,}",
          delta=f"of {len(sights):,} total")
k2.metric("Unique oblasts hit",
          f"{win['oblast'].nunique()}")
multi = win[win["all_oblasts"].fillna("").str.contains(",")]
k3.metric("Movement events (multi-oblast posts)", f"{len(multi):,}")
if len(win):
    latest = win["posted_at"].max()
    age_min = int((now_utc - latest).total_seconds() / 60)
    k4.metric("Most recent sighting", f"{age_min}m ago",
              delta=latest.strftime("%Y-%m-%d %H:%M UTC"),
              delta_color="off")


# ---------- Build pydeck layers ----------
heatmap_data = win[["lat", "lon", "recency_weight"]].rename(
    columns={"lat": "latitude", "lon": "longitude"}
)
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=heatmap_data,
    get_position="[longitude, latitude]",
    get_weight="recency_weight",
    radius_pixels=60,
    intensity=1,
    threshold=0.03,
    color_range=[
        [255, 255, 178, 40],
        [254, 217, 118, 90],
        [254, 178, 76, 140],
        [253, 141, 60, 180],
        [240, 59, 32, 220],
        [189, 0, 38, 255],
    ],
    pickable=False,
)

arc_layer = None
if show_arcs and len(multi):
    arc_rows = []
    for _, row in multi.iterrows():
        obs = [o.strip() for o in str(row["all_oblasts"]).split(",") if o.strip()]
        if len(obs) < 2:
            continue
        src_ob = obs[0]
        for dst_ob in obs[1:]:
            src = centroids[centroids["oblast"] == src_ob]
            dst = centroids[centroids["oblast"] == dst_ob]
            if src.empty or dst.empty:
                continue
            arc_rows.append({
                "src_lat": float(src.iloc[0]["lat"]),
                "src_lon": float(src.iloc[0]["lon"]),
                "dst_lat": float(dst.iloc[0]["lat"]),
                "dst_lon": float(dst.iloc[0]["lon"]),
                "age_hours": row["age_hours"],
                "recency_weight": row["recency_weight"],
                "src_oblast": src_ob,
                "dst_oblast": dst_ob,
                "text": row["text"][:120],
            })
    if arc_rows:
        arcs_df = pd.DataFrame(arc_rows)
        arcs_df["red"]   = 220
        arcs_df["green"] = (200 * (1 - arcs_df["recency_weight"])).astype(int).clip(50, 200)
        arcs_df["blue"]  = 40
        arcs_df["alpha"] = (255 * arcs_df["recency_weight"]).astype(int).clip(30, 255)
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=arcs_df,
            get_source_position="[src_lon, src_lat]",
            get_target_position="[dst_lon, dst_lat]",
            get_source_color="[red, green, blue, alpha]",
            get_target_color="[red, green, blue, alpha]",
            get_width=2,
            pickable=True,
            auto_highlight=True,
        )

scatter_layer = None
if show_predicted:
    daily_budget = this_week_budget / 7.0
    scatter_df = centroids.copy()
    scatter_df["daily_predicted"] = scatter_df["share"] * daily_budget
    scatter_df["radius"] = (scatter_df["daily_predicted"] * 2500).clip(5000, 60000)
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_df.rename(columns={"lat": "latitude", "lon": "longitude"}),
        get_position="[longitude, latitude]",
        get_radius="radius",
        get_fill_color="[0, 61, 122, 60]",
        get_line_color="[0, 61, 122, 180]",
        line_width_min_pixels=1,
        pickable=True,
    )

layers = [l for l in [heatmap_layer, arc_layer, scatter_layer] if l is not None]

view_state = pdk.ViewState(
    latitude=49.2, longitude=32.0, zoom=5.2, pitch=35, bearing=0,
)

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip={
        "html": "<b>{oblast}</b><br>Predicted daily: {daily_predicted:.1f} drones"
                "<br>{src_oblast} → {dst_oblast}",
    },
    map_style=None,
))

st.caption(
    "**Reading the map.**  "
    "• **Heat colour** = density of sightings, weighted by recency "
    "(24-hour half-life; newer = redder).  "
    "• **Blue arcs** = successive-oblast mentions from one sighting "
    "(the drone's observed path).  Arc opacity ∝ recency.  "
    "• **Blue circles** = each oblast's predicted share × this week's "
    f"daily budget ({this_week_budget/7:.0f} drones/night); larger = more expected."
)


# ---------- Per-oblast prediction vs observed ----------
st.markdown("---")
st.subheader("Per-oblast: predicted vs observed (this window)")

daily_budget = this_week_budget / 7.0
days_in_window = window_hours / 24 if window_hours else 30

by_oblast = win.groupby("oblast").agg(
    observed_sightings=("posted_at", "count"),
    latest=("posted_at", "max"),
    weighted_intensity=("recency_weight", "sum"),
).reset_index()
by_oblast = by_oblast.merge(centroids, on="oblast", how="left")
by_oblast["predicted_drones"] = (by_oblast["share"] * daily_budget * days_in_window).round(1)
by_oblast["latest_age_hr"] = ((now_utc - by_oblast["latest"]).dt.total_seconds() / 3600).round(1)
by_oblast["obs_per_pred"] = np.where(
    by_oblast["predicted_drones"] > 0,
    (by_oblast["observed_sightings"] / by_oblast["predicted_drones"]).round(2),
    np.nan,
)
by_oblast = by_oblast.sort_values("weighted_intensity", ascending=False)

show = by_oblast[[
    "oblast", "observed_sightings", "predicted_drones", "obs_per_pred",
    "latest_age_hr", "weighted_intensity",
]].rename(columns={
    "oblast": "Oblast",
    "observed_sightings": "Sightings (window)",
    "predicted_drones": f"Predicted (over {days_in_window:.1f}d)",
    "obs_per_pred": "Obs / Pred ratio",
    "latest_age_hr": "Last sighting (hr ago)",
    "weighted_intensity": "Recency-weighted score",
})
show["Recency-weighted score"] = show["Recency-weighted score"].round(2)
st.dataframe(show, use_container_width=True, hide_index=True)

st.caption(
    "**Obs/Pred ratio** > 1 → this oblast is seeing MORE sightings than the "
    "structural-share model predicted for the window.  Ratios near 1 mean the "
    "model is calibrated for that oblast.  Ratios < 1 mean the model expected "
    "activity that didn't materialise.  Note: sighting counts ≠ drone counts "
    "(one drone crossing 3 oblasts = 3 sightings)."
)


# ---------- Latest movement traces ----------
if len(multi):
    st.markdown("---")
    st.subheader("Most recent movement events (multi-oblast sightings)")

    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    recent_moves = multi.head(20).copy()
    dist_rows = []
    for _, row in recent_moves.iterrows():
        obs = [o.strip() for o in str(row["all_oblasts"]).split(",") if o.strip()]
        if len(obs) < 2:
            continue
        src_ob, dst_ob = obs[0], obs[1]
        src = centroids[centroids["oblast"] == src_ob]
        dst = centroids[centroids["oblast"] == dst_ob]
        if src.empty or dst.empty:
            continue
        km = _haversine_km(src.iloc[0]["lat"], src.iloc[0]["lon"],
                           dst.iloc[0]["lat"], dst.iloc[0]["lon"])
        age_sec = (now_utc - row["posted_at"]).total_seconds()
        if age_sec < 3600:
            age_str = f"{int(age_sec // 60)}m ago"
        elif age_sec < 86400:
            age_str = f"{int(age_sec // 3600)}h ago"
        else:
            age_str = f"{int(age_sec // 86400)}d ago"
        dist_rows.append({
            "Age": age_str,
            "From": src_ob,
            "To": dst_ob,
            "Distance (km)": int(round(km)),
            "Message": row["text"][:150],
        })
    if dist_rows:
        st.dataframe(pd.DataFrame(dist_rows), use_container_width=True,
                     hide_index=True)


st.markdown("---")
st.subheader("Hypothesis tests — corridor vs target, regimes, ring, gate")
st.caption(
    "Four tests from the public LinkedIn writeup, run live against the "
    "last 30 days of `drone_sightings.csv`.  Overcount caveat: one drone "
    "can appear in several posts along its path, which is EXACTLY why "
    "the corridor-vs-target split matters — it separates transit from "
    "intended destination."
)

_test_window_days = 30
_test_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=_test_window_days)
_s = sights[sights["posted_at"] >= _test_cutoff].copy()

if _s.empty:
    st.info("No sightings in the last 30 days.")
else:
    # Parse ordered oblast lists per post
    def _parse(obs):
        if not isinstance(obs, str): return []
        return [o.strip() for o in obs.split(",") if o.strip()]
    _s["obs_list"] = _s["all_oblasts"].fillna(_s["oblast"]).apply(_parse)

    # ---- Test 1: corridor vs target ----
    origins, endpoints = {}, {}
    for lst in _s["obs_list"]:
        if len(lst) >= 2:
            origins[lst[0]]   = origins.get(lst[0], 0) + 1
            endpoints[lst[-1]] = endpoints.get(lst[-1], 0) + 1
    all_obl = sorted(set(list(origins) + list(endpoints)))
    ce_tbl = pd.DataFrame([{
        "Oblast": o,
        "As arc origin":   origins.get(o, 0),
        "As arc endpoint": endpoints.get(o, 0),
        "Origin − Endpoint": origins.get(o, 0) - endpoints.get(o, 0),
    } for o in all_obl]).sort_values("Origin − Endpoint", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test 1 — Corridor vs Target (last 30 d)**")
        st.caption("Positive Δ = corridor (seen more entering than ending). "
                   "Negative Δ = target (ends up here).")
        st.dataframe(ce_tbl.head(12), use_container_width=True, hide_index=True)

    # ---- Test 2: two regimes (deep-strike vs frontline) ----
    with c2:
        st.markdown("**Test 2 — Two regimes (deep-strike vs frontline)**")
        feats = pd.read_csv(DATA_DIR / "oblast_features.csv")
        bd = dict(zip(feats["oblast"], feats["border_dist"]))
        _s["max_border_dist"] = _s["obs_list"].apply(
            lambda lst: max((bd.get(o, 0) for o in lst), default=0))
        _s["night"] = _s["posted_at"].dt.date
        night_max = _s.groupby("night")["max_border_dist"].max()
        deep_nights = set(night_max[night_max > 700].index)
        _s["regime"] = _s["night"].apply(
            lambda d: "deep" if d in deep_nights else "frontline")

        # Fit decay: sightings-per-oblast vs border_dist, per regime
        def _decay(sub):
            per = sub.explode("obs_list").groupby("obs_list").size()
            dd = pd.DataFrame({"n": per})
            dd["bd"] = dd.index.map(bd)
            dd = dd.dropna()
            if len(dd) < 4: return None, None
            # log-linear slope
            x = dd["bd"].to_numpy(dtype=float)
            y = np.log(dd["n"].to_numpy(dtype=float).clip(min=1))
            slope = np.polyfit(x, y, 1)[0] if x.std() > 0 else np.nan
            return slope, len(dd)

        f_slope, f_n = _decay(_s[_s["regime"] == "frontline"])
        d_slope, d_n = _decay(_s[_s["regime"] == "deep"])
        st.caption("Log-linear decay slope of sightings per oblast vs "
                   "border distance (km).  More negative = steeper decay.")
        st.dataframe(pd.DataFrame([
            {"Regime": "Frontline nights",
             "Slope (log-count / km)": f"{f_slope:.5f}" if f_slope is not None else "n/a",
             "Oblasts": f_n or 0,
             "Nights": len(set(_s[_s['regime']=='frontline']['night']))},
            {"Regime": "Deep-strike nights (any sighting > 700 km in)",
             "Slope (log-count / km)": f"{d_slope:.5f}" if d_slope is not None else "n/a",
             "Oblasts": d_n or 0,
             "Nights": len(deep_nights)},
        ]), use_container_width=True, hide_index=True)
        if f_slope is not None and d_slope is not None:
            st.caption(f"Frontline-vs-deep slope ratio: "
                       f"{(d_slope / f_slope):.2f}× "
                       f"(deep regime {'flatter' if abs(d_slope) < abs(f_slope) else 'steeper'} → "
                       f"deep-strike nights distribute drones further from the border).")

    # ---- Test 3: capital ring ----
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Test 3 — Capital ring (Kyiv Oblast / Kyiv City, last 30 d)**")
        _s_exp = _s.explode("obs_list")
        _s_exp["day"] = _s_exp["posted_at"].dt.date
        daily = _s_exp.groupby(["day","obs_list"]).size().unstack(fill_value=0)
        if "Kyiv Oblast" in daily.columns and "Kyiv City" in daily.columns:
            daily["ratio"] = daily["Kyiv Oblast"] / daily["Kyiv City"].replace(0, np.nan)
            above_2 = (daily["ratio"] > 2).sum()
            total_d = daily["ratio"].notna().sum()
            mean_r = daily["ratio"].dropna().mean()
            st.metric("Days with ratio > 2", f"{above_2} / {total_d}",
                      f"Mean ratio: {mean_r:.2f}")
            st.caption("Hypothesis predicts ratio stays > 2 → drones "
                       "reported on approach, not inside the city.")
            st.line_chart(daily[["ratio"]].dropna().reset_index()
                          .rename(columns={"day":"index"})
                          .set_index("index"))
        else:
            st.caption("Kyiv Oblast / Kyiv City not both present.")

    # ---- Test 4: entry gate ----
    with c4:
        st.markdown("**Test 4 — Entry gate (arcs starting in Chernihiv/Sumy)**")
        arcs = _s[_s["obs_list"].apply(len) >= 2]
        starts = arcs["obs_list"].apply(lambda l: l[0])
        gate = starts.isin(["Chernihiv", "Sumy"]).sum()
        total = len(starts)
        pct = (100 * gate / total) if total else 0
        st.metric("Arcs starting in Chernihiv or Sumy",
                  f"{gate} / {total}", f"{pct:.1f}%")
        top_starts = starts.value_counts().head(8)
        st.dataframe(top_starts.rename_axis("Origin").reset_index(name="Arc count"),
                     use_container_width=True, hide_index=True)
        st.caption("Hypothesis: northern border oblasts are transit "
                   "corridors (high arc-origin count), not targets.")


st.markdown("---")
st.caption(
    "**Method.**  Sightings from `drone_sightings.csv` (populated by "
    "the sync-loop container from @kpszsu).  Recency weight = "
    "exp(−age_hours / 24) → 24-hour half-life.  Movement arcs from the "
    "`all_oblasts` column: when a Telegram post mentions multiple oblasts, "
    "arcs are drawn between successive centroids.  Per-oblast prediction = "
    "structural-share regression coefficient × current weekly budget × "
    "days-in-window / 7.  Distances computed by great-circle (Haversine)."
    "  Source: [github.com/connormmc777/ukraine-drone-forecast]"
    "(https://github.com/connormmc777/ukraine-drone-forecast)."
)
