"""
Naval Strikes — Ukraine → Russia sea-drone campaign against Black Sea Fleet.

Mirror of the air-drone page, right-sized for the naval domain:
  • event-level log (10s of events per year, not 100s per night)
  • target frequency + cumulative counter
  • structural target-share OLS fitted on whatever N we currently have
    (honest small-sample banner surfaces when n < 15)
  • map of Black Sea launch/target geometry

Data: naval_events.csv (populated by naval_ingest.py at each sync tick).
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Naval Strikes — Ukraine → Russia",
                   page_icon="🌊", layout="wide")


# ---------- helpers ----------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(ttl=60)
def load_events(mtime_key: float) -> pd.DataFrame:
    csv = DATA_DIR / "naval_events.csv"
    if not csv.exists():
        return pd.DataFrame(columns=[
            "posted_at", "target", "drones_launched", "action",
            "ship_hit", "source_channel", "message_id", "text",
        ])
    df = pd.read_csv(csv)
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True, errors="coerce")
    return df.dropna(subset=["posted_at"]).sort_values("posted_at", ascending=False)


@st.cache_data(ttl=300)
def load_targets(mtime_key: float) -> pd.DataFrame:
    return pd.read_csv(APP_DIR / "naval_targets.csv")


# ---------- data load ----------
events = load_events(_mtime(DATA_DIR / "naval_events.csv"))
targets = load_targets(_mtime(APP_DIR / "naval_targets.csv"))

st.title("🌊 Naval Strikes — Ukraine → Russia")
st.markdown(
    "**Ukrainian sea-drone (USV) and long-range strike campaign against the "
    "Black Sea Fleet, coastal ports, and oil infrastructure.**  Mirror of the "
    "air-drone page — same modeling shape, right-sized for the rare-event "
    "domain.  Source: [@GeneralStaffZSU](https://t.me/s/GeneralStaffZSU) "
    "with a naval-keyword parser."
)

if events.empty:
    st.warning(
        "🕐 **No naval events on file yet.**  The sync sidecar populates "
        "`data/naval_events.csv` on each 60-second tick.  If this stays empty "
        "past the next sync, check the parser: "
        "`kubectl logs -l app.kubernetes.io/name=dronespredictions -c sync-loop "
        "| grep naval`"
    )
    st.stop()


# ---------- top KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total events on file", f"{len(events):,}")
c2.metric("Days covered",
          f"{(events['posted_at'].max() - events['posted_at'].min()).days:,}")
c3.metric("Unique targets hit", f"{events['target'].nunique():,}")
last_dt = events['posted_at'].max()
_age_h = (pd.Timestamp.utcnow() - last_dt).total_seconds() / 3600
c4.metric("Hours since last event", f"{_age_h:.1f}h")


# ---------- live event log ----------
st.markdown("---")
st.subheader("Recent naval-strike events")
_targets_filter = st.multiselect(
    "Filter by target (empty = all)",
    sorted(events["target"].unique().tolist()),
)
_ev = events if not _targets_filter else events[events["target"].isin(_targets_filter)]

_display = _ev.copy()
_display["Age"] = _display["posted_at"].apply(
    lambda dt: f"{int((pd.Timestamp.utcnow() - dt).total_seconds() // 3600)}h ago"
                if (pd.Timestamp.utcnow() - dt).total_seconds() < 86400 * 3
                else f"{int((pd.Timestamp.utcnow() - dt).total_seconds() // 86400)}d ago"
)
_display["Posted (UTC)"] = _display["posted_at"].dt.strftime("%Y-%m-%d %H:%M")
_show = _display[["Age", "Posted (UTC)", "target", "action",
                    "ship_hit", "text"]].rename(columns={
    "target": "Target", "action": "Action", "ship_hit": "Ship / object",
    "text": "Source text",
})
st.dataframe(_show.head(25), use_container_width=True, hide_index=True)
st.caption(f"Showing 25 most recent of {len(_ev):,} filtered events "
           f"(source: @GeneralStaffZSU · parser: naval_ingest.py)")


# ---------- cumulative + weekly ----------
st.markdown("---")
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Cumulative events")
    ce = events.sort_values("posted_at").copy()
    ce["cum"] = np.arange(1, len(ce) + 1)
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(ce["posted_at"], ce["cum"], color="#003d7a", lw=2)
    ax1.fill_between(ce["posted_at"], 0, ce["cum"], color="#003d7a", alpha=0.15)
    ax1.set_ylabel("cumulative naval events")
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    fig1.autofmt_xdate()
    st.pyplot(fig1); plt.close(fig1)

with c_right:
    st.subheader("Events per calendar week")
    we = events.copy()
    we["week"] = we["posted_at"].dt.to_period("W").dt.start_time
    wk_counts = we.groupby("week").size().reset_index(name="n")
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.bar(wk_counts["week"], wk_counts["n"], width=6,
            color="#cc0033", edgecolor="none")
    ax2.set_ylabel("events / week")
    ax2.grid(alpha=0.3, axis="y")
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    fig2.autofmt_xdate()
    st.pyplot(fig2); plt.close(fig2)


# ---------- target frequency ----------
st.markdown("---")
st.subheader("Target frequency")
freq = (events.groupby("target").size().reset_index(name="events")
        .sort_values("events", ascending=False))
freq_with_geom = freq.merge(targets, on="target", how="left")
freq_with_geom["events_share"] = freq_with_geom["events"] / freq_with_geom["events"].sum()
st.dataframe(
    freq_with_geom[["target", "events", "events_share",
                     "distance_odesa_km", "fleet_ships_est", "defense_score"]]
        .round({"events_share": 3})
        .rename(columns={
            "target": "Target", "events": "Events", "events_share": "Share",
            "distance_odesa_km": "Dist from Odesa (km)",
            "fleet_ships_est": "Est. ships in port",
            "defense_score": "Defense (0-10)",
        }),
    use_container_width=True, hide_index=True,
)


# ---------- structural target-share regression ----------
st.markdown("---")
st.subheader("Structural target-share regression")
st.caption(
    r"Model:  $\text{share}_i \propto \beta_1 \cdot \exp(-d_i/300) + "
    r"\beta_2 \cdot \text{ships}_i - \beta_3 \cdot \text{defense}_i$   "
    "— distance-decay from Odesa, weighted by fleet presence and inverse defense."
)

reg = freq_with_geom.dropna(subset=["distance_odesa_km"]).copy()
reg = reg[reg["target"] != "Odesa"]  # exclude the launch point itself
if reg.empty:
    st.info("No fittable events yet.")
else:
    reg["dist_decay"] = np.exp(-reg["distance_odesa_km"] / 300.0)
    y = reg["events_share"].values
    X = reg[["dist_decay", "fleet_ships_est", "defense_score"]].values
    n, k = X.shape

    if n < 5:
        st.warning(
            f"⚠️  **Only {n} distinct targets with events so far.**  Regression "
            f"needs n > k+1 = {k+1} to even fit; results below are "
            "unreliable and shown for completeness only.  Wait for more events."
        )

    try:
        X_ = np.column_stack([np.ones(n), X])
        beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
        resid = y - X_ @ beta
        rss = (resid ** 2).sum()
        tss = ((y - y.mean()) ** 2).sum() if y.var() > 0 else 1.0
        r2 = 1 - rss / tss
        dof = max(n - k - 1, 1)
        mse = rss / dof
        try:
            cov = mse * np.linalg.inv(X_.T @ X_)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)
        t = beta / se
        pvals = 2 * (1 - stats.t.cdf(np.abs(t), df=dof))
        adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan
        f_stat = (r2 / max(1 - r2, 1e-9)) * (dof / k) if k else np.nan
        f_p = 1 - stats.f.cdf(f_stat, k, dof) if k and dof > 0 else np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("n (targets w/ events)", n)
        m2.metric("R²", f"{r2:.3f}")
        m3.metric("Adj R²", f"{adj:.3f}")
        m4.metric(f"F({k}, {dof})", f"{f_stat:.2f}", delta=f"p = {f_p:.4f}")

        coef_df = pd.DataFrame({
            "variable": ["intercept", "exp(−dist/300)",
                         "fleet_ships_est", "defense_score"],
            "β":  [f"{b:+.4f}" for b in beta],
            "SE": [f"{s:.4f}" if not np.isnan(s) else "n/a" for s in se],
            "t":  [f"{v:+.2f}" if not np.isnan(v) else "n/a" for v in t],
            "p":  [f"{p:.4f}" if not np.isnan(p) else "n/a" for p in pvals],
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Small-sample honesty note
        if n < 15:
            st.warning(
                f"📉 **Honest small-sample warning.**  With n = {n}, "
                f"individual β SEs are wide and p-values are noisy.  The "
                "structural regression will stabilise once the events log "
                "grows past ~20.  Directional signs are meaningful; "
                "significance claims are not."
            )
    except Exception as e:
        st.info(f"Regression could not be fit: {type(e).__name__}: {e}")


# ---------- map ----------
st.markdown("---")
st.subheader("Black Sea theatre — targets + strike origin")
map_df = targets[["target", "lat", "lon"]].dropna().copy()
_ev_counts = events.groupby("target").size().to_dict()
map_df["hits"] = map_df["target"].map(_ev_counts).fillna(0).astype(int)
map_df = map_df.rename(columns={"lat": "latitude", "lon": "longitude"})
# scatter with hit count sizing
map_df["size"] = 50000 + map_df["hits"] * 30000
st.map(map_df, size="size", color="#cc0033", zoom=5)
st.caption(
    "Odesa is the reference launch point.  Circle size scales with total "
    "recorded strike events per target."
)


# ---------- footer ----------
st.markdown("---")
st.caption(
    "Data flow: `naval_ingest.py` scrapes @GeneralStaffZSU every 60s, "
    "filters for USV/ship-strike keywords + strike verbs, writes to "
    "`data/naval_events.csv`.  Parser is deliberately conservative — the raw "
    "source text is retained so you can audit any row.  False positives from "
    "the ingest (e.g. exercise mentions like 'Sea Breeze') can be filtered "
    "here or removed from the CSV manually."
)
