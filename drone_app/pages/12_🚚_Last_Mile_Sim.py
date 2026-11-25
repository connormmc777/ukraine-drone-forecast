"""
Last-Mile Simulation — NATO autonomous trucks + LICTOR MESH, read through
Goldratt's Theory of Constraints.

The question: if NATO supplied autonomous last-mile logistics trucks with a
LICTOR MESH escort swarm, how much further would a Botzkrieg advance go, would
it reach Russian rail, how much ground across all fronts, and is it a door-kick
or a marginal gain?

The method: an advance is a production line with four stations in series --
RAIL, HAUL, GREY, FORCE. Advance rate = MIN of the four. Goldratt's rule is
that elevating a non-constraint returns nothing, so the primary output of this
page is not a number but WHICH STATION IS HERBIE, per axis, per scenario.

Engine: drone_app/lastmile_sim.py (calibrated on vivaldi_ops.csv, n=5).
Concept: concepts/lictor-mesh.md (defensive publication, CC0).
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lastmile_sim import (                                   # noqa: E402
    Params, SCENARIOS, STATIONS, advance_rate, station_haul, station_grey,
    station_rail, station_force, stall_depth_km, simulate_axis,
    simulate_axis_mc, rail_verdict, load_axes, calibration_check,
    door_kick_index, lictor_cost, STALL_KM2_WK, INTERDICT_KM, TAU_FORCE_WK,
)

st.set_page_config(page_title="Last-Mile Sim", page_icon="🚚", layout="wide")

# Validated categorical palette (light surface #fcfcfb, all-pairs):
# blue / yellow / aqua / red. CVD warn band -> every mark is direct-labelled
# and every chart ships a table view. See references/palette.md.
C = {"RAIL": "#2a78d6", "HAUL": "#eda100", "GREY": "#1baf7a",
     "FORCE": "#e34948"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"

STATION_DESC = {
    "RAIL":  "Railhead throughput — how much the rail can push",
    "HAUL":  "Railhead → forward depot — fleet round-trip + interdiction "
             "(Watling's 90-km envelope)",
    "GREY":  "Forward depot → fighting line — the FPV kill-zone "
             "(what Lictor Mesh attacks)",
    "FORCE": "Operators + infantry available to consolidate ground "
             "(what trucks cannot buy)",
}

p = Params()
axes = load_axes()

st.title("🚚 Last-Mile Simulation — Trucks, Lictor Mesh, and Herbie")
st.markdown(
    "**If NATO supplied autonomous last-mile trucks with a LICTOR MESH escort "
    "swarm — how far does the advance go, does it reach Russian rail, how much "
    "ground across all fronts, and is it a door-kick or a marginal gain?**"
)

st.info(
    "**The method is Goldratt's *The Goal*.** An advance is a production line. "
    "Tonnage enters at a railhead and must clear four stations in series before "
    "it becomes ground taken. Advance rate = **MIN** of the four. Goldratt's "
    "rule: *an hour saved at a non-bottleneck is a mirage.* So the output that "
    "matters is not a km² figure — it is **which station is Herbie**."
)

c = st.columns(4)
for col, s in zip(c, STATIONS):
    col.markdown(f"<div style='border-left:4px solid {C[s]};padding-left:.6rem'>"
                 f"<b>{s}</b><br><span style='font-size:.82rem;color:{INK2}'>"
                 f"{STATION_DESC[s]}</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- calibration
st.subheader("Calibration — does the line reproduce Vivaldi?")
cal = calibration_check(p)
st.dataframe(cal, use_container_width=True, hide_index=True)
st.caption(
    f"Fitted to the five full-sequence observations in `vivaldi_ops.csv` "
    f"(from 2026-09-28, when stage 5 first appears). **n = 5.** The model "
    f"reproduces both anchors to within 0.1% *and* picks the right binding "
    f"station at each: GREY inside the envelope at 77 km, HAUL past it at "
    f"95 km. Everything downstream is extrapolation from two anchor points."
)

# ------------------------------------------------------- station curves chart
st.subheader("The four stations vs distance from railhead")

sel = st.selectbox("Scenario", [s.label for s in SCENARIOS], index=0)
sc = next(s for s in SCENARIOS if s.label == sel)
ready = st.radio("Force generation on this axis", ["proven", "partial", "none"],
                 horizontal=True, index=0,
                 help="'proven' = an axis that has actually run the full "
                      "5-stage Botzkrieg sequence (Vivaldi). 'none' = no UA "
                      "offensive posture there at all.")

# domain starts below the shallowest real axis (Lyman at 20 km) so every
# triangle sits inside the plot rather than floating in the margin
dd = np.linspace(15, 185, 440)
curves = {
    "RAIL":  np.full_like(dd, station_rail("high")),
    "HAUL":  np.array([station_haul(x, p, sc) for x in dd]),
    "GREY":  np.full_like(dd, station_grey(p, sc)),
    "FORCE": np.full_like(dd, station_force(ready, sc, week=1)),
}
env = np.min(np.vstack([curves[s] for s in STATIONS]), axis=0)

fig, ax = plt.subplots(figsize=(11.5, 5.6))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

ax.axvspan(60, 90, color="#eda100", alpha=0.10, zorder=0)
ax.axvline(90, color=MUTED, lw=1.0, ls="--", zorder=1)
ax.axhline(STALL_KM2_WK, color=MUTED, lw=1.0, ls=":", zorder=1)

for s in STATIONS:
    ax.plot(dd, curves[s], lw=2.0, color=C[s], zorder=3, solid_capstyle="round")
ax.plot(dd, env, lw=4.0, color=INK, alpha=0.82, zorder=4,
        solid_capstyle="round", label="Actual advance rate = MIN")

# direct labels (required: CVD warn band + contrast relief)
for s in STATIONS:
    y = curves[s][-1]
    ax.annotate(s, xy=(dd[-1], y), xytext=(6, 0), textcoords="offset points",
                color=C[s], fontsize=10.5, fontweight="bold", va="center")
ax.annotate("MIN = what you actually get", xy=(dd[-1], env[-1]),
            xytext=(6, -14), textcoords="offset points", color=INK,
            fontsize=9.5, fontweight="bold", va="center")
ax.annotate("90-km rail envelope", xy=(90, 0.985), xycoords=("data", "axes fraction"),
            xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=9,
            va="top")
ax.annotate(f"stalled (<{STALL_KM2_WK:.0f} km²/wk)", xy=(18, STALL_KM2_WK),
            xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=9)

# stagger the labels: Lyman (20 km) and Kostiantynivka (28.7 km) sit close
# enough together that a single row of rotated text collides
for i, (_, a) in enumerate(axes.sort_values("front_dist_km").iterrows()):
    ax.plot([a["front_dist_km"]], [0], marker="^", ms=9,
            color=INK, clip_on=False, zorder=6)
    ax.annotate(a["axis"].split()[0][:13], xy=(a["front_dist_km"], 0),
                xytext=(2, -18 if i % 2 == 0 else -36),
                textcoords="offset points", rotation=45,
                ha="right", va="top", fontsize=7.4, color=INK2)

ax.set_xlabel("Distance of the fighting line from its own railhead (km)",
              color=INK2, labelpad=60)
ax.set_ylabel("Sustainable advance rate (km² / week)", color=INK2)
ax.set_title(f"Constraint curves — {sc.key}, force = {ready}", color=INK,
             fontsize=12, fontweight="bold", loc="left")
ax.set_ylim(0, max(160, float(np.max(env)) * 1.25))
ax.grid(True, alpha=0.16, color=MUTED)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=INK2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.30)
st.pyplot(fig); plt.close(fig)

st.caption(
    "The black line is the only one that matters — it is the minimum, which is "
    "what the advance actually gets. Notice how little of it the GREY curve "
    "touches. **Raising a curve that is not already the lowest moves the black "
    "line nowhere.** Triangles mark where each real axis sits today."
)

with st.expander("Table view — station rates at each axis's current depth"):
    rows = []
    for _, a in axes.iterrows():
        r, b, v = advance_rate(a["front_dist_km"], a["rail_capacity"],
                               a["botzkrieg_ready"], p, sc)
        rows.append({"Axis": a["axis"], "Depth (km)": a["front_dist_km"],
                     **{k: round(v[k], 1) for k in STATIONS},
                     "Rate": round(r, 1), "HERBIE": b})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------- binding matrix
st.subheader("Who is Herbie? — axis × scenario")

mat, rate_mat = [], []
for _, a in axes.iterrows():
    mrow, rrow = [], []
    for s in SCENARIOS:
        r, b, _ = advance_rate(a["front_dist_km"], a["rail_capacity"],
                               a["botzkrieg_ready"], p, s)
        mrow.append(b); rrow.append(r)
    mat.append(mrow); rate_mat.append(rrow)

fig, ax = plt.subplots(figsize=(11.5, 4.3))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
for i in range(len(axes)):
    for j in range(len(SCENARIOS)):
        st_name = mat[i][j]
        ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94,
                                    facecolor=C[st_name], alpha=0.90,
                                    edgecolor="#fcfcfb", linewidth=2))
        ax.text(j + 0.5, i + 0.58, st_name, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color="#ffffff")
        ax.text(j + 0.5, i + 0.30, f"{rate_mat[i][j]:.0f}", ha="center",
                va="center", fontsize=7.8, color="#ffffff", alpha=0.95)
ax.set_xlim(0, len(SCENARIOS)); ax.set_ylim(0, len(axes))
ax.set_xticks(np.arange(len(SCENARIOS)) + 0.5)
ax.set_xticklabels([s.key for s in SCENARIOS], fontsize=10)
ax.set_yticks(np.arange(len(axes)) + 0.5)
ax.set_yticklabels([a[:26] for a in axes["axis"]], fontsize=9)
ax.invert_yaxis()
ax.set_title("Binding station + rate (km²/wk) — the station name is written in "
             "every cell, so colour is never the only cue",
             color=INK, fontsize=11, fontweight="bold", loc="left")
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, colors=INK2)
fig.tight_layout()
st.pyplot(fig); plt.close(fig)

st.caption(
    "**Read the GREY cells.** There are none. At the depths these axes actually "
    "sit at today, the last mile is not the binding constraint anywhere — not "
    "even in the scenarios that fix it. That is the headline result."
)

# ------------------------------------------------------- stall depth + rail
st.subheader("How deep can it go, and does it reach Russian rail?")

rows = []
for _, a in axes.iterrows():
    for s in SCENARIOS:
        sd = stall_depth_km(a["rail_capacity"], a["botzkrieg_ready"], p, s)
        v, need, margin = rail_verdict(a, sd)
        rows.append({"Axis": a["axis"], "Scenario": s.key,
                     "Stall depth (km)": None if sd != sd else round(sd, 1),
                     "RU railhead": a["ru_railhead"],
                     "Its distance (km)": a["ru_railhead_km"],
                     "Depth needed (km)": round(need, 1),
                     "Margin (km)": None if margin != margin else round(margin, 1),
                     "Verdict": v})
sdf = pd.DataFrame(rows)
st.dataframe(sdf, use_container_width=True, hide_index=True, height=330)
st.caption(
    f"**Interdiction radius = {INTERDICT_KM:.0f} km**, anchored on this repo's "
    f"own data rather than assumed: Pokrovsk's railhead is listed *degraded* at "
    f"12.8 km from the line, and Kramatorsk and Sloviansk are *under fire* at "
    f"~28 km. Reaching a railhead is not required — bringing it under sustained "
    f"FPV and artillery observation is enough to stop it working."
)

_sd = {s.key: stall_depth_km("high", "proven", p, s) for s in SCENARIOS}
k1, k2, k3 = st.columns(3)
k1.metric("Status-quo stall depth", f"{_sd['S0']:.0f} km",
          "already past the 90-km envelope")
k2.metric("+ Lictor Mesh (S2)", f"{_sd['S2']:.0f} km",
          "no change — GREY was not binding", delta_color="off")
k3.metric("+ autonomous long-haul (S3)", f"{_sd['S3']:.0f} km",
          f"+{_sd['S3'] - _sd['S0']:.0f} km")

# ------------------------------------------------------- all fronts
st.subheader("All fronts — the Goldratt error, quantified")

march = []
for _, a in axes.iterrows():
    for s in SCENARIOS:
        df = simulate_axis(a, p, s, weeks=26)
        march.append({"axis": a["axis"], "scen": s.key,
                      "km2": 0 if df.empty else float(df["cum_km2"].max())})
mdf = pd.DataFrame(march)
agg = mdf.groupby("scen", sort=False).agg(
    geometric=("km2", "sum"),
    force_limited=("km2", lambda x: x.nlargest(2).sum())).reset_index()

fig, ax = plt.subplots(figsize=(11.5, 4.6))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = np.arange(len(agg)); w = 0.38
# deliberately NOT the station palette: blue and red carry station meaning
# everywhere else on this page, and these two bars are not stations.
b1 = ax.bar(x - w/2 - 0.01, agg["geometric"], w, color="#c9c8c3",
            label="All 6 axes at once (not force-feasible)")
b2 = ax.bar(x + w/2 + 0.01, agg["force_limited"], w, color=INK, alpha=0.85,
            label="Best 2 axes (what UA can actually generate)")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}", xy=(r.get_x() + r.get_width()/2,
                    r.get_height()), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.6, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(agg["scen"], fontsize=10)
ax.set_ylabel("Cumulative km² over 26 weeks", color=INK2)
ax.set_title("Ground taken across all fronts, by scenario", color=INK,
             fontsize=12, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(True, axis="y", alpha=0.16, color=MUTED)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=INK2)
fig.tight_layout()
st.pyplot(fig); plt.close(fig)

st.dataframe(agg.rename(columns={"scen": "Scenario",
                                 "geometric": "All 6 axes (km²)",
                                 "force_limited": "Best 2 axes (km²)"}),
             use_container_width=True, hide_index=True)
g0 = agg.loc[agg.scen == "S0", "geometric"].iloc[0]
g2 = agg.loc[agg.scen == "S2", "geometric"].iloc[0]
f0 = agg.loc[agg.scen == "S0", "force_limited"].iloc[0]
f2 = agg.loc[agg.scen == "S2", "force_limited"].iloc[0]
st.caption(
    f"S0 → S2 is **flat**: {g0:,.0f} → {g2:,.0f} km² on the geometric sum, "
    f"{f0:,.0f} → {f2:,.0f} km² force-limited. Spreading trucks across every front is precisely "
    "the local-optimum error *The Goal* was written to prevent — it elevates a "
    f"non-constraint on six axes simultaneously and buys nothing on any of "
    f"them. Only S5 (forward rail repair) and S6 (more force) move the total."
)

# ------------------------------------------------------- counterfactual
st.subheader("The one case where Lictor Mesh WAS the answer")

peak = pd.Series(dict(axis="Vivaldi @ peak", ua_railhead="Pavlohrad",
                      rail_capacity="high", front_dist_km=77.0,
                      ru_railhead="Volnovakha", ru_railhead_km=158.5,
                      ru_rail_capacity="high", width_km=13.4,
                      botzkrieg_ready="proven"))
rows = []
for s in SCENARIOS[:4]:
    r, b, _ = advance_rate(77.0, "high", "proven", p, s)
    det = simulate_axis(peak, p, s, weeks=12)["cum_km2"].max()
    mc = simulate_axis_mc(peak, p, s, weeks=12, n_runs=4000)
    rows.append({"Scenario": s.key, "Week-1 rate": round(r, 1), "HERBIE": b,
                 "Grey-leg CV": mc["cv"], "Deterministic km²": round(det),
                 "Monte-Carlo mean km²": round(mc["mean_km2"]),
                 "MC p10": round(mc["p10"]), "MC p90": round(mc["p90"]),
                 "Fluctuation tax": f"{100*(det-mc['mean_km2'])/det:.1f}%"})
cf = pd.DataFrame(rows)
st.dataframe(cf, use_container_width=True, hide_index=True)
st.markdown(
    """
Rewind to **Sept–Oct 2026**, Vivaldi at 77 km from Pavlohrad — the one moment
on the whole map when GREY genuinely was Herbie. Over a 12-week run:

- **S0 → S2 is 353 → 366 km². A 1.04× gain.** Four percent, at the single most
  favourable point in time and space.
- The reason is Goldratt's *balanced line*: HAUL sat at 53 km²/wk, just 9% behind
  GREY's 49. Unloading Herbie's pack when the next hiker is two metres back buys
  you two metres.
- **S3 — autonomous *long-haul*, not last-mile — gives 1.27×**, and the
  constraint jumps to FORCE.

**On the "sigma" question, you were half right.** Goldratt's dice game says a
line of dependent events with statistical fluctuation yields *less* than the
mean of its constraint, because a bad week cannot be recovered downstream.
Lictor Mesh cuts grey-leg variance hard (CV 0.45 → 0.18), and the Monte Carlo
does show a fluctuation tax — **but only 3%**, because a near-balanced line has
nowhere for the variance to propagate. Variance reduction is real. It is worth
3 points, not a collapse.
    """
)

# ------------------------------------------------------- verdict
st.subheader("Door-kick or marginal? — the verdict")

hits = []
for s in SCENARIOS:
    h, t = door_kick_index(axes, p, s)
    hits.append({"Scenario": s.key, "Label": s.label,
                 "High-capacity RU railheads interdicted": f"{h} / {t}",
                 "What it changes": s.note})
st.dataframe(pd.DataFrame(hits), use_container_width=True, hide_index=True)

st.error(
    "**Marginal — decisively so.** Last-mile autonomy is not a door-kick. "
    "At the most favourable point on the map it is worth **4%**. Across all "
    "fronts it is worth **zero**, because the last mile is not the binding "
    "constraint on any axis at the depth it actually sits. The door-kick lever "
    "exists, but it is not trucks: it is **force generation** (S6) and "
    "**forward rail repair** (S5)."
)

st.warning(
    "**The decisive axis needs no trucks at all.** Orikhiv → Melitopol is "
    "112.8 km railhead-to-railhead — the shortest reach to a high-capacity "
    "Russian hub anywhere on the front, and Melitopol is the hinge of the "
    "Crimea–Donbas land bridge. Interdicting it needs 92.8 km of sustained "
    "depth; the status-quo stall depth is already **107 km**. Sustainment is "
    "*not* what stops Ukraine there — **force generation is**, and it gains "
    "exactly 0 km² in every truck scenario. Trucks delivered to Melitopol "
    "would be a textbook Goldratt mirage."
)

st.success(
    "**Where the trucks DO pay.** Not in ground taken — in vehicles not lost, "
    "crews not killed, and delivery you can plan around. Over 10 sorties "
    "through a 15-km grey zone the crewed baseline loses ~4.5 trucks (~$675k) "
    "and ~9 crew; Lictor-escorted UGVs lose ~1.2 vehicles (~$240k) and zero "
    "crew. That is a real case. It is an **attrition** case, not a "
    "**manoeuvre** case."
)

nodes, lo, hi = lictor_cost(2)
st.caption(
    f"Programme cost, 2 axes × 6 convoys × 4 UGVs = {nodes} protected nodes: "
    f"**${lo/1e6:.1f}M – ${hi/1e6:.1f}M**. Concentrate it on the one axis where "
    f"GREY is about to become Herbie, which is Goldratt's actual advice — not "
    f"spread it across six."
)

with st.expander("Assumptions, parameters and caveats — read before quoting"):
    st.markdown(f"""
**Free parameters, fitted to n = 5 observations.** `haul_0` = {p.haul_0}
km²/wk at the envelope floor, `d_decay` = {p.d_decay} km post-envelope decay
length, `grey_base` = {p.grey_base}. Three fitted parameters on five data
points from **one campaign**. This is a structural toy that makes its logic
inspectable, not a forecast.

**Assumed, not fitted.** Grey-leg delivery fractions (crewed 0.55, bare UGV
0.66, UGV + Lictor Mesh 0.88); convoy speed 28 km/h; 12-hour crewed night
window vs 22-hour autonomous; 15-km grey-zone depth; force ceilings
(proven 90, partial 45, none 6 km²/wk); culmination time constant
τ = {TAU_FORCE_WK:.0f} weeks. Every one of these is arguable. The *qualitative*
result — GREY is not Herbie — holds across wide ranges of all of them,
because it depends on orderings, not levels.

**Known weaknesses.**
1. **n = 5, one campaign, one axis.** Vivaldi may not generalise.
2. **Russia is static here.** The adaptation path exists in the engine
   (`ru_adapt_weeks`) but the headline runs do not use it. Russia fields
   logistics UGVs and fiber FPV already and could counter an escort swarm in
   roughly 6–12 months, so S2's gain is time-boxed and the model overstates it.
3. **Straight-line distances.** Road distance is 15–30% longer, so every
   depth figure here is optimistic.
4. **Force ceilings are the weakest numbers in the model** and they are what
   the verdict turns on. If `FORCE` for the Melitopol axis is wrong, the
   headline is wrong.
5. **No terrain, weather, minefields, or river lines.** A 20-km advance across
   the Zaporizhzhia steppe and a 20-km advance into Horlivka's urban sprawl are
   not the same thing, and this model cannot tell them apart.
6. **km² is a poor measure of success** — it was chosen because it is what
   `vivaldi_ops.csv` records, not because ground is the objective.

**Falsifiers.** (a) A full-sequence op sustains 35–45 km²/wk past 120 km from
its railhead for 8+ weeks → the HAUL curve is wrong and the envelope is not
binding. (b) An axis given escorted last-mile UGVs more than doubles its rate
without any change to haul or force → GREY was Herbie after all and this whole
page is wrong.
    """)

st.caption(
    "Engine: `drone_app/lastmile_sim.py` · Concept: `concepts/lictor-mesh.md` "
    "(CC0 defensive publication) · Calibration: `vivaldi_ops.csv` · "
    "Geography: `railheads.csv`, `salients.csv` · Sources: ISW Aug–Nov 2026, "
    "Deep State, RUSI (Watling) on Russian rail dependency · "
    f"bounded-arithmetic method · generated {date.today().isoformat()}"
)
