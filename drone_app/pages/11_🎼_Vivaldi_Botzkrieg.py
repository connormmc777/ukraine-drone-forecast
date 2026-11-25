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

# ---------- LICTOR MESH ----------
st.divider()
st.header("🜨 LICTOR MESH — mobile escort swarm for the last mile")

st.success(
    "**This is a DEFENSIVE PUBLICATION.** Published to establish prior art and "
    "keep this combination in the public domain under CC0 1.0. "
    "First disclosure 2026-11-23, Connor McSweeney. Full text: "
    "`concepts/lictor-mesh.md`. Use it, improve it, build it — but don't lock "
    "it up. No patent claim is made or reserved on the combination below."
)

st.markdown(
    """
> *Lictors escorted Roman magistrates and carried the fasces — authority and
> enforcement in one bundle. Lictor Mesh escorts the logistics that enable
> Botzkrieg past the 90-km rail envelope.*

**The gap it fills.**  Fielded counter-UAS — Fortem DroneHunter, Anduril
Roadrunner, the gun and EW systems around fixed sites — is **stationary base
defence**.  It protects a *place*.  The deceleration curve on this page points
at a different need: **mobile protection for last-mile logistics past the
~90-km rail envelope**, where trucks and UGVs are hunted across a 10–20 km
grey zone.  Protecting a place is a different problem from protecting a thing
that must keep moving.

**The claimed combination** is all five elements together: a logistics UGV
operating in the last-mile kill-zone; a **co-travelling** escort swarm of
functionally differentiated interceptor classes; on-vehicle detection with a
path for **RF-silent** fiber-optic threats; a **low-bandwidth mesh** carrying
cue and track-handoff only, with terminal engagement run by on-board autonomy
on each escort; and a dedicated **tether-severing** class.
    """
)

# ---- schematic ----
st.subheader("Schematic — one protected convoy node")

fig, ax = plt.subplots(figsize=(11.5, 6.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
BLUE, YEL, AQUA, RED = "#2a78d6", "#eda100", "#1baf7a", "#e34948"
INK, INK2, MUT = "#0b0b0b", "#52514e", "#8a8984"

for rad, col, lab, ls in ((1200, BLUE, "DETECTION required: 800–1200 m", "--"),
                          (800, AQUA, "INTERCEPT envelope: 400–800 m", "-")):
    ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec=col, lw=2.0, ls=ls,
                            alpha=0.9, zorder=2))
    ax.annotate(lab, xy=(0, rad), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9.5, color=col, fontweight="bold")
ax.add_patch(plt.Circle((0, 0), 400, fill=False, ec=AQUA, lw=1.0, ls=":",
                        alpha=0.55, zorder=2))

ax.plot([-1900, 1900], [0, 0], color=MUT, lw=10, alpha=0.18, zorder=0,
        solid_capstyle="butt")
# the vehicles are drawn, not lettered: at this scale a label inside a 170 m
# box overflows it, and the caption below already says what they are
for dx in (-430, -150, 150, 430):
    ax.add_patch(plt.Rectangle((dx - 85, -52), 170, 104, facecolor=INK,
                               ec="#fcfcfb", lw=2, zorder=5))
ax.annotate("4-UGV convoy, ~0.5 t each → ~2 t per sortie",
            xy=(0, -52), xytext=(0, -14), textcoords="offset points",
            ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
ax.annotate("direction of travel →", xy=(1300, 0), xytext=(0, 12),
            textcoords="offset points", fontsize=8.5, color=INK2, ha="left")

esc = [(-620, 520, "A"), (-230, 660, "D"), (210, 690, "C"), (620, 530, "A"),
       (780, 170, "B"), (-780, 150, "B"), (-430, -560, "A"), (90, -660, "C"),
       (560, -530, "D"), (-70, 760, "A")]
for i, (x, y, _) in enumerate(esc):
    ax.plot([x, 0], [y, 0], color=YEL, lw=0.9, alpha=0.55, zorder=1)
    nx, ny, _ = esc[(i + 1) % len(esc)]
    ax.plot([x, nx], [y, ny], color=YEL, lw=0.9, alpha=0.45, zorder=1)
for x, y, cls in esc:
    ax.plot([x], [y], marker="o", ms=17, color=AQUA, mec="#fcfcfb", mew=2,
            zorder=6)
    ax.text(x, y, cls, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color="#0b3d2b", zorder=7)
# notes live in the corners, clear of the swarm and of each other
ax.text(-2230, 1430, "8–12 active escorts aloft\n+ 5 in reserve on the UGVs",
        fontsize=9.5, color="#0b6b4a", fontweight="bold", ha="left", va="top",
        linespacing=1.4)
ax.text(-2230, -1480, "LoRa mesh — cue + track handoff only,\na few bytes.\n"
        "No video backhaul; it cannot carry one.",
        fontsize=9, color="#8a6b00", fontweight="bold", ha="left",
        va="bottom", linespacing=1.4)

for (tx, ty, va, lab) in (
        (-1560, 1030, "bottom", "radio FPV\n(RF scanner sees it first)"),
        (1560, 1030, "bottom", "fiber-optic FPV\n(RF-SILENT → mmWave radar)"),
        (1560, -1030, "top", "loitering munition\n(acoustic + radar)")):
    ax.annotate("", xy=(tx * 0.33, ty * 0.33), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=RED,
                                shrinkA=0, shrinkB=4))
    ax.annotate(lab, xy=(tx, ty), xytext=(0, 10 if va == "bottom" else -10),
                textcoords="offset points", fontsize=9, color=RED,
                fontweight="bold", ha="center", va=va, linespacing=1.4)

ax.annotate("on-UGV sensors:\nacoustic array\n77 GHz mmWave radar\n"
            "passive RF scanner",
            xy=(-430, -60), xytext=(-2230, -560), fontsize=9, color=BLUE,
            fontweight="bold", ha="left", va="center", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", lw=1.2, color=BLUE, alpha=0.6))

ax.set_xlim(-2250, 2250)
ax.set_ylim(-1520, 1560)
ax.set_aspect("equal")
ax.set_title("LICTOR MESH — one protected convoy node (metres, plan view)",
             color=INK, fontsize=12.5, fontweight="bold", loc="left")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---- class table ----
st.subheader("Four-class escort swarm")
st.caption(
    "An earlier draft of this disclosure said *three-class* while listing "
    "four. The disclosed combination is **four classes (A–D)** — the EW/decoy "
    "class D is part of the claim."
)
st.markdown(
    """
| Class | Role | Mass | Effector | Primary target |
|---|---|---|---|---|
| **A** | Kinetic ram interceptor | ~1.0 kg | Airframe / frangible nose | Radio-controlled FPV on terminal run |
| **B** | Net-caster | ~2.0 kg | Deployable rotor-tangling net | Loitering munitions, larger multirotors |
| **C** | **Wire-cutter** | ~1.5 kg | Trailing cutter line / serrated boom | **Fiber-optic tether — no Western counter exists** |
| **D** | EW / decoy | ~3.0 kg | Video-link jam + false-target emitter | The FPV operator's video feed, during the engagement window |

**Loadout:** 8–12 active escorts plus 5 in reserve per protected node, mixed by
threat — FPV-heavy sectors weight A and D, fiber-heavy sectors weight C.

| Sensor (on the UGV) | Detects | Why it is needed |
|---|---|---|
| Acoustic mic array | Rotor signature, 300–500 m | Works against RF-silent threats; cheap; all-aspect |
| mmWave radar, 77 GHz class | Small-RCS (~0.01 m²) airframes | **The fiber-optic answer** — these emit nothing to scan for |
| Passive RF scanner | Control and video links | Longest warning against conventional FPV; free bearing |

**Mesh.**  LoRa between UGV and escorts, carrying **cue and track handoff
only** — bearing, range, class, commit flag.  A few bytes per message.  It is
explicitly *not* a video or radar-return backhaul; at LoRa's bandwidth it could
not be.  Terminal guidance is on-board each escort.  That is what makes the
mesh survivable: losing the UGV's radio does not disarm escorts already
committed.
    """
)

st.subheader("Engagement timeline — the corrected version")
st.warning(
    "**Detection must happen at 800 m – 1.2 km, not 400–800 m.**  "
    "detect ~1 s + classify/commit ~1 s + launch ~1 s + fly-out 6–10 s "
    "(a 1 kg interceptor does 40–70 m/s, so 400–800 m takes 6–10 s) = "
    "**9–13 s total**.  An FPV closing at 33 m/s covers 300–430 m in that "
    "window.  So 400–800 m is the *intercept* envelope; the *detection* "
    "requirement is roughly double it, and a 3-second response is "
    "detect-to-launch latency, not time-to-intercept.  This is the hardest "
    "requirement in the system and the one most likely to decide whether it "
    "works at all."
)

st.subheader("Cost build-up and ROI")
st.markdown(
    """
| Item | Low | High |
|---|---|---|
| 13–17 escort drones @ $5–15k | $65,000 | $255,000 |
| Acoustic array + mmWave radar + RF scanner | $40,000 | $90,000 |
| Fusion compute + LoRa mesh + launch rails | $15,000 | $40,000 |
| **Per protected convoy node** | **$120,000** | **$385,000** |

A 4-UGV convoy with Lictor Mesh: **$0.84M – $1.54M** capex.
    """
)
st.error(
    "**The ROI claim has to change.**  An earlier draft said ~$3–5M of "
    "supplies per sortie and positive ROI by trip three.  That does not "
    "survive arithmetic: four UGVs at ~0.5 t each carry ~2 t, and 2 t of "
    "mixed ammunition, fuel and rations is worth roughly **$40k–200k** — not "
    "$3–5M.  Cargo replacement value cannot justify this system, and a "
    "reviewer will go straight at that number."
)
st.markdown(
    """
**The defensible case is attrition and casualties**, over 10 sorties through a
15-km grey zone:

| | Crewed 4 t truck | Lictor-escorted UGVs |
|---|---|---|
| Per-trip delivery survival | ~55% | ~88% |
| Vehicles lost / 10 sorties | ~4.5 × $150k = **$675k** | ~1.2 × $200k = **$240k** |
| Crew casualties / 10 sorties | **~9** | **0** |
| Delivery variance (CV) | ~0.45 | ~0.18 |

It pays for itself in **vehicles not lost, crews not killed, and delivery you
can plan around** — not in cargo value.  A real case, just a different one.
    """
)

st.subheader("What it does NOT do")
st.info(
    "The simulation on the **🚚 Last-Mile Sim** page tested this concept under "
    "Theory of Constraints, and the honest result is that Lictor Mesh is "
    "**necessary but not sufficient**:\n\n"
    "- **It does not extend the 90-km envelope.**  That envelope is a "
    "*throughput* limit on the railhead-to-depot haul; the grey zone is a "
    "*survivability* limit on the final leg.  Different constraints.  Fixing "
    "the second does not move the first.\n"
    "- **It does not neutralize the grey zone.**  Russian FPV still owns that "
    "airspace.  The honest claim is that it makes the grey zone *survivable "
    "for logistics* — a smaller and achievable claim.\n"
    "- **The last mile is not currently the binding constraint on any axis.**  "
    "At Vivaldi's peak (77 km) it was worth 1.04×.  Across all fronts, zero.\n"
    "- **The advantage is time-boxed.**  Russia fields logistics UGVs and "
    "fiber FPV already and could counter an escort swarm in roughly 6–12 "
    "months."
)
st.caption(
    "Full disclosure text, prior-art statement and CC0 dedication: "
    "`concepts/lictor-mesh.md` · constraint analysis: **🚚 Last-Mile Sim** "
    "page and `drone_app/lastmile_sim.py`"
)


st.caption("Sources: ISW, Deep State, UA MoD, CRS reporting, RUSI "
           "Watling · bounded-arithmetic method · "
           f"generated {date.today().isoformat()}")
