"""
Last-Mile Sustainment Simulation — NATO autonomous trucks + LICTOR MESH
========================================================================

The question
------------
If NATO supplied autonomous last-mile logistics trucks (escorted by a
LICTOR MESH counter-FPV swarm), how much further would a Botzkrieg-style
advance go, would it reach Russia's rail network, how much ground would
it take across all fronts, and is it a door-kick or a marginal gain?

The method — Theory of Constraints (Goldratt, "The Goal")
----------------------------------------------------------
An advance is a production line.  Tonnage enters at a railhead and must
survive four stations in series before it becomes ground taken:

    [1 RAIL]  railhead throughput
       |
    [2 HAUL]  railhead -> forward depot, fleet round-trip + interdiction
       |                  (this is Watling's 90-km rail envelope)
    [3 GREY]  forward depot -> fighting line, the FPV kill-zone
       |                  (this is what Lictor Mesh attacks)
    [4 FORCE] operators + infantry available to consolidate ground

Advance rate = MIN of the four.  Goldratt's rule: elevating a station
that is NOT the constraint buys exactly nothing.  So the model's primary
output is not a number — it is WHICH STATION IS HERBIE, per axis, per
scenario.  Everything else follows from that.

Calibration
-----------
Fitted to the five full-sequence observations in vivaldi_ops.csv (from
2026-09-28, when stage 5 first appears, onward).  n=5.  This is a
structural toy, not a forecast.  Two anchor points are reproduced:

    d= 77 km -> 49 km2/wk, binding = GREY  (inside the envelope)
    d= 95 km -> 28 km2/wk, binding = HAUL  (past the envelope)

Everything downstream is extrapolation from those two anchors.

Sources: ISW campaign updates Aug-Nov 2026, Deep State mapping, RUSI
(Watling) on Russian rail dependency, railheads.csv / salients.csv /
vivaldi_ops.csv in this repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------- constants
STALL_KM2_WK = 8.0     # below this an advance is operationally stalled
INTERDICT_KM = 20.0    # front within this range renders a railhead degraded
                       # (anchored on this repo's own data: Pokrovsk railhead
                       #  is "degraded" at 12.8 km from the line; Kramatorsk /
                       #  Sloviansk are "under fire" at ~28 km)

RAIL_CEILING = {"high": 140.0, "med": 70.0, "low": 22.0}   # km2/wk equivalent
FORCE_CEILING = {"proven": 90.0, "partial": 45.0, "none": 6.0}
TAU_FORCE_WK = 8.0     # offensives culminate: assault groups are consumed.
                       # F(wk) = F0 / (1 + wk/TAU). Vivaldi decelerated after
                       # ~10-12 weeks; historical offensives culminate 8-14 wks.


@dataclass(frozen=True)
class Params:
    """Physical + fitted parameters.  Fitted: haul_0, d_decay, grey_base."""
    # haul leg (station 2)
    haul_0:      float = 65.8   # FITTED km2/wk-equivalent at d = d_free
    d_free:      float = 60.0   # inside this, round-trip time is not limiting
    d_envelope:  float = 90.0   # Watling rail envelope
    d_decay:     float = 10.8   # FITTED post-envelope decay length (km)
    v_kmh:       float = 28.0   # convoy speed, night / damaged roads
    t_turn_h:    float = 2.0    # load + unload + disperse
    l_grey_km:   float = 15.0   # depth of the FPV kill-zone
    window_crewed_h: float = 12.0   # crewed trucks move in darkness
    window_auto_h:   float = 22.0   # autonomous trucks run around the clock

    # grey leg (station 3)
    grey_base:   float = 89.1   # FITTED capacity at perfect delivery
    g_crewed:    float = 0.55   # fraction of tonnage that survives the last mile
    g_ugv:       float = 0.66   # UGV: better survival, worse payload-per-vehicle
    g_ugv_mesh:  float = 0.88   # UGV + Lictor Mesh escort

    # force generation (station 4)
    force_mult:  float = 1.0


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    grey_g: float            # last-mile delivery fraction
    grey_fleet_mult: float   # last-mile vehicle-hours multiplier
    auto_haul: bool          # autonomous long-haul (24/7 ops)
    haul_fleet_mult: float   # long-haul fleet size multiplier
    railhead_shift_km: float # forward rail repair, km of envelope recovered
    force_mult: float        # force-generation multiplier
    note: str = ""


SCENARIOS = [
    Scenario("S0", "S0 — Status quo (crewed trucks, crewed last mile)",
             0.55, 1.00, False, 1.0, 0.0, 1.0,
             "What Vivaldi actually ran. Calibration baseline."),
    Scenario("S1", "S1 — NATO logistics UGVs, no escort",
             0.66, 1.15, False, 1.0, 0.0, 1.0,
             "UGVs survive better but carry ~0.5 t vs ~4 t. Payload penalty "
             "eats most of the survival gain."),
    Scenario("S2", "S2 — NATO UGVs + LICTOR MESH escort",
             0.88, 1.75, False, 1.0, 0.0, 1.0,
             "Escort swarm defeats radio-FPV, loitering munitions and "
             "fiber-optic drones inside 400-800 m. Grey zone elevated."),
    Scenario("S3", "S3 — Lictor Mesh + autonomous LONG-HAUL fleet",
             0.88, 1.75, True, 1.0, 0.0, 1.0,
             "Also removes the driver from the railhead-to-depot leg: "
             "22 h/day operations instead of a 12 h night window."),
    Scenario("S4", "S4 — S3 + 2x haul fleet (full NATO package)",
             0.88, 1.75, True, 2.0, 0.0, 1.0,
             "Fleet scale on top of autonomy."),
    Scenario("S5", "S5 — S4 + forward rail repair (+30 km railhead)",
             0.88, 1.75, True, 2.0, 30.0, 1.0,
             "Re-gauging and repairing captured track. The only lever that "
             "genuinely moves the envelope rather than stretching it."),
    Scenario("S6", "S6 — S4 + force generation doubled",
             0.88, 1.75, True, 2.0, 0.0, 2.0,
             "What trucks CANNOT buy: more operator teams and infantry to "
             "consolidate ground."),
]


# ------------------------------------------------------------ the 4 stations
def station_rail(rail_capacity: str) -> float:
    return RAIL_CEILING.get(rail_capacity, 70.0)


def station_haul(d_km: float, p: Params, sc: Scenario) -> float:
    """Railhead -> forward depot. Round-trip limited, then interdicted."""
    d_eff = max(0.0, d_km - sc.railhead_shift_km)
    d_haul = max(5.0, d_eff - p.l_grey_km)

    window = p.window_auto_h if sc.auto_haul else p.window_crewed_h
    ref_haul = max(5.0, p.d_free - p.l_grey_km)

    trips = window / (2 * d_haul / p.v_kmh + p.t_turn_h)
    trips_ref = p.window_crewed_h / (2 * ref_haul / p.v_kmh + p.t_turn_h)

    # Inside d_free the round trip is not what limits the leg -- loading,
    # dispersal and depot handling are. Without this floor the curve climbs
    # without bound as d -> 0, which the model does not claim. Binds nothing
    # at d >= d_free, so no calibrated result changes.
    trips = min(trips, window / (2 * ref_haul / p.v_kmh + p.t_turn_h))

    interdiction = np.exp(-max(0.0, d_eff - p.d_envelope) / p.d_decay)
    return p.haul_0 * (trips / trips_ref) * interdiction * sc.haul_fleet_mult


def station_grey(p: Params, sc: Scenario) -> float:
    """Forward depot -> fighting line, through the FPV kill-zone."""
    return p.grey_base * sc.grey_g * sc.grey_fleet_mult


def station_force(botzkrieg_ready: str, sc: Scenario, week: int = 1) -> float:
    """Force generation, with culmination. Trucks cannot buy this station."""
    f0 = FORCE_CEILING.get(botzkrieg_ready, 45.0) * sc.force_mult
    return f0 / (1.0 + max(0, week - 1) / TAU_FORCE_WK)


STATIONS = ("RAIL", "HAUL", "GREY", "FORCE")


def advance_rate(d_km, rail_capacity, botzkrieg_ready, p, sc, week: int = 1):
    """Returns (rate_km2_per_week, binding_station, all_station_rates)."""
    vals = {
        "RAIL":  station_rail(rail_capacity),
        "HAUL":  station_haul(d_km, p, sc),
        "GREY":  station_grey(p, sc),
        "FORCE": station_force(botzkrieg_ready, sc, week),
    }
    binding = min(vals, key=vals.get)
    return vals[binding], binding, vals


# ------------------------------------------------------------ stall depth
def stall_depth_km(rail_capacity, botzkrieg_ready, p, sc,
                   d_lo=40.0, d_hi=400.0, tol=0.05):
    """Deepest d at which the advance still exceeds STALL_KM2_WK."""
    r_lo, _, _ = advance_rate(d_lo, rail_capacity, botzkrieg_ready, p, sc)
    if r_lo <= STALL_KM2_WK:
        return float("nan")          # stalled before it starts
    r_hi, _, _ = advance_rate(d_hi, rail_capacity, botzkrieg_ready, p, sc)
    if r_hi > STALL_KM2_WK:
        return d_hi                  # never stalls in range
    while d_hi - d_lo > tol:
        mid = 0.5 * (d_lo + d_hi)
        r, _, _ = advance_rate(mid, rail_capacity, botzkrieg_ready, p, sc)
        if r > STALL_KM2_WK:
            d_lo = mid
        else:
            d_hi = mid
    return 0.5 * (d_lo + d_hi)


# ------------------------------------------------------------ dynamic march
def simulate_axis(axis_row, p: Params, sc: Scenario, weeks: int = 26,
                  ru_adapt_weeks: int | None = None):
    """
    Week-by-week march. Advancing pushes the line further from the railhead,
    which loads the HAUL station further -- Goldratt's rope getting longer.

    ru_adapt_weeks: if set, Russia fields counter-escort after this many
    weeks and the grey-zone gain decays back toward the crewed baseline.
    """
    d = float(axis_row["front_dist_km"])
    width = float(axis_row["width_km"])
    cap = axis_row["rail_capacity"]
    ready = axis_row["botzkrieg_ready"]

    rows, cum = [], 0.0
    for wk in range(1, weeks + 1):
        sc_wk = sc
        if ru_adapt_weeks is not None and wk > ru_adapt_weeks:
            decay = np.exp(-(wk - ru_adapt_weeks) / 12.0)
            g_now = p.g_crewed + (sc.grey_g - p.g_crewed) * decay
            fm_now = 1.0 + (sc.grey_fleet_mult - 1.0) * decay
            sc_wk = replace(sc, grey_g=g_now, grey_fleet_mult=fm_now)

        rate, binding, vals = advance_rate(d, cap, ready, p, sc_wk, week=wk)
        if rate <= STALL_KM2_WK:
            rows.append(dict(week=wk, d_km=d, rate=rate, binding=binding,
                             cum_km2=cum, stalled=True, **vals))
            break
        cum += rate
        d += rate / width          # km2 -> linear km of advance
        rows.append(dict(week=wk, d_km=d, rate=rate, binding=binding,
                         cum_km2=cum, stalled=False, **vals))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ rail verdict
def rail_verdict(axis_row, stall_km):
    """Does the achievable depth bring a Russian railhead under fire?"""
    need = float(axis_row["ru_railhead_km"]) - INTERDICT_KM
    if np.isnan(stall_km):
        return "no advance", need, float("nan")
    margin = stall_km - need
    if margin >= 0:
        verdict = "RAILHEAD INTERDICTED"
    elif margin > -25:
        verdict = "falls short (near miss)"
    else:
        verdict = "out of reach"
    return verdict, need, margin


# ------------------------------------------------- Goldratt: fluctuation tax
# The mean-constraint model above is deterministic. Goldratt's dice game says
# that in a line of DEPENDENT events with STATISTICAL FLUCTUATION, throughput
# is strictly worse than the mean of the constraint, because a bad week at any
# station cannot be recovered downstream -- lost tempo is lost.
#
# min() is concave, so E[min(stations)] < min(E[stations]). The gap is the tax.
# Lictor Mesh's real payoff is mostly here: not a higher mean last mile, but a
# far tighter one. CV is the coefficient of variation of weekly delivery.
GREY_CV = {0.55: 0.45,    # crewed last mile: nights the route is simply shut
           0.66: 0.32,    # bare UGV: survives more, still hunted
           0.88: 0.18}    # UGV + Lictor Mesh: predictable delivery


def grey_cv(sc: Scenario) -> float:
    return GREY_CV.get(round(sc.grey_g, 2), 0.30)


def simulate_axis_mc(axis_row, p: Params, sc: Scenario, weeks: int = 26,
                     n_runs: int = 2000, seed: int = 7):
    """Monte Carlo march with lognormal weekly noise on the grey leg."""
    rng = np.random.default_rng(seed)
    cv = grey_cv(sc)
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = -0.5 * sigma ** 2                      # so E[noise] = 1

    width = float(axis_row["width_km"])
    cap, ready = axis_row["rail_capacity"], axis_row["botzkrieg_ready"]
    d0 = float(axis_row["front_dist_km"])

    totals, depths = np.zeros(n_runs), np.zeros(n_runs)
    for r in range(n_runs):
        d, cum = d0, 0.0
        for wk in range(1, weeks + 1):
            noise = float(rng.lognormal(mu, sigma))
            vals = (station_rail(cap),
                    station_haul(d, p, sc),
                    station_grey(p, sc) * noise,
                    station_force(ready, sc, wk))
            rate = min(vals)
            if rate <= STALL_KM2_WK:
                break
            cum += rate
            d += rate / width
        totals[r], depths[r] = cum, d
    return dict(mean_km2=totals.mean(), p10=np.percentile(totals, 10),
                p90=np.percentile(totals, 90), mean_depth=depths.mean(),
                cv=cv)


def load_axes():
    return pd.read_csv(DATA_DIR / "axes.csv")


# ------------------------------------------------------------ Lictor cost
LICTOR_NODE_COST = (210_000, 385_000)   # $ per protected convoy node
SUPPLIES_PER_SORTIE = (3_000_000, 5_000_000)   # $ delivered per 4-UGV sortie
UGV_PER_CONVOY = 4


def lictor_cost(n_axes: int, convoys_per_axis: int = 6):
    """Programme cost to escort n_axes at convoys_per_axis convoys each."""
    nodes = n_axes * convoys_per_axis * UGV_PER_CONVOY
    return nodes, nodes * LICTOR_NODE_COST[0], nodes * LICTOR_NODE_COST[1]


def door_kick_index(axes: pd.DataFrame, p: Params, sc: Scenario):
    """Fraction of HIGH-capacity RU railheads brought under interdiction."""
    hi = axes[axes.ru_rail_capacity == "high"].copy()
    # several axes aim at the same railhead (Horlivka) -- count each once,
    # keeping the axis with the best chance of reaching it
    hi = hi.sort_values("ru_railhead_km").drop_duplicates("ru_railhead")
    hit = 0
    for _, a in hi.iterrows():
        sd = stall_depth_km(a["rail_capacity"], a["botzkrieg_ready"], p, sc)
        v, _, _ = rail_verdict(a, sd)
        hit += (v == "RAILHEAD INTERDICTED")
    return hit, len(hi)


# ------------------------------------------------------------ calibration ck
def calibration_check(p: Params):
    s0 = SCENARIOS[0]
    out = []
    for d, obs, exp_bind in ((77.0, 49.0, "GREY"), (95.0, 28.0, "HAUL")):
        r, b, vals = advance_rate(d, "high", "proven", p, s0)
        out.append(dict(d_km=d, observed=obs, modelled=round(r, 1),
                        err_pct=round(100 * (r - obs) / obs, 1),
                        binding=b, expected=exp_bind,
                        ok=(b == exp_bind and abs(r - obs) / obs < 0.05)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    p = Params()

    print("=" * 78)
    print("CALIBRATION CHECK — does the model reproduce Vivaldi's anchors?")
    print("=" * 78)
    print(calibration_check(p).to_string(index=False))

    axes = load_axes()

    print("\n" + "=" * 78)
    print("STATION RATES + BINDING CONSTRAINT — each axis at its CURRENT depth")
    print("=" * 78)
    rec = []
    for _, a in axes.iterrows():
        for sc in SCENARIOS:
            r, b, v = advance_rate(a["front_dist_km"], a["rail_capacity"],
                                    a["botzkrieg_ready"], p, sc)
            rec.append(dict(axis=a["axis"][:28], scen=sc.key,
                            d=a["front_dist_km"], rate=round(r, 1),
                            HERBIE=b, RAIL=round(v["RAIL"]),
                            HAUL=round(v["HAUL"], 1), GREY=round(v["GREY"], 1),
                            FORCE=round(v["FORCE"])))
    print(pd.DataFrame(rec).to_string(index=False))

    print("\n" + "=" * 78)
    print("STALL DEPTH + RAIL VERDICT")
    print("=" * 78)
    rec = []
    for _, a in axes.iterrows():
        for sc in SCENARIOS:
            sd = stall_depth_km(a["rail_capacity"], a["botzkrieg_ready"], p, sc)
            v, need, margin = rail_verdict(a, sd)
            rec.append(dict(axis=a["axis"][:28], scen=sc.key,
                            stall_km=round(sd, 1) if sd == sd else None,
                            ru_rail=a["ru_railhead"][:12],
                            ru_km=a["ru_railhead_km"],
                            need_km=round(need, 1),
                            margin=round(margin, 1) if margin == margin else None,
                            verdict=v))
    print(pd.DataFrame(rec).to_string(index=False))

    print("\n" + "=" * 78)
    print("26-WEEK MARCH — cumulative km2 per axis per scenario")
    print("=" * 78)
    rec = []
    for _, a in axes.iterrows():
        for sc in SCENARIOS:
            df = simulate_axis(a, p, sc, weeks=26)
            if df.empty:
                rec.append(dict(axis=a["axis"][:28], scen=sc.key, km2=0,
                                end_d=a["front_dist_km"], wks=0, HERBIE="-"))
                continue
            rec.append(dict(axis=a["axis"][:28], scen=sc.key,
                            km2=round(df["cum_km2"].max()),
                            end_d=round(df["d_km"].max(), 1),
                            wks=int(df["week"].max()),
                            HERBIE=df["binding"].mode()[0]))
    march = pd.DataFrame(rec)
    print(march.to_string(index=False))

    print("\n" + "=" * 78)
    print("GOLDRATT FLUCTUATION TAX — deterministic vs Monte Carlo (Vivaldi axis)")
    print("=" * 78)
    viv = axes.iloc[0]
    rec = []
    for sc in SCENARIOS:
        det = simulate_axis(viv, p, sc, weeks=26)["cum_km2"].max()
        det = 0.0 if pd.isna(det) else det
        mc = simulate_axis_mc(viv, p, sc, weeks=26)
        rec.append(dict(scen=sc.key, grey_CV=mc["cv"],
                        determ_km2=round(det),
                        MC_mean_km2=round(mc["mean_km2"]),
                        MC_p10=round(mc["p10"]), MC_p90=round(mc["p90"]),
                        tax_km2=round(det - mc["mean_km2"]),
                        tax_pct=round(100 * (det - mc["mean_km2"]) / det, 1)
                        if det > 0 else None))
    print(pd.DataFrame(rec).to_string(index=False))

    print("\n" + "=" * 78)
    print("RETROSPECTIVE COUNTERFACTUAL — Vivaldi's PEAK phase (d = 77 km)")
    print("What would Lictor Mesh have been worth in Sept-Oct 2026, when the")
    print("grey zone WAS the binding constraint?  12-week run, MC n=4000.")
    print("=" * 78)
    peak = pd.Series(dict(axis="Vivaldi @ peak", ua_railhead="Pavlohrad",
                          rail_capacity="high", front_dist_km=77.0,
                          ru_railhead="Volnovakha", ru_railhead_km=158.5,
                          ru_rail_capacity="high", width_km=13.4,
                          botzkrieg_ready="proven"))
    rec = []
    for sc in SCENARIOS[:4]:
        r, b, v = advance_rate(77.0, "high", "proven", p, sc)
        det = simulate_axis(peak, p, sc, weeks=12)["cum_km2"].max()
        mc = simulate_axis_mc(peak, p, sc, weeks=12, n_runs=4000)
        rec.append(dict(scen=sc.key, wk1_rate=round(r, 1), HERBIE=b,
                        grey_CV=mc["cv"], determ_km2=round(det),
                        MC_mean=round(mc["mean_km2"]),
                        MC_p10=round(mc["p10"]), MC_p90=round(mc["p90"]),
                        tax_pct=round(100 * (det - mc["mean_km2"]) / det, 1)))
    cf = pd.DataFrame(rec)
    print(cf.to_string(index=False))
    base = cf.loc[cf.scen == "S0", "MC_mean"].iloc[0]
    for _, row in cf.iterrows():
        print(f"  {row.scen}: MC mean {row.MC_mean:4.0f} km2  "
              f"= {row.MC_mean / base:4.2f}x the S0 baseline")

    print("\n" + "=" * 78)
    print("DOOR-KICK INDEX — high-capacity RU railheads brought under interdiction")
    print("=" * 78)
    for sc in SCENARIOS:
        hit, tot = door_kick_index(axes, p, sc)
        print(f"{sc.key}  {hit}/{tot} high-capacity RU railheads interdicted")

    print("\n" + "=" * 78)
    print("LICTOR MESH PROGRAMME COST")
    print("=" * 78)
    for n in (1, 2, 6):
        nodes, lo, hi = lictor_cost(n)
        print(f"{n} axis/axes, 6 convoys each -> {nodes:4d} protected nodes"
              f"   ${lo/1e6:6.1f}M - ${hi/1e6:6.1f}M")

    print("\n" + "=" * 78)
    print("ALL-FRONTS TOTAL, 26 weeks (geometric sum vs force-limited)")
    print("=" * 78)
    for sc in SCENARIOS:
        sub = march[march.scen == sc.key]
        geo = sub["km2"].sum()
        top2 = sub.nlargest(2, "km2")["km2"].sum()
        print(f"{sc.key}  geometric-sum(all 6 axes) {geo:6.0f} km2  | "
              f"force-limited(best 2 axes) {top2:6.0f} km2")
