# LICTOR MESH — Mobile Escort Swarm for Autonomous Last-Mile Logistics

**A DEFENSIVE PUBLICATION.** Published to establish prior art and keep this
combination in the public domain. Use it, improve it, build it — but don't
lock it up.

- **Author:** Connor McSweeney
- **First public disclosure:** 2026-11-23
- **Canonical location:** `concepts/lictor-mesh.md`, github.com/connormmc777/ukraine-drone-forecast
- **Live pages:** dronespredictions.net → 🎼 Vivaldi × Botzkrieg (concept +
  schematic) and 🚚 Last-Mile Sim (the constraint model that tests it)
- **Licence:** CC0 1.0 Universal (public domain dedication). No patent claim is
  made or reserved by the author on the combination disclosed here.

> Lictors escorted Roman magistrates and carried the fasces — authority and
> enforcement in one bundle. Lictor Mesh escorts the logistics that enable
> Botzkrieg past the 90-km rail envelope.

---

## 1. The claimed combination

A **networked escort drone swarm that travels WITH an autonomous logistics
UGV** and defeats radio-FPV, loitering-munition, and **fiber-optic-guided**
attack drones before they reach the convoy.

The combination claimed as prior art is all five elements together:

1. A ground logistics UGV (or convoy of UGVs) operating beyond the rail
   sustainment envelope, in the last-mile FPV kill-zone;
2. A **co-travelling** escort swarm of multiple, functionally differentiated
   interceptor classes — not a stationary base-defence installation;
3. On-vehicle multi-modal detection explicitly including a sensor path for
   **RF-silent** (fiber-optic-tethered) threats;
4. A **low-bandwidth mesh** (e.g. LoRa) carrying only cue and track-handoff
   data between escorts, with terminal engagement executed by on-board
   autonomy on each escort — no single point of failure, no video backhaul;
5. A dedicated **tether-severing** class targeting the fiber-optic control
   link — the FPV threat class with no fielded Western counter.

## 2. The gap it fills

Fielded and announced counter-UAS is overwhelmingly **stationary base
defence** — Fortem DroneHunter, Anduril Roadrunner, and the Ukrainian and
Russian gun/EW systems protecting fixed sites. These protect a *place*.

The Vivaldi × Botzkrieg analysis in this repository shows the binding
operational need is different: **mobile protection for last-mile logistics
past the ~90-km rail envelope**, where trucks and UGVs are hunted by FPVs
in a 10–20 km deep grey zone.

Protecting a place is a different problem from protecting a moving thing
that must keep moving.

## 3. Four-class escort swarm per convoy node

> Note: an earlier draft of this disclosure said "three-class" while listing
> four. The disclosed combination is **four classes (A–D)**. The EW/decoy
> class D is part of the claim.

| Class | Role | Mass | Effector | Primary target |
|---|---|---|---|---|
| **A** | Kinetic ram interceptor | ~1.0 kg | Airframe itself / frangible nose | Radio-controlled FPV on terminal run |
| **B** | Net-caster | ~2.0 kg | Deployable net, rotor-tangling | Loitering munitions, larger multirotors |
| **C** | **Wire-cutter** | ~1.5 kg | Trailing cutter line / serrated boom | **Fiber-optic tether** — no Western counter exists |
| **D** | EW / decoy | ~3.0 kg | Video-link jam + false-target emitter | FPV operator's video feed during engagement |

**Loadout per protected convoy node:** 8–12 active escorts aloft or ready,
plus 5 in reserve on the UGV. Mixed by threat: FPV-heavy sectors weight A
and D; fiber-heavy sectors weight C.

## 4. Detection, mounted on the UGV itself

| Sensor | Detects | Why it's needed |
|---|---|---|
| Acoustic mic array | Rotor signature, 300–500 m | Works against RF-silent threats; cheap; all-aspect |
| mmWave radar (77 GHz class) | Small RCS (~0.01 m²) airframes | **The fiber-optic answer** — these emit no RF to scan for |
| Passive RF scanner | Control + video links | Longest warning against conventional FPV; free bearing |

**Mesh:** LoRa between UGV and escorts. LoRa carries **cue and track handoff
only** — bearing, range, class, commit flag. A few bytes per message. It is
explicitly *not* a video or radar-return backhaul; it cannot be. Terminal
guidance is on-board each escort. This is what makes the mesh survivable:
losing the UGV's radio does not disarm the escorts already committed.

## 5. Engagement envelope — corrected timeline

The honest timeline, because enablement matters for prior art:

```
  detect            ~1 s     (acoustic/radar/RF first contact)
  classify + commit ~1 s     (on-UGV fusion, threat class -> escort class)
  launch            ~1 s     (escort already spun up on rail)
  fly-out         6-10 s     (1 kg interceptor at 40-70 m/s over 400-800 m)
  ------------------------------------------------------------------
  total           9-13 s
```

**Therefore detection must happen at 800 m – 1.2 km, not 400–800 m.** An
attacking FPV closing at 33 m/s (120 km/h) covers 300–430 m during the
engagement timeline. The 400–800 m figure is the *intercept* envelope; the
*detection* requirement is roughly double it. A "3-second response" is
detect-to-launch latency, not time-to-intercept — stating it as the latter
would not be enabling.

This is the hardest requirement in the system and the one most likely to
decide whether it works.

## 6. Cost build-up per protected node

Auditable, rather than a single headline number:

| Item | Low | High |
|---|---|---|
| 13–17 escort drones @ $5–15k | $65,000 | $255,000 |
| Acoustic array + mmWave radar + RF scanner | $40,000 | $90,000 |
| Fusion compute + LoRa mesh + launch rails | $15,000 | $40,000 |
| **Total per protected convoy node** | **$120,000** | **$385,000** |

A 4-UGV convoy with Lictor Mesh: **$0.84M – $1.54M** capex.

## 7. ROI — the defensible version

**An earlier draft claimed ~$3–5M of supplies delivered per sortie and
positive ROI by trip three. That claim does not survive arithmetic and
should not be used.** A 4-UGV convoy at ~0.5 t per UGV carries ~2 t. Two
tonnes of mixed ammunition, fuel and rations is worth roughly **$40k–200k**,
not $3–5M. Cargo replacement value cannot justify this system.

The real case is **attrition and casualties**, over 10 sorties through a
15-km grey zone:

| | Crewed 4 t truck | Lictor-escorted UGVs |
|---|---|---|
| Per-trip delivery survival | ~55% | ~88% |
| Vehicles lost / 10 sorties | ~4.5 × $150k = **$675k** | ~1.2 × $200k = **$240k** |
| Crew casualties / 10 sorties | **~9** | **0** |
| Delivery variance (CV) | ~0.45 | ~0.18 |

The system pays for itself in **vehicles not lost, crews not killed, and
delivery you can plan around** — not in cargo value. That is a real case.
It is simply a different case than the one originally stated.

## 8. What this system does NOT do

Stated plainly, because the simulation in `lastmile_sim.py` tested it and
this is the honest result:

- **It does not extend the 90-km rail envelope.** The envelope is a
  *throughput* limit on the railhead-to-depot haul. The grey zone is a
  *survivability* limit on the final leg. Different constraints. Fixing the
  second does not move the first.
- **It does not neutralize the grey zone.** Russian FPV still owns that
  airspace. Lictor Mesh makes the grey zone *survivable for logistics*,
  which is a smaller and more achievable claim.
- **It is not a war-winning capability.** Under Theory-of-Constraints
  analysis the last mile is not currently the binding constraint on any
  front-line axis. See §9.
- **Its advantage is time-boxed.** Russia fields logistics UGVs and fiber
  FPVs already and can field counter-escort within roughly 6–12 months.

## 9. Where it sits in the constraint chain

From `lastmile_sim.py` (Theory of Constraints, four stations in series):

```
  [RAIL] railhead throughput
  [HAUL] railhead -> depot, fleet round-trip + interdiction   <- binds NOW
  [GREY] depot -> fighting line, the FPV kill-zone            <- Lictor Mesh
  [FORCE] operators + infantry to consolidate ground          <- binds everywhere else
```

The grey zone was the binding constraint during Vivaldi's peak phase
(Sept–Oct 2026, ~77 km from railhead). It is not the binding constraint at
Vivaldi's current depth of 95 km, where HAUL binds, nor on any secondary
axis, where FORCE binds. Elevating a non-constraint returns nothing.

**Lictor Mesh is necessary but not sufficient.** It is the right answer to
the wrong question unless it is paired with autonomous long-haul and
forward rail repair.

## 10. Prior art statement

This document discloses the combination in §1 publicly, with sufficient
detail to enable a skilled practitioner, as of the disclosure date above.
It is dedicated to the public domain under CC0 1.0. Independent invention
elsewhere is likely and is not disputed; this disclosure exists so that the
combination cannot be removed from public use.

If you build it, the only thing asked is that you say where it came from.

---

*Method: bounded arithmetic. Sources: ISW campaign updates Aug–Nov 2026,
Deep State mapping, RUSI (Watling) on Russian rail dependency, and the
`railheads.csv` / `salients.csv` / `vivaldi_ops.csv` datasets in this
repository. Simulation: `drone_app/lastmile_sim.py`.*
