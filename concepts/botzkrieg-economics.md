# BOTZKRIEG ECONOMICS — Ground as a cost-exchange problem

**The thesis in one sentence.** Ground is not retaken by manoeuvre any more; it
is retaken by whichever side can hold the cheapest exchange rate across every
station of the sustainment line at once, and the side that runs that
optimisation as a *closed loop* — re-finding its binding constraint faster than
the enemy can counter-adapt — takes ground from the side that runs it as a
procurement list.

- **Author:** Connor McSweeney
- **Date:** 2026-11-25
- **Canonical location:** `concepts/botzkrieg-economics.md`
- **Companion:** `concepts/lictor-mesh.md` (the escort-swarm concept this paper prices)
- **Model:** `drone_app/lastmile_sim.py`, live at dronespredictions.net → 🚚 Last-Mile Sim
- **Licence:** CC0 1.0 Universal

---

## 1. The claim, stated formally

Let **u** be the control vector a force actually owns: convoy packet size,
operating window, escort density, interceptor price point, uncrewed vs crewed
mix at the point of contact, artillery effort spent contesting the grey zone,
and holding density behind the line. Let **B** be budget and **K** a casualty
ceiling.

    maximise   J(u) = Σ_t  A(u, t)              [km² held at time t]
    subject to Σ cost(u) ≤ B
               casualties(u) ≤ K
    where      A(u, t) = min( RAIL, HAUL, GREY, FORCE )

The objective is a **min()**. That single fact drives everything below: the
gradient of J is zero in every direction except the one acting on the station
that is currently lowest. Money spent anywhere else is not a small gain — it is
*exactly* zero gain. The Last-Mile Sim exists to find which direction is
non-zero, per axis, per week.

The thesis adds the second half: because the adversary responds, the binding
station **moves**. So this is not an optimisation, it is a control problem with
a drifting setpoint. A procurement programme solves it once, open-loop. A
self-regulating system solves it continuously, closed-loop. Over a year of
adaptation cycles, the closed-loop force takes ground from the open-loop one
even with an identical budget.

## 2. The objective is three-termed, and two of the terms fight

Land retaken, casualties, and money are not separable. The honest objective is
**km² held per casualty per million dollars**, and the classic infantry answer
maximises the first term by spending the second. Robot-first sequencing —
machines take and screen the ground, people arrive afterwards at the lowest
density that holds it — is an attempt to *decouple* those terms rather than
trade them.

That makes the programme measurable. The test of a Botzkrieg force is not
"did it advance" but:

    casualties per km² retaken, and $ per km² held per month, both falling.

If either rises while the line moves, the programme is buying ground with
bodies and has failed its own objective, whatever the map says.

## 3. Where robots actually attack the four stations

The uncomfortable result from the model, and the reason this paper exists:
across six real axes and seven scenarios, **GREY — the last mile — is never the
binding station** at the depth those axes actually sit. Not once. Logistics
robots, escort swarms and autonomous convoys all elevate a station that is
already not the lowest.

| Station | What binds it | Can robots raise it? |
|---|---|---|
| RAIL | Railhead throughput, track gauge, repair capacity | Barely. Engineering, not autonomy. |
| HAUL | Round-trips and interdiction past ~90 km | **Yes** — 22 h/day instead of a 12 h night window is worth 1.27× |
| GREY | Survival through the 10–20 km FPV kill-zone | Yes, and it buys 1.04× because it was not binding |
| FORCE | Troops to consolidate and hold | **This is the one that binds, and it is the hardest to automate** |

Over 26 weeks on the two axes Ukraine can actually resource, the model gives
1,077 km² under the status quo and **1,077 km² with robot convoys and escort
swarms** — identical through five scenarios of logistics spending. Forward rail
repair gives 1,283. Doubling force generation gives 2,079.

So the thesis is right in its sequencing — robots take, humans hold — but the
leverage is not where the hardware industry is pointed. **The Botzkrieg
programme that matters is not the one that moves supplies. It is the one that
lowers the number of human beings required to hold a kilometre of ground.**

## 4. Holding density is the real target

Advancing multiplies the frontage you have to hold. Take the calibration axis:
a 13.4-km-wide wedge, status quo, 26 weeks. It gains 172 km², which is 12.9 km
of depth — and that depth creates 25.7 km of new flank. Frontage to hold goes
from 13.4 km to roughly 39 km. **A 2.9× holding bill for the ground taken.**

That is why offensives culminate on a schedule rather than at a terrain
feature. The model's force ceiling decays with an 8-week constant:

| Week | 1 | 4 | 8 | 13 | 26 |
|---|---|---|---|---|---|
| FORCE ceiling (km²/wk) | 90 | 66 | 48 | 36 | 22 |

Assault groups are consumed; the ground they took has to be garrisoned by the
same pool. Depth eats its own enabler.

Now put a holding density **h** (troops per km of perimeter) on it. *These
figures are illustrative, not fitted — the model has no h term and this is the
weakest arithmetic in the paper:* at h = 40/km, that wedge goes from ~540
troops to ~1,560 to hold what it took. Cut h to 15/km with sensor pickets,
remote weapon stations, uncrewed screening vehicles, persistent loitering
munitions and mined approaches, and the same wedge holds for ~590. That is not
a logistics saving. **It is the difference between an offensive that culminates
in week 13 and one that does not.**

Every km²/week in the FORCE column is a statement about h. Nothing else in the
model moves that column.

## 5. The interior optima — where the maximisation actually lives

The thesis names the right levers. Each one has an interior optimum, which
means each can be over-bought as easily as under-bought.

**Packet size.** Dispersing a convoy into *n* smaller packets lowers the value
at risk per prosecuted target, but raises cost per ton: more vehicles, more
escort nodes, more hours, more coordination. Take per-packet overhead *a* and
convoy value *V*:

    C(n) ≈ a·n + b·V/n      →     n* ≈ √(bV / a)

Optimal dispersal scales with the **square root** of what the convoy is worth.
That is a toy, but it kills two intuitions at once: "disperse everything" is
wrong, and so is "one big escorted push".

**Operating window.** Crewed convoys get a 12-hour night window; autonomy
allows 22. But threat intensity λ(t) is not flat, and the marginal hour added
is the *worst* remaining hour. Run until the expected loss in the next hour
exceeds the value it delivers — not until the clock runs out. Autonomy's real
gift is not 22 hours; it is the freedom to *choose* which hours, and to stop.

**Interceptor price point.** An interceptor is worth fielding when
`cost_interceptor < P(kill) × (threat cost + value it would have destroyed)`.
Against a $600 radio FPV about to kill a $200k UGV carrying $1M of cargo, the
ceiling is high and almost any interceptor clears it. Against a $3,000
fibre-optic drone hunting a $40k pickup, it does not. Interceptor mix should be
priced per threat class, not bought as a single line item.

**Escort density.** The model's own attrition line: ten sorties through a 15-km
grey zone cost ~4.5 crewed trucks (~$675k) and ~9 crew; Lictor-escorted UGVs
cost ~1.2 vehicles (~$240k) and no crew. Escorting one 4-UGV convoy costs
$840k–1.54M at the concept's node prices, so **the escort pays for itself after
roughly 19–35 sorties on vehicle attrition alone** — $58M–177M of supplies
moved — before counting the nine crew, who are the point.

That is a sound attrition case. It is still not a manoeuvre case, because GREY
was not binding. Both things are true at once, and the programme should be sold
on the first, not the second.

## 6. Impose cost on the enemy's constraint, not on his cheapest input

The thesis says: minimise your cost relative to what Russian drones can do. The
sharper version applies Goldratt in the other direction — **find the enemy's
Herbie too.**

Russian airframes are the cheap input: $400–1,000 for radio FPV, $1,500–3,000
with fibre. You will never win a dollar war against a factory output. What is
genuinely scarce on that side is **trained operator crews, ISR coverage, and
the bandwidth to prosecute separate targets**. Dispersal, deception, timing and
jamming are valuable precisely because they tax *that* input: they force more
operator-hours and more ISR dwell per ton of Ukrainian supply destroyed.

So the cost-imposition metric should not be dollars. It should be
**operator-hours and ISR dwell-minutes denied per ton delivered**. A measure
denominated in dollars will keep recommending that you out-buy a drone factory,
which is the one competition that cannot be won.

## 7. The self-regulating system

The closed loop, stated as a procedure:

1. **Measure** the four stations on each axis, weekly, from observed data.
2. **Identify** the binding one. (Today: FORCE on four axes, HAUL on two.)
3. **Elevate** it at least cost — and only it.
4. **Expect decay.** Every elevation is counter-adapted. The model's adaptation
   path decays a grey-zone gain with a 12-week constant once the enemy fields a
   counter; the companion concept estimates 6–12 months before an escort swarm
   is countered at all.
5. **Re-measure**, because step 3 moved the constraint somewhere else. It
   always does. Elevate HAUL and FORCE binds. Elevate FORCE and RAIL binds.

The winning condition is a latency inequality:

    τ_own re-optimisation cycle  <  τ_enemy counter-adaptation cycle

A force that re-finds its constraint monthly and a force that re-finds it
annually can hold identical inventories and produce completely different front
lines. **This, not any particular robot, is the thesis.** The hardware is
replaceable; the loop is the weapon.

Note the corollary: a self-regulating system has to be allowed to *stop*
things. A programme that can add escorts but cannot cancel them when GREY stops
binding is open-loop with extra steps.

## 8. What this predicts going forward

**1. The lines mostly stay where they are.** Both sides are FORCE-limited and
neither can convert money into that input quickly. The model's own verdict:
under current parameters, six axes, no scenario of logistics spending moves the
total at all.

**2. The two sides have different Herbies, so symmetric advice is wrong.**
Russia's constraint is rail-tethered logistics; its surplus is manpower.
Ukraine's constraint is manpower; its surplus is technical adaptation speed.
Each side's correct investment is the one the other side cannot make. This is
why a Ukrainian Botzkrieg programme is not a luxury item: substituting capital
for manpower is the *only* move that attacks Ukraine's own binding station.
And it is why Russia does not run one — not because it could not buy the
hardware, but because automating the station you have in surplus buys you
nothing. The same Goldratt rule, read from the other chair.

**3. Ground only moves when three things happen together.** Forward rail repair
(moves the envelope), a fall in holding density h (makes depth affordable), and
a loop that reallocates between them faster than the adaptation cycle. One or
two of the three produces a bulge that culminates on schedule. The model's
stall depths say the same thing quantitatively: 107 km under the status quo,
107 km with escorted autonomous logistics, 150 km with rail repair.

**4. The decisive axis needs no trucks.** Orikhiv → Melitopol is 112.8 km
railhead-to-railhead, the shortest reach to a high-capacity Russian hub on the
whole front, and interdicting it needs 92.8 km of depth against a 107 km
status-quo stall depth. The geometry is already solved. The force to hold it is
not. Trucks delivered there would be a textbook mirage.

## 9. What to buy, in order

1. **The loop itself** — weekly per-axis measurement of all four stations, and
   the authority to reallocate against it. Cheapest item on the list and the
   only one that compounds.
2. **Holding-density reducers** — sensor pickets, remote weapon stations,
   uncrewed screening vehicles, persistent munitions, rapid obstacle emplacement.
   This is where "robots take, humans hold" earns its keep, and it is the FORCE
   column, not the GREY column.
3. **Forward rail repair and re-gauging capacity** — the only lever that moves
   the envelope instead of stretching it.
4. **Autonomous long-haul** (the 12 h → 22 h window, 1.27×), ahead of escorted
   last mile.
5. **Escorted last mile**, concentrated on the one axis where GREY is *about to*
   become binding — never spread across six. Bought on the attrition case:
   vehicles not lost, crews not killed, delivery you can plan around.

## 10. Assumptions, weaknesses, falsifiers

**Fitted:** three parameters on five observations from one campaign. Everything
downstream is extrapolation from two anchor points.

**Assumed, not fitted:** every holding-density figure in §4; the dispersal and
operating-window optima in §5 are structural toys with no data behind their
coefficients; adversary adaptation times; all cost figures are order-of-magnitude.

**Known weaknesses.** Russia is static in the headline runs, which overstates
every gain. Straight-line distances understate road distance by 15–30%. There
is no terrain, weather, mine or river term — a 20-km advance across the
Zaporizhzhia steppe and 20 km into Horlivka's urban sprawl are not the same
event. And km² is a poor measure of success; it is used because it is what the
data records.

**Falsifiers.**
- An axis given escorted last-mile robots more than doubles its rate with no
  change to haul or force → GREY was binding after all and §3 is wrong.
- A force sustains 35–45 km²/week past 120 km from its railhead for 8+ weeks →
  the haul curve is wrong and the envelope does not bind.
- A sector holds at materially lower troop density **without** a fall in the
  lines held, over a season → §4's h-term is real and measurable, which would
  make this the most important number in the model rather than the least
  supported one.
- Russia fields a functioning closed-loop robotic logistics and holding system
  at scale within 12 months → §8.2 is wrong and the asymmetry argument fails.

## 11. The Marion Formula — combat interconnection, and the shifting of the scale

Sections 1–10 argue in words and arithmetic. This section states the same thing
as a **two-sided** problem, because the optimisation in §1 is incomplete on its
own: the adversary is optimising against you, on the same clock, with the same
kind of reasoning.

**The objects.**

- **a(t)** — your allocation across n levers: escort density, long-haul
  autonomy, rail repair, holding automation, dispersal, timing. Sums to your
  budget.
- **d(t)** — the adversary's allocation across m counters: radio FPV, fibre
  FPV, operator training, ISR dwell, EW, artillery. Sums to theirs.
- **M(t)** — the n × m **exchange matrix**. M_ij is the cost imposed on the
  adversary per unit cost you spend, when lever i meets counter j. A cell above
  1 is a cell you are winning.
- **S(a, t)** — the four station capacities from §3.

**The scale is a matrix game, not a ratio.**

    V(t) = max over a of min over d of   a' M(t) d  /  (|a| |d|)

You maximise what the adversary minimises, over one shared matrix. Neither side
sets V; the pair sets it. That is the interconnection, and it is why quoting a
single exchange ratio — a $600 airframe against a $200k vehicle — is a category
error. That ratio is one cell. The front runs on the saddle point.

**The gate: winning the exchange is not taking ground.**

    dG/dt  =  min over k of  S_k(a, t)   ·   g(V − 1)

with g rising from zero: ground accrues only while the exchange is survivable
*and* the binding station is one the allocation is actually moving. This is §1
and §6 compressed into a line.

**The dance: why the scale shifts under you.**

M is not fixed. Use reveals. Every time lever i is used against counter j, the
adversary observes it, prices it, and buys it down:

    dM_ij/dt  =  − λ_j · u_ij(t) · M_ij  +  μ_i(t)

- **u_ij** — exposure: how hard that lever is being used against that counter.
- **λ_j** — the adversary's counter-adaptation rate. The companion concept puts
  6–12 months on countering an escort swarm; the model decays a countered gain
  with a 12-week constant.
- **μ_i** — your own innovation rate, restoring the cell.

Their matrix obeys the mirror equation with your λ. Two coupled flows over one
shared matrix, each side's optimum feeding the other's decay term. That is the
dance, and it has no fixed point in the hardware — only in the rates.

**Three consequences, which are why it is worth writing down.**

1. *An advantage has to be rationed.* Exposure sits inside the decay term, so
   using a winning cell consumes it. The optimal use rate is interior: hard
   enough to matter before the cell decays, restrained enough not to teach the
   counter. That is an exhaustible-resource problem, not a procurement decision,
   and the usual acquisition vocabulary cannot express it.

2. *The two optimisations are coupled through your own allocation.* **a** moves
   the gate and the decay at once. Spending to win a ratio can close the gate
   and burn the cell in the same quarter — §1's mirage, restated with a
   derivative.

3. *Stasis is an equilibrium, not a failure.* When both sides are gated at a
   station neither can elevate — force generation, for both — dG/dt goes to zero
   while V oscillates near 1 and both treasuries keep emptying. A static front
   line is not an absence of optimisation. It is two optimisations cancelling.

**The condition for taking ground.** Over a horizon T, ground goes to the side
that sustains

    E[ min over k of S_k(a, t) · g(V − 1) ]  >  0

which needs three things at once, and this is the formula's whole content:

- the binding station is one the budget can actually move (§3);
- μ beats the adversary's λ in the cells being depended on — innovation faster
  than counter-adaptation (§7);
- the re-allocation period is shorter than their counter-adaptation period, so
  **a** sits on the right cell during the window when V is above 1.

Fail any one and the expression collapses to the static front of §8.

**What it is not.** λ, μ and u are estimated nowhere in this repository, and M
is not measured. This section is structure, not calibration — a bookkeeping
device that forces both sides, the gate and the clock into one expression. Its
practical value is that it names what would have to be measured: exposure,
counter-adaptation times, and the exchange cells themselves.

---

*Method: bounded arithmetic. Sources: ISW campaign updates Aug–Nov 2026, Deep
State mapping, RUSI (Watling) on Russian rail dependency, and the
`railheads.csv` / `salients.csv` / `vivaldi_ops.csv` / `axes.csv` datasets in
this repository. Simulation: `drone_app/lastmile_sim.py`. Every quantitative
claim above is reproducible from that file.*
