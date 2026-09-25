# Multimodal travel: walk, bike-share, car and taxi

Until September 2026 every trip in the model was a car trip (`CAR_ONLY_MODE`).
Agents now choose between walking, bike-share, car and taxi, and each mode is
costed on its own street network. This page records what was built, what each
choice rests on, and what it deliberately leaves out.

## Who can use which mode

| Agent | Car | Taxi | Walk | Bike-share |
|---|---|---|---|---|
| Lives in the core, owns a car | yes, where the car is (see below) | no | yes | yes |
| Lives in the core, no car | no | yes, full fare | yes | yes |
| Lives outside the core, owns a car | yes | no | no | no |
| Lives outside the core, no car, **NYC** | no | yes, **half** fare | no | no |
| Lives outside the core, no car, elsewhere | not in the population | | | |

- **The core** is the principal city (Manhattan, Seattle, SF + Oakland, …). All
  workplaces and venues are inside it, so "lives in the core" is decided by the
  home alone (`Agent.lives_in_core`, from the core polygon).
- **Car ownership** is the agent's own PUMS household vehicle count.
- **Walking** is ruled out above age 85, and limited to 1 km for anyone with a
  PUMS ambulatory difficulty. **Cycling** is ruled out above age 75 and for
  ambulatory difficulty. Beyond those rules, walking stops at 5 km and cycling
  at 12 km (network plus access distance). These are physical ceilings, not
  behavioural ones: how far people are *willing* to walk comes from the time it
  takes, priced by their value of time. (The walking ceiling used to be
  `walk_tolerance_min / 12` = 1.25 km for everyone, because walking tolerance is
  pinned at 15 minutes; that one unsourced constant would have set the walk
  share.)
- **Unreachable means unavailable.** Walk and bike networks keep genuinely
  separate pieces (San Francisco and Oakland have no walkable link across the
  Bay), and a pair in different pieces has no walk or bike option. It is never
  priced at straight-line distance.

### The car goes where its owner drives it

Mode is chosen **per trip**, with one rule for owned cars:

- the car starts the day at home;
- an owner who drives away from home keeps the car, and car is the only mode
  until they drive back home;
- an owner who leaves home on foot or by bike cannot drive later that day — the
  car is at home;
- leaving home without the car is only offered if every remaining leg of the
  day can be walked or cycled (the day plan fixes the leisure venue in the
  morning, so the whole tour is known).

Bike-share and rides need no such rule: the bike is docked, the ride leaves.
This follows the vehicle-availability logic of tour-based models (Miller,
Roorda & Carrasco 2005, *Transportation* 32) without making mode a tour-level
choice.

### Taxi: how people without a car travel by car

A household without a vehicle never drives. Its car trips are **taxi** trips
(`Simulation._taxi_leg`):

- **Who:** every carless agent living in the core, in every metro; carless
  agents living outside the core only in NYC (`outer_taxi_metros`), where
  14.7% of agents are carless and live outside Manhattan, beyond the walk and
  bike networks.
- **Fare:** each city's regulated meter rate, a flag fall plus a per-mile rate
  (`params.TAXI_FARES`, sources beside each value). NYC's outside-core carless
  pay **half** the fare, a stand-in for the cheaper options they really use
  (subway, shared rides) until transit exists. Surcharges, slow-traffic time
  charges and tips are left out, so fares are somewhat understated.
- **Travel:** the drive network at car speed, under the same congestion, and
  counted in road volume; a 5-minute pickup wait; car emissions.
- **Everything else is car's:** comfort, status, enjoyment, community norm,
  the survey-measured attitude towards car, and car's leisure taste shifts.
- **No minimum distance and no penalty.** The old "ride" (car cost plus a
  2.2 utility penalty, for trips of 3 km or more) is gone. The fare and the
  wait are what make a short taxi ride unattractive, and charging a penalty on
  top would count the same cost twice.
- **Habit:** PUMS taxi commuters (JWTRNS 7) start with a taxi habit.

Carless agents outside the core anywhere but NYC have no mode at all, and are
left out of the population; the next row of the built population takes their
place (`Simulation._population_row_eligible`). All conditions of a study apply
the same rule to the same rows, so the paired No-RS comparison still pairs
identical agents. In reality most of these people take transit.

## Networks

Built by `welfare_rs/active_modes.py`, attached to the drive graph as
`RoadNetwork.mode_networks`, cached like the drive graph (GraphML, warm pickle,
routing matrices). See that module's docstring for the construction; in short:

- **Extent**: the core polygon with holes filled (enclaves such as Beverly Hills
  inside Los Angeles, Piedmont inside Oakland), buffered by 1 km so routes can
  use streets just past the city line and bridges reach the far bank.
- **Walk**: OSMnx's walk network (every edge both ways), keeping streets whose
  sidewalks are mapped as separate ways. OSMnx drops those, but separate
  sidewalks are often missing across bridges and causeways; in Miami that cut
  Watson Island off on foot while bikes and cars crossed.
- **Bike**: OSMnx's bike network plus footways where cycling is explicitly
  allowed, plus the reverse direction of one-way streets tagged
  `oneway:bicycle=no` (contraflow lanes).
- **Pieces**: every connected piece of at least 50 nodes is kept (its largest
  strongly connected part); smaller fragments are dropped so they cannot capture
  a home's snap. The drive graph's "largest component only" rule would have
  deleted Oakland.
- **POIs** are inserted as mid-block nodes, the same venues as the drive graph.
- **Routing matrices**: shortest-path metres over the core subset of the
  endpoint universe (`<metro>_<mode>_routing/`, `dist_m` only). Walk and bike
  route by length; time is distance over speed.

Agents keep their real census block points (`home_true`, `work_true`). Walk and
bike snap those, not the drive-snapped node, and the snap gap is charged as an
access leg at the mode's own speed. (Car keeps its 25 km/h access speed.)

`scripts/verify_active_networks.py <metro>` audits a metro's networks: every
piece and how many homes, workplaces and POIs land in it, snap distances,
unreachable share and detour factors. `scripts/verify_routing_matrix.py` now
checks the walk and bike matrices too.

## Cost of a trip

Nothing about how travel is valued changed; only the inputs did. Every mode
still pays generalised cost = time × value of time × time weight + money ×
cost weight, plus comfort, emissions, the MTTC motivation terms (derived-demand
penalty, enjoyment of travel time, escape, positionality), the community-norm
and survey mode biases, and the five decision paradigms.

- **Distance**: each mode's own network. Walking uses footpaths and ignores
  one-way rules; cycling obeys them except on contraflow lanes.
- **Speed**: walk 4.8 km/h and bike 14 km/h, flat, times the existing weather
  factors. (Age-specific walking speed and hills are deferred; see below.)
- **Money**: walking is free. Bike-share is priced as an annual member rides:
  nothing for the first 45 minutes, then $0.27/min — Citi Bike's 2026 member
  terms, applied to every metro until per-system fares are added. The annual fee
  is sunk and does not enter a per-trip choice.
- **Emissions**: zero for walking and cycling. Road congestion (BPR) counts only
  cars, so shifting trips off the road relieves it.
- **Alternative-specific constants** (`params.MODE_ASC`, per metro) are
  calibrated; see below.

### Planning prices venues at the generalised cost the agent can expect

When an agent plans its evening, each candidate venue used to cost a flat 0.12
per km of drive distance, whatever mode would carry the agent there. Now each
leg is priced at the **generalised cost this agent can expect**:
`Σ_m P_m · C_m`, where `C_m` is time × value of time + money (the paper's C in
`U = V − C`) and `P_m` is the probability the agent takes mode m under its own
decision rule (`Simulation.mode_choice_probabilities`, which mirrors
`choose_mode` without drawing). A 600 m walk is cheap for a carless core
resident; a suburban driver pays the drive. The whole round trip is priced,
as the predecessor paper defines C ("reaching and returning from the activity
location"). The accept-or-decline comparison with a recommendation uses the
same price, as the paper requires, and the price is exactly what the welfare
metric later charges the realised trip (`activity − gen_cost / 8`). It remains
the ex-ante view: free-flow roads, whereas the trip is charged under the
congestion it meets. The agent's trip-burden multiplier scales the cost.

**Two alternatives were tried and rejected on measurement:**

- **The logsum** `ln Σ_m exp(V_m)` (Ben-Akiva & Lerman 1985). Its choice-value
  term is up to ln 3 ≈ 1.1 per leg with three modes at this model's utility
  scale — as large as the travel cost itself — and no realised trip is ever
  credited with it. Miami core car owners' mean round-trip price went from
  +1.11 to −0.41 (trips looked net-positive), and participation rose from
  0.30 to 0.40.
- **Expected total travel utility** `Σ_m P_m · V_m`. `V` carries the mode
  constants, which a choice model identifies only relative to car (car = 0);
  their level is arbitrary. Priced as a level, calibrated walk and bike
  constants and the NYC ride penalty made every non-car option look
  absolutely worse, and NYC participation fell to 0.06. Constants still decide
  WHICH mode the agent expects to take, through `P_m`.

Because the scale of the planning price changed, `beta_delta_cost` was
re-derived the way it was originally set (Manhattan, 400 agents, 6 days,
proximity-led vs footfall-led; `scripts/measure_delta_cost.py`): 0.195 divided
by the mean cost gap between the two rankers. With taxis the gap is 1.93
(seeds 42 and 43), driven by a long tail of distant venues reached by taxi, so
the value is **0.10** (was 0.30 with per-km pricing, then 0.22 before taxis).

### The five decision rules now weigh the same things

Agents differ in HOW they choose a mode, not in WHAT they care about. Every
rule works from the same travel utility, split into two groups
(`Simulation._mode_outcome`):

- **attributes** of the mode on this trip: time and money (priced by value of
  time and cost sensitivity), comfort, emissions, the time-based MTTC terms
  (derived-demand penalty, enjoyment of travel), escape and status;
- **constants**, the agent's standing leaning towards a mode: the calibrated
  constant, community norm, survey-measured mode attitude, leisure-type taste
  shift and the no-car penalty.

| Rule | How it chooses | Source |
|---|---|---|
| Utility | Logit over total travel utility | McFadden 1974 |
| Regret | **Switchable, pending a decision** (`params.DECISION_RULE_PARAMS["regret_rule"]`). Default `"chorus_rrm"`: random regret minimisation, each mode compared with every other attribute by attribute, `R_i = Σ_j Σ_m ln(1 + exp(a_jm − a_im))`, logit over −R plus constants. Alternative `"max_utility_gap"`: the predecessor paper's `R_i = max_j U_j − U_i`, lowest taken — which always picks the highest-utility mode | Chorus 2010 (cited by GeoRecSim); Uğurel & Yabe 2026 Eq. 6 / SI Eq. 4 |
| Prospect | Utility plus reference-dependent gains and losses in time and money, losses weighted by loss aversion | Kahneman & Tversky 1979 |
| Satisficing | Search modes (habitual first, rest random); take the first whose full cost (travel utility in dollars) is within the aspiration threshold, else logit | Simon 1955; Caplin, Dean & Martin 2011 |
| Habit | Repeat the habitual mode if available, else logit | Gärling & Axhausen 2003 |

The predecessor paper (Uğurel & Yabe 2026, arXiv 2608.16922, and its SI)
writes regret down only for the recommender's filter, over venues. Applied to
agents' mode choice it makes regret agents pick the best mode every time, the
utility rule without its random term. Chorus's model is the standard form of
regret theory for choices between options described by attributes, and keeps
regret a distinct rule. Both are implemented; `MODE_ASC` and `beta_delta_cost`
are fitted under `"chorus_rrm"`, so switching means re-running
`scripts/calibrate_mode_constants.py` and `scripts/measure_delta_cost.py`.

**Where the styles act.** Only on mode choice (`decision_paradigms["mode_choice"]`).
Venue choice and recommendation acceptance use the same rules for every agent;
the style reaches them only through the expected travel cost of a venue. Under
the old car-only model the style was never consulted at all, so any
difference between styles in car-only results is chance.

Three things were wrong in the old rules and never showed while every trip was
a car trip:

1. **Regret picked the most expensive mode.** It summed how much *worse* the
   other modes were (`max(0, cost_j − cost_i)`) and took the minimum, so with
   car at $10 and walk at $20 it walked. That is why regret agents walked 54% of
   Miami commutes.
2. **Regret and satisficing saw only money and time.** A bike-share member ride
   costs $0, so they chose bike whatever else it cost: regret agents biked 28%
   of commutes even with the sign fixed, and no mode constant could move them,
   which made calibration impossible (Miami bike stuck at ~10% against 1.1%
   observed). Regret on total utility instead (`max U − U`) would pick exactly
   what a utility maximiser picks; comparing attributes one by one, as Chorus's
   model does, is what keeps regret a distinct rule. It is also the form the
   predecessor paper (Uğurel & Yabe 2026) and GeoRecSim both point to: regret
   over the full value of the options.
3. **Satisficing searched in `MODE_PARAMS` order**, which lists walk first. It now
   searches the habitual mode first, then the rest at random.

Also fixed:

- **Habit** starts from the agent's own PUMS commute mode (`JWTRNS`: car, taxi
  and motorcycle → car, bicycle → bike, walked → walk; transit and work-from-home
  → no habit).
- **Access legs** are resolved by location rather than activity type. A
  remote-work day's "work" activity is at home, and was being charged the
  workplace's snap gap.

## Calibration

`scripts/calibrate_mode_constants.py` fits one walk and one bike constant per
metro (car is the reference) so that the simulated commute walk and bike shares
of core-resident workers match those same agents' PUMS commute modes (work
from home and "other" excluded, transit riders kept in the denominator).
Shares are computed as exact expected probabilities under each agent's own
decision rule, so the fit has no sampling noise. Results are in
`params.MODE_ASC`, with the observed and fitted shares beside each value.

**Only commuters with a choice are fitted.** A constant can only move someone
with two or more permitted modes. Commuters with one are reported separately
and left out of both the simulated and the observed side. Fitting over everyone
made bike unreachable in Miami: 2.1% of its core commuters are bike-only
(carless, commute too far to walk), which is more than the 1.15% who really
bike, and the bike constant was driven to its −15 bound — which would have made
cycling all but unchoosable for everyone who does have a choice.

**Forced bike-share commutes (resolved by the taxi).** Before the taxi mode, a
carless person outside NYC whose commute was within cycling reach but beyond
walking reach could only cycle (7.6% of DC's core commuters, most of whom
really ride transit). They now choose between bike-share and taxi.

With no transit mode, calibration can only match walk and bike shares; "car"
absorbs everyone else, transit riders included. In Manhattan that makes car
the dominant simulated commute mode although few Manhattan residents drive to
work — the largest known distortion of this version.

## Effect on results (2026-09-23)

Same agents, same seed (42), 1,000 agents × 3 days, default recommender
stack; car-only is the `car-only-final` tag. Final model: walk, bike-share,
car and taxi; calibrated constants; `beta_delta_cost` 0.10.

| | Miami | NYC | DC |
|---|---|---|---|
| Leisure participation, car-only → multimodal | 0.289 → 0.260 | 0.276 → 0.112 | 0.290 → 0.160 |
| Leisure net utility (activity − cost) | 0.364 → 0.306 | 0.154 → 0.365 | 0.146 → 0.289 |
| Taxi share of all trips | 0.7% | 28.3% | 3.3% |
| Leisure trips walked / cycled | 13.5% / 1.5% | 54.5% / 2.7% | 26.3% / 3.1% |
| Legs with no permitted mode | — | 0 | 0 |

**Fewer outings, each worth more, in NYC and DC.** The car-only planner priced
a venue at 0.12/km, but the trip was then charged time × value of time +
money. In the two cities with long, slow trips and high values of time, agents
set out on outings that left them worse off (realised net utility 0.15). Now
planning and charging use the same cost, so agents skip outings that are not
worth it.

**NYC falls furthest because it has no transit.** Most New Yorkers are carless.
Without a subway they walk short trips and pay taxi fares for the rest ($3.50 a
mile in Manhattan, half that for carless residents outside it), so going out
looks, and is charged as, far costlier than it really is. The 31% of NYC
commutes made by taxi are overwhelmingly subway riders in reality. NYC results
should be read with this in mind until transit exists.

The no-database demo (`scripts/run_demo.py`) runs on a legacy preset with no
calibrated constants, so its mode shares are uncalibrated.

## Deferred

- **Transit.** The dominant commute mode in NYC, DC, SF and Chicago's core.
- **Hills.** Flat speeds overstate cycling in San Francisco and Seattle.
- **Bike-share docks and per-system fares.** Bikes are available anywhere in the
  core at Citi Bike member prices.
- **Age-specific walking speed.**
- **Out-of-sample validation** of leisure mode shares and trip lengths against
  NHTS.
