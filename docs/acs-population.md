# The ACS/PUMS agent population

How an agent gets built, which source supplies which part of it, and what the
result does and does not represent. Written 2026-09-21, alongside the build.

The short version: an agent is a real commute from LODES, a real person from
ACS PUMS reweighted to that commute's home tract, and a real survey respondent
matched to that person's demographics. Nothing about an agent is invented, and
nothing is sampled while a simulation runs.

---

## 1. Why three sources

No public dataset holds person-level detail at block level — that would
identify people, so the Census does not publish it. Every approach is therefore
some way of combining a coarse joint distribution with fine-grained local
totals. The three sources are used for what each alone can do:

| Source | Gives | Cannot give |
|---|---|---|
| **LODES** OD | Home block → work block, weighted by jobs; the job's age band, earnings band, industry sector | Anything else about the person; only covers workers |
| **ACS PUMS** | One real person's age, own earnings, household income, vehicles, disability, household size — jointly | Location finer than a PUMA (100k+ people, ~34 tracts) |
| **ACS tables** | Exact local totals per tract: income distribution, vehicles, age/sex, employment, disability, commute mode | Any joint structure — one margin at a time |
| **The survey** | Psychometrics, platform attitudes, the behavioural coefficients | Geography; a national sample of 473 |

The whole LODES OD record is `w_geocode, h_geocode, S000, SA01-03, SE01-03,
SI01-03` — a home, a workplace, and three coarse splits. It places an agent and
describes almost nobody.

## 2. The pipeline

```
db/fetch_census_data.py     ACS tables + PUMS (14 states) + tract→PUMA crosswalk
       |                    every file SHA-256'd into manifest_acs_pums.json
       v
db/load_pums.py             census.pums_persons / pums_households / tract_puma
db/load_acs.py              census.acs_estimates (long format)
       |
       v
db/build_tract_weights.py   census.pums_tract_weights   <- IPU, the fitting step
       |   --what validate  checks every recode against published state totals
       |   --what home-blocks  public.metro_home_blocks (core blocks)
       v
db/build_population.py      public.metro_population     <- one row per agent
       |
       +-- db/export_population.py -> data/population/*.csv  (no-database path)
       v
welfare_rs.datasource.population()  ->  Simulation(population_rows=...)
```

Verified by `scripts/verify_acs_population.py`.

## 2a. Two geography mismatches, both resolved

`db/build_tract_weights.py` resolves each tract to the ACS geography that
actually describes it, and stops if it cannot:

* **Connecticut.** In 2022 the state replaced its eight counties with nine
  planning regions as county-equivalents. ACS 2023 publishes CT tracts under
  planning-region prefixes (09110–09190) while LODES and TIGER 2020 blocks
  still carry the old county FIPS. The 6-digit tract code survives the change,
  and all 423 CT home tracts match exactly one ACS tract on it. Without this,
  every Connecticut commuter into New York — 34,341 jobs — would be unplaceable.
* **Tracts revised after the 2020 TIGER vintage**, which ACS 2023 no longer
  publishes: 15 tracts in Suffolk County, NY, carrying 0.03% of jobs. They fall
  back to their county's control distribution, scaled by the tract's share of
  county housing units. Weaker than a real tract fit, and reported as such.

## 3. Fourteen states, not seven

The metros sit in seven states, but their commuter sheds reach seven more.
Grouping `public.metro_od_pairs` on the home state gives: New Jersey 3.5% of
all jobs, Maryland 2.7%, Virginia 1.7%, Connecticut 0.4%, Indiana 0.3%,
Pennsylvania and Delaware 0.01% each — 4,856 tracts in total. PUMS and ACS are
needed wherever an agent **lives**, not where it works, so all fourteen load.
Both loaders stop rather than fit a tract with no controls behind it.

## 4. The fitting step (IPU)

For each of the ~28,400 home tracts, the PUMS sample of its PUMA is reweighted
until its totals match that tract's ACS figures, using Iterative Proportional
Updating (Ye et al. 2009 — the method behind PopGen and ActivitySim). The
household stays intact; only how many of it the tract is thought to hold
changes. So the joint structure survives while the marginals become local.

IPU rather than plain IPF because the controls are of two kinds — households
(income, vehicles, size) and persons (age, sex, employment, disability,
commute) — and one weight per household has to satisfy both at once.

**Fitted at tract level, not block group.** Measured on Manhattan's 1,292 block
groups: the average holds ~600 households with a ±33% margin of error on that
total, and a single income bucket inside it carries a margin of error of **134%
of the estimate**. Fitting to numbers that noisy fits noise.

**The controls contradict each other, and that sets the accuracy floor.** ACS
publishes each table from its own weighting, so at tract level they do not
reconcile. For tract 36047003000 the household-size table (B11016) implies *at
least* 1,954 people living in households while the population table (B01001)
reports 1,818 for the whole tract — a 7% contradiction between two published
figures. No set of weights satisfies both, so the fit settles on a compromise
and a residual remains. Consequences, all measured rather than assumed:

* More sweeps do not help. On the worst tracts, 200 sweeps land no closer than
  50 and sometimes further, because the weights oscillate between conflicting
  controls. The fit therefore caps at 50 sweeps and **keeps the best sweep, not
  the last**.
* "Converged" means every control within **5%**, or within 5% of 25 units for
  cells smaller than that. A tighter bar would be chasing noise: ACS itself
  publishes small tract cells with margins of error of 50–130%.
* About 80% of tracts reach that bar. The rest are fitted as well as the sample
  allows, flagged `converged = false` in `census.pums_tract_weight_meta`, and
  reported by `scripts/verify_acs_population.py` rather than hidden.
* Residuals land on person controls, not household ones — the household side
  (income, vehicles, size) matches almost exactly. "Person controls" are the
  targets counted in people (age x sex, employment, disability, commute mode)
  as against those counted in households (income, vehicles, size); one weight
  per household has to satisfy both, since a household of four contributes 1
  to every household count and 4 to every person count.
* The extreme tail is group quarters. The worst tract has **36 households and
  1,640 residents** (a prison or campus), another **0 households and 2,945
  people**; 13 of the 73 tracts above 20% error look like this. A sample
  holding few group-quarters records cannot put 1,600 people into 36
  households.
* **Nothing is dropped.** A tract that misses the bar still gets the closest
  weights the sample allows and is flagged `converged = false`; no tract, agent
  or record is discarded anywhere in the build.

**Universes are enforced, not assumed.** Household controls are fitted over
occupied housing units only; group-quarters records carry no household weight
and no household income (verified: 447,021 GQ records, all at weight 0) but
their *persons* carry full person weights, so person controls include them.
`--what validate` sums each recode over a whole state and compares with the
same ACS control summed over that state's counties. That check earned its
keep: it showed every person control sitting 2–7% low, which turned out to be
PUMS calibrating household and person weights separately — validation must use
person weights, while the fitting correctly *starts* from household weights.

## 5. Workers and non-workers are placed differently

A worker's leisure trip departs from its **workplace**, which is always in the
core, a few km from the venues. A non-worker has no workplace, so its trip
departs from **home** (`welfare_rs/agent.py`, `origin_after_work`).

Drawing non-worker homes across the whole metro would put much of the
population 30–60 km from every venue in the catalog, across a shell layer that
carries arterials only, and those trips would dominate every distance-normalised
metric — while describing a behaviour the model does not represent, since a
suburban non-worker's real options are suburban venues that are not in the
catalog.

So **non-workers are drawn from core blocks only**, weighted by each block's
expected count of non-working adults (block population × the tract's rate of
16+ adults not employed). The population is therefore *people whose day is
anchored in the core*: they work there, or they live there.

The worker share is not a parameter. It is core jobs against core-resident
non-working adults, both measured.

**Known gap:** core residents who work *outside* the core belong in this
population — they come home to the core in the evening — but LODES pairs were
filtered to work-block-in-core and no longer carry them.

**Remote workers follow the same rule (2026-09-28).** `agent.py` sends a
remote worker's leisure trip out from home, and worker homes are metro-wide
LODES homes, so the long home→venue trips the core-only rule exists to prevent
used to occur at the remote-work rate (12–35% of employed agents on a given
day). Now only core residents work remotely (`Agent.plan_day`); a worker living
outside the core commutes every day, and its leisure trip departs from its
core workplace. Every home→venue trip therefore starts in the core. The cost is
a lower remote-work share overall: remote days now apply only to the core
residents among workers.

## 5a. Car commutes are capped at what people actually drive

LODES records *where* people commute, never *how*. When this cap was set the
model drove every pair it drew (car-only), and the raw OD tail contains commutes
nobody makes daily by car: NYC's longest pair is **488 km**, and 9.5% of NYC jobs
(16.7% in LA) sit on pairs over 50 km.

The cap comes from what car commuters report. PUMS `JWMNP` for people whose
mode is car (`JWTRNS = 1`): New Jersey median 25 min, p90 60, **p99 134**; New
York median 20, p90 55, p99 138. So **135 minutes one way** is the 99th
percentile of real car commuting, and a pair implying more is dropped from the
draw entirely. Distance is converted to time with two stated assumptions — a
1.25 detour factor and 55 km/h effective door-to-door speed — because the
builder has no road graph; 135 min is then ~99 km straight-line.

Within the surviving range, each pair is **matched to a person whose own
reported commute time, at their own mode's speed, could cover it**. Every PUMS
worker gets a plausible distance range from their reported time (`JWMNP`) and a
door-to-door speed range for their mode (`JWTRNS`): walk 3.5–6 km/h, bike
10–20, car 8–60, transit 8–50, over the 1.25 detour factor. The pair's
straight-line distance must fall inside it, in shared distance bands. People who
work from home, report "other" or give no time carry no range and match on the
other keys only. Industry and earnings are relaxed before distance in the draw
ladder, because distance is what keeps the geography honest while industry only
sharpens it. `scripts/verify_acs_population.py` checks that at least 90% of
workers hold a commute their own mode and time could cover.

**Why ranges, and what they replaced (2026-09-22).** The first version converted
every pair to a time at car speed (55 km/h) and matched that against reported
time. A 3 km Manhattan commute came out at "4 minutes", matchable only to
people reporting under 15 — walkers and very short drives. Among core residents
working in the core, 74% of NYC agents walked against 24.5% in ACS B08301 for
the same tracts, and transit riders all but vanished (Chicago 6.8% vs 23.8%, DC
6.9% vs 32.4%, SF 4.9% vs 26.5%), because a 25-minute subway ride never
matched. A single speed per mode fails the other way: at 55 km/h no driver
reports a time short enough for a 1 km pair, so only walkers fitted. Ranges
admit a walker at 37–64 minutes, a driver at 4–28 and a subway rider at 5–28
for the same 3 km pair, which is what the city looks like.

Result, core residents working in the core (replicates 0–3), against ACS
B08301 for the same core tracts (which also counts residents who work
outside the core, so walking there is expected to run a little lower):

| Metro | Walk pop / ACS | Bike | Transit | Car |
|---|---|---|---|---|
| NYC | 26.6 / 24.5 | 4.2 / 3.3 | 56.5 / 60.0 | 8.3 / 9.4 |
| DC | 14.6 / 14.6 | 4.5 / 4.8 | 27.4 / 32.4 | 52.1 / 47.1 |
| SF Bay | 12.4 / 10.7 | 3.9 / 3.9 | 23.3 / 26.5 | 58.3 / 57.2 |
| Chicago | 5.8 / 7.1 | 1.5 / 1.7 | 24.9 / 23.8 | 66.9 / 66.4 |
| Seattle | 9.9 / 12.1 | 3.3 / 3.7 | 18.5 / 20.4 | 67.8 / 63.2 |
| Miami | 11.4 / 6.1 | 0.7 / 0.9 | 4.7 / 8.4 | 81.3 / 83.1 |
| LA | 2.6 / 3.8 | 0.4 / 0.8 | 6.5 / 8.2 | 89.7 / 86.4 |
| Houston | 0.8 / 2.0 | 0.3 / 0.4 | 4.0 / 3.4 | 94.7 / 93.8 |

**What this does not fix.** New Brunswick to Manhattan is ~55 km, about 75
minutes driving — inside what people report — yet **62.6% of New Jersey
residents working in New York take transit and only 35.4% drive**. The agents
now carry those transit riders, but the model has no transit mode to put them
on. Every agent carries its own PUMS
mode. Walking and cycling now exist (`multimodal.md`), but only for core
residents; transit does not yet, so a New Jersey transit commuter still drives
(or, in NYC, takes a ride if carless). That is a stated limitation rather than
something the cap resolves.

## 5b. Homes outside the graph's counties are skipped at run time

The drive graph, and the commute pairs attached to it, cover only the counties
that hold 95% of a metro's commuters, minus the metro's excluded counties
(`params.METRO_PARAMS[...]["exclude_counties"]`). The rest is real LODES data
with unusable geometry. This population build draws homes from every county,
though, so some agents lived where there is no graph: 3.3% in DC, 2.7% NYC,
1.8% Seattle, 1.6% Chicago, 1.0% Houston, 0.6% LA, 0.2% SF, none in Miami.
Their homes snapped to the graph's edge, and every trip started with a
straight-line access leg of 7-18 km at the median (up to 68 km), charged at
25 km/h.

`Simulation` now skips those rows, in row order, and the next rows take their
places (`dropped_outside_graph`); the service fetches 20% more rows than
agents to allow for it. Every condition of a study applies the same rule to
the same rows. The rows stay in the database: the rule belongs to the graph,
not to the census geography.

## 6. What each agent now carries

Measured, and jointly consistent because they come from one real person: age,
sex, employment, industry, **own earnings**, household income, vehicles,
disability, ambulatory difficulty, household size, commute mode, plus the home
tract's transit-commute share.

**Own earnings** is what fixes the model's standing income problem: the survey
reports household income, LODES segments job earnings, and the two are not the
same quantity. A PUMS person carries both.

Still constant for everyone, because nothing supplies them: walking tolerance,
preferred time window, risk salience, primary leisure interest, and the seven
unmeasured survey item slots (see `survey-coverage.md`).

Vehicles and ambulatory difficulty are live in mode choice (`multimodal.md`):
a household without a vehicle has no car outside NYC, and ambulatory
difficulty rules out cycling and caps walking at 1 km. The PUMS commute mode
starts each agent's habit and is the target the walk and bike constants are
calibrated to. Transit share only feeds perceived behavioural control (TPB),
because there is no transit mode yet.

## 6a. Survey matching reweights the survey, and that has a cost

Each agent takes its psychometrics from the survey respondent matched on sex,
employment, income band and age band (keys dropped in that order when a cell is
empty; `--match-keys` changes the set). That buys a real link — a low-income
agent gets a low-income respondent's attitudes — but 473 respondents spread
across a metro have to be reused, and unevenly, because the metro's demographic
mix concentrates agents into some cells.

Measured, four keys unless noted:

| | Chicago, 20,000 agents | DC, 2,000 | DC, 2,000, no sex key |
|---|---|---|---|
| respondents used (of 473) | **473** | 412 | 413 |
| most-used respondent | 419 agents | 72 | 58 |
| **effective sample size** | **239** | 133 | 142 |
| trust x autonomy correlation | −0.459 | −0.401 | −0.372 |
| (survey's own) | −0.515 | −0.515 | −0.515 |

The 20,000-agent column is the operative one — that is the built population
size. Every respondent appears, and the correlation lands closer to the
survey's than the small-run figure suggested.

Two things follow.

**The correlation shift is expected, not a defect.** Matching reweights the
survey toward the metro's demographics, so a statistic of the Prolific sample
is not a target the metro population must reproduce. Psychometric *spreads* are
preserved (sd ratios 0.92–0.98), and the relationship keeps its sign and most
of its strength. What the verifier checks is exactly that, rather than equality
with a number the population has no reason to match.

**Clustering is the number that matters for inference.** The effective sample
size for anything psychometric is ~240 in a 20,000-agent population, not 473. Results
that lean on attitudes should be clustered on `respondent_id`, which every
agent carries. Dropping sex as a key barely moves it, because the cause is the
metro's demographic concentration rather than thin cells.

## 6b. The routing precompute had to learn about core homes

The engine precomputes all-pairs travel between every point a trip can start or
end at (`welfare_rs/routing_matrix.py`). That universe covered venue nodes and
every node a LODES commute pair snaps to — which already includes every worker
home. Non-worker homes come from core blocks instead, and a block where people
live but nobody works was not in it; those agents would have fallen through to
the lazy per-source Dijkstra, the quadratic path that once made a 6,000-agent
run take hours.

`endpoint_universe` now takes core home blocks as a fourth contributor, via
`RoadNetwork.attach_home_blocks`, attached in `metro.py` wherever commutes are.

**The cost turned out to be nothing.** DC's universe went from 15,336 nodes to
**15,340** — four new nodes — because core residents overwhelmingly already
appear in LODES as home blocks. The matrices rebuild in about a minute per
metro. An earlier worst-case estimate of hundreds of gigabytes assumed the
universe might grow toward every graph node; it does not come close.

## 7. Approximations, stated

- **LODES age × earnings × industry.** LODES publishes the three splits
  separately per pair, never their cross-tabulation, so a job's three bands are
  drawn independently within its pair. Exact for the ~80% of jobs sitting on a
  pair that carries a single job (measured: 69–86% by metro); an assumption for
  the rest.
- **Earnings mapping.** LODES segments *monthly job* earnings; PUMS `PERNP` is
  *annual* earnings from all jobs, divided by 12 here. Reduced, not removed.
- **PUMA resolution.** Two agents on the same block can differ only through
  their LODES bands and their tract's fitted weights; below tract level the
  demographics are as sharp as public data allows.
- **The income tail is real and uncapped, by decision.** PUMS household income
  is used as reported, so a metro population spans $0 to about $2M (Chicago:
  median $93,784, p99 $752k, 2.5% above $500k). The old band draw capped at
  $300k. Value of time is `income / 2000 x 0.5`, linear and unbounded, so the
  top of that tail values time at hundreds of dollars an hour and those agents
  refuse almost every leisure trip. That is a deliberate choice (user, 2026-09-21)
  to keep real incomes rather than cap the VOT input or make it concave — the
  behaviour at the tail is driven by the linearity of the VOT formula, and
  should be read that way rather than as a behavioural finding.
- **LODES noise infusion.** Block-level LODES counts are privacy-protected, so
  the block geography is sharper than the data underneath it really supports.
  Chasing finer demographic precision would be false precision.

## 8. Reproducing it

The repo carries the scripts, never the data. From a clone:

```bash
python db/fetch_census_data.py --what all       # ~2.4 GB, no API key needed
python db/load_pums.py --what all
python db/load_acs.py
python db/build_tract_weights.py --what home-blocks
python db/build_tract_weights.py --what validate     # optional, ~10 min
python db/build_tract_weights.py --what fit
python db/build_population.py --all
python scripts/verify_acs_population.py --metro dc
```

Every download is checksummed into `data/census/manifest_acs_pums.json`, and
the loaders refuse a file whose bytes do not match it — so a rebuild can prove
it used the same vintage. Vintage is pinned: ACS 2019–2023 5-year with LODES8
2023, both on 2020 census geography.

The POI catalog is a separate, licensed dataset and is not redistributable; see
the note in `.gitignore`. The survey is IRB-protected (IRB-FY2026-11354) and
never leaves institutional storage.
