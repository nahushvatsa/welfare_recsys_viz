# Survey coverage: what is grounded, what is not, and what ACS could fix

What the April 2026 Prolific survey (IRB-FY2026-11354, N=473) supplies to the
ABM, what it does not, and — for each gap — what was done about it and why.
Written 2026-09-03, after the survey integration.

The short version: every agent attribute that varies across the population is
now survey-measured. Everything the survey does not measure is either held
constant for the whole population or dropped. Nothing is drawn from a
distribution that pretends to be empirical.

---

## 0. Where the data lives

Two paths reach the same 473 agent profiles, and
`scripts/verify_survey_personas_view.py` asserts they are interchangeable —
all 25,069 persona cells equal, and every attribute of every agent in a
600-agent population equal, resampled agents included.

```
data_handoff/algo_leisure_survey_2026.csv        (IRB-FY2026-11354, N=473)
  |
  |-- data/build_survey_personas.py --> data/survey_personas_2026.csv
  |                                     LocalDataSource.personas()   [no DB]
  |
  '-- db/load_survey.py --> survey.respondents (136 typed columns)
                              |
                              '-- survey.personas (view, 53 columns)
                                  PostgresDataSource.personas()   [lab server]
```

The `survey` schema also holds the rest of the package: `data_dictionary`
(item wording, valid values, declared missingness), `construct_items` (the
composite scoring map), `marginals`, `correlations` + `calibration_meta` (the
33-attribute Pearson matrix and its pairwise N), and `package_files` (every
shipped file's SHA-256). `db/load_survey.py` re-verifies the checksums on every
run and aborts if the DB would hold anything other than what was shipped.

**Access.** `survey.respondents` is respondent-level human-subjects data and
the API-facing `welfare_app` role cannot read it, nor any other table in the
schema. It holds SELECT on `survey.personas` alone; a Postgres view runs with
its owner's privileges, so the 53 ABM columns are readable while the
136-column file underneath is not. The package and everything derived from it
at respondent level are gitignored (`.gitignore`, "Human-subjects data").

**Typing note.** `survey.respondents` types each column by what it *means*, not
by how pandas happened to write it: the shipped file carries 27 integer
response items as floats (`"3.0"`) purely because a missing cell upcast the
column, and 8 of the 11 `plat_freq_*` items differ in storage type from the
other 3 for that reason alone. Those are `smallint` in the DB. The one place
this is visible downstream — `LeisureSpend_num`, which is copied verbatim as a
string onto `Agent.characteristics` — is cast back to `double precision` inside
the view so the two paths stay byte-identical. See the comment there.

---

## 1. Grounded in the survey

**Psychometrics (9).** Big Five, Maximization, `Recommendation_Trust`,
`Algorithmic_Awareness`, and `autonomy_preference` from the single item
`s6_autonomy_control_a` — *not* the `Autonomy_Control` composite, which is
positively loaded on liking personalisation and would invert beta_O. See the
comment in `data/build_survey_personas.py`.

**Survey items (18 of 25 slots).** Leisure frequency, overnight trips,
spontaneity, city familiarity, follow-through by source (friend / platform / AI),
cross-platform search, price filtering, group coordination, popularity herding,
unexpected discovery, social posting, multi-platform use, AI itinerary comfort,
explanation need, algorithmic literacy, budget tightness.

**Demographics.** Age and household income (band label -> dollar range),
sex, employment. Education, race/ethnicity, home language, census region and
division, residence length and leisure spend are carried on
`Agent.characteristics` for reporting but do not drive behaviour.

**Behaviourally load-bearing demographics.** `Employment` decides whether an
agent holds a job, which sets both the day structure (work-anchored vs
home-anchored) and the value-of-time share. 161 of 473 respondents (34%) are
retired, unemployed, out of the labour force or studying.

---

## 2. Not measured by the survey — attributes

Checked against all 136 columns of the shipped dataset. None of these has any
counterpart in the instrument. The survey's `s4_group_coord_*` / `s5_group_coord_*`
items are about coordinating *with a group when using a platform*, which is a
different construct from travel-party composition and is already wired to
`group_coordination_preference`.

These were previously drawn per agent from the synthetic persona file, whose
level shares came from a **D-optimal experimental design** — balanced by
construction (32% ADA mobility needs, 33% carless, 50% non-English), not
representative of any population. That made them look like a source of empirical
heterogeneity when they were an artefact of the design.

| Attribute | Now | Where |
|---|---|---|
| Car access | fixed `own car` | `PERSONA_MAPPING["fixed_attributes"]` |
| Transit access | fixed `medium` (0.55) | same |
| Mobility needs | fixed `none` | same |
| Walking tolerance | fixed 15 min | same |
| Time window | fixed `weekday evening` | same |
| Risk salience | fixed `low` | same |
| Primary leisure interest | **dropped** (`""`, matches no `interest_map` key) | same |
| Environmental attitude | **dropped**; green weight keeps its random base plus the Openness shift | `simulation.py` |
| Travel-party composition | **dropped**; travel affinity keeps its base plus survey shifts | removed 2026-09-03 |

Constants rather than draws is the honest choice: they contribute no
heterogeneity and cannot be mistaken for an empirical source of it. Primary
interest and environmental attitude were dropped outright rather than pinned
because both applied a *utility bonus*, and a constant bonus for every agent is
only a shift in the origin.

---

## 3. Not measured by the survey — item slots

Seven `Agent.survey_items` slots have no survey counterpart. **A constant 0.5 is
not neutral**: it pins part of a coefficient at its midpoint, which compresses
that coefficient's spread. Removing it and renormalising the survivors does not
restore neutrality — it *amplifies* whatever indicator is left.

Measured on 400 agents (Manhattan, seed 42):

| Coefficient | Unsourced weight | Now | If dropped + renormalised |
|---|---|---|---|
| `feedback_loop_strength` | 0.70 (`review_posting` .25, `switch_when_dissatisfied` .45) | 0.585, range [0.35, 0.65] | 0.783, range [0.00, 1.00] |
| `budget_sensitivity` | 0.40 (three `incentive_*`) | 0.539, [0.20, 0.80] | 0.564, [0.00, 1.00] |
| `variety_seeking` | 0.20 (`advice_goal_directed`) | 0.657, [0.10, 0.90] | 0.696, [0.00, 1.00] |
| `risk_aversion` | 0.10 (`negative_recommendation_experience`) | 0.460, [0.08, 0.90] | 0.455, [0.03, 0.94] |

### Removed

`negative_recommendation_experience` — **gone as of 2026-09-03.** Both its uses
were no-ops at the constant: the `risk_aversion` term carried only 10% weight,
and its participation-signal term was literally `(0.5 - 0.5) = 0`. Removing it
moved `risk_aversion` by 0.005. `risk_aversion` weights renormalised to
0.72 / 0.17 / 0.11.

### Kept, deliberately — do not remove without re-running the concentration panels

- **`review_posting` and `switch_when_dissatisfied`.** Two-thirds of
  `feedback_loop_strength`. Dropping them makes that coefficient *entirely*
  `multi_platform_parallel`: mean shifts +0.20 and spread widens **3.3x**. It
  drives `feedback_sensitivity`, which controls how strongly agent feedback moves
  venue ratings — the public feedback loop the paper exists to study. Tripling its
  spread would materially change the concentration results.
- **The three `incentive_*` items.** 40% of `budget_sensitivity`. Mean barely
  moves but the spread doubles.
- **`advice_goal_directed`.** 20% of `variety_seeking`. Mean +0.04, spread 1.25x.

Keeping them at 0.5 is the conservative choice and avoids asserting that the
surviving indicator alone measures the construct. State in the paper that these
coefficients are partly pinned because the survey lacks the items.

---

## 4. What ACS could supply

Planned. ACS can plausibly close four of the nine attribute gaps:

| Attribute | ACS table | Note |
|---|---|---|
| Car access | B08201 / B25044 (vehicles available) | Household-level; the ABM wants person-level access. Tenure and household size help. |
| Transit access | B08301 (means of transportation to work) | A *revealed commute mode*, not an access measure. A tract-level transit-commute share is a reasonable proxy for `transit_access_level`. |
| Mobility needs | B18101 (disability status by age) | Gives a disability rate, not the ADA/stroller/none split the ABM uses. Would need recoding. |
| Travel-party composition | B11016 (household type by size) | Closest thing to the dropped `Group`, if it is ever reinstated. |

ACS **cannot** supply: walking tolerance, primary leisure interest, preferred
time window, environmental attitude, risk salience, or any of the seven item
slots in section 3. Those need either a new survey wave or removal from the
model.

**Sequencing note.** ACS is geographic (tract/block-group), while the survey
population is a national sample with ZIP removed. Joining them means assigning
each agent a tract — which the LODES commute pair already does. So the natural
path is: draw the LODES pair, take the home block group, then draw car access /
transit share / disability from that block group's ACS marginals. That makes
these attributes vary *by geography* rather than by respondent, which is a
different and defensible kind of heterogeneity — but it is not joint with the
survey attributes, and that limitation should be stated.

---

## 5. Survey data collected but not used by the ABM

Not gaps in the ABM — data the model has no home for.

- **`plat_freq_*` (11 items) and `plat_search` / `plat_social` / `plat_ai_asst`.**
  Platform-specific use intensity. The ABM models a single monopoly platform
  (paper section 5 limitation), so there is nowhere to put per-platform frequency.
  Three of these are among the 33 calibration attributes.
- **`s2_disc_method` (Q15).** How respondents last discovered a new venue:
  188/473 (39.7%) from a person, 90 (19.0%) social media, 79 (16.7%) map/review
  app. This is a direct empirical measure of the organic-vs-platform split the
  whole model turns on, and it is currently unused. Best available calibration
  target for the acceptance rate.
- **The s4/s5 scenario batteries** beyond the six items wired.
- **`Platform_Comfort`** and **`Autonomy_Control`** — carried for analysis, not
  wired to behaviour.
- Ordinal duplicates (`leisure_freq_ord`, `overnight_ord`) and `Duration_s`.

---

## 6. Code left in place that looks dead but is not

For anyone doing a future cleanup pass:

- **`DERIVED_DEMAND_ONLY` branches** (`agent.py`, `simulation.py`). The toggle is
  now `False`, so these never fire — but they are the ablation machinery the
  toggle exists for. Deleting them removes the capability, not dead code.
- **The transit stand-in** (`MODE_PARAMS["transit"]`, crowding, the transit
  branch of the mode code). `CAR_ONLY_MODE` is gone and walk/bike/car choice is
  live (see `multimodal.md`), but transit is disabled by default
  (`params.DISABLED_MODES`): it is a flat speed over road distance, not a
  transit network. Kept for when one exists.
- **`over_recommendation_cost` / `welfare_gain`** (`experiment_harness.py`). A
  *deliberately different* quantity from the paper's ORC — clipped loss over all
  agents rather than per-harmed-agent. Its output key was renamed
  `orc_all_agents` on 2026-09-03 so the two cannot be confused. The paper's ORC
  is `table2_from_utilities`, which is what the service reports.
- **`walk_tolerance_min` outdoor discount** (`agent.py`). Never fires because the
  fixed tolerance is 15 and the threshold is 30. Parameterised, not dead.

Genuine cleanup candidate, left alone pending a decision:
**`welfare_rs/oracle_rs.py`** (215 lines, `OracleRecommender`). Not imported by
the package, the service, the simulation or the notebook. Not one of the paper's
two controls.
