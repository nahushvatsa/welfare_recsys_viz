import { useMemo, useState } from "react";
import {
  condLabel,
  conditionColors,
  DayLines,
  DivergingHeatmap,
  FigureMode,
  GroupedBars,
  LorenzChart,
  Panel,
} from "./charts";
import WelfareMap from "./WelfareMap";
import type { ConditionMetrics, RunMeta } from "../types";

function pct(v: number | null | undefined, digits = 1): string {
  return v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
}
function num(v: number | null | undefined, digits = 3): string {
  return v == null ? "—" : v.toFixed(digits);
}

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "loop", label: "Feedback loop" },
  { id: "geography", label: "Geography" },
  { id: "who", label: "Who gains & loses" },
] as const;
type TabId = (typeof TABS)[number]["id"];

/** Category labels, prettier than the raw engine keys. */
const CATEGORY_LABEL: Record<string, string> = {
  restaurant: "Restaurant",
  food_takeout: "Takeout",
  cafe: "Café",
  park: "Park",
  museum: "Museum",
  fitness_studio: "Fitness",
  gym: "Gym",
  live_music: "Live music",
  concert_venue: "Concert venue",
  music_event: "Music event",
  running_route: "Running",
};

export default function Results({ run, condition }: { run: RunMeta; condition: string }) {
  const [tab, setTab] = useState<TabId>("overview");
  // Screenshot mode: panels drop to a fixed paper-column width. See FigureMode.
  const [figure, setFigure] = useState(false);
  const shown = run.headlines[condition] ? condition : run.default_condition;
  const control = run.conditions[0];
  const colors = useMemo(
    () => conditionColors(run.conditions, control),
    [run.conditions, control]
  );
  const metrics = run.metrics ?? {};
  const treated = run.conditions.filter((c) => c !== control);
  const nSeeds = run.seeds.length;

  return (
    <div className={figure ? "results figure-mode" : "results"}>
      <div className="results-head">
        <h3>
          Results · {run.city_label}
          <span className="muted">
            {" "}
            · {run.config.num_agents.toLocaleString()} agents · {run.num_days} days ·{" "}
            {nSeeds} seed{nSeeds > 1 ? "s" : ""} · {run.conditions.length} conditions
          </span>
        </h3>
        <div className="head-tools">
          <nav className="tabs" role="tablist">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                className={tab === t.id ? "tab active" : "tab"}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <button
            className={figure ? "figure-toggle on" : "figure-toggle"}
            aria-pressed={figure}
            onClick={() => setFigure((f) => !f)}
            title="Shrink each panel to one paper column (~3.3 in) so a screenshot drops straight into the manuscript at a readable point size. Panels marked wide span the full text width."
          >
            Figure mode
          </button>
        </div>
      </div>

      <FigureMode.Provider value={figure}>
        {tab === "overview" && <Overview run={run} shown={shown} />}
        {tab === "loop" && (
          <FeedbackLoop run={run} metrics={metrics} colors={colors} control={control} />
        )}
        {tab === "geography" && (
          <Geography run={run} metrics={metrics} colors={colors} control={control} treated={treated} />
        )}
        {tab === "who" && (
          <WhoGainsLoses run={run} metrics={metrics} colors={colors} control={control} />
        )}
      </FigureMode.Provider>
    </div>
  );
}

// ── Overview ─────────────────────────────────────────────────────────────────

function Overview({ run, shown }: { run: RunMeta; shown: string }) {
  const h = run.headlines[shown];
  const agg = run.aggregate[shown] ?? {};
  const nSeeds = run.seeds.length;

  return (
    <>
      <p className="muted small">
        Headline figures are for <b>{condLabel(shown)}</b> — the condition the map is
        showing. Tables cover every condition. Both are computed on the final
        (evaluation) day, matching the paper; the other tabs pool all{" "}
        {run.num_days} days for resolution.
      </p>

      <div className="metric-grid">
        <Metric label="Mean net utility (Ū)" value={`${num(h.mean_utility)} ± ${num(h.sigma_u)}`}
          help="Mean net trip utility over the eval-day leisure trips (activity benefit V − travel cost C), averaged across seeds; ± is the between-seed std σ_U. Paper Table 1." />
        <Metric label="Negative rate" value={pct(h.neg_rate)}
          help="Fraction of leisure trips whose realized net utility U < 0 — the trip left the agent worse off than not going." />
        <Metric label="Harmed %" value={pct(h.harmed_pct)}
          help="Fraction of agents whose eval-day leisure trip yielded lower utility than their matched No-RS trip. Paper Table 2." />
        <Metric label="Improved %" value={pct(h.improved_pct)}
          help="Fraction whose utility increased vs the matched No-RS trip." />
        <Metric label="Mean ORC" value={num(h.mean_orc)}
          help="Over-recommendation cost: mean utility loss (U_organic − U_rec) over harmed agents only (Eq. 14)." />
      </div>

      <div className="card">
        <h4>Table 1 · Aggregate welfare by condition</h4>
        <div className="table-scroll">
          <table className="metrics-table">
            <thead>
              <tr><th>Condition</th><th>Ū</th><th>σ_U</th><th>Neg. rate</th><th>Abst.</th><th>Gini</th></tr>
            </thead>
            <tbody>
              {run.table1.map((r) => (
                <tr key={r.condition} className={r.condition === shown ? "row-hi" : ""}>
                  <td>{condLabel(r.condition)}</td>
                  <td>{num(r.mean_utility)}</td>
                  <td>{num(r.sigma_u)}</td>
                  <td>{pct(r.neg_rate)}</td>
                  <td>{pct(r.abstention_rate)}</td>
                  <td>{num(r.gini)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {run.table2.length > 0 && (
        <div className="card">
          <h4>Table 2 · Over-recommendation cost (vs matched control)</h4>
          <div className="table-scroll">
            <table className="metrics-table">
              <thead>
                <tr><th>Condition</th><th>Harmed %</th><th>Improved %</th><th>Mean ORC</th><th>Matched agents</th></tr>
              </thead>
              <tbody>
                {run.table2.map((r) => (
                  <tr key={r.condition} className={r.condition === shown ? "row-hi" : ""}>
                    <td>{condLabel(r.condition)}</td>
                    <td>{pct(r.harmed_pct)}</td>
                    <td>{pct(r.improved_pct)}</td>
                    <td>{num(r.mean_orc)}</td>
                    <td>{r.n_matched}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <p className="muted small">
        {condLabel(shown)}, across {nSeeds} seed{nSeeds > 1 ? "s" : ""}: rec. acceptance{" "}
        {pct(agg.acceptance_rate)} · avg travel time {num(agg.avg_travel_time_min, 1)} min.
      </p>
    </>
  );
}

// ── Feedback loop ────────────────────────────────────────────────────────────

function FeedbackLoop({
  run, metrics, colors, control,
}: {
  run: RunMeta;
  metrics: Record<string, ConditionMetrics>;
  colors: Record<string, string>;
  control: string;
}) {
  const days = alignDays(run.conditions, metrics, "effective_places");
  const top10 = alignDays(run.conditions, metrics, "top10_share");
  const taste = alignDays(run.conditions, metrics, "recommended_rate");
  const lorenz = Object.fromEntries(
    run.conditions.map((c) => [c, metrics[c]?.lorenz ?? []])
  );

  return (
    <div className="panel-grid">
      <Panel title="Effective number of venues">
        <DayLines data={days} series={run.conditions} colors={colors}
                  yLabel="effective venues" format={(v) => v.toFixed(1)} />
      </Panel>

      <Panel
        title="Share of visits at the ten busiest venues"
        hint="An absolute head-share, so it stays meaningful however large the catalog is. Rising means the same handful of places is taking more of the traffic."
      >
        <DayLines data={top10} series={run.conditions} colors={colors}
                  yLabel="share of visits" format={(v) => `${(v * 100).toFixed(1)}%`} />
      </Panel>

      <Panel title="Concentration of footfall (Lorenz)">
        <LorenzChart curves={lorenz} colors={colors} />
      </Panel>

      <Panel
        title="Taste discovery"
        hint={
          <>
            Share of accepted recommendations that matched the agent's latent
            taste. The platform never observes taste — it can only infer it from
            feedback on recommendations agents <i>accepted</i>, so a rising line
            is personalization learning. {feedbackNote(run, metrics, control)}
          </>
        }
      >
        <DayLines data={taste} series={run.conditions.filter((c) => c !== control)}
                  colors={colors} yLabel="taste match"
                  format={(v) => `${(v * 100).toFixed(1)}%`} />
        {/* The daily series is thin — a few dozen accepted recommendations a
            day — so it reads as noise even when the effect is real. The pooled
            rate over the whole run is the number to compare, and the organic
            column is the natural floor: it is what taste match looks like when
            nothing is learning. */}
        <div className="table-scroll">
          <table className="metrics-table">
            <thead>
              <tr><th>Condition</th><th>Recommended</th><th>Organic</th><th>Accepted recs</th></tr>
            </thead>
            <tbody>
              {run.conditions.map((c) => {
                const t = metrics[c]?.taste;
                if (!t) return null;
                return (
                  <tr key={c} className={c === control ? "row-control" : ""}>
                    <td>
                      <span className="swatch" style={{ background: colors[c] }} />
                      {condLabel(c)}{c === control ? " (control)" : ""}
                    </td>
                    <td>{c === control ? "—" : pct(t.recommended_rate)}</td>
                    <td>{pct(t.organic_rate)}</td>
                    <td>{Math.round(t.recommended_n)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

// ── Geography ────────────────────────────────────────────────────────────────

function Geography({
  run, metrics, colors, control, treated,
}: {
  run: RunMeta;
  metrics: Record<string, ConditionMetrics>;
  colors: Record<string, string>;
  control: string;
  treated: string[];
}) {
  const bands = run.income_bands ?? [];
  const baseline = metrics[control]?.spatial;

  const distanceRows = bands.map((band) => {
    const row: Record<string, string | number> = { band };
    for (const c of run.conditions) {
      row[c] = metrics[c]?.spatial?.by_income?.[band]?.mean_distance_km ?? 0;
    }
    return row;
  });

  const co2Rows = bands.map((band) => {
    const row: Record<string, string | number> = { band };
    const base = baseline?.by_income?.[band]?.mean_emissions_g ?? 0;
    for (const c of treated) {
      const v = metrics[c]?.spatial?.by_income?.[band]?.mean_emissions_g ?? 0;
      row[c] = (v - base) / 1000; // grams → kg per leisure trip
    }
    return row;
  });

  const detourRows = run.conditions.map((c) => ({
    condition: c,
    ratio: metrics[c]?.spatial?.overall?.median_detour ?? 0,
    excess: metrics[c]?.spatial?.overall?.median_excess_km ?? 0,
  }));

  return (
    <div className="panel-grid">
      <Panel
        title="How far agents travel for leisure"
        hint="Mean network distance per leisure trip, by household income. Pooled over the whole run."
      >
        <GroupedBars data={distanceRows} categoryKey="band" series={run.conditions}
                     colors={colors} yLabel="km per trip" format={(v) => `${v.toFixed(2)} km`} />
      </Panel>

      <Panel
        title="Extra CO₂ from being recommended to"
        hint={<>Change in tailpipe emissions per leisure trip versus the <b>{condLabel(control)}</b> control. Above the line, the recommender is adding emissions; below it, saving them.</>}
      >
        <GroupedBars data={co2Rows} categoryKey="band" series={treated} colors={colors}
                     yLabel="kg CO₂e per trip" zeroLine
                     format={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)} kg`} />
      </Panel>

      <Panel
        title="Farther than necessary"
        hint="Distance to the venue actually visited, against the nearest venue of the same kind. Both straight-line, so the comparison isolates the choice. In a dense core the nearest option is often only 100–200 m away, which is why the ratio runs high even without a recommender — the kilometre column is the readable one."
      >
        <div className="table-scroll">
          <table className="metrics-table">
            <thead>
              <tr><th>Condition</th><th>Median ratio</th><th>Median extra distance</th></tr>
            </thead>
            <tbody>
              {detourRows.map((r) => (
                <tr key={r.condition} className={r.condition === control ? "row-control" : ""}>
                  <td>
                    <span className="swatch" style={{ background: colors[r.condition] }} />
                    {condLabel(r.condition)}{r.condition === control ? " (control)" : ""}
                  </td>
                  <td>{r.ratio.toFixed(2)}×</td>
                  <td>{r.excess >= 0 ? "+" : ""}{r.excess.toFixed(2)} km</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Where the winners and losers live" wide>
        <WelfareMap run={run} conditions={treated} />
      </Panel>
    </div>
  );
}

// ── Who gains and loses ──────────────────────────────────────────────────────

function WhoGainsLoses({
  run, metrics, colors, control,
}: {
  run: RunMeta;
  metrics: Record<string, ConditionMetrics>;
  colors: Record<string, string>;
  control: string;
}) {
  const bands = run.income_bands ?? [];
  const trust = run.trust_groups ?? [];
  const treated = run.conditions.filter((c) => c !== control);

  const incomeRows = bands.map((band) => {
    const row: Record<string, string | number> = { band };
    for (const c of run.conditions) row[c] = metrics[c]?.segments?.income?.[band]?.mean_utility ?? 0;
    return row;
  });
  const trustRows = trust.map((group) => {
    const row: Record<string, string | number> = { group };
    for (const c of run.conditions) row[c] = metrics[c]?.segments?.trust?.[group]?.mean_utility ?? 0;
    return row;
  });

  // Category mix shift: percentage points of each band's leisure visits, minus
  // the control's. Shares, not counts — the bands hold very different numbers of
  // agents, so raw counts would just redraw the income distribution.
  const categories = useMemo(() => {
    const seen = new Set<string>();
    for (const c of run.conditions) {
      for (const row of Object.values(metrics[c]?.category_by_income ?? {})) {
        Object.keys(row).forEach((k) => seen.add(k));
      }
    }
    return [...seen].sort();
  }, [run.conditions, metrics]);

  const [heatCondition, setHeatCondition] = useState(treated[0] ?? control);

  function shareDelta(band: string, category: string): number | null {
    const share = (cond: string) => {
      const row = metrics[cond]?.category_by_income?.[band];
      if (!row) return null;
      const total = Object.values(row).reduce((a, b) => a + b, 0);
      return total > 0 ? (row[category] ?? 0) / total : null;
    };
    const a = share(heatCondition);
    const b = share(control);
    return a == null || b == null ? null : (a - b) * 100;
  }

  return (
    <div className="panel-grid">
      <Panel title="Utility by household income">
        <GroupedBars data={incomeRows} categoryKey="band" series={run.conditions}
                     colors={colors} yLabel="mean net utility" zeroLine />
      </Panel>

      <Panel title="Utility by trust in platforms">
        <GroupedBars data={trustRows} categoryKey="group" series={run.conditions}
                     colors={colors} yLabel="mean net utility" zeroLine />
      </Panel>

      <Panel
        title="What each income group ends up visiting"
        hint={<>Change in the share of a band's leisure visits going to each venue type, in percentage points versus <b>{condLabel(control)}</b>. Blue means the recommender sends that group there more often than they would have gone on their own.</>}
        wide
      >
        {treated.length > 1 && (
          <label className="inline-picker">
            Condition
            <select value={heatCondition} onChange={(e) => setHeatCondition(e.target.value)}>
              {treated.map((c) => <option key={c} value={c}>{condLabel(c)}</option>)}
            </select>
          </label>
        )}
        <DivergingHeatmap
          rows={bands}
          cols={categories.map((c) => CATEGORY_LABEL[c] ?? c)}
          rowLabel="Income"
          value={(band, colLabel) => {
            const key = categories.find((c) => (CATEGORY_LABEL[c] ?? c) === colLabel);
            return key ? shareDelta(band, key) : null;
          }}
          format={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`}
        />
        <p className="muted small">Percentage points. Rows sum to zero by construction.</p>
      </Panel>
    </div>
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * Say whether this study gave personalization enough to learn from.
 *
 * Feedback arrives only on *accepted* recommendations, at roughly 0.15 leisure
 * outings per agent per day times the acceptance rate — so a short study leaves
 * most agents with zero or one signal and the curve is flat no matter how the
 * recommender is configured. That is a property of the study, not a result, and
 * the panel should say so rather than let it be read as "personalization does
 * not work".
 */
function feedbackNote(
  run: RunMeta,
  metrics: Record<string, ConditionMetrics>,
  control: string
): string {
  const treated = run.conditions.filter((c) => c !== control);
  if (!treated.length) return "";
  const perAgent =
    treated.reduce((sum, c) => sum + (metrics[c]?.taste?.recommended_n ?? 0), 0) /
    treated.length /
    Math.max(1, run.config.num_agents);
  const rounded = perAgent.toFixed(1);
  return perAgent < 2
    ? `In this study each agent produced only about ${rounded} accepted recommendations, which is too few to learn from — run more days to see the curve move.`
    : `Each agent produced about ${rounded} accepted recommendations here, enough for the signal to accumulate.`;
}

/** Build a per-day chart frame: one row per day, one column per condition. */
function alignDays(
  conditions: string[],
  metrics: Record<string, ConditionMetrics>,
  key: string
): Record<string, number>[] {
  const lengths = conditions.map((c) => metrics[c]?.per_day?.length ?? 0).filter((n) => n > 0);
  const days = lengths.length ? Math.min(...lengths) : 0;
  const out: Record<string, number>[] = [];
  for (let i = 0; i < days; i += 1) {
    const row: Record<string, number> = { day: i + 1 };
    for (const c of conditions) {
      const rec = metrics[c]?.per_day?.[i];
      if (rec && typeof rec[key] === "number") row[c] = rec[key];
    }
    out.push(row);
  }
  return out;
}

function Metric({ label, value, help }: { label: string; value: string; help?: string }) {
  return (
    <div className="metric" title={help}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
