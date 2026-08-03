import type { RunMeta } from "../types";

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
function num(v: number | null | undefined, digits = 3): string {
  return v == null ? "—" : v.toFixed(digits);
}

export default function Results({ run, condition }: { run: RunMeta; condition: string }) {
  // Headline tiles track whichever condition the map is showing, so the numbers
  // and the animation always describe the same run.
  const shown = run.headlines[condition] ? condition : run.default_condition;
  const h = run.headlines[shown];
  const nSeeds = run.seeds.length;
  const agg = run.aggregate[shown] ?? {};

  return (
    <div className="results">
      <h3>
        Welfare metrics (final day, paper-aligned) · {run.city_label}
        <span className="muted">
          {" "}
          · {nSeeds} seed{nSeeds > 1 ? "s" : ""} · {run.conditions.length} conditions
        </span>
      </h3>

      <p className="muted small">
        Headline figures below are for <b>{shown}</b> — the condition the map is
        showing. The tables cover every condition in the study.
      </p>

      <div className="metric-grid">
        <Metric
          label="Mean net utility (Ū)"
          value={`${num(h.mean_utility)} ± ${num(h.sigma_u)}`}
          help="Mean net trip utility over the eval-day leisure trips (activity benefit V − travel cost C), averaged across seeds; ± is the between-seed std σ_U. Paper Table 1."
        />
        <Metric
          label="Negative rate"
          value={pct(h.neg_rate)}
          help="Fraction of leisure trips whose realized net utility U < 0 — the trip left the agent worse off than not going. Paper Table 1. (Distinct from rejection rate.)"
        />
        <Metric
          label="Harmed %"
          value={pct(h.harmed_pct)}
          help="Fraction of agents whose eval-day leisure trip yielded lower utility than their matched No-RS (organic) trip. Paper Table 2."
        />
        <Metric
          label="Improved %"
          value={pct(h.improved_pct)}
          help="Fraction whose utility increased vs the matched No-RS trip. Paper Table 2."
        />
        <Metric
          label="Mean ORC"
          value={num(h.mean_orc)}
          help="Over-recommendation cost: mean utility loss (U_organic − U_rec) over harmed agents only (Eq. 14)."
        />
      </div>

      <div className="card">
        <h4>Table 1 · Aggregate welfare by condition</h4>
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Condition</th>
              <th>Ū</th>
              <th>σ_U</th>
              <th>Neg. rate</th>
              <th>Abst.</th>
              <th>Gini</th>
            </tr>
          </thead>
          <tbody>
            {run.table1.map((r) => (
              <tr key={r.condition} className={r.condition === shown ? "row-hi" : ""}>
                <td>{r.condition}</td>
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

      {run.table2.length > 0 && (
        <div className="card">
          <h4>Table 2 · Over-recommendation cost (vs matched No RS)</h4>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Condition</th>
                <th>Harmed %</th>
                <th>Improved %</th>
                <th>Mean ORC</th>
                <th>Matched agents</th>
              </tr>
            </thead>
            <tbody>
              {run.table2.map((r) => (
                <tr key={r.condition} className={r.condition === shown ? "row-hi" : ""}>
                  <td>{r.condition}</td>
                  <td>{pct(r.harmed_pct)}</td>
                  <td>{pct(r.improved_pct)}</td>
                  <td>{num(r.mean_orc)}</td>
                  <td>{r.n_matched}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="muted small">
        {shown}, across {nSeeds} seed{nSeeds > 1 ? "s" : ""}: rec. acceptance{" "}
        {pct(agg.acceptance_rate)} · avg travel time {num(agg.avg_travel_time_min, 1)} min ·{" "}
        {run.table1[0]?.n_leisure_trips ?? 0} leisure trips (No RS).
      </p>
    </div>
  );
}

function Metric({ label, value, help }: { label: string; value: string; help?: string }) {
  return (
    <div className="metric" title={help}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
