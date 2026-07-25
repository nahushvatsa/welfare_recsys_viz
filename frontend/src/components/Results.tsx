import type { RunMeta } from "../types";

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
function num(v: number | null | undefined, digits = 3): string {
  return v == null ? "—" : v.toFixed(digits);
}

export default function Results({ run }: { run: RunMeta }) {
  const h = run.headline;
  const t2 = run.table2;
  const nSeeds = run.seeds.length;

  return (
    <div className="results">
      <h3>
        Welfare metrics (final day, paper-aligned) · {run.city_label} · {run.config.treatment}
        <span className="muted">
          {" "}
          · {nSeeds} seed{nSeeds > 1 ? "s" : ""}
        </span>
      </h3>

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
              <tr key={r.condition} className={r.condition === run.config.treatment ? "row-hi" : ""}>
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

      {t2 && (
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
              <tr className="row-hi">
                <td>{t2.condition}</td>
                <td>{pct(t2.harmed_pct)}</td>
                <td>{pct(t2.improved_pct)}</td>
                <td>{num(t2.mean_orc)}</td>
                <td>{t2.n_matched}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <p className="muted small">
        Across {nSeeds} seed{nSeeds > 1 ? "s" : ""}: rec. acceptance {pct(run.aggregate.acceptance_rate)} ·
        avg travel time {num(run.aggregate.avg_travel_time_min, 1)} min ·{" "}
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
