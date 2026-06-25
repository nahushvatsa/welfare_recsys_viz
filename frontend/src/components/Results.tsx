import type { RunMeta } from "../types";
import { BarChart, LineChart } from "./Charts";

function pct(v: number | undefined): string {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}
function num(v: number | undefined, digits = 0): string {
  return v == null ? "—" : v.toFixed(digits);
}

const TREND_SERIES = [
  { key: "avg_trip_utility", label: "avg trip utility", color: "#3b82f6" },
  { key: "recommendation_acceptance_rate", label: "rec. acceptance", color: "#10b981" },
  { key: "feedback_like_rate", label: "feedback like rate", color: "#a855f7" },
  { key: "avg_travel_time_min", label: "avg travel time (min)", color: "#f59e0b" },
];

export default function Results({ run }: { run: RunMeta }) {
  const s = run.summary;
  const modeShare = (s.mode_share ?? {}) as Record<string, number>;
  const leisureMix = (s.leisure_subtype_counts ?? {}) as Record<string, number>;
  const leisureTrips = Object.values(leisureMix).reduce((a, b) => a + b, 0);

  return (
    <div className="results">
      <h3>
        Results (final day) · {run.area_label} · {run.config.treatment}
      </h3>

      <div className="metric-grid">
        <Metric label="Trips" value={num(s.trips)} />
        <Metric label="Leisure trips" value={String(leisureTrips)} />
        <Metric label="Rec. acceptance" value={pct(s.recommendation_acceptance_rate)} />
        <Metric
          label="Leisure net utility"
          value={num(s.mean_leisure_net_utility, 3)}
          help="Mean net utility of leisure trips (activity benefit − travel cost). The welfare-relevant signal: Standard RS often pushes this below No RS."
        />
        <Metric label="Avg travel time" value={`${num(s.avg_travel_time_min, 1)} min`} />
      </div>

      <div className="chart-grid">
        <div className="card">
          <h4>Mode share</h4>
          <BarChart data={modeShare} color="#3b82f6" format={(v) => `${Math.round(v * 100)}%`} />
        </div>
        <div className="card">
          <h4>Leisure activity mix</h4>
          <BarChart data={leisureMix} color="#a855f7" />
        </div>
      </div>

      {run.day_summaries.length > 1 && (
        <div className="card">
          <h4>Daily trends</h4>
          <LineChart
            rows={run.day_summaries as Array<Record<string, number>>}
            xKey="day"
            series={TREND_SERIES.filter((s2) => run.day_summaries.some((d) => d[s2.key] != null))}
          />
        </div>
      )}
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
