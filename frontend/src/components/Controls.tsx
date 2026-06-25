import type { CitiesResponse, RunConfig, RunMeta } from "../types";

export type RunStatus = "idle" | "running" | "ready" | "error";

interface ControlsProps {
  cities: CitiesResponse;
  config: RunConfig;
  setConfig: (patch: Partial<RunConfig>) => void;
  onRun: () => void;
  status: RunStatus;
  progress: { current: number; total: number } | null;
  runMeta: RunMeta | null;
  error: string | null;
  dirty: boolean;
}

export default function Controls({
  cities,
  config,
  setConfig,
  onRun,
  status,
  progress,
  runMeta,
  error,
  dirty,
}: ControlsProps) {
  const showPup = config.treatment === "PUP" || config.treatment === "PUP+RM";
  const showRm = config.treatment === "RM" || config.treatment === "PUP+RM";
  const running = status === "running";

  return (
    <aside className="sidebar">
      <h1>Welfare-RS · NYC</h1>
      <p className="muted">Welfare-oriented activity-travel simulation</p>

      <h2>Area</h2>
      <div className="area-grid">
        {cities.areas.map((a) => (
          <button
            key={a.key}
            className={`area-btn ${config.city === a.key ? "active" : ""}`}
            onClick={() => setConfig({ city: a.key })}
          >
            {a.label}
          </button>
        ))}
      </div>

      <h2>Settings</h2>
      <label>
        Agents
        <input
          type="number"
          min={1}
          max={5000}
          value={config.num_agents}
          onChange={(e) => setConfig({ num_agents: clampInt(e.target.value, 1, 5000) })}
        />
      </label>
      <label>
        Days
        <input
          type="number"
          min={1}
          max={60}
          value={config.num_days}
          onChange={(e) => setConfig({ num_days: clampInt(e.target.value, 1, 60) })}
        />
      </label>
      <label>
        Random seed
        <input
          type="number"
          value={config.seed}
          onChange={(e) => setConfig({ seed: parseInt(e.target.value || "0", 10) })}
        />
      </label>
      <label>
        Recommender
        <select
          value={config.treatment}
          onChange={(e) => setConfig({ treatment: e.target.value })}
        >
          {cities.treatments.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>

      <label className="toggle">
        <input
          type="checkbox"
          checked={config.multimodal}
          onChange={(e) => setConfig({ multimodal: e.target.checked })}
        />
        Multimodal (car + walk + bike)
      </label>
      <label className="toggle">
        <input
          type="checkbox"
          checked={config.use_real_pois}
          disabled={!cities.pois_available}
          onChange={(e) => setConfig({ use_real_pois: e.target.checked })}
        />
        Real NYC POIs {cities.pois_available ? "" : "(dataset not found)"}
      </label>

      {showPup && (
        <label>
          PUP α (min P[U≥0]): {config.pup_alpha.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={config.pup_alpha}
            onChange={(e) => setConfig({ pup_alpha: parseFloat(e.target.value) })}
          />
        </label>
      )}
      {showRm && (
        <label>
          RM ε (regret ceiling): {config.rm_epsilon.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={config.rm_epsilon}
            onChange={(e) => setConfig({ rm_epsilon: parseFloat(e.target.value) })}
          />
        </label>
      )}

      <button className="run-btn" onClick={onRun} disabled={running}>
        {running ? "Running…" : "▶ Run simulation"}
      </button>

      {running && progress && (
        <div className="progress">
          <div
            className="progress-bar"
            style={{ width: `${progress.total ? (progress.current / progress.total) * 100 : 5}%` }}
          />
          <span className="progress-text">
            {progress.current === 0
              ? "Preparing simulation…"
              : `Running day ${progress.current} of ${progress.total}…`}
          </span>
        </div>
      )}

      {dirty && status === "ready" && (
        <p className="hint">⚙️ Settings changed — press Run to apply.</p>
      )}
      {error && <p className="error">{error}</p>}

      {runMeta && status === "ready" && (
        <div className="captions">
          <p>Network: {runMeta.num_intersections.toLocaleString()} road intersections</p>
          <p>
            POIs: {runMeta.poi_count.toLocaleString()} ({runMeta.poi_source})
          </p>
        </div>
      )}
    </aside>
  );
}

function clampInt(v: string, lo: number, hi: number): number {
  const n = parseInt(v || "0", 10);
  if (Number.isNaN(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}
