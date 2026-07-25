import { useState } from "react";
import type { CitiesResponse, RunConfig, RunMeta } from "../types";

export type RunStatus = "idle" | "running" | "ready" | "error";

interface ControlsProps {
  cities: CitiesResponse;
  config: RunConfig;
  setConfig: (patch: Partial<RunConfig>) => void;
  onRun: () => void;
  onStop: () => void;
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
  onStop,
  status,
  progress,
  runMeta,
  error,
  dirty,
}: ControlsProps) {
  const showPup = config.treatment === "PUP" || config.treatment === "PUP+RM";
  const showRm = config.treatment === "RM" || config.treatment === "PUP+RM";
  const running = status === "running";

  // Numeric fields are edited as raw text so they can be fully erased/retyped
  // (a controlled number coerced empty -> 1, causing "type 5, get 15"). The
  // config's number is only updated when the text is valid; while it's
  // empty/invalid we surface an error and block Run.
  const [agentsText, setAgentsText] = useState(String(config.num_agents));
  const [daysText, setDaysText] = useState(String(config.num_days));
  const [seedsText, setSeedsText] = useState(String(config.num_seeds));
  const agents = validateInt(agentsText, 1, 5000);
  const days = validateInt(daysText, 1, 60);
  const seeds = validateInt(seedsText, 1, 12);
  const inputError = agents.error ?? days.error ?? seeds.error;

  const poisAvailable = cities.pois_available[config.city] ?? false;

  return (
    <aside className="sidebar">
      <h1>Welfare-RS</h1>
      <p className="muted">Welfare-oriented activity-travel simulation</p>

      <h2>City</h2>
      <label title="Two-layer metro: homes across the commuter counties, work and leisure POIs inside the principal city.">
        <select
          className="city-select"
          value={config.city}
          onChange={(e) => setConfig({ city: e.target.value })}
        >
          {cities.cities.map((c) => (
            <option key={c.key} value={c.key}>
              {c.label}
            </option>
          ))}
        </select>
      </label>
      <p className="muted">
        Homes span the metro's commuter counties; work &amp; leisure stay in the
        principal city. A city's first-ever run downloads its road network
        (several minutes) — later runs load from cache.
      </p>

      <h2>Settings</h2>
      <label>
        Agents
        <input
          type="number"
          min={1}
          max={5000}
          value={agentsText}
          aria-invalid={agents.error ? true : undefined}
          onChange={(e) => {
            setAgentsText(e.target.value);
            const v = validateInt(e.target.value, 1, 5000).value;
            if (v !== null) setConfig({ num_agents: v });
          }}
        />
        {agents.error && <span className="field-error">{agents.error}</span>}
      </label>
      <label>
        Days
        <input
          type="number"
          min={1}
          max={60}
          value={daysText}
          aria-invalid={days.error ? true : undefined}
          onChange={(e) => {
            setDaysText(e.target.value);
            const v = validateInt(e.target.value, 1, 60).value;
            if (v !== null) setConfig({ num_days: v });
          }}
        />
        {days.error && <span className="field-error">{days.error}</span>}
      </label>
      <label>
        Base random seed
        <input
          type="number"
          value={config.seed}
          onChange={(e) => setConfig({ seed: parseInt(e.target.value || "0", 10) })}
        />
      </label>
      <label title="Number of random seeds to sweep. The study runs all of them (in parallel), reports Table 1 with σ_U = std across seeds, and lets you switch which seed the map shows. Seeds are base .. base+N−1.">
        Seeds (run in parallel)
        <input
          type="number"
          min={1}
          max={12}
          value={seedsText}
          aria-invalid={seeds.error ? true : undefined}
          onChange={(e) => {
            setSeedsText(e.target.value);
            const v = validateInt(e.target.value, 1, 12).value;
            if (v !== null) setConfig({ num_seeds: v });
          }}
        />
        {seeds.error && <span className="field-error">{seeds.error}</span>}
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
          disabled={!poisAvailable}
          onChange={(e) => setConfig({ use_real_pois: e.target.checked })}
        />
        Real POIs {poisAvailable ? "" : "(no dataset for this city — synthetic)"}
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

      {running ? (
        <button className="stop-btn" onClick={onStop}>■ Stop simulation</button>
      ) : (
        <button
          className="run-btn"
          onClick={() => { if (!inputError) onRun(); }}
          disabled={!!inputError}
        >
          ▶ Run simulation
        </button>
      )}
      {!running && inputError && (
        <p className="error">Fix the highlighted settings above to run.</p>
      )}

      {running && progress && (
        <div className="progress">
          <div
            className="progress-bar"
            style={{ width: `${progress.total ? (progress.current / progress.total) * 100 : 5}%` }}
          />
          <span className="progress-text">
            {progress.current === 0
              ? "Preparing study (building network)…"
              : `Completed ${progress.current} of ${progress.total} seed${progress.total > 1 ? "s" : ""}…`}
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

function validateInt(
  text: string,
  lo: number,
  hi: number
): { value: number | null; error: string | null } {
  const t = text.trim();
  if (t === "") return { value: null, error: "Required" };
  if (!/^\d+$/.test(t)) return { value: null, error: "Whole number only" };
  const n = parseInt(t, 10);
  if (n < lo || n > hi) return { value: null, error: `Must be ${lo}–${hi}` };
  return { value: n, error: null };
}
