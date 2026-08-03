import { useEffect, useState } from "react";
import type { CitiesResponse, RunConfig, RunMeta, RunProgress } from "../types";

export type RunStatus = "idle" | "running" | "ready" | "error";

interface ControlsProps {
  cities: CitiesResponse;
  config: RunConfig;
  setConfig: (patch: Partial<RunConfig>) => void;
  onRun: () => void;
  onStop: () => void;
  status: RunStatus;
  progress: RunProgress | null;
  runMeta: RunMeta | null;
  error: string | null;
  dirty: boolean;
  /** Rendered at the foot of the sidebar (the run list). */
  children?: React.ReactNode;
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
  children,
}: ControlsProps) {
  // "No RS" is the control: always run, never a choice, so it is pinned rather
  // than listed. Everything else is togglable.
  const recommenders = cities.treatments.filter((t) => t !== "No RS");
  const selected = config.conditions ?? [];
  const showPup = selected.some((t) => t === "PUP" || t === "PUP+RM");
  const showRm = selected.some((t) => t === "RM" || t === "PUP+RM");
  const running = status === "running";

  function toggleCondition(name: string, on: boolean) {
    const next = new Set(selected);
    if (on) next.add(name);
    else next.delete(name);
    // Emit in the server's canonical order, so ticking A-then-B produces the
    // same array as B-then-A. Otherwise the parent's JSON.stringify dirty check
    // would report a change when nothing actually changed.
    setConfig({ conditions: recommenders.filter((t) => next.has(t)) });
  }

  // Numeric fields are edited as raw text so they can be fully erased/retyped
  // (a controlled number coerced empty -> 1, causing "type 5, get 15"). The
  // config's number is only updated when the text is valid; while it's
  // empty/invalid we surface an error and block Run.
  const [agentsText, setAgentsText] = useState(String(config.num_agents));
  const [daysText, setDaysText] = useState(String(config.num_days));
  const [seedsText, setSeedsText] = useState(String(config.num_seeds));
  const agents = validateInt(agentsText, 1, 100000);
  const days = validateInt(daysText, 1, 60);
  const seeds = validateInt(seedsText, 1, 12);
  const inputError = agents.error ?? days.error ?? seeds.error;

  // Study size, for the cost hint below. Falls back to the committed seed count
  // so the number doesn't blank out while the field is mid-edit.
  const nSeeds = seeds.value ?? config.num_seeds;
  const nConditions = selected.length + 1; // + the No-RS control

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
          max={100000}
          value={agentsText}
          aria-invalid={agents.error ? true : undefined}
          onChange={(e) => {
            setAgentsText(e.target.value);
            const v = validateInt(e.target.value, 1, 100000).value;
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
      <label title="Number of random seeds to sweep. The study runs all of them (in parallel), reports Table 1 with σ_U = std across seeds, and lets you switch which seed the map shows. The base seed is drawn fresh on every run — the seeds used are listed with the results, so a study can be replayed by posting them to the API.">
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

      <h2>Recommenders</h2>
      <p className="muted">
        Pick any number. Each one runs as its own simulation, in parallel,
        against the same No-RS control — then you can compare their metrics and
        switch the map between them.
      </p>
      <label
        className="toggle"
        title="The baseline every comparison is paired against. Always run."
      >
        <input type="checkbox" checked disabled readOnly />
        No RS <span className="muted">(control)</span>
      </label>
      {recommenders.map((t) => (
        <label className="toggle" key={t}>
          <input
            type="checkbox"
            checked={selected.includes(t)}
            onChange={(e) => toggleCondition(t, e.target.checked)}
          />
          {t}
        </label>
      ))}
      <p className="run-cost">
        {nConditions} condition{nConditions === 1 ? "" : "s"} × {nSeeds} seed
        {nSeeds === 1 ? "" : "s"} = <b>{nConditions * nSeeds} simulations</b>
        {nConditions * nSeeds > 1 ? ", run in parallel" : ""}
      </p>
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
        <div className="progress-panel">
          <div className="progress">
            <div
              className="progress-bar"
              style={{ width: `${progress.total ? (progress.current / progress.total) * 100 : 5}%` }}
            />
          </div>
          <p className="progress-text">
            <span>
              {progress.current === 0
                ? "Preparing study (loading network)…"
                : `${progress.current} of ${progress.total} simulated days`}
            </span>
            <span className="elapsed">
              <LiveTimer
                baseSec={progress.elapsed_sec ?? 0}
                observedAt={progress.observed_at ?? Date.now()}
              />
            </span>
          </p>
          {/* Every condition runs at the same time, on its own cores. One row
              each, so you can watch them advance together — an overall figure
              alone reads as if the study were sequential. */}
          {progress.per_condition && progress.cond_total ? (
            <ul className="cond-progress">
              {Object.entries(progress.per_condition).map(([name, done]) => (
                <li key={name}>
                  <span className="cond-name">{name}</span>
                  <span className="cond-track">
                    <span
                      className="cond-fill"
                      style={{ width: `${(done / progress.cond_total!) * 100}%` }}
                    />
                  </span>
                  <span className="cond-count">
                    {done}/{progress.cond_total}
                  </span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      )}

      {dirty && status === "ready" && (
        <p className="hint">⚙️ Settings changed — press Run to apply.</p>
      )}
      {error && <p className="error">{error}</p>}

      {runMeta && status === "ready" && (
        <div className="captions">
          {runMeta.duration_sec != null && (
            <p className="took">
              ⏱ Ran in <b>{fmtDuration(runMeta.duration_sec)}</b>
            </p>
          )}
          <p>Network: {runMeta.num_intersections.toLocaleString()} road intersections</p>
          <p>
            POIs: {runMeta.poi_count.toLocaleString()} ({runMeta.poi_source})
          </p>
        </div>
      )}

      {children}
    </aside>
  );
}

/** "2m 14s" / "1h 03m 20s" — compact, no leading zero units. */
export function fmtDuration(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m ${String(r).padStart(2, "0")}s`;
  if (m) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${r}s`;
}

/**
 * Ticking elapsed time for a study in flight.
 *
 * Anchored on the server's `elapsed_sec` and advanced with the local clock's
 * *delta* since that value arrived — never with absolute local time, which may
 * disagree with the server's by minutes and would show a nonsense duration on
 * an attached run.
 */
function LiveTimer({ baseSec, observedAt }: { baseSec: number; observedAt: number }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, []);
  return <>{fmtDuration(baseSec + (Date.now() - observedAt) / 1000)}</>;
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
