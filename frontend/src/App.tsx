import { useEffect, useMemo, useRef, useState } from "react";
import Controls, { type RunStatus } from "./components/Controls";
import MapView from "./components/MapView";
import Results from "./components/Results";
import { cancelRun, createRun, getCities, getRun, subscribeProgress } from "./api";
import type { CitiesResponse, RunConfig, RunMeta } from "./types";

export default function App() {
  const [cities, setCities] = useState<CitiesResponse | null>(null);
  const [config, setConfigState] = useState<RunConfig | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [runMeta, setRunMeta] = useState<RunMeta | null>(null);
  const [committed, setCommitted] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const runningIdRef = useRef<string | null>(null);  // the in-flight study, for Stop

  // When a study finishes, point the map at its first seed.
  useEffect(() => {
    if (runMeta) setSelectedSeed(runMeta.default_seed);
  }, [runMeta]);

  useEffect(() => {
    getCities()
      .then((c) => {
        setCities(c);
        setConfigState(c.defaults);
      })
      .catch((e) => setError(`Could not reach backend: ${e.message}`));
  }, []);

  const setConfig = (patch: Partial<RunConfig>) =>
    setConfigState((prev) => (prev ? { ...prev, ...patch } : prev));

  const dirty = useMemo(
    () => (config ? JSON.stringify(config) !== committed : false),
    [config, committed]
  );

  async function onRun() {
    if (!config) return;
    setError(null);
    setStatus("running");
    setProgress({ current: 0, total: config.num_seeds });
    try {
      const resp = await createRun(config);
      const finalize = async (runId: string) => {
        const meta = await getRun(runId);
        setRunMeta(meta);
        setCommitted(JSON.stringify(config));
        setStatus("ready");
        setProgress(null);
      };
      if (resp.status === "ready") {
        if (resp.meta) {
          setRunMeta(resp.meta);
          setCommitted(JSON.stringify(config));
          setStatus("ready");
          setProgress(null);
        } else {
          await finalize(resp.run_id);
        }
        return;
      }
      runningIdRef.current = resp.run_id;
      subscribeProgress(resp.run_id, (e) => {
        if (e.type === "progress") {
          setProgress({ current: e.current, total: e.total });
        } else if (e.type === "done") {
          runningIdRef.current = null;
          void finalize(e.run_id);
        } else if (e.type === "cancelled") {
          // Clean stop: reset to a fresh, empty state (no partial viz shown).
          runningIdRef.current = null;
          setStatus("idle");
          setProgress(null);
          setRunMeta(null);
        } else if (e.type === "error") {
          runningIdRef.current = null;
          setError(e.message);
          setStatus("error");
          setProgress(null);
        }
      });
    } catch (e: any) {
      setError(e.message ?? String(e));
      setStatus("error");
      setProgress(null);
    }
  }

  function onStop() {
    const id = runningIdRef.current;
    if (id) void cancelRun(id);  // SSE 'cancelled' will reset the UI to idle
  }

  if (!cities || !config) {
    return (
      <div className="loading-screen">
        {error ? <p className="error">{error}</p> : <p>Loading…</p>}
      </div>
    );
  }

  return (
    <div className="layout">
      <Controls
        cities={cities}
        config={config}
        setConfig={setConfig}
        onRun={onRun}
        onStop={onStop}
        status={status}
        progress={progress}
        runMeta={runMeta}
        error={error}
        dirty={dirty}
      />
      <main className="content">
        {status === "idle" && (
          <div className="placeholder">
            <p>Pick an area, set options on the left, then press ▶ Run simulation.</p>
          </div>
        )}
        {runMeta && (
          <>
            <section>
              <div className="section-head">
                <h3>
                  Activity-travel over the day · {runMeta.area_label} · {runMeta.config.treatment}
                </h3>
                {runMeta.seeds.length > 1 && (
                  <label className="seed-picker">
                    Seed
                    <select
                      value={selectedSeed ?? runMeta.default_seed}
                      onChange={(e) => setSelectedSeed(parseInt(e.target.value, 10))}
                    >
                      {runMeta.seeds.map((s, i) => (
                        <option key={s} value={s}>
                          {s}
                          {i === 0 ? " (base)" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
              <MapView run={runMeta} seed={selectedSeed ?? runMeta.default_seed} />
              <p className="legend">
                <b>Dots = agents:</b> <Dot c="#e65038" /> in transit · <Dot c="#50aa5a" /> home ·{" "}
                <Dot c="#f0961e" /> work · <Dot c="#a05ad2" /> leisure · <Dot c="#6e6e78" /> POIs ·{" "}
                hover a dot for details, <b>click an in-transit dot</b> to trace its street route.
              </p>
            </section>
            <Results run={runMeta} />
          </>
        )}
      </main>
    </div>
  );
}

function Dot({ c }: { c: string }) {
  return <span className="legend-dot" style={{ background: c }} />;
}
