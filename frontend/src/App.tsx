import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Controls, { type RunStatus } from "./components/Controls";
import MapView from "./components/MapView";
import Results from "./components/Results";
import RunList from "./components/RunList";
import { BLANK_SPEC } from "./components/RecommenderBuilder";
import { condLabel } from "./components/charts";
import { cancelRun, createRun, fetchRun, getCities, getRun, subscribeProgress } from "./api";
import type { CitiesResponse, RunConfig, RunMeta, RunProgress } from "./types";

/** Read/write the ?run=<id> query param, so a run survives reload and can be
 *  opened in another window by pasting the URL. */
function urlRunId(): string | null {
  return new URLSearchParams(window.location.search).get("run");
}
function setUrlRunId(runId: string | null) {
  const url = new URL(window.location.href);
  if (runId) url.searchParams.set("run", runId);
  else url.searchParams.delete("run");
  window.history.replaceState(null, "", url.toString());
}

export default function App() {
  const [cities, setCities] = useState<CitiesResponse | null>(null);
  const [config, setConfigState] = useState<RunConfig | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [runMeta, setRunMeta] = useState<RunMeta | null>(null);
  const [committed, setCommitted] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const [selectedCondition, setSelectedCondition] = useState<string | null>(null);
  const [runsToken, setRunsToken] = useState(0);   // bump to refresh the run list
  const runningIdRef = useRef<string | null>(null);  // the in-flight study, for Stop
  // Live SSE stream, if any. Held so switching runs can close the previous one:
  // two open streams would both write progress/status and fight over the UI.
  const unsubRef = useRef<(() => void) | null>(null);
  const bumpRuns = () => setRunsToken((t) => t + 1);

  const stopWatching = useCallback(() => {
    unsubRef.current?.();
    unsubRef.current = null;
    runningIdRef.current = null;
  }, []);

  // Close the stream if the app unmounts mid-run.
  useEffect(() => () => unsubRef.current?.(), []);

  // When a study finishes, point the map at its first seed and first recommender.
  useEffect(() => {
    if (runMeta) {
      setSelectedSeed(runMeta.default_seed);
      setSelectedCondition(runMeta.default_condition);
    }
  }, [runMeta]);

  useEffect(() => {
    getCities()
      .then((c) => {
        setCities(c);
        // `prev ?? defaults` because a ?run= attach may have already loaded that
        // run's settings into the form; this fetch resolving later must not
        // clobber them. Whichever lands first, the adopted run wins.
        //
        // The server's default recommender list is empty (a control-only study
        // is valid), but an empty builder is a poor first screen — seed it with
        // the first preset so there is something to run and something to edit.
        setConfigState((prev) => {
          if (prev) return prev;
          const seeded = c.presets[0]
            ? [{ ...BLANK_SPEC, ...c.presets[0], label: c.presets[0].name }]
            : [];
          return { ...c.defaults, recommenders: c.defaults.recommenders?.length
            ? c.defaults.recommenders
            : seeded };
        });
      })
      .catch((e) => setError(`Could not reach backend: ${e.message}`));
  }, []);

  const setConfig = (patch: Partial<RunConfig>) =>
    setConfigState((prev) => (prev ? { ...prev, ...patch } : prev));

  const dirty = useMemo(
    () => (config ? JSON.stringify(config) !== committed : false),
    [config, committed]
  );

  // Which condition the map and the headline metrics are showing. Validated
  // against the current study rather than trusted: on the render between a new
  // runMeta arriving and the reset effect firing, selectedCondition still holds
  // the previous study's pick, which may not exist in this one.
  const condition =
    selectedCondition && runMeta?.conditions.includes(selectedCondition)
      ? selectedCondition
      : runMeta?.default_condition ?? "";

  // Show a finished study: adopt its settings into the form so the sidebar
  // describes what you are looking at, and name it in the URL. The concrete
  // seed is dropped — keeping it would make the next Run silently replay this
  // study instead of drawing a fresh sample.
  const adoptRun = useCallback((meta: RunMeta) => {
    const { seed: _seed, ...cfg } = meta.config;
    const adopted = cfg as RunConfig;
    setConfigState(adopted);
    setCommitted(JSON.stringify(adopted));
    setRunMeta(meta);
    setStatus("ready");
    setProgress(null);
    setUrlRunId(meta.run_id);
    bumpRuns();
  }, []);

  // Follow a study to completion. Used both for runs this tab started and for
  // ones it attached to — Job.subscribe replays the current tick on connect, so
  // an attaching tab gets a populated progress bar immediately.
  const watchRun = useCallback(
    (runId: string) => {
      stopWatching(); // never leave a previous run's stream open
      runningIdRef.current = runId;
      setUrlRunId(runId);
      unsubRef.current = subscribeProgress(runId, (e) => {
        if (e.type === "progress") {
          setProgress({
            current: e.current,
            total: e.total,
            cond_total: e.cond_total,
            per_condition: e.per_condition,
            elapsed_sec: e.elapsed_sec,
            observed_at: Date.now(),
          });
        } else if (e.type === "done") {
          stopWatching();
          getRun(e.run_id)
            .then(adoptRun)
            .catch((err) => {
              setError(err.message ?? String(err));
              setStatus("error");
              setProgress(null);
            });
        } else if (e.type === "cancelled") {
          // Clean stop: reset to a fresh, empty state (no partial viz shown).
          stopWatching();
          setStatus("idle");
          setProgress(null);
          setRunMeta(null);
          setUrlRunId(null);
          bumpRuns();
        } else if (e.type === "error") {
          stopWatching();
          setError(e.message);
          setStatus("error");
          setProgress(null);
          bumpRuns();
        }
      });
    },
    [adoptRun, stopWatching]
  );

  // Open a run this tab may not have started: a ?run= link, a reload, or a pick
  // from the run list.
  const attachTo = useCallback(
    async (runId: string) => {
      setError(null);
      try {
        const found = await fetchRun(runId);
        if (found.state === "ready") {
          stopWatching(); // leaving a run that was still streaming
          adoptRun(found.meta);
        } else if (found.state === "running") {
          setRunMeta(null);
          setStatus("running");
          setProgress({ current: 0, total: 0, elapsed_sec: 0, observed_at: Date.now() });
          watchRun(runId);
        } else {
          stopWatching();
          setError(`That run can't be opened — ${found.reason}`);
          setStatus("idle");
          setRunMeta(null);
          setUrlRunId(null);
        }
      } catch (e: any) {
        setError(e.message ?? String(e));
      }
      bumpRuns();
    },
    [adoptRun, watchRun, stopWatching]
  );

  // On first load, open whatever run the URL names.
  const bootedRef = useRef(false);
  useEffect(() => {
    if (bootedRef.current) return;
    bootedRef.current = true;
    const id = urlRunId();
    if (id) void attachTo(id);
  }, [attachTo]);

  async function onRun() {
    if (!config) return;
    setError(null);
    setStatus("running");
    // Placeholder until the first real tick; the server owns the true total
    // (simulated days across the whole seed x condition grid).
    setProgress({ current: 0, total: 0, elapsed_sec: 0, observed_at: Date.now() });
    try {
      const resp = await createRun(config);
      bumpRuns();
      if (resp.status === "ready") {
        // Cache hit — identical config already computed.
        if (resp.meta) adoptRun(resp.meta);
        else await attachTo(resp.run_id);
        return;
      }
      watchRun(resp.run_id);
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
      >
        <RunList
          activeRunId={runMeta?.run_id ?? runningIdRef.current}
          onOpen={(id) => void attachTo(id)}
          refreshToken={runsToken}
        />
      </Controls>
      <main className="content">
        {status === "idle" && (
          <div className="placeholder">
            <p>Pick a city, set options on the left, then press ▶ Run simulation.</p>
          </div>
        )}
        {runMeta && (
          <>
            <section>
              <div className="section-head">
                <h3>
                  Activity-travel over the day · {runMeta.city_label} ·{" "}
                  {condLabel(condition)}
                </h3>
                <div className="viz-pickers">
                  {runMeta.conditions.length > 1 && (
                    <label className="seed-picker">
                      Showing
                      <select
                        value={condition}
                        onChange={(e) => setSelectedCondition(e.target.value)}
                      >
                        {runMeta.conditions.map((c) => (
                          <option key={c} value={c}>
                            {condLabel(c)}
                            {c === "No RS" ? " (control)" : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
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
              </div>
              <MapView
                run={runMeta}
                seed={selectedSeed ?? runMeta.default_seed}
                condition={condition}
              />
              <p className="legend">
                <b>Dots = agents:</b> <Dot c="#e65038" /> in transit · <Dot c="#50aa5a" /> home ·{" "}
                <Dot c="#f0961e" /> work · <Dot c="#a05ad2" /> leisure · <Dot c="#6e6e78" /> POIs ·{" "}
                hover a dot for details, <b>click an in-transit dot</b> to trace its street route.
              </p>
            </section>
            <Results run={runMeta} condition={condition} />
          </>
        )}
      </main>
    </div>
  );
}

function Dot({ c }: { c: string }) {
  return <span className="legend-dot" style={{ background: c }} />;
}
