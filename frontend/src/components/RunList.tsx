import { useCallback, useEffect, useState } from "react";
import { listRuns } from "../api";
import { fmtDuration } from "./Controls";
import type { RunSummary } from "../types";

interface RunListProps {
  activeRunId: string | null;
  onOpen: (runId: string) => void;
  /** Bumped by the parent to refresh immediately instead of waiting for a poll. */
  refreshToken: number;
}

function ago(ts: number | null): string {
  if (!ts) return "";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/**
 * Studies known to the backend, including ones this browser never started.
 *
 * Run state lives in the server process, not in the tab — so without this list
 * a second window had no way to reach a study already in flight, and a reload
 * lost it entirely.
 */
export default function RunList({ activeRunId, onOpen, refreshToken }: RunListProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);

  const refresh = useCallback(() => {
    // A failed poll is not worth surfacing: the next tick retries, and the run
    // the user is actually watching has its own SSE stream for errors.
    listRuns()
      .then(setRuns)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, refreshToken]);

  useEffect(() => {
    const id = window.setInterval(() => {
      if (!document.hidden) refresh(); // don't poll a backgrounded tab
    }, 4000);
    return () => window.clearInterval(id);
  }, [refresh]);

  if (runs.length === 0) return null;

  return (
    <div className="run-list-wrap">
      <h2>Runs</h2>
      <p className="muted">
        Studies on this server — including any started in another window.
      </p>
      <ul className="run-list">
        {runs.map((r) => {
          // A finished study whose results were evicted is listed for the
          // record but can't be opened.
          const openable = r.status === "running" || r.available;
          const pct = r.progress.total
            ? (r.progress.current / r.progress.total) * 100
            : 0;
          return (
            <li
              key={r.run_id}
              className={
                "run-row" +
                (r.run_id === activeRunId ? " run-row-active" : "") +
                (openable ? "" : " run-row-dead")
              }
              onClick={() => openable && onOpen(r.run_id)}
              title={openable ? "Open this run" : "Results no longer held in memory"}
            >
              <div className="run-row-head">
                <span className={`run-badge run-badge-${r.status}`}>{r.status}</span>
                <span className="run-city">{r.city_label ?? r.city ?? "—"}</span>
                <span className="run-when">{ago(r.finished_at ?? r.created_at)}</span>
              </div>
              <div className="run-row-sub">
                {r.num_agents != null ? r.num_agents.toLocaleString() : "?"} agents ·{" "}
                {r.num_days}d · {r.num_seeds} seed{r.num_seeds === 1 ? "" : "s"} ·{" "}
                {r.conditions.length} condition{r.conditions.length === 1 ? "" : "s"}
                {r.created_at && r.finished_at
                  ? ` · ran in ${fmtDuration(r.finished_at - r.created_at)}`
                  : ""}
              </div>
              {r.status === "running" && (
                <span className="run-track">
                  <span className="run-fill" style={{ width: `${pct}%` }} />
                </span>
              )}
              {!openable && <div className="run-row-sub run-dead">results expired</div>}
              {r.status === "error" && r.error && (
                <div className="run-row-sub run-dead">{r.error}</div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
