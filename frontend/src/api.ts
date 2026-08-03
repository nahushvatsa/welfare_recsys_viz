// Typed client for the FastAPI backend. Same-origin in prod (FastAPI serves the
// built app); Vite proxies /api to uvicorn in dev.

import type {
  BBox,
  CitiesResponse,
  CreateRunResponse,
  Poi,
  ProgressEvent,
  RunConfig,
  RunMeta,
  RunSummary,
  TimelineResponse,
  TripGeometry,
} from "./types";

const API = "/api";

function bboxParam(bbox?: BBox): string {
  return bbox ? `&bbox=${bbox.join(",")}` : "";
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return res.json() as Promise<T>;
}

export function getCities(): Promise<CitiesResponse> {
  return getJSON<CitiesResponse>(`${API}/cities`);
}

export async function createRun(config: RunConfig): Promise<CreateRunResponse> {
  const res = await fetch(`${API}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`createRun failed: ${res.status}`);
  return res.json() as Promise<CreateRunResponse>;
}

export function getRun(runId: string): Promise<RunMeta> {
  return getJSON<RunMeta>(`${API}/runs/${runId}`);
}

/** All runs this backend knows about, newest first. */
export async function listRuns(): Promise<RunSummary[]> {
  const data = await getJSON<{ runs: RunSummary[] }>(`${API}/runs`);
  return data.runs;
}

export type RunLookup =
  | { state: "ready"; meta: RunMeta }
  | { state: "running" }
  | { state: "gone"; reason: string };

/**
 * Resolve a run id that this tab may not have started (a ?run= link, a reload,
 * a pick from the run list).
 *
 * The 202 check MUST come before `res.ok`: FastAPI returns 202 with a
 * `{detail: ...}` body for "still computing", and 202 is inside the ok range —
 * so testing `res.ok` first would parse that detail object as a RunMeta and
 * hand the UI a run with no tables, no seeds and no bounds.
 */
export async function fetchRun(runId: string): Promise<RunLookup> {
  const res = await fetch(`${API}/runs/${runId}`);
  if (res.status === 202) return { state: "running" };
  if (res.ok) return { state: "ready", meta: (await res.json()) as RunMeta };
  let reason = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body?.detail) reason = String(body.detail);
  } catch {
    /* non-JSON error body — keep the status line */
  }
  return { state: "gone", reason };
}

/** Request a clean stop of a running study (aborts within ~one day). */
export async function cancelRun(runId: string): Promise<void> {
  await fetch(`${API}/runs/${runId}/cancel`, { method: "POST" });
}

function seedParam(seed?: number): string {
  return seed == null ? "" : `&seed=${seed}`;
}

/**
 * Condition names must be percent-encoded: "PUP+RM" contains a literal '+',
 * which a query string would otherwise decode as a space ("PUP RM") and the
 * server would not recognise.
 */
function conditionParam(condition?: string): string {
  return condition == null ? "" : `&condition=${encodeURIComponent(condition)}`;
}

export function getTimeline(
  runId: string,
  t0: number,
  t1: number,
  bbox?: BBox,
  seed?: number,
  condition?: string
): Promise<TimelineResponse> {
  return getJSON<TimelineResponse>(
    `${API}/runs/${runId}/timeline?t0=${t0}&t1=${t1}` +
      `${bboxParam(bbox)}${seedParam(seed)}${conditionParam(condition)}`
  );
}

export function getTripGeometry(
  runId: string,
  tripId: string,
  seed?: number,
  condition?: string
): Promise<TripGeometry> {
  return getJSON<TripGeometry>(
    `${API}/runs/${runId}/trips/${tripId}/geometry?_=1` +
      `${seedParam(seed)}${conditionParam(condition)}`
  );
}

export async function getPois(runId: string, bbox?: BBox, seed?: number): Promise<Poi[]> {
  const data = await getJSON<{ pois: Poi[] }>(
    `${API}/runs/${runId}/pois?limit=4000${bboxParam(bbox)}${seedParam(seed)}`
  );
  return data.pois;
}

/**
 * Subscribe to per-day progress for a run via SSE. Returns an unsubscribe fn.
 */
export function subscribeProgress(
  runId: string,
  onEvent: (e: ProgressEvent) => void
): () => void {
  const src = new EventSource(`${API}/runs/${runId}/events`);
  src.onmessage = (msg) => {
    try {
      const e = JSON.parse(msg.data) as ProgressEvent;
      onEvent(e);
      if (e.type === "done" || e.type === "error" || e.type === "cancelled") src.close();
    } catch {
      /* ignore keep-alive comments */
    }
  };
  src.onerror = () => {
    // Network/stream error — close and let the caller fall back to polling.
    src.close();
  };
  return () => src.close();
}
