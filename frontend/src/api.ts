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

export function getTimeline(
  runId: string,
  t0: number,
  t1: number,
  bbox?: BBox
): Promise<TimelineResponse> {
  return getJSON<TimelineResponse>(
    `${API}/runs/${runId}/timeline?t0=${t0}&t1=${t1}${bboxParam(bbox)}`
  );
}

export function getTripGeometry(runId: string, tripId: string): Promise<TripGeometry> {
  return getJSON<TripGeometry>(`${API}/runs/${runId}/trips/${tripId}/geometry`);
}

export async function getPois(runId: string, bbox?: BBox): Promise<Poi[]> {
  const data = await getJSON<{ pois: Poi[] }>(
    `${API}/runs/${runId}/pois?limit=4000${bboxParam(bbox)}`
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
      if (e.type === "done" || e.type === "error") src.close();
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
