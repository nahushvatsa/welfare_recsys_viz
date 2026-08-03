// Shared API types (mirror backend/models.py and backend/viz.py outputs).

export interface RunConfig {
  city: string;
  num_agents: number;
  num_days: number;
  // Base seed for the sweep (seed .. seed + num_seeds - 1). Omitted by the UI
  // so the backend draws a fresh one per run; send it only to replay a study.
  seed?: number;
  num_seeds: number;
  // Recommenders to run. The No-RS control is always added server-side, so an
  // empty array is a valid "control only" study.
  conditions: string[];
  multimodal: boolean;
  use_real_pois: boolean;
  pup_alpha: number;
  rm_epsilon: number;
}

// Paper Table 1 (aggregate welfare) — one row per condition, averaged over seeds.
export interface Table1Row {
  condition: string;
  mean_utility: number; // Ū
  sigma_u: number; // std of per-seed Ū
  neg_rate: number; // fraction of leisure trips with U < 0
  abstention_rate: number; // fraction of opportunities the RS withheld
  gini: number;
  n_leisure_trips: number;
}

// Paper Table 2 (over-recommendation cost), treatment vs matched No-RS.
export interface Table2Row {
  condition: string;
  harmed_pct: number;
  improved_pct: number;
  mean_orc: number;
  n_matched: number;
}

export interface Headline {
  mean_utility: number;
  sigma_u: number;
  neg_rate: number;
  harmed_pct: number | null;
  improved_pct: number | null;
  mean_orc: number | null;
}

export interface City {
  key: string;
  label: string;
}

export interface CitiesResponse {
  cities: City[];
  treatments: string[]; // includes "No RS"; the UI pins it as the control
  pois_available: Record<string, boolean>; // per city key
  warmed?: Record<string, boolean>; // road network cached on disk
  precomputed?: Record<string, boolean>; // routing matrices precomputed
  defaults: RunConfig;
}

export interface RunMeta {
  run_id: string;
  config: RunConfig;
  seeds: number[];
  default_seed: number;
  conditions: string[]; // "No RS" first, then the recommenders that ran
  default_condition: string; // first recommender, else "No RS"
  // {condition: {seed: time span}} — the scrubber's extent depends on both.
  time_spans: Record<string, Record<string, number>>;
  table1: Table1Row[]; // one row per condition
  table2: Table2Row[]; // one row per recommender (the control has none)
  headlines: Record<string, Headline>; // per condition
  aggregate: Record<string, Record<string, number>>; // per condition
  view: { latitude: number; longitude: number };
  bounds: { south: number; west: number; north: number; east: number }; // full metro
  core_bounds: { south: number; west: number; north: number; east: number }; // principal city
  num_days: number;
  num_intersections: number;
  poi_count: number;
  poi_source: string;
  city_label: string;
  created_at: number | null; // epoch seconds, when the study was submitted
  duration_sec: number | null; // wall-clock time the study took
}

/** One row of GET /api/runs — enough to list and open a run, no geometry. */
export interface RunSummary {
  run_id: string;
  status: "running" | "done" | "error" | "cancelled";
  /** False when a finished run's results have been evicted from memory. */
  available: boolean;
  city: string | null;
  city_label: string | null;
  conditions: string[];
  num_agents: number | null;
  num_days: number | null;
  num_seeds: number | null;
  created_at: number | null; // epoch seconds
  finished_at: number | null;
  error: string | null;
  progress: RunProgress;
}

export interface CreateRunResponse {
  run_id: string;
  status: "ready" | "running";
  meta?: RunMeta;
}

export type RGB = [number, number, number];

export interface Trip {
  agent_id: number;
  trip_id: string;
  path: [number, number][]; // [lon, lat]
  timestamps: number[];
  mode: string;
  color: RGB;
}

export interface Stay {
  agent_id: number;
  t0: number;
  t1: number;
  lon: number;
  lat: number;
  state: string;
  color: RGB;
}

export interface TimelineResponse {
  condition: string; // which condition the server actually resolved to
  seed: number;
  t0: number;
  t1: number;
  trips: Trip[];
  stays: Stay[];
}

export interface TripGeometry {
  trip_id: string;
  agent_id: number;
  path: [number, number][];
  timestamps: number[];
  mode: string;
  purpose: string;
  color: RGB;
}

export interface Poi {
  lon: number;
  lat: number;
  category: string;
  label: string;
}

export type BBox = [number, number, number, number]; // south, west, north, east

/**
 * Study progress, counted in simulated days. `current`/`total` cover the whole
 * (seed x condition) grid; `per_condition` breaks it down per arm — all of them
 * run at once, so a single figure would hide that.
 */
export interface RunProgress {
  current: number;
  total: number;
  cond_total?: number; // per-arm denominator (seeds x days)
  per_condition?: Record<string, number>; // condition -> sim-days done
  /** Seconds since the study was submitted, measured on the SERVER. */
  elapsed_sec?: number;
  /** Client-side only: Date.now() when elapsed_sec was received, so the UI can
   *  advance the timer between ticks without trusting the browser clock to
   *  agree with the server's. */
  observed_at?: number;
}

export type ProgressEvent =
  | ({ type: "progress" } & RunProgress)
  | { type: "done"; run_id: string }
  | { type: "error"; message: string }
  | { type: "cancelled"; run_id: string };
