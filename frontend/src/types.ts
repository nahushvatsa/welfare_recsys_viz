// Shared API types (mirror backend/models.py and backend/viz.py outputs).

export interface RunConfig {
  city: string;
  num_agents: number;
  num_days: number;
  // Base seed for the sweep (seed .. seed + num_seeds - 1). Omitted by the UI
  // so the backend draws a fresh one per run; send it only to replay a study.
  seed?: number;
  num_seeds: number;
  treatment: string;
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
  treatments: string[];
  pois_available: Record<string, boolean>; // per city key
  defaults: RunConfig;
}

export interface RunMeta {
  run_id: string;
  config: RunConfig;
  seeds: number[];
  default_seed: number;
  time_spans: Record<string, number>; // per-seed time span (keyed by seed string)
  table1: Table1Row[];
  table2: Table2Row | null;
  headline: Headline;
  aggregate: Record<string, number>;
  view: { latitude: number; longitude: number };
  bounds: { south: number; west: number; north: number; east: number }; // full metro
  core_bounds: { south: number; west: number; north: number; east: number }; // principal city
  num_days: number;
  num_intersections: number;
  poi_count: number;
  poi_source: string;
  city_label: string;
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

export type ProgressEvent =
  | { type: "progress"; current: number; total: number }
  | { type: "done"; run_id: string }
  | { type: "error"; message: string }
  | { type: "cancelled"; run_id: string };
