// Shared API types (mirror backend/models.py and backend/viz.py outputs).

export interface RunConfig {
  city: string;
  num_agents: number;
  num_days: number;
  seed: number;
  treatment: string;
  multimodal: boolean;
  use_real_pois: boolean;
  pup_alpha: number;
  rm_epsilon: number;
}

export interface Area {
  key: string;
  label: string;
}

export interface CitiesResponse {
  areas: Area[];
  treatments: string[];
  pois_available: boolean;
  defaults: RunConfig;
}

export interface RunMeta {
  run_id: string;
  config: RunConfig;
  summary: Record<string, any>;
  day_summaries: Array<Record<string, any>>;
  view: { latitude: number; longitude: number };
  bounds: { south: number; west: number; north: number; east: number };
  num_days: number;
  num_intersections: number;
  poi_count: number;
  poi_source: string;
  time_span: number;
  area_label: string;
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
  | { type: "error"; message: string };
