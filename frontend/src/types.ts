// Shared API types (mirror backend/models.py and backend/viz.py outputs).

/** One recommender the user built. Every field is a knob on the same scorer. */
export interface RecommenderSpec {
  label: string;
  w_rating: number;
  w_reviews: number;
  w_relevance: number;
  w_proximity: number;
  w_popularity: number;
  w_personalization: number;
  distance_scale_km: number;
  popularity_gamma: number;
  learning_rate: number;
  /** "" | "pup" | "rm" | "pup_rm" */
  welfare_gate: string;
  pup_alpha: number;
  rm_epsilon: number;
  random_ranking: boolean;
}

/** The six weighted components, in the order they are shown to the user. */
export const WEIGHT_KEYS = [
  "w_rating",
  "w_reviews",
  "w_relevance",
  "w_proximity",
  "w_popularity",
  "w_personalization",
] as const;
export type WeightKey = (typeof WEIGHT_KEYS)[number];

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
  recommenders: RecommenderSpec[];
  use_real_pois: boolean;
}

/** A preset is a set of values for the same knobs, not a different design. */
export interface Preset extends Partial<RecommenderSpec> {
  name: string;
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
  presets: Preset[]; // starting points for the recommender builder
  control: string; // the pinned No-RS control's label
  max_recommenders: number;
  treatments: string[]; // deprecated fixed vocabulary
  pois_available: Record<string, boolean>; // per city key
  warmed?: Record<string, boolean>; // road network cached on disk
  precomputed?: Record<string, boolean>; // routing matrices precomputed
  defaults: RunConfig;
}

// ── Dashboard metrics ────────────────────────────────────────────────────────

export interface SegmentCell {
  n: number;
  mean_utility: number;
  neg_rate: number;
}

export interface SpatialCell {
  n: number;
  mean_distance_km: number;
  mean_emissions_g: number;
  mean_travel_min: number;
  median_detour: number;
  median_excess_km: number;
}

/** One simulated day, averaged across seeds. Indexable, because the chart
 *  frame builder pulls a metric out by name. */
export interface DayRecord {
  [metric: string]: number;
  day: number;
  leisure_trips: number;
  acceptance_rate: number;
  mean_utility: number;
  gini: number;
  top1_share: number;
  top5_share: number;
  top10_share: number;
  coverage: number;
  total_visits: number;
  n_places: number;
  all_rate: number;
  recommended_rate: number;
  organic_rate: number;
  all_n: number;
  recommended_n: number;
  organic_n: number;
}

export interface ConditionMetrics {
  per_day: DayRecord[];
  segments: Record<string, Record<string, SegmentCell>>; // income | trust | paradigm
  category_by_income: Record<string, Record<string, number>>;
  spatial: { overall: SpatialCell; by_income: Record<string, SpatialCell> };
  lorenz: [number, number][];
  footfall: Record<string, number>;
  taste: Record<string, number>;
}

/** One agent's home, carrying each arm's utility delta vs the control. */
export interface WelfareMapRow {
  lat: number;
  lon: number;
  income: string;
  delta: Record<string, number>;
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
  metrics: Record<string, ConditionMetrics>; // dashboard panels, per condition
  welfare_map: WelfareMapRow[]; // agent homes + per-condition utility delta
  income_bands: string[]; // poorest-first, for stable chart axes
  trust_groups: string[]; // Q1..Q4
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
