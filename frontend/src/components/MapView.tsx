import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { ScatterplotLayer } from "@deck.gl/layers";
import { TripsLayer } from "@deck.gl/geo-layers";
import { getPois, getTimeline, getTripGeometry } from "../api";
import type { BBox, Poi, RunMeta, Trip, Stay, TripGeometry } from "../types";

const DAY = 1440;
const TRAIL = 60; // minutes of trail for the inspected trip

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

interface MapViewProps {
  run: RunMeta;
  seed: number;
}

interface Marker {
  position: [number, number];
  color: [number, number, number];
  agentId: number;
  tripId?: string;
  label: string;
}

interface Want {
  day: number;
  bbox: BBox | undefined;
  key: string;
}

function keyOf(day: number, bbox: BBox | undefined, seed: number): string {
  return `${seed}|${day}|${bbox ? bbox.map((n) => n.toFixed(3)).join(",") : ""}`;
}

// Piecewise-linear position along a downsampled trip at minute t -> [lon, lat].
function interp(trip: Trip, t: number): [number, number] {
  const ts = trip.timestamps;
  const p = trip.path;
  if (t <= ts[0]) return p[0];
  if (t >= ts[ts.length - 1]) return p[p.length - 1];
  for (let i = 1; i < ts.length; i++) {
    if (ts[i] >= t) {
      const f = (t - ts[i - 1]) / (ts[i] - ts[i - 1] || 1);
      return [p[i - 1][0] + (p[i][0] - p[i - 1][0]) * f, p[i - 1][1] + (p[i][1] - p[i - 1][1]) * f];
    }
  }
  return p[p.length - 1];
}

function fmtClock(t: number): string {
  const day = Math.floor(t / DAY) + 1;
  const m = t % DAY;
  const hh = String(Math.floor(m / 60) % 24).padStart(2, "0");
  const mm = String(Math.floor(m) % 60).padStart(2, "0");
  return `Day ${day} · ${hh}:${mm}`;
}

export default function MapView({ run, seed }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  // Animation state lives in refs so the rAF loop never re-subscribes.
  const timeRef = useRef(0);
  const playingRef = useRef(true);
  const speedRef = useRef(240); // simulated minutes per real second
  const lastRef = useRef(performance.now());

  // Streamed data for the current window+viewport.
  const tripsByAgent = useRef<Map<number, Trip[]>>(new Map());
  const staysByAgent = useRef<Map<number, Stay[]>>(new Map());
  const poisRef = useRef<Poi[]>([]);
  const selectedTripRef = useRef<TripGeometry | null>(null);

  // Single-in-flight data pump: `want` is the desired (day, viewport); `loadedKey`
  // is what's currently in the refs. The loop nudges `want`; the pump reconciles.
  const loadingRef = useRef(false);
  const wantRef = useRef<Want>({ day: 0, bbox: undefined, key: "" });
  const loadedKeyRef = useRef<string>("");

  // Which seed's trips are being shown, and that seed's time span. Held in refs so
  // the rAF loop and pump (defined once per map init) always read the live value,
  // and a separate [seed] effect can swap data without re-creating the map.
  const seedRef = useRef(seed);
  const shownSeedRef = useRef(seed); // last seed the [seed] effect acted on
  const timeSpanRef = useRef(run.time_spans?.[String(seed)] ?? run.num_days * DAY);
  const pumpRef = useRef<(() => void) | null>(null);

  const clockRef = useRef<HTMLSpanElement | null>(null);
  const scrubRef = useRef<HTMLInputElement | null>(null);

  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(240);
  const [loading, setLoading] = useState(false);
  const [inspected, setInspected] = useState<string | null>(null);

  const runId = run.run_id;
  const timeSpan = run.time_spans?.[String(seed)] ?? run.num_days * DAY;

  function currentBBox(): BBox | undefined {
    const map = mapRef.current;
    if (!map) return undefined;
    const b = map.getBounds();
    return [b.getSouth(), b.getWest(), b.getNorth(), b.getEast()];
  }

  function markersAt(t: number): Marker[] {
    const out: Marker[] = [];
    const tba = tripsByAgent.current;
    const sba = staysByAgent.current;
    for (const [agentId, trips] of tba) {
      let moving: Trip | null = null;
      for (const tr of trips) {
        if (t >= tr.timestamps[0] && t <= tr.timestamps[tr.timestamps.length - 1]) {
          moving = tr;
          break;
        }
      }
      if (moving) {
        out.push({
          position: interp(moving, t),
          color: [230, 80, 60],
          agentId,
          tripId: moving.trip_id,
          label: `Agent ${agentId} · in transit${moving.mode ? ` (${moving.mode})` : ""}`,
        });
      }
    }
    for (const [agentId, stays] of sba) {
      if (tba.has(agentId) && out.some((m) => m.agentId === agentId)) continue;
      for (const st of stays) {
        if (t >= st.t0 && t <= st.t1) {
          out.push({
            position: [st.lon, st.lat],
            color: st.color,
            agentId,
            label: `Agent ${agentId} · ${st.state}`,
          });
          break;
        }
      }
    }
    return out;
  }

  function buildLayers(t: number) {
    const sel = selectedTripRef.current;
    const layers: any[] = [
      new ScatterplotLayer({
        id: "pois",
        data: poisRef.current,
        getPosition: (d: Poi) => [d.lon, d.lat],
        getFillColor: [110, 110, 120],
        getRadius: 4,
        radiusMinPixels: 1,
        radiusMaxPixels: 2.5,
        opacity: 0.4,
        pickable: true,
        autoHighlight: true,
        highlightColor: [80, 140, 255, 200],
      }),
    ];
    if (sel) {
      layers.push(
        new TripsLayer({
          id: "inspected",
          data: [sel],
          getPath: (d: TripGeometry) => d.path,
          getTimestamps: (d: TripGeometry) => d.timestamps,
          getColor: (d: TripGeometry) => d.color,
          currentTime: t,
          trailLength: TRAIL,
          widthMinPixels: 4,
          capRounded: true,
          jointRounded: true,
          opacity: 0.9,
        })
      );
    }
    layers.push(
      new ScatterplotLayer({
        id: "agents",
        data: markersAt(t),
        getPosition: (d: Marker) => d.position,
        getFillColor: (d: Marker) => d.color,
        getRadius: 14,
        radiusMinPixels: 3,
        radiusMaxPixels: 6,
        opacity: 0.95,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 180],
        updateTriggers: { getPosition: t },
      })
    );
    return layers;
  }

  // ── Map init (once per run) ───────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;

    // Reset load state for this (re)mount so a stale in-flight flag from a prior
    // mount (React StrictMode / run change) can't wedge the pump.
    loadingRef.current = false;
    loadedKeyRef.current = "";
    wantRef.current = { day: 0, bbox: undefined, key: "" };
    seedRef.current = seed;
    shownSeedRef.current = seed;
    timeSpanRef.current = run.time_spans?.[String(seed)] ?? run.num_days * DAY;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: [run.view.longitude, run.view.latitude],
      zoom: 11.5,
      attributionControl: false,
    });
    mapRef.current = map;
    const overlay = new MapboxOverlay({
      interleaved: false,
      layers: [],
      getTooltip: ({ object }: any) =>
        object && (object.label || object.category)
          ? {
              html: `<b>${object.label ?? object.category}</b>`,
              style: {
                background: "#1f2430",
                color: "#fff",
                fontSize: "12px",
                padding: "5px 9px",
                borderRadius: "6px",
              },
            }
          : null,
    });
    overlayRef.current = overlay;

    // Fetch the wanted day-window + viewport. Exactly one runs at a time; if the
    // desired window changed while loading, it re-runs once on completion. This
    // replaces the old per-frame fetch (which dispatched ~60 self-superseding
    // requests/sec and could stall for minutes before any "won").
    async function pump() {
      if (loadingRef.current) return;
      const want = wantRef.current;
      if (want.key === loadedKeyRef.current) return;
      loadingRef.current = true;
      setLoading(true);
      const sd = seedRef.current;
      const t0 = want.day * DAY;
      const t1 = Math.min(timeSpanRef.current, t0 + DAY);
      try {
        const [timeline, pois] = await Promise.all([
          getTimeline(runId, t0, t1, want.bbox, sd),
          getPois(runId, want.bbox, sd),
        ]);
        const tba = new Map<number, Trip[]>();
        for (const tr of timeline.trips) {
          const arr = tba.get(tr.agent_id) ?? [];
          arr.push(tr);
          tba.set(tr.agent_id, arr);
        }
        const sba = new Map<number, Stay[]>();
        for (const st of timeline.stays) {
          const arr = sba.get(st.agent_id) ?? [];
          arr.push(st);
          sba.set(st.agent_id, arr);
        }
        tripsByAgent.current = tba;
        staysByAgent.current = sba;
        poisRef.current = pois;
        loadedKeyRef.current = want.key;
      } catch (err) {
        console.error("loadWindow failed", err);
      } finally {
        loadingRef.current = false;
        setLoading(false);
      }
      // Desired window moved while we were loading → reconcile once more.
      if (wantRef.current.key !== loadedKeyRef.current) void pump();
    }
    pumpRef.current = pump;

    let frame = 0;
    let lastClock = 0;
    const loop = (now: number) => {
      const dt = (now - lastRef.current) / 1000;
      lastRef.current = now;
      if (playingRef.current) {
        timeRef.current += speedRef.current * dt;
        if (timeRef.current > timeSpanRef.current) timeRef.current = 0;
      }
      const t = timeRef.current;
      // Want the day-window for the current clock position; pump reconciles.
      const dayIndex = Math.floor(t / DAY);
      if (dayIndex !== wantRef.current.day) {
        wantRef.current = { ...wantRef.current, day: dayIndex, key: keyOf(dayIndex, wantRef.current.bbox, seedRef.current) };
      }
      void pump();
      overlay.setProps({ layers: buildLayers(t) });
      if (scrubRef.current && document.activeElement !== scrubRef.current) {
        scrubRef.current.value = String(t);
      }
      // Update the clock via the DOM (not React state) so the toolbar doesn't
      // re-render ~10x/sec — that re-render was the speed-slider flicker.
      if (clockRef.current && now - lastClock > 100) {
        clockRef.current.textContent = fmtClock(t);
        lastClock = now;
      }
      frame = requestAnimationFrame(loop);
    };

    const onLoad = () => {
      map.addControl(overlay as unknown as maplibregl.IControl);
      const bbox = currentBBox();
      wantRef.current = { day: 0, bbox, key: keyOf(0, bbox, seedRef.current) };
      void pump();
      frame = requestAnimationFrame(loop);
    };
    map.on("load", onLoad);

    // Re-stream on pan/zoom (debounced) for viewport culling.
    let moveTimer: number | undefined;
    const onMoveEnd = () => {
      window.clearTimeout(moveTimer);
      moveTimer = window.setTimeout(() => {
        const bbox = currentBBox();
        wantRef.current = { ...wantRef.current, bbox, key: keyOf(wantRef.current.day, bbox, seedRef.current) };
        void pump();
      }, 250);
    };
    map.on("moveend", onMoveEnd);

    // Click an agent dot -> fetch its current trip's full geometry (on demand).
    overlay.setProps({
      onClick: ({ object }: any) => {
        if (object && object.tripId) {
          void getTripGeometry(runId, object.tripId, seedRef.current).then((g) => {
            selectedTripRef.current = g;
            setInspected(object.tripId);
          });
        } else {
          selectedTripRef.current = null;
          setInspected(null);
        }
      },
    });

    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(moveTimer);
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  // ── Seed switch (swap data only; the map itself stays put) ───────────────────
  useEffect(() => {
    if (shownSeedRef.current === seed) return; // initial render handled by map init
    shownSeedRef.current = seed;
    seedRef.current = seed;
    timeSpanRef.current = run.time_spans?.[String(seed)] ?? run.num_days * DAY;
    // Drop the old seed's geometry so its dots/paths clear immediately, then force
    // the single-flight pump to refetch this seed's window.
    tripsByAgent.current = new Map();
    staysByAgent.current = new Map();
    selectedTripRef.current = null;
    setInspected(null);
    if (timeRef.current > timeSpanRef.current) timeRef.current = 0;
    loadedKeyRef.current = "";
    wantRef.current = {
      ...wantRef.current,
      key: keyOf(wantRef.current.day, wantRef.current.bbox, seed),
    };
    pumpRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed]);

  // ── Controls ───────────────────────────────────────────────────────────────
  function togglePlay() {
    playingRef.current = !playingRef.current;
    setPlaying(playingRef.current);
  }
  function onScrub(e: React.ChangeEvent<HTMLInputElement>) {
    timeRef.current = parseFloat(e.target.value);
    playingRef.current = false;
    setPlaying(false);
  }
  function onSpeed(e: React.ChangeEvent<HTMLInputElement>) {
    const v = parseFloat(e.target.value);
    speedRef.current = v;
    setSpeed(v);
  }
  function clearInspect() {
    selectedTripRef.current = null;
    setInspected(null);
  }

  const secPerDay = DAY / speed;

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map" />
      <div className="map-ctl">
        <button onClick={togglePlay}>{playing ? "⏸ Pause" : "▶ Play"}</button>
        <span className="clock" ref={clockRef}>Day 1 · 00:00</span>
        <input
          ref={scrubRef}
          type="range"
          min={0}
          max={timeSpan}
          step={1}
          defaultValue={0}
          onChange={onScrub}
          className="scrub"
        />
        <span className="speed-label">⏩ {secPerDay < 1 ? secPerDay.toFixed(1) : Math.round(secPerDay)}s/day</span>
        <input type="range" min={30} max={1440} step={30} value={speed} onChange={onSpeed} className="speed" />
        {loading && <span className="loading-dot" title="streaming viewport data" />}
      </div>
      {inspected && (
        <div className="inspect-chip" onClick={clearInspect}>
          inspecting trip {inspected} · click to clear
        </div>
      )}
    </div>
  );
}
