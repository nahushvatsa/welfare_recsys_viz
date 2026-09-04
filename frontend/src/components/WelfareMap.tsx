import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { HexagonLayer } from "@deck.gl/aggregation-layers";
import { condLabel } from "./charts";
import type { RunMeta, WelfareMapRow } from "../types";

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

/**
 * Where a recommender's winners and losers live.
 *
 * Each agent contributes its home point and the change in its mean leisure net
 * utility versus the No-RS control. Hexagons average those deltas, so a cell
 * answers "for people who live here, did this recommender help or hurt".
 *
 * Averaged, never summed: a hexagon over a dense residential block holds far
 * more agents than one in the outer counties, and a sum would render population
 * density as welfare impact.
 */

// Diverging orange→blue. Mid is near-transparent so unaffected areas recede and
// the eye goes to the extremes, which is the whole point of the panel.
const COLOR_RANGE: [number, number, number][] = [
  [180, 78, 12],
  [214, 130, 62],
  [235, 190, 150],
  [170, 195, 230],
  [86, 138, 210],
  [28, 82, 172],
];

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export default function WelfareMap({
  run,
  conditions,
}: {
  run: RunMeta;
  conditions: string[];
}) {
  const [condition, setCondition] = useState(conditions[0] ?? "");
  const [radius, setRadius] = useState(600);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  const rows: WelfareMapRow[] = run.welfare_map ?? [];
  const points = useMemo(
    () => rows.filter((r) => r.delta[condition] != null),
    [rows, condition]
  );

  // Symmetric colour domain from the 5th/95th percentile of |delta|, so a
  // single outlier cannot flatten the whole map to one shade, and zero always
  // sits at the palette's midpoint (otherwise "blue" would not mean "better").
  const domain = useMemo<[number, number]>(() => {
    const values = points.map((p) => Math.abs(p.delta[condition])).sort((a, b) => a - b);
    const extent = Math.max(1e-4, quantile(values, 0.95));
    return [-extent, extent];
  }, [points, condition]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: [run.view.longitude, run.view.latitude],
      zoom: 8.6,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
    map.addControl(overlay as unknown as maplibregl.IControl);
    mapRef.current = map;
    overlayRef.current = overlay;
    return () => {
      overlayRef.current = null;
      mapRef.current = null;
      map.remove();
    };
  }, [run.view.latitude, run.view.longitude]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.setProps({
      layers: [
        new HexagonLayer<WelfareMapRow>({
          id: `welfare-${condition}-${radius}`,
          data: points,
          getPosition: (d) => [d.lon, d.lat],
          getColorWeight: (d) => d.delta[condition],
          colorAggregation: "MEAN",
          radius,
          coverage: 0.92,
          extruded: false,
          opacity: 0.75,
          colorRange: COLOR_RANGE,
          colorDomain: domain,
          pickable: true,
        }),
      ],
      getTooltip: ({ object }: { object?: { points?: unknown[]; colorValue?: number } }) => {
        if (!object) return null;
        const n = object.points?.length ?? 0;
        const mean = object.colorValue ?? 0;
        return {
          html: `<b>${mean >= 0 ? "+" : ""}${mean.toFixed(3)}</b> mean ΔU<br/>${n} agent${n === 1 ? "" : "s"} living here`,
          style: { fontSize: "12px" },
        };
      },
    });
  }, [points, condition, radius, domain]);

  if (!rows.length) {
    return (
      <p className="muted small">
        No paired agents to map — this needs agents who took a leisure trip both
        with and without a recommender.
      </p>
    );
  }

  return (
    <div className="welfare-map">
      <div className="welfare-map-controls">
        {conditions.length > 1 && (
          <label className="inline-picker">
            Condition
            <select value={condition} onChange={(e) => setCondition(e.target.value)}>
              {conditions.map((c) => (
                <option key={c} value={c}>{condLabel(c)}</option>
              ))}
            </select>
          </label>
        )}
        <label className="inline-picker">
          Hex size
          <select value={radius} onChange={(e) => setRadius(parseInt(e.target.value, 10))}>
            <option value={300}>300 m</option>
            <option value={600}>600 m</option>
            <option value={1200}>1.2 km</option>
            <option value={2500}>2.5 km</option>
          </select>
        </label>
        <span className="muted small">{points.length.toLocaleString()} agents mapped</span>
      </div>
      <div ref={containerRef} className="welfare-map-canvas" />
      <div className="welfare-legend">
        <span className="muted small">worse off</span>
        <span className="legend-ramp" />
        <span className="muted small">better off</span>
        <span className="muted small legend-range">
          ±{Math.abs(domain[1]).toFixed(2)} utility
        </span>
      </div>
    </div>
  );
}
