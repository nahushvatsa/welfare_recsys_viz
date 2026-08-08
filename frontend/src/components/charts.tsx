/**
 * Shared chart primitives for the results dashboard.
 *
 * One palette, one set of axis/grid/tooltip conventions, so every panel reads as
 * part of the same system. The control is always drawn in neutral grey and
 * always sits first, because every panel is a comparison against it and the eye
 * needs a fixed reference.
 */
import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/** Categorical series colours. Distinguishable in both themes and for the
 *  common forms of colour-blindness (blue/orange/green/purple/red ramp). */
export const SERIES_COLORS = [
  "#4c78d8",
  "#f58518",
  "#54a24b",
  "#b279a2",
  "#e45756",
  "#72b7b2",
];
export const CONTROL_COLOR = "#8a8a94";

/** Stable colour per condition: the control is grey, the rest cycle. */
export function conditionColors(
  conditions: string[],
  control: string
): Record<string, string> {
  const out: Record<string, string> = {};
  let i = 0;
  for (const c of conditions) {
    if (c === control) out[c] = CONTROL_COLOR;
    else out[c] = SERIES_COLORS[i++ % SERIES_COLORS.length];
  }
  return out;
}

/** Recharts types its tooltip formatter over `ValueType | undefined`; this
 *  narrows to the numeric case once, instead of at every call site. */
function fmt(format?: (v: number) => string) {
  return (value: unknown): string => {
    const n = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(n)) return "—";
    return format ? format(n) : n.toFixed(3);
  };
}

const AXIS = { stroke: "var(--chart-axis)", fontSize: 11 };
const GRID = { stroke: "var(--chart-grid)", strokeDasharray: "3 3" };

export function Panel({
  title,
  hint,
  children,
  wide,
}: {
  title: string;
  hint?: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <section className={`panel${wide ? " wide" : ""}`}>
      <h4>{title}</h4>
      {hint && <p className="panel-hint">{hint}</p>}
      {children}
    </section>
  );
}

function tooltipStyle() {
  return {
    contentStyle: {
      background: "var(--chart-tooltip-bg)",
      border: "1px solid var(--chart-grid)",
      borderRadius: 6,
      fontSize: 12,
    },
    labelStyle: { color: "var(--fg)" },
  };
}

/** Multi-series line over simulated days. */
export function DayLines({
  data,
  series,
  colors,
  yLabel,
  height = 240,
  format,
}: {
  data: Record<string, number>[];
  series: string[];
  colors: Record<string, string>;
  yLabel: string;
  height?: number;
  format?: (v: number) => string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="day" {...AXIS} tickLine={false}
               label={{ value: "simulated day", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--chart-axis)" }} />
        <YAxis {...AXIS} tickLine={false} width={48}
               label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--chart-axis)" }} />
        <Tooltip {...tooltipStyle()} formatter={fmt(format)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((s) => (
          <Line key={s} type="monotone" dataKey={s} stroke={colors[s]}
                strokeWidth={2} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Grouped bars over a categorical axis, with an optional control reference. */
export function GroupedBars({
  data,
  categoryKey,
  series,
  colors,
  yLabel,
  height = 260,
  format,
  zeroLine,
}: {
  data: Record<string, string | number>[];
  categoryKey: string;
  series: string[];
  colors: Record<string, string>;
  yLabel: string;
  height?: number;
  format?: (v: number) => string;
  zeroLine?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey={categoryKey} {...AXIS} tickLine={false} />
        <YAxis {...AXIS} tickLine={false} width={52}
               label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--chart-axis)" }} />
        <Tooltip {...tooltipStyle()} cursor={{ fill: "var(--chart-grid)", opacity: 0.25 }}
                 formatter={fmt(format)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {zeroLine && <ReferenceLine y={0} stroke="var(--chart-axis)" strokeWidth={1} />}
        {series.map((s) => (
          <Bar key={s} dataKey={s} fill={colors[s]} isAnimationActive={false} radius={[2, 2, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/**
 * Diverging heatmap as a CSS grid.
 *
 * Deliberately not a chart library shape — Recharts has no heatmap, and a grid
 * of coloured cells with the number printed in each is both easier to read and
 * fully accessible to a screen reader.
 */
export function DivergingHeatmap({
  rows,
  cols,
  value,
  format,
  rowLabel,
  maxAbs,
}: {
  rows: string[];
  cols: string[];
  value: (row: string, col: string) => number | null;
  format: (v: number) => string;
  rowLabel: string;
  maxAbs?: number;
}) {
  const values = rows.flatMap((r) => cols.map((c) => value(r, c))).filter((v): v is number => v != null);
  const scale = maxAbs ?? Math.max(0.0001, ...values.map(Math.abs));

  function cellStyle(v: number | null) {
    if (v == null) return { background: "transparent", color: "var(--muted)" };
    const t = Math.min(1, Math.abs(v) / scale);
    // Blue for "more than the control", orange for "less" — the same two hues
    // the series palette opens with, so the dashboard keeps one colour language.
    const hue = v >= 0 ? "76, 120, 216" : "245, 133, 24";
    return {
      background: `rgba(${hue}, ${(0.12 + 0.68 * t).toFixed(3)})`,
      color: t > 0.55 ? "#fff" : "var(--fg)",
    };
  }

  return (
    <div className="heatmap-scroll">
      <div className="heatmap" style={{ gridTemplateColumns: `minmax(84px, auto) repeat(${cols.length}, minmax(58px, 1fr))` }}>
        <div className="hm-corner">{rowLabel}</div>
        {cols.map((c) => (
          <div className="hm-colhead" key={c}>{c}</div>
        ))}
        {rows.map((r) => (
          <FragmentRow key={r} label={r} cols={cols} value={value} format={format} cellStyle={cellStyle} />
        ))}
      </div>
    </div>
  );
}

function FragmentRow({
  label,
  cols,
  value,
  format,
  cellStyle,
}: {
  label: string;
  cols: string[];
  value: (row: string, col: string) => number | null;
  format: (v: number) => string;
  cellStyle: (v: number | null) => Record<string, string>;
}) {
  return (
    <>
      <div className="hm-rowhead">{label}</div>
      {cols.map((c) => {
        const v = value(label, c);
        return (
          <div className="hm-cell" key={c} style={cellStyle(v)}
               title={`${label} · ${c}: ${v == null ? "no data" : format(v)}`}>
            {v == null ? "—" : format(v)}
          </div>
        );
      })}
    </>
  );
}

/** Lorenz curves, with the equality diagonal as the reference. */
export function LorenzChart({
  curves,
  colors,
  height = 260,
}: {
  curves: Record<string, [number, number][]>;
  colors: Record<string, string>;
  height?: number;
}) {
  const names = Object.keys(curves);
  const grid = curves[names[0]] ?? [];
  const data = grid.map((point, i) => {
    const row: Record<string, number> = { x: point[0], equality: point[0] };
    for (const name of names) row[name] = curves[name][i]?.[1] ?? 0;
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="x" type="number" domain={[0, 1]} {...AXIS} tickLine={false}
               tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
               label={{ value: "POIs, least visited first", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--chart-axis)" }} />
        <YAxis domain={[0, 1]} {...AXIS} tickLine={false} width={48}
               tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
               label={{ value: "share of visits", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--chart-axis)" }} />
        <Tooltip {...tooltipStyle()}
                 formatter={fmt((v) => `${(v * 100).toFixed(1)}%`)}
                 labelFormatter={(v) => `${(Number(v) * 100).toFixed(0)}% of POIs`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="equality" stroke="var(--chart-axis)" strokeDasharray="4 4"
              strokeWidth={1} dot={false} isAnimationActive={false} name="perfect equality" />
        {names.map((n) => (
          <Line key={n} dataKey={n} stroke={colors[n]} strokeWidth={2} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
