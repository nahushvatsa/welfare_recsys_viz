/**
 * Shared chart primitives for the results dashboard.
 *
 * One palette, one set of axis/grid/tooltip conventions, so every panel reads as
 * part of the same system. The control is always drawn in neutral grey and
 * always sits first, because every panel is a comparison against it and the eye
 * needs a fixed reference.
 */
import { createContext, useContext, type ReactNode } from "react";
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

/**
 * Display names for conditions.
 *
 * Presentation only. The raw condition string stays the key everywhere it
 * matters — it indexes `metrics`, `colors`, `welfare_map.delta` and the API's
 * `?condition=` — so this must never be used to look anything up. Matched
 * case-insensitively so a study labelled "footfall-led" reads the same as
 * "Footfall-led".
 */
const CONDITION_LABEL: Record<string, string> = {
  "footfall-led": "Popularity RS",
  "proximity-led": "Proximity RS",
  personalized: "Personalized RS",
};

export function condLabel(condition: string): string {
  return CONDITION_LABEL[condition.toLowerCase()] ?? condition;
}

/**
 * Figure mode: render panels at the proportions of a paper figure.
 *
 * A screenshot is scaled to the column, so what matters is the ratio of text to
 * plot area, not either one's pixel size. At the dashboard's ~700px panel width
 * the 14px chart text lands at ~4.8pt once it is squeezed into an ACM column —
 * illegible. Fixing the panel at 470px (see .figure-mode in styles.css) puts the
 * same text at ~7pt. The chart also loses a little height here so the captured
 * block is nearer the 3:2 an ACM column figure usually wants.
 */
export const FigureMode = createContext(false);

function useChartHeight(height: number): number {
  return useContext(FigureMode) ? Math.round(height * 0.78) : height;
}

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

const AXIS = { stroke: "var(--chart-axis)", fontSize: 13 };
const GRID = { stroke: "var(--chart-grid)", strokeDasharray: "3 3" };

/**
 * Axis titles and legend, sized to be readable in a projected slide rather than
 * only on a laptop. The panels that carry these charts no longer print a
 * description under the heading, which is where the room came from.
 *
 * `textAnchor` is not cosmetic. Recharts derives it from `position`, and
 * `insideLeft` yields "start" — so a title rotated -90° starts at the axis
 * midpoint and grows *upward*, taking its full length out of the top half
 * alone. At figure-mode heights that half is ~64px against a ~105px title, and
 * the top of the word is clipped. "middle" centres the rotated text on the
 * midpoint instead, spending half its length each way. Harmless on the X titles
 * below, whose position already anchors them "middle".
 */
const AXIS_TITLE = { fontSize: 14, fill: "var(--chart-axis)", textAnchor: "middle" as const };
const LEGEND_STYLE = { fontSize: 14, paddingTop: 6 };
/** X axes that carry a title need room for ticks + title; 30 (the Recharts
 *  default) clips the title once the fonts go up. */
const X_AXIS_HEIGHT = 52;
/**
 * How far the X title sits above the bottom of that 52px band.
 *
 * `insideBottom` measures up from the band's lower edge (y = bottom - offset),
 * so bigger means higher. The ticks end ~22px down and the legend opens 6px
 * below the band, which leaves the title ~10px of air above and ~12px below —
 * centred in the gap rather than sitting on the legend, which is where the old
 * -2 (tuned for the default 30px band) had pushed it.
 */
const X_TITLE_OFFSET = 6;

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
      fontSize: 13.5,
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
  const h = useChartHeight(height);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="day" {...AXIS} tickLine={false} height={X_AXIS_HEIGHT}
               label={{ value: "simulated day", position: "insideBottom", offset: X_TITLE_OFFSET, ...AXIS_TITLE }} />
        <YAxis {...AXIS} tickLine={false} width={62}
               label={{ value: yLabel, angle: -90, position: "insideLeft", ...AXIS_TITLE }} />
        <Tooltip {...tooltipStyle()} formatter={fmt(format)} />
        <Legend wrapperStyle={LEGEND_STYLE} />
        {series.map((s) => (
          <Line key={s} type="monotone" dataKey={s} name={condLabel(s)} stroke={colors[s]}
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
  const h = useChartHeight(height);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey={categoryKey} {...AXIS} tickLine={false} height={34} />
        <YAxis {...AXIS} tickLine={false} width={66}
               label={{ value: yLabel, angle: -90, position: "insideLeft", ...AXIS_TITLE }} />
        <Tooltip {...tooltipStyle()} cursor={{ fill: "var(--chart-grid)", opacity: 0.25 }}
                 formatter={fmt(format)} />
        <Legend wrapperStyle={LEGEND_STYLE} />
        {zeroLine && <ReferenceLine y={0} stroke="var(--chart-axis)" strokeWidth={1} />}
        {series.map((s) => (
          <Bar key={s} dataKey={s} name={condLabel(s)} fill={colors[s]} isAnimationActive={false} radius={[2, 2, 0, 0]} />
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
  const h = useChartHeight(height);
  const names = Object.keys(curves);
  const grid = curves[names[0]] ?? [];
  const data = grid.map((point, i) => {
    const row: Record<string, number> = { x: point[0], equality: point[0] };
    for (const name of names) row[name] = curves[name][i]?.[1] ?? 0;
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={h}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="x" type="number" domain={[0, 1]} {...AXIS} tickLine={false}
               height={X_AXIS_HEIGHT}
               tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
               label={{ value: "POIs, least visited first", position: "insideBottom", offset: X_TITLE_OFFSET, ...AXIS_TITLE }} />
        <YAxis domain={[0, 1]} {...AXIS} tickLine={false} width={62}
               tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
               label={{ value: "share of visits", angle: -90, position: "insideLeft", ...AXIS_TITLE }} />
        <Tooltip {...tooltipStyle()}
                 formatter={fmt((v) => `${(v * 100).toFixed(1)}%`)}
                 labelFormatter={(v) => `${(Number(v) * 100).toFixed(0)}% of POIs`} />
        <Legend wrapperStyle={LEGEND_STYLE} />
        <Line dataKey="equality" stroke="var(--chart-axis)" strokeDasharray="4 4"
              strokeWidth={1} dot={false} isAnimationActive={false} name="perfect equality" />
        {names.map((n) => (
          <Line key={n} dataKey={n} name={condLabel(n)} stroke={colors[n]} strokeWidth={2} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
