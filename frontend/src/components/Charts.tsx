// Tiny dependency-free SVG charts (bar + multi-series line).

interface BarChartProps {
  data: Record<string, number>;
  height?: number;
  color?: string;
  format?: (v: number) => string;
}

export function BarChart({ data, height = 140, color = "#3b82f6", format }: BarChartProps) {
  const entries = Object.entries(data);
  if (entries.length === 0) return <div className="muted">No data</div>;
  const max = Math.max(...entries.map(([, v]) => v), 1e-9);
  const fmt = format ?? ((v: number) => (v >= 1 ? String(Math.round(v)) : v.toFixed(2)));
  return (
    <div className="barchart" style={{ height }}>
      {entries.map(([k, v]) => (
        <div className="bar-row" key={k} title={`${k}: ${fmt(v)}`}>
          <span className="bar-label">{k}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{ width: `${(v / max) * 100}%`, background: color }}
            />
          </span>
          <span className="bar-value">{fmt(v)}</span>
        </div>
      ))}
    </div>
  );
}

interface LineChartProps {
  rows: Array<Record<string, number>>;
  xKey: string;
  series: { key: string; label: string; color: string }[];
  height?: number;
}

export function LineChart({ rows, xKey, series, height = 200 }: LineChartProps) {
  if (rows.length === 0) return <div className="muted">No data</div>;
  const W = 480;
  const H = height;
  const padL = 36;
  const padB = 22;
  const padT = 10;
  const padR = 10;
  const xs = rows.map((r) => r[xKey]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const allY = series.flatMap((s) => rows.map((r) => r[s.key]).filter((v) => v != null));
  const yMin = Math.min(...allY, 0);
  const yMax = Math.max(...allY, 1e-9);
  const sx = (x: number) =>
    padL + ((x - xMin) / (xMax - xMin || 1)) * (W - padL - padR);
  const sy = (y: number) =>
    H - padB - ((y - yMin) / (yMax - yMin || 1)) * (H - padT - padB);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="linechart" preserveAspectRatio="xMidYMid meet">
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="#d1d5db" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="#d1d5db" />
        <text x={padL - 4} y={sy(yMax)} className="axis" textAnchor="end">
          {yMax.toFixed(2)}
        </text>
        <text x={padL - 4} y={sy(yMin)} className="axis" textAnchor="end">
          {yMin.toFixed(2)}
        </text>
        {series.map((s) => {
          const pts = rows
            .filter((r) => r[s.key] != null)
            .map((r) => `${sx(r[xKey])},${sy(r[s.key])}`)
            .join(" ");
          return <polyline key={s.key} points={pts} fill="none" stroke={s.color} strokeWidth={2} />;
        })}
        {xs.map((x) => (
          <text key={x} x={sx(x)} y={H - 6} className="axis" textAnchor="middle">
            {x}
          </text>
        ))}
      </svg>
      <div className="legend-row">
        {series.map((s) => (
          <span key={s.key} className="legend-item">
            <span className="legend-swatch" style={{ background: s.color }} /> {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
