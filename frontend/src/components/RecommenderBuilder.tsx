import { useState } from "react";
import type { Preset, RecommenderSpec, WeightKey } from "../types";
import { WEIGHT_KEYS } from "../types";

/**
 * The "＋ Add recommender" panel.
 *
 * Every recommender in a study is the *same* scorer with different knob values;
 * a preset seeds the sliders and the user takes it from there. Weights are shown
 * as the percentages they normalize to rather than as their raw values, because
 * the raw numbers are meaningless on their own — the engine renormalizes them —
 * and a slider that reads "0.35" next to five others is impossible to reason
 * about, while "34% of the ranking" is not.
 */

const WEIGHT_LABELS: Record<WeightKey, { name: string; help: string }> = {
  w_rating: {
    name: "Rating",
    help: "The POI's star rating. Rises and falls as agents like or dislike it, so it is a slow feedback channel.",
  },
  w_reviews: {
    name: "Reviews",
    help: "How many reviews the POI has, on a log scale. A proxy for established reputation.",
  },
  w_relevance: {
    name: "Relevance",
    help: "Category fit — is this a restaurant when the agent wants dinner. Near-identical for every candidate of a subtype, so it barely changes the ranking. It is not personal: agents do not tell the platform what they want.",
  },
  w_proximity: {
    name: "Proximity",
    help: "Straight-line closeness to where the agent is, decaying over the distance scale below.",
  },
  w_popularity: {
    name: "Popularity",
    help: "Live footfall relative to the busiest candidate. This is the feedback loop: visits make a place more recommendable, which brings more visits.",
  },
  w_personalization: {
    name: "Personalization",
    help: "What the platform has learned this agent likes, from their past feedback. The only channel carrying anything agent-specific — and the only way it can ever discover an agent's latent taste.",
  },
};

const GATES: { value: string; label: string }[] = [
  { value: "", label: "None" },
  { value: "pup", label: "PUP — drop low P[U≥0]" },
  { value: "rm", label: "RM — drop high regret" },
  { value: "pup_rm", label: "PUP + RM" },
];

export const BLANK_SPEC: RecommenderSpec = {
  label: "Custom RS",
  w_rating: 0.2,
  w_reviews: 0.15,
  w_relevance: 0.1,
  w_proximity: 0.25,
  w_popularity: 0.2,
  w_personalization: 0.1,
  distance_scale_km: 5,
  popularity_gamma: 1,
  learning_rate: 0.12,
  welfare_gate: "",
  pup_alpha: 0.6,
  rm_epsilon: 0.3,
  random_ranking: false,
};

/** Weight shares as the engine will compute them, so the UI cannot disagree. */
export function normalizedWeights(spec: RecommenderSpec): Record<WeightKey, number> {
  const total = WEIGHT_KEYS.reduce((sum, k) => sum + Math.max(0, spec[k]), 0);
  const out = {} as Record<WeightKey, number>;
  for (const k of WEIGHT_KEYS) {
    out[k] = total > 0 ? Math.max(0, spec[k]) / total : 1 / WEIGHT_KEYS.length;
  }
  return out;
}

/** Compact one-line summary for a collapsed card. */
export function specSummary(spec: RecommenderSpec): string {
  if (spec.random_ranking) return "uniform random · control";
  const w = normalizedWeights(spec);
  const top = WEIGHT_KEYS.map((k) => ({ k, v: w[k] }))
    .sort((a, b) => b.v - a.v)
    .filter((x) => x.v > 0.01)
    .slice(0, 3)
    .map((x) => `${WEIGHT_LABELS[x.k].name} ${Math.round(x.v * 100)}%`);
  const extra: string[] = [];
  if (spec.w_popularity > 0 && spec.popularity_gamma !== 1) {
    extra.push(`γ ${spec.popularity_gamma.toFixed(1)}`);
  }
  if (spec.welfare_gate) extra.push(spec.welfare_gate.toUpperCase().replace("_", "+"));
  return [...top, ...extra].join(" · ");
}

interface Props {
  specs: RecommenderSpec[];
  presets: Preset[];
  max: number;
  control: string;
  disabled: boolean;
  onChange: (specs: RecommenderSpec[]) => void;
}

export default function RecommenderBuilder({
  specs,
  presets,
  max,
  control,
  disabled,
  onChange,
}: Props) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  // Controlled rather than a bare <details>, so picking a preset closes the
  // menu and reveals the card it just created instead of burying it.
  const [pickerOpen, setPickerOpen] = useState(false);

  function addFrom(preset: Preset) {
    if (specs.length >= max) return;
    const { name, ...values } = preset;
    // Labels are the identity of a study arm everywhere downstream (results
    // rows, the map picker, the run id), so a duplicate has to be made distinct
    // here rather than silently collapsing into the existing one.
    let label = name;
    for (let n = 2; specs.some((s) => s.label === label); n += 1) label = `${name} ${n}`;
    onChange([...specs, { ...BLANK_SPEC, ...values, label }]);
    setOpenIndex(specs.length);
    setPickerOpen(false);
  }

  function patch(index: number, change: Partial<RecommenderSpec>) {
    onChange(specs.map((s, i) => (i === index ? { ...s, ...change } : s)));
  }

  function remove(index: number) {
    onChange(specs.filter((_, i) => i !== index));
    setOpenIndex(null);
  }

  return (
    <div className="builder">
      <div className="rec-card control" title="The baseline every comparison is paired against. Always run.">
        <div className="rec-card-head">
          <span className="rec-name">{control}</span>
          <span className="muted small">control · always run</span>
        </div>
      </div>

      {specs.map((spec, index) => (
        <div className={`rec-card${openIndex === index ? " open" : ""}`} key={index}>
          <div className="rec-card-head">
            <button
              className="rec-toggle"
              onClick={() => setOpenIndex(openIndex === index ? null : index)}
              aria-expanded={openIndex === index}
            >
              <span className="chev">{openIndex === index ? "▾" : "▸"}</span>
              <span className="rec-name">{spec.label}</span>
            </button>
            <button
              className="rec-remove"
              onClick={() => remove(index)}
              disabled={disabled}
              title="Remove this recommender"
            >
              ✕
            </button>
          </div>
          <div className="rec-chips">{specSummary(spec)}</div>

          {openIndex === index && (
            <SpecEditor spec={spec} onPatch={(c) => patch(index, c)} disabled={disabled} />
          )}
        </div>
      ))}

      {specs.length < max ? (
        <details
          className="add-rec"
          open={pickerOpen}
          onToggle={(e) => setPickerOpen((e.target as HTMLDetailsElement).open)}
        >
          <summary>＋ Add recommender</summary>
          <div className="preset-list">
            {presets.map((p) => (
              <button key={p.name} className="preset-btn" onClick={() => addFrom(p)}>
                <b>{p.name}</b>
                <span className="muted small">
                  {specSummary({ ...BLANK_SPEC, ...p } as RecommenderSpec)}
                </span>
              </button>
            ))}
          </div>
        </details>
      ) : (
        <p className="muted small">
          Maximum of {max} recommenders. Remove one to add another.
        </p>
      )}
    </div>
  );
}

function SpecEditor({
  spec,
  onPatch,
  disabled,
}: {
  spec: RecommenderSpec;
  onPatch: (c: Partial<RecommenderSpec>) => void;
  disabled: boolean;
}) {
  const shares = normalizedWeights(spec);

  return (
    <div className="rec-body">
      <label className="rec-field">
        Name
        <input
          type="text"
          value={spec.label}
          maxLength={40}
          disabled={disabled}
          onChange={(e) => onPatch({ label: e.target.value })}
        />
      </label>

      <label className="toggle" title="Ignores every weight below and ranks candidates uniformly at random. A control for 'does the ranking matter at all'.">
        <input
          type="checkbox"
          checked={spec.random_ranking}
          disabled={disabled}
          onChange={(e) => onPatch({ random_ranking: e.target.checked })}
        />
        Random ranking (control)
      </label>

      {!spec.random_ranking && (
        <>
          <h5>
            Ranking weights
            <span className="muted small"> — shown as the share each gets</span>
          </h5>
          {WEIGHT_KEYS.map((key) => (
            <label className="rec-weight" key={key} title={WEIGHT_LABELS[key].help}>
              <span className="rec-weight-head">
                <span>{WEIGHT_LABELS[key].name}</span>
                <b>{Math.round(shares[key] * 100)}%</b>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={spec[key]}
                disabled={disabled}
                onChange={(e) => onPatch({ [key]: parseFloat(e.target.value) } as Partial<RecommenderSpec>)}
              />
            </label>
          ))}

          <label className="rec-field" title="Proximity decays as exp(-distance / scale). Smaller keeps agents local; larger makes the recommender willing to send them across the metro.">
            Distance scale: <b>{spec.distance_scale_km.toFixed(1)} km</b>
            <input
              type="range"
              min={1}
              max={20}
              step={0.5}
              value={spec.distance_scale_km}
              disabled={disabled}
              onChange={(e) => onPatch({ distance_scale_km: parseFloat(e.target.value) })}
            />
          </label>

          {spec.w_popularity > 0 && (
            <label className="rec-field" title="Exponent on relative footfall. Below 1 flattens the advantage of already-busy places; above 1 concentrates demand on whatever is already winning. This is the dial on the rich-get-richer loop.">
              Popularity damping γ: <b>{spec.popularity_gamma.toFixed(1)}</b>
              <input
                type="range"
                min={0}
                max={3}
                step={0.1}
                value={spec.popularity_gamma}
                disabled={disabled}
                onChange={(e) => onPatch({ popularity_gamma: parseFloat(e.target.value) })}
              />
              <span className="muted small">
                {spec.popularity_gamma < 0.9
                  ? "flattens — the long tail keeps a chance"
                  : spec.popularity_gamma > 1.1
                  ? "concentrates — winners take more"
                  : "linear in relative footfall"}
              </span>
            </label>
          )}

          {spec.w_personalization > 0 && (
            <label className="rec-field" title="How fast a like or dislike moves the platform's model of this agent. Feedback only arrives on accepted recommendations, so learning is slow — a longer study shows more of it.">
              Learning rate: <b>{spec.learning_rate.toFixed(2)}</b>
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.01}
                value={spec.learning_rate}
                disabled={disabled}
                onChange={(e) => onPatch({ learning_rate: parseFloat(e.target.value) })}
              />
            </label>
          )}
        </>
      )}

      <label className="rec-field" title="An optional welfare filter applied after ranking: it withholds recommendations predicted to leave the agent worse off, rather than re-ranking them.">
        Welfare gate
        <select
          value={spec.welfare_gate}
          disabled={disabled}
          onChange={(e) => onPatch({ welfare_gate: e.target.value })}
        >
          {GATES.map((g) => (
            <option key={g.value} value={g.value}>
              {g.label}
            </option>
          ))}
        </select>
      </label>

      {(spec.welfare_gate === "pup" || spec.welfare_gate === "pup_rm") && (
        <label className="rec-field">
          PUP α (min P[U≥0]): <b>{spec.pup_alpha.toFixed(2)}</b>
          <input
            type="range" min={0} max={1} step={0.05}
            value={spec.pup_alpha}
            disabled={disabled}
            onChange={(e) => onPatch({ pup_alpha: parseFloat(e.target.value) })}
          />
        </label>
      )}
      {(spec.welfare_gate === "rm" || spec.welfare_gate === "pup_rm") && (
        <label className="rec-field">
          RM ε (regret ceiling): <b>{spec.rm_epsilon.toFixed(2)}</b>
          <input
            type="range" min={0} max={1.5} step={0.05}
            value={spec.rm_epsilon}
            disabled={disabled}
            onChange={(e) => onPatch({ rm_epsilon: parseFloat(e.target.value) })}
          />
        </label>
      )}
    </div>
  );
}
