import type { AffectiveState } from "@/lib/types";

const dimensions = [
  { key: "valence", label: "Valence", low: "unpleasant", high: "pleasant", min: -1, max: 1 },
  { key: "arousal", label: "Activation", low: "quiet", high: "charged", min: 0, max: 1 },
  { key: "agency", label: "Agency", low: "stuck", high: "capable", min: 0, max: 1 },
] as const;

export function StateControls({ state, onChange }: { state: AffectiveState; onChange: (state: AffectiveState) => void }) {
  return (
    <div className="state-controls">
      {dimensions.map((dimension) => (
        <label className="dimension" key={dimension.key}>
          <span className="dimension-head">
            <strong>{dimension.label}</strong>
            <output>{state[dimension.key].toFixed(2)}</output>
          </span>
          <input
            type="range"
            min={dimension.min}
            max={dimension.max}
            step="0.05"
            value={state[dimension.key]}
            onChange={(event) => onChange({ ...state, [dimension.key]: Number(event.target.value) })}
          />
          <span className="dimension-axis">
            <small>{dimension.low}</small>
            <small>{dimension.high}</small>
          </span>
        </label>
      ))}
    </div>
  );
}
