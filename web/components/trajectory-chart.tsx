import type { Insights } from "@/lib/types";

type Point = Insights["state_trajectory"][number];

const dimensions = [
  { key: "valence", label: "Valence", color: "#a7462d", min: -1, max: 1 },
  { key: "arousal", label: "Activation", color: "#c68a31", min: 0, max: 1 },
  { key: "agency", label: "Agency", color: "#1d6b68", min: 0, max: 1 },
] as const;

function line(points: Point[], key: "valence" | "arousal" | "agency", min: number, max: number) {
  return points
    .map((point, index) => {
      const x = points.length === 1 ? 50 : (index / (points.length - 1)) * 100;
      const normalized = (point[key] - min) / (max - min);
      return `${x},${92 - normalized * 84}`;
    })
    .join(" ");
}

export function TrajectoryChart({ points }: { points: Point[] }) {
  if (points.length < 2) {
    return <p className="chart-empty">Two or more reflections are needed to draw a trajectory.</p>;
  }
  return (
    <figure className="trajectory-figure">
      <svg viewBox="0 0 100 100" role="img" aria-label="Valence, activation, and agency over time" preserveAspectRatio="none">
        {[8, 29, 50, 71, 92].map((y) => <line key={y} x1="0" y1={y} x2="100" y2={y} className="chart-rule" />)}
        {dimensions.map((dimension) => (
          <polyline
            key={dimension.key}
            points={line(points, dimension.key, dimension.min, dimension.max)}
            fill="none"
            stroke={dimension.color}
            strokeWidth="1.8"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      <figcaption>
        {dimensions.map((dimension) => <span key={dimension.key}><i style={{ background: dimension.color }} />{dimension.label}</span>)}
      </figcaption>
    </figure>
  );
}
