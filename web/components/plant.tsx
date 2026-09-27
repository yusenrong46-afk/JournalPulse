export type PlantStage = "seed" | "sprout" | "leafy" | "flower";

const PETALS = ["#FF9A5A", "#B9A8F0", "#FFB4A2", "#F6C453"];

/** One plant per check-in: it grows with how much the small step helped. */
export function Plant({ stage, index = 0, size = 56, label }: { stage: PlantStage; index?: number; size?: number; label?: string }) {
  const petal = PETALS[index % PETALS.length];
  const tall = stage === "flower" ? 16 : stage === "leafy" ? 24 : stage === "sprout" ? 36 : 50;
  return (
    <svg
      className="plant"
      width={size}
      height={size * 1.25}
      viewBox="0 0 64 80"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <path d="M14 66 h36 l-4 12 h-28 z" fill="#E7A27A" />
      <rect x="12" y="62" width="40" height="6" rx="3" fill="#D98E66" />
      {stage === "seed" ? (
        <>
          <ellipse cx="32" cy="60" rx="6" ry="4" fill="#8A6A4A" />
          <path d="M32 58 C32 54 33 52 35 50" stroke="#6FA56F" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeDasharray="2 3" />
        </>
      ) : (
        <>
          <path d={`M32 62 C32 50 31 40 32 ${tall}`} stroke="#5E9A62" strokeWidth="3" fill="none" strokeLinecap="round" />
          <path d="M32 50 C24 44 17 46 16 51 C22 55 28 54 32 50 Z" fill="#9FC8A8" />
          {stage !== "sprout" && <path d="M32 42 C40 35 47 37 48 42 C42 47 36 46 32 42 Z" fill="#8DBF8B" />}
          {stage === "leafy" && <path d="M32 30 C27 24 22 25 21 29 C25 33 29 33 32 30 Z" fill="#9FC8A8" />}
          {stage === "flower" && (
            <g>
              {[0, 72, 144, 216, 288].map((angle) => (
                <ellipse key={angle} cx="32" cy="8" rx="5.5" ry="8" fill={petal} transform={`rotate(${angle} 32 16)`} />
              ))}
              <circle cx="32" cy="16" r="5" fill="#FFE08A" />
            </g>
          )}
        </>
      )}
    </svg>
  );
}

export function plantStage(helpfulness: number | null | undefined, completed: boolean | undefined): PlantStage {
  if (completed === undefined) return "seed";
  if (!completed || !helpfulness || helpfulness <= 2) return "sprout";
  if (helpfulness === 3) return "leafy";
  return "flower";
}
