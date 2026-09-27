import { useId } from "react";

export type LunaMood =
  | "idle"
  | "listening"
  | "thinking"
  | "answering"
  | "proud"
  | "oops"
  | "sleepy"
  | "checkin"
  | "support";

const LABELS: Record<LunaMood, string> = {
  idle: "Luna is here",
  listening: "Luna is listening",
  thinking: "Luna is thinking",
  answering: "Luna is answering",
  proud: "Luna is proud of you",
  oops: "Luna hit a snag",
  sleepy: "Luna is sleepy",
  checkin: "Luna is checking in",
  support: "Luna is here with you",
};

const INK = "#2B2540";

type LunaProps = {
  mood?: LunaMood;
  size?: number;
  /** Hide from assistive technology when the surrounding text already says what Luna is doing. */
  decorative?: boolean;
  className?: string;
};

export function Luna({ mood = "idle", size = 120, decorative = false, className }: LunaProps) {
  const id = useId().replaceAll(":", "");
  const body = `luna-body-${id}`;
  const glow = `luna-glow-${id}`;
  const label = LABELS[mood];
  const happyEyes = mood === "answering" || mood === "proud";
  const armsUp = mood === "proud";

  return (
    <svg
      className={["luna", `luna-${mood}`, className].filter(Boolean).join(" ")}
      width={size}
      height={size}
      viewBox="0 0 160 160"
      role={decorative ? undefined : "img"}
      aria-label={decorative ? undefined : label}
      aria-hidden={decorative ? true : undefined}
      focusable="false"
    >
      <defs>
        <radialGradient id={body} cx="42%" cy="34%" r="75%">
          <stop offset="0%" stopColor="#FFFBEA" />
          <stop offset="62%" stopColor="#FCEBC0" />
          <stop offset="100%" stopColor="#F2D597" />
        </radialGradient>
        <radialGradient id={glow} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#FFD58A" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#FFD58A" stopOpacity="0" />
        </radialGradient>
      </defs>

      <ellipse className="luna-shadow" cx="80" cy="148" rx="30" ry="5" fill={INK} opacity="0.1" />

      <g className="luna-float">
        {mood === "thinking" && (
          <g className="luna-orbit" aria-hidden="true">
            <Star x={36} y={40} size={6} color="#B9A8F0" />
            <Star x={124} y={34} size={5} color="#FF9A5A" />
            <Star x={130} y={78} size={4} color="#9FC8A8" />
          </g>
        )}

        {mood !== "sleepy" && (
          <g className="luna-sprout">
            <path d="M80 46 C80 38 81 33 83 28" stroke="#6FA56F" strokeWidth="3" strokeLinecap="round" fill="none" />
            <path d="M82 30 C73 20 62 22 60 30 C68 34 76 34 82 30 Z" fill="#9FC8A8" />
            <path d="M83 28 C90 16 102 16 105 23 C98 30 90 31 83 28 Z" fill="#8DBF8B" />
          </g>
        )}

        <g className={armsUp ? "luna-arms-up" : "luna-arms"}>
          {armsUp ? (
            <>
              <ellipse cx="36" cy="74" rx="7" ry="12" transform="rotate(-28 36 74)" fill={`url(#${body})`} />
              <ellipse cx="124" cy="74" rx="7" ry="12" transform="rotate(28 124 74)" fill={`url(#${body})`} />
            </>
          ) : (
            <>
              <ellipse cx="40" cy="108" rx="7" ry="11" transform="rotate(24 40 108)" fill={`url(#${body})`} />
              <ellipse
                className={mood === "checkin" ? "luna-wave" : undefined}
                cx={mood === "checkin" ? 126 : 120}
                cy={mood === "checkin" ? 80 : 108}
                rx="7"
                ry="11"
                transform={mood === "checkin" ? "rotate(-30 126 80)" : "rotate(-24 120 108)"}
                fill={`url(#${body})`}
              />
            </>
          )}
        </g>

        <path
          d="M80 44 C104 44 120 66 122 94 C124 120 104 134 80 134 C56 134 36 120 38 94 C40 66 56 44 80 44 Z"
          fill={`url(#${body})`}
        />

        {mood === "sleepy" && (
          <g aria-hidden="true">
            <path d="M44 72 C46 44 70 36 92 40 C110 43 124 36 134 26 C136 46 126 60 116 66 C96 58 64 60 44 72 Z" fill="#B9A8F0" />
            <path d="M42 72 C62 60 98 58 118 68" stroke="#F4F0FF" strokeWidth="7" strokeLinecap="round" fill="none" />
            <circle cx="135" cy="25" r="7" fill="#F4F0FF" />
          </g>
        )}

        {mood === "thinking" && (
          <ellipse cx="94" cy="118" rx="6.5" ry="9" transform="rotate(-40 94 118)" fill="#F6DFA8" />
        )}

        <ellipse cx="57" cy="102" rx="8" ry="5" fill="#FFB4A2" opacity="0.7" />
        <ellipse cx="103" cy="102" rx="8" ry="5" fill="#FFB4A2" opacity="0.7" />

        <g className="luna-eyes">
          <Eyes mood={mood} happy={happyEyes} />
        </g>
        <Mouth mood={mood} />

        {mood === "answering" && (
          <g className="luna-twinkle" aria-hidden="true">
            <Star x={128} y={48} size={9} color="#FF9A5A" />
          </g>
        )}
        {mood === "proud" && (
          <g className="luna-petals" aria-hidden="true">
            <Petal x={30} y={30} color="#FF9A5A" />
            <Petal x={130} y={24} color="#FFB4A2" />
            <Petal x={112} y={10} color="#B9A8F0" />
            <Petal x={46} y={8} color="#FFB4A2" />
          </g>
        )}
        {mood === "oops" && (
          <path
            className="luna-sweat"
            d="M118 62 C122 70 124 74 120 78 C116 80 112 76 114 72 C115 69 117 66 118 62 Z"
            fill="#9CC9F0"
            aria-hidden="true"
          />
        )}
        {mood === "checkin" && (
          <g aria-hidden="true">
            <path d="M18 128 h20 l-3 14 h-14 z" fill="#E7A27A" />
            <path d="M28 128 C28 118 30 114 34 112" stroke="#6FA56F" strokeWidth="2.5" fill="none" strokeLinecap="round" />
            <path d="M33 114 C38 106 46 108 46 112 C41 116 37 116 33 114 Z" fill="#9FC8A8" />
          </g>
        )}
        {mood === "support" && (
          <g className="luna-lantern" aria-hidden="true">
            <circle cx="128" cy="118" r="22" fill={`url(#${glow})`} className="luna-lantern-glow" />
            <path d="M128 96 v6" stroke={INK} strokeWidth="2" strokeLinecap="round" />
            <rect x="119" y="102" width="18" height="24" rx="6" fill="#FFE3A3" stroke="#6B5A8E" strokeWidth="2" />
            <rect x="122" y="126" width="12" height="4" rx="2" fill="#6B5A8E" />
            <rect x="122" y="99" width="12" height="4" rx="2" fill="#6B5A8E" />
          </g>
        )}
      </g>

      {mood === "thinking" && (
        <g className="luna-bubble" aria-hidden="true">
          <circle cx="108" cy="46" r="3" fill="#FFFFFF" stroke="#E4DDF7" />
          <circle cx="116" cy="36" r="4.5" fill="#FFFFFF" stroke="#E4DDF7" />
          <rect x="112" y="8" width="42" height="22" rx="11" fill="#FFFFFF" stroke="#E4DDF7" />
          <text x="133" y="23" textAnchor="middle" fontSize="10" fontWeight="700" fill="#6B5A8E">hmm…</text>
        </g>
      )}
      {mood === "sleepy" && (
        <g className="luna-zz" aria-hidden="true">
          <text x="120" y="70" fontSize="14" fontWeight="800" fill="#B9A8F0">z</text>
          <text x="132" y="56" fontSize="11" fontWeight="800" fill="#B9A8F0">z</text>
        </g>
      )}
    </svg>
  );
}

function Eyes({ mood, happy }: { mood: LunaMood; happy: boolean }) {
  if (happy) {
    return (
      <>
        <path d="M60 90 Q66 83 72 90" stroke={INK} strokeWidth="3.2" strokeLinecap="round" fill="none" />
        <path d="M88 90 Q94 83 100 90" stroke={INK} strokeWidth="3.2" strokeLinecap="round" fill="none" />
      </>
    );
  }
  if (mood === "sleepy") {
    return (
      <>
        <path d="M60 90 Q66 95 72 90" stroke={INK} strokeWidth="3" strokeLinecap="round" fill="none" />
        <path d="M88 90 Q94 95 100 90" stroke={INK} strokeWidth="3" strokeLinecap="round" fill="none" />
      </>
    );
  }
  if (mood === "oops") {
    const swirl = (cx: number) =>
      `M${cx} 89 m-1 0 a1 1 0 1 1 2 0 a3 3 0 1 1 -5 -1 a5 5 0 1 1 9 2`;
    return (
      <>
        <path d={swirl(66)} stroke="#E07A4F" strokeWidth="2" strokeLinecap="round" fill="none" />
        <path d={swirl(94)} stroke="#E07A4F" strokeWidth="2" strokeLinecap="round" fill="none" />
      </>
    );
  }
  if (mood === "support") {
    return (
      <>
        <ellipse cx="66" cy="90" rx="3.8" ry="4.2" fill={INK} />
        <ellipse cx="94" cy="90" rx="3.8" ry="4.2" fill={INK} />
        <circle cx="64.8" cy="88.6" r="1.3" fill="#FFFFFF" />
        <circle cx="92.8" cy="88.6" r="1.3" fill="#FFFFFF" />
      </>
    );
  }
  const big = mood === "listening";
  const radius = big ? 5.2 : 4.4;
  const dx = mood === "thinking" ? -2 : 0;
  const dy = mood === "thinking" ? -3 : 0;
  return (
    <>
      <circle cx={66 + dx} cy={89 + dy} r={radius} fill={INK} />
      <circle cx={94 + dx} cy={89 + dy} r={radius} fill={INK} />
      <circle cx={64.6 + dx} cy={87.3 + dy} r={big ? 1.8 : 1.4} fill="#FFFFFF" />
      <circle cx={92.6 + dx} cy={87.3 + dy} r={big ? 1.8 : 1.4} fill="#FFFFFF" />
    </>
  );
}

function Mouth({ mood }: { mood: LunaMood }) {
  switch (mood) {
    case "listening":
      return <ellipse cx="80" cy="105" rx="2.8" ry="3.4" fill={INK} />;
    case "thinking":
      return <path d="M75 105 Q80 103 85 105" stroke={INK} strokeWidth="2.4" strokeLinecap="round" fill="none" />;
    case "answering":
    case "proud":
      return (
        <>
          <path d="M72 101 Q80 112 88 101 Z" fill={INK} />
          <path d="M76 106 Q80 109 84 106" stroke="#FF9A9A" strokeWidth="2.5" strokeLinecap="round" fill="none" />
        </>
      );
    case "oops":
      return <path d="M72 106 q4 -4 8 0 q4 4 8 0" stroke={INK} strokeWidth="2.4" strokeLinecap="round" fill="none" />;
    case "sleepy":
      return <ellipse cx="80" cy="105" rx="2.4" ry="2" fill={INK} opacity="0.8" />;
    case "support":
      return <path d="M75 103 Q80 106 85 103" stroke={INK} strokeWidth="2.2" strokeLinecap="round" fill="none" />;
    default:
      return <path d="M74 102 Q80 108 86 102" stroke={INK} strokeWidth="2.6" strokeLinecap="round" fill="none" />;
  }
}

function Star({ x, y, size, color }: { x: number; y: number; size: number; color: string }) {
  const s = size;
  return (
    <path
      d={`M${x} ${y - s} Q${x + s * 0.2} ${y - s * 0.2} ${x + s} ${y} Q${x + s * 0.2} ${y + s * 0.2} ${x} ${y + s} Q${x - s * 0.2} ${y + s * 0.2} ${x - s} ${y} Q${x - s * 0.2} ${y - s * 0.2} ${x} ${y - s} Z`}
      fill={color}
    />
  );
}

function Petal({ x, y, color }: { x: number; y: number; color: string }) {
  return (
    <path
      className="luna-petal"
      d={`M${x} ${y} c3 -4 8 -3 8 1 c0 4 -5 6 -8 5 c-2 -1 -2 -4 0 -6 z`}
      fill={color}
    />
  );
}
