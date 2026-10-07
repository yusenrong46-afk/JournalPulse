/** A day mark: empty, a journal/chat moment (sage half), an activity check-in (amber), or today. */
export type MoonKind = "empty" | "moment" | "activity";

export function MoonMark({ kind = "empty", today = false, size = 24, label }: {
  kind?: MoonKind; today?: boolean; size?: number; label?: string;
}) {
  const fill = kind === "activity" ? "#F4E4C8" : kind === "moment" ? "#E3EADB" : "none";
  const stroke = today ? "#24493C" : kind === "activity" ? "#D9A35B" : kind === "moment" ? "#6E9474" : "#C2C9B9";
  return (
    <svg className="moon-mark" width={size} height={size} viewBox="0 0 24 24"
      role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true} focusable="false">
      <circle cx="12" cy="12" r="8.5" fill={fill} stroke={stroke} strokeWidth={today ? 1.8 : 1.4} />
      {kind === "activity" && <path d="M12 3.5a8.5 8.5 0 0 1 0 17a4.5 8.5 0 0 0 0-17z" fill="#D9A35B" />}
      {kind === "moment" && <path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="#6E9474" />}
    </svg>
  );
}
