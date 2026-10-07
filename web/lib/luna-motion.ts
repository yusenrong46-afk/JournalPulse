/** A present-tense visual response, never a diagnosis or a new stored emotion. */
export type LunaMood =
  | "idle" | "listening" | "thinking" | "answering" | "proud" | "oops"
  | "sleepy" | "checkin" | "support" | "reflecting" | "comforting"
  | "grounded" | "offering" | "resting" | "encouraging";

export type ActivityReaction = {
  conversationId?: string;
  messageId?: string | null;
  status?: string;
  followUp?: string;
  participation?: string;
  stateChange?: string | null;
};

type MotionContext = {
  support?: boolean; error?: boolean; waiting?: boolean; typing?: boolean;
  closed?: boolean; move?: string; hasOffer?: boolean; hasReply?: boolean;
  confirmedFeelings?: readonly string[] | null;
  replyFeelings?: readonly string[] | null;
  openingMood?: number | null;
  activity?: ActivityReaction | null;
};
const HEAVY = new Set(["anxious", "stressed", "sad", "frustrated", "lonely", "overwhelmed", "numb"]);
const LIGHT = new Set(["happy", "hopeful"]);

export function resolveLunaMood(context: MotionContext): LunaMood {
  // Safety and lifecycle outrank inferred feelings and decorative gestures.
  if (context.support) return "support";
  if (context.error) return "oops";
  if (context.closed || context.move === "pause") return "resting";
  if (context.waiting || context.activity?.followUp === "generating") return "thinking";
  if (context.typing) return "listening";
  if (context.activity?.status === "awaiting_report") return "checkin";
  if (context.hasOffer) return "offering";
  // A report is stronger evidence than a timer ending. Not-tried, unchanged
  // and uncertain results never trigger celebration.
  if (context.activity?.participation) {
    if (context.activity.stateChange === "away_from_target") return "comforting";
    if (context.activity.stateChange === "toward_target"
      && ["completed", "partial"].includes(context.activity.participation)) return "grounded";
    return "reflecting";
  }
  // Confirmations must come from the CURRENT user turn. An empty correction
  // clears suggestions rather than restoring a previous interpretation.
  if (context.confirmedFeelings == null && context.openingMood != null) {
    // A direct mood choice outranks a model's tentative label on that opening
    // turn. The caller stops supplying this score after the next user message.
    if (context.openingMood <= 2) return "comforting";
    if (context.openingMood >= 4) return "encouraging";
    return "reflecting";
  }
  const feelings = context.confirmedFeelings ?? context.replyFeelings ?? [];
  const heavy = feelings.some((feeling) => HEAVY.has(feeling));
  const light = feelings.some((feeling) => LIGHT.has(feeling));
  if (heavy && light) return "reflecting";
  if (heavy) return "comforting";
  if (feelings.includes("tired")) return "sleepy";
  if (feelings.includes("calm")) return "grounded";
  if (light) return "encouraging";
  if (context.activity?.status === "paused") return "resting";
  if (context.activity?.status === "active") return "grounded";
  if (context.hasReply || context.move === "reflect" || context.move === "clarify") return "reflecting";
  return "idle";
}
