import type { AffectiveState } from "./types";

export type Feeling = {
  id: string;
  label: string;
  emoji: string;
  valence: number;
  arousal: number;
  agency: number;
};

// Must match FEELINGS in src/journalpulse/domain.py.
export const FEELINGS: Feeling[] = [
  { id: "tired", label: "Tired", emoji: "😴", valence: -0.3, arousal: 0.2, agency: 0.4 },
  { id: "anxious", label: "Anxious", emoji: "😟", valence: -0.5, arousal: 0.8, agency: 0.35 },
  { id: "stressed", label: "Stressed", emoji: "😣", valence: -0.45, arousal: 0.75, agency: 0.4 },
  { id: "sad", label: "Sad", emoji: "😢", valence: -0.6, arousal: 0.3, agency: 0.35 },
  { id: "frustrated", label: "Frustrated", emoji: "😤", valence: -0.5, arousal: 0.75, agency: 0.45 },
  { id: "lonely", label: "Lonely", emoji: "🥺", valence: -0.5, arousal: 0.3, agency: 0.35 },
  { id: "overwhelmed", label: "Overwhelmed", emoji: "🌊", valence: -0.65, arousal: 0.85, agency: 0.2 },
  { id: "numb", label: "Numb", emoji: "😶", valence: -0.3, arousal: 0.15, agency: 0.3 },
  { id: "calm", label: "Calm", emoji: "😌", valence: 0.45, arousal: 0.25, agency: 0.65 },
  { id: "hopeful", label: "Hopeful", emoji: "🌱", valence: 0.5, arousal: 0.5, agency: 0.7 },
  { id: "okay", label: "Okay", emoji: "🙂", valence: 0.1, arousal: 0.4, agency: 0.55 },
  { id: "happy", label: "Happy", emoji: "😊", valence: 0.7, arousal: 0.6, agency: 0.7 },
];

export type Mood = { score: number; label: string; emoji: string; valence: number; sentence: string };

export const MOODS: Mood[] = [
  { score: 5, label: "Great", emoji: "😄", valence: 0.75, sentence: "I'm feeling great today." },
  { score: 4, label: "Good", emoji: "🙂", valence: 0.4, sentence: "I'm feeling pretty good." },
  { score: 3, label: "Okay", emoji: "😐", valence: 0, sentence: "I'm feeling okay, just so-so." },
  { score: 2, label: "Low", emoji: "😔", valence: -0.4, sentence: "I'm feeling kind of low." },
  { score: 1, label: "Rough", emoji: "😣", valence: -0.75, sentence: "Honestly, I'm having a rough time." },
];

export type GoalOption = { id: "settle" | "move" | "understand" | "connect" | "act"; label: string; emoji: string; phrase: string };

export const GOALS: GoalOption[] = [
  { id: "settle", label: "Calm down", emoji: "🍃", phrase: "calm down" },
  { id: "move", label: "Get some energy back", emoji: "☀️", phrase: "get some energy back" },
  { id: "understand", label: "Make sense of it", emoji: "💭", phrase: "make sense of it" },
  { id: "connect", label: "Feel less alone", emoji: "💜", phrase: "feel less alone" },
  { id: "act", label: "Take one small step", emoji: "👣", phrase: "take one small step" },
];

export function moodByScore(score: number | null | undefined): Mood | undefined {
  return MOODS.find((item) => item.score === score);
}

export function feelingById(id: string): Feeling | undefined {
  return FEELINGS.find((item) => item.id === id);
}

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

function round(value: number) {
  return Math.round(value * 100) / 100;
}

export const DERIVATION = "feeling-buttons-v1";

/**
 * A derived state built from the feelings the person tapped and, if they chose one, the
 * mood face they started with. The mood face anchors valence because it is the most
 * direct answer to "how are you?". Must match src/journalpulse/self_report.py. No
 * confidence is claimed: nobody measured how sure the person was.
 */
export function selfReport(feelingIds: string[], moodValence: number | null): AffectiveState {
  const chosen = feelingIds.map(feelingById).filter((item): item is Feeling => Boolean(item));
  if (chosen.length === 0) {
    return {
      valence: round(moodValence ?? 0),
      arousal: 0.5,
      agency: 0.5,
      emotion_tags: [],
      confidence: null,
      uncertainty: moodValence === null ? "No feelings were reported." : "Only an overall mood was reported.",
      derivation: DERIVATION,
    };
  }
  const average = (key: "valence" | "arousal" | "agency") =>
    chosen.reduce((sum, item) => sum + item[key], 0) / chosen.length;
  const feelingValence = average("valence");
  const valence = moodValence === null ? feelingValence : (feelingValence + moodValence * 2) / 3;
  return {
    valence: round(clamp(valence, -1, 1)),
    arousal: round(clamp(average("arousal"), 0, 1)),
    agency: round(clamp(average("agency"), 0, 1)),
    emotion_tags: chosen.map((item) => item.id).slice(0, 6),
    confidence: null,
    uncertainty: "Derived from the feeling buttons the person chose.",
    derivation: DERIVATION,
  };
}

export function goalSentence(feelingIds: string[], goal: GoalOption): string {
  const names = feelingIds
    .map((id) => feelingById(id)?.label.toLowerCase())
    .filter((item): item is string => Boolean(item));
  const feeling =
    names.length === 0
      ? ""
      : names.length === 1
        ? `I'm feeling ${names[0]}. `
        : `I'm feeling ${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}. `;
  return `${feeling}I'd like to ${goal.phrase}.`;
}

const STYLE_EMOJI: Record<string, string> = {
  move: "🌿",
  watch: "🎧",
  read: "📖",
  play: "🎲",
};

export function actionEmoji(copingStyle: string, resourceType?: string): string {
  if (resourceType === "support") return "🏮";
  return STYLE_EMOJI[copingStyle] ?? "✨";
}

export function actionTone(copingStyle: string, resourceType?: string): string {
  if (resourceType === "support") return "support";
  return ["move", "watch", "read", "play"].includes(copingStyle) ? copingStyle : "read";
}
