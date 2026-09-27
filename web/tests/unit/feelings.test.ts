import { describe, expect, test } from "vitest";

import { GOALS, goalSentence, selfReport } from "@/lib/feelings";

describe("self-report from feeling buttons", () => {
  test("averages the chosen feelings and keeps them as tags", () => {
    const state = selfReport(["tired", "anxious"], null);
    expect(state.emotion_tags).toEqual(["tired", "anxious"]);
    expect(state.valence).toBeCloseTo(-0.4);
    expect(state.arousal).toBeCloseTo(0.5);
    expect(state.confidence).toBeGreaterThan(0.5);
    expect(state.confidence).toBeLessThan(1);
  });

  test("the opening mood face weighs more than the feelings for valence", () => {
    const state = selfReport(["calm"], -0.75);
    expect(state.valence).toBeLessThan(0);
  });

  test("an empty choice is low-confidence, not a fake certainty", () => {
    expect(selfReport([], null).confidence).toBeLessThanOrEqual(0.1);
    expect(selfReport(["not-a-feeling"], null).emotion_tags).toEqual([]);
  });
});

describe("goal sentence", () => {
  test("reads like something a person would say", () => {
    const settle = GOALS.find((goal) => goal.id === "settle")!;
    expect(goalSentence(["tired", "anxious", "sad"], settle)).toBe(
      "I'm feeling tired, anxious and sad. I'd like to calm down.",
    );
    expect(goalSentence([], settle)).toBe("I'd like to calm down.");
  });
});
