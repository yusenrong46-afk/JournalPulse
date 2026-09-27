import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { reportedState } from "@/lib/reported-state";

describe("reported affective state", () => {
  test("does not replace a model or fallback confidence with 1", () => {
    expect(
      reportedState({
        valence: -0.2,
        arousal: 0.4,
        agency: 0.3,
        emotion_tags: ["avoidance"],
        confidence: 0.42,
      }).confidence,
    ).toBe(0.42);
    expect(
      reportedState({
        valence: 0,
        arousal: 0.5,
        agency: 0.5,
        emotion_tags: ["unclassified"],
        confidence: 0,
      }).confidence,
    ).toBe(0);
  });

  test("reflect and check-in do not hard-code confidence 1", () => {
    const reflect = readFileSync(resolve(process.cwd(), "app/reflect/page.tsx"), "utf8");
    const checkIn = readFileSync(resolve(process.cwd(), "app/check-in/page.tsx"), "utf8");
    expect(reflect).not.toContain("confidence: 1");
    expect(checkIn).not.toContain("confidence: 1");
  });
});
