import { beforeEach, describe, expect, test } from "vitest";

import {
  clearReflectionDraft,
  loadReflectionDraft,
  type ReflectionDraft,
  saveReflectionDraft,
} from "@/lib/reflection-draft";

const draft: ReflectionDraft = {
  clientRequestId: "30000000-0000-4000-8000-000000000001",
  step: 2,
  text: "An unfinished private thought.",
  situation: "after work",
  energy: "medium",
  socialContext: "alone",
  consentOverride: null,
  retainOverride: false,
  analysis: null,
  state: {
    valence: -0.2,
    arousal: 0.6,
    agency: 0.4,
    emotion_tags: ["unfinished"],
    confidence: 1,
  },
  target: { goal: "settle", arousal: 0.35, agency: 0.65 },
  preview: null,
  selectedAction: "",
};

beforeEach(async () => {
  await clearReflectionDraft();
});

describe("encrypted reflection drafts", () => {
  test("round-trips an active draft and removes it explicitly", async () => {
    await saveReflectionDraft(draft);

    await expect(loadReflectionDraft()).resolves.toEqual(draft);
    await clearReflectionDraft();
    await expect(loadReflectionDraft()).resolves.toBeNull();
  });
});
