import { expect, test } from "@playwright/test";

const analysis = {
  state: {
    valence: -0.4,
    arousal: 0.7,
    agency: 0.35,
    emotion_tags: ["frustration", "work stress"],
    confidence: 0.82,
    uncertainty: "The desired outcome is not explicit.",
  },
  reflection: {
    summary: "The meeting still feels unresolved.",
    interpretation: "The language suggests frustration and reduced agency.",
    reflection_question: "What would make this feel complete?",
  },
  safety: {
    mode: "normal",
    reasons: [],
    locale: "CA",
    exploration_allowed: true,
    resource_ids: [],
  },
  model_run: {
    model: "deterministic-fallback",
    provider: "openrouter",
    latency_ms: 0,
    schema_valid: true,
    used_fallback: true,
  },
  resource_intent: "reflect",
};

test("guided reflection preserves the human correction step", async ({ page }) => {
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith("/analyze")) return route.fulfill({ json: analysis });
    if (url.pathname === "/v1/reflections") {
      return route.fulfill({
        status: 201,
        json: {
          id: "10000000-0000-4000-8000-000000000001",
          user_id: "00000000-0000-4000-8000-000000000001",
          created_at: "2026-07-12T12:00:00Z",
          text: null,
          text_retained: false,
          context: {},
          state: analysis.state,
          target: { goal: "settle", valence: 0, arousal: 0.35, agency: 0.65 },
          reflection: analysis.reflection,
          safety: analysis.safety,
          decision: {
            decision_id: "20000000-0000-4000-8000-000000000001",
            action_id: "mindful_breathing_ucla",
            propensity: 1,
            policy_name: "fixed-baseline",
            policy_version: "1.0.0",
            safe_action_ids: ["mindful_breathing_ucla"],
            context_snapshot: {},
            explanation: "Selected deterministically from the reviewed safe set.",
          },
          model_run: analysis.model_run,
        },
      });
    }
    if (url.pathname === "/v1/resources") {
      return route.fulfill({
        json: {
          items: [
            {
              id: "mindful_breathing_ucla",
              title: "A two-minute breathing reset",
              url: "https://www.uclahealth.org/",
              summary: "A short guided pause from a reviewed source.",
              provider: "UCLA Health",
              resource_type: "website",
              coping_style: "reflect",
            },
          ],
        },
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Unhandled test route" } });
  });

  await page.goto("/reflect");
  await page.getByPlaceholder("Write without trying to sound composed…").fill(
    "The meeting is replaying in my head and I cannot settle.",
  );
  await page.getByRole("button", { name: "Check the signal" }).click();
  await expect(page.getByText("The system’s read is a proposal, not a verdict.")).toBeVisible();
  await page.getByRole("button", { name: "This reflects me" }).click();
  await page.getByRole("button", { name: "Find one next move" }).click();
  await expect(page.getByRole("heading", { name: "A two-minute breathing reset" })).toBeVisible();
  await expect(page.getByText("fixed-baseline · propensity 1.00")).toBeVisible();
});

test("mobile Today screen has a stable scientific-journal composition", async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.startsWith("mobile"), "Mobile visual baseline only");
  await page.route("http://127.0.0.1:8000/**", (route) => {
    const pathname = new URL(route.request().url()).pathname;
    return route.fulfill({
      json:
        pathname === "/v1/insights"
          ? {
              reflection_count: 0,
              completed_outcomes: 0,
              action_counts: {},
              average_helpfulness_by_action: {},
              average_state_change: null,
              note: "Descriptive only.",
            }
          : { items: [] },
    });
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Begin with one honest observation" })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth),
  ).toBe(true);
  await expect(page).toHaveScreenshot("today-mobile.png", {
    animations: "disabled",
    fullPage: true,
    maxDiffPixelRatio: 0.02,
  });
});
