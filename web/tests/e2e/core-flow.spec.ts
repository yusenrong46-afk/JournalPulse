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

const preferences = {
  onboarded: true,
  llmConsent: false,
  retainText: false,
  encryptedDrafts: false,
  followUpMinutes: 10,
  locale: "CA",
};

async function onboard(page: import("@playwright/test").Page) {
  await page.addInitScript((value) => {
    window.localStorage.setItem("journalpulse_preferences_v1", JSON.stringify(value));
  }, preferences);
}

test("guided reflection preserves the human correction step", async ({ page }) => {
  await onboard(page);
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/system/status") {
      return route.fulfill({ json: { analysis_mode: "ai_configured", persistence_mode: "this_device", message: "Ready." } });
    }
    if (url.pathname.endsWith("/analyze")) return route.fulfill({ json: analysis });
    if (url.pathname === "/v1/actions/preview") {
      return route.fulfill({
        json: {
          decision: {
            decision_id: "20000000-0000-4000-8000-000000000001",
            action_id: "mindful_breathing_ucla",
            propensity: 1,
            policy_name: "fixed-baseline",
            policy_version: "1.0.0",
            safe_action_ids: ["mindful_breathing_ucla", "nature_reset"],
            context_snapshot: {},
            explanation: "Selected deterministically from the reviewed safe set.",
            recommended_action_id: null,
            selection_source: "policy",
            eligible_for_ope: true,
          },
          actions: [
            {
              id: "mindful_breathing_ucla",
              title: "A two-minute breathing reset",
              url: "https://www.uclahealth.org/",
              summary: "A short guided pause from a reviewed source.",
              provider: "UCLA Health",
              resource_type: "website",
              coping_style: "reflect",
              duration_minutes: 2,
            },
            {
              id: "nature_reset",
              title: "A quiet nature reset",
              url: "https://www.bbc.com/earth",
              summary: "A visual slowdown from a reviewed source.",
              provider: "BBC Earth",
              resource_type: "video",
              coping_style: "watch",
              duration_minutes: 5,
            },
          ],
        },
      });
    }
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
            recommended_action_id: "mindful_breathing_ucla",
            selection_source: "policy_accepted",
            eligible_for_ope: true,
          },
          model_run: analysis.model_run,
        },
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Unhandled test route" } });
  });

  await page.goto("/reflect");
  await page.getByPlaceholder("Write without trying to sound composed…").fill(
    "The meeting is replaying in my head and I cannot settle.",
  );
  await page.getByRole("button", { name: "Continue to my state" }).click();
  await expect(page.getByText("The system’s read is a proposal, not a verdict.")).toBeVisible();
  await expect(page.getByText("Local reflection mode")).toBeVisible();
  await page.getByRole("button", { name: "This reflects me" }).click();
  await page.getByRole("button", { name: "Show reviewed options" }).click();
  await expect(page.getByText("Choose the action you are actually willing to try.")).toBeVisible();
  await page.getByRole("button", { name: "Use this action" }).click();
  await expect(page.getByRole("heading", { name: "A two-minute breathing reset" })).toBeVisible();
  await expect(page.getByText(/policy accepted · eligible for policy evaluation/)).toBeVisible();
  const reminder = await page.evaluate(() => localStorage.getItem("journalpulse_reminders_v1"));
  expect(JSON.parse(reminder ?? "[]")).toHaveLength(1);
});

test("welcome flow stores explicit defaults before the first reflection", async ({ page }) => {
  await page.goto("/welcome");
  await page.getByRole("checkbox", { name: /Allow private AI analysis/ }).check();
  await page.getByRole("button", { name: "Set my preferences" }).click();
  await expect(page).toHaveURL(/\/reflect$/);
  const saved = await page.evaluate(() => window.localStorage.getItem("journalpulse_preferences_v1"));
  expect(JSON.parse(saved ?? "{}").llmConsent).toBe(true);
});

test("delayed check-in records a post-action state", async ({ page }) => {
  await onboard(page);
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "journalpulse_reminders_v1",
      JSON.stringify([{ decisionId: "20000000-0000-4000-8000-000000000001", actionId: "mindful_breathing_ucla", actionTitle: "A two-minute breathing reset", dueAt: "2026-07-12T12:10:00Z" }]),
    );
  });
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/system/status") return route.fulfill({ json: { analysis_mode: "ai_configured", persistence_mode: "this_device", message: "Ready." } });
    if (url.pathname === "/v1/reflections") {
      return route.fulfill({ json: { items: [{ id: "reflection-1", created_at: "2026-07-12T12:00:00Z", text_retained: false, context: {}, state: analysis.state, target: { goal: "settle" }, reflection: analysis.reflection, safety: analysis.safety, decision: { decision_id: "20000000-0000-4000-8000-000000000001", action_id: "mindful_breathing_ucla", propensity: 1, policy_name: "fixed-baseline", policy_version: "1.0.0", safe_action_ids: ["mindful_breathing_ucla"], context_snapshot: {}, explanation: "Baseline", selection_source: "policy_accepted", eligible_for_ope: true } }] } });
    }
    if (url.pathname === "/v1/outcomes" && route.request().method() === "GET") return route.fulfill({ json: { items: [] } });
    if (url.pathname === "/v1/outcomes" && route.request().method() === "POST") return route.fulfill({ status: 201, json: { id: "outcome-1", decision_id: "20000000-0000-4000-8000-000000000001", created_at: "2026-07-12T12:10:00Z", completed: true } });
    if (url.pathname === "/v1/resources") return route.fulfill({ json: { items: [{ id: "mindful_breathing_ucla", title: "A two-minute breathing reset", url: "https://www.uclahealth.org/", summary: "Pause.", provider: "UCLA", resource_type: "website", coping_style: "reflect" }] } });
    return route.fulfill({ status: 404, json: { detail: "Unhandled test route" } });
  });
  await page.goto("/check-in?decision=20000000-0000-4000-8000-000000000001");
  await expect(page.getByRole("heading", { name: "A two-minute breathing reset" })).toBeVisible();
  await page.getByRole("button", { name: "Record this outcome" }).click();
  await expect(page.getByRole("heading", { name: "One observation recorded." })).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("journalpulse_reminders_v1"))).toBe("[]");
});

test("mobile Today screen has a stable scientific-journal composition", async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.startsWith("mobile"), "Mobile visual baseline only");
  await onboard(page);
  await page.route("http://127.0.0.1:8000/**", (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === "/v1/system/status") return route.fulfill({ json: { analysis_mode: "ai_configured", persistence_mode: "this_device", message: "Ready." } });
    return route.fulfill({
      json:
        pathname === "/v1/insights"
          ? {
              reflection_count: 0,
              completed_outcomes: 0,
              action_counts: {},
              average_helpfulness_by_action: {},
              average_state_change: null,
              completion_rate: 0,
              pending_decision_ids: [],
              state_trajectory: [],
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

test("encrypted draft recovery survives a refresh only after opt-in", async ({ page }) => {
  await page.addInitScript((value) => {
    window.localStorage.setItem(
      "journalpulse_preferences_v1",
      JSON.stringify({ ...value, encryptedDrafts: true }),
    );
  }, preferences);
  await page.route("http://127.0.0.1:8000/**", (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === "/v1/system/status") {
      return route.fulfill({
        json: { analysis_mode: "local_fallback", persistence_mode: "this_device", message: "Local mode." },
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Not needed for draft recovery" } });
  });

  await page.goto("/reflect");
  const entry = page.getByPlaceholder("Write without trying to sound composed…");
  await entry.fill("This unfinished thought should survive one accidental refresh.");
  await expect(page.getByText("Encrypted draft saved on this device.")).toBeVisible();
  await page.reload();
  await expect(entry).toHaveValue("This unfinished thought should survive one accidental refresh.");
  await expect(page.getByText("Encrypted draft restored on this device.")).toBeVisible();
});
