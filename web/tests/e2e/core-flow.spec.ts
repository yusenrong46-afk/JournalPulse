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

const conversationId = "10000000-0000-4000-8000-000000000010";
const talkAction = {
  id: "mindful_breathing_ucla",
  title: "A two-minute breathing reset",
  url: "https://www.uclahealth.org/",
  summary: "A short guided pause from a reviewed source.",
  provider: "UCLA Health",
  resource_type: "website",
  coping_style: "reflect",
  duration_minutes: 2,
};

function openConversation(safetyMode: "normal" | "support" = "normal") {
  return {
    id: conversationId,
    user_id: "00000000-0000-4000-8000-000000000001",
    created_at: "2026-09-24T12:00:00Z",
    updated_at: "2026-09-24T12:00:00Z",
    status: "open",
    llm_consent: true,
    retain_text: false,
    safety_mode: safetyMode,
    summary: "The meeting still feels unresolved.",
    card: safetyMode === "support" ? null : {
      resource_intent: "reflect",
      card_reason: "A short pause matches what you asked for.",
      decision_preview: {
        decision_id: "20000000-0000-4000-8000-000000000010",
        action_id: talkAction.id,
        propensity: 1,
        policy_name: "fixed-baseline",
        policy_version: "1.0.0",
        safe_action_ids: [talkAction.id],
        explanation: "Selected from the reviewed catalog.",
        selection_source: "policy",
        eligible_for_ope: true,
      },
      actions: [talkAction],
      offered_message_id: "30000000-0000-4000-8000-000000000010",
    },
    safety: safetyMode === "support"
      ? { mode: "support", reasons: ["risk"], locale: "CA", exploration_allowed: false, support_message: "Contact 9-8-8.", resource_ids: [] }
      : { mode: "normal", reasons: [], locale: "CA", exploration_allowed: true, resource_ids: [] },
    reflection_id: null,
    locale: "CA",
    prompt_version: "2026-09-24.1",
  };
}

test("talk saves a reviewed action and a follow-up reminder", async ({ page }) => {
  await onboard(page);
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/system/status") {
      return route.fulfill({ json: { analysis_mode: "ai_configured", persistence_mode: "this_device", message: "Ready." } });
    }
    if (url.pathname === "/v1/conversations" && route.request().method() === "POST") {
      return route.fulfill({ status: 201, json: openConversation() });
    }
    if (url.pathname === `/v1/conversations/${conversationId}/messages`) {
      return route.fulfill({
        json: {
          conversation: openConversation(),
          user_message: { id: "30000000-0000-4000-8000-000000000011", conversation_id: conversationId, role: "user", content: "The meeting is still in my head.", created_at: "2026-09-24T12:01:00Z", safety_mode: "normal" },
          assistant_message: { id: "30000000-0000-4000-8000-000000000010", conversation_id: conversationId, role: "assistant", content: "That meeting is still taking up space. A short pause is one option.", created_at: "2026-09-24T12:01:01Z", safety_mode: "normal" },
        },
      });
    }
    if (url.pathname === `/v1/conversations/${conversationId}` && route.request().method() === "GET") {
      return route.fulfill({ json: { conversation: openConversation(), messages: [] } });
    }
    if (url.pathname === `/v1/conversations/${conversationId}/accept`) {
      return route.fulfill({
        status: 201,
        json: {
          id: "10000000-0000-4000-8000-000000000099",
          created_at: "2026-09-24T12:02:00Z",
          text_retained: false,
          context: { source: "conversation", conversation_id: conversationId },
          state: analysis.state,
          target: { goal: "understand" },
          reflection: analysis.reflection,
          safety: analysis.safety,
          decision: {
            decision_id: "20000000-0000-4000-8000-000000000010",
            action_id: talkAction.id,
            propensity: 1,
            policy_name: "fixed-baseline",
            policy_version: "1.0.0",
            safe_action_ids: [talkAction.id],
            explanation: "You accepted the baseline action.",
            selection_source: "policy_accepted",
            eligible_for_ope: true,
          },
        },
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Unhandled talk route" } });
  });

  await page.goto("/talk");
  await page.getByRole("checkbox", { name: /Use private AI analysis for this conversation/ }).check();
  await page.getByRole("button", { name: "Start this conversation" }).click();
  await page.getByPlaceholder("Write the next thing you want to say…").fill("The meeting is still in my head.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("That meeting is still taking up space.")).toBeVisible();
  await page.getByRole("button", { name: /A two-minute breathing reset/ }).click();
  await page.getByRole("button", { name: "Use this action" }).click();
  await expect(page.getByRole("link", { name: "Check in afterward" })).toBeVisible();
  const reminder = await page.evaluate(() => localStorage.getItem("journalpulse_reminders_v1"));
  expect(JSON.parse(reminder ?? "[]")).toHaveLength(1);
});

test("talk support mode shows 9-8-8 and hides the composer", async ({ page }) => {
  await onboard(page);
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/system/status") {
      return route.fulfill({ json: { analysis_mode: "ai_configured", persistence_mode: "this_device", message: "Ready." } });
    }
    if (url.pathname === "/v1/conversations" && route.request().method() === "POST") {
      return route.fulfill({ status: 201, json: openConversation() });
    }
    if (url.pathname === `/v1/conversations/${conversationId}/messages`) {
      const support = openConversation("support");
      return route.fulfill({
        json: {
          conversation: support,
          user_message: { id: "30000000-0000-4000-8000-000000000021", conversation_id: conversationId, role: "user", content: "I have a suicide plan", created_at: "2026-09-24T12:01:00Z", safety_mode: "support" },
          assistant_message: { id: "30000000-0000-4000-8000-000000000022", conversation_id: conversationId, role: "assistant", content: "Contact 9-8-8.", created_at: "2026-09-24T12:01:01Z", safety_mode: "support" },
        },
      });
    }
    if (url.pathname === `/v1/conversations/${conversationId}` && route.request().method() === "GET") {
      return route.fulfill({ json: { conversation: openConversation(), messages: [] } });
    }
    return route.fulfill({ status: 404, json: { detail: "Unhandled support route" } });
  });

  await page.goto("/talk");
  await page.getByRole("checkbox", { name: /Use private AI analysis for this conversation/ }).check();
  await page.getByRole("button", { name: "Start this conversation" }).click();
  await page.getByPlaceholder("Write the next thing you want to say…").fill("I have a suicide plan");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByRole("link", { name: "Open 9-8-8 Canada" })).toBeVisible();
  await expect(page.getByPlaceholder("Write the next thing you want to say…")).toHaveCount(0);
});

test("talk restores an open conversation from the server", async ({ page }) => {
  await onboard(page);
  await page.addInitScript((id) => {
    window.localStorage.setItem("journalpulse_open_conversation_v1", id);
  }, conversationId);
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/system/status") {
      return route.fulfill({ json: { analysis_mode: "ai_configured", persistence_mode: "this_device", message: "Ready." } });
    }
    if (url.pathname === `/v1/conversations/${conversationId}`) {
      return route.fulfill({
        json: {
          conversation: openConversation(),
          messages: [
            { id: "30000000-0000-4000-8000-000000000031", conversation_id: conversationId, role: "user", content: "Restored from the server.", created_at: "2026-09-24T12:01:00Z", safety_mode: "normal" },
            { id: "30000000-0000-4000-8000-000000000032", conversation_id: conversationId, role: "assistant", content: "I still have that from the server.", created_at: "2026-09-24T12:01:01Z", safety_mode: "normal" },
          ],
        },
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Unhandled restore route" } });
  });
  await page.goto("/talk");
  await expect(page.getByText("Restored from the server.")).toBeVisible();
  await page.reload();
  await expect(page.getByText("I still have that from the server.")).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("journalpulse_open_conversation_v1"))).toBe(conversationId);
  expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain("Restored from the server.");
});

test("talk links to the guided reflection when private AI is unavailable", async ({ page }) => {
  await onboard(page);
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/system/status") {
      return route.fulfill({ json: { analysis_mode: "local_only", persistence_mode: "this_device", message: "AI is off." } });
    }
    return route.fulfill({ status: 404, json: { detail: "Unavailable" } });
  });
  await page.goto("/talk");
  await expect(page.getByRole("heading", { name: "Talk needs private AI analysis." })).toBeVisible();
  await expect(page.getByRole("link", { name: /Open a guided reflection/ })).toHaveAttribute("href", "/reflect");
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
