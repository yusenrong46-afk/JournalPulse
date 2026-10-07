import { expect, type Page, type Route, test } from "@playwright/test";

const API = "http://127.0.0.1:8000";
const CONVERSATION_ID = "10000000-0000-4000-8000-000000000010";
const DECISION_ID = "20000000-0000-4000-8000-000000000001";

const onboarded = {
  onboarded: true,
  llmConsent: false,
  retainText: false,
  encryptedDrafts: false,
  followUpMinutes: 10,
  locale: "CA",
};

const breathing = {
  id: "site_nhs_breathing",
  title: "NHS Breathing Exercises for Stress",
  url: "https://www.nhs.uk/mental-health/self-help/guides-tools-and-activities/breathing-exercises-for-stress/",
  summary: "A short breathing exercise.",
  provider: "NHS",
  resource_type: "website",
  coping_style: "move",
  duration_minutes: 5,
};

const walk = { ...breathing, id: "move_nhs_walking", title: "Walking for Health (NHS)", duration_minutes: 10 };

const decision = {
  decision_id: DECISION_ID,
  action_id: breathing.id,
  propensity: 1,
  policy_name: "fixed-baseline",
  policy_version: "1.0.0",
  safe_action_ids: [breathing.id, walk.id],
  explanation: "Baseline pick.",
  selection_source: "policy",
  eligible_for_ope: true,
};

function conversation(overrides: Record<string, unknown> = {}) {
  return {
    id: CONVERSATION_ID,
    user_id: "00000000-0000-4000-8000-000000000001",
    created_at: "2026-09-27T20:00:00Z",
    updated_at: "2026-09-27T20:00:00Z",
    status: "open",
    llm_consent: false,
    retain_text: false,
    safety_mode: "normal",
    summary: null,
    card: null,
    locale: "CA",
    prompt_version: "guided-2026-09-27.1",
    mode: "guided",
    feelings: [],
    ready_for_action: false,
    ...overrides,
  };
}

function message(role: "user" | "assistant", content: string, safety = "normal") {
  return {
    id: crypto.randomUUID(),
    conversation_id: CONVERSATION_ID,
    role,
    content,
    created_at: "2026-09-27T20:00:01Z",
    safety_mode: safety,
  };
}

const reflection = {
  id: "30000000-0000-4000-8000-000000000001",
  created_at: "2026-09-27T20:05:00Z",
  text: null,
  text_retained: false,
  context: { source: "conversation" },
  state: { valence: -0.4, arousal: 0.6, agency: 0.4, emotion_tags: ["tired", "stressed"], confidence: 0.6 },
  target: { goal: "settle" },
  reflection: { summary: "You checked in with Luna.", interpretation: "Calm down.", reflection_question: "What changed?" },
  safety: { mode: "normal", reasons: [], locale: "CA", exploration_allowed: true, resource_ids: [] },
  decision,
};

async function fulfilJson(route: Route, json: unknown, status = 200) {
  if (new URL(route.request().url()).pathname === "/v1/account/data-revision") {
    return route.fulfill({ status: 200, json: { revision: 0 } });
  }
  return route.fulfill({ status, json });
}

async function seed(page: Page, preferences: Record<string, unknown> | null = onboarded) {
  await page.addInitScript((value) => {
    if (value) window.localStorage.setItem("journalpulse_preferences_v1", JSON.stringify(value));
  }, preferences);
}

test("the chat composer keeps pasted messages within the API's limit", async ({ page }) => {
  await seed(page);
  let sentText = "";
  await page.route(`${API}/**`, (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/conversations") return fulfilJson(route, conversation(), 201);
    if (path === `/v1/conversations/${CONVERSATION_ID}/messages`) {
      sentText = route.request().postDataJSON().text;
      return fulfilJson(route, {
        conversation: conversation(), user_message: message("user", sentText),
        assistant_message: message("assistant", "I hear you."),
      });
    }
    return fulfilJson(route, { items: [] });
  });
  await page.goto("/talk");
  const input = page.getByLabel("Message Luna");
  await input.fill("a".repeat(2001));
  await expect(input).toHaveValue("a".repeat(2000));
  await page.keyboard.press("End");
  await page.keyboard.insertText("b");
  await expect(input).toHaveValue("a".repeat(2000));
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log")).toContainText("Luna said: I hear you.");
  expect(sentText).toHaveLength(2000);
  await expect(input).toHaveValue("");
});

test("chat privacy and options support keyboard focus and dismissal", async ({ page }) => {
  await seed(page);
  await page.route(`${API}/**`, (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/system/status") return fulfilJson(route, { analysis_mode: "ai_configured" });
    if (path === `/v1/conversations/${CONVERSATION_ID}`) {
      return fulfilJson(route, { conversation: conversation(), messages: [] });
    }
    return fulfilJson(route, { items: [] });
  });
  await page.goto("/talk");
  const privacy = page.getByRole("button", { name: /Change chat privacy/ });
  await privacy.click();
  const dialog = page.getByRole("dialog", { name: "How this chat is kept" });
  const first = dialog.getByRole("checkbox").first();
  const done = dialog.getByRole("button", { name: "Done", exact: true });
  await expect(first).toBeFocused();
  await done.focus();
  await page.keyboard.press("Tab");
  await expect(first).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(done).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(privacy).toBeFocused();

  await page.goto(`/talk?c=${CONVERSATION_ID}`);
  const options = page.getByRole("button", { name: "Chat options" });
  await options.click();
  const menu = page.getByRole("menu");
  const details = menu.getByRole("menuitem", { name: "How this chat is kept" });
  await expect(details).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(menu.getByRole("menuitem", { name: "End this chat" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect(options).toBeFocused();
  await options.click();
  await page.keyboard.press("Enter");
  await expect(dialog).toBeVisible();
  await expect(done).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(options).toBeFocused();
});

test("a new person meets Luna and chooses how Luna replies", async ({ page }) => {
  await seed(page, null);
  await page.route(`${API}/**`, (route) => fulfilJson(route, { items: [] }));
  await page.goto("/");
  await expect(page).toHaveURL(/\/welcome/);
  await expect(page.getByRole("heading", { name: "Hi, I’m Luna." })).toBeVisible();
  await page.getByRole("button", { name: "Nice to meet you" }).click();
  const next = page.getByRole("button", { name: "Continue" });
  await expect(next).toBeDisabled();
  await page.getByRole("button", { name: /Simple Luna/ }).click();
  await next.click();
  await page.getByRole("button", { name: "Let’s begin" }).click();
  await expect(page).toHaveURL(/\/talk/);
  await expect(page.getByRole("button", { name: /AI help off/ })).toBeVisible();
  const saved = await page.evaluate(() => JSON.parse(window.localStorage.getItem("journalpulse_preferences_v1") ?? "{}"));
  expect(saved).toMatchObject({ onboarded: true, llmConsent: false });
});

test("a chat goes from a mood tap to one saved small step", async ({ page }) => {
  await seed(page);
  const turns: Array<Record<string, unknown>> = [];
  let accepted: Record<string, unknown> | null = null;
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    if (url.pathname === "/v1/conversations" && method === "POST") {
      const body = route.request().postDataJSON();
      expect(body.llm_consent).toBe(false);
      return fulfilJson(route, conversation(), 201);
    }
    if (url.pathname === `/v1/conversations/${CONVERSATION_ID}/messages`) {
      const body = route.request().postDataJSON();
      turns.push(body);
      if (body.goal) {
        return fulfilJson(route, {
          conversation: conversation({
            ready_for_action: true,
            feelings: ["tired"],
            card: { resource_intent: "ground", card_reason: "Calm.", decision_preview: decision, actions: [breathing, walk], goal: body.goal },
          }),
          user_message: message("user", body.text),
          assistant_message: message("assistant", "Here are three small ways to calm things down."),
        });
      }
      return fulfilJson(route, {
        conversation: conversation({ ready_for_action: turns.length >= 2, feelings: ["tired", "stressed"] }),
        user_message: message("user", body.text),
        assistant_message: message("assistant", turns.length >= 2 ? "Want to find one small thing to try together?" : "What part of that is sitting with you most?"),
      });
    }
    if (url.pathname === `/v1/conversations/${CONVERSATION_ID}/accept`) {
      accepted = route.request().postDataJSON();
      return fulfilJson(route, reflection, 201);
    }
    return fulfilJson(route, { items: [] });
  });

  await page.goto("/talk");
  await page.getByRole("button", { name: /Low/ }).click();
  await expect(page.getByText("What part of that is sitting with you most?")).toBeVisible();
  expect(turns[0]).toMatchObject({ text: "I'm feeling kind of low.", mood_score: 2 });

  await page.getByLabel("Message Luna").fill("Work is a lot and I'm worn out.");
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByRole("button", { name: /Yes, let’s find one small thing/ }).click();

  const feelings = page.getByRole("group", { name: "Feelings" });
  await expect(feelings.getByRole("button", { name: /Tired/ })).toHaveAttribute("aria-pressed", "true");
  await expect(feelings.getByRole("button", { name: /Stressed/ })).toHaveAttribute("aria-pressed", "true");
  await feelings.getByRole("button", { name: /Stressed/ }).click();
  await page.getByRole("button", { name: "That’s it" }).click();
  await page.getByRole("button", { name: /Calm down/ }).click();

  // The click returns before the stubbed request is recorded; wait for it.
  await expect.poll(() => turns.at(-1)).toMatchObject({
    goal: "settle",
    text: "I'm feeling tired. I'd like to calm down.",
    confirmed_feelings: ["tired"],
  });
  const pick = page.getByRole("button", { name: /Luna’s pick/ });
  await expect(pick).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: /Walking for Health/ }).click();
  await page.getByRole("button", { name: "Let’s try it" }).click();

  await expect(page.getByRole("heading", { name: "Nice choice." })).toBeVisible();
  expect(accepted).toMatchObject({ action_id: walk.id });
  const report = (accepted as unknown as { self_report: { emotion_tags: string[]; valence: number; confidence: number | null } }).self_report;
  expect(report.emotion_tags).toEqual(["tired"]);
  expect(report.valence).toBeLessThan(0);
  expect(report.confidence).toBeNull();
});

test("after a reload the chat keeps the feelings the person confirmed, not Luna's guess", async ({ page }) => {
  await seed(page);
  await page.addInitScript((id) => window.localStorage.setItem("journalpulse_open_conversation_v1", id), CONVERSATION_ID);
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === `/v1/conversations/${CONVERSATION_ID}`) {
      return fulfilJson(route, {
        conversation: conversation({ feelings: ["anxious"], confirmed_feelings: ["tired", "sad"], reported_mood: 2, ready_for_action: true }),
        messages: [message("user", "I'm feeling kind of low."), message("assistant", "What part of that is sitting with you most?")],
      });
    }
    return fulfilJson(route, { items: [] });
  });
  await page.goto("/talk");
  await page.getByRole("button", { name: /Yes, let’s find one small thing/ }).click();
  const feelings = page.getByRole("group", { name: "Feelings" });
  await expect(feelings.getByRole("button", { name: /Tired/ })).toHaveAttribute("aria-pressed", "true");
  await expect(feelings.getByRole("button", { name: /Sad/ })).toHaveAttribute("aria-pressed", "true");
  await expect(feelings.getByRole("button", { name: /Anxious/ })).toHaveAttribute("aria-pressed", "false");
});

test("a reply to a chat that closed elsewhere is not shown as saved", async ({ page }) => {
  await seed(page);
  let closed = false;
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/conversations" && route.request().method() === "POST") return fulfilJson(route, conversation(), 201);
    if (url.pathname.endsWith("/messages")) {
      closed = true;
      return fulfilJson(route, { detail: "This conversation is closed." }, 409);
    }
    if (url.pathname === `/v1/conversations/${CONVERSATION_ID}`) {
      return fulfilJson(route, { conversation: conversation({ status: closed ? "closed" : "open" }), messages: [] });
    }
    return fulfilJson(route, { items: [] });
  });
  await page.goto("/talk");
  await page.getByLabel("Message Luna").fill("Are you still there?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.locator("p[role=alert]")).toHaveText("This conversation is closed.");
  await expect(page.getByRole("button", { name: "Start a new chat" })).toBeVisible();
  // Submitted writing stays visible, but must never look like a confirmed save.
  await expect(page.locator(".msg.from-me:not([data-message-status]) .bubble", { hasText: "Are you still there?" })).toHaveCount(0);
  await expect(page.locator(".msg.from-me[data-message-status='failed'] .bubble")).toContainText("Are you still there?");
  await expect(page.locator(".msg.from-me[data-message-status='failed'] .bubble"))
    .toContainText("Delivery not confirmed. Start a new chat to continue.");
  await expect(page.getByLabel("Message Luna")).toHaveValue("Are you still there?");
});

test("support mode puts people first and hides the chat box", async ({ page }) => {
  await seed(page);
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/conversations") return fulfilJson(route, conversation(), 201);
    if (url.pathname.endsWith("/messages")) {
      const body = route.request().postDataJSON();
      return fulfilJson(route, {
        conversation: conversation({
          safety_mode: "support",
          card: {
            resource_intent: "pause",
            card_reason: "Human support first.",
            decision_preview: { ...decision, action_id: "support_988_canada", safe_action_ids: ["support_988_canada"] },
            actions: [{ ...breathing, id: "support_988_canada", title: "9-8-8 Suicide Crisis Helpline — Canada", resource_type: "support", url: "https://988.ca/" }],
          },
        }),
        user_message: message("user", body.text, "support"),
        assistant_message: message("assistant", "Please contact 9-8-8 now.", "support"),
      });
    }
    return fulfilJson(route, { items: [] });
  });
  await page.goto("/talk");
  await page.getByLabel("Message Luna").fill("I want to end my life.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("You don’t have to handle this alone.")).toBeVisible();
  await expect(page.getByRole("link", { name: "Call or text 9-8-8" })).toHaveAttribute("href", "https://988.ca/");
  await expect(page.getByLabel("Message Luna")).toHaveCount(0);
  await expect(page.getByRole("img", { name: "Luna is here with you" })).toBeVisible();
});

test("a check-in records how much the step helped", async ({ page }) => {
  await seed(page);
  let outcome: Record<string, unknown> | null = null;
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    if (url.pathname === "/v1/reflections") return fulfilJson(route, { items: [reflection] });
    if (url.pathname === "/v1/resources") return fulfilJson(route, { items: [breathing, walk] });
    if (url.pathname === "/v1/outcomes" && method === "POST") {
      outcome = route.request().postDataJSON();
      return fulfilJson(route, { id: "o1", decision_id: DECISION_ID, created_at: "2026-09-27T20:20:00Z", completed: true }, 201);
    }
    return fulfilJson(route, { items: [] });
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /NHS Breathing Exercises for Stress.*Did it help/ })).toBeVisible();
  await page.getByRole("link", { name: /A lot/ }).click();
  await expect(page.getByRole("heading", { name: "How much did it help?" })).toBeVisible();
  await expect(page.getByRole("button", { name: /A lot/ })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: /Calm$/ }).click();
  await page.getByRole("button", { name: "Save my check-in" }).click();
  await expect(page.getByRole("heading", { name: "Thank you!" })).toBeVisible();
  expect(outcome).toMatchObject({ decision_id: DECISION_ID, completed: true, helpfulness: 5 });
  expect((outcome as unknown as { post_state: { emotion_tags: string[] } }).post_state.emotion_tags).toEqual(["calm"]);
});

test("journey grows a plant for each check-in and names what helped", async ({ page }) => {
  await seed(page);
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/reflections") return fulfilJson(route, { items: [reflection] });
    if (url.pathname === "/v1/resources") return fulfilJson(route, { items: [breathing] });
    if (url.pathname === "/v1/outcomes") {
      return fulfilJson(route, { items: [{ id: "o1", decision_id: DECISION_ID, created_at: "2026-09-27T20:20:00Z", completed: true, helpfulness: 5 }] });
    }
    return fulfilJson(route, { items: [] });
  });
  await page.goto("/journey");
  // The journey names what helped from the person's own rating, never from a timer.
  await expect(page.getByRole("heading", { name: "Last two weeks" })).toBeVisible();
  await expect(page.getByText("1 rating · 1 rated helpful", { exact: false })).toBeVisible();
  await expect(page.locator(".entry .chips").getByText("Tired", { exact: true })).toBeVisible();
});

test("the main navigation has three calm destinations", async ({ page }) => {
  await seed(page);
  await page.route(`${API}/**`, (route) => fulfilJson(route, { items: [] }));
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Main" }).or(page.getByRole("complementary", { name: "Main" }));
  const visible = nav.locator("visible=true").first();
  await expect(visible.getByRole("link", { name: "Home", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(visible.getByRole("link", { name: "Journey", exact: true })).toHaveAttribute("href", "/journey");
  await expect(visible.getByRole("link", { name: "Settings", exact: true })).toHaveAttribute("href", "/me");
  await expect(page.getByRole("link", { name: "Talk with Luna" }).first()).toBeVisible();
  await expect(page.getByText(/Observe|Orient/)).toHaveCount(0);
});

for (const [from, to] of [["/reflect", "/talk"], ["/history", "/journey"], ["/patterns", "/journey"], ["/privacy", "/me"]]) {
  test(`${from} now leads to ${to}`, async ({ page }) => {
    await seed(page);
    await page.route(`${API}/**`, (route) => fulfilJson(route, { items: [] }));
    await page.goto(from);
    await expect(page).toHaveURL(new RegExp(`${to}$`));
  });
}
