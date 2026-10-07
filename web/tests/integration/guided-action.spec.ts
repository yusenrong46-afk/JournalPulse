// Real UI/API/PostgREST/PostgreSQL. Auth, Luna and Brave remain named local
// test doubles; these checks establish the activity loop, not emotional benefit.
import { expect, type APIRequestContext, type Page, test } from "@playwright/test";
import { currentDataRevisionHeaders } from "../helpers/data-revision";

type Session = { user_id: string; access_token: string };
type Activity = {
  id: string;
  status: string;
  remaining_seconds: number;
  follow_up_status: string;
  report: { participation: string; state_change: string | null } | null;
};

async function sessionFor(request: APIRequestContext, name: string): Promise<Session> {
  const response = await request.get(`http://127.0.0.1:54321/test/session/${name}`);
  expect(response.ok()).toBeTruthy();
  return response.json();
}

function headers(who: Session) {
  return { Authorization: `Bearer ${who.access_token}` };
}

async function clearAccount(request: APIRequestContext, who: Session) {
  expect((await request.delete("/v1/account/data", { headers: headers(who) })).ok()).toBeTruthy();
}

async function signIn(page: Page, who: Session) {
  await page.addInitScript((session: Session) => {
    window.localStorage.setItem(`journalpulse_preferences_v1:${session.user_id}`, JSON.stringify({
      onboarded: true, llmConsent: true, retainText: false, encryptedDrafts: false,
      followUpMinutes: 10, locale: "CA",
    }));
    window.localStorage.setItem("sb-127-auth-token", JSON.stringify({
      access_token: session.access_token, refresh_token: "integration-refresh",
      token_type: "bearer", expires_in: 86400,
      expires_at: Math.floor(Date.now() / 1000) + 86400,
      user: { id: session.user_id, aud: "authenticated", role: "authenticated" },
    }));
  }, who);
}

async function getActivity(request: APIRequestContext, who: Session, chatId: string): Promise<Activity> {
  const response = await request.get(`/v1/conversations/${chatId}/activity-sessions`, { headers: headers(who) });
  expect(response.ok()).toBeTruthy();
  return response.json();
}

async function askForQuietPause(page: Page) {
  await page.goto("/talk/");
  await page.getByLabel("Message Luna").fill("A fictional busy afternoon has left my head feeling crowded.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  await expect(page.getByLabel("Message Luna")).toBeEnabled();
  await page.getByLabel("Message Luna").fill("I have two minutes and would like a quiet seated pause, without audio.");
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  await expect(page.getByRole("button", { name: "Start activity", exact: true })).toBeVisible();
}

test("meditation stays in chat and a paused session survives reload before an honest check-in", async ({ page, request }, testInfo) => {
  const who = await sessionFor(request, "grace");
  await clearAccount(request, who);
  try {
    await signIn(page, who);
    await askForQuietPause(page);
    const chatId = new URL(page.url()).searchParams.get("c");
    expect(chatId).toBeTruthy();
    const activityPanel = page.getByRole("region", { name: "Activity with Luna", exact: true });
    await expect(activityPanel.getByRole("heading", { name: "Two-minute quiet meditation", exact: true })).toBeVisible();
    // A keyboard activation follows the same owner-bound start command as a tap.
    const start = activityPanel.getByRole("button", { name: "Start activity", exact: true });
    await start.focus();
    await page.keyboard.press("Enter");
    // Once started, controls live in the compact bar above the composer, outside the chat log.
    const activityBar = page.getByRole("region", { name: "Current activity", exact: true });
    await expect(activityBar.getByRole("button", { name: "Pause", exact: true })).toBeVisible();
    await expect(activityBar.getByRole("timer")).toHaveAttribute("aria-live", "off");
    await expect(page.getByRole("log").getByRole("button", { name: "Pause", exact: true })).toHaveCount(0);
    let activity = await getActivity(request, who, chatId!);
    expect(activity.status).toBe("active");
    const chat = await (await request.get(`/v1/conversations/${chatId}`, { headers: headers(who) })).json();
    expect(chat.conversation.status).toBe("open");
    expect(chat.messages.some((message: { content: string | null }) => message.content?.includes("fictional busy afternoon"))).toBeTruthy();

    await activityBar.getByRole("button", { name: "Pause", exact: true }).click();
    await expect(activityBar.getByRole("button", { name: "Resume", exact: true })).toBeVisible();
    activity = await getActivity(request, who, chatId!);
    expect(activity.status).toBe("paused");
    const pausedRemaining = activity.remaining_seconds;
    await page.reload();
    await expect(activityBar.getByRole("button", { name: "Resume", exact: true })).toBeVisible();
    const recovered = await getActivity(request, who, chatId!);
    expect(recovered.id).toBe(activity.id);
    expect(recovered.remaining_seconds).toBe(pausedRemaining);
    expect(recovered.report).toBeNull();

    await activityBar.getByRole("button", { name: "Finish early", exact: true }).click();
    const checkIn = page.getByRole("form", { name: "Activity check-in", exact: true });
    await expect(page.getByText("Did you try it?", { exact: true })).toHaveCount(1);
    await checkIn.getByRole("radio", { name: "Not tried", exact: true }).check();
    await checkIn.getByRole("button", { name: "Save check-in", exact: true }).click();
    await expect.poll(async () => (await getActivity(request, who, chatId!)).follow_up_status).toBe("ready");
    const reported = await getActivity(request, who, chatId!);
    expect(reported.report?.participation).toBe("not_tried");
    // Opening or timing an activity cannot create a legacy completed outcome.
    const exported = await (await request.get("/v1/export", { headers: headers(who) })).json();
    expect(exported.outcomes).toEqual([]);
    expect(exported.activity_sessions).toHaveLength(1);
    await expect(page.getByLabel("Message Luna")).toBeEnabled();
    const stillOpen = await (await request.get(`/v1/conversations/${chatId}`, { headers: headers(who) })).json();
    expect(stillOpen.conversation.status).toBe("open");
    await page.screenshot({ path: testInfo.outputPath("guided-action-mobile-check-in.png"), fullPage: true });

    // The report, not the timer, reaches the garden through PostgREST and owner RLS.
    const history = await (await request.get("/v1/activity-history", { headers: headers(who) })).json();
    expect(history.items).toHaveLength(1);
    expect(history.items[0]).toMatchObject({ id: activity.id, participation: "not_tried" });
    const stranger = await sessionFor(request, "frank");
    expect((await (await request.get("/v1/activity-history", { headers: headers(stranger) })).json()).items).toEqual([]);
    await page.goto("/journey/");
    const fromChats = page.getByRole("region", { name: "Activities from your chats" });
    await expect(fromChats.getByText("Two-minute quiet meditation", { exact: true })).toBeVisible();
    await expect(fromChats.getByText(/Didn.t try it/)).toBeVisible();
  } finally {
    await clearAccount(request, who);
  }
});

test("changing account while an activity is running removes its private panel and controls", async ({ page, request }) => {
  const first = await sessionFor(request, "grace");
  const second = await sessionFor(request, "frank");
  for (const who of [first, second]) await clearAccount(request, who);
  try {
    await signIn(page, first);
    await askForQuietPause(page);
    await page.getByRole("button", { name: "Start activity", exact: true }).click();
    await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeVisible();
    const chatId = new URL(page.url()).searchParams.get("c");
    expect(chatId).toBeTruthy();
    const activity = await getActivity(request, first, chatId!);
    const originalUrl = page.url();
    await page.evaluate((who: Session) => {
      const nextSession = {
        access_token: who.access_token, refresh_token: "integration-refresh", token_type: "bearer",
        expires_in: 86400, expires_at: Math.floor(Date.now() / 1000) + 86400,
        user: { id: who.user_id, aud: "authenticated", role: "authenticated" },
      };
      window.localStorage.setItem("sb-127-auth-token", JSON.stringify(nextSession));
      const channel = new BroadcastChannel("sb-127-auth-token");
      channel.postMessage({ event: "SIGNED_IN", session: nextSession });
      channel.close();
    }, second);
    await expect(page.getByRole("region", { name: "Activity with Luna", exact: true })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Pause", exact: true })).toHaveCount(0);
    await expect(page.getByText("Did you try it?", { exact: true })).toHaveCount(0);
    await expect(page.getByText("A fictional busy afternoon has left my head feeling crowded.", { exact: true })).toHaveCount(0);
    expect(page.url()).toBe(originalUrl);
    // UI disappearance is backed by owner enforcement on the real API.
    expect((await request.get(`/v1/activity-sessions/${activity.id}`, { headers: headers(second) })).status()).toBe(404);
    expect((await request.get(`/v1/activity-sessions/${activity.id}`, { headers: headers(first) })).status()).toBe(200);
  } finally {
    for (const who of [first, second]) await clearAccount(request, who);
  }
});

test("inline search uses the real request contract and saves a refined signed offer in chat", async ({ page, request }) => {
  const who = await sessionFor(request, "grace");
  await clearAccount(request, who);
  try {
    await signIn(page, who);
    await page.goto("/talk/");
    await page.getByLabel("Message Luna").fill("A fictional busy afternoon has left my head feeling crowded.");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect(page.getByLabel("Message Luna")).toHaveValue("");
    await page.getByRole("button", { name: "Find another resource", exact: true }).click();
    const discovery = page.getByRole("region", { name: "Find another activity", exact: true });
    await discovery.getByLabel("Allow this general activity search with Brave and Luna").check();
    await discovery.getByLabel("General activity topic (optional)").fill("quiet meditation");
    const searched = page.waitForResponse((response) => response.url().endsWith("/discover") && response.request().method() === "POST");
    await discovery.getByRole("button", { name: "Search activities", exact: true }).click();
    const response = await searched;
    expect(response.status()).toBe(200);
    const firstRequest = response.request().postDataJSON();
    expect(firstRequest.constraints).toEqual({
      time_minutes: null, no_audio: false, no_video: false, seated: false, avoid_breath_focus: false,
    });
    expect(firstRequest).not.toHaveProperty("no_audio");
    await expect(discovery.getByRole("heading", { name: "Fictional reflection resource 1", exact: true })).toBeVisible();

    const feedback = discovery.getByLabel("What would fit better? Keep it general.");
    await expect(feedback).toHaveAttribute("maxlength", "160");
    // The UI's own example must pass the real public-topic validator.
    await feedback.fill((await feedback.getAttribute("placeholder"))!.replace("For example: ", ""));
    const refined = page.waitForResponse((value) => value.url().endsWith("/discover") && value.request().method() === "POST");
    await discovery.getByRole("button", { name: "Find different sources", exact: true }).click();
    const refinedResponse = await refined;
    expect(refinedResponse.status()).toBe(200);
    expect(refinedResponse.request().postDataJSON().excluded_urls).toEqual([
      "https://example.org/reflection-1", "https://example.org/reflection-2",
    ]);
    await expect(discovery.getByRole("heading", { name: "Fictional reflection resource 3", exact: true })).toBeVisible();
    await expect(discovery.getByRole("heading", { name: "Fictional reflection resource 1", exact: true })).toHaveCount(0);
    await discovery.getByRole("button", { name: "Save this activity", exact: true }).first().click();

    const panel = page.getByRole("region", { name: "Activity with Luna", exact: true });
    await expect(panel.getByRole("heading", { name: "Fictional reflection resource 3", exact: true })).toBeVisible();
    await panel.getByRole("button", { name: "Start activity", exact: true }).click();
    await expect(page.getByRole("region", { name: "Current activity", exact: true })
      .getByRole("button", { name: "Done / check in", exact: true })).toBeVisible();
    const chatId = new URL(page.url()).searchParams.get("c")!;
    expect((await getActivity(request, who, chatId)).status).toBe("active");
    const exported = await (await request.get("/v1/export", { headers: headers(who) })).json();
    expect(exported.activity_sessions).toHaveLength(1);
    expect(exported.activity_sessions[0].resource.url).toBe("https://example.org/reflection-3");
    expect(exported.outcomes).toEqual([]);
  } finally {
    await clearAccount(request, who);
  }
});

test("a newer recommendation replaces an unstarted saved search offer", async ({ page, request }) => {
  const who = await sessionFor(request, "grace");
  await clearAccount(request, who);
  try {
    const chat = await (await request.post("/v1/conversations", {
      headers: await currentDataRevisionHeaders(request, headers(who)), data: { llm_consent: true, retain_text: false },
    })).json();
    const discoveredResponse = await request.post(`/v1/conversations/${chat.id}/discover`, {
      headers: await currentDataRevisionHeaders(request, headers(who)), data: {
        expected_revision: chat.revision, llm_consent: true, original_query: "quiet meditation",
      },
    });
    expect(discoveredResponse.status()).toBe(200);
    const { offers } = await discoveredResponse.json();
    expect(offers.length).toBeGreaterThan(0);
    const createdResponse = await request.post(`/v1/conversations/${chat.id}/activity-sessions`, {
      headers: await currentDataRevisionHeaders(request, headers(who)), data: {
        client_request_id: crypto.randomUUID(), expected_conversation_revision: chat.revision,
        resource_id: offers[0].resource.id, resource_token: offers[0].resource_token,
      },
    });
    expect(createdResponse.ok()).toBeTruthy();
    const oldOffer = await createdResponse.json();
    expect(oldOffer.status).toBe("offered");
    await signIn(page, who);
    await page.goto(`/talk/?c=${chat.id}`);
    await expect(page.getByRole("heading", { name: offers[0].resource.title, exact: true })).toBeVisible();
    await page.getByLabel("Message Luna").fill("I have two minutes and would like a quiet seated pause, without audio.");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect(page.getByLabel("Message Luna")).toHaveValue("");
    const newOffer = page.getByRole("region", { name: "Activity with Luna", exact: true })
      .filter({ has: page.getByRole("heading", { name: "Two-minute quiet meditation", exact: true }) });
    await expect(newOffer.getByRole("button", { name: "Start activity", exact: true })).toBeVisible();
    expect((await getActivity(request, who, chat.id)).status).toBe("stopped");
    await newOffer.getByRole("button", { name: "Start activity", exact: true }).click();
    await expect(page.getByRole("region", { name: "Current activity", exact: true })
      .getByRole("button", { name: "Pause", exact: true })).toBeVisible();
    const activity = await getActivity(request, who, chat.id);
    expect(activity.id).not.toBe(oldOffer.id);
    expect(activity.status).toBe("active");
    expect(activity.remaining_seconds).toBeLessThanOrEqual(120);
    const oldState = await (await request.get(`/v1/activity-sessions/${oldOffer.id}`, { headers: headers(who) })).json();
    expect(oldState.status).toBe("stopped");
    expect(oldState.started_at).toBeNull();
    expect(oldState.report).toBeNull();
  } finally {
    await clearAccount(request, who);
  }
});
