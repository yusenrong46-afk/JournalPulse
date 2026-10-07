// Integrated end-to-end tests: nothing below is mocked in the browser. Requests go to the
// real FastAPI app, which talks to PostgREST and PostgreSQL with every migration applied.
// The only stand-ins are the auth token issuer and a deterministic model provider.
import { type APIRequestContext, expect, type Page, test } from "@playwright/test";
import { currentDataRevisionHeaders } from "../helpers/data-revision";

const GATEWAY = "http://127.0.0.1:54321";

type Session = { user_id: string; access_token: string };

async function session(request: APIRequestContext, name: string): Promise<Session> {
  const response = await request.get(`${GATEWAY}/test/session/${name}`);
  expect(response.ok()).toBeTruthy();
  return response.json();
}

async function signIn(page: Page, who: Session, llmConsent = true) {
  await page.addInitScript(
    ([token, userId, consent]) => {
      window.localStorage.setItem(
        `journalpulse_preferences_v1:${userId}`,
        JSON.stringify({ onboarded: true, llmConsent: consent, retainText: false, encryptedDrafts: false, followUpMinutes: 10, locale: "CA" }),
      );
      // supabase-js reads this key for the gateway host 127.0.0.1.
      window.localStorage.setItem(
        "sb-127-auth-token",
        JSON.stringify({
          access_token: token,
          refresh_token: "integration-refresh",
          token_type: "bearer",
          expires_in: 86400,
          expires_at: Math.floor(Date.now() / 1000) + 86400,
          user: { id: userId, aud: "authenticated", role: "authenticated", email: `${userId}@example.test` },
        }),
      );
    },
    [who.access_token, who.user_id, llmConsent] as const,
  );
}

function api(request: APIRequestContext, who: Session) {
  const headers = { Authorization: `Bearer ${who.access_token}` };
  return {
    get: (path: string) => request.get(path, { headers }),
    post: async (path: string, data?: unknown) => request.post(path, { headers: await currentDataRevisionHeaders(request, headers), data }),
    delete: (path: string) => request.delete(path, { headers }),
  };
}

test("the legacy guided loop runs against the real API and database", async ({ page, request }) => {
  const alex = await session(request, "alex");
  const as = api(request, alex);
  expect((await as.delete("/v1/account/data")).ok()).toBeTruthy();
  // This preserves the original feelings → goal → accept → standalone check-in
  // contract. AI chats use the separately tested, open-chat activity session loop.
  await signIn(page, alex, false);

  await page.goto("/talk/");
  await page.getByRole("button", { name: /Low/ }).click();
  await expect(page.getByText(/What part of that is sitting with you most right now/)).toBeVisible();
  await page.getByLabel("Message Luna").fill("Work has been so stressful and I'm exhausted. Please find one small step.");
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByRole("button", { name: /Yes, let’s find one small thing/ }).click();

  const feelings = page.getByRole("group", { name: "Feelings" });
  await expect(feelings.getByRole("button", { name: /Tired/ })).toHaveAttribute("aria-pressed", "true");
  await expect(feelings.getByRole("button", { name: /Stressed/ })).toHaveAttribute("aria-pressed", "true");
  await feelings.getByRole("button", { name: /Stressed/ }).click();
  await feelings.getByRole("button", { name: /Sad/ }).click();
  await page.getByRole("button", { name: "That’s it" }).click();
  await page.getByRole("button", { name: /Calm down/ }).click();
  await expect(page.getByRole("button", { name: "Let’s try it" })).toBeVisible();

  // Reload before choosing: the person's corrections must come back from the server.
  const conversationId = new URL(page.url()).searchParams.get("c");
  expect(conversationId).toBeTruthy();
  await page.reload();
  await expect(page.getByRole("button", { name: "Let’s try it" })).toBeVisible();
  const reloaded = await (await as.get(`/v1/conversations/${conversationId}`)).json();
  expect(reloaded.conversation.confirmed_feelings).toEqual(["tired", "sad"]);
  expect(reloaded.conversation.reported_mood).toBe(2);
  expect(reloaded.conversation.feelings).toEqual(["tired", "stressed"]);

  await page.getByRole("button", { name: "Let’s try it" }).click();
  await expect(page.getByRole("heading", { name: "Nice choice." })).toBeVisible();

  const history = await (await as.get("/v1/reflections")).json();
  expect(history.items).toHaveLength(1);
  const saved = history.items[0];
  expect(saved.self_report_input).toEqual({ feelings: ["tired", "sad"], mood_score: 2 });
  expect(saved.state.emotion_tags).toEqual(["tired", "sad"]);
  expect(saved.state.confidence).toBeNull();
  expect(saved.model_run.model).toBe("luna-guided");
  expect(saved.model_run.provider).toBe("local");
  const closed = await (await as.get(`/v1/conversations/${conversationId}`)).json();
  expect(closed.conversation.status).toBe("closed");
  expect(closed.conversation.reflection_id).toBe(saved.id);
  expect(closed.messages.every((item: { content: string | null }) => item.content === null)).toBe(true);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Did it help/ })).toBeVisible();
  await page.getByRole("link", { name: "Helped", exact: true }).click();
  await page.getByRole("button", { name: /Calm$/ }).click();
  await page.getByRole("button", { name: "Save my check-in" }).click();
  await expect(page.getByRole("heading", { name: "Thank you!" })).toBeVisible();

  await page.goto("/journey/");
  await expect(page.getByText("1 rating · 1 rated helpful", { exact: false })).toBeVisible();
  await expect(page.locator(".entry .chips").getByText("Sad", { exact: true })).toBeVisible();

  await page.goto("/me/");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download my data" }).click();
  const file = await (await download).path();
  const exported = JSON.parse(await (await import("node:fs/promises")).readFile(file, "utf8"));
  expect(exported.reflections).toHaveLength(1);
  expect(exported.outcomes).toHaveLength(1);
  expect(exported.outcomes[0].helpfulness).toBe(4);
  expect(exported.policy_decisions).toHaveLength(1);
  expect(exported.conversation_messages.every((item: { content: string | null }) => item.content === null)).toBe(true);

  // Deletion is a disclosed, deliberate step; open it before confirming.
  await page.getByText("Delete saved data", { exact: true }).click();
  await page.getByLabel(/Type “delete my journal” to confirm/).fill("delete my journal");
  await page.getByRole("button", { name: "Delete my journal" }).click();
  await expect(page.getByText(/saved items were deleted/)).toBeVisible();
  const after = await (await as.get("/v1/export")).json();
  for (const key of ["reflections", "outcomes", "conversations", "conversation_messages", "policy_decisions", "model_runs"]) {
    expect(after[key], key).toEqual([]);
  }
});

test("each person sees only their own journal and cannot write provenance directly", async ({ request }) => {
  const alex = api(request, await session(request, "alex"));
  const blairSession = await session(request, "blair");
  const blair = api(request, blairSession);
  const chat = await (await alex.post("/v1/conversations", { llm_consent: true, retain_text: false })).json();
  await alex.post(`/v1/conversations/${chat.id}/messages`, { client_message_id: crypto.randomUUID(), text: "Alex only." });

  expect((await blair.get(`/v1/conversations/${chat.id}`)).status()).toBe(404);
  expect((await blair.post(`/v1/conversations/${chat.id}/close`)).status()).toBe(404);
  expect((await (await blair.get("/v1/export")).json()).conversations).toEqual([]);

  const direct = { Authorization: `Bearer ${blairSession.access_token}`, "Content-Type": "application/json" };
  const readAlex = await request.get(`${GATEWAY}/rest/v1/conversation_messages?select=record`, { headers: direct });
  expect(await readAlex.json()).toEqual([]);
  const forged = await request.post(`${GATEWAY}/rest/v1/model_runs`, {
    headers: direct,
    data: { user_id: blairSession.user_id, model: "forged", provider: "me", latency_ms: 0, schema_valid: true },
  });
  expect(forged.status()).toBe(403);
  const unsigned = await request.post(`${GATEWAY}/rest/v1/rpc/jp_save_reflection`, {
    headers: direct,
    data: { payload: JSON.stringify({ purpose: "save_reflection", user_id: blairSession.user_id }), signature: "0".repeat(64) },
  });
  expect(unsigned.status()).toBe(403);
  await alex.delete(`/v1/conversations/${chat.id}`);
});

test("Just talk survives reload and later AI flags, then resumes with a fresh activity offer", async ({ page, request }) => {
  const who = await session(request, "alex");
  const as = api(request, who);
  await as.delete("/v1/account/data");
  await signIn(page, who);
  await page.goto("/talk/");
  await page.getByLabel("Message Luna").fill("I feel tired.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("What feels heaviest right now?")).toBeVisible();
  await page.getByLabel("Message Luna").fill("I have two minutes and would like a quiet seated pause without audio.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByRole("button", { name: "Start activity", exact: true })).toBeVisible();
  const id = new URL(page.url()).searchParams.get("c");
  const old = (await (await as.get(`/v1/conversations/${id}`)).json()).conversation;
  expect(old.activity_card).toBeTruthy();
  expect(old.card).toBeNull();

  await page.getByRole("button", { name: "Just talk", exact: true }).click();
  await expect(page.getByText("Just talking. We’ll stay with your thoughts.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Start activity", exact: true })).toHaveCount(0);
  await page.reload();
  await expect(page.getByRole("button", { name: "Find a small step" })).toBeVisible();
  // The old-contract fake deliberately sets offer_action on later turns. The
  // server's Listen preference must suppress even that adversarial readiness flag.
  await page.getByLabel("Message Luna").fill("I still have more to say.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  const listening = (await (await as.get(`/v1/conversations/${id}`)).json()).conversation;
  expect(listening.interaction_preference).toBe("listen");
  expect(listening.ready_for_action).toBe(false);
  expect(listening.card).toBeNull();
  expect(listening.activity_card).toBeNull();
  expect((await (await as.get("/v1/reflections")).json()).items).toEqual([]);
  await expect(page.getByRole("button", { name: /Yes, let’s find one small thing/ })).toHaveCount(0);

  await page.getByRole("button", { name: "Find a small step" }).click();
  await page.getByLabel("Message Luna").fill("I have two minutes and want a quiet seated pause now.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByRole("button", { name: "Start activity", exact: true })).toBeVisible();
  const resumed = (await (await as.get(`/v1/conversations/${id}`)).json()).conversation;
  expect(resumed.revision).toBeGreaterThan(old.revision);
  expect(resumed.activity_card.offered_message_id).not.toBe(old.activity_card.offered_message_id);
  const stale = await as.post(`/v1/conversations/${id}/activity-sessions`, {
    client_request_id: crypto.randomUUID(), resource_id: old.activity_card.decision_preview.action_id,
    expected_conversation_revision: old.revision,
  });
  expect(stale.status()).toBe(409);
  await page.getByRole("button", { name: "Start activity", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeVisible();
  const current = (await (await as.get(`/v1/conversations/${id}`)).json()).conversation;
  expect(current.status).toBe("open");
  expect((await (await as.get("/v1/reflections")).json()).items).toEqual([]);
  const activity = await (await as.get(`/v1/conversations/${id}/activity-sessions`)).json();
  expect(activity.status).toBe("active");
  expect(activity.offered_message_id).toBe(resumed.activity_card.offered_message_id);
  await as.delete(`/v1/conversations/${id}`);
});

test("a lost choice response can be retried without losing the draft or applying twice", async ({ page, request }) => {
  const who = await session(request, "blair");
  const as = api(request, who);
  await as.delete("/v1/account/data");
  await signIn(page, who, false);
  await page.goto("/talk/");
  await page.getByLabel("Message Luna").fill("Starting a guided conversation.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  await page.getByLabel("Message Luna").fill("Words I am still writing.");
  const commands: unknown[] = [];
  await page.route("**/v1/conversations/*/preference", async route => {
    commands.push(route.request().postDataJSON());
    const response = await route.fetch();
    expect(response.ok()).toBeTruthy();
    if (commands.length <= 2) await route.abort("failed");
    else await route.fulfill({ response });
  });
  await page.getByRole("button", { name: "Just talk", exact: true }).click();
  await expect(page.getByRole("button", { name: "Try saving choice again" })).toBeVisible();
  await expect(page.getByLabel("Message Luna")).toHaveValue("Words I am still writing.");
  await page.getByRole("button", { name: "Try saving choice again" }).click();
  await expect(page.getByRole("button", { name: "Find a small step" })).toBeVisible();
  expect(commands).toHaveLength(3); // Automatic retry, then the person's explicit retry.
  expect(commands[0]).toEqual(commands[1]);
  expect(commands[0]).toEqual(commands[2]);
  const id = new URL(page.url()).searchParams.get("c");
  const stored = (await (await as.get(`/v1/conversations/${id}`)).json()).conversation;
  expect(stored.revision).toBe(2); // One turn and one choice, despite two HTTP attempts.
  const exported = await (await as.get("/v1/export")).json();
  expect(exported.conversation_preference_requests).toHaveLength(1);
  expect(exported.reflections).toEqual([]);
  await as.delete(`/v1/conversations/${id}`);
  expect((await (await as.get("/v1/export")).json()).conversation_preference_requests).toEqual([]);
});

test("a delayed browser response cannot replace a newer listening choice", async ({ page, request }) => {
  const who = await session(request, "blair");
  const as = api(request, who);
  await as.delete("/v1/account/data");
  expect((await (await as.get("/v1/export")).json()).conversation_preference_requests).toEqual([]);
  await signIn(page, who, false);
  await page.goto("/talk/");
  await page.getByLabel("Message Luna").fill("My first thought.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");

  let release!: () => void;
  let reached!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  const arrived = new Promise<void>(resolve => { reached = resolve; });
  await page.route("**/v1/conversations/*/messages", async route => {
    const response = await route.fetch(); // Real commit; delay only HTTP delivery.
    expect(response.ok()).toBeTruthy();
    reached();
    await held;
    await route.fulfill({ response });
  });
  await page.getByLabel("Message Luna").fill("My second thought.");
  await page.getByRole("button", { name: "Send" }).click();
  await arrived;
  try {
    // The first choice uses the old revision; the page must refresh before retry.
    await page.getByRole("button", { name: "Just talk", exact: true }).click();
    await expect(page.locator("p[role=alert]")).toBeVisible();
    await page.getByRole("button", { name: "Just talk", exact: true }).click();
    await expect(page.getByText("Just talking. We’ll stay with your thoughts.")).toBeVisible();
  } finally {
    release();
  }
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  await expect(page.getByRole("button", { name: "Find a small step" })).toBeEnabled();
  await expect(page.getByRole("button", { name: /Yes, let’s find one small thing/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Skip ahead/ })).toHaveCount(0);
  const id = new URL(page.url()).searchParams.get("c");
  const stored = await (await as.get(`/v1/conversations/${id}`)).json();
  expect(stored.conversation.interaction_preference).toBe("listen");
  expect(stored.conversation.revision).toBe(3);
  expect(stored.messages).toHaveLength(4);
});

test("idle chat text is purged by the scheduled job even if the person never returns", async ({ request }) => {
  const casey = api(request, await session(request, "casey"));
  const chat = await (await casey.post("/v1/conversations", { llm_consent: true, retain_text: false })).json();
  await casey.post(`/v1/conversations/${chat.id}/messages`, { client_message_id: crypto.randomUUID(), text: "Private words." });
  await request.post(`${GATEWAY}/test/age/${chat.id}`);
  const purge = await (await request.post(`${GATEWAY}/test/purge`)).json();
  expect(purge.closed).toBeGreaterThanOrEqual(1);
  expect(purge.purged_messages).toBeGreaterThanOrEqual(2);
  const stored = await (await casey.get(`/v1/conversations/${chat.id}`)).json();
  expect(stored.conversation.status).toBe("closed");
  expect(stored.messages.every((item: { content: string | null }) => item.content === null)).toBe(true);
  const late = await casey.post(`/v1/conversations/${chat.id}/messages`, { client_message_id: crypto.randomUUID(), text: "Still here?" });
  expect(late.status()).toBe(409);
});

test("a reply to a chat closed in another tab is refused by the database", async ({ page, request }) => {
  const caseySession = await session(request, "casey");
  const casey = api(request, caseySession);
  await signIn(page, caseySession);
  await page.goto("/talk/");
  await page.getByLabel("Message Luna").fill("Starting a chat.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("What feels heaviest right now?")).toBeVisible();
  const conversationId = new URL(page.url()).searchParams.get("c");
  expect((await casey.post(`/v1/conversations/${conversationId}/close`)).ok()).toBeTruthy();
  await page.getByLabel("Message Luna").fill("One more thing.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.locator("p[role=alert]")).toHaveText("This conversation is closed.");
  await expect(page.getByRole("button", { name: "Start a new chat" })).toBeVisible();
  const stored = await (await casey.get(`/v1/conversations/${conversationId}`)).json();
  expect(stored.messages).toHaveLength(2);
});

test("the generation limit is enforced by the database, not the browser", async ({ request }) => {
  const dana = api(request, await session(request, "dana"));
  const chat = await (await dana.post("/v1/conversations", { llm_consent: true, retain_text: false })).json();
  const statuses: number[] = [];
  for (let index = 0; index < 13; index += 1) {
    const response = await dana.post(`/v1/conversations/${chat.id}/messages`, {
      client_message_id: crypto.randomUUID(),
      text: `Message ${index}`,
    });
    statuses.push(response.status());
    if (response.status() === 429) {
      const retryAfter = Number(response.headers()["retry-after"]);
      expect(retryAfter).toBeGreaterThanOrEqual(1);
      expect(retryAfter).toBeLessThanOrEqual(60);
    }
  }
  expect(statuses.slice(0, 12).every((status) => status === 200)).toBe(true);
  expect(statuses[12]).toBe(429);
});
