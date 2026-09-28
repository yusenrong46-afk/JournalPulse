// Integrated end-to-end tests: nothing below is mocked in the browser. Requests go to the
// real FastAPI app, which talks to PostgREST and PostgreSQL with every migration applied.
// The only stand-ins are the auth token issuer and a deterministic model provider.
import { type APIRequestContext, expect, type Page, test } from "@playwright/test";

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
        "journalpulse_preferences_v1",
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
    post: (path: string, data?: unknown) => request.post(path, { headers, data }),
    delete: (path: string) => request.delete(path, { headers }),
  };
}

test("the whole loop runs against the real API and database", async ({ page, request }) => {
  const alex = await session(request, "alex");
  const as = api(request, alex);
  expect((await as.delete("/v1/account/data")).ok()).toBeTruthy();
  await signIn(page, alex);

  await page.goto("/talk/");
  await page.getByRole("button", { name: /Low/ }).click();
  await expect(page.getByText("What feels heaviest right now?")).toBeVisible();
  await page.getByLabel("Message Luna").fill("Work has been so stressful and I'm exhausted.");
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
  expect(saved.model_run.model).toBe("integration-fake-luna");
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
  await expect(page.getByText("helped 1 of 1")).toBeVisible();
  await expect(page.getByText("😢 Sad")).toBeVisible();

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
