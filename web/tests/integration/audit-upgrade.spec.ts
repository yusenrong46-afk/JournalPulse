// Real UI/API/database; auth issuance and AI/search are local test doubles.
import { expect, test } from "@playwright/test";
import { currentDataRevisionHeaders } from "../helpers/data-revision";

test("an account change on the same chat URL discards private replies, drafts and inherited consent", async ({ page, request }) => {
  const alice = await (await request.get("http://127.0.0.1:54321/test/session/alex")).json();
  const bob = await (await request.get("http://127.0.0.1:54321/test/session/blair")).json();
  for (const who of [alice, bob]) {
    expect((await request.delete("/v1/account/data", { headers: { Authorization: `Bearer ${who.access_token}` } })).ok()).toBeTruthy();
  }
  try {
    await page.goto("/login/");
    await page.evaluate((who) => {
      window.localStorage.setItem(`journalpulse_preferences_v1:${who.user_id}`, JSON.stringify({
        onboarded: true, llmConsent: true, retainText: false, encryptedDrafts: false, followUpMinutes: 10, locale: "CA",
      }));
      window.localStorage.setItem("sb-127-auth-token", JSON.stringify({
        access_token: who.access_token, refresh_token: "integration-refresh", token_type: "bearer",
        expires_in: 86400, expires_at: Math.floor(Date.now() / 1000) + 86400,
        user: { id: who.user_id, aud: "authenticated", role: "authenticated" },
      }));
    }, alice);
    await page.goto("/talk/");
    await page.getByLabel("Message Luna").fill("ALICE_PRIVATE_SENT_WRITING: a fictional meeting left me tired.");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect(page.getByLabel("Message Luna")).toHaveValue("");
    await expect(page.getByText(/ALICE_PRIVATE_SENT_WRITING/)).toBeVisible();
    await page.getByLabel("Message Luna").fill("ALICE_PRIVATE_UNSENT_DRAFT");
    const sameChatUrl = page.url();
    await page.evaluate((who) => {
      const nextSession = {
        access_token: who.access_token, refresh_token: "integration-refresh", token_type: "bearer",
        expires_in: 86400, expires_at: Math.floor(Date.now() / 1000) + 86400,
        user: { id: who.user_id, aud: "authenticated", role: "authenticated" },
      };
      window.localStorage.setItem("sb-127-auth-token", JSON.stringify(nextSession));
      // This is the same event supabase-js sends when another tab signs in. The
      // real SDK listener and auth boundary handle it; the UI is not mocked.
      const channel = new BroadcastChannel("sb-127-auth-token");
      channel.postMessage({ event: "SIGNED_IN", session: nextSession });
      channel.close();
    }, bob);
    await expect(page.getByLabel("Message Luna")).toHaveValue("");
    await expect(page.getByText(/ALICE_PRIVATE_SENT_WRITING/)).toHaveCount(0);
    expect(page.url()).toBe(sameChatUrl);
    await page.getByRole("button", { name: "Start a new chat", exact: true }).click();
    await page.getByLabel("Message Luna").fill("Bob's fictional thought for a new guided chat.");
    const created = page.waitForRequest((req) => req.url().endsWith("/v1/conversations") && req.method() === "POST");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    const outgoing = (await created).postDataJSON();
    expect(outgoing.llm_consent).toBe(false);
    expect(outgoing.retain_text).toBe(false);
    await expect(page.getByLabel("Message Luna")).toHaveValue("");
  } finally {
    // These users are shared with the RLS suite; restore its empty-account fixture
    // even when the browser assertions fail.
    for (const who of [alice, bob]) {
      expect((await request.delete("/v1/account/data", {
        headers: { Authorization: `Bearer ${who.access_token}` },
      })).ok()).toBeTruthy();
    }
  }
});

test("a linked chat opens an editable general search and returns with its source", async ({ page, request }) => {
  const session = await (await request.get("http://127.0.0.1:54321/test/session/frank")).json();
  const headers = { Authorization: `Bearer ${session.access_token}` };
  expect((await request.delete("/v1/account/data", { headers })).ok()).toBeTruthy();
  await page.addInitScript((who) => {
    window.localStorage.setItem(`journalpulse_preferences_v1:${who.user_id}`, JSON.stringify({
      onboarded: true, llmConsent: true, retainText: false, encryptedDrafts: false, followUpMinutes: 10, locale: "CA",
    }));
    window.localStorage.setItem("sb-127-auth-token", JSON.stringify({
      access_token: who.access_token, refresh_token: "integration-refresh", token_type: "bearer",
      expires_in: 86400, expires_at: Math.floor(Date.now() / 1000) + 86400,
      user: { id: who.user_id, aud: "authenticated", role: "authenticated" },
    }));
  }, session);
  const saved = await request.post("/v1/journal/entries", { headers: await currentDataRevisionHeaders(request, headers), data: { text: "PRIVATE_FICTIONAL_SOURCE for a busy afternoon." } });
  expect(saved.status()).toBe(201);
  const entry = await saved.json();
  await page.goto(`/talk/?entry=${entry.id}`);
  await page.getByRole("button", { name: "Use this entry in a new AI chat" }).click();
  await page.getByLabel("Message Luna").fill("A fictional thought I only want to explore.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  const chat = new URL(page.url()).searchParams.get("c");
  expect(chat).toBeTruthy();
  await expect(page.getByRole("button", { name: "Yes, let’s find one small thing" })).toHaveCount(0);
  // AI chats search inline now; the separate library is reached from Home and must still
  // return to the linked chat without carrying its source or chat identity.
  await page.goto("/");
  await page.getByRole("link", { name: /Explore useful resources/ }).click();
  await expect(page.getByLabel("What would you like to explore?")).toHaveValue("");
  await expect(page.getByRole("checkbox", { name: /Share this topic/ })).not.toBeChecked();
  expect(page.url()).not.toContain(entry.id);
  expect(page.url()).not.toContain(chat!);
  expect(await page.locator("body").innerText()).not.toContain("PRIVATE_FICTIONAL_SOURCE");
  await page.getByLabel("What would you like to explore?").fill("brief grounding guides");
  await page.getByRole("checkbox", { name: /Share this topic/ }).check();
  const requested = page.waitForRequest((req) => req.url().endsWith("/v1/discovery/search") && req.method() === "POST");
  await page.getByRole("button", { name: "Search this topic", exact: true }).click();
  const payload = (await requested).postDataJSON();
  expect(payload.original_query).toBe("brief grounding guides");
  expect(JSON.stringify(payload)).not.toContain(entry.id);
  expect(JSON.stringify(payload)).not.toContain(chat!);
  expect(payload).not.toHaveProperty("journal_text");
  await expect(page.getByRole("link", { name: "Fictional reflection resource 1", exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Return to chat", exact: true }).click();
  await expect(page.getByRole("link", { name: "Open entry", exact: true })).toHaveAttribute("href", `/journal/?entry=${entry.id}`);
  expect((await request.delete(`/v1/journal/entries/${entry.id}`, { headers })).status()).toBe(204);
});

test("retry after a committed turn loses its response keeps one stored turn", async ({ page, request }) => {
  const session = await (await request.get("http://127.0.0.1:54321/test/session/grace")).json();
  const headers = { Authorization: `Bearer ${session.access_token}` };
  expect((await request.delete("/v1/account/data", { headers })).ok()).toBeTruthy();
  const created = await request.post("/v1/conversations", { headers: await currentDataRevisionHeaders(request, headers), data: { llm_consent: true, retain_text: false } });
  expect(created.status()).toBe(201);
  const chat = await created.json();
  await page.addInitScript((who) => {
    window.localStorage.setItem(`journalpulse_preferences_v1:${who.user_id}`, JSON.stringify({
      onboarded: true, llmConsent: true, retainText: false, encryptedDrafts: false, followUpMinutes: 10, locale: "CA",
    }));
    window.localStorage.setItem("sb-127-auth-token", JSON.stringify({
      access_token: who.access_token, refresh_token: "integration-refresh", token_type: "bearer",
      expires_in: 86400, expires_at: Math.floor(Date.now() / 1000) + 86400,
      user: { id: who.user_id, aud: "authenticated", role: "authenticated" },
    }));
  }, session);
  await page.goto(`/talk/?c=${chat.id}`);
  await expect(page.getByLabel("Message Luna")).toBeEnabled();
  const ids: string[] = [];
  await page.route(`**/v1/conversations/${chat.id}/messages`, async (route) => {
    ids.push(route.request().postDataJSON().client_message_id);
    if (ids.length <= 2) {
      // The actual signed database write commits before the client loses the response.
      expect((await route.fetch()).status()).toBe(200);
      await route.abort("failed");
    } else await route.continue();
  });
  await page.route(`**/v1/conversations/${chat.id}`, (route) => route.fulfill({
    status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Temporary recovery-read outage" }),
  }));
  await page.getByLabel("Message Luna").fill("A fictional thought whose response gets lost.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("button", { name: "Try again", exact: true })).toBeVisible();
  expect(ids).toHaveLength(2);
  await page.unroute(`**/v1/conversations/${chat.id}`);
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  expect(ids).toHaveLength(3);
  expect(new Set(ids).size).toBe(1);
  const stored = await (await request.get(`/v1/conversations/${chat.id}`, { headers })).json();
  expect(stored.messages).toHaveLength(2);
  await expect(page.locator(".from-me .bubble").filter({ hasText: "A fictional thought whose response gets lost." })).toHaveCount(1);
  expect((await request.delete(`/v1/conversations/${chat.id}`, { headers })).status()).toBe(204);
});
