import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import { currentDataRevisionHeaders } from "../helpers/data-revision";

async function setup(page: Page, request: APIRequestContext) {
  const session = await (await request.get("http://127.0.0.1:54321/test/session/erin")).json();
  const headers = { Authorization: `Bearer ${session.access_token}` };
  expect((await request.delete("/v1/account/data", { headers })).ok()).toBeTruthy();
  await page.addInitScript((who) => {
    window.localStorage.setItem("sb-127-auth-token", JSON.stringify({
      access_token: who.access_token, refresh_token: "integration-refresh", token_type: "bearer", expires_in: 86400,
      expires_at: Math.floor(Date.now() / 1000) + 86400,
      user: { id: who.user_id, aud: "authenticated", role: "authenticated", email: "erin@example.test" },
    }));
    window.localStorage.setItem(`journalpulse_preferences_v1:${who.user_id}`, JSON.stringify({ onboarded: false, llmConsent: false, retainText: false, locale: "CA", followUpMinutes: 10 }));
  }, session);
  return headers;
}

test("audit: draft recovery, optional Home and interrupted-save reconciliation", async ({ page, request }, info) => {
  await setup(page, request);
  const draft = "QA TEST audit repair — a fictional green kite.\nExact writing 🙂";
  await page.goto("/journal/"); await page.getByLabel("What would you like to remember?", { exact: true }).fill(draft);
  await page.getByRole("link", { name: "Home", exact: true }).click();
  await page.getByRole("link", { name: "Explore useful resources", exact: false }).click();
  await page.getByRole("link", { name: "Journal", exact: true }).click();
  await expect(page.getByLabel("What would you like to remember?", { exact: true })).toHaveValue(draft);
  await page.reload(); await expect(page.getByLabel("What would you like to remember?", { exact: true })).toHaveValue(draft);
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/v1/journal/entries", async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    await gate; const response = await route.fetch(); await route.fulfill({ response });
  });
  await page.getByRole("button", { name: "Save entry", exact: true }).click();
  await page.getByRole("link", { name: "Home", exact: true }).click();
  await page.getByRole("link", { name: "Explore useful resources", exact: false }).click();
  await page.getByRole("link", { name: "Journal", exact: true }).click();
  release();
  await expect(page.getByText(/Entry saved\./)).toBeVisible();
  await expect(page.locator(".journal-entry-link").filter({ hasText: "fictional green kite" })).toBeVisible();
  await page.screenshot({ path: info.outputPath("draft-save-recovery.png"), fullPage: true });
  await page.getByRole("link", { name: "Home", exact: true }).click();
  await expect(page).toHaveURL(/\/$/); await expect(page.getByText(/private defaults/)).toBeVisible();
});

test("stale application chunks in an old worker cache cannot replace the current runtime", async ({ page, request }) => {
  await setup(page, request);
  await page.goto("/journal/");
  await expect(page.getByLabel("What would you like to remember?", { exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
    const current = [...document.querySelectorAll<HTMLScriptElement>('script[src]')].find((script) => script.src.includes('/_next/static/chunks/'))!;
    const old = await caches.open('journalpulse-shell-v5');
    await old.put(current.src, new Response('throw new Error("QA stale runtime must never execute");', { headers: { 'Content-Type': 'application/javascript' } }));
  });
  await page.reload();
  await expect(page.getByLabel("What would you like to remember?", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Oops, Luna tripped." })).toHaveCount(0);
});

test("audit: accepted legacy activity and timer recover through refresh", async ({ page, request }, info) => {
  const headers = await setup(page, request);
  const revisionHeaders = await currentDataRevisionHeaders(request, headers);
  const chat = await (await request.post("/v1/conversations", { headers: revisionHeaders, data: { llm_consent: false, retain_text: false, locale: "CA" } })).json();
  const turn = await (await request.post(`/v1/conversations/${chat.id}/messages`, {
    headers: revisionHeaders, data: { client_message_id: crypto.randomUUID(), text: "QA TEST fictional stress", goal: "settle", confirmed_feelings: ["stressed"] },
  })).json();
  const card = turn.conversation.card;
  const savedResponse = await request.post(`/v1/conversations/${chat.id}/accept`, {
    headers: revisionHeaders, data: { client_request_id: crypto.randomUUID(), action_id: card.actions[0].id, expected_revision: turn.conversation.revision },
  });
  expect(savedResponse.status()).toBe(201);
  const saved = await savedResponse.json();
  await page.goto(`/talk/?c=${chat.id}`);
  await expect(page.getByRole("heading", { name: "Nice choice." })).toBeVisible();
  await expect(page.getByRole("link", { name: "I’m done, check in now" })).toHaveAttribute("href", `/check-in/?decision=${saved.decision.decision_id}`);
  await page.getByRole("button", { name: /Start a .*minute timer/ }).click();
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeVisible();
  await page.reload(); await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.reload(); await expect(page.getByRole("button", { name: "Keep going", exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath("accepted-timer-recovery.png"), fullPage: true });
  await expect(page.getByLabel("Message Luna")).toHaveCount(0);
  await page.getByRole("link", { name: "I’m done, check in now", exact: true }).click();
  await page.getByRole("button", { name: "Not yet", exact: true }).click();
  const outcome = page.waitForResponse((response) => response.url().endsWith("/v1/outcomes") && response.request().method() === "POST");
  const clickAt = Date.now();
  await page.getByRole("button", { name: "Skip this one", exact: true }).click();
  const response = await outcome;
  await expect(page.getByRole("heading", { name: "Thank you!" })).toBeVisible();
  const timing = response.request().timing();
  const app = Number(/app;dur=([\d.]+)/.exec(response.headers()["server-timing"] ?? "")?.[1] ?? NaN);
  const measurements = { environment: "local PostgreSQL; auth stand-in", total_to_visible_confirmation_ms: Date.now() - clickAt,
    revision_and_auth_before_write_ms: Math.max(0, timing.startTime - clickAt),
    request_to_headers_ms: timing.responseStart, app_to_headers_ms: Number.isFinite(app) ? app : null,
    transport_and_scheduling_ms: Number.isFinite(app) ? Math.max(0, timing.responseStart - app) : null };
  await info.attach("check-in-timing", { body: JSON.stringify(measurements, null, 2), contentType: "application/json" });
  console.log(JSON.stringify({ check_in_timing: measurements }));
});
