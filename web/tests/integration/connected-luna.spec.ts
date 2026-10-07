// Real UI -> API -> PostgREST -> PostgreSQL. Auth issuance, Luna and search are
// explicit local test doubles; passing these checks does not establish AI quality.
import { type APIRequestContext, expect, type Page, test } from "@playwright/test";
import { currentDataRevisionHeaders } from "../helpers/data-revision";

type Session = { user_id: string; access_token: string };

async function setup(page: Page, request: APIRequestContext) {
  const response = await request.get("http://127.0.0.1:54321/test/session/erin");
  expect(response.ok()).toBeTruthy();
  const who: Session = await response.json();
  const headers = { Authorization: `Bearer ${who.access_token}` };
  expect((await request.delete("/v1/account/data", { headers })).ok()).toBeTruthy();
  await page.addInitScript((session: Session) => {
    window.localStorage.setItem(`journalpulse_preferences_v1:${session.user_id}`, JSON.stringify({
      onboarded: true, llmConsent: true, retainText: false, encryptedDrafts: false,
      followUpMinutes: 10, locale: "CA",
    }));
    window.localStorage.setItem("sb-127-auth-token", JSON.stringify({
      access_token: session.access_token, refresh_token: "integration-refresh",
      token_type: "bearer", expires_in: 86400,
      expires_at: Math.floor(Date.now() / 1000) + 86400,
      user: { id: session.user_id, aud: "authenticated", role: "authenticated", email: "erin@example.test" },
    }));
  }, who);
  return headers;
}

test("slice 1: save, reopen, reflect and delete standalone writing", async ({ page, request }, testInfo) => {
  const headers = await setup(page, request);
  const writing = "A fictional cancelled plan left me disappointed.\nI wanted time with my friend.";
  await page.goto("/journal/");
  await page.getByLabel("What would you like to remember?").fill(writing);
  await expect(page.getByRole("button", { name: "Save entry", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Save entry", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Your saved entry" })).toBeVisible();
  await expect(page.getByText("Saved writing", { exact: true })).toBeVisible();
  await expect(page.getByText("This reply disappears when you leave this entry or reload.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Reflect on this entry", exact: true })).toBeDisabled();
  const id = new URL(page.url()).searchParams.get("entry");
  expect(id).toBeTruthy();
  const exported = await (await request.get("/v1/export", { headers })).json();
  expect(exported.journal_entries).toHaveLength(1);
  expect(exported.journal_entries[0].text).toBe(writing);
  expect(exported.reflections).toEqual([]);
  await page.reload();
  await expect(page.locator(".journal-entry-text").first()).toHaveText(writing);
  await page.getByRole("checkbox", { name: /Allow this entry to be sent/ }).check();
  await page.getByRole("button", { name: "Reflect on this entry" }).click();
  await expect(page.getByText("Luna’s reflection", { exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("slice-1-journal-mobile.png"), fullPage: true });
  await page.reload();
  await expect(page.locator(".journal-entry-text").first()).toHaveText(writing);
  await expect(page.getByText("Luna’s reflection", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("checkbox", { name: /Allow this entry to be sent/ })).not.toBeChecked();
  await expect(page.getByRole("button", { name: "Reflect on this entry", exact: true })).toBeDisabled();
  await expect(page.getByRole("link", { name: "Discuss with Luna", exact: true })).toHaveAttribute("href", `/talk/?entry=${id}`);
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete entry", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Your saved entry" })).toHaveCount(0);
  expect((await request.get(`/v1/journal/entries/${id}`, { headers })).status()).toBe(404);
});

test("slice 2: selected journal context survives chat reload and source deletion invalidates it", async ({ page, request }, testInfo) => {
  const headers = await setup(page, request);
  const saved = await request.post("/v1/journal/entries", {
    headers: await currentDataRevisionHeaders(request, headers), data: { text: "A fictional meeting left me stressed and tired." },
  });
  expect(saved.status()).toBe(201);
  const entry = await saved.json();
  await page.goto(`/journal/?entry=${entry.id}`);
  await page.getByRole("link", { name: "Discuss with Luna" }).click();
  await page.getByRole("button", { name: "Use this entry in a new AI chat" }).click();
  await page.getByLabel("Message Luna").fill("What might I want to explore about that meeting?");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
  await expect(page.getByText("Using your selected journal entry", { exact: false })).toBeVisible();
  await expect(page.getByRole("link", { name: "Open entry", exact: true })).toHaveAttribute("href", `/journal/?entry=${entry.id}`);
  const chatId = new URL(page.url()).searchParams.get("c");
  expect(chatId).toBeTruthy();
  const chat = await (await request.get(`/v1/conversations/${chatId}`, { headers })).json();
  expect(chat.conversation.source_entry_id).toBe(entry.id);
  expect(chat.messages.some((item: { content: string | null }) => item.content === entry.text)).toBe(false);
  await page.reload();
  await expect(page.getByRole("link", { name: "Open entry", exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("slice-2-chat-mobile.png"), fullPage: true });
  expect((await request.delete(`/v1/journal/entries/${entry.id}`, { headers })).status()).toBe(204);
  expect((await request.get(`/v1/conversations/${chatId}`, { headers })).status()).toBe(404);
  await page.reload();
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
});

test("slice 3: approve a topic and refine results without repeating sources", async ({ page, request }, testInfo) => {
  await setup(page, request);
  await page.goto("/discover/");
  await page.getByLabel("What would you like to explore?").fill("Understanding overthinking");
  await expect(page.getByRole("button", { name: "Search this topic" })).toBeDisabled();
  await page.getByRole("checkbox", { name: /Share this topic/ }).check();
  await page.getByRole("button", { name: "Search this topic" }).click();
  await expect(page.getByRole("link", { name: "Fictional reflection resource 1", exact: true })).toBeVisible();
  await page.getByLabel("Feedback for Luna").fill("Something shorter and practical");
  await page.getByRole("button", { name: "Find different sources" }).click();
  await expect(page.getByRole("link", { name: "Fictional reflection resource 3", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Fictional reflection resource 1", exact: true })).toHaveCount(0);
  // The original topic stays the visible goal while refinement narrows the search.
  await expect(page.getByRole("heading", { name: "Understanding overthinking", exact: true })).toBeVisible();
  await expect(page.getByText(
    "Latest search: Understanding overthinking Something shorter and practical", { exact: true },
  )).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("slice-3-discovery-mobile.png"), fullPage: true });
});
