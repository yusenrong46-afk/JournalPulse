// Portfolio demo sandbox against the real API, PostgREST and PostgreSQL. The gateway's
// /auth/v1/signup stands in for Supabase anonymous sign-ins; everything else is real.
import { type APIRequestContext, expect, test } from "@playwright/test";

type DemoSession = { user_id: string; label: string; access_token: string; refresh_token: string };

async function demo(request: APIRequestContext): Promise<DemoSession> {
  const response = await request.post("/v1/demo/sessions");
  expect(response.status(), await response.text()).toBe(201);
  const body: DemoSession = await response.json();
  expect(body.label).toBe("Demo: fictional data, AI off");
  expect(body.access_token).toBeTruthy();
  return body;
}

function as(request: APIRequestContext, who: DemoSession) {
  const headers = { Authorization: `Bearer ${who.access_token}` };
  return {
    get: (path: string) => request.get(path, { headers }),
    post: (path: string, data: unknown) =>
      request.post(path, { headers: { ...headers, "X-JournalPulse-Data-Revision": "0" }, data }),
    delete: (path: string) => request.delete(path, { headers }),
  };
}

test("two demo identities on real Postgres see only their own seeded fiction", async ({ request }) => {
  const first = await demo(request);
  const second = await demo(request);
  expect(first.user_id).not.toBe(second.user_id);
  const firstEntries = (await (await as(request, first).get("/v1/journal/entries")).json()).items;
  const secondEntries = (await (await as(request, second).get("/v1/journal/entries")).json()).items;
  expect(firstEntries).toHaveLength(3);
  expect(secondEntries).toHaveLength(3);
  for (const item of firstEntries) expect(item.text).toMatch(/^Demo entry \(fictional\):/);
  const firstIds = new Set(firstEntries.map((item: { id: string }) => item.id));
  expect(secondEntries.some((item: { id: string }) => firstIds.has(item.id))).toBe(false);
  const target = firstEntries[0].id;
  expect((await as(request, second).get(`/v1/journal/entries/${target}`)).status()).toBe(404);
  expect((await as(request, second).delete(`/v1/journal/entries/${target}`)).status()).toBe(404);
  expect((await as(request, first).get(`/v1/journal/entries/${target}`)).status()).toBe(200);
  const history = (await (await as(request, first).get("/v1/activity-history")).json()).items;
  expect(history).toHaveLength(1);
  expect(history[0].participation).toBe("completed");
});

test("a demo identity asking for AI gets Simple mode", async ({ request }) => {
  const who = await demo(request);
  const chat = await as(request, who).post("/v1/conversations", { client_request_id: crypto.randomUUID(), llm_consent: true });
  expect(chat.status(), await chat.text()).toBe(201);
  expect((await chat.json()).mode).toBe("guided");
  expect((await chat.json()).llm_consent).toBe(false);
  const reflect = await as(request, who).post("/v1/discovery/search", { original_query: "short walk", llm_consent: true });
  expect(reflect.status()).toBe(403);
});

test("Try the demo signs in anonymously and labels the workspace", async ({ page }) => {
  await page.goto("/login/");
  await page.getByRole("button", { name: "Try the demo" }).click();
  await expect(page.getByRole("note", { name: "Demo" })).toContainText("Demo: fictional data, AI off");
  await expect(page.getByRole("note", { name: "Demo" }).getByRole("link", { name: "9-8-8" })).toBeVisible();
  await page.goto("/journal/");
  await expect(page.getByText(/Watered the balcony tomatoes/).first()).toBeVisible();
  await expect(page.getByRole("note", { name: "Demo" })).toBeVisible();
});
