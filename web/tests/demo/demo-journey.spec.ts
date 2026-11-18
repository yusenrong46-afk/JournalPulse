// The portfolio demo journey, end to end against the local demo server: Try the demo,
// write and save an entry, talk with Simple Luna, choose an activity, check in, and see
// it in Journey. Every request must stay on this machine.
import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

const DOCS = path.resolve(__dirname, "../../../docs/demo");
const LABEL = "Demo: fictional data, AI off";

test("the demo journey works with fictional data, AI off and no outside requests", async ({ browser, baseURL }) => {
  mkdirSync(DOCS, { recursive: true });
  const allowed = new URL(baseURL!).host;
  const outside: string[] = [];
  const context = await browser.newContext({
    ...test.info().project.use,
    baseURL,
    recordVideo: { dir: test.info().outputPath("video"), size: { width: 412, height: 915 } },
  });
  // Anything not served by the demo server is recorded and refused before it leaves.
  await context.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.protocol === "data:" || url.protocol === "blob:" || url.host === allowed) return route.continue();
    outside.push(`${route.request().method()} ${url.protocol}//${url.host}`);
    return route.abort();
  });
  const page = await context.newPage();
  const banner = page.getByRole("note", { name: "Demo" });

  // 1. Try the demo: no email, an isolated identity with three fictional entries.
  await page.goto("/login/");
  await expect(page.getByText(/9-8-8 Suicide Crisis Helpline/)).toBeVisible();
  await page.getByRole("button", { name: "Try the demo" }).click();
  await expect(banner).toContainText(LABEL);
  await expect(banner.getByRole("link", { name: "9-8-8" })).toBeVisible();

  // 2. Write and save an entry.
  await page.goto("/journal/");
  await expect(page.getByText(/3 entries/)).toBeVisible();
  await page.getByRole("textbox", { name: "What would you like to remember?" })
    .fill("Demo entry (fictional): A short walk after lunch cleared my head.");
  await page.getByRole("button", { name: "Save entry" }).click();
  await expect(page.getByText("Entry saved.", { exact: false })).toBeVisible();
  await expect(page.getByText(/4 entries/)).toBeVisible();
  await expect(banner).toContainText(LABEL);
  await page.screenshot({ path: path.join(DOCS, "01-journal-saved.png") });

  // 3. Simple Luna (no AI), then choose an activity.
  await page.goto("/talk/");
  await expect(page.getByText(/Simple mode · no AI/)).toBeVisible();
  await page.getByRole("button", { name: /Low/ }).click();
  await page.getByLabel("Message Luna").fill("Demo message (fictional): work was busy and I feel tired. Please find one small step.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.getByRole("button", { name: /Yes, let’s find one small thing/ }).click();
  await page.getByRole("button", { name: "That’s it" }).click();
  await page.getByRole("button", { name: /Calm down/ }).click();
  await expect(page.getByRole("button", { name: "Let’s try it" })).toBeVisible();
  await expect(banner).toContainText(LABEL);
  await page.screenshot({ path: path.join(DOCS, "02-simple-luna-activity.png") });
  await page.getByRole("button", { name: "Let’s try it" }).click();
  await expect(page.getByRole("heading", { name: "Nice choice." })).toBeVisible();

  // 4. Check in.
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Did it help/ })).toBeVisible();
  await page.getByRole("link", { name: "Helped", exact: true }).click();
  await page.getByRole("button", { name: /Calm$/ }).click();
  await page.getByRole("button", { name: "Save my check-in" }).click();
  await expect(page.getByRole("heading", { name: "Thank you!" })).toBeVisible();

  // 5. See it in Journey.
  await page.goto("/journey/");
  await expect(page.getByText(/1 rating · 1 rated helpful/)).toBeVisible();
  await expect(banner).toContainText(LABEL);
  await page.screenshot({ path: path.join(DOCS, "03-journey.png") });

  const video = page.video();
  await context.close();
  copyFileSync(await video!.path(), path.join(DOCS, "demo-journey.webm"));
  expect(outside, "requests that tried to leave this machine").toEqual([]);
});
