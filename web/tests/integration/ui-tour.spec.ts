// Screenshot tour for before/after UI review on the real local stack. Luna, Brave and auth
// are the stack's deterministic stand-ins, so these images show layout and state, not live
// model quality. Runs only when JP_SCREENSHOT_DIR is set, so CI time is unchanged.
// Selectors accept both the original and the redesigned labels, so one script produces
// comparable before/after images.
import AxeBuilder from "@axe-core/playwright";
import { expect, type APIRequestContext, type Page, test } from "@playwright/test";
import path from "node:path";

type Session = { user_id: string; access_token: string };
const OUT = process.env.JP_SCREENSHOT_DIR;

test.skip(!OUT, "Set JP_SCREENSHOT_DIR to capture the UI tour.");

async function sessionFor(request: APIRequestContext, name: string): Promise<Session> {
  const response = await request.get(`http://127.0.0.1:54321/test/session/${name}`);
  expect(response.ok()).toBeTruthy();
  return response.json();
}

const headers = (who: Session) => ({ Authorization: `Bearer ${who.access_token}` });

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

async function say(page: Page, text: string) {
  await expect(page.getByLabel("Message Luna")).toBeEnabled();
  await page.getByLabel("Message Luna").fill(text);
  await page.getByRole("button", { name: /^Send( message)?$/ }).click();
  await expect(page.getByLabel("Message Luna")).toHaveValue("");
}

async function shot(page: Page, label: string, name: string) {
  // Let entry animations settle so images compare like for like.
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT!, `${label}-${name}.png`), fullPage: false });
  // Each captured state is also checked for serious automated accessibility violations.
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter((item) => ["serious", "critical"].includes(item.impact ?? ""));
  expect(serious.map((item) => `${name}: ${item.id} ${item.nodes.map((node) => node.target).join(" | ")}`)).toEqual([]);
}

async function finishEarly(page: Page) {
  const direct = page.getByRole("button", { name: /^(Finish early|I.m done)$/ });
  if (await direct.count()) {
    await direct.first().click();
    return;
  }
  await page.getByRole("button", { name: /More activity options/ }).click();
  await page.getByRole("menuitem", { name: /I.m done/ }).click();
}

for (const [label, use] of [
  ["mobile", {}],
  ["desktop", { viewport: { width: 1280, height: 820 }, isMobile: false, hasTouch: false, deviceScaleFactor: 1 }],
] as const) {
  test.describe(label, () => {
    test.use(use);

    test(`${label} activity loop`, async ({ page, request }) => {
      const who = await sessionFor(request, "grace");
      await clearAccount(request, who);
      try {
        await signIn(page, who);
        await page.goto("/talk/");
        await expect(page.getByLabel("Message Luna")).toBeVisible();
        await shot(page, label, "01-welcome");
        await say(page, "A fictional busy afternoon has left my head feeling crowded.");
        await say(page, "I have two minutes and would like a quiet seated pause, without audio.");
        const start = page.getByRole("button", { name: /^Start( activity|\b)/ }).first();
        await expect(start).toBeVisible();
        await shot(page, label, "02-offer");
        await start.click();
        await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeVisible();
        await shot(page, label, "03-running");
        await page.getByRole("button", { name: "Pause", exact: true }).click();
        await expect(page.getByRole("button", { name: "Resume", exact: true })).toBeVisible();
        await shot(page, label, "04-paused");
        await finishEarly(page);
        await expect(page.getByRole("button", { name: "Save check-in", exact: true })).toBeVisible();
        await shot(page, label, "05-check-in");
        await page.getByRole("radio", { name: /^(Not tried|Didn.t try)$/ }).check();
        await page.getByRole("button", { name: "Save check-in", exact: true }).click();
        await expect(page.getByRole("button", { name: "Save check-in", exact: true })).toHaveCount(0);
        await page.waitForTimeout(1500);
        await shot(page, label, "06-follow-up");
      } finally {
        await clearAccount(request, who);
      }
    });

    test(`${label} search and journal`, async ({ page, request }) => {
      const who = await sessionFor(request, "grace");
      await clearAccount(request, who);
      try {
        await signIn(page, who);
        await page.goto("/talk/");
        await say(page, "A fictional busy afternoon has left my head feeling crowded.");
        await page.getByRole("button", { name: /^(Find another resource|Something else|Search for public ideas)$/ }).first().click();
        const consent = page.getByRole("checkbox", { name: /search/i }).first();
        await consent.check();
        await page.getByLabel(/topic/i).first().fill("quiet meditation");
        await page.getByRole("button", { name: /^Search( activities)?$/ }).click();
        await expect(page.getByRole("heading", { name: "Fictional reflection resource 1", exact: true })
          .or(page.getByText("Fictional reflection resource 1", { exact: true }))).toBeVisible();
        await shot(page, label, "07-search");

        const saved = await request.post("/v1/journal/entries", {
          headers: headers(who), data: { text: "A fictional meeting left me stressed and tired." },
        });
        const entry = await saved.json();
        // Accept "end the current chat?" so the entry really starts a new chat. The baseline
        // run dismissed it, which exposed the composer-under-chooser bug fixed in this audit.
        page.on("dialog", (dialog) => void dialog.accept());
        await page.goto(`/journal/?entry=${entry.id}`);
        await page.getByRole("link", { name: "Discuss with Luna" }).click();
        await page.getByRole("button", { name: "Use this entry in a new AI chat" }).click();
        await say(page, "What might I want to explore about that meeting?");
        await shot(page, label, "08-journal");
      } finally {
        await clearAccount(request, who);
      }
    });
  });
}
