import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "journalpulse_preferences_v1",
      JSON.stringify({
        onboarded: true,
        llmConsent: false,
        retainText: false,
        encryptedDrafts: false,
        followUpMinutes: 10,
        locale: "CA",
      }),
    );
  });
  await page.route("http://127.0.0.1:8000/**", (route) => route.fulfill({ json: { items: [] } }));
});

for (const path of ["/", "/talk", "/journey", "/me", "/welcome", "/login", "/check-in"]) {
  test(`${path} has no serious automated accessibility violations`, async ({ page }) => {
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    const results = await new AxeBuilder({ page }).analyze();
    const serious = results.violations.filter((item) => ["serious", "critical"].includes(item.impact ?? ""));
    expect(serious).toEqual([]);
  });
}
