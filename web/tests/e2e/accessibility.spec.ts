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

for (const path of ["/", "/talk", "/journal", "/discover", "/journey", "/me", "/welcome", "/login", "/check-in"]) {
  test(`${path} has no serious automated accessibility violations`, async ({ page }) => {
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    // Axe samples colours as rendered. Mid-fade text blends with its background, which made
    // this check flaky; wait for finite entrance animations so the settled design is tested.
    await page.evaluate(() => Promise.all(document.getAnimations()
      .filter((animation) => animation.effect?.getComputedTiming().iterations !== Infinity)
      .map((animation) => animation.finished.catch(() => undefined))));
    const results = await new AxeBuilder({ page }).analyze();
    const serious = results.violations.filter((item) => ["serious", "critical"].includes(item.impact ?? ""));
    expect(serious).toEqual([]);
  });
}
