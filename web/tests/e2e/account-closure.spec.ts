import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

for (const configured of [false, true]) {
  test(`account closure disclosure is accessible when ${configured ? "configured" : "unavailable"}`, async ({ page }) => {
    await page.addInitScript(() => window.localStorage.setItem("journalpulse_preferences_v1",
      JSON.stringify({ onboarded: true, llmConsent: false, retainText: false, encryptedDrafts: false,
        followUpMinutes: 10, locale: "CA" })));
    let closureRequests = 0;
    await page.route("http://127.0.0.1:8000/**", async route => {
      if (new URL(route.request().url()).pathname === "/v1/account") closureRequests++;
      await route.fulfill({ json: { items: [], account_closure: configured ? "configured" : "unavailable" } });
    });
    await page.goto("/me");
    await page.getByText("Close account", { exact: true }).click();
    await expect(page.getByRole("heading", { name: "Permanently close your account" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Close my account", exact: true })).toBeDisabled();
    await expect(page.getByRole("link", { name: "yusenrong46@gmail.com", exact: true }).first()).toBeVisible();
    if (!configured) await expect(page.getByText(/Online account closure is currently unavailable/)).toBeVisible();
    await page.getByLabel("Type “delete my account” to confirm").fill("delete my account");
    if (configured) await expect(page.getByRole("button", { name: "Close my account", exact: true })).toBeEnabled();
    else await expect(page.getByRole("button", { name: "Close my account", exact: true })).toBeDisabled();
    expect(closureRequests).toBe(0);
    await page.evaluate(() => Promise.all(document.getAnimations()
      .filter(animation => animation.effect?.getComputedTiming().iterations !== Infinity)
      .map(animation => animation.finished.catch(() => undefined))));
    const results = await new AxeBuilder({ page }).analyze();
    expect(results.violations.filter(item => ["serious", "critical"].includes(item.impact ?? ""))).toEqual([]);
  });
}
