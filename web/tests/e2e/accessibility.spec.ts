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
  await page.route("http://127.0.0.1:8000/**", (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === "/v1/system/status") {
      return route.fulfill({
        json: { analysis_mode: "ai_configured", persistence_mode: "this_device", message: "Ready." },
      });
    }
    if (pathname === "/v1/insights") {
      return route.fulfill({
        json: {
          reflection_count: 0,
          completed_outcomes: 0,
          action_counts: {},
          average_helpfulness_by_action: {},
          average_state_change: null,
          completion_rate: 0,
          pending_decision_ids: [],
          state_trajectory: [],
          note: "Descriptive only.",
        },
      });
    }
    return route.fulfill({ json: { items: [] } });
  });
});

for (const path of ["/", "/reflect", "/talk", "/privacy"]) {
  test(`${path} has no serious automated accessibility violations`, async ({ page }) => {
    await page.goto(path);
    const results = await new AxeBuilder({ page }).analyze();
    const serious = results.violations.filter((item) => ["serious", "critical"].includes(item.impact ?? ""));
    expect(serious).toEqual([]);
  });
}
