import { expect, test } from "@playwright/test";

for (const [path, heading] of [
  ["/privacy", "Your privacy"],
  ["/terms", "Terms of use"],
  ["/how-ai-works", "How data and AI are used"],
]) {
  test(`${path} is a public disclosure, not an account-settings redirect`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
    await expect(page.getByRole("navigation", { name: "About JournalPulse" })).toBeVisible();
    await expect(page.getByRole("complementary", { name: "Crisis support" })).toBeVisible();
    // Inspect targets only: automated tests must never place a helpline call.
    await expect(page.locator('a[href="tel:988"]')).toHaveText("988");
    await expect(page.locator('a[href="sms:988"]')).toHaveText("988");
    expect(new URL(page.url()).pathname.replace(/\/$/, "")).toBe(path);
  });
}

test("sign-in links to disclosures before an email is submitted", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("textbox", { name: "Email", exact: true })).toBeEmpty();
  await page.getByRole("link", { name: "Privacy", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Your privacy", exact: true })).toBeVisible();
});
