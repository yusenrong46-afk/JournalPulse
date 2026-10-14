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
    expect(new URL(page.url()).pathname.replace(/\/$/, "")).toBe(path);
  });
}

test("sign-in links to disclosures before an email is submitted", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("textbox", { name: "Email", exact: true })).toBeEmpty();
  await page.getByRole("link", { name: "Privacy", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Your privacy", exact: true })).toBeVisible();
});
