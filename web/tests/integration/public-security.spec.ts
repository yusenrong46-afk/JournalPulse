import { expect, test } from "@playwright/test";

test("exported pages enforce script hashes and the exact local Auth origin", async ({ page }, info) => {
  const response = await page.goto("/login/");
  const csp = response!.headers()["content-security-policy"];
  expect(csp).toContain("connect-src 'self' http://127.0.0.1:54321");
  const scriptPolicy = csp.split(";").find((part) => part.trim().startsWith("script-src "))!;
  expect(scriptPolicy).toContain("'sha256-");
  expect(scriptPolicy).not.toContain("unsafe-inline");
  expect(scriptPolicy).not.toContain("unsafe-eval");
  await page.getByRole("textbox", { name: "Email", exact: true }).fill("fictional@example.test");
  await expect(page.getByRole("button", { name: "Email me a link" })).toBeEnabled();
  const violations = await page.evaluate(async () => {
    const blocked: string[] = [];
    document.addEventListener("securitypolicyviolation", (event) => blocked.push(event.violatedDirective));
    const script = document.createElement("script");
    script.textContent = 'document.documentElement.dataset.unapprovedScript = "ran"';
    document.head.appendChild(script);
    try { await fetch("https://unapproved.example.test/auth"); } catch { /* expected CSP block */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
    return { blocked, ran: document.documentElement.dataset.unapprovedScript };
  });
  expect(violations.ran).toBeUndefined();
  expect(violations.blocked).toContain("script-src-elem");
  expect(violations.blocked).toContain("connect-src");
  await info.attach("response-csp", { body: csp, contentType: "text/plain" });
});
