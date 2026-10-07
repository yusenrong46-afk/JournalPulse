import { defineConfig, devices } from "@playwright/test";

// Browser -> FastAPI -> PostgREST -> PostgreSQL with the real migrations. See
// scripts/integration_stack.py for what is real and what is a stand-in.
export default defineConfig({
  testDir: "./tests/integration",
  outputDir: "./test-results-integration",
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: "list",
  timeout: 60_000,
  use: {
    baseURL: "http://127.0.0.1:8100",
    trace: "retain-on-failure",
    timezoneId: "UTC",
  },
  projects: [{ name: "integration-chromium", use: { ...devices["Pixel 7"], browserName: "chromium" } }],
  webServer: {
    command: "uv run python ../scripts/integration_stack.py",
    url: "http://127.0.0.1:8100/ready",
    reuseExistingServer: !process.env.CI,
    timeout: 300_000,
    stdout: "pipe",
  },
});
