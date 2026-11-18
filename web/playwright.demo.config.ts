import { defineConfig, devices } from "@playwright/test";

// `make demo-check`: drives the local demo (scripts/demo_server.py) through the main
// journey, saves screenshots and a short video under docs/demo/, and fails if any
// request leaves this machine. Fictional data only; AI and search are off.
export const DEMO_PORT = Number(process.env.JP_DEMO_PORT ?? "8300");
// Set by `make onboard`: drive an environment that is already running instead.
const EXTERNAL = process.env.JP_DEMO_BASE_URL;

export default defineConfig({
  testDir: "./tests/demo",
  outputDir: "./test-results-demo",
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: "list",
  timeout: 120_000,
  use: {
    ...devices["Pixel 7"],
    browserName: "chromium",
    baseURL: EXTERNAL ?? `http://127.0.0.1:${DEMO_PORT}`,
    timezoneId: "America/Vancouver",
    trace: "retain-on-failure",
  },
  webServer: EXTERNAL ? undefined : {
    command: `uv run python ../scripts/demo_server.py --port ${DEMO_PORT}${process.env.JP_SKIP_WEB_BUILD === "1" ? " --skip-build" : ""}`,
    url: `http://127.0.0.1:${DEMO_PORT}/ready`,
    reuseExistingServer: false,
    timeout: 400_000,
    stdout: "pipe",
  },
});
