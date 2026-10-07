// Outlast the server's 100-second generation budget and 120-second runtime
// ceiling, leaving time for database work and the response to reach the browser.
export const LUNA_REQUEST_TIMEOUT_MS = 125_000;
