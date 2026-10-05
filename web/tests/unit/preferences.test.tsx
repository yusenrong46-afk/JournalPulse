import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { usePreferences } from "@/lib/preferences";

let root: Root;
let container: HTMLDivElement;
function SettingsProbe() {
  const [preferences, save] = usePreferences();
  return createElement("button", { onClick: () => save({ ...preferences, llmConsent: true }) }, JSON.stringify(preferences));
}
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  vi.restoreAllMocks();
});
async function mount() { await act(async () => { root.render(createElement(SettingsProbe)); }); }
describe("stored preference validation", () => {
  test.each([JSON.stringify({ llmConsent: "false", retainText: "false", followUpMinutes: -20, locale: 2 }), "null"])(
    "malformed values cannot enable AI or retention (%s)", async (value) => {
      window.localStorage.setItem("journalpulse_preferences_v1", value);
      await mount();
      expect(container.textContent).toContain('"llmConsent":false');
      expect(container.textContent).toContain('"retainText":false');
      expect(container.textContent).toContain('"followUpMinutes":10');
      expect(container.textContent).toContain('"locale":"CA"');
    },
  );
  test("blocked browser storage keeps settings usable for this session", async () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("Storage blocked", "SecurityError"); });
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new DOMException("Storage blocked", "SecurityError"); });
    await mount();
    expect(container.textContent).toContain('"llmConsent":false');
    await act(async () => { container.querySelector("button")!.click(); });
    expect(container.textContent).toContain('"llmConsent":true');
  });
});
