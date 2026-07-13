import { afterEach, describe, expect, test, vi } from "vitest";

import { apiRequest } from "@/lib/api";

afterEach(() => {
  vi.useRealTimers();
});

describe("apiRequest", () => {
  test("retries a safe read after a transient network failure", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError("network unavailable"))
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "ok" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const request = apiRequest<{ status: string }>("/health");
    await vi.runAllTimersAsync();

    await expect(request).resolves.toEqual({ status: "ok" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test("does not retry an unsafe write unless the caller marks it idempotent", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockRejectedValue(new TypeError("network unavailable"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(apiRequest("/v1/reflections", { method: "POST", body: "{}" })).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
