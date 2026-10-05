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

  test("validation errors explain the field without dumping submitted private input", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      detail: [{ type: "string_too_long", loc: ["body", "text"],
        msg: "String should have at most 2000 characters", input: "private journal words", ctx: { max_length: 2000 } }],
    }), { status: 422 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(apiRequest("/v1/conversations/chat/messages", { method: "POST", body: "{}" }))
      .rejects.toThrow("Message: String should have at most 2000 characters");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  test("unexpected error objects have a readable fallback and validation output is bounded", async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: { input: "private words" } }), { status: 400 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: Array.from({ length: 20 }, (_, index) => ({
        loc: ["body", "feedback"], msg: `${index} ${"too long ".repeat(200)}`, input: "private words",
      })) }), { status: 422 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(apiRequest("/request", { method: "POST" }))
      .rejects.toThrow("JournalPulse could not complete the request. Please check your answers and try again.");
    const error = await apiRequest("/request", { method: "POST" }).catch((reason: Error) => reason);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message.length).toBeLessThanOrEqual(600);
    expect((error as Error).message).not.toContain("private words");
    expect((error as Error).message).not.toContain("[object Object]");
  });
});
