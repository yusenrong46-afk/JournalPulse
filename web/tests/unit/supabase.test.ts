import { afterEach, beforeEach, expect, test, vi } from "vitest";

const mocks = vi.hoisted(() => ({ createClient: vi.fn() }));
vi.mock("@supabase/supabase-js", () => ({ createClient: mocks.createClient }));

beforeEach(() => {
  vi.resetModules();
  mocks.createClient.mockReset();
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://journal-test.supabase.co");
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "fixture-public-key");
});
afterEach(() => { vi.unstubAllEnvs(); });

test("concurrent SDK requests initialize a single client and reuse it afterward", async () => {
  const client = { auth: {} };
  mocks.createClient.mockReturnValue(client);
  const { getSupabase } = await import("@/lib/supabase");
  const first = getSupabase();
  const second = getSupabase();
  expect(first).toBe(second);
  expect(await first).toBe(client);
  expect(await getSupabase()).toBe(client);
  expect(mocks.createClient).toHaveBeenCalledTimes(1);
});

test("failed SDK initialization is shared by current callers but a later attempt can recover", async () => {
  const error = new Error("Initialization failed");
  const client = { auth: {} };
  mocks.createClient.mockImplementationOnce(() => { throw error; }).mockReturnValue(client);
  const { getSupabase } = await import("@/lib/supabase");
  const first = getSupabase();
  const second = getSupabase();
  expect(first).toBe(second);
  await expect(first).rejects.toBe(error);
  await expect(second).rejects.toBe(error);
  const retry = getSupabase();
  expect(retry).not.toBe(first);
  expect(await retry).toBe(client);
  expect(mocks.createClient).toHaveBeenCalledTimes(2);
});
