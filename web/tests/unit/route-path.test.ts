import { describe, expect, test } from "vitest";

import { normalizeRoutePath } from "@/lib/route-path";

describe("route path normalization", () => {
  test("treats trailing-slash export paths as the same route", () => {
    expect(normalizeRoutePath("/login/")).toBe("/login");
    expect(normalizeRoutePath("/login")).toBe("/login");
    expect(normalizeRoutePath("/talk//")).toBe("/talk");
  });

  test("keeps the root route", () => {
    expect(normalizeRoutePath("/")).toBe("/");
    expect(normalizeRoutePath("")).toBe("/");
    expect(normalizeRoutePath(null)).toBe("/");
  });
});
