import { expect, test } from "vitest";

import { moonDays } from "@/lib/moon-days";

const NOW = new Date(2026, 9, 6, 20);

test("marks each local day once, oldest first, with today last", () => {
  const days = moonDays([], [], 7, NOW);
  expect(days).toHaveLength(7);
  expect(days.at(-1)?.today).toBe(true);
  expect(days.every((day) => day.kind === "empty")).toBe(true);
  expect(days[0].date.getDate()).toBe(30);
});

test("an activity check-in outranks a moment on the same day, and labels never grade outcomes", () => {
  const days = moonDays(
    [new Date(2026, 9, 4, 9).toISOString(), new Date(2026, 9, 5, 9).toISOString()],
    [new Date(2026, 9, 5, 18).toISOString()],
    7, NOW,
  );
  const byDate = new Map(days.map((day) => [day.date.getDate(), day]));
  expect(byDate.get(4)?.kind).toBe("moment");
  expect(byDate.get(5)?.kind).toBe("activity");
  expect(byDate.get(6)?.kind).toBe("empty");
  expect(days.map((day) => day.label).join(" ")).not.toMatch(/helped|better|worse/i);
});
