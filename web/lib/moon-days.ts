import type { MoonKind } from "@/components/moon-mark";

export type MoonDay = { key: string; date: Date; kind: MoonKind; today: boolean; label: string };

/**
 * One mark per local day for the last `days` days, oldest first. A day with an activity
 * check-in shows the activity mark; otherwise any saved reflection or chat moment shows the
 * moment mark. Marks record that the person showed up, never how well anything worked.
 */
export function moonDays(
  moments: readonly string[],
  activities: readonly string[],
  days = 7,
  now: Date = new Date(),
): MoonDay[] {
  const dayKey = (value: Date) => `${value.getFullYear()}-${value.getMonth()}-${value.getDate()}`;
  const momentDays = new Set(moments.map((value) => dayKey(new Date(value))));
  const activityDays = new Set(activities.map((value) => dayKey(new Date(value))));
  return Array.from({ length: days }, (_, index) => {
    const date = new Date(now.getFullYear(), now.getMonth(), now.getDate() - (days - 1 - index));
    const key = dayKey(date);
    const kind: MoonKind = activityDays.has(key) ? "activity" : momentDays.has(key) ? "moment" : "empty";
    const name = date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
    const what = kind === "activity" ? "activity check-in" : kind === "moment" ? "a saved moment" : "nothing recorded";
    return { key, date, kind, today: index === days - 1, label: `${name}: ${what}` };
  });
}
