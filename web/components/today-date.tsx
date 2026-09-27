"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

function localDay() {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function format(isoDay: string) {
  const [year, month, day] = isoDay.split("-").map(Number);
  return new Intl.DateTimeFormat("en-CA", { weekday: "long", month: "long", day: "numeric" }).format(
    new Date(year, month - 1, day),
  );
}

/**
 * The static export is built once, so a date rendered on the server would stay
 * frozen at build time. Resolving it on the client keeps it the reader's own day.
 */
export function TodayDate() {
  const isoDay = useSyncExternalStore(subscribe, localDay, () => null);

  if (!isoDay) return <time suppressHydrationWarning />;
  return (
    <time dateTime={isoDay} suppressHydrationWarning>
      {format(isoDay)}
    </time>
  );
}
