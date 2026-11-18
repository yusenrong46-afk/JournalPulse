"use client";

import { DEMO_LABEL, useDemoIdentity } from "@/lib/demo";

/** Shown on every view while the browser acts as a demo identity. The crisis line
 * stays in the banner, so it is visible on every demo screen, not only at sign-in. */
export function DemoBanner() {
  const demo = useDemoIdentity();
  if (!demo) return null;
  return (
    <aside className="demo-banner" role="note" aria-label="Demo">
      <strong>{DEMO_LABEL}</strong>
      <span>
        {" "}Nothing here is a real person&apos;s writing. In crisis in Canada? Call or text{" "}
        <a href="tel:988">9-8-8</a>.
      </span>
    </aside>
  );
}
