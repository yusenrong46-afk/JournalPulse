"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const navigation = [
  { href: "/", label: "Today", mark: "01" },
  { href: "/reflect", label: "Reflect", mark: "+" },
  { href: "/history", label: "History", mark: "02" },
  { href: "/patterns", label: "Patterns", mark: "03" },
  { href: "/privacy", label: "Privacy", mark: "04" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="app-frame">
      <aside className="rail" aria-label="Primary navigation">
        <Link className="wordmark" href="/" aria-label="JournalPulse home">
          <span className="wordmark-mark">JP</span>
          <span>JournalPulse</span>
        </Link>
        <p className="rail-note">A personal laboratory for noticing what actually helps.</p>
        <nav className="rail-nav">
          {navigation.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link key={item.href} href={item.href} className={active ? "nav-link active" : "nav-link"}>
                <span>{item.mark}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="rail-boundary">
          <span className="status-dot" />
          Non-clinical reflection
        </div>
      </aside>
      <main className="page-surface">{children}</main>
      <nav className="bottom-nav" aria-label="Mobile navigation">
        {navigation.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link key={item.href} href={item.href} className={active ? "active" : ""}>
              <span>{item.mark}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
