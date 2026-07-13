"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { AuthBoundary } from "@/components/auth-boundary";
import { NavIcon } from "@/components/nav-icon";
import { SystemStatus } from "@/components/system-status";

const navigation = [
  { href: "/", label: "Today", icon: "today", note: "Current loop" },
  { href: "/reflect", label: "Reflect", icon: "reflect", note: "New observation" },
  { href: "/history", label: "History", icon: "history", note: "Private record" },
  { href: "/patterns", label: "Patterns", icon: "patterns", note: "Your evidence" },
  { href: "/privacy", label: "Privacy", icon: "privacy", note: "Data controls" },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const immersive = pathname === "/welcome" || pathname === "/login";
  return (
    <div className={immersive ? "app-frame immersive" : "app-frame"}>
      <aside className="rail" aria-label="Primary navigation">
        <Link className="wordmark" href="/" aria-label="JournalPulse home">
          <span className="wordmark-mark">JP</span>
          <span className="wordmark-copy"><strong>JournalPulse</strong><small>Research beta</small></span>
        </Link>
        <p className="rail-note">A private field journal for noticing what changes, then testing one useful next move.</p>
        <nav className="rail-nav">
          {navigation.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link key={item.href} href={item.href} prefetch={false} aria-current={active ? "page" : undefined} className={active ? "nav-link active" : "nav-link"}>
                <span className="nav-icon"><NavIcon name={item.icon} /></span>
                <span className="nav-copy"><strong>{item.label}</strong><small>{item.note}</small></span>
              </Link>
            );
          })}
        </nav>
        <div className="rail-boundary" aria-label="Product boundary">
          <span className="status-dot" aria-hidden="true" />
          <span><strong>Reflection, not diagnosis</strong><small>You make the final call.</small></span>
        </div>
      </aside>
      <header className="mobile-masthead">
        <Link className="wordmark" href="/" aria-label="JournalPulse home">
          <span className="wordmark-mark">JP</span>
          <span className="wordmark-copy"><strong>JournalPulse</strong><small>Research beta</small></span>
        </Link>
        <Link className="mobile-new-entry" href="/reflect" prefetch={false} aria-label="Start a new reflection">
          <NavIcon name="reflect" />
        </Link>
      </header>
      <main className="page-surface">
        <AuthBoundary>
          <SystemStatus />
          {children}
        </AuthBoundary>
      </main>
      <nav className="bottom-nav" aria-label="Mobile navigation">
        {navigation.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link key={item.href} href={item.href} prefetch={false} aria-current={active ? "page" : undefined} className={`${active ? "active" : ""} ${item.href === "/reflect" ? "primary-destination" : ""}`}>
              <span className="nav-icon"><NavIcon name={item.icon} /></span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
