"use client";

import Link from "next/link";
import { type ReactNode, useEffect } from "react";

import { AuthBoundary } from "@/components/auth-boundary";
import { Luna } from "@/components/luna";
import { Icon, type IconName } from "@/components/nav-icon";
import { SystemStatus } from "@/components/system-status";
import { useRoutePath } from "@/lib/route-path";
import { useTimeOfDay } from "@/lib/time-of-day";

const navigation: { href: string; label: string; icon: IconName }[] = [
  { href: "/", label: "Home", icon: "home" },
  { href: "/journey", label: "Journey", icon: "journey" },
  { href: "/me", label: "Me", icon: "me" },
];

const IMMERSIVE = new Set(["/talk", "/welcome", "/login", "/check-in"]);

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRoutePath();
  const time = useTimeOfDay();
  const immersive = IMMERSIVE.has(pathname);

  useEffect(() => {
    if (time) document.documentElement.dataset.time = time;
  }, [time]);

  const isActive = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));

  return (
    <div className={immersive ? "app immersive" : "app"}>
      <a className="skip-link" href="#main">Skip to content</a>
      {!immersive && (
        <aside className="side-nav" aria-label="Main">
          <Link className="brand" href="/">
            <Luna mood="idle" size={40} decorative />
            JournalPulse
          </Link>
          {navigation.map((item) => (
            <Link
              key={item.href}
              className="nav-link"
              href={item.href}
              aria-current={isActive(item.href) ? "page" : undefined}
            >
              <Icon name={item.icon} />
              {item.label}
            </Link>
          ))}
          <Link className="btn btn-primary side-talk" href="/talk">Talk with Luna</Link>
          <p className="side-note">Luna is a companion, not a therapist.</p>
        </aside>
      )}
      <main id="main" className="app-main">
        <SystemStatus />
        <AuthBoundary>{children}</AuthBoundary>
      </main>
      {!immersive && (
        <nav className="tab-bar" aria-label="Main">
          {navigation.map((item) => (
            <Link key={item.href} href={item.href} aria-current={isActive(item.href) ? "page" : undefined}>
              <Icon name={item.icon} />
              {item.label}
            </Link>
          ))}
        </nav>
      )}
    </div>
  );
}
