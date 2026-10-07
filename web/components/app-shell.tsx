"use client";

import Link from "next/link";
import { useEffect, type ReactNode } from "react";
import { Ambient } from "@/components/ambient";
import { AuthBoundary } from "@/components/auth-boundary";
import { Icon, type IconName } from "@/components/nav-icon";
import { SystemStatus } from "@/components/system-status";
import { useRoutePath } from "@/lib/route-path";
import { useTimeOfDay } from "@/lib/time-of-day";
import { usePreferences } from "@/lib/preferences";

const navigation: { href: string; label: string; icon: IconName }[] = [
  { href: "/", label: "Home", icon: "home" },
  { href: "/talk", label: "Chat", icon: "chat" },
  { href: "/journal", label: "Journal", icon: "journal" },
  { href: "/discover", label: "Explore", icon: "explore" },
  { href: "/journey", label: "Journey", icon: "journey" },
  { href: "/me", label: "Settings", icon: "settings" },
];
const PRIVATE_FRAME = new Set(["/welcome", "/login", "/check-in"]);
// Phones get a thumb-reach tab bar with the five daily places; Explore stays one tap away
// from Home and inside chat, so the bar never needs six cramped targets.
const TAB_BAR = new Set(["/", "/talk", "/journal", "/journey", "/me"]);

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRoutePath();
  const time = useTimeOfDay();
  const [preferences] = usePreferences();
  const quiet = PRIVATE_FRAME.has(pathname);
  const chat = pathname === "/talk";
  useEffect(() => { if (time) document.documentElement.dataset.time = time; }, [time]);
  useEffect(() => {
    const sync = () => { document.documentElement.dataset.motion = document.hidden ? "paused" : "active"; };
    sync();
    document.addEventListener("visibilitychange", sync);
    return () => document.removeEventListener("visibilitychange", sync);
  }, []);
  const isActive = (href: string) => href === "/" ? pathname === "/" : pathname.startsWith(href);
  return (
    <div className={`app nook-app paper-lamp${quiet ? " immersive" : ""}${chat ? " nook-chat-app" : ""}`}
      data-luna-motion={preferences.animateLuna === false ? "off" : "on"}>
      <Ambient />
      <a className="skip-link" href="#main">Skip to content</a>
      {/* Chat is its own quiet room with a compact header, so the site navigation steps away. */}
      {!quiet && !chat && <header className="nook-navigation">
        <Link className="nook-brand" href="/"><Icon name="leaf" /><span>JournalPulse</span></Link>
        <nav className="nook-links" aria-label="Main">
          {navigation.map((item) => <Link key={item.href} href={item.href}
            aria-current={isActive(item.href) ? "page" : undefined}>
            <Icon name={item.icon} /><span>{item.label}</span>
          </Link>)}
        </nav>
      </header>}
      <main id="main" className="app-main">
        <SystemStatus />
        <AuthBoundary>{children}</AuthBoundary>
      </main>
      {!quiet && !chat && <nav className="tab-bar" aria-label="Main">
        {navigation.filter((item) => TAB_BAR.has(item.href)).map((item) => <Link key={item.href} href={item.href}
          aria-current={isActive(item.href) ? "page" : undefined}>
          <Icon name={item.icon} /><span>{item.label}</span>
        </Link>)}
      </nav>}
    </div>
  );
}
