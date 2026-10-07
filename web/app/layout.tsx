import type { Metadata, Viewport } from "next";

import { AppShell } from "@/components/app-shell";
import { ServiceWorkerRegistration } from "@/components/service-worker";

import "./globals.css";
import "@fontsource-variable/dm-sans";
import "./quiet-nook.css";
// Paper & Lamp prototype: a reading serif for words people read or write, loaded locally.
import "@fontsource-variable/newsreader/opsz.css";
import "@fontsource-variable/newsreader/opsz-italic.css";
import "./paper-lamp.css";

export const metadata: Metadata = {
  title: { default: "JournalPulse", template: "%s · JournalPulse" },
  description: "Check in with Luna, a calm companion for one small step at a time.",
  applicationName: "JournalPulse",
};

// resizes-content lets Android Chrome shrink the page for the keyboard; /talk also follows the
// visual viewport for iOS, which ignores this hint.
export const viewport: Viewport = { themeColor: "#f0f3ed", colorScheme: "light", interactiveWidget: "resizes-content" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <AppShell>{children}</AppShell>
        <ServiceWorkerRegistration />
      </body>
    </html>
  );
}
