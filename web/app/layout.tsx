import type { Metadata, Viewport } from "next";
import { Inter, Nunito } from "next/font/google";

import { AppShell } from "@/components/app-shell";
import { ServiceWorkerRegistration } from "@/components/service-worker";

import "./globals.css";

const display = Nunito({
  subsets: ["latin"],
  weight: ["600", "700", "800"],
  variable: "--font-display",
  display: "swap",
});

const body = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "JournalPulse", template: "%s · JournalPulse" },
  description: "Check in with Luna, a calm companion for one small step at a time.",
  applicationName: "JournalPulse",
};

// resizes-content lets Android Chrome shrink the page for the keyboard; /talk also follows the
// visual viewport for iOS, which ignores this hint.
export const viewport: Viewport = { themeColor: "#fbf5ec", colorScheme: "light", interactiveWidget: "resizes-content" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>
        <AppShell>{children}</AppShell>
        <ServiceWorkerRegistration />
      </body>
    </html>
  );
}
