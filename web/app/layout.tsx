import type { Metadata, Viewport } from "next";

import { AppShell } from "@/components/app-shell";
import { ServiceWorkerRegistration } from "@/components/service-worker";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "JournalPulse", template: "%s · JournalPulse" },
  description: "A personal laboratory for adaptive reflection.",
  applicationName: "JournalPulse",
};

export const viewport: Viewport = { themeColor: "#f1ecdf", colorScheme: "light" };

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
