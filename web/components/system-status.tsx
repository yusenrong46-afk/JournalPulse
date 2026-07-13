"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

import { apiRequest } from "@/lib/api";
import type { SystemStatus as Status } from "@/lib/types";

export function SystemStatus() {
  const pathname = usePathname();
  const [status, setStatus] = useState<Status | null>(null);
  const [online, setOnline] = useState(true);

  useEffect(() => {
    if (pathname === "/login" || pathname === "/welcome") return;
    const updateConnection = () => setOnline(navigator.onLine);
    updateConnection();
    window.addEventListener("online", updateConnection);
    window.addEventListener("offline", updateConnection);
    apiRequest<Status>("/v1/system/status").then(setStatus).catch(() => setOnline(false));
    return () => {
      window.removeEventListener("online", updateConnection);
      window.removeEventListener("offline", updateConnection);
    };
  }, [pathname]);

  if (pathname === "/login" || pathname === "/welcome") return null;
  if (!online) {
    return <div className="service-strip warning" role="status"><span className="service-indicator" aria-hidden="true" /><span><strong>Connection paused</strong> Unsaved writing stays on this page.</span></div>;
  }
  if (!status) {
    return <div className="service-strip checking" role="status"><span className="service-indicator" aria-hidden="true" /><span><strong>Privacy first</strong> Checking your processing settings.</span></div>;
  }
  if (status.analysis_mode === "ai_configured") {
    return <div className="service-strip ready" role="status"><span className="service-indicator" aria-hidden="true" /><span><strong>Private analysis ready</strong> You approve the final state before anything is saved.</span></div>;
  }
  return <div className="service-strip" role="status"><span className="service-indicator" aria-hidden="true" /><span><strong>Local reflection mode</strong> {status.message}</span></div>;
}
