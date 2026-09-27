"use client";

import { useEffect, useState } from "react";

import { Luna } from "@/components/luna";

/** Only speaks up when the device goes offline; everything else is shown where it matters. */
export function SystemStatus() {
  const [online, setOnline] = useState(true);

  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    update();
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, []);

  if (online) return null;
  return (
    <div className="status-banner" role="status">
      <Luna mood="oops" size={34} decorative />
      You’re offline. Nothing new will be sent until you reconnect.
    </div>
  );
}
