"use client";

import { useEffect, useState } from "react";
import { apiRequest } from "./api";

export function useDiscoveryCapability() {
  const [status, setStatus] = useState<"loading" | "configured" | "unavailable" | "unknown">("loading");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    void apiRequest<{ discovery: "configured" | "unavailable" }>("/v1/capabilities", { signal: controller.signal })
      .then((result) => { if (!controller.signal.aborted) setStatus(result?.discovery === "configured" ? "configured" : "unavailable"); })
      .catch(() => { if (!controller.signal.aborted) setStatus("unknown"); });
    return () => controller.abort();
  }, [attempt]);
  return { status, retry: () => { setStatus("loading"); setAttempt((value) => value + 1); } };
}
