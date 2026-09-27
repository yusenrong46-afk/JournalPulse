"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { Luna } from "@/components/luna";

/** Client-side redirect for retired routes; a static export cannot answer with HTTP redirects. */
export function Redirect({ to, keepQuery = false }: { to: string; keepQuery?: boolean }) {
  const router = useRouter();
  useEffect(() => {
    const query = keepQuery ? window.location.search : "";
    router.replace(`${to}${query}`);
  }, [keepQuery, router, to]);
  return (
    <div className="loading-luna" role="status">
      <Luna mood="idle" size={90} decorative />
      <span>One moment…</span>
    </div>
  );
}
