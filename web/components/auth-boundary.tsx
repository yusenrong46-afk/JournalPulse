"use client";

import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import { getSupabase, isSupabaseConfigured } from "@/lib/supabase";

const PUBLIC_PATHS = new Set(["/login", "/welcome"]);

export function AuthBoundary({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const configured = isSupabaseConfigured();
  const publicPath = PUBLIC_PATHS.has(pathname);
  const [authorizedPath, setAuthorizedPath] = useState<string | null>(null);

  useEffect(() => {
    if (!configured || publicPath) return;

    let active = true;
    let unsubscribe: () => void = () => undefined;
    getSupabase().then((client) => {
      if (!active || !client) return;
      client.auth.getSession().then(({ data }) => {
        if (!active) return;
        if (data.session) setAuthorizedPath(pathname);
        else router.replace(`/login?next=${encodeURIComponent(pathname)}`);
      });
      const { data: subscription } = client.auth.onAuthStateChange((_event, session) => {
        if (!active) return;
        if (session) setAuthorizedPath(pathname);
        else {
          setAuthorizedPath(null);
          router.replace(`/login?next=${encodeURIComponent(pathname)}`);
        }
      });
      unsubscribe = () => subscription.subscription.unsubscribe();
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, [configured, pathname, publicPath, router]);

  if (!configured || publicPath) return children;
  if (authorizedPath !== pathname) {
    return (
      <div className="page-wrap narrow" aria-busy="true">
        <section className="paper-card skeleton-card" aria-label="Checking your private session" />
      </div>
    );
  }
  return children;
}
