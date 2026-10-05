"use client";

import { useRouter } from "next/navigation";
import { Fragment, type ReactNode, useEffect, useState } from "react";

import { Luna } from "@/components/luna";
import { activateBrowserAccount } from "@/lib/account-storage";
import { useRoutePath } from "@/lib/route-path";
import { getSupabase, isSupabaseConfigured } from "@/lib/supabase";

const PUBLIC_PATHS = new Set(["/login", "/welcome"]);

export function AuthBoundary({ children }: { children: ReactNode }) {
  const pathname = useRoutePath();
  const router = useRouter();
  const configured = isSupabaseConfigured();
  const publicPath = PUBLIC_PATHS.has(pathname);
  const [authorization, setAuthorization] = useState<{ path: string; userId: string | null } | null>(null);
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    if (!configured) return;

    let active = true;
    let unsubscribe: () => void = () => undefined;
    let sessionRevision = 0;
    function acceptSession(userId: string | null) {
      if (!active) return;
      activateBrowserAccount(userId);
      setError(false);
      setAuthorization({ path: pathname, userId });
      if (!userId && !publicPath) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    }
    getSupabase().then((client) => {
      if (!active || !client) return;
      const { data: subscription } = client.auth.onAuthStateChange((_event, session) => {
        sessionRevision += 1;
        acceptSession(session?.user.id ?? null);
      });
      unsubscribe = () => subscription.subscription.unsubscribe();
      const initialRevision = sessionRevision;
      client.auth.getSession().then(({ data, error: sessionError }) => {
        // An auth event is newer than the asynchronous initial snapshot.
        if (!active || initialRevision !== sessionRevision) return;
        if (sessionError) setError(true);
        else acceptSession(data.session?.user.id ?? null);
      }).catch(() => { if (active && initialRevision === sessionRevision) setError(true); });
    }).catch(() => { if (active) setError(true); });
    return () => {
      active = false;
      unsubscribe();
    };
  }, [configured, pathname, publicPath, retry, router]);

  if (!configured) return children;
  if (error) {
    return <div className="loading-luna" role="alert"><span>We couldn’t check your sign-in.</span><button className="btn btn-soft" type="button" onClick={() => setRetry((value) => value + 1)}>Try again</button></div>;
  }
  if (authorization?.path !== pathname || (!publicPath && !authorization.userId)) {
    return (
      <div className="loading-luna" role="status" aria-busy="true">
        <Luna mood="sleepy" size={96} decorative />
        <span>Waking Luna up…</span>
      </div>
    );
  }
  // The account key discards private component state, pending drafts and responses
  // when a second account signs in on the same route.
  return <Fragment key={authorization.userId ?? "guest"}>{children}</Fragment>;
}
