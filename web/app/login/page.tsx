"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { Luna } from "@/components/luna";
import { getSupabase } from "@/lib/supabase";

function LoginWorkspace() {
  const searchParams = useSearchParams();
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const mounted = useRef(true);
  const inFlight = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const client = await getSupabase();
      // A delayed SDK import must not send mail after this page has been left.
      if (!mounted.current) return;
      if (!client) {
        setError("Sign-in isn’t set up here yet.");
        return;
      }
      const requestedPath = searchParams.get("next");
      const nextPath = requestedPath?.startsWith("/") && !requestedPath.startsWith("//") ? requestedPath : "/";
      const { error: signInError } = await client.auth.signInWithOtp({
        email: email.trim(),
        options: { emailRedirectTo: `${window.location.origin}${nextPath}` },
      });
      if (!mounted.current) return;
      if (signInError) setError("Luna couldn’t send the link. Check the address and try again.");
      else setMessage("Check your email and tap the link to come in. You can close this tab.");
    } catch {
      if (mounted.current) setError("Luna couldn’t send the link. Check your connection and try again.");
    } finally {
      inFlight.current = false;
      if (mounted.current) setLoading(false);
    }
  }

  return (
    <div className="focus-page">
      <Luna mood={message ? "proud" : error ? "oops" : "idle"} size={150} />
      <h1>{message ? "Link on its way!" : "Welcome to JournalPulse"}</h1>
      <p>{message || "No password needed. Luna will email you a sign-in link."}</p>
      {!message && (
        <form className="stack" onSubmit={submit}>
          <label className="text-field" style={{ textAlign: "left" }}>
            Email
            <input
              autoComplete="email"
              inputMode="email"
              type="email"
              required
              disabled={loading}
              value={email}
              placeholder="you@example.com"
              onChange={(event) => setEmail(event.target.value)}
            />
          </label>
          <button className="btn btn-primary btn-big btn-block" disabled={loading}>
            {loading ? "Sending…" : "Email me a link"}
          </button>
        </form>
      )}
      {message && (
        <button className="btn btn-ghost" type="button" onClick={() => setMessage("")}>Use a different email</button>
      )}
      {error && <p className="note error" role="alert">{error}</p>}
      <p className="small">Luna is a companion, not a therapist.</p>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="loading-luna"><Luna mood="idle" size={100} decorative /></div>}>
      <LoginWorkspace />
    </Suspense>
  );
}
