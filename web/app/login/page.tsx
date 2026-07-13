"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { getSupabase } from "@/lib/supabase";

function LoginWorkspace() {
  const searchParams = useSearchParams();
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const client = await getSupabase();
    if (!client) {
      setError("Private sign-in is not configured in this environment.");
      return;
    }
    setLoading(true);
    setError("");
    setMessage("");
    const requestedPath = searchParams.get("next");
    const nextPath = requestedPath?.startsWith("/") ? requestedPath : "/";
    const { error: signInError } = await client.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}${nextPath}` },
    });
    if (signInError) setError("The private link could not be sent. Please check the address and retry.");
    else setMessage("Check your email for a single-use sign-in link. You can close this tab afterward.");
    setLoading(false);
  }

  return (
    <div className="page-wrap narrow reveal">
      <section className="flow-sheet login-sheet">
        <span className="folio">Invitation-only beta</span>
        <h1>Enter your private field journal.</h1>
        <p>No password to remember. We will send a single-use sign-in link.</p>
        <form onSubmit={submit}>
          <label className="field-label">
            Email address
            <input
              autoComplete="email"
              inputMode="email"
              type="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </label>
          <button className="button primary" disabled={loading}>
            {loading ? "Sending a private link…" : "Send private link"}
          </button>
        </form>
        {message && <p className="success-note" role="status">{message}</p>}
        {error && <p className="error-note" role="alert">{error}</p>}
      </section>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="page-wrap narrow"><section className="paper-card skeleton-card" /></div>}>
      <LoginWorkspace />
    </Suspense>
  );
}
