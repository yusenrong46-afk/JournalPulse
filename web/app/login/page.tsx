"use client";

import { useState } from "react";

import { getSupabase } from "@/lib/supabase";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  async function submit(event: React.FormEvent) { event.preventDefault(); const client = getSupabase(); if (!client) return setMessage("Supabase is not configured in this environment."); const { error } = await client.auth.signInWithOtp({ email, options: { emailRedirectTo: window.location.origin } }); setMessage(error ? error.message : "Check your email for a private sign-in link."); }
  return <div className="page-wrap narrow reveal"><section className="flow-sheet login-sheet"><span className="folio">Invitation-only beta</span><h1>Enter your private field journal.</h1><p>No password. We will send a single-use link.</p><form onSubmit={submit}><label className="field-label">Email<input type="email" required value={email} onChange={(event) => setEmail(event.target.value)} /></label><button className="button primary">Send private link</button></form>{message && <p className="method-note">{message}</p>}</section></div>;
}
