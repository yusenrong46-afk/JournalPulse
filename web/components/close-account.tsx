"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { closureReceipt } from "@/lib/account-closure";
import { invalidateAccountDataRequests } from "@/lib/account-data";
import { activateBrowserAccount, browserAccount, clearClosedAccountStorage } from "@/lib/account-storage";
import { apiRequest, ApiError } from "@/lib/api";
import { getSupabase } from "@/lib/supabase";
import { clearTabSession } from "@/lib/tab-session";

const PHRASE = "delete my account";

export function CloseAccount({ onBusyChange, disabled = false }: { onBusyChange: (busy: boolean) => void; disabled?: boolean }) {
  const router = useRouter();
  const [available, setAvailable] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const mounted = useRef(true);
  const inFlight = useRef(false);
  const receipt = useRef<ReturnType<typeof closureReceipt> | null>(null);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    apiRequest<{ account_closure?: string }>("/v1/system/status", { signal: controller.signal, retry: false })
      .then((status) => { if (mounted.current) setAvailable(status.account_closure === "configured"); })
      .catch(() => { /* contact remains available if configuration cannot be checked */ });
    return () => { mounted.current = false; controller.abort(); };
  }, []);

  function start() {
    if (inFlight.current || disabled) return false;
    inFlight.current = true;
    setBusy(true); onBusyChange(true); setError(""); setMessage("");
    return true;
  }
  function finish() {
    inFlight.current = false;
    if (mounted.current) { setBusy(false); onBusyChange(false); }
  }

  async function freshSignIn() {
    if (!start()) return;
    const owner = browserAccount();
    try {
      const client = await getSupabase();
      const user = await client?.auth.getUser();
      if (!mounted.current || browserAccount() !== owner) return;
      if (!client || user?.error || !user?.data.user?.email || user.data.user.id !== owner) {
        throw new Error("No verified email");
      }
      const result = await client.auth.signInWithOtp({ email: user.data.user.email, options: {
        shouldCreateUser: false, emailRedirectTo: `${window.location.origin}/me/`,
      } });
      if (result.error) throw result.error;
      if (mounted.current) setMessage("A fresh sign-in link was sent. Open it, return to Settings, and confirm account closure within five minutes.");
    } catch {
      if (mounted.current) setError("The fresh sign-in link could not be sent. No account closure was started.");
    } finally { finish(); }
  }

  async function closeAccount() {
    if (confirmation.trim().toLowerCase() !== PHRASE || !available || !start()) return;
    const owner = browserAccount();
    let confirmed = false;
    try {
      const client = await getSupabase();
      if (!mounted.current || !client || typeof owner !== "string" || browserAccount() !== owner) return;
      receipt.current ??= closureReceipt();
      await apiRequest<void>("/v1/account", { method: "DELETE", retry: false, timeoutMs: 90_000,
        body: JSON.stringify({ confirmation: PHRASE, ...receipt.current }) });
      confirmed = true;
      if (browserAccount() !== owner) return;
      invalidateAccountDataRequests();
      clearTabSession(owner);
      clearClosedAccountStorage(owner);
      if (mounted.current) setMessage("Your account is closed. Signing out of this browser…");
      const signedOut = await client.auth.signOut({ scope: "local" });
      if (signedOut.error) throw signedOut.error;
      if (browserAccount() === owner) activateBrowserAccount(null);
      router.replace("/login");
    } catch (reason) {
      if (mounted.current) setError(confirmed
        ? "Your account was closed, but browser sign-out did not finish. Close this tab."
        : reason instanceof ApiError ? reason.message
        : "Account closure could not be confirmed. Keep this page open and retry, or contact us.");
    } finally { finish(); }
  }

  return (
    <details className="explain danger-card">
      <summary>Close account</summary>
      <div className="stack">
        <h3>Permanently close your account</h3>
        <p>This removes your sign-in account and saved app data. You cannot undo it. Download anything you want to keep first.</p>
        <p>Backup and service-log limits still apply. A minimal deletion record is retained to prevent an old backup from restoring your account. <Link href="/privacy">Read the privacy notice</Link>.</p>
        <p>A fresh sign-in is required. Once closure starts, new writing is paused until it finishes. If a response is lost, retry from this page.</p>
        {!available && <p>Online account closure is currently unavailable. Contact <a href="mailto:yusenrong46@gmail.com">yusenrong46@gmail.com</a> for help.</p>}
        <button type="button" className="btn btn-soft" disabled={busy || disabled || !available} onClick={() => void freshSignIn()}>Send a fresh sign-in link</button>
        <label className="text-field">Type “{PHRASE}” to confirm
          <input autoComplete="off" value={confirmation} disabled={busy || disabled}
            onChange={(event) => setConfirmation(event.target.value)} />
        </label>
        <button type="button" className="btn btn-danger" disabled={busy || disabled || !available || confirmation.trim().toLowerCase() !== PHRASE}
          onClick={() => void closeAccount()}>{busy ? "Working…" : "Close my account"}</button>
        {message && <p className="note ok" role="status">{message}</p>}
        {error && <p className="note error" role="alert">{error}</p>}
      </div>
    </details>
  );
}
