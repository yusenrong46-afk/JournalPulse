"use client";

// Portfolio demo sandbox. The API decides everything that matters (an isolated
// identity, fictional seed data, no model or search request); this file only starts a
// demo session in the browser and tells the shell when the current identity is one.
import { useEffect, useState } from "react";

import { ACCOUNT_CHANGED_EVENT } from "./account-storage";
import { apiBase } from "./api";
import { getSupabase, isSupabaseConfigured } from "./supabase";

export const DEMO_LABEL = "Demo: fictional data, AI off";
// Must match LOCAL_DEMO_PREFIX in src/journalpulse/auth.py.
export const LOCAL_DEMO_PREFIX = "de300000-";
const CLIENT_ID_KEY = "journalpulse-preview-client-id";
const PREFERENCES_KEY = "journalpulse_preferences_v1";
const DEMO_STARTED_EVENT = "journalpulse-demo-started";

export function demoEnabled(): boolean {
  return process.env.NEXT_PUBLIC_JOURNALPULSE_DEMO_ENABLED === "true";
}

export function isLocalDemoUser(userId: string | null | undefined): boolean {
  return typeof userId === "string" && userId.startsWith(LOCAL_DEMO_PREFIX);
}

type DemoSession = {
  user_id: string;
  label: string;
  access_token: string | null;
  refresh_token: string | null;
};

export class DemoUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DemoUnavailableError";
  }
}

const DEMO_PREFERENCES = {
  onboarded: true,
  llmConsent: false,
  retainText: false,
  encryptedDrafts: false,
  followUpMinutes: 10,
  locale: "CA",
};

/** Create a demo identity on the API and switch this browser to it. */
export async function startDemo(fetcher: typeof fetch = fetch): Promise<string> {
  let response: Response;
  try {
    response = await fetcher(`${apiBase()}/v1/demo/sessions`, {
      method: "POST",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(30_000),
    });
  } catch {
    throw new DemoUnavailableError("The demo could not start. Check your connection and try again.");
  }
  if (response.status === 429) {
    throw new DemoUnavailableError("Too many demo sessions from this network. Please try again later.");
  }
  if (!response.ok) throw new DemoUnavailableError("The demo is unavailable right now. Please try again later.");
  const session = (await response.json()) as DemoSession;
  if (session.access_token) {
    const client = await getSupabase();
    if (!client || !session.refresh_token) throw new DemoUnavailableError("The demo is unavailable right now.");
    const { error } = await client.auth.setSession({
      access_token: session.access_token,
      refresh_token: session.refresh_token,
    });
    if (error) throw new DemoUnavailableError("The demo is unavailable right now. Please try again later.");
    window.localStorage.setItem(`${PREFERENCES_KEY}:${session.user_id}`, JSON.stringify(DEMO_PREFERENCES));
  } else {
    // Local development: the API trusts this header-borne identity, as for any local user.
    window.localStorage.setItem(CLIENT_ID_KEY, session.user_id);
    window.localStorage.setItem(PREFERENCES_KEY, JSON.stringify(DEMO_PREFERENCES));
  }
  window.dispatchEvent(new Event(DEMO_STARTED_EVENT));
  return session.user_id;
}

async function currentIdentityIsDemo(): Promise<boolean> {
  if (!isSupabaseConfigured()) {
    try {
      return isLocalDemoUser(window.localStorage.getItem(CLIENT_ID_KEY));
    } catch {
      return false;
    }
  }
  const client = await getSupabase();
  if (!client) return false;
  const { data } = await client.auth.getSession();
  return data.session?.user.is_anonymous === true;
}

/** True while the browser acts as a demo identity, so every view can say so. */
export function useDemoIdentity(): boolean {
  const [demo, setDemo] = useState(false);
  useEffect(() => {
    let active = true;
    const refresh = () => {
      currentIdentityIsDemo().then((value) => { if (active) setDemo(value); }).catch(() => undefined);
    };
    refresh();
    window.addEventListener(ACCOUNT_CHANGED_EVENT, refresh);
    window.addEventListener(DEMO_STARTED_EVENT, refresh);
    window.addEventListener("storage", refresh);
    return () => {
      active = false;
      window.removeEventListener(ACCOUNT_CHANGED_EVENT, refresh);
      window.removeEventListener(DEMO_STARTED_EVENT, refresh);
      window.removeEventListener("storage", refresh);
    };
  }, []);
  return demo;
}
