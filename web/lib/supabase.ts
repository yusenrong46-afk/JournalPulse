import type { SupabaseClient } from "@supabase/supabase-js";

let clientPromise: Promise<SupabaseClient | null> | undefined;

export function isSupabaseConfigured(): boolean {
  return Boolean(process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY);
}

export function getSupabase(): Promise<SupabaseClient | null> {
  if (clientPromise) return clientPromise;
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) {
    clientPromise = Promise.resolve(null);
    return clientPromise;
  }
  clientPromise = import("@supabase/supabase-js")
    .then(({ createClient }) => createClient(url, key))
    .catch((error) => {
      // A failed SDK load or initialization is not a permanent sign-in state.
      // Keep concurrent callers on one promise, but let the next attempt retry.
      clientPromise = undefined;
      throw error;
    });
  return clientPromise;
}
