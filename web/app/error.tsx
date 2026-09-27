"use client";

import Link from "next/link";
import { useEffect } from "react";

import { Luna } from "@/components/luna";

export default function ErrorPage({ error, reset }: { error: Error; reset: () => void }) {
  useEffect(() => {
    // Keep private entry content out of telemetry; only the error class is retained locally.
    console.error("JournalPulse page error", error.name);
  }, [error.name]);

  return (
    <div className="focus-page">
      <Luna mood="oops" size={140} />
      <h1>Oops, Luna tripped.</h1>
      <p>Something went wrong on this page. Nothing you wrote was included in the error.</p>
      <div className="stack">
        <button className="btn btn-primary btn-block" type="button" onClick={reset}>Try again</button>
        <Link className="btn btn-ghost btn-block" href="/">Go home</Link>
      </div>
    </div>
  );
}
