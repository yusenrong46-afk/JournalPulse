"use client";

import Link from "next/link";
import { useEffect } from "react";

export default function ErrorPage({ error, reset }: { error: Error; reset: () => void }) {
  useEffect(() => {
    // Keep private entry content out of telemetry; only the error class is retained locally.
    console.error("JournalPulse page error", error.name);
  }, [error.name]);

  return (
    <div className="page-wrap narrow">
      <section className="paper-card empty-card">
        <span className="folio">Recovery note</span>
        <h2>This page lost its place.</h2>
        <p>Your journal text was not included in this error report. Retry the page or return to Today.</p>
        <div className="button-row">
          <button className="button primary" type="button" onClick={reset}>Try this page again</button>
          <Link className="button secondary" href="/">Return to Today</Link>
        </div>
      </section>
    </div>
  );
}
