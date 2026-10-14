import Link from "next/link";
import type { ReactNode } from "react";

export function PublicInformationLinks() {
  return (
    <nav className="public-information-links" aria-label="About JournalPulse">
      <Link href="/privacy">Privacy</Link>
      <Link href="/terms">Terms of use</Link>
      <Link href="/how-ai-works">Data &amp; AI</Link>
    </nav>
  );
}

export function CrisisSupport() {
  return (
    <aside className="public-support" aria-label="Crisis support">
      <p>JournalPulse is not a crisis service. Messages are not monitored by a crisis responder.</p>
      <p>In Canada, call <a href="tel:988?oai_link_source=model_response_hotline">988</a> or text <a href="sms:988?oai_link_source=model_response_hotline">988</a> for the 9-8-8 Suicide Crisis Helpline, available 24/7 in English and French. <a href="https://988.ca/?oai_link_source=model_response_hotline" target="_blank" rel="noreferrer">Visit 9-8-8</a>.</p>
      <p>Outside Canada, contact a crisis service or emergency service in your country.</p>
    </aside>
  );
}

export function PublicInformation({ title, children }: { title: string; children: ReactNode }) {
  return (
    <article className="page public-information">
      <header className="page-head">
        <Link href="/login">JournalPulse · Sign in</Link>
        <p className="eyebrow">Deployed research beta</p>
        <h1>{title}</h1>
        <p className="small muted">Updated October 14, 2026</p>
        <PublicInformationLinks />
      </header>
      <div className="public-information-body">{children}</div>
      <CrisisSupport />
    </article>
  );
}
