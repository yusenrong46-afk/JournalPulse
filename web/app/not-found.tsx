import Link from "next/link";

export default function NotFound() {
  return (
    <div className="page-wrap narrow">
      <section className="paper-card empty-card">
        <span className="folio">Missing page</span>
        <h2>This field note does not exist.</h2>
        <p>The address may be outdated. No journal data was changed.</p>
        <Link className="button primary" href="/">Return to Today</Link>
      </section>
    </div>
  );
}
