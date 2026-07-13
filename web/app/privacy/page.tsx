"use client";

import Link from "next/link";
import { useState } from "react";

import { apiRequest } from "@/lib/api";

const DELETE_PHRASE = "delete my journal data";

export default function PrivacyPage() {
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function exportData() {
    setError("");
    try {
      const payload = await apiRequest<Record<string, unknown>>("/v1/export");
      const href = URL.createObjectURL(
        new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
      );
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = "journalpulse-export.json";
      anchor.click();
      URL.revokeObjectURL(href);
      setMessage("Your export was created locally in this browser.");
    } catch {
      setError("The export could not be created. Your stored data was not changed.");
    }
  }

  async function deleteData() {
    if (confirmation !== DELETE_PHRASE) return;
    setBusy(true);
    setError("");
    try {
      const result = await apiRequest<{ deleted_records: number }>("/v1/account/data", {
        method: "DELETE",
      });
      setConfirmation("");
      setMessage(`${result.deleted_records} stored records were deleted. Your sign-in remains active.`);
    } catch {
      setError("Deletion did not complete. No success is assumed; please retry.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-wrap narrow reveal">
      <header className="page-header">
        <div>
          <span className="kicker">Data boundary</span>
          <h1>Your reflections are not the product.</h1>
        </div>
      </header>
      <section className="flow-sheet">
        <span className="folio">Processing choices</span>
        <div className="privacy-list">
          <article>
            <strong>AI analysis</strong>
            <p>Opt in per reflection. Requests require zero-data-retention routing.</p>
          </article>
          <article>
            <strong>Original text</strong>
            <p>Off by default. You may save only the structured state you approved.</p>
          </article>
          <article>
            <strong>Personal memory</strong>
            <p>Disabled until retrieval, provenance, and deletion behavior are validated.</p>
          </article>
          <article>
            <strong>Safety</strong>
            <p>Support routing happens before AI analysis, retrieval, or adaptive selection.</p>
          </article>
        </div>
        <div className="button-row">
          <button className="button secondary" onClick={exportData} type="button">
            Export my data
          </button>
          <Link className="text-button" href="/memory">
            Inspect memory status
          </Link>
        </div>
      </section>
      <section className="flow-sheet danger-zone" aria-labelledby="delete-data-heading">
        <span className="folio">Irreversible</span>
        <h2 id="delete-data-heading">Delete all journal data</h2>
        <p>
          This removes reflections, outcomes, model records, safety events, and memories. It does
          not remove your sign-in identity.
        </p>
        <label className="field-label">
          Type “{DELETE_PHRASE}” to confirm
          <input
            autoComplete="off"
            onChange={(event) => setConfirmation(event.target.value)}
            value={confirmation}
          />
        </label>
        <button
          className="button urgent"
          disabled={busy || confirmation !== DELETE_PHRASE}
          onClick={deleteData}
          type="button"
        >
          {busy ? "Deleting…" : "Delete journal data"}
        </button>
      </section>
      {message && <p className="success-note" role="status">{message}</p>}
      {error && <p className="error-note" role="alert">{error}</p>}
    </div>
  );
}
