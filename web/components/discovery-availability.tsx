"use client";

export function DiscoveryAvailability({ status, onRetry }: {
  status: "loading" | "configured" | "unavailable" | "unknown"; onRetry(): void;
}) {
  if (status === "configured") return null;
  return <div className="note" role="status">
    <p>{status === "loading" ? "Checking web search availability…" : status === "unknown"
      ? "We couldn’t check web search right now. You can browse reviewed app activities below."
      : "Web search isn’t available right now. You can browse reviewed app activities below."}</p>
    {status !== "loading" && <button className="btn btn-soft" type="button" onClick={onRetry}>Check search availability again</button>}
  </div>;
}
