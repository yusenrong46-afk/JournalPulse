import { Luna } from "@/components/luna";

export default function Loading() {
  return (
    <div className="loading-luna" role="status" aria-busy="true">
      <Luna mood="idle" size={96} decorative />
      <span>One moment…</span>
    </div>
  );
}
