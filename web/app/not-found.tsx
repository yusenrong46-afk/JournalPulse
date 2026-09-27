import Link from "next/link";

import { Luna } from "@/components/luna";

export default function NotFound() {
  return (
    <div className="focus-page">
      <Luna mood="thinking" size={140} />
      <h1>Hmm, this page wandered off.</h1>
      <p>The link may be old. Nothing in your journal changed.</p>
      <Link className="btn btn-primary btn-block" href="/">Go home</Link>
    </div>
  );
}
