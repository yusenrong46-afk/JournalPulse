import type { Metadata } from "next";
import Link from "next/link";
import { PublicInformation } from "@/components/public-information";

export const metadata: Metadata = { title: "Terms of use" };

export default function TermsPage() {
  return (
    <PublicInformation title="Terms of use">
      <section><h2>A research beta</h2>
        <p>JournalPulse is an experimental tool for journaling and reflection. Features may change, fail, or become unavailable. Keep your own copy of writing you need to preserve.</p>
      </section>
      <section><h2>Its intended use</h2>
        <p>Use the app voluntarily for personal reflection. It is not therapy, diagnosis, treatment, medical advice, or crisis care. AI replies and suggested activities can be inaccurate or unsuitable; you can decline them.</p>
      </section>
      <section><h2>Your account and writing</h2>
        <p>Keep access to your email and sign-in links secure. Only submit material you have permission to share. Avoid adding other people’s private information. You retain ownership of your writing; using the features requires the processing described in the <Link href="/privacy">privacy notice</Link>.</p>
      </section>
      <section><h2>Responsible use</h2>
        <p>Do not attempt to access another person’s records, bypass access controls or usage limits, or disrupt the service. External resources are operated by other organizations and have their own terms.</p>
      </section>
      <section><h2>Leaving the beta</h2>
        <p>You can stop using JournalPulse at any time. Settings provides download, saved-data deletion, and a separate Close account control. Deleting saved data keeps the sign-in identity; closing the account removes it after a fresh sign-in and explicit confirmation. Backup and service-log retention limits are described in the privacy notice.</p>
      </section>
      <section><h2>Contact</h2>
        <p>For privacy questions or help closing your account, email <a href="mailto:yusenrong46@gmail.com">yusenrong46@gmail.com</a>. Please do not include journal entries, passwords, or sign-in links in your message.</p>
      </section>
    </PublicInformation>
  );
}
