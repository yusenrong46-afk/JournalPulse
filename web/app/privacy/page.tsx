import type { Metadata } from "next";
import Link from "next/link";
import { PublicInformation } from "@/components/public-information";

export const metadata: Metadata = { title: "Privacy" };

export default function PrivacyPage() {
  return (
    <PublicInformation title="Your privacy">
      <p>This notice describes the JournalPulse research beta. Please understand these limits before adding sensitive information.</p>
      <section><h2>What is stored</h2>
        <p>Supabase handles email sign-in and hosts the app’s database. Saved journal entries, chat summaries, reported feelings, activity choices, and feedback stay until you delete them. The app also stores consent choices and technical records needed to operate it.</p>
        <p>Open chat messages are stored so you can resume a conversation. With “Keep my messages” off, message text is cleared when the chat is accepted or closed. Idle chats are scheduled for cleanup after 24 hours; a delayed or failed cleanup job can delay removal. Summaries and activity records remain even when message text is cleared.</p>
        <p>With “Keep my messages” on, chat text remains until you delete it. Saved journal entries have their own lifetime and are not removed by turning this setting off.</p>
      </section>
      <section><h2>What is sent to AI and search services</h2>
        <p>When you enable AI, the conversation and any journal entry you explicitly share can be sent through OpenRouter to the configured AI provider. Requests require zero-data-retention routing. That provider requirement does not mean JournalPulse stores no data.</p>
        <p>Simple Luna uses scripted replies without model calls. Optional web search requires separate permission and sends a general activity query to Brave Search. The app constructs that query from allowed topics, rather than forwarding your journal or chat text.</p>
        <p><Link href="/how-ai-works">Read more about data and AI</Link>.</p>
      </section>
      <section><h2>On your device</h2>
        <p>Your browser holds sign-in session information and preferences. Unsaved journal drafts are kept for the browser tab session so they can survive a refresh. Downloaded exports are files on your device; deleting data in the app does not delete those files.</p>
      </section>
      <section><h2>Access and technical records</h2>
        <p>The app uses account ownership checks to limit access to saved records. It is not end-to-end encrypted: the server processes writing, and authorized infrastructure operators may have access. AI processing also requires sending the selected text to the provider.</p>
        <p>Application request logs record technical information such as request identifiers, paths, status, and timing. The application is designed not to log journal text or API keys. Hosting and authentication services may keep their own operational records.</p>
      </section>
      <section><h2>Download and delete</h2>
        <p>After signing in, use Settings to download your data or delete saved journal data. The deletion control removes journal and chat content from the active app database. It keeps your sign-in identity; it is not account closure.</p>
        <p>Limited technical records remain to prevent delayed requests from recreating deleted content and to enforce usage limits. Deletion markers contain account and object identifiers, not your writing.</p>
        <p>Deleting active records does not promise immediate removal from infrastructure backups or service logs. Backup retention and a hosted restore exercise have not yet been verified for this beta. Do not rely on JournalPulse as your only copy of important writing.</p>
      </section>
    </PublicInformation>
  );
}
