import type { Metadata } from "next";
import { PublicInformation } from "@/components/public-information";

export const metadata: Metadata = { title: "Data & AI" };

export default function HowAIWorksPage() {
  return (
    <PublicInformation title="How data and AI are used">
      <section><h2>English only</h2>
        <p>JournalPulse currently supports English only. Please use English when chatting with Luna or sharing a journal entry for AI reflection. Replies and safety checks may misunderstand other languages.</p>
        <p>Using English does not guarantee that Luna will understand you correctly or recognize distress.</p>
      </section>
      <section><h2>You choose whether to use AI</h2>
        <p>Simple Luna offers a scripted conversation without sending text to a model. Smart Luna uses an AI model through OpenRouter. AI and message-retention choices apply to new chats; changing a setting does not rewrite earlier chats.</p>
      </section>
      <section><h2>What Luna can read</h2>
        <p>AI replies use the current conversation and, if you choose to share one, a specific journal entry. Luna does not automatically read your whole journal. A saved summary or suggestion can still contain personal information.</p>
      </section>
      <section><h2>Suggestions can be wrong</h2>
        <p>Luna can misunderstand you, invent details, or suggest an unsuitable activity. You can decline suggestions or stop an activity. JournalPulse does not diagnose conditions, provide treatment, or establish that an activity will improve your wellbeing.</p>
        <p>Optional web results are selected from search snippets. Full pages are not read or fact-checked. Opening a link takes you to a separate service with its own privacy practices.</p>
      </section>
      <section><h2>Safety checks have limits</h2>
        <p>The current safety check uses English phrase rules. It can miss distress, especially unfamiliar wording or other languages, and can misread harmless statements. A detected phrase switches the conversation to human-support information and stops ordinary chat replies.</p>
        <p>No person is watching messages to provide emergency help. Do not wait for Luna to recognize an emergency.</p>
      </section>
      <section><h2>What has been evaluated</h2>
        <p>Software checks test parts of the app’s behavior. A small comparison of an earlier AI prompt used fictional scenarios and an AI judge. It does not establish that the current version is better, clinically safe, or beneficial to wellbeing.</p>
      </section>
    </PublicInformation>
  );
}
