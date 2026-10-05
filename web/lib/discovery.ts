import { apiRequest } from "./api";
import type { GoalOption } from "./feelings";

const GENERAL_TOPICS: Record<GoalOption["id"], string> = {
  settle: "short grounding exercises for everyday stress",
  move: "gentle movement for an everyday energy break",
  understand: "reflective writing prompts to understand everyday feelings",
  connect: "small ways to build everyday social connection",
  act: "breaking an everyday task into one manageable step",
};

/** Only a fixed goal travels between pages; private writing and chat IDs stay out. */
export function discoveryTopicForGoal(goal: string | null | undefined): string {
  return goal && Object.hasOwn(GENERAL_TOPICS, goal) ? GENERAL_TOPICS[goal as GoalOption["id"]] : "";
}

export function discoveryHref(goal?: string | null): string {
  return discoveryTopicForGoal(goal) ? `/discover?goal=${goal}` : "/discover";
}

export const MAX_EXCLUDED_SOURCES = 30;
// A refinement can make two 25s model calls and one 20s search. The stream
// deadline adds at most a final 5s read per call; allow room for network overhead.
export const DISCOVERY_TIMEOUT_MS = 110_000;

export type DiscoveryRequest = {
  original_query: string;
  previous_query?: string | null;
  feedback?: string | null;
  excluded_urls: string[];
  llm_consent: boolean;
  locale?: string;
};

export type DiscoveryCandidate = {
  title: string;
  url: string;
  description: string;
  why_selected: string;
  evidence_kind: "search_snippet";
};

export type DiscoveryResponse = {
  original_query: string;
  updated_query: string;
  candidates: DiscoveryCandidate[];
  provenance: {
    search_provider: "brave";
    prompt_version: string;
    retrieved_at: string;
    candidate_count: number;
    search_calls: 1;
    page_fetches: 0;
    model_runs: {
      model: string;
      provider: string;
      latency_ms: number;
      prompt_tokens?: number | null;
      completion_tokens?: number | null;
      prompt_version?: string | null;
      schema_valid: boolean;
    }[];
  };
  limitations: string[];
};

/** Keep exclusions explicit and bounded; server validation is authoritative. */
export function excludedSources(seen: string[], manuallyExcluded: string): string[] {
  const manual = manuallyExcluded.split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
  const urls = [...new Set([...seen, ...manual])];
  if (urls.length > MAX_EXCLUDED_SOURCES) {
    throw new Error("You can skip up to 30 sources. Start a new search to reset the list.");
  }
  for (const value of urls) {
    let url: URL;
    try {
      url = new URL(value);
    } catch {
      throw new Error("Each skipped source needs a full public HTTPS link, one per line.");
    }
    if (url.protocol !== "https:" || url.username || url.password) {
      throw new Error("Each skipped source needs a public HTTPS link without sign-in details.");
    }
  }
  return urls;
}

/** Each refinement carries the original goal, previous query, and all seen URLs. */
export function refinementRequest(input: {
  previous: DiscoveryResponse;
  feedback: string;
  seen: string[];
  manuallyExcluded: string;
  consent: boolean;
  locale?: string;
}): DiscoveryRequest {
  const feedback = input.feedback.trim();
  if (!feedback) throw new Error("Tell Luna what you would like to change.");
  if (feedback.length > 600) throw new Error("Keep your feedback within 600 characters.");
  return {
    original_query: input.previous.original_query,
    previous_query: input.previous.updated_query,
    feedback,
    excluded_urls: excludedSources(input.seen, input.manuallyExcluded),
    llm_consent: input.consent,
    ...(input.locale ? { locale: input.locale } : {}),
  };
}

export function searchDiscovery(payload: DiscoveryRequest, signal?: AbortSignal): Promise<DiscoveryResponse> {
  return apiRequest<DiscoveryResponse>("/v1/discovery/search", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: DISCOVERY_TIMEOUT_MS,
    signal,
  });
}
