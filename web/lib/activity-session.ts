import { apiRequest } from "./api";
import type { Conversation } from "./types";

export type ActivityStatus = "offered" | "active" | "paused" | "awaiting_report" | "completed" | "stopped" | "declined";
export type ActivityCommand = "start" | "pause" | "resume" | "finish_early" | "expire" | "stop" | "decline";
export type ActivityResource = {
  id: string; title: string; url: string | null; provider: string; resource_type: string;
  format: "timer" | "external" | "manual";
  kind: "meditation" | "movement" | "reflection" | "connection" | "focus" | "video" | "reading" | "other";
  duration_seconds: number | null; instructions: string[];
  provenance: "builtin" | "catalog" | "search_snippet";
};
export type ActivityReport = {
  participation: "completed" | "partial" | "not_tried" | "stopped";
  fit?: "good" | "mixed" | "poor" | "unsure" | null;
  state_change?: "toward_target" | "same" | "away_from_target" | "unsure" | null;
  goal_progress?: "closer" | "same" | "further" | "unsure" | null;
  before_rating?: number | null; after_rating?: number | null;
  helpfulness?: number | null; effort?: number | null; note?: string | null;
};
export type ActivitySession = {
  id: string; user_id: string; conversation_id: string; source_entry_id: string | null;
  offered_message_id?: string | null;
  revision: number; status: ActivityStatus; resource: ActivityResource;
  goal?: "settle" | "move" | "understand" | "connect" | "act" | null;
  recommendation_reason: string | null;
  selection: { selection_source: "llm" | "guided" | "user" | "search"; recommended_resource_id: string;
    selected_resource_id: string; eligible_for_ope: false; propensity: null };
  duration_seconds: number; remaining_seconds: number; expires_at: string | null;
  created_at: string; updated_at: string; started_at: string | null;
  check_in_issued: boolean; report: ActivityReport | null; reported_at: string | null;
  follow_up_status: "none" | "pending" | "generating" | "ready" | "failed";
  follow_up_reply: string | null; follow_up_message_id?: string | null;
  follow_up_attempts: number; final_follow_up: boolean;
  server_now: string | null; conversation_revision: number | null;
};
export type ActivityReceipt = {
  client_request_id: string; expected_revision: number; expected_conversation_revision: number;
};

export function readCurrentActivity(conversationId: string, signal?: AbortSignal): Promise<ActivitySession | null> {
  return apiRequest(`/v1/conversations/${conversationId}/activity-sessions`, { signal });
}

export function createActivity(conversationId: string, payload: {
  client_request_id: string; expected_conversation_revision: number; resource_id: string; resource_token?: string;
}, signal?: AbortSignal): Promise<ActivitySession> {
  return apiRequest(`/v1/conversations/${conversationId}/activity-sessions`, {
    method: "POST", body: JSON.stringify(payload), retry: true, signal,
  });
}

export function commandActivity(sessionId: string, payload: ActivityReceipt & { command: ActivityCommand }, signal?: AbortSignal): Promise<ActivitySession> {
  return apiRequest(`/v1/activity-sessions/${sessionId}/commands`, {
    method: "POST", body: JSON.stringify(payload), retry: true, signal,
  });
}

export function reportActivity(sessionId: string, payload: ActivityReceipt & ActivityReport, signal?: AbortSignal): Promise<ActivitySession> {
  return apiRequest(`/v1/activity-sessions/${sessionId}/report`, {
    method: "POST", body: JSON.stringify(payload), retry: true, signal,
  });
}

/** Saving a report is separate from a paid follow-up; a timeout cannot erase the report. */
export function followUpActivity(sessionId: string, payload: ActivityReceipt, signal?: AbortSignal): Promise<ActivitySession> {
  return apiRequest(`/v1/activity-sessions/${sessionId}/follow-up`, {
    method: "POST", body: JSON.stringify(payload), timeoutMs: 60_000, retry: false, signal,
  });
}

export function activityReceipt(session: ActivitySession, conversation: Conversation, requestId: string): ActivityReceipt {
  return {
    client_request_id: requestId,
    expected_revision: session.revision,
    expected_conversation_revision: Math.max(session.conversation_revision ?? 0, conversation.revision ?? 0),
  };
}

/** Responses from another chat or an older tab must not rewind a canonical session. */
export function canApplyActivity(current: ActivitySession | null, next: ActivitySession, conversationId: string): boolean {
  if (next.conversation_id !== conversationId) return false;
  if (!current) return true;
  if (current.id === next.id) return next.revision >= current.revision;
  return Date.parse(next.created_at) >= Date.parse(current.created_at);
}

export type ActivityClock = { serverMillis: number; monotonicMillis: number };

export function activityClock(session: ActivitySession, monotonicMillis: number): ActivityClock {
  return { serverMillis: Date.parse(session.server_now ?? session.updated_at), monotonicMillis };
}

/** The interval only repaints. A server deadline and monotonic elapsed time decide expiry. */
export function activitySecondsLeft(session: ActivitySession, clock: ActivityClock, monotonicMillis: number): number {
  if (session.status !== "active" || !session.expires_at) return session.remaining_seconds;
  const estimatedServerNow = clock.serverMillis + Math.max(0, monotonicMillis - clock.monotonicMillis);
  return Math.max(0, Math.ceil((Date.parse(session.expires_at) - estimatedServerNow) / 1000));
}
