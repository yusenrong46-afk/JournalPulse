export type AffectiveState = {
  valence: number;
  arousal: number;
  agency: number;
  emotion_tags: string[];
  confidence: number;
  uncertainty?: string | null;
};

export type TargetState = {
  valence?: number | null;
  arousal?: number | null;
  agency?: number | null;
  goal: string;
};

export type PreparedAnalysis = {
  state: AffectiveState;
  reflection: {
    summary: string;
    interpretation: string;
    reflection_question: string;
  };
  safety: {
    mode: "normal" | "support";
    reasons: string[];
    locale: string;
    exploration_allowed: boolean;
    support_message?: string | null;
    resource_ids: string[];
  };
  model_run: {
    model: string;
    provider: string;
    latency_ms: number;
    schema_valid: boolean;
    used_fallback: boolean;
    fallback_reason?: string | null;
  };
  resource_intent: string;
};

export type ReflectionRecord = {
  id: string;
  created_at: string;
  text?: string | null;
  text_retained: boolean;
  context: Record<string, string>;
  state: AffectiveState;
  target: TargetState;
  reflection: PreparedAnalysis["reflection"];
  safety: PreparedAnalysis["safety"];
  decision: {
    decision_id: string;
    action_id: string;
    propensity: number;
    policy_name: string;
    policy_version: string;
    safe_action_ids: string[];
    explanation: string;
    recommended_action_id?: string | null;
    selection_source: "policy" | "policy_accepted" | "user_override";
    eligible_for_ope: boolean;
  };
  model_run?: PreparedAnalysis["model_run"] | null;
};

export type Resource = {
  id: string;
  title: string;
  url: string;
  summary: string;
  provider: string;
  resource_type: string;
  coping_style: string;
  duration_minutes?: number | null;
  source_tier?: string;
  tone_tags?: string[];
};

export type ActionPreview = {
  decision: ReflectionRecord["decision"];
  actions: Resource[];
};

export type OutcomeRecord = {
  id: string;
  decision_id: string;
  created_at: string;
  completed: boolean;
  post_state?: AffectiveState | null;
  helpfulness?: number | null;
  effort?: number | null;
  elapsed_minutes?: number | null;
  note?: string | null;
};

export type Insights = {
  reflection_count: number;
  completed_outcomes: number;
  action_counts: Record<string, number>;
  average_helpfulness_by_action: Record<string, number>;
  average_state_change?: Record<string, number> | null;
  completion_rate: number;
  pending_decision_ids: string[];
  state_trajectory: Array<{
    reflection_id: string;
    created_at: string;
    valence: number;
    arousal: number;
    agency: number;
  }>;
  note: string;
};

export type SystemStatus = {
  analysis_mode: "ai_configured" | "local_fallback" | "local_only";
  persistence_mode: "account" | "server_sqlite";
  message: string;
};
