export type AffectiveState = {
  valence: number;
  arousal: number;
  agency: number;
  emotion_tags: string[];
  /** A model's own estimate. Null for states derived from the person's taps. */
  confidence?: number | null;
  uncertainty?: string | null;
  derivation?: string | null;
};

export type SelfReportInput = {
  feelings: string[];
  mood_score?: number | null;
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
  self_report_input?: SelfReportInput | null;
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

export type ConversationMessage = {
  id: string;
  conversation_id: string;
  client_message_id?: string | null;
  role: "user" | "assistant";
  content?: string | null;
  created_at: string;
  safety_mode: "normal" | "support";
  model_run?: PreparedAnalysis["model_run"] | null;
  request_inputs?: { goal?: string | null; confirmed_feelings?: string[] | null; activity_constraints?: ActivityConstraints | null } | null;
};

export type ActionCard = {
  resource_intent: string;
  card_reason: string;
  decision_preview: ReflectionRecord["decision"];
  actions: Resource[];
  offered_message_id?: string | null;
  goal?: "settle" | "move" | "understand" | "connect" | "act" | null;
};

export type ActivityConstraints = {
  time_minutes: number | null;
  no_audio: boolean; no_video: boolean; seated: boolean; avoid_breath_focus: boolean;
};

export type Conversation = {
  id: string;
  incarnation_id?: string | null;
  activity_card?: ActionCard | null;
  activity_constraints?: ActivityConstraints;
  activity_goal?: ActionCard["goal"];
  user_id: string;
  created_at: string;
  updated_at: string;
  status: "open" | "closed";
  llm_consent: boolean;
  /** One explicitly selected journal entry; the original text remains in the journal. */
  source_entry_id?: string | null;
  source_entry_created_at?: string | null;
  retain_text: boolean;
  safety_mode: "normal" | "support";
  summary?: string | null;
  card?: ActionCard | null;
  safety?: PreparedAnalysis["safety"] | null;
  reflection_id?: string | null;
  locale: string;
  prompt_version: string;
  mode?: "ai" | "guided";
  /** Luna's suggestion only. */
  feelings?: string[];
  ready_for_action?: boolean;
  /** The person's choice for this chat; absent legacy values mean automatic. */
  interaction_preference?: "auto" | "listen" | "act";
  /** What the person reported. Null until they report it. */
  reported_mood?: number | null;
  confirmed_feelings?: string[] | null;
  revision?: number;
};

export type JournalEntry = {
  id: string;
  user_id: string;
  created_at: string;
  text: string;
};

export type ConversationDetail = {
  conversation: Conversation;
  messages: ConversationMessage[];
  accepted_reflection?: ReflectionRecord | null;
};

export type ConversationTurn = {
  conversation: Conversation;
  user_message: ConversationMessage;
  assistant_message: ConversationMessage;
};
