from __future__ import annotations

from typing import Protocol

from .domain import AffectiveState, PolicyDecision, SelectionSource, TargetState


class ReflectionPolicy(Protocol):
    def decide(
        self,
        *,
        state: AffectiveState,
        target: TargetState,
        actions: list[dict],
        context: dict[str, str],
    ) -> PolicyDecision: ...


class FixedBaselinePolicy:
    """Transparent non-learning baseline; adaptive policies are user-owned work."""

    name = "fixed-baseline"
    version = "1.0.0"

    def decide(
        self,
        *,
        state: AffectiveState,
        target: TargetState,
        actions: list[dict],
        context: dict[str, str],
    ) -> PolicyDecision:
        if not actions:
            return PolicyDecision(
                action_id="pause",
                propensity=1.0,
                policy_name=self.name,
                policy_version=self.version,
                safe_action_ids=["pause"],
                context_snapshot={
                    "state": state.model_dump(),
                    "target": target.model_dump(),
                    "context": context,
                },
                explanation="No approved resource matched, so the baseline selected a short pause.",
            )
        action = actions[0]
        return PolicyDecision(
            action_id=action["id"],
            propensity=1.0,
            policy_name=self.name,
            policy_version=self.version,
            safe_action_ids=[item["id"] for item in actions],
            context_snapshot={"state": state.model_dump(), "target": target.model_dump(), "context": context},
            explanation=(
                "This is a transparent baseline choice from the approved catalog. "
                "It does not yet claim to be personalized."
            ),
        )


def apply_user_choice(decision: PolicyDecision, chosen_action_id: str | None) -> PolicyDecision:
    """Record an explicit accept or a safe override. The caller supplies the server decision."""
    if not chosen_action_id:
        return decision
    if chosen_action_id not in decision.safe_action_ids:
        raise ValueError("Chosen action is not in the safe set")
    recommended = decision.action_id
    if chosen_action_id == recommended:
        return decision.model_copy(
            update={
                "recommended_action_id": recommended,
                "selection_source": SelectionSource.POLICY_ACCEPTED,
            }
        )
    return decision.model_copy(
        update={
            "action_id": chosen_action_id,
            "recommended_action_id": recommended,
            "propensity": 1.0,
            "policy_name": "user-choice",
            "policy_version": "1.0.0",
            "selection_source": SelectionSource.USER_OVERRIDE,
            "eligible_for_ope": False,
            "explanation": (
                "You chose a safe alternative. This decision is recorded as a user "
                "override and excluded from off-policy evaluation."
            ),
        }
    )
