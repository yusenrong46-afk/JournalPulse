from __future__ import annotations

from typing import Protocol

from .domain import AffectiveState, PolicyDecision, TargetState


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
