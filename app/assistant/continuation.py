from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.assistant.planner import AgentStep, RoutePlan


class ContinuationDecision(BaseModel):
    action: Literal["continue", "insert_step", "finish"]
    step: AgentStep | None = None
    reason: str = ""


class ContinuationDecider:
    def decide(
        self,
        *,
        plan: RoutePlan,
        completed_step: AgentStep,
        result: dict[str, Any],
        previous_results: dict[str, dict[str, Any]],
        queued_steps: list[AgentStep],
        iteration: int,
    ) -> ContinuationDecision:
        if not plan.can_replan:
            return ContinuationDecision(action="continue", reason="Replanning disabled for this plan.")
        if iteration >= plan.max_iterations:
            return ContinuationDecision(action="continue", reason="Maximum execution iterations reached.")
        if completed_step.skill_id not in {"forecast_cashflow", "get_cash_balance"}:
            return ContinuationDecision(action="continue", reason="No continuation rule for completed skill.")
        if any(step.skill_id in {"analyze_liquidity_gap", "recommend_funding_plan"} for step in queued_steps):
            return ContinuationDecision(action="continue", reason="Treasury analysis is already queued.")
        if any(skill_id in previous_results for skill_id in {"analyze_liquidity_gap", "recommend_funding_plan"}):
            return ContinuationDecision(action="continue", reason="Treasury analysis has already completed.")
        if not has_positive_liquidity_gap(result):
            return ContinuationDecision(action="continue", reason="No liquidity gap trigger found.")

        return ContinuationDecision(
            action="insert_step",
            step=AgentStep(
                skill_id="recommend_funding_plan",
                task="Recommend treasury actions based on observed liquidity gap",
                depends_on=[completed_step.skill_id],
            ),
            reason="Cash skill reported a positive liquidity gap.",
        )


def has_positive_liquidity_gap(result: dict[str, Any]) -> bool:
    data = result.get("data")
    if not isinstance(data, dict):
        return False
    gap = data.get("liquidity_gap")
    if isinstance(gap, int | float):
        return gap > 0
    return False
