from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from app.assistant.plan_validator import PlanValidationError, PlanValidator
from app.assistant.prompts import build_router_system_prompt
from app.core.registry import AgentRegistry
from app.core.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


class SkillStep(BaseModel):
    skill_id: str
    task: str
    depends_on: list[str] = Field(default_factory=list)


AgentStep = SkillStep


class RoutePlan(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    execution_mode: str
    steps: list[SkillStep]
    can_replan: bool = True
    replan_triggers: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=4, ge=1)
    needs_clarification: bool = False
    clarification_question: str | None = None
    reason: str = ""


class RoutePlanner:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        llm_client: LLMClient | None = None,
        min_confidence: float = 0.55,
    ) -> None:
        self.registry = registry
        self.skill_registry = SkillRegistry.from_agent_registry(registry)
        self.plan_validator = PlanValidator(skill_registry=self.skill_registry)
        self.llm_client = llm_client
        self.min_confidence = min_confidence

    async def plan(self, message: str) -> RoutePlan:
        logger.info("route_planner.plan.started", extra={"message_length": len(message)})
        rule_plan = plan_by_rules(message)
        if rule_plan is not None:
            logger.info(
                "route_planner.plan.rule_matched",
                extra={"intent": rule_plan.intent, "execution_mode": rule_plan.execution_mode, "step_count": len(rule_plan.steps)},
            )
            return rule_plan

        if self.llm_client is None:
            logger.info("route_planner.plan.no_llm_fallback")
            return general_parallel_plan(reason="No keyword matched and no LLM planner is configured.")

        llm_plan = await self._plan_with_llm(message)
        if llm_plan is None or llm_plan.confidence < self.min_confidence or not llm_plan.steps:
            logger.warning(
                "route_planner.plan.llm_fallback",
                extra={"has_plan": llm_plan is not None, "confidence": llm_plan.confidence if llm_plan else None},
            )
            return general_parallel_plan(reason="LLM route was missing or below confidence threshold.")
        normalized = normalize_plan(llm_plan, skill_registry=self.skill_registry)
        logger.info(
            "route_planner.plan.completed",
            extra={"intent": normalized.intent, "execution_mode": normalized.execution_mode, "step_count": len(normalized.steps)},
        )
        return normalized

    async def _plan_with_llm(self, message: str) -> RoutePlan | None:
        response = await self.llm_client.complete(
            [
                {"role": "system", "content": build_router_system_prompt(self.registry)},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        try:
            return RoutePlan.model_validate_json(extract_json_object(response))
        except (ValidationError, ValueError, json.JSONDecodeError):
            logger.error("route_planner.llm.invalid_json")
            return None


def plan_by_rules(message: str) -> RoutePlan | None:
    normalized = message.lower()
    cash_keywords = ("余额", "流水", "账户", "现金流", "balance", "cash")
    treasury_keywords = ("融资", "策略", "限额", "风控", "funding", "policy")
    liquidity_keywords = ("缺口", "调拨", "流动性", "liquidity", "transfer")

    if any(keyword in normalized for keyword in liquidity_keywords):
        return RoutePlan(
            intent="liquidity_analysis",
            confidence=1.0,
            execution_mode="sequential",
            steps=[
                AgentStep(skill_id="forecast_cashflow", task="Collect cash balances and cashflow context"),
                AgentStep(
                    skill_id="analyze_liquidity_gap",
                    task="Analyze liquidity gap and recommend treasury actions",
                    depends_on=["forecast_cashflow"],
                ),
            ],
            reason="Matched liquidity keywords.",
        )
    if any(keyword in normalized for keyword in cash_keywords):
        return RoutePlan(
            intent="cash",
            confidence=1.0,
            execution_mode="single",
            steps=[AgentStep(skill_id="get_cash_balance", task="Handle cash-related request")],
            reason="Matched cash keywords.",
        )
    if any(keyword in normalized for keyword in treasury_keywords):
        return RoutePlan(
            intent="treasury",
            confidence=1.0,
            execution_mode="single",
            steps=[AgentStep(skill_id="recommend_funding_plan", task="Handle treasury-related request")],
            reason="Matched treasury keywords.",
        )
    return None


def general_parallel_plan(reason: str) -> RoutePlan:
    return RoutePlan(
        intent="general",
        confidence=0.45,
        execution_mode="parallel",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Provide cash context for the user request"),
            AgentStep(skill_id="analyze_liquidity_gap", task="Provide treasury context for the user request"),
        ],
        reason=reason,
    )


def normalize_plan(plan: RoutePlan, *, skill_registry: SkillRegistry) -> RoutePlan:
    validator = PlanValidator(skill_registry=skill_registry)
    steps = validator.supported_steps(plan)
    if not steps:
        return general_parallel_plan(reason="LLM route had no supported skills.")

    supported_intents = {"cash", "treasury", "liquidity_analysis", "general"}
    intent = plan.intent if plan.intent in supported_intents else "general"
    candidate = plan.model_copy(update={"intent": intent, "steps": steps})
    try:
        return validator.normalize(candidate)
    except PlanValidationError as exc:
        return general_parallel_plan(reason=f"LLM route failed validation: {exc}")


def extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return text[start : end + 1]
