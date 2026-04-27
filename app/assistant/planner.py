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
            normalized_rule_plan = normalize_plan(rule_plan, skill_registry=self.skill_registry)
            logger.info(
                "route_planner.plan.rule_matched",
                extra={
                    "intent": normalized_rule_plan.intent,
                    "execution_mode": normalized_rule_plan.execution_mode,
                    "step_count": len(normalized_rule_plan.steps),
                },
            )
            return normalized_rule_plan

        if self.llm_client is None:
            logger.info("route_planner.plan.no_llm_fallback")
            return fallback_plan_for_supported_skills(
                self.skill_registry,
                reason="No keyword matched and no LLM planner is configured.",
            )

        try:
            llm_plan = await self._plan_with_llm(message)
        except Exception:
            logger.exception("route_planner.plan.llm_error")
            return fallback_plan_for_supported_skills(
                self.skill_registry,
                reason="LLM route failed; using supported discovered skills.",
            )
        if llm_plan is None or llm_plan.confidence < self.min_confidence or not llm_plan.steps:
            logger.warning(
                "route_planner.plan.llm_fallback",
                extra={"has_plan": llm_plan is not None, "confidence": llm_plan.confidence if llm_plan else None},
            )
            return fallback_plan_for_supported_skills(
                self.skill_registry,
                reason="LLM route was missing or below confidence threshold.",
            )
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
    dynamic_keywords = ("动态编排", "动态进行agent编排", "dynamic orchestration", "dynamic")
    parallel_keywords = ("并发编排", "并发执行", "parallel orchestration", "parallel")
    cash_keywords = ("余额", "流水", "账户", "现金", "现金流", "balance", "cash")
    treasury_keywords = ("融资", "策略", "限额", "风控", "funding", "policy")
    liquidity_keywords = ("缺口", "调拨", "流动性", "liquidity", "transfer")
    read_keywords = ("读取", "读", "查看状态", "read")
    change_keywords = ("修改", "变更", "调整", "change", "update")

    if any(keyword in normalized for keyword in dynamic_keywords):
        return RoutePlan(
            intent="dynamic_liquidity_check",
            confidence=1.0,
            execution_mode="single",
            steps=[AgentStep(skill_id="forecast_cashflow", task="Assess cash safety for dynamic orchestration")],
            can_replan=True,
            replan_triggers=["liquidity_gap_detected"],
            reason="Matched dynamic orchestration case keywords.",
        )

    if any(keyword in normalized for keyword in parallel_keywords):
        return RoutePlan(
            intent="general",
            confidence=1.0,
            execution_mode="parallel",
            steps=[
                AgentStep(skill_id="forecast_cashflow", task="Provide cash context for parallel orchestration"),
                AgentStep(skill_id="analyze_liquidity_gap", task="Provide treasury context for parallel orchestration"),
            ],
            reason="Matched parallel orchestration case keywords.",
        )

    if any(keyword in normalized for keyword in read_keywords):
        if any(keyword in normalized for keyword in cash_keywords):
            return RoutePlan(
                intent="cash_read",
                confidence=1.0,
                execution_mode="single",
                steps=[AgentStep(skill_id="read_cash_state", task="Read mock cash runtime state")],
                reason="Matched cash read keywords.",
            )
        if any(keyword in normalized for keyword in treasury_keywords):
            return RoutePlan(
                intent="treasury_read",
                confidence=1.0,
                execution_mode="single",
                steps=[AgentStep(skill_id="read_treasury_state", task="Read mock treasury runtime state")],
                reason="Matched treasury read keywords.",
            )
    if any(keyword in normalized for keyword in change_keywords):
        if any(keyword in normalized for keyword in cash_keywords):
            return RoutePlan(
                intent="cash_change",
                confidence=1.0,
                execution_mode="single",
                steps=[AgentStep(skill_id="change_cash_state", task="Prepare mock cash runtime state change")],
                reason="Matched cash change keywords.",
            )
        if any(keyword in normalized for keyword in treasury_keywords):
            return RoutePlan(
                intent="treasury_change",
                confidence=1.0,
                execution_mode="single",
                steps=[AgentStep(skill_id="change_treasury_state", task="Prepare mock treasury runtime state change")],
                reason="Matched treasury change keywords.",
            )

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
            steps=[AgentStep(skill_id="analyze_liquidity_gap", task="Handle treasury-related request")],
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


def unavailable_plan(reason: str) -> RoutePlan:
    return RoutePlan(
        intent="unavailable",
        confidence=0.0,
        execution_mode="single",
        steps=[],
        can_replan=False,
        needs_clarification=True,
        clarification_question="No A2A subagent skills are currently discoverable.",
        reason=reason,
    )


def fallback_plan_for_supported_skills(skill_registry: SkillRegistry, *, reason: str) -> RoutePlan:
    known_skills = {skill.skill_id for skill in skill_registry.list()}
    if {"forecast_cashflow", "analyze_liquidity_gap"}.issubset(known_skills):
        return general_parallel_plan(reason=reason)
    if "get_cash_balance" in known_skills:
        return RoutePlan(
            intent="cash",
            confidence=0.35,
            execution_mode="single",
            steps=[AgentStep(skill_id="get_cash_balance", task="Provide available cash context for the user request")],
            reason=reason,
        )
    if "analyze_liquidity_gap" in known_skills:
        return RoutePlan(
            intent="treasury",
            confidence=0.35,
            execution_mode="single",
            steps=[AgentStep(skill_id="analyze_liquidity_gap", task="Provide treasury context for the user request")],
            reason=reason,
        )
    return unavailable_plan(reason=reason)


def normalize_plan(plan: RoutePlan, *, skill_registry: SkillRegistry) -> RoutePlan:
    validator = PlanValidator(skill_registry=skill_registry)
    steps = validator.supported_steps(plan)
    if not steps:
        return fallback_plan_for_supported_skills(skill_registry, reason="Route had no supported discovered skills.")

    supported_intents = {
        "cash",
        "treasury",
        "liquidity_analysis",
        "general",
        "unavailable",
        "cash_read",
        "cash_change",
        "treasury_read",
        "treasury_change",
        "dynamic_liquidity_check",
    }
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
