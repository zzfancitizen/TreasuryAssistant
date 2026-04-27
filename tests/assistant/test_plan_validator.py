import pytest

from app.assistant.plan_validator import PlanValidationError, PlanValidator
from app.assistant.planner import AgentStep, RoutePlan
from app.core.registry import AgentRegistry
from app.core.skill_registry import SkillRegistry


def build_validator() -> PlanValidator:
    return PlanValidator(skill_registry=SkillRegistry.from_agent_registry(AgentRegistry.default_builtin()))


def test_normalizes_execution_mode_from_dependencies() -> None:
    validator = build_validator()
    plan = RoutePlan(
        intent="liquidity_analysis",
        confidence=0.9,
        execution_mode="parallel",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Forecast cash"),
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity",
                depends_on=["forecast_cashflow"],
            ),
        ],
    )

    normalized = validator.normalize(plan)

    assert normalized.execution_mode == "sequential"


def test_rejects_unknown_skill() -> None:
    validator = build_validator()
    plan = RoutePlan(
        intent="general",
        confidence=0.9,
        execution_mode="single",
        steps=[AgentStep(skill_id="unknown_skill", task="Do unknown work")],
    )

    with pytest.raises(PlanValidationError, match="Unknown skill"):
        validator.validate(plan)


def test_rejects_missing_dependency() -> None:
    validator = build_validator()
    plan = RoutePlan(
        intent="general",
        confidence=0.9,
        execution_mode="sequential",
        steps=[
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity",
                depends_on=["forecast_cashflow"],
            )
        ],
    )

    with pytest.raises(PlanValidationError, match="Missing dependency"):
        validator.validate(plan)


def test_rejects_circular_dependency() -> None:
    validator = build_validator()
    plan = RoutePlan(
        intent="general",
        confidence=0.9,
        execution_mode="sequential",
        steps=[
            AgentStep(
                skill_id="forecast_cashflow",
                task="Forecast cash",
                depends_on=["analyze_liquidity_gap"],
            ),
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity",
                depends_on=["forecast_cashflow"],
            ),
        ],
    )

    with pytest.raises(PlanValidationError, match="Circular dependency"):
        validator.validate(plan)


def test_rejects_duplicate_skill_steps() -> None:
    validator = build_validator()
    plan = RoutePlan(
        intent="general",
        confidence=0.9,
        execution_mode="parallel",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Forecast cash"),
            AgentStep(skill_id="forecast_cashflow", task="Forecast cash again"),
        ],
    )

    with pytest.raises(PlanValidationError, match="Duplicate skill"):
        validator.validate(plan)
