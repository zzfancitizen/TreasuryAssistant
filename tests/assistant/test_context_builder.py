from app.assistant.context_builder import ContextBuilder
from app.memory import ExecutionState
from app.assistant.planner import AgentStep, RoutePlan
from app.core.skill_registry import SkillDescriptor


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.messages.append(messages)
        return self.response


def build_plan() -> RoutePlan:
    return RoutePlan(
        intent="liquidity_analysis",
        confidence=0.9,
        execution_mode="sequential",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Forecast cashflow"),
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity gap",
                depends_on=["forecast_cashflow"],
            ),
        ],
    )


def test_context_builder_passes_only_dependency_results() -> None:
    plan = build_plan()
    state = ExecutionState.create(user_goal="分析未来两周资金情况", plan=plan)
    state.record_step_result(
        "forecast_cashflow",
        {
            "status": "completed",
            "summary": "未来两周预计缺口 500 万 CNY",
            "data": {"liquidity_gap": 5_000_000, "currency": "CNY"},
        },
    )
    state.record_step_result(
        "unrelated_skill",
        {"status": "completed", "summary": "Unrelated result"},
    )

    context = ContextBuilder().build(
        state=state,
        step=plan.steps[1],
        skill=SkillDescriptor(skill_id="analyze_liquidity_gap", provider_agent_id="treasury_agent", name="Analyze"),
    )

    assert "forecast_cashflow" in context["dependency_results"]
    assert "unrelated_skill" not in context["dependency_results"]
    assert context["user_goal"] == "分析未来两周资金情况"
    assert context["current_step"]["skill_id"] == "analyze_liquidity_gap"


def test_context_builder_respects_token_budget() -> None:
    plan = build_plan()
    state = ExecutionState.create(user_goal="分析资金情况", plan=plan)
    state.record_step_result(
        "forecast_cashflow",
        {
            "status": "completed",
            "summary": "x" * 1000,
            "data": {"large_payload": "y" * 5000},
        },
    )

    context = ContextBuilder(max_context_tokens=80).build(
        state=state,
        step=plan.steps[1],
        skill=SkillDescriptor(skill_id="analyze_liquidity_gap", provider_agent_id="treasury_agent", name="Analyze"),
    )

    assert context["context_budget"]["max_tokens"] == 80
    assert context["context_budget"]["estimated_tokens"] <= 80
    assert context["context_budget"]["truncated"] is True


async def test_context_builder_uses_llm_compression_when_configured() -> None:
    plan = build_plan()
    state = ExecutionState.create(user_goal="分析资金情况", plan=plan)
    state.record_step_result(
        "forecast_cashflow",
        {
            "status": "completed",
            "summary": "x" * 1000,
            "data": {"liquidity_gap": 5_000_000, "currency": "CNY", "large_payload": "y" * 5000},
        },
    )
    llm = FakeLLMClient("缺口500万CNY")

    context = await ContextBuilder(max_context_tokens=260, compression_llm_client=llm).build_async(
        state=state,
        step=plan.steps[1],
        skill=SkillDescriptor(skill_id="analyze_liquidity_gap", provider_agent_id="treasury_agent", name="Analyze"),
    )

    assert llm.messages
    assert context["context_budget"]["compression_strategy"] == "llm"
