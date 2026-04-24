from unittest.mock import patch

from app.assistant.planner import RoutePlanner
from app.core.registry import AgentRegistry


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.messages.append(messages)
        return self.response


async def test_rule_route_keeps_high_confidence_cash_fast_path() -> None:
    planner = RoutePlanner(registry=AgentRegistry.default_local())

    plan = await planner.plan("查询公司银行账户余额")

    assert plan.intent == "cash"
    assert plan.confidence == 1.0
    assert [step.skill_id for step in plan.steps] == ["get_cash_balance"]


async def test_llm_route_handles_no_keyword_user_request() -> None:
    llm = FakeLLMClient(
        """
        {
          "intent": "liquidity_analysis",
          "confidence": 0.86,
          "execution_mode": "sequential",
          "steps": [
            {"skill_id": "forecast_cashflow", "task": "Get cash position", "depends_on": []},
            {"skill_id": "analyze_liquidity_gap", "task": "Analyze liquidity risk", "depends_on": ["forecast_cashflow"]}
          ],
          "needs_clarification": false,
          "clarification_question": null,
          "reason": "The user asks whether liquidity can hold over the next two weeks."
        }
        """
    )
    planner = RoutePlanner(registry=AgentRegistry.default_local(), llm_client=llm)

    plan = await planner.plan("帮我看下这两周能不能撑过去")

    assert plan.intent == "liquidity_analysis"
    assert plan.confidence == 0.86
    assert plan.execution_mode == "sequential"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow", "analyze_liquidity_gap"]
    assert llm.messages


async def test_low_confidence_llm_route_falls_back_to_general_parallel() -> None:
    llm = FakeLLMClient(
        """
        {
          "intent": "unknown",
          "confidence": 0.2,
          "execution_mode": "sequential",
          "steps": [],
          "needs_clarification": true,
          "clarification_question": "What do you want to analyze?",
          "reason": "Ambiguous request."
        }
        """
    )
    planner = RoutePlanner(registry=AgentRegistry.default_local(), llm_client=llm)

    plan = await planner.plan("看一下")

    assert plan.intent == "general"
    assert plan.confidence == 0.45
    assert plan.execution_mode == "parallel"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow", "analyze_liquidity_gap"]


async def test_invalid_llm_json_falls_back_to_general_parallel() -> None:
    planner = RoutePlanner(
        registry=AgentRegistry.default_local(),
        llm_client=FakeLLMClient("not json"),
    )

    with patch("app.assistant.planner.logger.error") as log_error:
        plan = await planner.plan("看一下")

    assert plan.intent == "general"
    assert plan.execution_mode == "parallel"
    log_error.assert_called_once_with("route_planner.llm.invalid_json")


async def test_llm_parallel_claim_is_overridden_when_steps_have_dependencies() -> None:
    llm = FakeLLMClient(
        """
        {
          "intent": "liquidity_analysis",
          "confidence": 0.91,
          "execution_mode": "parallel",
          "steps": [
            {"skill_id": "forecast_cashflow", "task": "Get cash position", "depends_on": []},
            {"skill_id": "analyze_liquidity_gap", "task": "Recommend treasury actions", "depends_on": ["forecast_cashflow"]}
          ],
          "needs_clarification": false,
          "clarification_question": null,
          "reason": "LLM incorrectly claimed parallel despite a dependency."
        }
        """
    )
    planner = RoutePlanner(registry=AgentRegistry.default_local(), llm_client=llm)

    plan = await planner.plan("先看现金情况，再给我资金建议")

    assert plan.execution_mode == "sequential"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow", "analyze_liquidity_gap"]
