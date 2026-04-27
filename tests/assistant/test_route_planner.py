from unittest.mock import patch

from app.assistant.planner import RoutePlanner
from app.core.registry import AgentEndpoint
from app.core.registry import AgentRegistry, build_endpoint_from_agent_card
from app.cash.server import agent_card as cash_agent_card
from app.treasury.server import agent_card as treasury_agent_card


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.messages.append(messages)
        return self.response


class FailingLLMClient:
    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        raise RuntimeError("LLM unavailable")


async def test_rule_route_keeps_high_confidence_cash_fast_path() -> None:
    planner = RoutePlanner(registry=AgentRegistry.default_builtin())

    plan = await planner.plan("查询公司银行账户余额")

    assert plan.intent == "cash"
    assert plan.confidence == 1.0
    assert [step.skill_id for step in plan.steps] == ["get_cash_balance"]


async def test_rule_route_supports_parallel_execution_case() -> None:
    planner = RoutePlanner(registry=AgentRegistry.default_builtin())

    plan = await planner.plan("并发编排：请综合分析当前财资状况")

    assert plan.intent == "general"
    assert plan.confidence == 1.0
    assert plan.execution_mode == "parallel"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow", "analyze_liquidity_gap"]


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
    planner = RoutePlanner(registry=AgentRegistry.default_builtin(), llm_client=llm)

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
    planner = RoutePlanner(registry=AgentRegistry.default_builtin(), llm_client=llm)

    plan = await planner.plan("看一下")

    assert plan.intent == "general"
    assert plan.confidence == 0.45
    assert plan.execution_mode == "parallel"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow", "analyze_liquidity_gap"]


async def test_llm_route_failure_falls_back_to_general_parallel() -> None:
    planner = RoutePlanner(registry=AgentRegistry.default_builtin(), llm_client=FailingLLMClient())

    plan = await planner.plan("看一下")

    assert plan.intent == "general"
    assert plan.execution_mode == "parallel"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow", "analyze_liquidity_gap"]


async def test_invalid_llm_json_falls_back_to_general_parallel() -> None:
    planner = RoutePlanner(
        registry=AgentRegistry.default_builtin(),
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
    planner = RoutePlanner(registry=AgentRegistry.default_builtin(), llm_client=llm)

    plan = await planner.plan("帮我判断这两周能不能撑过去")

    assert plan.execution_mode == "sequential"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow", "analyze_liquidity_gap"]


async def test_rule_route_uses_skill_published_by_discovered_treasury_card() -> None:
    registry = AgentRegistry(
        [
            build_endpoint_from_agent_card(
                agent_id="cash_agent",
                seed_url="http://localhost:8001",
                card=cash_agent_card.model_dump(),
            ),
            build_endpoint_from_agent_card(
                agent_id="treasury_agent",
                seed_url="http://localhost:8002",
                card=treasury_agent_card.model_dump(),
            ),
        ]
    )
    planner = RoutePlanner(registry=registry)

    plan = await planner.plan("有什么融资策略建议")
    planner.plan_validator.validate(plan)

    published_skills = {skill.skill_id for endpoint in registry.list() for skill in endpoint.skills}
    assert plan.steps[0].skill_id in published_skills


async def test_rule_route_maps_runtime_read_and_change_requests() -> None:
    planner = RoutePlanner(registry=AgentRegistry.default_builtin())

    read_cash = await planner.plan("读取 cash runtime state")
    change_cash = await planner.plan("修改现金池状态")
    read_treasury = await planner.plan("读取 treasury policy state")
    change_treasury = await planner.plan("修改融资计划")

    assert read_cash.steps[0].skill_id == "read_cash_state"
    assert change_cash.steps[0].skill_id == "change_cash_state"
    assert read_treasury.steps[0].skill_id == "read_treasury_state"
    assert change_treasury.steps[0].skill_id == "change_treasury_state"


async def test_rule_route_supports_dynamic_runtime_orchestration_case() -> None:
    planner = RoutePlanner(registry=AgentRegistry.default_builtin())

    plan = await planner.plan("动态编排：先读取现金流，如果发现资金缺口再追加融资计划")

    assert plan.intent == "dynamic_liquidity_check"
    assert plan.execution_mode == "single"
    assert [step.skill_id for step in plan.steps] == ["forecast_cashflow"]
    assert plan.can_replan is True
    assert plan.replan_triggers == ["liquidity_gap_detected"]


async def test_route_planner_does_not_invent_skills_when_discovery_has_no_skill_metadata() -> None:
    registry = AgentRegistry(
        [
            AgentEndpoint(agent_id="cash_agent", name="CashAgent", url="http://localhost:8001", capabilities=()),
            AgentEndpoint(agent_id="treasury_agent", name="TreasuryAgent", url="http://localhost:8002", capabilities=()),
        ]
    )
    planner = RoutePlanner(registry=registry)

    plan = await planner.plan("查询公司银行账户余额")

    assert plan.intent == "unavailable"
    assert plan.steps == []
    assert plan.needs_clarification is True
