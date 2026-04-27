from app.assistant.plan_executor import PlanExecutor
from app.assistant.planner import AgentStep, RoutePlan
from app.core.registry import AgentEndpoint, AgentRegistry


class RecordingA2AClient:
    def __init__(self, results_by_agent: dict[str, dict] | None = None) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.results_by_agent = results_by_agent or {}

    async def invoke(self, endpoint: AgentEndpoint, message: str, context: dict | None = None) -> dict:
        self.calls.append((endpoint.agent_id, message, context))
        if endpoint.agent_id in self.results_by_agent:
            return self.results_by_agent[endpoint.agent_id]
        return {
            "agent": endpoint.name,
            "status": "completed",
            "summary": f"{endpoint.name} completed {message}",
        }


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.messages.append(messages)
        return self.response


def build_registry_with_risk_agent() -> AgentRegistry:
    endpoints = AgentRegistry.default_builtin().list()
    endpoints.append(
        AgentEndpoint(
            agent_id="risk_agent",
            name="RiskAgent",
            url="http://localhost:8003",
            capabilities=("check_counterparty_risk",),
        )
    )
    return AgentRegistry(endpoints)


async def test_executes_single_step_for_any_registered_agent() -> None:
    client = RecordingA2AClient()
    executor = PlanExecutor(registry=build_registry_with_risk_agent(), a2a_client=client)
    plan = RoutePlan(
        intent="risk",
        confidence=0.8,
        execution_mode="single",
        steps=[AgentStep(skill_id="check_counterparty_risk", task="Check counterparty risk")],
    )

    results = await executor.execute(plan, "检查交易对手风险")

    assert [call[0] for call in client.calls] == ["risk_agent"]
    assert client.calls[0][1] == "Check counterparty risk"
    assert results == [{"agent": "RiskAgent", "status": "completed", "summary": "RiskAgent completed Check counterparty risk"}]


async def test_sequential_plan_passes_previous_results_generically() -> None:
    client = RecordingA2AClient()
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client)
    plan = RoutePlan(
        intent="liquidity_analysis",
        confidence=0.9,
        execution_mode="sequential",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Collect cash data"),
            AgentStep(skill_id="analyze_liquidity_gap", task="Analyze treasury action", depends_on=["forecast_cashflow"]),
        ],
    )

    await executor.execute(plan, "分析资金情况")

    assert [call[0] for call in client.calls] == ["cash_agent", "treasury_agent"]
    assert client.calls[1][2] is not None
    assert "dependency_results" in client.calls[1][2]
    assert "forecast_cashflow" in client.calls[1][2]["dependency_results"]


async def test_parallel_plan_invokes_registered_agents_without_agent_specific_code() -> None:
    client = RecordingA2AClient()
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client)
    plan = RoutePlan(
        intent="general",
        confidence=0.7,
        execution_mode="parallel",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Provide cash context"),
            AgentStep(skill_id="analyze_liquidity_gap", task="Provide treasury context"),
        ],
    )

    results = await executor.execute(plan, "综合分析")

    assert sorted(call[0] for call in client.calls) == ["cash_agent", "treasury_agent"]
    assert len(results) == 2


async def test_single_cash_plan_replans_to_treasury_when_liquidity_gap_is_observed() -> None:
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "agent": "CashAgent",
                "status": "completed",
                "data": {"liquidity_gap": 5_000_000, "currency": "CNY"},
                "summary": "Projected liquidity gap detected.",
            }
        }
    )
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client)
    plan = RoutePlan(
        intent="cash",
        confidence=0.8,
        execution_mode="single",
        steps=[AgentStep(skill_id="forecast_cashflow", task="Assess cash safety")],
        can_replan=True,
        replan_triggers=["liquidity_gap_detected"],
    )

    results = await executor.execute(plan, "帮我看下这两周能不能撑过去")

    assert [call[0] for call in client.calls] == ["cash_agent", "treasury_agent"]
    assert client.calls[1][1] == "Recommend treasury actions based on observed liquidity gap"
    assert client.calls[1][2] is not None
    assert client.calls[1][2]["dependency_results"]["forecast_cashflow"]["data"]["liquidity_gap"] == 5_000_000
    assert len(results) == 2


async def test_single_cash_plan_finishes_when_no_replan_trigger_is_observed() -> None:
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "agent": "CashAgent",
                "status": "completed",
                "data": {"liquidity_gap": 0, "currency": "CNY"},
                "summary": "No liquidity gap detected.",
            }
        }
    )
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client)
    plan = RoutePlan(
        intent="cash",
        confidence=0.8,
        execution_mode="single",
        steps=[AgentStep(skill_id="forecast_cashflow", task="Assess cash safety")],
        can_replan=True,
        replan_triggers=["liquidity_gap_detected"],
    )

    results = await executor.execute(plan, "帮我看下这两周能不能撑过去")

    assert [call[0] for call in client.calls] == ["cash_agent"]
    assert len(results) == 1


async def test_replanning_respects_max_iterations() -> None:
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "agent": "CashAgent",
                "status": "completed",
                "data": {"liquidity_gap": 5_000_000},
                "summary": "Projected liquidity gap detected.",
            }
        }
    )
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client)
    plan = RoutePlan(
        intent="cash",
        confidence=0.8,
        execution_mode="single",
        steps=[AgentStep(skill_id="forecast_cashflow", task="Assess cash safety")],
        can_replan=True,
        max_iterations=1,
        replan_triggers=["liquidity_gap_detected"],
    )

    results = await executor.execute(plan, "帮我看下这两周能不能撑过去")

    assert [call[0] for call in client.calls] == ["cash_agent"]
    assert len(results) == 1


async def test_executor_stops_when_subagent_awaits_confirmation() -> None:
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "status": "await_confirm",
                "summary": "需要确认是否继续生成资金调拨建议",
                "approval_request": {
                    "approval_id": "approval-1",
                    "question": "是否确认继续？",
                    "options": ["approve", "reject"],
                },
            }
        }
    )
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client)
    plan = RoutePlan(
        intent="cash",
        confidence=0.8,
        execution_mode="sequential",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Assess cash safety"),
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity gap",
                depends_on=["forecast_cashflow"],
            ),
        ],
    )

    results = await executor.execute(plan, "帮我看下这两周能不能撑过去")

    assert [call[0] for call in client.calls] == ["cash_agent"]
    assert results[0]["status"] == "await_confirm"
    assert results[0]["human_action"]["action_type"] == "await_confirm"


async def test_executor_resumes_remaining_steps_after_pending_confirmation_is_approved() -> None:
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "status": "await_confirm",
                "summary": "需要确认是否继续生成资金调拨建议",
                "approval_request": {
                    "approval_id": "approval-1",
                    "question": "是否确认继续？",
                    "options": ["approve", "reject"],
                },
            },
            "treasury_agent": {
                "agent": "TreasuryAgent",
                "status": "completed",
                "summary": "已完成调拨建议",
            },
        }
    )
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client)
    plan = RoutePlan(
        intent="cash",
        confidence=0.8,
        execution_mode="sequential",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Assess cash safety"),
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity gap",
                depends_on=["forecast_cashflow"],
            ),
        ],
    )

    await executor.execute(plan, "帮我看下这两周能不能撑过去", task_id="context-1")
    resumed = await executor.resume_pending_human_action("context-1", "确认，可以执行")

    assert [call[0] for call in client.calls] == ["cash_agent", "treasury_agent"]
    assert resumed == [{"agent": "TreasuryAgent", "status": "completed", "summary": "已完成调拨建议"}]


async def test_executor_context_is_budgeted_and_dependency_scoped() -> None:
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "agent": "CashAgent",
                "status": "completed",
                "summary": "x" * 1000,
                "data": {"liquidity_gap": 5_000_000, "large_payload": "y" * 5000},
            }
        }
    )
    executor = PlanExecutor(registry=AgentRegistry.default_builtin(), a2a_client=client, max_context_tokens=120)
    plan = RoutePlan(
        intent="liquidity_analysis",
        confidence=0.8,
        execution_mode="sequential",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Assess cash safety"),
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity gap",
                depends_on=["forecast_cashflow"],
            ),
        ],
        can_replan=False,
    )

    await executor.execute(plan, "帮我看下这两周能不能撑过去")

    assert client.calls[1][2] is not None
    assert "dependency_results" in client.calls[1][2]
    assert "previous_results" not in client.calls[1][2]
    assert client.calls[1][2]["context_budget"]["estimated_tokens"] <= 120


async def test_executor_uses_llm_context_compression_when_configured() -> None:
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "agent": "CashAgent",
                "status": "completed",
                "summary": "x" * 1000,
                "data": {"liquidity_gap": 5_000_000, "currency": "CNY", "large_payload": "y" * 5000},
            }
        }
    )
    llm = FakeLLMClient("缺口500万CNY")
    executor = PlanExecutor(
        registry=AgentRegistry.default_builtin(),
        a2a_client=client,
        max_context_tokens=260,
        compression_llm_client=llm,
    )
    plan = RoutePlan(
        intent="liquidity_analysis",
        confidence=0.8,
        execution_mode="sequential",
        steps=[
            AgentStep(skill_id="forecast_cashflow", task="Assess cash safety"),
            AgentStep(
                skill_id="analyze_liquidity_gap",
                task="Analyze liquidity gap",
                depends_on=["forecast_cashflow"],
            ),
        ],
        can_replan=False,
    )

    await executor.execute(plan, "帮我看下这两周能不能撑过去")

    assert llm.messages
    assert client.calls[1][2]["context_budget"]["compression_strategy"] == "llm"
