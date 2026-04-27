from app.assistant.orchestrator import TreasuryAssistantOrchestrator
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
            "data": {"source": endpoint.agent_id},
            "summary": f"{endpoint.name} handled {message}",
        }


def build_orchestrator() -> tuple[TreasuryAssistantOrchestrator, RecordingA2AClient]:
    registry = AgentRegistry.default_builtin()
    client = RecordingA2AClient()
    return TreasuryAssistantOrchestrator(registry=registry, a2a_client=client), client


class FixedRoutePlanner:
    def __init__(self, plan: RoutePlan) -> None:
        self.route_plan = plan

    async def plan(self, message: str) -> RoutePlan:
        return self.route_plan


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.messages.append(messages)
        return self.response


async def test_routes_cash_questions_to_cash_agent() -> None:
    orchestrator, client = build_orchestrator()

    result = await orchestrator.invoke("查询公司银行账户余额")

    assert result.intent == "cash"
    assert [call[0] for call in client.calls] == ["cash_agent"]


async def test_routes_policy_questions_to_treasury_agent() -> None:
    orchestrator, client = build_orchestrator()

    result = await orchestrator.invoke("有什么融资策略建议")

    assert result.intent == "treasury"
    assert [call[0] for call in client.calls] == ["treasury_agent"]


async def test_liquidity_analysis_calls_cash_then_treasury() -> None:
    orchestrator, client = build_orchestrator()

    result = await orchestrator.invoke("分析未来两周资金缺口并给调拨建议")

    assert result.intent == "liquidity_analysis"
    assert [call[0] for call in client.calls] == ["cash_agent", "treasury_agent"]
    assert client.calls[1][2] is not None
    assert "dependency_results" in client.calls[1][2]
    assert "forecast_cashflow" in client.calls[1][2]["dependency_results"]


async def test_general_question_calls_agents_in_parallel() -> None:
    orchestrator, client = build_orchestrator()

    result = await orchestrator.invoke("请综合分析当前财资状况")

    assert result.intent == "general"
    assert sorted(call[0] for call in client.calls) == ["cash_agent", "treasury_agent"]
    assert len(result.agent_results) == 2


async def test_stream_emits_step_level_progress_and_final_result() -> None:
    orchestrator, client = build_orchestrator()

    events = [event async for event in orchestrator.stream("请综合分析当前财资状况")]

    step_events = [event for event in events if event.event_type in {"step_started", "step_completed"}]

    assert [event.event_type for event in step_events] == [
        "step_started",
        "step_started",
        "step_completed",
        "step_completed",
    ]
    assert {event.metadata["skill_id"] for event in step_events} == {"forecast_cashflow", "analyze_liquidity_gap"}
    assert {event.metadata["execution_mode"] for event in step_events} == {"parallel"}
    assert events[-1].result is not None
    assert events[-1].result.intent == "general"
    assert sorted(call[0] for call in client.calls) == ["cash_agent", "treasury_agent"]


async def test_stream_emits_dynamic_step_insertion() -> None:
    registry = AgentRegistry.default_builtin()
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
    planner = FixedRoutePlanner(
        RoutePlan(
            intent="dynamic_liquidity_check",
            confidence=1.0,
            execution_mode="single",
            steps=[AgentStep(skill_id="forecast_cashflow", task="Assess cash safety for dynamic orchestration")],
            replan_triggers=["liquidity_gap_detected"],
        )
    )
    orchestrator = TreasuryAssistantOrchestrator(registry=registry, a2a_client=client, route_planner=planner)

    events = [event async for event in orchestrator.stream("动态编排：先读取现金流，如果发现资金缺口再追加融资计划")]

    inserted_events = [event for event in events if event.event_type == "step_inserted"]
    assert len(inserted_events) == 1
    assert inserted_events[0].metadata["skill_id"] == "recommend_funding_plan"
    assert inserted_events[0].metadata["depends_on"] == ["forecast_cashflow"]
    assert [call[0] for call in client.calls] == ["cash_agent", "treasury_agent"]


async def test_no_keyword_request_can_follow_llm_generated_plan() -> None:
    registry = AgentRegistry.default_builtin()
    client = RecordingA2AClient()
    planner = FixedRoutePlanner(
        RoutePlan(
            intent="liquidity_analysis",
            confidence=0.88,
            execution_mode="sequential",
            steps=[
                AgentStep(skill_id="forecast_cashflow", task="Collect liquidity data"),
                AgentStep(
                    skill_id="analyze_liquidity_gap",
                    task="Analyze whether liquidity can hold",
                    depends_on=["forecast_cashflow"],
                ),
            ],
            reason="LLM mapped colloquial request to liquidity analysis.",
        )
    )
    orchestrator = TreasuryAssistantOrchestrator(
        registry=registry,
        a2a_client=client,
        route_planner=planner,
    )

    result = await orchestrator.invoke("帮我看下这两周能不能撑过去")

    assert result.intent == "liquidity_analysis"
    assert [call[0] for call in client.calls] == ["cash_agent", "treasury_agent"]


async def test_synthesizer_uses_llm_prompt_when_client_is_configured() -> None:
    registry = AgentRegistry.default_builtin()
    client = RecordingA2AClient()
    llm = FakeLLMClient("LLM synthesized treasury answer")
    orchestrator = TreasuryAssistantOrchestrator(
        registry=registry,
        a2a_client=client,
        llm_client=llm,
    )

    result = await orchestrator.invoke("查询余额")

    assert result.summary == "LLM synthesized treasury answer"
    assert llm.messages
    assert "synthesizer" in llm.messages[0][0]["content"]


async def test_orchestrator_surfaces_await_confirm_result() -> None:
    registry = AgentRegistry.default_builtin()
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "status": "await_confirm",
                "summary": "需要确认",
                "approval_request": {
                    "approval_id": "approval-1",
                    "question": "是否确认继续？",
                    "options": ["approve", "reject"],
                },
            }
        }
    )
    planner = FixedRoutePlanner(
        RoutePlan(
            intent="cash",
            confidence=0.8,
            execution_mode="single",
            steps=[AgentStep(skill_id="forecast_cashflow", task="Assess cash safety")],
        )
    )
    orchestrator = TreasuryAssistantOrchestrator(registry=registry, a2a_client=client, route_planner=planner)

    result = await orchestrator.invoke("帮我看下这两周能不能撑过去")

    assert result.status == "await_confirm"
    assert result.human_action is not None
    assert result.human_action["action_id"] == "approval-1"
    assert result.summary == "是否确认继续？"


async def test_orchestrator_resumes_pending_confirmation_by_context_id() -> None:
    registry = AgentRegistry.default_builtin()
    client = RecordingA2AClient(
        {
            "cash_agent": {
                "status": "await_confirm",
                "summary": "需要确认",
                "approval_request": {
                    "approval_id": "approval-1",
                    "question": "是否确认继续？",
                    "options": ["approve", "reject"],
                },
            },
            "treasury_agent": {
                "status": "completed",
                "summary": "已完成调拨建议",
            },
        }
    )
    planner = FixedRoutePlanner(
        RoutePlan(
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
        )
    )
    orchestrator = TreasuryAssistantOrchestrator(registry=registry, a2a_client=client, route_planner=planner)

    first = await orchestrator.invoke("帮我看下这两周能不能撑过去", context_id="context-1")
    resumed = await orchestrator.invoke("确认，可以执行", context_id="context-1")

    assert first.status == "await_confirm"
    assert resumed.status == "completed"
    assert resumed.summary == "已完成调拨建议"
    assert [call[0] for call in client.calls] == ["cash_agent", "treasury_agent"]
