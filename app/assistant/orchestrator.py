from __future__ import annotations

from typing import Any
from typing import Protocol

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from app.assistant.plan_executor import A2AInvoker, PlanExecutor
from app.assistant.planner import RoutePlan, RoutePlanner
from app.assistant.prompts import SYNTHESIZER_SYSTEM_PROMPT
from app.assistant.types import AssistantResult, AssistantStreamEvent
from app.core.registry import AgentRegistry


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


class OrchestratorState(TypedDict, total=False):
    message: str
    intent: str
    route_plan: RoutePlan
    agent_results: list[dict[str, Any]]
    summary: str
    status: str
    human_action: dict[str, Any] | None


class TreasuryAssistantOrchestrator:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        a2a_client: A2AInvoker,
        route_planner: RoutePlanner | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.registry = registry
        self.a2a_client = a2a_client
        self.route_planner = route_planner or RoutePlanner(registry=registry)
        self.plan_executor = PlanExecutor(
            registry=registry,
            a2a_client=a2a_client,
            compression_llm_client=llm_client,
        )
        self.llm_client = llm_client
        self.graph = self._build_graph()

    async def invoke(self, message: str) -> AssistantResult:
        state = await self.graph.ainvoke({"message": message})
        return AssistantResult(
            intent=state["intent"],
            summary=state["summary"],
            agent_results=state["agent_results"],
            status=state.get("status", "completed"),
            human_action=state.get("human_action"),
        )

    async def stream(self, message: str):
        yield AssistantStreamEvent(event_type="working", message="正在识别用户意图")
        plan = await self.route_planner.plan(message)
        yield AssistantStreamEvent(event_type="working", message=f"已生成执行计划: {plan.intent} ({plan.execution_mode})")

        if plan.execution_mode == "sequential":
            yield AssistantStreamEvent(event_type="working", message="正在按依赖顺序调用下游 agents")
        elif plan.execution_mode == "parallel":
            yield AssistantStreamEvent(event_type="working", message="正在并发调用下游 agents")
        else:
            yield AssistantStreamEvent(event_type="working", message=f"正在调用 skill {plan.steps[0].skill_id}")

        result = await self._execute_plan_direct(message, plan)
        yield AssistantStreamEvent(event_type="working", message="正在汇总 agent 结果")
        if result.status in {"await_input", "await_confirm"}:
            yield AssistantStreamEvent(event_type="completed", message="等待用户输入或确认", result=result)
        else:
            yield AssistantStreamEvent(event_type="completed", message="财资分析已完成", result=result)

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("plan_route", self._plan_route)
        graph.add_node("execute_plan", self._execute_plan)
        graph.add_node("synthesize", self._synthesize)

        graph.set_entry_point("plan_route")
        graph.add_edge("plan_route", "execute_plan")
        graph.add_edge("execute_plan", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    async def _plan_route(self, state: OrchestratorState) -> OrchestratorState:
        plan = await self.route_planner.plan(state["message"])
        return {"intent": plan.intent, "route_plan": plan}

    async def _execute_plan(self, state: OrchestratorState) -> OrchestratorState:
        agent_results = await self.plan_executor.execute(state["route_plan"], state["message"])
        return {"agent_results": agent_results}

    async def _execute_plan_direct(self, message: str, plan: RoutePlan) -> AssistantResult:
        state: OrchestratorState = {
            "message": message,
            "intent": plan.intent,
            "route_plan": plan,
            "agent_results": await self.plan_executor.execute(plan, message),
        }
        summary_state = await self._synthesize(state)
        return AssistantResult(
            intent=plan.intent,
            summary=summary_state["summary"],
            agent_results=state["agent_results"],
            status=summary_state.get("status", "completed"),
            human_action=summary_state.get("human_action"),
        )

    async def _synthesize(self, state: OrchestratorState) -> OrchestratorState:
        human_action_result = next(
            (
                result
                for result in state["agent_results"]
                if result.get("status") in {"await_input", "await_confirm"} and result.get("human_action")
            ),
            None,
        )
        if human_action_result is not None:
            human_action = human_action_result["human_action"]
            return {
                "status": human_action["action_type"],
                "human_action": human_action,
                "summary": human_action.get("question") or human_action_result.get("summary", "等待用户输入或确认。"),
            }

        summaries = [
            result.get("summary", f"{result.get('agent', 'Agent')} completed")
            for result in state["agent_results"]
        ]
        if self.llm_client is not None:
            response = await self.llm_client.complete(
                [
                    {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Subagent results:\n{state['agent_results']}"},
                ],
                temperature=0.1,
                max_tokens=800,
            )
            return {"summary": response}
        return {"summary": "\n".join(summaries)}
