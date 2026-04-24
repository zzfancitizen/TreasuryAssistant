from app.memory import ExecutionState
from app.memory import InMemoryMemoryService
from app.assistant.planner import AgentStep, RoutePlan


def test_in_memory_service_saves_and_loads_execution_state() -> None:
    service = InMemoryMemoryService()
    state = ExecutionState.create(
        user_goal="分析资金情况",
        plan=RoutePlan(
            intent="cash",
            confidence=0.8,
            execution_mode="single",
            steps=[AgentStep(skill_id="forecast_cashflow", task="Forecast cashflow")],
        ),
        task_id="task-1",
    )
    state.record_step_result("forecast_cashflow", {"status": "completed", "summary": "完成"})

    service.save(state)

    loaded = service.get("task-1")
    assert loaded is state
    assert loaded.step_results["forecast_cashflow"]["summary"] == "完成"


def test_in_memory_service_deletes_execution_state() -> None:
    service = InMemoryMemoryService()
    state = ExecutionState.create(
        user_goal="分析资金情况",
        plan=RoutePlan(
            intent="cash",
            confidence=0.8,
            execution_mode="single",
            steps=[AgentStep(skill_id="forecast_cashflow", task="Forecast cashflow")],
        ),
        task_id="task-1",
    )

    service.save(state)
    service.delete("task-1")

    assert service.get("task-1") is None
