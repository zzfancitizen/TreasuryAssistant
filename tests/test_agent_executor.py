from app.agent_executor import AgentExecutor


def test_agent_executor_lazily_builds_orchestrator() -> None:
    executor = AgentExecutor()

    assert executor._agent is None
    assert executor.agent is executor.agent
