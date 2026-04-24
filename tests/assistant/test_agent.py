from app.assistant.agent import TreasuryAssistantAgent
from app.assistant.orchestrator import TreasuryAssistantOrchestrator
from app.core.registry import AgentEndpoint, AgentRegistry


class RecordingA2AClient:
    async def invoke(self, endpoint: AgentEndpoint, message: str, context: dict | None = None) -> dict:
        return {
            "agent": endpoint.name,
            "status": "completed",
            "summary": f"{endpoint.name} handled {message}",
        }


async def test_assistant_agent_invokes_orchestrator() -> None:
    orchestrator = TreasuryAssistantOrchestrator(
        registry=AgentRegistry.default_local(),
        a2a_client=RecordingA2AClient(),
    )
    agent = TreasuryAssistantAgent(orchestrator=orchestrator)

    result = await agent.invoke("查询余额")

    assert result.intent == "cash"
