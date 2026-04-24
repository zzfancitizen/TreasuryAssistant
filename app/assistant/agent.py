from __future__ import annotations

import os

from app.assistant.orchestrator import TreasuryAssistantOrchestrator
from app.assistant.planner import RoutePlanner
from app.assistant.types import AssistantResult, AssistantStreamEvent
from app.core.a2a_client import A2AClient
from app.core.llm_client import LiteLLMClient
from app.core.registry import AgentRegistry


class TreasuryAssistantAgent:
    def __init__(self, orchestrator: TreasuryAssistantOrchestrator | None = None) -> None:
        self._orchestrator = orchestrator

    @property
    def orchestrator(self) -> TreasuryAssistantOrchestrator:
        if self._orchestrator is None:
            registry = AgentRegistry.default_local()
            route_planner = None
            if os.getenv("LITELLM_MODEL"):
                route_planner = RoutePlanner(registry=registry, llm_client=LiteLLMClient())
            self._orchestrator = TreasuryAssistantOrchestrator(
                registry=registry,
                a2a_client=A2AClient(),
                route_planner=route_planner,
                llm_client=LiteLLMClient() if os.getenv("LITELLM_MODEL") else None,
            )
        return self._orchestrator

    async def invoke(self, message: str) -> AssistantResult:
        return await self.orchestrator.invoke(message)

    async def stream(self, message: str):
        async for event in self.orchestrator.stream(message):
            yield event


__all__ = ["AssistantResult", "AssistantStreamEvent", "TreasuryAssistantAgent"]
