from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Part,
    TaskState,
    TextPart,
)


AgentHandler = Callable[[str, dict[str, Any] | None], Awaitable[dict[str, Any]]]


class JsonAgentExecutor(AgentExecutor):
    def __init__(self, handler: AgentHandler) -> None:
        self.handler = handler

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        message = extract_text_message(context.message)
        metadata = context.metadata or {}
        result = await self.handler(message, metadata.get("context"))
        text = json.dumps(result, ensure_ascii=False, indent=2)

        await updater.add_artifact(
            [Part(TextPart(text=text))],
            name="agent_result",
            last_chunk=True,
        )
        await updater.update_status(
            TaskState.completed,
            message=updater.new_agent_message([Part(TextPart(text=text))]),
            final=True,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def extract_text_message(message: Message | None) -> str:
    if message is None:
        return ""

    texts: list[str] = []
    for part in message.parts:
        root = part.root
        if isinstance(root, TextPart):
            texts.append(root.text)
    return "\n".join(texts)


def build_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    skills: list[AgentSkill],
    streaming: bool = True,
) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        url=url,
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=streaming, stateTransitionHistory=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["application/json"],
        skills=skills,
    )


def build_skill(*, skill_id: str, name: str, description: str, tags: list[str]) -> AgentSkill:
    return AgentSkill(
        id=skill_id,
        name=name,
        description=description,
        tags=tags,
        inputModes=["text/plain"],
        outputModes=["application/json"],
    )


def build_a2a_app(*, agent_card: AgentCard, executor: AgentExecutor, title: str):
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    return A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build(title=title)
