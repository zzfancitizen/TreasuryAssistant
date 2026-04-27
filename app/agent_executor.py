from __future__ import annotations

import asyncio
import json
import logging

from a2a.server.agent_execution import AgentExecutor as SDKAgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Part, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from app.assistant.agent import TreasuryAssistantAgent

logger = logging.getLogger(__name__)


class AgentExecutor(SDKAgentExecutor):
    def __init__(self) -> None:
        self._agent: TreasuryAssistantAgent | None = None

    @property
    def agent(self) -> TreasuryAssistantAgent:
        if self._agent is None:
            self._agent = TreasuryAssistantAgent()
        return self._agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        logger.info(
            "assistant.request.received",
                extra={
                    "context_id": task.context_id,
                    "task_id": task.id,
                    "user_input_length": len(query or ""),
                },
            )
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            await updater.start_work()
            payload = None
            result = None
            async for event in self.agent.stream(query, context_id=task.context_id):
                if event.event_type != "completed":
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(event.message, task.context_id, task.id),
                    )
                elif event.event_type == "completed" and event.result is not None:
                    result = event.result
                    payload = {
                        "agent": "TreasuryAssistant",
                        "status": result.status,
                        "intent": result.intent,
                        "summary": result.summary,
                        "agent_results": result.agent_results,
                        "human_action": result.human_action,
                    }

            if payload is None or result is None:
                raise RuntimeError("Assistant stream completed without a final result")

            text = json.dumps(payload, ensure_ascii=False, indent=2)
            await updater.add_artifact(
                [Part(root=TextPart(text=text))],
                name="agent_result",
                last_chunk=True,
            )
            await updater.complete()
            logger.info(
                "assistant.request.completed",
                extra={
                    "context_id": task.context_id,
                    "task_id": task.id,
                    "intent": result.intent,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "assistant.request.failed",
                extra={
                    "context_id": task.context_id,
                    "task_id": task.id,
                    "error": str(exc),
                },
            )
            raise ServerError(error=InternalError()) from exc

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.current_task and context.current_task.id
        logger.info("assistant.request.cancelled", extra={"task_id": task_id})
        raise ServerError(error=UnsupportedOperationError())
