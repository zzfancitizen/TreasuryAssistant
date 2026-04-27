from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AssistantResult(BaseModel):
    intent: str
    summary: str
    agent_results: list[dict[str, Any]]
    status: str = "completed"
    human_action: dict[str, Any] | None = None


class AssistantStreamEvent(BaseModel):
    event_type: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    result: AssistantResult | None = None
