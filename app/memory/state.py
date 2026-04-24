from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.assistant.planner import AgentStep, RoutePlan

HumanActionType = Literal["await_input", "await_confirm"]
ExecutionStatus = Literal["planning", "executing", "await_input", "await_confirm", "completed", "failed"]


class HumanAction(BaseModel):
    action_type: HumanActionType
    action_id: str
    source_skill_id: str
    question: str
    reason: str = ""
    options: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    resume_step: AgentStep | None = None


class ExecutionState(BaseModel):
    task_id: str
    user_goal: str
    plan: RoutePlan
    status: ExecutionStatus = "planning"
    current_step: AgentStep | None = None
    step_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    pending_human_action: HumanAction | None = None

    @classmethod
    def create(cls, *, user_goal: str, plan: RoutePlan, task_id: str | None = None) -> "ExecutionState":
        return cls(task_id=task_id or f"task-{uuid4().hex}", user_goal=user_goal, plan=plan)

    def start_step(self, step: AgentStep) -> None:
        self.status = "executing"
        self.current_step = step

    def record_step_result(self, skill_id: str, result: dict[str, Any]) -> None:
        self.step_results[skill_id] = result
        self._extract_facts(skill_id, result)
        self._extract_artifacts(skill_id, result)

    def set_pending_human_action(self, action: HumanAction) -> None:
        self.pending_human_action = action
        self.status = action.action_type

    def complete(self) -> None:
        self.status = "completed"
        self.current_step = None
        self.pending_human_action = None

    def _extract_facts(self, skill_id: str, result: dict[str, Any]) -> None:
        data = result.get("data")
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if isinstance(value, str | int | float | bool) or value is None:
                self.facts.append({"source_skill_id": skill_id, "key": key, "value": value})

    def _extract_artifacts(self, skill_id: str, result: dict[str, Any]) -> None:
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list):
            return
        for artifact in artifacts:
            if isinstance(artifact, dict):
                self.artifacts.append({"source_skill_id": skill_id, **artifact})
