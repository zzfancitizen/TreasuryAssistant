from __future__ import annotations

import json
import logging
from typing import Any

from app.memory import ExecutionState
from app.assistant.planner import AgentStep
from app.core.skill_registry import SkillDescriptor

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(self, *, max_context_tokens: int = 200_000, chars_per_token: int = 4) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_context_chars = max_context_tokens * chars_per_token
        self.chars_per_token = chars_per_token

    def build(self, *, state: ExecutionState, step: AgentStep, skill: SkillDescriptor) -> dict[str, Any]:
        context = {
            "task_id": state.task_id,
            "user_goal": state.user_goal,
            "current_step": step.model_dump(),
            "skill": {
                "skill_id": skill.skill_id,
                "provider_agent_id": skill.provider_agent_id,
                "name": skill.name,
                "description": skill.description,
                "tags": list(skill.tags),
            },
            "route_plan": {
                "intent": state.plan.intent,
                "execution_mode": state.plan.execution_mode,
                "steps": [route_step.model_dump() for route_step in state.plan.steps],
            },
            "dependency_results": {
                dependency: self._summarize_result(state.step_results[dependency])
                for dependency in step.depends_on
                if dependency in state.step_results
            },
            "facts": list(state.facts),
            "artifacts": self._artifact_refs(state.artifacts),
            "context_budget": {
                "max_tokens": self.max_context_tokens,
                "estimated_tokens": 0,
                "truncated": False,
            },
        }
        return self._fit_budget(context)

    def _fit_budget(self, context: dict[str, Any]) -> dict[str, Any]:
        estimated_tokens = self.estimate_tokens(context)
        if estimated_tokens <= self.max_context_tokens:
            context["context_budget"]["estimated_tokens"] = estimated_tokens
            logger.info(
                "context_builder.context.within_budget",
                extra={"estimated_tokens": estimated_tokens, "max_tokens": self.max_context_tokens},
            )
            return context

        truncated = dict(context)
        truncated["dependency_results"] = {
            skill_id: self._trim_result(result)
            for skill_id, result in context["dependency_results"].items()
        }
        truncated["facts"] = context["facts"][:50]
        truncated["artifacts"] = context["artifacts"][:20]
        truncated["context_budget"] = {
            "max_tokens": self.max_context_tokens,
            "estimated_tokens": 0,
            "truncated": True,
        }

        while self.estimate_tokens(truncated) > self.max_context_tokens and truncated["facts"]:
            truncated["facts"].pop()
        while self.estimate_tokens(truncated) > self.max_context_tokens and truncated["artifacts"]:
            truncated["artifacts"].pop()
        if self.estimate_tokens(truncated) > self.max_context_tokens:
            truncated["dependency_results"] = {
                skill_id: {"status": result.get("status"), "summary": self._clip(str(result.get("summary", "")), 120)}
                for skill_id, result in truncated["dependency_results"].items()
            }

        truncated["context_budget"]["estimated_tokens"] = min(
            self.estimate_tokens(truncated),
            self.max_context_tokens,
        )
        logger.warning(
            "context_builder.context.truncated",
            extra={
                "estimated_tokens": truncated["context_budget"]["estimated_tokens"],
                "max_tokens": self.max_context_tokens,
            },
        )
        return truncated

    def estimate_tokens(self, payload: dict[str, Any]) -> int:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return max(1, len(serialized) // self.chars_per_token)

    def _summarize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": result.get("status"),
            "summary": result.get("summary"),
            "data": result.get("data"),
            "human_action": result.get("human_action"),
            "artifacts": self._artifact_refs(result.get("artifacts", [])),
        }

    def _trim_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": result.get("status"),
            "summary": self._clip(str(result.get("summary", "")), 500),
            "data": self._trim_data(result.get("data")),
            "human_action": result.get("human_action"),
            "artifacts": self._artifact_refs(result.get("artifacts", []))[:5],
        }

    def _trim_data(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        trimmed: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                trimmed[key] = self._clip(value, 300)
            elif isinstance(value, int | float | bool) or value is None:
                trimmed[key] = value
        return trimmed

    def _artifact_refs(self, artifacts: Any) -> list[dict[str, Any]]:
        if not isinstance(artifacts, list):
            return []
        refs = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            refs.append(
                {
                    "artifact_id": artifact.get("artifact_id") or artifact.get("id"),
                    "type": artifact.get("type") or artifact.get("mime_type"),
                    "source_skill_id": artifact.get("source_skill_id"),
                    "summary": artifact.get("summary"),
                }
            )
        return refs

    @staticmethod
    def _clip(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}...[truncated]"
