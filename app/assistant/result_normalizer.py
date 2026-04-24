from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from app.memory import HumanAction

logger = logging.getLogger(__name__)

NormalizedStatus = Literal["completed", "failed", "await_input", "await_confirm"]


class NormalizedAgentResult(BaseModel):
    skill_id: str
    status: NormalizedStatus
    result: dict[str, Any]
    human_action: HumanAction | None = None


def normalize_agent_result(skill_id: str, result: dict[str, Any]) -> NormalizedAgentResult:
    raw_status = str(result.get("status", "completed"))
    if raw_status in {"await_input", "needs_clarification", "waiting_for_input"}:
        action = _build_await_input_action(skill_id, result)
        logger.info("agent_result.await_input", extra={"skill_id": skill_id, "action_id": action.action_id})
        return NormalizedAgentResult(skill_id=skill_id, status="await_input", result=result, human_action=action)

    data = result.get("data")
    requires_human_approval = isinstance(data, dict) and data.get("requires_human_approval") is True
    if raw_status in {"await_confirm", "requires_approval", "waiting_for_confirmation"} or requires_human_approval:
        action = _build_await_confirm_action(skill_id, result)
        logger.info("agent_result.await_confirm", extra={"skill_id": skill_id, "action_id": action.action_id})
        return NormalizedAgentResult(skill_id=skill_id, status="await_confirm", result=result, human_action=action)

    if raw_status == "failed":
        logger.error("agent_result.failed", extra={"skill_id": skill_id})
        return NormalizedAgentResult(skill_id=skill_id, status="failed", result=result)

    return NormalizedAgentResult(skill_id=skill_id, status="completed", result=result)


def result_with_human_action(normalized: NormalizedAgentResult) -> dict[str, Any]:
    if normalized.human_action is None:
        return normalized.result
    enriched = dict(normalized.result)
    enriched["status"] = normalized.status
    enriched["human_action"] = normalized.human_action.model_dump()
    return enriched


def _build_await_input_action(skill_id: str, result: dict[str, Any]) -> HumanAction:
    return HumanAction(
        action_type="await_input",
        action_id=str(result.get("input_request_id") or result.get("request_id") or f"input-{uuid4().hex}"),
        source_skill_id=skill_id,
        question=str(result.get("question") or result.get("summary") or "Please provide additional input."),
        reason=str(result.get("reason") or result.get("summary") or ""),
        missing_fields=tuple(result.get("missing_fields") or ()),
    )


def _build_await_confirm_action(skill_id: str, result: dict[str, Any]) -> HumanAction:
    approval_request = result.get("approval_request") if isinstance(result.get("approval_request"), dict) else {}
    return HumanAction(
        action_type="await_confirm",
        action_id=str(
            approval_request.get("approval_id")
            or result.get("approval_id")
            or result.get("confirmation_id")
            or f"confirm-{uuid4().hex}"
        ),
        source_skill_id=skill_id,
        question=str(
            approval_request.get("question")
            or result.get("question")
            or result.get("summary")
            or "Please confirm before continuing."
        ),
        reason=str(approval_request.get("reason") or result.get("reason") or result.get("summary") or ""),
        options=tuple(approval_request.get("options") or result.get("options") or ("approve", "reject", "modify")),
    )
