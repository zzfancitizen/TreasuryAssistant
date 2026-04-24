from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.memory import ExecutionState

TurnType = Literal[
    "answer_to_pending_human_action",
    "followup_on_current_task",
    "new_task",
    "memory_query",
    "correction_or_override",
]
TurnDecision = Literal["approve", "reject", "modify"] | None


class UserTurn(BaseModel):
    turn_type: TurnType
    decision: TurnDecision = None
    reason: str = ""


def classify_user_turn(message: str, state: ExecutionState | None) -> UserTurn:
    normalized = message.strip().lower()
    if state and state.pending_human_action:
        decision = _classify_pending_action_answer(normalized)
        if decision:
            return UserTurn(
                turn_type="answer_to_pending_human_action",
                decision=decision,
                reason="Matched pending human action response.",
            )

    if _contains_any(normalized, ("上次", "刚才", "之前", "last time", "previous")):
        return UserTurn(turn_type="memory_query", reason="Message references prior memory.")
    if _contains_any(normalized, ("改成", "换成", "重新", "instead", "change to")):
        return UserTurn(turn_type="correction_or_override", reason="Message modifies current constraints.")
    if state and state.step_results and _contains_any(normalized, ("为什么", "原因", "怎么", "如果", "why", "how", "what if")):
        return UserTurn(turn_type="followup_on_current_task", reason="Message asks about current task results.")
    return UserTurn(turn_type="new_task", reason="No active task reference detected.")


def _classify_pending_action_answer(message: str) -> TurnDecision:
    if _contains_any(message, ("确认", "同意", "可以", "批准", "approve", "yes", "ok")):
        return "approve"
    if _contains_any(message, ("拒绝", "不同意", "不要", "取消", "reject", "no")):
        return "reject"
    if _contains_any(message, ("改成", "修改", "换成", "modify", "change")):
        return "modify"
    return None


def _contains_any(message: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in message for keyword in keywords)
