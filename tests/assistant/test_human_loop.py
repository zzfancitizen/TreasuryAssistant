from unittest.mock import patch

from app.memory import ExecutionState, HumanAction
from app.assistant.planner import AgentStep, RoutePlan
from app.assistant.result_normalizer import normalize_agent_result
from app.assistant.turn_classifier import classify_user_turn


def build_state() -> ExecutionState:
    return ExecutionState.create(
        user_goal="分析资金情况",
        plan=RoutePlan(
            intent="general",
            confidence=0.8,
            execution_mode="single",
            steps=[AgentStep(skill_id="forecast_cashflow", task="Forecast cashflow")],
        ),
    )


def test_normalizes_await_input_result() -> None:
    normalized = normalize_agent_result(
        "forecast_cashflow",
        {
            "status": "await_input",
            "summary": "需要补充时间范围",
            "question": "请确认分析未来一周还是两周？",
            "missing_fields": ["time_horizon"],
        },
    )

    assert normalized.status == "await_input"
    assert normalized.human_action is not None
    assert normalized.human_action.action_type == "await_input"
    assert normalized.human_action.missing_fields == ("time_horizon",)


def test_normalizes_await_confirm_result() -> None:
    normalized = normalize_agent_result(
        "recommend_cash_transfer",
        {
            "status": "await_confirm",
            "summary": "需要确认关键操作",
            "approval_request": {
                "approval_id": "approval-1",
                "question": "是否确认生成调拨方案？",
                "options": ["approve", "reject"],
            },
        },
    )

    assert normalized.status == "await_confirm"
    assert normalized.human_action is not None
    assert normalized.human_action.action_type == "await_confirm"
    assert normalized.human_action.action_id == "approval-1"


def test_execution_state_persists_pending_human_action() -> None:
    state = build_state()
    action = HumanAction(
        action_type="await_confirm",
        action_id="approval-1",
        source_skill_id="recommend_cash_transfer",
        question="是否确认？",
    )

    state.set_pending_human_action(action)

    assert state.status == "await_confirm"
    assert state.pending_human_action == action


def test_turn_classifier_routes_pending_confirmation_answer() -> None:
    state = build_state()
    state.set_pending_human_action(
        HumanAction(
            action_type="await_confirm",
            action_id="approval-1",
            source_skill_id="recommend_cash_transfer",
            question="是否确认？",
        )
    )

    turn = classify_user_turn("确认，可以执行", state)

    assert turn.turn_type == "answer_to_pending_human_action"
    assert turn.decision == "approve"


def test_turn_classifier_identifies_followup_without_pending_action() -> None:
    state = build_state()
    state.record_step_result(
        "forecast_cashflow",
        {"status": "completed", "summary": "预计存在 500 万缺口"},
    )

    turn = classify_user_turn("为什么会有这个缺口？", state)

    assert turn.turn_type == "followup_on_current_task"


def test_failed_agent_result_logs_error() -> None:
    with patch("app.assistant.result_normalizer.logger.error") as log_error:
        normalized = normalize_agent_result("forecast_cashflow", {"status": "failed", "summary": "boom"})

    assert normalized.status == "failed"
    log_error.assert_called_once()
    assert log_error.call_args.args == ("agent_result.failed",)
