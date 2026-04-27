from __future__ import annotations

from typing import Any


class CashAgent:
    async def invoke(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        skill_id = _current_skill_id(context)
        if skill_id == "read_cash_state":
            return self.read_state(message, context)
        if skill_id == "change_cash_state":
            return self.change_state(message, context)
        data: dict[str, Any] = {
            "currency": "CNY",
            "available_balance": 12_500_000,
            "restricted_cash": 1_000_000,
            "cashflow_forecast": [
                {"period": "week_1", "net_flow": -2_000_000},
                {"period": "week_2", "net_flow": 3_200_000},
            ],
            "accounts": [
                {"bank": "Mock Bank A", "balance": 8_000_000},
                {"bank": "Mock Bank B", "balance": 4_500_000},
            ],
        }
        summary = "Mock cash position shows sufficient CNY liquidity over the next two weeks."
        if _is_dynamic_orchestration_case(message, context):
            data["liquidity_gap"] = 5_000_000
            summary = "Mock cash forecast detected a CNY 5,000,000 liquidity gap for dynamic orchestration."
        return {
            "agent": "CashAgent",
            "status": "completed",
            "input": {"message": message, "context": context or {}},
            "data": data,
            "summary": summary,
        }

    def read_state(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": "CashAgent",
            "status": "completed",
            "operation": "read",
            "input": {"message": message, "context": context or {}},
            "data": {
                "ledger_version": "cash-ledger-v7",
                "currency": "CNY",
                "available_balance": 12_500_000,
                "restricted_cash": 1_000_000,
                "bank_accounts": [
                    {"account_id": "CASH-CNY-001", "bank": "Mock Bank A", "balance": 8_000_000},
                    {"account_id": "CASH-CNY-002", "bank": "Mock Bank B", "balance": 4_500_000},
                ],
                "last_updated": "2026-04-27T09:00:00+08:00",
            },
            "summary": "Read mock cash ledger state with two active CNY bank accounts.",
        }

    def change_state(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": "CashAgent",
            "status": "await_confirm",
            "operation": "change",
            "input": {"message": message, "context": context or {}},
            "data": {
                "change_id": "cash-change-001",
                "target": "cash_pool_buffer",
                "proposed_buffer": 10_000_000,
                "currency": "CNY",
                "requires_human_approval": True,
            },
            "approval_request": {
                "approval_id": "cash-change-001",
                "question": "是否确认模拟调整现金池目标余额？",
                "options": ["approve", "reject", "modify"],
                "reason": "Cash state changes require treasury operator confirmation in runtime simulation.",
            },
            "summary": "Prepared a mock cash state change and paused for human confirmation.",
        }


def _current_skill_id(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return ""
    current_step = context.get("current_step")
    if not isinstance(current_step, dict):
        return ""
    return str(current_step.get("skill_id") or "")


def _is_dynamic_orchestration_case(message: str, context: dict[str, Any] | None) -> bool:
    text = message.lower()
    if "动态编排" in text or "dynamic orchestration" in text or "dynamic" in text:
        return True
    if not isinstance(context, dict):
        return False
    current_step = context.get("current_step")
    if not isinstance(current_step, dict):
        return False
    task = str(current_step.get("task") or "").lower()
    return "dynamic orchestration" in task
