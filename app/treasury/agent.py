from __future__ import annotations

from typing import Any


class TreasuryAgent:
    async def invoke(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        skill_id = _current_skill_id(context)
        if skill_id == "read_treasury_state":
            return self.read_state(message, context)
        if skill_id == "change_treasury_state":
            return self.change_state(message, context)
        return {
            "agent": "TreasuryAgent",
            "status": "completed",
            "input": {"message": message, "context": context or {}},
            "data": {
                "recommendation_type": "liquidity_management",
                "liquidity_gap": 0,
                "recommended_actions": [
                    "Keep current cash buffer unchanged.",
                    "Monitor week-1 outflow before initiating transfers.",
                ],
                "risk_level": "low",
                "requires_human_approval": False,
            },
            "summary": "Mock treasury analysis recommends no immediate transfer under current assumptions.",
        }

    def read_state(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": "TreasuryAgent",
            "status": "completed",
            "operation": "read",
            "input": {"message": message, "context": context or {}},
            "data": {
                "policy_version": "treasury-policy-v4",
                "liquidity_buffer_minimum": 8_000_000,
                "funding_lines": [
                    {"facility_id": "RCF-001", "limit": 30_000_000, "drawn": 0},
                    {"facility_id": "BILATERAL-002", "limit": 15_000_000, "drawn": 2_000_000},
                ],
                "risk_limits": {"single_bank_limit": 0.45, "minimum_days_cash": 14},
                "last_updated": "2026-04-27T09:15:00+08:00",
            },
            "summary": "Read mock treasury policy, funding line, and risk limit state.",
        }

    def change_state(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": "TreasuryAgent",
            "status": "await_confirm",
            "operation": "change",
            "input": {"message": message, "context": context or {}},
            "data": {
                "change_id": "treasury-change-001",
                "target": "funding_plan",
                "recommended_facility": "RCF-001",
                "proposed_draw": 5_000_000,
                "currency": "CNY",
                "requires_human_approval": True,
            },
            "approval_request": {
                "approval_id": "treasury-change-001",
                "question": "是否确认模拟修改融资计划？",
                "options": ["approve", "reject", "modify"],
                "reason": "Treasury state changes require approval before execution.",
            },
            "summary": "Prepared a mock treasury state change and paused for human confirmation.",
        }


def _current_skill_id(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return ""
    current_step = context.get("current_step")
    if not isinstance(current_step, dict):
        return ""
    return str(current_step.get("skill_id") or "")
