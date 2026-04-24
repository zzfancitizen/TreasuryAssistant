from __future__ import annotations

from typing import Any


class CashAgent:
    async def invoke(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "agent": "CashAgent",
            "status": "completed",
            "input": {"message": message, "context": context or {}},
            "data": {
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
            },
            "summary": "Mock cash position shows sufficient CNY liquidity over the next two weeks.",
        }
