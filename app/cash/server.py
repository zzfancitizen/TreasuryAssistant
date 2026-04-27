from __future__ import annotations

import os

from app.cash.agent import CashAgent
from app.core.a2a_sdk import JsonAgentExecutor, build_a2a_app, build_agent_card, build_skill
from app.core.logging_config import configure_logging

configure_logging()

agent = CashAgent()
agent_card = build_agent_card(
    name="CashAgent",
    description="Handles cash balances, bank accounts, transactions, and cashflow forecasts.",
    url=os.environ.get("CASH_AGENT_PUBLIC_URL", "http://localhost:8001"),
    skills=[
        build_skill(
            skill_id="get_cash_balance",
            name="Get cash balance",
            description="Returns mock available balance, restricted cash, and account balances.",
            tags=["cash", "balance"],
        ),
        build_skill(
            skill_id="forecast_cashflow",
            name="Forecast cashflow",
            description="Returns mock two-week cashflow forecast data.",
            tags=["cash", "forecast"],
        ),
        build_skill(
            skill_id="read_cash_state",
            name="Read cash state",
            description="Reads mock cash ledger, account, and liquidity state for runtime simulation.",
            tags=["cash", "read", "runtime"],
        ),
        build_skill(
            skill_id="change_cash_state",
            name="Change cash state",
            description="Prepares a mock cash state change and requires confirmation.",
            tags=["cash", "change", "runtime"],
        ),
    ],
)
app = build_a2a_app(
    agent_card=agent_card,
    executor=JsonAgentExecutor(agent.invoke),
    title="CashAgent",
)
