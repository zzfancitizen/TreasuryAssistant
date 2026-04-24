from __future__ import annotations

from app.cash.agent import CashAgent
from app.core.a2a_sdk import JsonAgentExecutor, build_a2a_app, build_agent_card, build_skill
from app.core.logging_config import configure_logging

configure_logging()

agent = CashAgent()
agent_card = build_agent_card(
    name="CashAgent",
    description="Handles cash balances, bank accounts, transactions, and cashflow forecasts.",
    url="http://localhost:8001",
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
    ],
)
app = build_a2a_app(
    agent_card=agent_card,
    executor=JsonAgentExecutor(agent.invoke),
    title="CashAgent",
)
