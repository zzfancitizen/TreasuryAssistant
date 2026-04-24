from __future__ import annotations

from app.core.a2a_sdk import JsonAgentExecutor, build_a2a_app, build_agent_card, build_skill
from app.core.logging_config import configure_logging
from app.treasury.agent import TreasuryAgent

configure_logging()

agent = TreasuryAgent()
agent_card = build_agent_card(
    name="TreasuryAgent",
    description="Handles liquidity analysis, funding plans, transfer recommendations, and risk checks.",
    url="http://localhost:8002",
    skills=[
        build_skill(
            skill_id="analyze_liquidity_gap",
            name="Analyze liquidity gap",
            description="Returns mock liquidity gap and risk analysis.",
            tags=["treasury", "liquidity"],
        ),
        build_skill(
            skill_id="recommend_cash_transfer",
            name="Recommend cash transfer",
            description="Returns mock cash transfer recommendations.",
            tags=["treasury", "transfer"],
        ),
    ],
)
app = build_a2a_app(
    agent_card=agent_card,
    executor=JsonAgentExecutor(agent.invoke),
    title="TreasuryAgent",
)
