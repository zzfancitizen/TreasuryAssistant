from __future__ import annotations

import os

from app.core.a2a_sdk import JsonAgentExecutor, build_a2a_app, build_agent_card, build_skill
from app.core.logging_config import configure_logging
from app.treasury.agent import TreasuryAgent

configure_logging()

agent = TreasuryAgent()
agent_card = build_agent_card(
    name="TreasuryAgent",
    description="Handles liquidity analysis, funding plans, transfer recommendations, and risk checks.",
    url=os.environ.get("TREASURY_AGENT_PUBLIC_URL", "http://localhost:8002"),
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
        build_skill(
            skill_id="recommend_funding_plan",
            name="Recommend funding plan",
            description="Returns mock funding recommendations for treasury planning.",
            tags=["treasury", "funding"],
        ),
        build_skill(
            skill_id="read_treasury_state",
            name="Read treasury state",
            description="Reads mock treasury policy, funding line, and risk state for runtime simulation.",
            tags=["treasury", "read", "runtime"],
        ),
        build_skill(
            skill_id="change_treasury_state",
            name="Change treasury state",
            description="Prepares a mock treasury state change and requires confirmation.",
            tags=["treasury", "change", "runtime"],
        ),
    ],
)
app = build_a2a_app(
    agent_card=agent_card,
    executor=JsonAgentExecutor(agent.invoke),
    title="TreasuryAgent",
)
