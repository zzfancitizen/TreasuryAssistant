from __future__ import annotations

import os

import click
import uvicorn

from app.agent_executor import AgentExecutor
from app.core.a2a_sdk import build_a2a_app, build_agent_card, build_skill
from app.core.logging_config import configure_logging

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_PUBLIC_HOST = "localhost" if HOST in {"0.0.0.0", "::"} else HOST

configure_logging()

agent_card = build_agent_card(
    name="TreasuryAssistant",
    description="Super agent for treasury business orchestration via A2A subagents.",
    url=os.environ.get("AGENT_PUBLIC_URL") or f"http://{DEFAULT_PUBLIC_HOST}:{PORT}/",
    skills=[
        build_skill(
            skill_id="answer_treasury_questions",
            name="Answer treasury questions",
            description="Routes treasury questions to mock CashAgent and TreasuryAgent subagents.",
            tags=["treasury", "assistant", "orchestration"],
        ),
        build_skill(
            skill_id="analyze_cash_position",
            name="Analyze cash position",
            description="Combines cash and treasury subagent outputs for liquidity analysis.",
            tags=["cash", "liquidity", "analysis"],
        ),
    ],
)
app = build_a2a_app(
    agent_card=agent_card,
    executor=AgentExecutor(),
    title="TreasuryAssistant",
)


@click.command()
@click.option("--host", default=HOST)
@click.option("--port", default=PORT)
def main(host: str, port: int) -> None:
    configure_logging()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
