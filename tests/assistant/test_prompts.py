from app.assistant.prompts import (
    ASSISTANT_SYSTEM_PROMPT,
    CONTINUATION_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_router_system_prompt,
)
from app.core.registry import AgentEndpoint, AgentRegistry


def test_assistant_system_prompt_defines_super_agent_role() -> None:
    assert "TreasuryAssistant" in ASSISTANT_SYSTEM_PROMPT
    assert "A2A" in ASSISTANT_SYSTEM_PROMPT
    assert "subagents" in ASSISTANT_SYSTEM_PROMPT


def test_router_prompt_includes_agent_card_capabilities() -> None:
    registry = AgentRegistry(
        [
            AgentEndpoint(
                agent_id="risk_agent",
                name="RiskAgent",
                url="http://localhost:8003",
                capabilities=("check_counterparty_risk",),
                description="Handles risk checks.",
            )
        ]
    )

    prompt = build_router_system_prompt(registry)

    assert "risk_agent" in prompt
    assert "check_counterparty_risk" in prompt
    assert "execution_mode" in prompt
    assert "depends_on" in prompt


def test_synthesizer_prompt_requires_grounded_output() -> None:
    assert "Do not invent" in SYNTHESIZER_SYSTEM_PROMPT
    assert "subagent" in SYNTHESIZER_SYSTEM_PROMPT


def test_continuation_prompt_defines_runtime_replanning() -> None:
    assert "insert_step" in CONTINUATION_SYSTEM_PROMPT
    assert "finish" in CONTINUATION_SYSTEM_PROMPT
