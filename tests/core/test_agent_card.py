from app.core.a2a_sdk import build_agent_card, build_skill


def test_build_agent_card_exposes_sdk_agent_card_metadata() -> None:
    card = build_agent_card(
        name="CashAgent",
        description="Handles cash data",
        url="http://localhost:8001",
        skills=[
            build_skill(
                skill_id="get_cash_balance",
                name="Get cash balance",
                description="Returns balances",
                tags=["cash"],
            )
        ],
    )

    dumped = card.model_dump(by_alias=True, exclude_none=True)
    assert dumped["name"] == "CashAgent"
    assert dumped["description"] == "Handles cash data"
    assert dumped["url"] == "http://localhost:8001"
    assert dumped["protocolVersion"] == "0.3.0"
    assert dumped["preferredTransport"] == "JSONRPC"
    assert dumped["skills"][0]["id"] == "get_cash_balance"
