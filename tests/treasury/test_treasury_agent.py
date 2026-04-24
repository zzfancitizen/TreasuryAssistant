from app.treasury.agent import TreasuryAgent


async def test_treasury_agent_returns_mock_recommendation() -> None:
    result = await TreasuryAgent().invoke(
        "给出资金调拨建议",
        context={"cash_result": {"data": {"available_balance": 100}}},
    )

    assert result["agent"] == "TreasuryAgent"
    assert result["status"] == "completed"
    assert result["data"]["recommendation_type"] == "liquidity_management"
    assert result["data"]["requires_human_approval"] is False
