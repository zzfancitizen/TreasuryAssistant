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


async def test_treasury_agent_mocks_read_and_change_methods() -> None:
    read_result = await TreasuryAgent().invoke(
        "读取财资策略",
        context={"current_step": {"skill_id": "read_treasury_state"}},
    )
    change_result = await TreasuryAgent().invoke(
        "修改融资计划",
        context={"current_step": {"skill_id": "change_treasury_state"}},
    )

    assert read_result["operation"] == "read"
    assert read_result["data"]["policy_version"]
    assert change_result["operation"] == "change"
    assert change_result["data"]["change_id"]
    assert change_result["data"]["requires_human_approval"] is True
