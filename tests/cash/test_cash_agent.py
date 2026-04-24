from app.cash.agent import CashAgent


async def test_cash_agent_returns_mock_cash_position() -> None:
    result = await CashAgent().invoke("查看未来两周现金情况")

    assert result["agent"] == "CashAgent"
    assert result["status"] == "completed"
    assert result["data"]["currency"] == "CNY"
    assert "available_balance" in result["data"]
    assert "cashflow_forecast" in result["data"]
