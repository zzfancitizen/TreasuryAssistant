from app.cash.agent import CashAgent


async def test_cash_agent_returns_mock_cash_position() -> None:
    result = await CashAgent().invoke("查看未来两周现金情况")

    assert result["agent"] == "CashAgent"
    assert result["status"] == "completed"
    assert result["data"]["currency"] == "CNY"
    assert "available_balance" in result["data"]
    assert "cashflow_forecast" in result["data"]


async def test_cash_agent_returns_mock_gap_for_dynamic_orchestration_case() -> None:
    result = await CashAgent().invoke("动态编排：先读取现金流，如果发现资金缺口再追加融资计划")

    assert result["status"] == "completed"
    assert result["data"]["liquidity_gap"] == 5_000_000
    assert "liquidity gap" in result["summary"]


async def test_cash_agent_mocks_read_and_change_methods() -> None:
    read_result = await CashAgent().invoke(
        "读取现金状态",
        context={"current_step": {"skill_id": "read_cash_state"}},
    )
    change_result = await CashAgent().invoke(
        "调整现金池目标余额",
        context={"current_step": {"skill_id": "change_cash_state"}},
    )

    assert read_result["operation"] == "read"
    assert read_result["data"]["ledger_version"]
    assert change_result["operation"] == "change"
    assert change_result["data"]["change_id"]
    assert change_result["data"]["requires_human_approval"] is True
