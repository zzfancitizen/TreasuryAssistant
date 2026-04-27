from unittest.mock import patch

from fastapi.testclient import TestClient

from app.assistant.server import app
from app.cash.server import agent_card as cash_agent_card
from app.treasury.server import agent_card as treasury_agent_card


def reset_assistant_agent() -> None:
    for route in app.routes:
        if getattr(route, "path", None) == "/":
            route.endpoint.__self__.handler.request_handler.agent_executor._agent = None
            return


def message_send_payload(message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "test-request",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": "test-message",
                "parts": [{"kind": "text", "text": message}],
            }
        },
    }


def message_stream_payload(message: str) -> dict:
    payload = message_send_payload(message)
    payload["method"] = "message/stream"
    return payload


def test_assistant_server_exposes_agent_card() -> None:
    response = TestClient(app).get("/.well-known/agent-card.json")

    assert response.status_code == 200
    assert response.json()["name"] == "TreasuryAssistant"


def test_assistant_server_exposes_a2a_jsonrpc_route() -> None:
    response = TestClient(app).post("/", json=message_send_payload("查询公司银行账户余额"))

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["result"]["status"]["state"] == "completed"


def test_assistant_server_streams_sse_thinking_events() -> None:
    reset_assistant_agent()

    def fetch_card(url: str) -> dict:
        if url == "http://localhost:8001":
            return cash_agent_card.model_dump()
        if url == "http://localhost:8002":
            return treasury_agent_card.model_dump()
        raise AssertionError(f"Unexpected agent card URL: {url}")

    with patch("app.core.registry.fetch_agent_card", side_effect=fetch_card):
        with TestClient(app) as client:
            with client.stream("POST", "/", json=message_stream_payload("查询公司银行账户余额")) as response:
                lines = list(response.iter_lines())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert any("status-update" in line for line in lines)
    assert any("正在识别用户意图" in line for line in lines)
    assert any("step_started" in line for line in lines)
    assert any("step_completed" in line for line in lines)
    assert any("artifact-update" in line for line in lines)
