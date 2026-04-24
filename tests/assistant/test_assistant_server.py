from fastapi.testclient import TestClient

from app.assistant.server import app


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


def test_assistant_server_exposes_agent_card() -> None:
    response = TestClient(app).get("/.well-known/agent-card.json")

    assert response.status_code == 200
    assert response.json()["name"] == "TreasuryAssistant"


def test_assistant_server_exposes_a2a_jsonrpc_route() -> None:
    response = TestClient(app).post("/", json=message_send_payload("请综合分析当前财资状况"))

    assert response.status_code in {200, 500}
    assert response.json()["jsonrpc"] == "2.0"
