from fastapi.testclient import TestClient

from app.treasury.server import app


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


def test_treasury_server_exposes_agent_card() -> None:
    response = TestClient(app).get("/.well-known/agent-card.json")

    assert response.status_code == 200
    assert response.json()["name"] == "TreasuryAgent"
    skill_ids = {skill["id"] for skill in response.json()["skills"]}
    assert {"read_treasury_state", "change_treasury_state"}.issubset(skill_ids)


def test_treasury_server_accepts_a2a_message_send() -> None:
    response = TestClient(app).post("/", json=message_send_payload("给调拨建议"))

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["result"]["status"]["state"] == "completed"
    assert body["result"]["artifacts"][0]["parts"][0]["text"]


def test_treasury_server_accepts_a2a_message_stream() -> None:
    with TestClient(app) as client:
        with client.stream("POST", "/", json=message_stream_payload("给调拨建议")) as response:
            lines = list(response.iter_lines())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert any("status-update" in line for line in lines)
    assert any("artifact-update" in line for line in lines)
