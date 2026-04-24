from fastapi.testclient import TestClient

from app.main import app


def test_main_publishes_treasury_assistant_card() -> None:
    response = TestClient(app).get("/.well-known/agent-card.json")

    assert response.status_code == 200
    assert response.json()["name"] == "TreasuryAssistant"
