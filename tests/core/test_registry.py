from pathlib import Path
from unittest.mock import patch

from app.core.registry import AgentRegistry, build_endpoint_from_agent_card


def test_loads_registry_seed_from_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "agents.yaml"
    config_path.write_text(
        """
agents:
  - agent_id: risk_agent
    url: http://localhost:8003
""",
        encoding="utf-8",
    )

    registry = AgentRegistry.from_yaml(config_path)

    endpoint = registry.get("risk_agent")
    assert endpoint.name == "risk_agent"
    assert endpoint.url == "http://localhost:8003"
    assert endpoint.capabilities == ()


def test_builds_endpoint_capabilities_from_agent_card() -> None:
    endpoint = build_endpoint_from_agent_card(
        agent_id="risk_agent",
        seed_url="http://localhost:8003",
        card={
            "name": "RiskAgent",
            "description": "Handles risk checks.",
            "url": "http://localhost:8003",
            "skills": [
                {"id": "check_counterparty_risk", "name": "Check counterparty risk"},
                {"id": "assess_transaction_risk", "name": "Assess transaction risk"},
            ],
        },
    )

    assert endpoint.name == "RiskAgent"
    assert endpoint.description == "Handles risk checks."
    assert endpoint.capabilities == ("check_counterparty_risk", "assess_transaction_risk")
    assert endpoint.skills[0].skill_id == "check_counterparty_risk"
    assert endpoint.skills[0].name == "Check counterparty risk"


def test_discovers_registry_from_agent_cards(tmp_path: Path) -> None:
    config_path = tmp_path / "agents.yaml"
    config_path.write_text(
        """
agents:
  - agent_id: risk_agent
    url: http://localhost:8003
""",
        encoding="utf-8",
    )

    def fetch_card(url: str) -> dict:
        assert url == "http://localhost:8003"
        return {
            "name": "RiskAgent",
            "description": "Handles risk checks.",
            "url": "http://localhost:8003",
            "skills": [{"id": "check_counterparty_risk", "name": "Check counterparty risk"}],
        }

    registry = AgentRegistry.from_yaml_with_discovery(config_path, card_fetcher=fetch_card)

    endpoint = registry.get("risk_agent")
    assert endpoint.name == "RiskAgent"
    assert endpoint.capabilities == ("check_counterparty_risk",)
    assert endpoint.skills[0].name == "Check counterparty risk"


def test_default_local_prefers_yaml_when_present(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "agents.yaml"
    config_path.write_text(
        """
agents:
  - agent_id: report_agent
    url: http://localhost:8004
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_REGISTRY_PATH", str(config_path))

    registry = AgentRegistry.default_local()

    assert [endpoint.agent_id for endpoint in registry.list()] == ["report_agent"]


def test_default_local_falls_back_to_builtin_agents(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_REGISTRY_PATH", raising=False)

    registry = AgentRegistry.default_local(config_path="missing-agents.yaml")

    assert [endpoint.agent_id for endpoint in registry.list()] == ["cash_agent", "treasury_agent"]


def test_explicit_agent_card_discovery_failure_logs_error(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "agents.yaml"
    config_path.write_text(
        """
agents:
  - agent_id: report_agent
    url: http://localhost:8004
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_REGISTRY_PATH", str(config_path))

    with patch("app.core.registry.logger.error") as log_error:
        registry = AgentRegistry.default_local()

    assert [endpoint.agent_id for endpoint in registry.list()] == ["report_agent"]
    log_error.assert_called_once()
    assert log_error.call_args.args == ("agent_registry.discovery.failed",)
