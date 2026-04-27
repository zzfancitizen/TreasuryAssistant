from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import httpx
from pydantic import BaseModel
import yaml

logger = logging.getLogger(__name__)


class AgentSkill(BaseModel):
    skill_id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()


class AgentEndpoint(BaseModel):
    agent_id: str
    name: str
    url: str
    capabilities: tuple[str, ...]
    description: str = ""
    skills: tuple[AgentSkill, ...] = ()

    def model_post_init(self, __context: Any) -> None:
        if not self.skills and self.capabilities:
            self.skills = tuple(AgentSkill(skill_id=capability, name=capability) for capability in self.capabilities)


class AgentRegistry:
    def __init__(self, endpoints: list[AgentEndpoint]) -> None:
        self._endpoints = {endpoint.agent_id: endpoint for endpoint in endpoints}

    @classmethod
    def default_local(cls, config_path: str | Path | None = None) -> "AgentRegistry":
        explicit_path = config_path is not None or os.getenv("AGENT_REGISTRY_PATH") is not None
        path = Path(config_path or os.getenv("AGENT_REGISTRY_PATH", "app/config/agents.yaml"))
        logger.info("agent_registry.load.started", extra={"config_path": str(path), "explicit_path": explicit_path})
        if path.exists():
            try:
                registry = cls.from_yaml_with_discovery(path)
                logger.info(
                    "agent_registry.load.discovered",
                    extra={"config_path": str(path), "agent_count": len(registry.list())},
                )
                return registry
            except Exception:
                log = logger.error if explicit_path else logger.warning
                log(
                    "agent_registry.discovery.failed",
                    extra={"config_path": str(path), "explicit_path": explicit_path},
                )
                return cls.from_yaml(path)
        logger.warning("agent_registry.config.missing", extra={"config_path": str(path)})
        return cls.default_builtin()

    @classmethod
    def default_builtin(cls) -> "AgentRegistry":
        logger.info("agent_registry.default_builtin.loaded")
        return cls(
            [
                AgentEndpoint(
                    agent_id="cash_agent",
                    name="CashAgent",
                    url="http://localhost:8001",
	                capabilities=(
	                    "get_cash_balance",
	                    "get_bank_accounts",
	                    "get_cash_transactions",
	                    "forecast_cashflow",
	                    "read_cash_state",
	                    "change_cash_state",
	                ),
                ),
                AgentEndpoint(
                    agent_id="treasury_agent",
                    name="TreasuryAgent",
                    url="http://localhost:8002",
	                capabilities=(
	                    "analyze_liquidity_gap",
	                    "recommend_funding_plan",
	                    "recommend_cash_transfer",
	                    "check_treasury_risk",
	                    "read_treasury_state",
	                    "change_treasury_state",
	                ),
                ),
            ]
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentRegistry":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        agents = payload.get("agents", [])
        endpoints = [
            build_endpoint_from_config(agent)
            for agent in agents
        ]
        return cls(endpoints)

    @classmethod
    def from_yaml_with_discovery(
        cls,
        path: str | Path,
        *,
        card_fetcher: Callable[[str], dict[str, Any]] | None = None,
    ) -> "AgentRegistry":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        agents = payload.get("agents", [])
        fetcher = card_fetcher or fetch_agent_card
        endpoints = [
            build_endpoint_from_agent_card(
                agent_id=agent["agent_id"],
                seed_url=agent["url"],
                card=fetcher(agent["url"]),
            )
            for agent in agents
        ]
        return cls(endpoints)

    def get(self, agent_id: str) -> AgentEndpoint:
        return self._endpoints[agent_id]

    def list(self) -> list[AgentEndpoint]:
        return list(self._endpoints.values())


def build_endpoint_from_config(agent: dict[str, Any]) -> AgentEndpoint:
    skills = tuple(
        AgentSkill(
            skill_id=skill["id"],
            name=skill.get("name", skill["id"]),
            description=skill.get("description", ""),
            tags=tuple(skill.get("tags", [])),
        )
        for skill in agent.get("skills", [])
        if skill.get("id")
    )
    capabilities = tuple(agent.get("capabilities", [])) or tuple(skill.skill_id for skill in skills)
    return AgentEndpoint(
        agent_id=agent["agent_id"],
        name=agent.get("name", agent["agent_id"]),
        url=agent["url"],
        capabilities=capabilities,
        description=agent.get("description", ""),
        skills=skills,
    )


def fetch_agent_card(url: str) -> dict[str, Any]:
    card_url = f"{url.rstrip('/')}/.well-known/agent-card.json"
    logger.info("agent_card.fetch.started", extra={"card_url": card_url})
    response = httpx.get(card_url, timeout=5.0)
    response.raise_for_status()
    logger.info("agent_card.fetch.completed", extra={"card_url": card_url, "status_code": response.status_code})
    return response.json()


def build_endpoint_from_agent_card(
    *,
    agent_id: str,
    seed_url: str,
    card: dict[str, Any],
) -> AgentEndpoint:
    skills = tuple(
        AgentSkill(
            skill_id=skill["id"],
            name=skill.get("name", skill["id"]),
            description=skill.get("description", ""),
            tags=tuple(skill.get("tags", [])),
        )
        for skill in card.get("skills", [])
        if skill.get("id")
    )
    capabilities = tuple(skill.skill_id for skill in skills)
    return AgentEndpoint(
        agent_id=agent_id,
        name=card.get("name", agent_id),
        url=card.get("url", seed_url),
        capabilities=capabilities,
        description=card.get("description", ""),
        skills=skills,
    )
