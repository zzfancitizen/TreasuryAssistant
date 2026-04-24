from app.core.registry import AgentEndpoint, AgentRegistry, AgentSkill
from app.core.skill_registry import SkillDescriptor, SkillRegistry


def test_builds_skill_registry_from_agent_registry() -> None:
    registry = AgentRegistry(
        [
            AgentEndpoint(
                agent_id="cash_agent",
                name="CashAgent",
                url="http://localhost:8001",
                capabilities=("forecast_cashflow", "get_cash_balance"),
                skills=(
                    AgentSkill(
                        skill_id="forecast_cashflow",
                        name="Forecast cashflow",
                        description="Forecasts cash liquidity.",
                        tags=("cash",),
                    ),
                    AgentSkill(skill_id="get_cash_balance", name="Get cash balance"),
                ),
            )
        ]
    )

    skills = SkillRegistry.from_agent_registry(registry)

    skill = skills.get("forecast_cashflow")
    assert skill.skill_id == "forecast_cashflow"
    assert skill.provider_agent_id == "cash_agent"
    assert skill.name == "Forecast cashflow"
    assert skill.description == "Forecasts cash liquidity."
    assert skill.tags == ("cash",)


def test_skill_registry_supports_explicit_descriptors() -> None:
    skills = SkillRegistry(
        [
            SkillDescriptor(
                skill_id="check_counterparty_risk",
                provider_agent_id="risk_agent",
                name="Check counterparty risk",
                description="Checks risk.",
                tags=("risk",),
            )
        ]
    )

    assert skills.get("check_counterparty_risk").provider_agent_id == "risk_agent"
    assert skills.list()[0].tags == ("risk",)
