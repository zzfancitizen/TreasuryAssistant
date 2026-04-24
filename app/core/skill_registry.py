from __future__ import annotations

from pydantic import BaseModel

from app.core.registry import AgentRegistry


class SkillDescriptor(BaseModel):
    skill_id: str
    provider_agent_id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()


class SkillRegistry:
    def __init__(self, skills: list[SkillDescriptor]) -> None:
        self._skills = {skill.skill_id: skill for skill in skills}

    @classmethod
    def from_agent_registry(cls, registry: AgentRegistry) -> "SkillRegistry":
        skills = [
            SkillDescriptor(
                skill_id=skill.skill_id,
                provider_agent_id=endpoint.agent_id,
                name=skill.name,
                description=skill.description,
                tags=skill.tags,
            )
            for endpoint in registry.list()
            for skill in endpoint.skills
        ]
        return cls(skills)

    def get(self, skill_id: str) -> SkillDescriptor:
        return self._skills[skill_id]

    def list(self) -> list[SkillDescriptor]:
        return list(self._skills.values())
