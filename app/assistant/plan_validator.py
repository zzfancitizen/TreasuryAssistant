from __future__ import annotations

from typing import Any

from app.core.skill_registry import SkillRegistry


class PlanValidationError(ValueError):
    pass


class PlanValidator:
    def __init__(self, *, skill_registry: SkillRegistry) -> None:
        self.skill_registry = skill_registry

    def validate(self, plan: Any) -> None:
        steps = list(plan.steps)
        known_skills = {skill.skill_id for skill in self.skill_registry.list()}
        step_ids = [step.skill_id for step in steps]
        step_id_set = set(step_ids)

        unknown_skills = sorted(skill_id for skill_id in step_ids if skill_id not in known_skills)
        if unknown_skills:
            raise PlanValidationError(f"Unknown skill in RoutePlan: {', '.join(unknown_skills)}")

        duplicate_skills = sorted({skill_id for skill_id in step_ids if step_ids.count(skill_id) > 1})
        if duplicate_skills:
            raise PlanValidationError(f"Duplicate skill in RoutePlan: {', '.join(duplicate_skills)}")

        for step in steps:
            if step.skill_id in step.depends_on:
                raise PlanValidationError(f"Step cannot depend on itself: {step.skill_id}")
            missing_dependencies = sorted(
                dependency for dependency in step.depends_on if dependency not in step_id_set
            )
            if missing_dependencies:
                raise PlanValidationError(
                    f"Missing dependency for {step.skill_id}: {', '.join(missing_dependencies)}"
                )

        self._assert_acyclic(steps)

    def normalize(self, plan: Any) -> Any:
        execution_mode = self.infer_execution_mode(plan.steps)
        normalized = plan.model_copy(update={"execution_mode": execution_mode})
        self.validate(normalized)
        return normalized

    def supported_steps(self, plan: Any) -> list[Any]:
        known_skills = {skill.skill_id for skill in self.skill_registry.list()}
        return [step for step in plan.steps if step.skill_id in known_skills]

    @staticmethod
    def infer_execution_mode(steps: list[Any]) -> str:
        if len(steps) == 1:
            return "single"
        if any(step.depends_on for step in steps):
            return "sequential"
        return "parallel"

    @staticmethod
    def _assert_acyclic(steps: list[Any]) -> None:
        dependencies_by_skill = {step.skill_id: set(step.depends_on) for step in steps}
        temporary_marks: set[str] = set()
        permanent_marks: set[str] = set()

        def visit(skill_id: str) -> None:
            if skill_id in permanent_marks:
                return
            if skill_id in temporary_marks:
                raise PlanValidationError("Circular dependency in RoutePlan.")

            temporary_marks.add(skill_id)
            for dependency in dependencies_by_skill[skill_id]:
                visit(dependency)
            temporary_marks.remove(skill_id)
            permanent_marks.add(skill_id)

        for step in steps:
            visit(step.skill_id)
