from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from app.assistant.continuation import ContinuationDecider
from app.assistant.context_builder import ContextBuilder
from app.assistant.context_compressor import CompressionLLMClient
from app.memory import ExecutionState, InMemoryMemoryService, MemoryService
from app.assistant.plan_validator import PlanValidator
from app.assistant.planner import AgentStep, RoutePlan
from app.assistant.result_normalizer import normalize_agent_result, result_with_human_action
from app.core.registry import AgentEndpoint, AgentRegistry
from app.core.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


class A2AInvoker(Protocol):
    async def invoke(
        self,
        endpoint: AgentEndpoint,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class PlanExecutor:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        a2a_client: A2AInvoker,
        continuation_decider: ContinuationDecider | None = None,
        memory_service: MemoryService | None = None,
        max_context_tokens: int = 200_000,
        compression_llm_client: CompressionLLMClient | None = None,
    ) -> None:
        self.registry = registry
        self.skill_registry = SkillRegistry.from_agent_registry(registry)
        self.plan_validator = PlanValidator(skill_registry=self.skill_registry)
        self.context_builder = ContextBuilder(
            max_context_tokens=max_context_tokens,
            compression_llm_client=compression_llm_client,
        )
        self.memory_service = memory_service or InMemoryMemoryService()
        self.a2a_client = a2a_client
        self.continuation_decider = continuation_decider or ContinuationDecider()

    async def execute(self, plan: RoutePlan, user_message: str) -> list[dict[str, Any]]:
        plan = self.plan_validator.normalize(plan)
        state = ExecutionState.create(user_goal=user_message, plan=plan)
        self.memory_service.save(state)
        logger.info(
            "plan_executor.execute.started",
            extra={
                "task_id": state.task_id,
                "intent": plan.intent,
                "execution_mode": plan.execution_mode,
                "step_count": len(plan.steps),
            },
        )
        has_dependencies = any(step.depends_on for step in plan.steps)
        if plan.execution_mode == "parallel" and not has_dependencies:
            return await self._execute_parallel(state)
        return await self._execute_sequential(state)

    async def _execute_parallel(self, state: ExecutionState) -> list[dict[str, Any]]:
        calls = [
            self._invoke_step(
                step,
                state=state,
            )
            for step in state.plan.steps
        ]
        results = await asyncio.gather(*calls)
        outputs: list[dict[str, Any]] = []
        for skill_id, result in results:
            normalized = normalize_agent_result(skill_id, result)
            output = result_with_human_action(normalized)
            state.record_step_result(skill_id, output)
            if normalized.human_action:
                state.set_pending_human_action(normalized.human_action)
            outputs.append(output)
        if not state.pending_human_action:
            state.complete()
        self.memory_service.save(state)
        logger.info(
            "plan_executor.execute.parallel_completed",
            extra={"task_id": state.task_id, "status": state.status, "result_count": len(outputs)},
        )
        return outputs

    async def _execute_sequential(self, state: ExecutionState) -> list[dict[str, Any]]:
        ordered_steps = order_steps_by_dependencies(state.plan.steps)
        results: list[dict[str, Any]] = []
        completed_iterations = 0

        while ordered_steps:
            step = ordered_steps.pop(0)
            skill_id, result = await self._invoke_step(
                step,
                state=state,
            )
            normalized = normalize_agent_result(skill_id, result)
            output = result_with_human_action(normalized)
            state.record_step_result(step.skill_id, output)
            results.append(output)
            completed_iterations += 1
            self.memory_service.save(state)

            if normalized.human_action:
                state.set_pending_human_action(normalized.human_action)
                self.memory_service.save(state)
                logger.info(
                    "plan_executor.execute.awaiting_human",
                    extra={
                        "task_id": state.task_id,
                        "action_type": normalized.human_action.action_type,
                        "source_skill_id": normalized.human_action.source_skill_id,
                    },
                )
                break

            decision = self.continuation_decider.decide(
                plan=state.plan,
                completed_step=step,
                result=output,
                previous_results=state.step_results,
                queued_steps=ordered_steps,
                iteration=completed_iterations,
            )
            if decision.action == "finish":
                break
            if decision.action == "insert_step" and decision.step is not None:
                ordered_steps.insert(0, decision.step)
        if not state.pending_human_action:
            state.complete()
            self.memory_service.save(state)
        logger.info(
            "plan_executor.execute.sequential_completed",
            extra={"task_id": state.task_id, "status": state.status, "result_count": len(results)},
        )
        return results

    async def _invoke_step(
        self,
        step: AgentStep,
        *,
        state: ExecutionState,
    ) -> tuple[str, dict[str, Any]]:
        state.start_step(step)
        skill = self.skill_registry.get(step.skill_id)
        endpoint = self.registry.get(skill.provider_agent_id)
        context = await self.context_builder.build_async(state=state, step=step, skill=skill)
        logger.info(
            "plan_executor.step.invoke",
            extra={
                "task_id": state.task_id,
                "skill_id": step.skill_id,
                "provider_agent_id": skill.provider_agent_id,
                "context_estimated_tokens": context["context_budget"]["estimated_tokens"],
                "context_truncated": context["context_budget"]["truncated"],
            },
        )
        return step.skill_id, await self.a2a_client.invoke(endpoint, step.task, context=context)


def order_steps_by_dependencies(steps: list[AgentStep]) -> list[AgentStep]:
    remaining = list(steps)
    ordered: list[AgentStep] = []
    completed: set[str] = set()

    while remaining:
        ready = [
            step
            for step in remaining
            if not step.depends_on or all(dependency in completed for dependency in step.depends_on)
        ]
        if not ready:
            raise ValueError("RoutePlan contains unresolved or circular dependencies.")

        for step in ready:
            ordered.append(step)
            completed.add(step.skill_id)
            remaining.remove(step)

    return ordered
