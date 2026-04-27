from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Protocol

from app.assistant.continuation import ContinuationDecider
from app.assistant.context_builder import ContextBuilder
from app.assistant.context_compressor import CompressionLLMClient
from app.memory import ExecutionState, InMemoryMemoryService, MemoryService
from app.assistant.plan_validator import PlanValidator
from app.assistant.planner import AgentStep, RoutePlan
from app.assistant.result_normalizer import normalize_agent_result, result_with_human_action
from app.assistant.turn_classifier import classify_user_turn
from app.core.registry import AgentEndpoint, AgentRegistry
from app.core.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


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

    async def execute(
        self,
        plan: RoutePlan,
        user_message: str,
        *,
        task_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        plan = self.plan_validator.normalize(plan)
        state = ExecutionState.create(user_goal=user_message, plan=plan, task_id=task_id)
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
            return await self._execute_parallel(state, progress_callback=progress_callback)
        return await self._execute_sequential(state, progress_callback=progress_callback)

    async def resume_pending_human_action(self, task_id: str, user_message: str) -> list[dict[str, Any]]:
        state = self.memory_service.get(task_id)
        if state is None or state.pending_human_action is None:
            return [{"status": "failed", "summary": "No pending human action exists for this context."}]

        turn = classify_user_turn(user_message, state)
        if turn.decision == "reject":
            state.complete()
            self.memory_service.save(state)
            return [{"status": "completed", "summary": "Pending action was rejected by the user."}]
        if turn.decision != "approve":
            return [
                {
                    "status": state.status,
                    "summary": state.pending_human_action.question,
                    "human_action": state.pending_human_action.model_dump(),
                }
            ]

        state.pending_human_action = None
        state.status = "executing"
        ordered_steps = [
            step
            for step in order_steps_by_dependencies(state.plan.steps)
            if step.skill_id not in state.step_results
        ]
        if not ordered_steps:
            state.complete()
            self.memory_service.save(state)
            return []
        return await self._execute_sequential_steps(
            state,
            ordered_steps=ordered_steps,
            completed_iterations=len(state.step_results),
            progress_callback=None,
        )

    async def _execute_parallel(
        self,
        state: ExecutionState,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        for step in state.plan.steps:
            await self._emit_progress(
                progress_callback,
                state=state,
                event_type="step_started",
                step=step,
            )
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
            step = next((candidate for candidate in state.plan.steps if candidate.skill_id == skill_id), None)
            if step is not None:
                await self._emit_progress(
                    progress_callback,
                    state=state,
                    event_type="step_completed",
                    step=step,
                    result=output,
                )
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

    async def _execute_sequential(
        self,
        state: ExecutionState,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        return await self._execute_sequential_steps(
            state,
            ordered_steps=order_steps_by_dependencies(state.plan.steps),
            completed_iterations=0,
            progress_callback=progress_callback,
        )

    async def _execute_sequential_steps(
        self,
        state: ExecutionState,
        *,
        ordered_steps: list[AgentStep],
        completed_iterations: int,
        progress_callback: ProgressCallback | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        while ordered_steps:
            step = ordered_steps.pop(0)
            await self._emit_progress(
                progress_callback,
                state=state,
                event_type="step_started",
                step=step,
            )
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
            await self._emit_progress(
                progress_callback,
                state=state,
                event_type="step_completed",
                step=step,
                result=output,
            )

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
                await self._emit_progress(
                    progress_callback,
                    state=state,
                    event_type="step_inserted",
                    step=decision.step,
                    reason=decision.reason,
                )
        if not state.pending_human_action:
            state.complete()
            self.memory_service.save(state)
        logger.info(
            "plan_executor.execute.sequential_completed",
            extra={"task_id": state.task_id, "status": state.status, "result_count": len(results)},
        )
        return results

    async def _emit_progress(
        self,
        progress_callback: ProgressCallback | None,
        *,
        state: ExecutionState,
        event_type: str,
        step: AgentStep,
        result: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        if progress_callback is None:
            return
        skill = self.skill_registry.get(step.skill_id)
        payload: dict[str, Any] = {
            "event_type": event_type,
            "task_id": state.task_id,
            "execution_mode": state.plan.execution_mode,
            "intent": state.plan.intent,
            "skill_id": step.skill_id,
            "task": step.task,
            "depends_on": step.depends_on,
            "agent_id": skill.provider_agent_id,
            "agent_name": self.registry.get(skill.provider_agent_id).name,
        }
        if result is not None:
            payload["status"] = result.get("status", "completed")
            payload["summary"] = result.get("summary", "")
            if result.get("human_action"):
                payload["human_action"] = result["human_action"]
        if reason:
            payload["reason"] = reason
        await progress_callback(payload)

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
