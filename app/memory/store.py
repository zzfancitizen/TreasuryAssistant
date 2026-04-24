from __future__ import annotations

from typing import Protocol

from app.memory.state import ExecutionState


class MemoryService(Protocol):
    def save(self, state: ExecutionState) -> None:
        raise NotImplementedError

    def get(self, task_id: str) -> ExecutionState | None:
        raise NotImplementedError

    def delete(self, task_id: str) -> None:
        raise NotImplementedError


class InMemoryMemoryService:
    def __init__(self) -> None:
        self._states: dict[str, ExecutionState] = {}

    def save(self, state: ExecutionState) -> None:
        self._states[state.task_id] = state

    def get(self, task_id: str) -> ExecutionState | None:
        return self._states.get(task_id)

    def delete(self, task_id: str) -> None:
        self._states.pop(task_id, None)


MemoryStore = MemoryService
InMemoryMemoryStore = InMemoryMemoryService
