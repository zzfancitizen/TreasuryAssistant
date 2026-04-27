from __future__ import annotations

import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CompressionLLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


class ContextCompressor:
    def __init__(
        self,
        *,
        max_context_tokens: int = 200_000,
        chars_per_token: int = 4,
        llm_client: CompressionLLMClient | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.chars_per_token = chars_per_token
        self.llm_client = llm_client

    async def compress(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.estimate_tokens(context) <= self.max_context_tokens:
            return self._with_budget(context, strategy="none", truncated=False)

        structured = self.compress_structured(context)
        if self.estimate_tokens(structured) <= self.max_context_tokens or self.llm_client is None:
            return structured

        try:
            llm_compressed = await self._compress_with_llm(structured)
            if self.estimate_tokens(llm_compressed) <= self.max_context_tokens:
                logger.warning(
                    "context_compressor.llm_compressed",
                    extra={"estimated_tokens": self.estimate_tokens(llm_compressed), "max_tokens": self.max_context_tokens},
                )
                return llm_compressed
            return self._hard_trim(llm_compressed, strategy="llm")
        except Exception as exc:
            logger.error("context_compressor.llm_failed", extra={"error": str(exc)})

        return self._hard_trim(structured, strategy="structured_fallback")

    def compress_structured(self, context: dict[str, Any]) -> dict[str, Any]:
        compressed = dict(context)
        compressed["dependency_results"] = {
            skill_id: self._compress_result(result)
            for skill_id, result in context.get("dependency_results", {}).items()
        }
        compressed["facts"] = context.get("facts", [])[:50]
        compressed["artifacts"] = self._artifact_refs(context.get("artifacts", []))[:20]
        compressed = self._with_budget(compressed, strategy="structured", truncated=True)
        if self.estimate_tokens(compressed) <= self.max_context_tokens:
            return compressed
        return self._hard_trim(compressed, strategy="structured")

    def estimate_tokens(self, payload: dict[str, Any]) -> int:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return max(1, len(serialized) // self.chars_per_token)

    async def _compress_with_llm(self, context: dict[str, Any]) -> dict[str, Any]:
        compressed = dict(context)
        compressed_results: dict[str, dict[str, Any]] = {}
        for skill_id, result in context.get("dependency_results", {}).items():
            summary = await self._compress_result_summary_with_llm(skill_id, result)
            compressed_results[skill_id] = {
                **result,
                "summary": summary,
            }
        compressed["dependency_results"] = compressed_results
        compressed["facts"] = []
        compressed["artifacts"] = []
        return self._with_budget(compressed, strategy="llm", truncated=True)

    async def _compress_result_summary_with_llm(self, skill_id: str, result: dict[str, Any]) -> str:
        response = await self.llm_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Compress this subagent result for downstream treasury reasoning. "
                        "Preserve amounts, currencies, dates, approval state, risks, and artifact references. "
                        "Return a concise plain-text summary only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "skill_id": skill_id,
                            "summary": result.get("summary"),
                            "facts": result.get("facts"),
                            "human_action": result.get("human_action"),
                            "artifact_refs": result.get("artifact_refs"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return self._clip(response.strip(), 800)

    def _compress_result(self, result: dict[str, Any]) -> dict[str, Any]:
        data = result.get("data")
        return {
            "status": result.get("status"),
            "summary": self._clip(str(result.get("summary", "")), 500),
            "facts": self._extract_primitive_facts(data),
            "human_action": result.get("human_action"),
            "artifact_refs": self._artifact_refs(result.get("artifacts", []))[:10],
            "compression": {"strategy": "structured"},
        }

    def _extract_primitive_facts(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        facts: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                if len(value) <= 300:
                    facts[key] = value
            elif isinstance(value, int | float | bool) or value is None:
                facts[key] = value
        return facts

    def _artifact_refs(self, artifacts: Any) -> list[dict[str, Any]]:
        if not isinstance(artifacts, list):
            return []
        refs = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            refs.append(
                {
                    "artifact_id": artifact.get("artifact_id") or artifact.get("id"),
                    "type": artifact.get("type") or artifact.get("mime_type"),
                    "source_skill_id": artifact.get("source_skill_id"),
                    "summary": artifact.get("summary"),
                }
            )
        return refs

    def _hard_trim(self, context: dict[str, Any], *, strategy: str) -> dict[str, Any]:
        trimmed = dict(context)
        trimmed["facts"] = []
        trimmed["artifacts"] = []
        trimmed["dependency_results"] = {
            skill_id: {
                "status": result.get("status"),
                "summary": self._clip(str(result.get("summary", "")), 120),
                "facts": result.get("facts", {}),
                "human_action": result.get("human_action"),
                "artifact_refs": result.get("artifact_refs", [])[:3],
                "compression": {"strategy": strategy},
            }
            for skill_id, result in context.get("dependency_results", {}).items()
        }
        return self._with_budget(trimmed, strategy=strategy, truncated=True)

    def _with_budget(self, context: dict[str, Any], *, strategy: str, truncated: bool) -> dict[str, Any]:
        budget = dict(context.get("context_budget", {}))
        budget.update(
            {
                "max_tokens": self.max_context_tokens,
                "estimated_tokens": min(self.estimate_tokens(context), self.max_context_tokens),
                "truncated": truncated,
                "compression_strategy": strategy,
            }
        )
        updated = dict(context)
        updated["context_budget"] = budget
        return updated

    @staticmethod
    def _clip(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}...[truncated]"
