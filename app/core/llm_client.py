from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()


def is_litellm_configured() -> bool:
    return bool(
        os.getenv("LITELLM_MODEL")
        and (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("LITELLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    )


class LiteLLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("LITELLM_MODEL", "mock/mock-model")
        self.api_base = os.getenv("LITELLM_API_BASE")
        self.api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("LITELLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        response = await acompletion(
            **completion_kwargs,
        )
        return response.choices[0].message.content or ""
