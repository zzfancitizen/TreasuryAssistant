from __future__ import annotations

import os
from typing import Any

from litellm import acompletion


class LiteLLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("LITELLM_MODEL", "mock/mock-model")

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return response.choices[0].message.content or ""
