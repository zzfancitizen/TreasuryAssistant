from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.registry import AgentEndpoint


class A2AClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def invoke(
        self,
        endpoint: AgentEndpoint,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                endpoint.url,
                json=build_message_send_request(message, context=context),
            )
            response.raise_for_status()
            return parse_message_send_response(response.json())

    async def stream(
        self,
        endpoint: AgentEndpoint,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                endpoint.url,
                json=build_message_stream_request(message, context=context),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    event = parse_sse_data_line(line)
                    if event is not None:
                        yield event


def build_message_send_request(message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_message_request("message/send", message, context=context)


def build_message_stream_request(message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_message_request("message/stream", message, context=context)


def build_message_request(
    method: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": {
            "metadata": {"context": context or {}},
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": message}],
            },
        },
    }


def parse_message_send_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "error" in payload:
        raise RuntimeError(payload["error"])

    result = payload["result"]
    artifacts = result.get("artifacts") or []
    if not artifacts:
        return result

    parts = artifacts[0].get("parts") or []
    if not parts:
        return result

    text = parts[0].get("text")
    if not text:
        return result
    return json.loads(text)


def parse_sse_data_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    return json.loads(line.removeprefix("data: "))
