from unittest.mock import AsyncMock, patch

from app.core.llm_client import LiteLLMClient, is_litellm_configured


async def test_litellm_client_passes_openrouter_qwen_config(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_MODEL", "openrouter/qwen/qwen3.5-plus-20260420")
    monkeypatch.setenv("LITELLM_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    completion = AsyncMock()
    completion.return_value.choices = [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]

    with patch("app.core.llm_client.acompletion", completion):
        result = await LiteLLMClient().complete([{"role": "user", "content": "hello"}])

    assert result == "ok"
    completion.assert_awaited_once()
    assert completion.await_args.kwargs["model"] == "openrouter/qwen/qwen3.5-plus-20260420"
    assert completion.await_args.kwargs["api_base"] == "https://openrouter.ai/api/v1"
    assert completion.await_args.kwargs["api_key"] == "sk-or-test"


def test_litellm_requires_model_and_api_key(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_MODEL", "openrouter/qwen/qwen3.5-plus-20260420")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert is_litellm_configured() is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert is_litellm_configured() is True
