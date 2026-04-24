from app.assistant.context_compressor import ContextCompressor


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.messages.append(messages)
        return self.response


def build_large_context() -> dict:
    return {
        "task_id": "task-1",
        "user_goal": "分析资金情况",
        "current_step": {"skill_id": "analyze_liquidity_gap"},
        "skill": {"skill_id": "analyze_liquidity_gap"},
        "route_plan": {"steps": []},
        "dependency_results": {
            "forecast_cashflow": {
                "status": "completed",
                "summary": "未来两周预计缺口 500 万 CNY。" + "x" * 1000,
                "data": {
                    "liquidity_gap": 5_000_000,
                    "currency": "CNY",
                    "large_payload": "y" * 5000,
                },
                "human_action": None,
                "artifacts": [{"artifact_id": "artifact-1", "type": "json", "summary": "full artifact"}],
            }
        },
        "facts": [{"key": f"k{i}", "value": i, "source_skill_id": "forecast_cashflow"} for i in range(100)],
        "artifacts": [{"artifact_id": f"a{i}", "type": "json"} for i in range(100)],
        "context_budget": {"max_tokens": 120, "estimated_tokens": 0, "truncated": False},
    }


def test_structured_compression_preserves_facts_and_refs() -> None:
    compressor = ContextCompressor(max_context_tokens=120)

    compressed = compressor.compress_structured(build_large_context())

    assert compressed["context_budget"]["compression_strategy"] == "structured"
    assert compressed["context_budget"]["estimated_tokens"] <= 120
    result = compressed["dependency_results"]["forecast_cashflow"]
    assert result["facts"]["liquidity_gap"] == 5_000_000
    assert result["facts"]["currency"] == "CNY"
    assert result["artifact_refs"][0]["artifact_id"] == "artifact-1"


async def test_llm_compression_runs_after_structured_compression_when_needed() -> None:
    llm = FakeLLMClient("压缩摘要：未来两周预计缺口 500 万 CNY。")
    compressor = ContextCompressor(max_context_tokens=180, llm_client=llm)

    compressed = await compressor.compress(build_large_context())

    assert llm.messages
    assert compressed["context_budget"]["compression_strategy"] == "llm"
    assert compressed["dependency_results"]["forecast_cashflow"]["summary"].startswith("压缩摘要")
    assert compressed["dependency_results"]["forecast_cashflow"]["facts"]["liquidity_gap"] == 5_000_000
