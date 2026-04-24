# TreasuryAssistant

`TreasuryAssistant` is a super agent that exposes an A2A SDK JSON-RPC surface and orchestrates two mock A2A subagents:

- `CashAgent`: cash balances, bank accounts, transactions, and cashflow forecasts.
- `TreasuryAgent`: liquidity analysis, funding plans, transfer recommendations, and risk checks.

The MVP keeps all agent code under `app/` and tests under `tests/`, following the sample project structure. Python dependencies are managed with `uv`.

## Architecture

```text
User / Agent Client
        ↓
TreasuryAssistant A2A Server
        ↓
LangGraph Orchestrator
        ↓
A2A Client
   ┌───────────────┬─────────────────┐
   ↓               ↓
CashAgent       TreasuryAgent
```

`app/main.py` publishes the `TreasuryAssistant` A2A service. `app/agent_executor.py` adapts A2A requests to the Assistant. `app/assistant/agent.py` is the Assistant's main business entrypoint. LLM calls are centralized through `app/core/llm_client.py` using LiteLLM, so third-party model providers can be changed without touching agent orchestration logic.

Assistant prompts live in `app/assistant/prompts.py`:

- `ASSISTANT_SYSTEM_PROMPT`: overall super-agent role and safety rules.
- `build_router_system_prompt(...)`: route planning prompt using discovered agent card skills.
- `SYNTHESIZER_SYSTEM_PROMPT`: grounded answer synthesis prompt.
- `CONTINUATION_SYSTEM_PROMPT`: runtime replanning decision prompt.

Routing is handled by `app/assistant/planner.py`:

- High-confidence treasury keywords use a deterministic fast path.
- Requests without obvious keywords use a LiteLLM-backed structured route planner when `LITELLM_MODEL` is configured.
- Low-confidence or invalid LLM plans fall back to a conservative general plan that queries both subagents in parallel.

Plan validation is handled by `app/assistant/plan_validator.py`:

- Unknown skills, duplicate skills, missing dependencies, and circular dependencies are rejected.
- `execution_mode` is inferred from `RoutePlan.steps[].depends_on`, so a plan with dependencies is forced to sequential execution even if the LLM claimed `parallel`.
- The planner uses validation to normalize LLM output; the executor validates again before calling downstream A2A agents.

Plan execution is handled by `app/assistant/plan_executor.py`. The LangGraph
orchestrator uses a fixed lifecycle graph:

```text
plan_route -> execute_plan -> synthesize
```

Execution state, memory, context, and human-in-the-loop handling are explicit:

- `app/memory/state.py` stores task status, current step, step results, extracted facts, artifact refs, and pending human actions.
- `app/memory/store.py` provides the `MemoryService` interface; the current implementation is in-memory and can be replaced with a remote memory service, Redis, or Postgres later.
- `app/assistant/context_builder.py` builds minimal subagent context from the current step dependencies instead of passing full conversation history. It defaults to a 200k-token budget.
- `app/assistant/context_compressor.py` applies two-layer compression before invoking A2A agents: deterministic structured compression first, then optional LLM semantic compression when a compression LLM is configured and structured compression is still too large.
- `app/assistant/result_normalizer.py` standardizes subagent pauses into `await_input` and `await_confirm`.
- `app/assistant/turn_classifier.py` classifies user turns as pending-action answers, follow-ups, memory queries, corrections, or new tasks.

Subagent context uses dependency-scoped payloads:

```text
ExecutionState + current SkillStep + dependency results
        ↓
ContextBuilder
        ↓
budgeted A2A metadata.context
```

Human-in-the-loop is represented as a normal execution state, not an exception:

```json
{
  "status": "await_confirm",
  "human_action": {
    "action_type": "await_confirm",
    "action_id": "approval-1",
    "source_skill_id": "recommend_cash_transfer",
    "question": "是否确认继续？",
    "options": ["approve", "reject", "modify"]
  }
}
```

Adding a new subagent should only require registering its seed endpoint in
`app/config/agents.yaml`; the Assistant discovers its A2A Agent Card, builds a
skill registry, and routes by `skill_id`. The orchestrator graph does not need a
new agent-specific node.

Subagents are discovered from seed URLs in `app/config/agents.yaml` by default. The registry
fetches each subagent's A2A agent card from `/.well-known/agent-card.json` and
uses the card skills as planner skills. Set
`AGENT_REGISTRY_PATH=/path/to/agents.yaml` to use a different seed file. If a
card is temporarily unavailable, the registry falls back to the seed metadata so
local development can still start.

Execution also supports runtime correction. Static plans are treated as the
default path, not a dead-end path: after each sequential step,
`app/assistant/continuation.py` evaluates the observation and can insert a new
step. For example, a `forecast_cashflow` plan can automatically add a
`recommend_funding_plan` step when the cash skill reports a positive
`data.liquidity_gap`, bounded by
`RoutePlan.max_iterations`.

## Run Locally

```bash
uv run uvicorn app.cash.server:app --port 8001
uv run uvicorn app.treasury.server:app --port 8002
uv run uvicorn app.main:app --port 8000
```

To enable LLM-based routing for non-keyword requests:

```bash
export LITELLM_MODEL=your-provider/your-model
export LLM_API_KEY=your-provider-key
```

Agent cards are available at:

- `http://localhost:8000/.well-known/agent-card.json`
- `http://localhost:8001/.well-known/agent-card.json`
- `http://localhost:8002/.well-known/agent-card.json`

Tasks are submitted to each agent's JSON-RPC endpoint with A2A `message/send`:

```bash
curl -X POST http://localhost:8001/ \
  -H 'content-type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "demo",
    "method": "message/send",
    "params": {
      "message": {
        "kind": "message",
        "role": "user",
        "messageId": "demo-message",
        "parts": [{"kind": "text", "text": "查余额"}]
      }
    }
  }'
```

Streaming uses A2A `message/stream` against the same endpoint. `TreasuryAssistant`
emits step-level status updates before returning the final artifact:

```bash
curl -N -X POST http://localhost:8000/ \
  -H 'content-type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "demo-stream",
    "method": "message/stream",
    "params": {
      "message": {
        "kind": "message",
        "role": "user",
        "messageId": "demo-stream-message",
        "parts": [{"kind": "text", "text": "请综合分析当前财资状况"}]
      }
    }
  }'
```

## Verify

```bash
uv run pytest
uv run ruff check app tests
```
