import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_AGENT_API_BASE || "/api";

const plannerExecutionCases = [
  {
    id: "parallel-analysis",
    label: "并发 Case",
    mode: "parallel",
    title: "综合财资并发分析",
    prompt: "并发编排：请综合分析当前财资状况",
    description: "无特定单一 skill 关键词，planner fallback 到 CashAgent 与 TreasuryAgent 并发。",
  },
  {
    id: "sequential-liquidity",
    label: "顺序 Case",
    mode: "sequential",
    title: "资金缺口顺序分析",
    prompt: "分析未来两周资金缺口并给调拨建议",
    description: "先 forecast_cashflow，再基于依赖结果执行 analyze_liquidity_gap。",
  },
  {
    id: "initial-read",
    label: "初始编排 Case",
    mode: "single",
    title: "启动时确定 Cash Read",
    prompt: "读取 cash runtime state",
    description: "planner 一开始就确定 read_cash_state，不等待运行时追加步骤。",
  },
  {
    id: "human-confirm-change",
    label: "确认 Case",
    mode: "single",
    title: "Treasury Change 确认",
    prompt: "修改融资计划",
    description: "触发 change_treasury_state，mock sub-agent 返回 await_confirm。",
  },
  {
    id: "dynamic-replan",
    label: "动态 Case",
    mode: "dynamic",
    title: "运行时追加融资步骤",
    prompt: "动态编排：先读取现金流，如果发现资金缺口再追加融资计划",
    description: "初始只派发 CashAgent；发现 liquidity_gap 后由 executor 动态追加 TreasuryAgent。",
  },
];

function createEmptyWorkflow(plannerCase = null) {
  return {
    intent: "waiting",
    mode: plannerCase?.mode || "idle",
    phase: "idle",
    case: plannerCase
      ? {
          id: plannerCase.id,
          title: plannerCase.title,
          mode: plannerCase.mode,
          description: plannerCase.description,
        }
      : null,
    events: [],
    steps: [],
    summary: "",
    humanAction: null,
  };
}

function buildStreamPayload(message) {
  return {
    jsonrpc: "2.0",
    id: crypto.randomUUID(),
    method: "message/stream",
    params: {
      message: {
        kind: "message",
        role: "user",
        messageId: crypto.randomUUID(),
        parts: [{ kind: "text", text: message }],
      },
    },
  };
}

async function streamAgent(message, handlers) {
  const response = await fetch(`${API_BASE}/`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(buildStreamPayload(message)),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Agent stream failed with HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (!raw || raw === "[DONE]") continue;
      try {
        handlers.onEvent(JSON.parse(raw));
      } catch {
        handlers.onText(raw);
      }
    }
  }
}

function extractTexts(value, texts = []) {
  if (value == null) return texts;
  if (typeof value === "string") {
    texts.push(value);
    return texts;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => extractTexts(item, texts));
    return texts;
  }
  if (typeof value === "object") {
    if (typeof value.text === "string") texts.push(value.text);
    Object.values(value).forEach((item) => extractTexts(item, texts));
  }
  return texts;
}

function parseFinalPayload(event) {
  const texts = extractTexts(event);
  for (const text of texts) {
    const trimmed = text.trim();
    if (!trimmed.startsWith("{")) continue;
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed.agent === "TreasuryAssistant" || parsed.agent_results) return parsed;
    } catch {
      // Ignore non-payload text.
    }
  }
  return null;
}

function readStatusText(event) {
  const texts = extractTexts(event);
  return (
    texts.find(
      (text) =>
        text.includes("正在") ||
        text.includes("已生成") ||
        text.includes("等待") ||
        text.startsWith("step_"),
    ) || ""
  );
}

function eventKind(event) {
  return (
    event?.result?.kind ||
    event?.result?.status?.state ||
    event?.kind ||
    event?.method ||
    "sse-event"
  );
}

function thinkingStepFromSse(raw, event = null) {
  const text = event ? readStatusText(event) || extractTexts(event)[0] || "" : raw;
  return {
    id: crypto.randomUUID(),
    time: new Date(),
    kind: event ? eventKind(event) : "raw",
    text: text || "SSE event received",
    raw,
  };
}

function updateWorkflowFromStatus(workflow, text) {
  const next = {
    ...workflow,
    events: text ? [...workflow.events, { id: crypto.randomUUID(), text, time: new Date() }] : workflow.events,
  };
  const stepEvent = parseStepEvent(text);
  if (stepEvent) {
    next.phase = "executing";
    next.mode = next.mode === "idle" ? "single" : next.mode;
    next.steps = upsertWorkflowStep(next.steps, stepEvent);
    return next;
  }
  const planMatch = text.match(/已生成执行计划:\s*([^(]+)\(([^)]+)\)/);
  if (planMatch) {
    next.intent = planMatch[1].trim();
    const plannedMode = planMatch[2].trim();
    next.mode = workflow.case?.mode === "dynamic" && next.intent === "dynamic_liquidity_check" ? "dynamic" : plannedMode;
    next.phase = "planned";
  } else if (text.includes("并发")) {
    next.mode = next.mode === "idle" ? "parallel" : next.mode;
    next.phase = "executing";
  } else if (text.includes("依赖顺序")) {
    next.mode = "sequential";
    next.phase = "executing";
  } else if (text.includes("调用 skill")) {
    next.mode = next.mode === "idle" ? "single" : next.mode;
    next.phase = "executing";
  } else if (text.includes("汇总")) {
    next.phase = "synthesizing";
  }
  return next;
}

function parseStepEvent(text) {
  const match = text.match(/^(step_started|step_completed|step_inserted):\s*([^\s]+)\s*->\s*([^;(]+)(?:\s*\(([^)]+)\))?/);
  if (!match) return null;
  const [, eventType, skillId, agentName, status] = match;
  if (eventType === "step_inserted") {
    return {
      id: skillId,
      skillId,
      agent: agentName.trim(),
      status: "queued",
      summary: "Runtime replanning inserted this step.",
    };
  }
  return {
    id: skillId,
    skillId,
    agent: agentName.trim(),
    status: eventType === "step_started" ? "working" : status || "completed",
    summary: eventType === "step_started" ? `Executing ${skillId}` : `Completed ${skillId}`,
  };
}

function upsertWorkflowStep(steps, stepEvent) {
  const existingIndex = steps.findIndex((step) => step.id === stepEvent.id || step.skillId === stepEvent.skillId);
  if (existingIndex === -1) {
    return [
      ...steps,
      {
        ...stepEvent,
        index: steps.length + 1,
      },
    ];
  }
  return steps.map((step, index) =>
    index === existingIndex
      ? {
          ...step,
          ...stepEvent,
          index: step.index || index + 1,
          data: step.data,
        }
      : step,
  );
}

function updateWorkflowFromPayload(workflow, payload) {
  const steps = (payload.agent_results || []).map((result, index) => ({
    id: `${result.agent || result.status || "agent"}-${index}`,
    index: index + 1,
    agent: result.agent || `Agent ${index + 1}`,
    status: result.status || "completed",
    summary: result.summary || "Step completed",
    data: result.data,
  }));

  return {
    ...workflow,
    intent: payload.intent || workflow.intent,
    phase: payload.status === "await_confirm" || payload.status === "await_input" ? "waiting" : "completed",
    steps,
    summary: payload.summary || workflow.summary,
    humanAction: payload.human_action || null,
  };
}

function modeLabel(mode) {
  if (mode === "parallel") return "并发执行";
  if (mode === "sequential") return "顺序执行";
  if (mode === "single") return "单步执行";
  if (mode === "dynamic") return "动态编排";
  if (mode === "idle") return "等待指令";
  return mode;
}

function App() {
  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "连接到 TreasuryAssistant 后，可以在这里发起财资分析、余额查询或调拨建议。",
    },
  ]);
  const [workflow, setWorkflow] = useState(createEmptyWorkflow);
  const [thinkingSteps, setThinkingSteps] = useState([]);
  const [draft, setDraft] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const canSend = draft.trim().length > 0 && !isStreaming;

  async function submitMessage(text = draft, plannerCase = null) {
    const message = text.trim();
    if (!message || isStreaming) return;

    setDraft("");
    setError("");
    setIsStreaming(true);
    setWorkflow(createEmptyWorkflow(plannerCase));
    setThinkingSteps([]);
    setMessages((items) => [
      ...items,
      { id: crypto.randomUUID(), role: "user", text: message },
      { id: crypto.randomUUID(), role: "assistant", text: "正在处理请求..." },
    ]);

    let finalPayload = null;
    try {
      await streamAgent(message, {
        onText: (textChunk) => {
          setThinkingSteps((items) => [...items, thinkingStepFromSse(textChunk)]);
          setWorkflow((current) => updateWorkflowFromStatus(current, textChunk));
        },
        onEvent: (event) => {
          setThinkingSteps((items) => [...items, thinkingStepFromSse(JSON.stringify(event), event)]);
          const statusText = readStatusText(event);
          if (statusText) {
            setWorkflow((current) => updateWorkflowFromStatus(current, statusText));
          }
          const payload = parseFinalPayload(event);
          if (payload) {
            finalPayload = payload;
            setWorkflow((current) => updateWorkflowFromPayload(current, payload));
          }
        },
      });

      const answer = finalPayload?.summary || "请求已完成，但没有返回可展示的摘要。";
      setMessages((items) =>
        items.map((item, index) =>
          index === items.length - 1 && item.role === "assistant" ? { ...item, text: answer } : item,
        ),
      );
    } catch (streamError) {
      const textError = streamError instanceof Error ? streamError.message : "Unknown stream error";
      setError(textError);
      setWorkflow((current) => ({ ...current, phase: "failed", summary: textError }));
      setMessages((items) =>
        items.map((item, index) =>
          index === items.length - 1 && item.role === "assistant"
            ? { ...item, text: `请求失败：${textError}` }
            : item,
        ),
      );
    } finally {
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-pane" aria-label="Agent chat">
        <header className="pane-header">
          <div>
            <p className="eyebrow">TreasuryAssistant</p>
            <h1>Agent Command Console</h1>
          </div>
          <span className={isStreaming ? "status-pill active" : "status-pill"}>{isStreaming ? "Streaming" : "Ready"}</span>
        </header>

        <div className="message-list">
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <span className="message-role">{message.role === "user" ? "You" : "Agent"}</span>
              <p>{message.text}</p>
            </article>
          ))}
        </div>

        <ThinkingSteps steps={thinkingSteps} isStreaming={isStreaming} />

        <div className="quick-actions" aria-label="Planner execution cases">
          {plannerExecutionCases.map((plannerCase) => (
            <button
              key={plannerCase.id}
              type="button"
              onClick={() => submitMessage(plannerCase.prompt, plannerCase)}
              disabled={isStreaming}
            >
              <span>{plannerCase.label}</span>
              <strong>{plannerCase.title}</strong>
              <small>{plannerCase.description}</small>
            </button>
          ))}
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            submitMessage();
          }}
        >
          <textarea
            ref={inputRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="输入给 agent 的财资指令..."
            rows={3}
          />
          <button type="submit" disabled={!canSend} aria-label="Send command">
            <span aria-hidden="true">↗</span>
            Send
          </button>
        </form>

        {error ? <p className="error-line">{error}</p> : null}
      </section>

      <WorkflowPane workflow={workflow} isStreaming={isStreaming} />
    </main>
  );
}

function ThinkingSteps({ steps, isStreaming }) {
  if (!steps.length && !isStreaming) return null;
  return (
    <section className="thinking-panel" aria-label="SSE thinking steps">
      <div className="thinking-heading">
        <h3>Thinking Steps</h3>
        <span>{isStreaming ? "live" : `${steps.length} events`}</span>
      </div>
      <div className="thinking-list">
        {steps.length ? (
          steps.map((step, index) => (
            <details key={step.id} className="thinking-step" open={index === steps.length - 1}>
              <summary>
                <span>{step.time.toLocaleTimeString()}</span>
                <strong>{step.kind}</strong>
                <em>{step.text}</em>
              </summary>
              <pre>{step.raw}</pre>
            </details>
          ))
        ) : (
          <p>等待 SSE 事件...</p>
        )}
      </div>
    </section>
  );
}

function WorkflowPane({ workflow, isStreaming }) {
  const computedSteps = useMemo(() => {
    if (workflow.steps.length) return workflow.steps;
    if (workflow.phase === "idle") return [];
    if (workflow.phase === "completed") {
      return [
        { id: "plan", index: 1, agent: "Planner", status: "completed", summary: "Route intent and execution mode" },
        {
          id: "execute",
          index: 2,
          agent: "Executor",
          status: workflow.intent === "unavailable" ? "skipped" : "completed",
          summary: workflow.intent === "unavailable" ? "No discoverable A2A skills to dispatch" : "Dispatch matching A2A skills",
        },
        { id: "synthesize", index: 3, agent: "Synthesizer", status: "completed", summary: "Merge agent results" },
      ];
    }
    return [
      { id: "plan", index: 1, agent: "Planner", status: workflow.phase === "planned" ? "completed" : "working", summary: "Route intent and execution mode" },
      { id: "execute", index: 2, agent: "Executor", status: workflow.phase === "executing" ? "working" : "queued", summary: "Dispatch matching A2A skills" },
      { id: "synthesize", index: 3, agent: "Synthesizer", status: workflow.phase === "synthesizing" ? "working" : "queued", summary: "Merge agent results" },
    ];
  }, [workflow.phase, workflow.steps]);

  return (
    <section className="workflow-pane" aria-label="Planner execution workflow">
      <header className="pane-header">
        <div>
          <p className="eyebrow">Planner / Executor</p>
          <h2>Workflow Pipeline</h2>
        </div>
        <span className={`mode-badge mode-${workflow.mode}`}>{modeLabel(workflow.mode)}</span>
      </header>

      <div className="workflow-summary">
        {workflow.case ? (
          <div>
            <span>Case</span>
            <strong>{workflow.case.title}</strong>
            <small>{workflow.case.description}</small>
          </div>
        ) : null}
        <div>
          <span>Intent</span>
          <strong>{workflow.intent}</strong>
        </div>
        <div>
          <span>Phase</span>
          <strong>{workflow.phase}</strong>
        </div>
      </div>

      <div className={`pipeline pipeline-${workflow.mode}`}>
        {computedSteps.map((step) => (
          <PipelineStep key={step.id} step={step} mode={workflow.mode} isStreaming={isStreaming} />
        ))}
        {!computedSteps.length ? (
          <div className="empty-workflow">
            <strong>等待执行计划</strong>
            <span>发送指令后，这里会展示顺序或并发 workflow。</span>
          </div>
        ) : null}
      </div>

      <div className="event-log">
        <h3>Live Updates</h3>
        {workflow.events.length ? (
          workflow.events.map((event) => (
            <div key={event.id} className="event-row">
              <time>{event.time.toLocaleTimeString()}</time>
              <span>{event.text}</span>
            </div>
          ))
        ) : (
          <p>暂无流式事件。</p>
        )}
      </div>

      {workflow.summary ? (
        <div className="result-box">
          <h3>Result Summary</h3>
          <p>{workflow.summary}</p>
        </div>
      ) : null}
    </section>
  );
}

function PipelineStep({ step, mode, isStreaming }) {
  return (
    <article className={`pipeline-step status-${step.status}`}>
      <div className="step-index">{mode === "parallel" ? "∥" : mode === "dynamic" ? "↻" : step.index}</div>
      <div className="step-body">
        <div className="step-title">
          <strong>{step.agent}</strong>
          <span>{step.status}</span>
        </div>
        <p>{step.summary}</p>
        {step.data ? <DataPreview data={step.data} /> : null}
      </div>
      {isStreaming && step.status === "working" ? <span className="pulse" aria-hidden="true" /> : null}
    </article>
  );
}

function DataPreview({ data }) {
  const entries = Object.entries(data).filter(([, value]) => typeof value !== "object").slice(0, 4);
  if (!entries.length) return null;
  return (
    <dl className="data-preview">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{String(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

createRoot(document.getElementById("root")).render(<App />);
