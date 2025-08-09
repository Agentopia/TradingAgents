# Trading Squad Streamlit UI Spec (CLI Parity)

Last updated: 2025-08-09
Owner: Trading Squad
Repo path: `agents/trading-squad/`

## 1) Goal & Scope

- Build a Streamlit UI that mirrors the CLI workflow and user experience with feature parity.
- Preserve all multi-agent orchestration signals (statuses, messages, tool calls, intermediate reports) and the final decision path.
- Make provider/model selection and run configuration as flexible as the CLI.

Out of scope: non-functional redesigns not required for parity, cloud authentication flows.

## 2) Parity Checklist (Must-Haves)

- Agent lifecycle parity
  - Live per-agent status: `pending → running → complete|error` for all roles
  - Teams: Analysts, Research (Bull, Bear, Manager), Trader, Risk, Portfolio Manager
- Streaming parity
  - Stream intermediate states when `debug=True` using graph streaming
  - Show live messages and tool calls
  - Update report sections incrementally as keys appear in state
- Configuration parity
  - LLM provider selection: OpenAI, Anthropic, Google, OpenRouter, Ollama
  - Backend/base URL input where applicable (OpenRouter base, Ollama host)
  - Separate shallow/quick and deep model selectors per provider
  - Research depth → rounds mapping and explicit `max_debate_rounds`, `max_risk_discuss_rounds`
  - Analyst subset selection (optional, default all)
  - Online tools toggle (on/off)
- Final output parity
  - Complete tabbed report sections identical to CLI nomenclature
  - Explicit final decision, with traceability to research/risk debates
  - Persist run metadata (symbol, date, timestamp, models used)
- Error and resilience
  - Per-agent error display; global error banner
  - API key and network error hints
  - Known Social Analyst loop visibility (mark stuck with timer fallback only if no stream)

## 3) Data & State Model

Use `st.session_state` to maintain UI state:

- `config: dict` — exact keys passed to `TradingAgentsGraph`
  - `llm_provider`: str
  - `backend_url`: str|None (provider-dependent)
  - `deep_think_llm`: str
  - `quick_think_llm`: str
  - `max_debate_rounds`: int
  - `max_risk_discuss_rounds`: int
  - `online_tools`: bool
  - `selected_analysts`: list[str]
  - `debug`: bool
- `agent_status: dict[str, "pending"|"running"|"complete"|"error"]`
- `progress_messages: list[dict]` — {ts, source, text}
- `tool_calls: list[dict]` — {ts, agent, tool, args|summary, status}
- `report_sections: dict` — keys align with CLI: `market_report`, `sentiment_report`, `news_report`, `fundamentals_report`, `investment_debate_state`, `trader_investment_plan`, `risk_debate_state`, `final_trade_decision`
- `analysis_running: bool`
- `final_result: dict|None` — final state snapshot from graph end
- `final_decision: str|dict|None`
- `run_meta: dict` — {symbol, date, timestamp, provider, models}

## 4) Execution Model

- Non-debug mode: call `TradingAgentsGraph(...).propagate(company, date)` synchronously as today.
- Debug mode: stream via graph to update UI live.

### 4.1 Streaming Integration (Debug)

- If `debug=True`, call a streaming entry (preferred):
  - Option A (best): add `propagate_stream(company, date)` wrapper in `tradingagents/graph/trading_graph.py` that yields `(node_name, partial_state)`.
  - Option B (minimal change): expose `graph.stream({"input": ...})` from `TradingAgentsGraph` and forward chunks to UI.
- For each streamed `chunk`:
  - Detect `node_name` → map to role/team → set `agent_status[role] = running`.
  - When node completes (chunk indicates result placed in state), set `complete`.
  - If exception detected, set `error` and append to `progress_messages`.
  - Extract and merge any known section keys into `report_sections`.
  - Capture tool usages (see §5) if available in state/log.

## 5) Tool Call Logging (Optional but Recommended)

- Add lightweight hooks where tools execute (market/news/social/fundamentals fetchers):
  - Wrap tool functions to emit a callback `on_tool_call(agent, tool_name, args, status)` when invoked/completed/error.
  - Pass a callback from UI when creating `TradingAgentsGraph` only in debug mode to avoid noise in production.
- Update `tool_calls` list and render in a collapsible panel.

## 6) UI Layout & Components

- Sidebar
  - API keys: `OPENAI_API_KEY`, `FINNHUB_API_KEY`, and provider-specific hints
  - Provider select → dynamic fields:
    - OpenAI: base URL (optional), models list
    - OpenRouter: base URL required
    - Ollama: host (e.g., http://localhost:11434), models list
    - Anthropic/Google: model lists
  - Quick/Deep model selects
  - Research depth (Beginner/Standard/Deep) → rounds
  - Advanced: explicit `max_debate_rounds`, `max_risk_discuss_rounds`
  - Analyst multi-select (defaults to all from `cli/utils.py::ANALYST_ORDER`)
  - Online tools toggle
  - Debug toggle

- Main content
  - Header: title, symbol/date inputs, Run/Stop buttons
  - Live dashboard (during run):
    - Team/agent cards with status badges and subtle animations
    - Live log panel (messages + tool calls)
    - Current report preview (the most recently updated section)
  - Final report (after run): Tabs
    - Final Decision
    - I. Analyst Team (Market, Social, News, Fundamentals)
    - II. Research Team (Bull, Bear, Manager summary)
    - III. Trading Team
    - IV. Risk Management
    - V. Portfolio Manager

## 7) Widget → Config Mapping

- Provider: `config["llm_provider"]`
- Backend URL/Host: `config["backend_url"]`
- Quick model: `config["quick_think_llm"]`
- Deep model: `config["deep_think_llm"]`
- Research depth: map to `max_debate_rounds` and `max_risk_discuss_rounds` via same thresholds as CLI
- Analyst multi-select: `config["selected_analysts"]`
- Online tools: `config["online_tools"]`
- Debug: `config["debug"]`

## 8) Error Handling & UX

- API key validation states: missing, invalid format, ready
- Per-agent error surfacing: card shows error icon + tooltip with message
- Global error banner with actionable tip (retry, check network/keys)
- Social Analyst loop guard: if no stream events for N seconds during Social phase, show “possible external API stall” tag

## 9) Acceptance Criteria

- Start a run in debug mode and observe the following:
  - Agent cards transition through statuses tied to actual node progress
  - Tool calls appear as they happen with basic metadata
  - Report sections update incrementally; final tab content matches CLI sections
  - Switching provider/model changes the config passed to the graph
  - Selecting a subset of analysts results in only those nodes running (when supported)
  - Online tools toggle affects data source path (online/offline)
- Non-debug mode
  - No live stream UI, but final report identical; statuses default to completed on finish

## 10) Implementation Plan (Phases)

- Phase 1: Config parity
  - Add provider/base URL + quick/deep model selectors
  - Add research depth → rounds mapping and analyst multi-select
  - Wire `config` into `TradingAgentsGraph`
- Phase 2: Streaming
  - Add `propagate_stream()` or direct `graph.stream` usage
  - Update live status, messages, report sections during stream
- Phase 3: Tool logging & polish
  - Optional tool-call hooks and UI
  - Replace timer-based animations with event-driven transitions
  - Per-agent error visualization and helpful retry UX

## 11) Dev Notes & References

- CLI references
  - `agents/trading-squad/cli/main.py` — `MessageBuffer`, live display, sections
  - `agents/trading-squad/cli/utils.py` — providers, model menus, analyst order, depth mapping
- Graph orchestration
  - `agents/trading-squad/tradingagents/graph/trading_graph.py` — `TradingAgentsGraph`, `propagate()`; add streaming wrapper here
- Current UIs
  - `agents/trading-squad/app/trading_agents_streamlit.py` — modern UI; extend to add streaming + config parity
  - `agents/trading-squad/app/trading_agents_streamlit_fixed.py` — debugging/timer approach; keep as reference

## 12) Privacy & Keys

- Never store API keys; use password inputs and `.env` auto-loading (already present)
- Support local-first (Ollama) to align with Agentopia privacy vision
- Make base URL fields explicit to support gateways and self-hosted endpoints

## 13) Testing

- Fixtures: offline/cached mode to make runs deterministic
- Smoke tests: OpenAI and Ollama minimal models
- Regression: ensure CLI and Streamlit final report sections remain consistent on same seed/config

## 14) Run Controls & Artifacts

- Run controls
  - Start: validates config and kicks off run
  - Stop/Cancel: sets a cancellation flag respected by streaming loop; marks in-progress agents as cancelled
  - Reset: clears `agent_status`, `progress_messages`, `tool_calls`, `report_sections`, `final_result`, `final_decision`, `analysis_running`
- Export artifacts
  - Download final report as Markdown (`final_report.md`) including decision and sections
  - Download run log as JSONL capturing `progress_messages`, `tool_calls`, timestamps, and config
  - Copy-to-clipboard quick action for final decision summary

## 15) Environment & Launch Notes

- Local imports rely on running from `agents/trading-squad/` (or repo root) so that `tradingagents/...` resolves
- `.env` loading: document keys (`OPENAI_API_KEY`, `FINNHUB_API_KEY`) and provider-specific endpoints
- Sidebar should surface basic connection checks (e.g., test call to Ollama host when selected)

## 16) Node → Role Mapping (for Status Updates)

Define a mapping used by streaming to update `agent_status` correctly:

- Market node → "Market Analyst"
- Social/Sentiment node → "Social Analyst"
- News node → "News Analyst"
- Fundamentals node → "Fundamentals Analyst"
- Bull researcher node → "Bull Researcher"
- Bear researcher node → "Bear Researcher"
- Research manager/judge node → "Research Manager"
- Trader node → "Trader"
- Risk debate/judge node → "Risk Management"
- Portfolio manager node → "Portfolio Manager"

If node names differ, provide a normalization function to map graph node IDs to these role labels.

## 17) Accessibility & UX Standards

- Keyboard and screen-reader friendly status badges with ARIA labels
- High-contrast mode compliance for status colors
- Avoid rapid reflow; batch UI updates (e.g., debounced stream updates every 200–300ms)
- Persist last successful run in session to aid comparison; include a "Clear history" button

## 18) Performance & Timeouts

- Streaming watchdog: if no chunk in N seconds during a running phase, surface a non-blocking warning
- Per-agent soft timeout hints (UX only); do not hard-kill unless user cancels
- Respect `online_tools` off by skipping external calls where supported

## 19) Expanded Acceptance & Verification

- Run controls verified: Start/Stop/Reset behave as expected; cancellation leaves a clear UI state
- Exports verified: downloaded Markdown and JSONL contain expected content
- Node→role mapping verified: statuses reflect actual node completion order during a streamed run
- Environment verified: running from `agents/trading-squad/` resolves local modules; `.env` keys load automatically
- Accessibility smoke check: status badges readable in light/dark modes; ARIA labels present

## 20) Implementation Readiness Checklist

- [ ] Provider + model selectors match `cli/utils.py` options
- [ ] Analyst multi-select wired and honored by graph (or no-op clearly documented)
- [ ] Research depth mapping identical to CLI
- [ ] Streaming path in debug mode updates: statuses, messages, sections
- [ ] Tool-call logging present (or explicitly deferred) with UI panel
- [ ] Run controls implemented and cancellation respected
- [ ] Export buttons generate Markdown + JSONL
- [ ] Environment & key validation in sidebar
- [ ] Tests for offline mode and two provider paths (OpenAI, Ollama)

## 21) Agentopia UI/UX Alignment (Non-Blocking)

Follow `docs/ui-ux-guidelines.md` to ensure brand consistency without altering functionality:

- Use per-agent `ui_components.py` (copy template from another agent if needed):
  - `display_sidebar_header()` at top of sidebar
  - `display_agent_title(icon, agent_name)` at top of main panel
  - `display_sidebar_footer()` pinned to bottom of sidebar
- Maintain two-panel layout (sidebar + main) and keep primary controls in sidebar
- Keep functional parity first: if a styling choice conflicts with streaming updates, prefer correctness and revisit styling after
- Add accessible labels to status badges and ensure contrast compliance

Implementation note: add these calls to `app/trading_agents_streamlit.py` structure, but do not refactor the streaming/event logic into the component file; keep logic in the page script for clarity and to avoid breaking parity.
