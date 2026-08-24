# US Equity Research Agent

An LLM agent that autonomously calls tools to research a US-listed stock —
pulling SEC EDGAR financials, sector/thematic rotation, peer comparisons,
filing text, and post-earnings price reaction — and produces an investment
research report with a visible reasoning trail.

Give it a ticker; the agent decides for itself which tools to call, how many
rounds it needs, and when it has enough to write a conclusion.

## Why this exists

Most "AI stock analysis" demos either hardcode a fixed pipeline (fetch →
summarize → done) or let the model free-associate numbers from memory. This
project deliberately separates the two:

- **Deterministic facts are computed in code, never guessed by the LLM.**
  Financial metrics, relative-strength/momentum sector positions, and
  ticker-to-industry matching are all plain Python — the model only reasons
  over numbers it was actually handed.
- **The agent loop is real multi-turn autonomy, not a scripted sequence.**
  A hand-written `while` loop drives Anthropic-protocol `tool_use` /
  `tool_result` round-trips; the model decides which of 8 tools to call, in
  what order, and when to stop (capped by `max_turns`).
- **The model must self-check its own draft.** After writing a first-pass
  report, the agent is required to make at least one `verify_number` call
  that re-derives a claimed figure from source data before finalizing.
- **Best-of-N, not RL training.** An optional "deep analysis" mode runs 3
  candidate reports at different temperatures, scores each with a rule-based
  check (does every claimed number trace back to real tool output?) plus an
  LLM-judge pass, and shows the highest-scoring one — reward-driven
  selection, with fixed hand-tuned weights, not policy training.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python + FastAPI |
| Agent loop | Hand-rolled `while` loop over the Anthropic Messages API (`tool_use` protocol) — no agent framework |
| LLM provider | [DeepSeek](https://www.deepseek.com/) (Anthropic-compatible endpoint, primary — far cheaper for multi-turn tool calls with large filing-text results in history) or Claude, swappable via one env var |
| Financial data | [SEC EDGAR](https://www.sec.gov/edgar) `data.sec.gov` (free, no key, rate-limited to 10 req/s) |
| Market/sector data | [Polygon.io](https://polygon.io/) (price/volume) + [Alpha Vantage](https://www.alphavantage.co/) (analyst EPS estimates) |
| Frontend | React + Recharts + Tailwind CSS |
| Testing | pytest (backend), Vitest + Testing Library (frontend) |
| Lint/type-check | ruff + mypy (backend), oxlint (frontend) |

## What the agent can do

8 tools it autonomously chooses among:

| Tool | What it returns |
|---|---|
| `get_financials` | Revenue, net income, margins, EPS, YoY deltas — from SEC EDGAR XBRL, computed in code |
| `get_sector_position` | Where the company's SPDR sector sits on a relative-rotation (RRG) graph vs. the broad market |
| `get_thematic_flow` | Same RRG methodology applied to 10 narrower supply-chain themes (semiconductors, storage, optical modules, cloud, power, …), with a deterministic ticker→theme match (basket membership + SIC industry classification) |
| `get_peer_comparison` | Key metrics side-by-side against same-sector peers |
| `get_filing_text` | Raw 10-K/10-Q text from SEC full-text search — handed to the model unprocessed; no NLP cleaning pipeline, the model reads and filters it itself. Transparently falls back to 20-F/20-F-A for Foreign Private Issuers (companies that don't file 10-K/10-Q at all), with the response's `form` field reflecting what was actually retrieved |
| `get_price_reaction` | Price/volume move after the last earnings release, plus how much of that initial move was given back in the following days (computed, not an intent judgment like "shakeout" or "bull trap") |
| `get_earnings_surprise` | Reported EPS vs. analyst consensus EPS (Alpha Vantage), with a code-computed beat/miss/inline verdict |
| `verify_number` | Re-derives one claimed figure from source data — the self-check step the agent is required to invoke before finalizing |

The final report is emitted as `<conclusion>/<evidence>/<flags>` XML, which
the frontend renders as three distinct sections.

## Architecture

### Layered backend

```mermaid
flowchart TD
    subgraph Client["Frontend — React"]
        UI["Panels: Financials, Sector, Company Profile,<br/>Peer, Thematic Flow, Report, Reasoning trace"]
        Hooks["useAgentAnalysis / useBestOfNAnalysis<br/>(SSE-consuming state machines)"]
    end

    subgraph API["API layer — FastAPI routers"]
        Sync["Sync data routes<br/>/api/financials, /api/financials-history,<br/>/api/sector-position, /api/company-profile,<br/>/api/peer-comparison, /api/thematic-flow,<br/>/api/filing-text"]
        Async["Async agent routes<br/>/api/analyze/{ticker}/start (+ /best-of-n/start)<br/>/api/analyze/stream/{task_id} (SSE)"]
        Guard["rate_limit + session_guard + ticker_path<br/>(throttling, single-in-flight guard, input validation)"]
    end

    subgraph Service["Service layer"]
        Data["sec_edgar / polygon_client / alpha_vantage_client<br/>sector_rotation + rrg / company_profile /<br/>peer_comparison / price_reaction / earnings_surprise /<br/>thematic_flow / filing_text"]
        CacheLock["cache_lock — file-lock guarded disk cache<br/>shared by the data services above"]
    end

    subgraph Agent["Agent subsystem — services/agent/"]
        Loop["loop.py — the Agent Loop"]
        LLMClient["llm_client.py — provider adapter<br/>(DeepSeek / Claude, same interface)"]
        Tools["tools.py — 8 tool schemas + dispatch"]
        Prompt["system_prompt.py — retrieval / analysis /<br/>generation, concatenated at runtime"]
        Trace["task_registry + trace_log —<br/>background task, SSE replay buffer,<br/>per-run structured log"]
        BON["best_of_n.py + reward.py —<br/>N candidates, rule + LLM-judge scoring"]
    end

    subgraph External["External services"]
        SEC["SEC EDGAR<br/>data.sec.gov"]
        Market["Polygon.io / Alpha Vantage"]
        LLM["DeepSeek / Claude<br/>(Anthropic Messages API)"]
    end

    UI --> Hooks --> Sync
    Hooks --> Async
    Sync --> Guard
    Async --> Guard
    Sync --> Data
    Async --> Trace --> Loop
    Loop --> Tools --> Data
    Loop --> LLMClient --> LLM
    Loop --> Prompt
    BON --> Loop
    Data --> CacheLock
    Data --> SEC
    Data --> Market
```

### Agent Loop

The core `while` loop in `loop.py` — the model drives every branch; the code
only enforces the turn budget and the deterministic gate checks.

```mermaid
flowchart TD
    Start(["run_agent_loop(ticker)"]) --> Call["LLM.create_message(system, messages, tools, temperature)"]
    Call --> Record["record reasoning_note, emit 'reasoning' event"]
    Record --> Decide{"stop_reason?"}

    Decide -- "tool_use" --> Parallel["asyncio.gather: run every tool_use<br/>call from this turn in parallel"]
    Parallel --> EmitTool["emit tool_call_started / tool_call_finished"]
    EmitTool --> Compact["compact old, already-seen large tool<br/>results into placeholders<br/>(structured-data tools exempt)"]
    Compact --> Append["append tool_results, next turn"]
    Append --> Budget{"turn < max_turns?"}
    Budget -- yes --> Call
    Budget -- no --> MaxTurns["return completed=false<br/>stop_reason=max_turns_exceeded"]

    Decide -- "end_turn, has &lt;conclusion&gt;,<br/>not yet gate-checked" --> Gates["batch-check 6 gates: self-verification /<br/>structure / tool coverage / verification<br/>mismatch / traceability / sentiment consistency"]
    Gates --> GateIssues{"any issues?"}
    GateIssues -- yes --> Nudge["inject one combined nudge<br/>(all issues at once)"]
    Nudge --> Call
    GateIssues -- no --> ReflexionCheck

    Decide -- "end_turn, already nudged,<br/>reply has no tags" --> FormatNudge["inject 'must re-emit the full<br/>tagged report' nudge (once)"]
    FormatNudge --> Call

    ReflexionCheck{"reflexion_check given?<br/>(Best-of-N only)"}
    ReflexionCheck -- "yes, critique returned" --> ReflexNudge["inject critique, ask to redo (once)"]
    ReflexNudge --> Call
    ReflexionCheck -- "no / already checked" --> Finalize["resolve final_report<br/>(fall back to last tagged reasoning<br/>note if this turn's text has none)"]
    Finalize --> Return["return AgentRunResult<br/>completed = (stop_reason == end_turn)"]

    Decide -- "refusal / max_tokens" --> Return
```

### Request sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as Frontend
    participant API as FastAPI (sync routes)
    participant Reg as task_registry (SSE)
    participant AL as Agent Loop
    participant LLM as DeepSeek / Claude
    participant Tools as Tools (SEC EDGAR / Polygon / Alpha Vantage)

    User->>FE: enter ticker, click "Analyze"
    par structured data — shown immediately
        FE->>API: GET /api/financials/{ticker}
        FE->>API: GET /api/sector-position/{ticker}
        FE->>API: GET /api/company-profile/{ticker}
        FE->>API: GET /api/peer-comparison/{ticker}
    end
    API-->>FE: financials, sector position, profile, peers

    FE->>Reg: POST /api/analyze/{ticker}/start
    Reg-->>FE: { task_id }
    Reg->>AL: run_agent_loop(ticker) — background task
    FE->>Reg: GET /api/analyze/stream/{task_id} (SSE: replay buffer, then live)

    loop until end_turn or max_turns
        AL->>LLM: create_message(system, messages, tools)
        LLM-->>AL: text + tool_use[] (stop_reason)
        Reg-->>FE: SSE "reasoning"
        par same-turn tool calls, run in parallel
            AL->>Tools: execute_tool(name, input)
            Tools-->>AL: result (disk-cached where applicable)
        end
        Reg-->>FE: SSE "tool_call_started" / "tool_call_finished"
        AL->>AL: compact old large tool results,<br/>append new tool_results
    end
    AL->>AL: batch gate checks, nudge + retry once<br/>if self-check / structure / traceability fail
    AL-->>Reg: AgentRunResult (final_report, transcript, gate flags)
    Reg-->>FE: SSE "done"
    FE-->>User: render ReportPanel (conclusion/evidence/flags)<br/>+ AgentReasoningPanel (full trace)

    opt user clicks "Deep analysis"
        FE->>Reg: POST /api/analyze/{ticker}/best-of-n/start
        Reg->>AL: run_agent_loop × 3 in parallel (temperature 0.3 / 0.6 / 1.0)
        AL-->>Reg: 3 candidates, scored (rule-based + LLM judge)
        Reg-->>FE: SSE "done" — selected candidate + all scores
        FE-->>User: render CandidateComparisonPanel
    end
```

## Project structure

```
backend/
  app/
    api/routes/       FastAPI routers (sync data endpoints + SSE streaming)
    services/          Data-fetching + computation (SEC EDGAR, Polygon, RRG, ...)
    services/agent/    The agent loop, LLM provider adapter, tool schemas,
                        Best-of-N reward scoring
    models/            Pydantic response schemas
  scripts/
    eval_report_quality.py   Offline regression check for report quality
                              (not wired into CI — costs real LLM calls)
  tests/
frontend/
  src/
    components/        Panels (financials, sector, thematic flow, report, ...)
    hooks/              SSE-consuming state machines for the two analysis modes
    lib/                Pure helpers (report XML parsing)
```

## Getting started

### Prerequisites

- Python 3.13
- Node 20
- A DeepSeek or Anthropic API key, and a Polygon.io API key (both have free
  tiers sufficient for local development)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # includes runtime deps + ruff/mypy
cp .env.example .env                  # fill in your API keys
uvicorn app.main:app --reload
```

The API listens on `http://localhost:8000`; `/health` is a liveness check,
`/docs` has interactive OpenAPI docs.

Required env vars (see `backend/.env.example`):

| Var | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` | LLM provider credentials |
| `LLM_PROVIDER` | `deepseek` (default) or `claude` |
| `SEC_EDGAR_USER_AGENT` | Required by SEC EDGAR — format: `"Your Name your_email@example.com"` |
| `POLYGON_API_KEY` | Market/sector data |
| `ALPHA_VANTAGE_API_KEY` | Analyst EPS estimates for `get_earnings_surprise` — optional, tool soft-degrades (`has_data: false`) without it |

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite dev server runs on `http://localhost:5173` and proxies `/api/*` to the
backend on port 8000.

### Tests, lint, type-check

```bash
# backend (from backend/)
pytest -q
ruff check .
mypy

# frontend (from frontend/)
npm run test
npm run lint
npm run build
```

All four run in CI on every push/PR to `main` (`.github/workflows/ci.yml`).

## Scope

This is a single-agent MVP, deliberately scoped:

- No earnings-call transcripts (would require a paid data source; only
  SEC EDGAR filing text is used — 10-K/10-Q for domestic filers, 20-F/20-F-A
  for Foreign Private Issuers)
- No multi-agent orchestration
- No real reinforcement-learning training — Best-of-N is reward-driven
  *selection* among sampled candidates with hand-tuned weights, not
  weight training
- No cross-session memory, batch scanning, or trading advice
