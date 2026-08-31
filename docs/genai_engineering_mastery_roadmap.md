# The 80/20 GenAI Engineering Mastery Roadmap — 30 Days, 450 Hours, Three Roles

## TL;DR
- **Master the 80/20 production stack first.** Across 30 days × 15 hrs (450 hrs total), 80% of production GenAI value comes from six skills: LangGraph for orchestration, OpenAI Responses API + Anthropic Messages API for LLM calls, hybrid RAG with rerankers and RAGAS evaluation, MCP for tool integration, LangSmith for observability, and **Microsoft Foundry** (the 2025 rebrand of Azure AI Foundry, announced at Microsoft Ignite on November 18, 2025) for deployment. Everything else is a "nice-to-know" specialization.
- **The three roles diverge late.** Days 1–20 are shared by all three personas. From Day 21 the GenAI/AI Engineer drills MCP + production patterns, the AI/ML Engineer drills containers/K8s/IaC/CI-CD, and the Senior DS (Agentic AI) drills evaluation rigor, red-teaming, and research methodology. The final 5 days converge on a capstone.
- **Default to LangGraph + OpenAI Responses + Anthropic + MCP + LangSmith + Microsoft Foundry.** Treat CrewAI Flows, OpenAI Agents SDK, Google ADK, AutoGen, AWS Bedrock AgentCore, and Vertex AI Agent Engine as comparative literacy, not deep practice — read docs and ship one "hello world" each, but don't build production on more than one orchestration framework or one cloud.

---

## Key Findings

1. **The orchestration market converged in 2025.** LangGraph (used inside LangChain v1's `create_agent`), CrewAI Flows, OpenAI Agents SDK, Microsoft Agent Framework, and Google ADK all settled on the same primitives: a state graph + tool nodes + handoffs + interrupts + checkpointers + tracing. Learn LangGraph deeply once; the others are 1–2 day transfers.
2. **The API surface converged too.** OpenAI's Responses API (launched March 11, 2025, per OpenAI's official announcement: "Today, we're releasing the first set of building blocks that will help developers and enterprises build useful and reliable agents") is now the recommended primitive for new projects, with built-in tools (web search, file search, code interpreter, remote MCP) and stateful conversations. Anthropic's Messages API adds prompt caching (cache reads billed at 0.1× the base input token price — a 90% reduction, per Anthropic's pricing docs: "Cache read tokens are 0.1 times the base input tokens price"), extended/adaptive thinking, and citations. Google Gemini 2.5/3 adds explicit + implicit context caching and grounding with Google Search. **Use LiteLLM only when you need provider-agnostic routing**; otherwise use the native SDKs.
3. **MCP is the new tool integration standard.** The Model Context Protocol (specification version 2025-11-25, released November 25, 2025 per the official MCP blog: "Today, MCP turns one year old … we're also releasing a brand-new MCP specification version") standardized on **stdio + Streamable HTTP** (SSE deprecated) with OAuth 2.1 for remote servers. It's natively supported in the OpenAI Responses API, Anthropic Claude, Claude Code, Cursor, and AWS Bedrock AgentCore Gateway. Building one custom MCP server is now mandatory literacy for any GenAI engineer.
4. **Observability has three viable options.** LangSmith (best LangChain/LangGraph DX, vendor-locked, trace-priced — collapses under agentic workloads), Langfuse (open-source, ClickHouse-backed, OpenTelemetry-native, transparent pricing), Arize Phoenix (framework-agnostic, single-Docker self-host, native RAGAS integration, free OSS features that Langfuse paywalls). For a production agent stack, **LangSmith + Langfuse/Phoenix as fallback** is a common pattern.
5. **RAG breaks predictably in five ways.** Bad chunking, embedding mismatch, lost-in-the-middle, multi-hop failures, and stale data. Hybrid search (BM25 + dense + Cohere/Voyage/Jina reranker) + RAGAS-driven eval-loop development fixes ~80% of cases without expensive architecture changes.
6. **Microsoft Foundry is the new umbrella.** Microsoft rebranded "Azure AI Foundry" to Microsoft Foundry at Microsoft Ignite on November 18, 2025; Azure OpenAI resources auto-upgrade to Foundry resources. Foundry Agent Service runs containerized hosted agents on Azure Container Apps and supports the A2A protocol in preview. **Azure AI Search was rebranded "Foundry IQ"** but the underlying hybrid + semantic ranker API is unchanged.
7. **AWS Bedrock AgentCore (preview, 2025)** offers 8 primitives — Runtime, Memory, Identity, Gateway, Browser, Code Interpreter, Observability, Evaluations — and is framework-agnostic (CrewAI, LangGraph, LlamaIndex, Strands). AgentCore Gateway is a managed MCP server.
8. **Google Vertex AI Agent Engine** is the managed runtime for ADK and other framework agents, with built-in `VertexAiSessionService` for multi-turn memory. Vertex AI is transitioning into **Gemini Enterprise Agent Platform** (2026), with A2A for cross-framework agent communication.
9. **Agentic coding tools are now production infrastructure.** Claude Code (CLI + VS Code + JetBrains + iOS), Cursor (.cursor/rules .mdc format with 4 activation modes), OpenAI Codex CLI, GitHub Copilot. Each agent persona should master Claude Code as the daily driver and Cursor as the IDE.
10. **Beads (Steve Yegge, 2025)** is the breakout "agent memory" tool — a git-native, dependency-tracking issue graph stored as JSONL that survives context compaction. Not a "must-master" but increasingly expected literacy.

---

## Details — The 30-Day Roadmap

### Conventions
- **Role markers:** 🤖 = GenAI/AI Engineer (production agents) · 🔧 = AI/ML Engineer (MLOps + infra) · 🔬 = Senior DS Agentic AI (eval + research). When all three apply, no marker.
- **Tier markers:** **[MUST]** = top-20% skill; **[NICE]** = optional / comparative literacy.
- Daily allocation = **15 hrs**: ~8 hrs deep work, ~4 hrs build/exercise, ~2 hrs reading docs/papers, ~1 hr review & journal. Build in a 15-min break each hour.
- **Pre-flight budget:** ~$200–$400 in API credits across providers; free tier for Langfuse/Phoenix; Microsoft Foundry free trial or Azure $200 credit.

---

## Phase 1 — Foundations: LLM APIs, Python/TypeScript & Prompt Engineering (Days 1–6, 90 hrs)

### Day 1 — Environment + Python for AI/ML (15 hrs)
- **Topics:** Python 3.12+ setup with **uv** (Astral) as primary package manager, **Poetry** as fallback (3 hrs); `asyncio`, `async/await`, `asyncio.gather`, async context managers (3 hrs); `typing` deep dive — `TypedDict`, `Protocol`, `Annotated`, `ParamSpec`, `TypeGuard` (2 hrs); **Pydantic v2** — `BaseModel`, validators, `model_validate`, `Field`, discriminated unions (3 hrs); **FastAPI** — async endpoints, dependency injection, streaming responses with SSE (3 hrs); project layout for AI services (1 hr).
- **Hands-on:** Build an async FastAPI service that streams chat completions from a mock LLM, with Pydantic-validated request/response models. Push to a fresh GitHub repo.
- **Role focus:** 🤖🔧🔬 all equal.
- **Resources:** uv docs (docs.astral.sh/uv) · FastAPI docs (fastapi.tiangolo.com) · Pydantic v2 docs (docs.pydantic.dev) · *Real Python* asyncio guide.

### Day 2 — TypeScript for AI (15 hrs)
- **Topics:** Node.js 22 LTS + **Bun 1.x** runtime tradeoffs (2 hrs); TypeScript 5.5+ strict mode, Zod for runtime validation (2 hrs); **Vercel AI SDK v5/v6** — `generateText`, `streamText`, `Output.object()`, `ToolLoopAgent`, dynamic tools, SSE-based streaming, tool execution approval (5 hrs); Next.js 15 App Router + **Server Actions** for AI endpoints (3 hrs); **tRPC v11** for type-safe internal APIs (2 hrs); Vercel AI Gateway model routing (1 hr).
- **Hands-on:** Port Day 1's FastAPI app to a Next.js Server Action + AI SDK app calling `openai/gpt-5.2` or `anthropic/claude-sonnet-4.6` with a Zod tool schema.
- **Role focus:** 🤖 (heavy — production UIs); 🔧 (light — read only); 🔬 (light).
- **Resources:** ai-sdk.dev/docs · Vercel AI SDK 5/6 blog posts (vercel.com/blog/ai-sdk-5, vercel.com/blog/ai-sdk-6) · tRPC docs · Next.js docs.
- **[NICE for 🔬]:** Skim only — most Senior DS work happens in Python.

### Day 3 — OpenAI API Deep Dive (15 hrs) **[MUST]**
- **Topics:** **Chat Completions API** vs. **Responses API** migration (2 hrs); **Responses API** stateful conversations (`store: true`), built-in tools (web search, file search, code interpreter, image generation, remote MCP), background mode, reasoning summaries, encrypted reasoning items (5 hrs); **function calling** with strict-mode tool schemas + **structured outputs** (strict JSON Schema mode) + classic **JSON mode** comparison and trade-offs (3 hrs); streaming, parallel tool calls (2 hrs); prompt caching (automatic + `prompt_cache_key` + `prompt_cache_retention`) (1 hr); rate limiting, exponential backoff with `tenacity`, cost tracking (1 hr); **Assistants API** sunset (2026-08-26) and migration paths to Responses (1 hr).
- **Hands-on:** Build a multi-tool agent in the Responses API with web_search + a custom function tool + structured Pydantic output. Add streaming, retry logic, and a token/cost meter.
- **Role focus:** 🤖🔧🔬 all critical.
- **Resources:** developers.openai.com/api/docs · OpenAI cookbook (github.com/openai/openai-cookbook) · *Why we built the Responses API* (developers.openai.com/blog/responses-api).

### Day 4 — Anthropic API Deep Dive (15 hrs) **[MUST]**
- **Topics:** Messages API basics, system prompts, message structure (2 hrs); **tool use** (`tools`, `tool_use`, `tool_result` blocks), parallel tool use, **Tool Runner SDK** (4 hrs); **prompt caching** — `cache_control: ephemeral`, 5-min vs. 1-hr TTL, breakpoint placement, cache reads at 0.1× the base input token price per Anthropic pricing docs (3 hrs); **extended/adaptive thinking** on Claude Opus 4.7 / Sonnet 4.6, interleaved thinking (`interleaved-thinking-2025-05-14` beta on older models, auto-enabled on Opus 4.6+), thinking budgets, cache interactions (3 hrs); **citations** for grounded outputs (1 hr); search results, server tools (web search, web fetch, code execution, computer use), Memory tool (2 hrs).
- **Hands-on:** Build a research agent using Claude Sonnet 4.6 with web_search + extended thinking + prompt caching of a 5K-token system prompt. Measure cost delta with vs. without caching.
- **Role focus:** 🤖🔧🔬 all critical.
- **Resources:** docs.anthropic.com · *Prompt caching* (platform.claude.com/docs/en/build-with-claude/prompt-caching) · *Building with extended thinking* (docs.anthropic.com/en/docs/build-with-claude/extended-thinking) · github.com/anthropics/anthropic-cookbook.

### Day 5 — Google Gemini API + SDK Comparison (15 hrs)
- **Topics [MUST]:** `google-genai` Python SDK; **Gemini 2.5 Pro / 2.5 Flash / 2.5 Flash-Lite** + **Gemini 3 Pro / 3 Flash** model selection (2 hrs); function calling, structured output with JSON schemas (2 hrs); **context caching** — implicit (default on 2.5+) + explicit (`client.caches.create`, TTL-billed) (2 hrs); **grounding with Google Search** (free tier 1,500 RPD on Pro, then $35/1K) + grounding with Google Maps (2 hrs); **thinking levels** (`thinkingLevel` for Gemini 3) and `thinkingBudget` for 2.5 (1 hr); Live API for real-time audio (1 hr).
- **Topics [NICE]:** Unified interfaces — **LiteLLM** for provider-agnostic routing, OpenRouter, Vercel AI Gateway (2 hrs); cost dashboards (1 hr); **OpenAI Python/JS vs. Anthropic Python/JS vs. google-genai vs. LiteLLM SDK side-by-side** comparison: tool calling, streaming, structured output, caching, error model (2 hrs).
- **Hands-on:** Build the same agent in OpenAI, Anthropic, Gemini, and LiteLLM. Compare latency, cost, and developer ergonomics.
- **Role focus:** 🤖🔧🔬 all equal.
- **Resources:** ai.google.dev/gemini-api/docs · LiteLLM docs (docs.litellm.ai)

### Day 6 — Prompt Engineering as a Discipline (15 hrs) **[MUST]**
- **Topics:** System prompts that work — role + constraints + format + examples + tools (2 hrs); zero-shot / **few-shot** / **chain-of-thought** / **tree-of-thought** / **ReAct** / **Reflexion** patterns (4 hrs); structured outputs vs. JSON mode vs. strict schemas — when to use which (2 hrs); tool-use patterns: planner/executor, single-shot, parallel tool use, "tool search" (2 hrs); meta-prompting + prompt optimization (DSPy preview) (2 hrs); 🔬 **eval-driven prompt iteration** — build a 20-prompt eval set, A/B compare temperatures and few-shots (3 hrs).
- **Hands-on:** Build a structured extraction pipeline (invoices → JSON) and a 30-example regression suite. Achieve ≥95% field-level accuracy.
- **Role focus:** 🤖 (heavy); 🔬 (heavy — eval mindset); 🔧 (moderate).
- **Resources:** *Prompt Engineering Guide* (promptingguide.ai) · *A practical guide to building agents* (OpenAI) · DeepLearning.AI *ChatGPT Prompt Engineering for Developers* · *AI Engineering* by Chip Huyen, Chapter 5.

**Phase 1 milestone:** Deploy a multi-provider, structured-output, tool-using agent as a FastAPI service. Cost meter ≤ $0.02 per call median.

---

## Phase 2 — Agent Orchestration: LangChain, LangGraph, CrewAI (Days 7–14, 120 hrs)

### Day 7 — LangChain v1 Fundamentals (15 hrs) **[MUST]**
- **Topics:** LangChain v1.0+ architecture — `langchain-core`, `langchain`, provider packages (1 hr); **LCEL (LangChain Expression Language)** — `Runnable`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, chains (3 hrs); ChatPromptTemplate, output parsers (Pydantic, JSON, structured) (2 hrs); retrievers, embeddings, vector store interfaces (2 hrs); tools (`@tool` decorator, structured tools) (2 hrs); memory (legacy — most agents now use LangGraph checkpointers) (1 hr); **`create_agent`** in v1 (built on LangGraph under the hood) (3 hrs); when LCEL is enough vs. when you need LangGraph (1 hr).
- **Hands-on:** Build a RAG chain in pure LCEL with a Pydantic output parser, then port it to `create_agent`.
- **Resources:** docs.langchain.com (v1) · github.com/langchain-ai/langchain · LangChain Academy *LangGraph Essentials*.

### Day 8 — LangGraph Core: StateGraph, Nodes, Edges (15 hrs) **[MUST — the keystone day]**
- **Topics:** Why graphs for agents (vs. linear chains) (1 hr); **`StateGraph`** with TypedDict + `Annotated[..., operator.add]` reducers (3 hrs); nodes as Python functions, conditional edges, `Command(goto=..., update=...)` for dynamic routing (3 hrs); **checkpointers** — `InMemorySaver`, `SqliteSaver`, `PostgresSaver` (Redis-backed for prod) (2 hrs); thread IDs, time-travel, state inspection (2 hrs); streaming modes (`updates`, `values`, `messages`, `debug`) (2 hrs); subgraphs and composability (2 hrs).
- **Hands-on:** Build a 5-node research agent with conditional routing and SQLite checkpointing. Stream updates to a terminal.
- **Resources:** docs.langchain.com/oss/python/langgraph · github.com/langchain-ai/langgraph · LangChain Academy *Intro to LangGraph* (free, 6 modules).

### Day 9 — LangGraph Advanced: HITL, Subgraphs, Multi-Agent Patterns (15 hrs) **[MUST]**
- **Topics:** **`interrupt()` + `Command(resume=...)`** for human-in-the-loop (3 hrs); dynamic interrupts, validation loops, multi-interrupt graphs (2 hrs); **canonical agent patterns** — the **ReAct agent** (reason → act → observe loop), the **tool-calling agent** (model-driven function calling with no explicit reasoning step), and **plan-and-execute** (separate planner + worker nodes) — implement each as a small graph (3 hrs); **multi-agent patterns** — supervisor, swarm, handoffs (3 hrs); `langgraph-supervisor` and `langgraph-swarm` libraries (2 hrs); ambient agents pattern (1 hr); state migration considerations (1 hr).
- **Hands-on:** Convert Day 8's agent into three side-by-side variants — a ReAct agent, a tool-calling agent, and a plan-and-execute agent — and benchmark cost + latency + task success against the same 10-prompt suite. Then convert to a supervisor pattern with three worker agents (researcher, writer, fact-checker) with an approval interrupt before final output.
- **Resources:** docs.langchain.com/oss/python/langgraph/interrupts · LangChain Academy *Ambient Agents with LangGraph* course · Sam Witteveen YouTube LangGraph series.

### Day 10 — LangGraph Production Concerns (15 hrs)
- **Topics:** Error handling and retries inside graphs (2 hrs); **LangGraph Platform** vs. self-deployment (2 hrs); **LangGraph Studio** for visual debugging (2 hrs); persistence backends (Postgres for prod) (2 hrs); long-running tasks and background execution (2 hrs); rate limit handling + backoff inside nodes (1 hr); 🔧 **deployment patterns** — FastAPI wrapper, async streaming endpoints, websocket integration (3 hrs); 🤖 hooks into LangSmith tracing (1 hr).
- **Hands-on:** Containerize the Day 9 supervisor agent with a FastAPI front-end and Postgres checkpointer. Deploy locally with docker-compose.
- **Role focus:** 🔧 (heavy); 🤖 (heavy); 🔬 (light).

### Day 11 — CrewAI: Crews + Flows (15 hrs)
- **Topics:** CrewAI architecture — agents, tasks, crews, processes (sequential, hierarchical) (3 hrs); tools, knowledge sources, memory (2 hrs); **CrewAI Flows** event-driven orchestration: `@start`, `@listen`, `@router`, `@persist`, `@human_feedback` decorators (4 hrs); `or_` / `and_` triggering; structured (Pydantic) vs. unstructured state (2 hrs); combining Flows with Crews (2 hrs); custom LLM integration (1 hr); CrewAI AMP Suite + Control Plane (1 hr).
- **Hands-on:** Rebuild Day 9's supervisor as a CrewAI Flow with three Crews and a router that branches on a "confidence score."
- **Role focus:** 🤖🔬 (moderate — comparative literacy); 🔧 (light).
- **Resources:** docs.crewai.com · github.com/crewAIInc/crewAI · DeepLearning.AI *Multi AI Agent Systems with crewAI* (Joao Moura).

### Day 12 — OpenAI Agents SDK + Microsoft Agent Framework / AutoGen (15 hrs)
- **Topics [MUST]:** **OpenAI Agents SDK** (Python + JS/TS) — `Agent`, `Runner.run` / `run_sync` / `run_streamed`, **handoffs** as tool calls (with `input_filter`, `nest_handoff_history`), **input/output/tool guardrails** with tripwires, function tools, hosted tools (FileSearch, WebSearch, ComputerTool, HostedMCPTool, ShellTool, ApplyPatchTool), sessions, sandbox agents with manifests (5 hrs); built-in tracing in the OpenAI Traces dashboard (1 hr); `needs_approval` pattern for human review (1 hr).
- **Topics [NICE]:** **Microsoft Agent Framework** — the .NET + Python successor to Semantic Kernel and AutoGen, designed for Foundry Agent Service (2 hrs); **Microsoft AutoGen v0.4** overview — `AssistantAgent`, `UserProxyAgent`, GroupChat patterns (2 hrs); **comparison matrix**: LangGraph vs. CrewAI vs. AutoGen vs. OpenAI Agents SDK vs. Microsoft Agent Framework — control, learning curve, ecosystem lock-in, observability, HITL support (3 hrs).
- **Hands-on:** Rebuild Day 9's flow in the OpenAI Agents SDK with a guardrail that blocks math-homework prompts and a handoff between Triage and Specialist agents.
- **Resources:** openai.github.io/openai-agents-python · github.com/openai/openai-agents-python · learn.microsoft.com/agent-framework · microsoft/autogen GitHub.

### Day 13 — Google ADK + Framework Selection Day (15 hrs)
- **Topics [NICE for 🤖🔬, MUST if deploying on GCP]:** **Google ADK** — `LlmAgent`, **workflow agents** (`SequentialAgent`, `ParallelAgent`, `LoopAgent`), custom agents extending `BaseAgent`, **AgentTool** for nested agents, Skills, Plugins, callbacks (5 hrs); ADK CLI + Web UI + Agent Builder visual editor (2 hrs); deployment to Vertex AI Agent Engine in one command (`adk deploy agent_engine`) (2 hrs); **A2A protocol** for cross-framework agent communication (2 hrs).
- **Topics [MUST]:** Framework selection rubric — LangGraph for max control, CrewAI for role-based teams, OpenAI Agents SDK for OpenAI-native + fast, ADK for GCP-native, Agent Framework for Microsoft-native (3 hrs); migration costs between frameworks (1 hr).
- **Hands-on:** Deploy a 2-agent ADK system to Vertex AI Agent Engine; build a comparison table across all 5 frameworks.
- **Resources:** google.github.io/adk-docs · github.com/google/adk-python · cloud.google.com/products/agent-builder.

### Day 14 — Capstone Mini-Project: Production Multi-Agent System (15 hrs) **[MUST]**
- **Topics + build:** Design and ship a customer-support multi-agent system in LangGraph: 4 agents (triage, knowledge, action, escalation) + 1 supervisor + Postgres checkpointer + LangSmith tracing + a HITL interrupt for refunds > $100 + structured output schema + 3 guardrails (PII, prompt injection, off-topic) + a 20-case eval set (15 hrs full day).
- **Deliverable:** Public GitHub repo + Loom demo + cost-per-conversation analysis.
- **Role focus:** All three roles ship this; 🔬 owns the eval set, 🔧 owns the docker-compose + Postgres, 🤖 owns the graph + prompts.

**Phase 2 milestone:** A production-shape multi-agent app, traced in LangSmith, with HITL and an evaluation suite.

---

## Phase 3 — RAG, Vector Stores & Agentic Coding Tools (Days 15–20, 90 hrs)

### Day 15 — RAG Architecture: Naive → Advanced → Agentic (15 hrs) **[MUST]**
- **Topics:** Naive RAG (chunk → embed → retrieve → stuff) and its failure modes (2 hrs); **the five ways RAG breaks**: bad chunking, embedding mismatch, lost-in-the-middle, multi-hop failures, hallucination from wrong context, stale data (3 hrs); advanced patterns — parent-document retrieval, multi-query, **HyDE**, **Anthropic Contextual Retrieval** (prepend chunk context with a Claude call before embedding), self-query (3 hrs); **agentic RAG** — agent-driven query planning + iterative retrieval + reflection (4 hrs); recursive retrieval, GraphRAG overview (2 hrs); when *not* to use RAG (1 hr).
- **Hands-on:** Build a naive RAG on 100 PDFs, then layer parent-document retrieval + multi-query + reflection. Measure recall@5 improvements.
- **Resources:** *AI Engineering* Chapter 6 (Huyen) · Anthropic *Contextual Retrieval* blog · LlamaIndex Advanced RAG cookbook.

### Day 16 — Vector Stores Deep Dive (15 hrs) **[MUST]**
- **Topics:** Embedding model selection — OpenAI `text-embedding-3-large`, Cohere `embed-v4`, Voyage `voyage-3`, BGE-M3 (2 hrs); **Chroma** for local dev (1 hr); **Weaviate** — schema-first, GraphQL, hybrid search with `alpha`, `relativeScoreFusion`, multi-tenancy with tenant lifecycle APIs (3 hrs); **Pinecone** — serverless, namespaces (up to 100K), Pinecone Assistant (GA Jan 2025), BYOC (2 hrs); **pgvector** + **pgvectorscale** for Postgres-native (HNSW, IVFFlat) (2 hrs); **Qdrant** — Rust core, payload filters, named vectors, prefetch + multi-stage queries (2 hrs); **Azure AI Search / Foundry IQ** — hybrid (BM25 + HNSW/eKNN) + semantic ranker (2 hrs); decision rubric: scale, latency, multi-tenancy, ops effort, cost (1 hr).
- **Hands-on:** Ingest 100K Wikipedia paragraphs into 3 stores (Qdrant, Weaviate, pgvector), benchmark latency and recall@10 against the same query set.
- **Resources:** Each vendor's docs + the Firecrawl/Xenoss comparison guides.

### Day 17 — Hybrid Search + Reranking (15 hrs) **[MUST]**
- **Topics:** **BM25** mechanics, why lexical still matters for proper nouns/IDs/codes (2 hrs); dense embedding similarity (cosine, dot product) (1 hr); **hybrid search** — RRF (Reciprocal Rank Fusion) and relativeScoreFusion (2 hrs); **rerankers** — **Cohere Rerank v3.5**, **Voyage rerank-2**, **Jina Reranker v2** — when each wins (4 hrs); chunking strategies — semantic chunking (LlamaIndex), recursive character splitter, **late chunking**, hierarchical chunks (3 hrs); metadata filtering and pre-filtering vs. post-filtering (2 hrs); 🔬 measuring chunk quality empirically (1 hr).
- **Hands-on:** Take Day 15's RAG pipeline and add BM25 + dense + Cohere rerank. Measure faithfulness lift with RAGAS.
- **Resources:** Cohere/Voyage/Jina docs · LlamaIndex `SemanticSplitterNodeParser` docs.

### Day 18 — RAG Evaluation with RAGAS + LangSmith Datasets (15 hrs) **[MUST for 🔬]**
- **Topics:** Why standard NLP metrics (BLEU, ROUGE) fail for RAG (1 hr); **RAGAS** core metrics — **faithfulness**, **answer relevancy**, **context precision**, **context recall**, **context entity recall**, **noise sensitivity** (4 hrs); reference-free vs. reference-based eval (1 hr); building a golden dataset from real queries + synthetic generation (3 hrs); **LLM-as-a-judge** — calibration, position bias, verbosity bias (2 hrs); **eval-driven development loop** — change → re-evaluate → ship (2 hrs); 🔬 statistical significance for prompt A/B tests (2 hrs).
- **Hands-on:** Build a 100-query golden set, run RAGAS against 4 RAG variants, produce a decision report.
- **Resources:** docs.ragas.io · Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert, *Ragas: Automated Evaluation of Retrieval Augmented Generation* (arXiv:2309.15217, submitted Sept 26, 2023; also EACL 2024 System Demonstrations pp. 150–158, St. Julians, Malta — ACL Anthology 2024.eacl-demo.16) · *AI Engineering* Chapters 3–4.

### Day 19 — Agentic Coding Tools: Claude Code, Cursor, Codex, Copilot (15 hrs) **[MUST]**
- **Topics:** **Claude Code CLI** (npm i -g `@anthropic-ai/claude-code`) — slash commands, `/init`, `/compact`, `/clear`, `/resume`, double-escape; **CLAUDE.md** project memory (project + ~/.claude/CLAUDE.md global); **hooks** (`PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `SessionStart`, etc.) for deterministic enforcement; **subagents** in `.claude/agents/*.md` for parallel context; **plugins** + **skills**; **MCP integration** via `/mcp`; permissions model + `/sandbox` (4 hrs).
- **Topics:** **Cursor** — `.cursor/rules/*.mdc` 4 activation modes (Always Apply, Auto Attached via globs, Agent Requested via description, Manual @rule-name); Composer + multi-file edit; codebase context; `.cursorignore`; legacy `.cursorrules` migration (3 hrs).
- **Topics:** **OpenAI Codex CLI** — sandbox execution (UnixLocalSandboxClient, ApplyPatchTool), full-auto mode, multi-file changes (2 hrs); **GitHub Copilot** — code completion, chat, workspace agent, custom instructions in `.github/copilot-instructions.md` (2 hrs).
- **Decision matrix:** Claude Code for autonomous + terminal-native; Cursor for IDE/multi-file edit; Codex CLI for OpenAI ecosystem + sandbox; Copilot for inline completion + GitHub PR integration (1 hr).
- **Hands-on:** Set up Claude Code with a CLAUDE.md, a PostToolUse hook for auto-formatting, 2 subagents (test-writer, doc-writer), and 1 MCP server (Context7 or Firecrawl). Refactor Day 14's capstone with it.
- **Resources:** docs.claude.com (Claude Code) · code.claude.com/docs · docs.cursor.com · github.com/openai/codex · docs.github.com/copilot.

### Day 20 — Evaluation-Driven RAG + Beads (15 hrs)
- **Topics:** **Eval-driven development for RAG** — build the eval first, *then* ship features (3 hrs); CI integration of RAGAS (2 hrs); production monitoring with online evals (2 hrs); 🔬 statistical rigor: paired tests, bootstrapping, multiple comparison corrections (2 hrs); 🤖🔧 **Beads** (Steve Yegge) — git-native dependency graph for agent memory, `bd init`, `bd add`, `bd link` commands (2 hrs).
- **Hands-on:** Wire RAGAS into a GitHub Action that fails PRs if faithfulness drops > 5%. Set up Beads on the capstone repo.
- **Resources:** github.com/steveyegge/beads · *Introducing Beads* by Steve Yegge (Medium).

**Phase 3 milestone:** Production RAG with hybrid search + reranker + evals as CI gates. Agentic coding workflow live.

---

## Phase 4 — MCP, Observability & Agent Evaluation (Days 21–25, 75 hrs)

### Day 21 — MCP Deep Dive + Building Servers (15 hrs) **[MUST for 🤖]**
- **Topics:** MCP architecture — host → client → server, JSON-RPC 2.0 base protocol (2 hrs); **three primitives**: Tools, Resources, Prompts (2 hrs); transports — **stdio** for local, **Streamable HTTP** (specification 2025-11-25) for remote, SSE deprecated (2 hrs); OAuth 2.1 + `.well-known` discovery for remote servers; structured tool annotations (read-only vs. mutating); async operations (2 hrs); **sampling** — server-requested LLM completions (1 hr); **elicitation** — server-defined input schemas (1 hr); building servers in **Python (FastMCP)** and **TypeScript (`@modelcontextprotocol/sdk`)** (5 hrs).
- **Hands-on:** Build a custom MCP server in Python that wraps a SQL database (read-only query + schema introspection tools + table resources). Connect it to Claude Code, Cursor, and the OpenAI Responses API. Verify in all three clients.
- **Resources:** modelcontextprotocol.io · github.com/modelcontextprotocol/servers · github.com/modelcontextprotocol/python-sdk · *Complete Guide to MCP in 2026* (Dev.to).

### Day 22 — Tracing & Observability: LangSmith / Langfuse / Phoenix (15 hrs) **[MUST]**
- **Topics:** Why semantic observability is different from infra observability (1 hr); **LangSmith** — projects, traces, runs, datasets, online evaluators, prompt hub, comparing experiments, human feedback (5 hrs); **Langfuse** — open-source, OpenTelemetry-native, ClickHouse architecture, self-host requirements (Clickhouse + Redis + S3), prompt management, LLM-as-judge evals (3 hrs); **Arize Phoenix** — single-Docker self-host, OpenInference instrumentation, native RAGAS support, AX upgrade path (3 hrs); decision rubric on lock-in vs. cost vs. features (1 hr); 🔧 Pydantic Logfire as the "general observability" play (1 hr); known sharp edges: tracing pattern with `langfuse.langchain.CallbackHandler` and LangGraph interrupts can split into multiple traces — track GH issue (1 hr).
- **Hands-on:** Instrument the Day 14 capstone with all three (LangSmith primary, Langfuse + Phoenix in parallel). Confirm traces look right after a HITL interrupt + resume.
- **Resources:** docs.smith.langchain.com · langfuse.com/docs · arize.com/docs/phoenix.

### Day 23 — Agent Evaluation Beyond RAG (15 hrs) **[MUST for 🔬]**
- **Topics:** Testing non-deterministic outputs — set temperature 0 for evals, fix seeds where possible (1 hr); **deterministic evals**: exact match, schema validity, tool-call correctness, side-effect assertions (3 hrs); **LLM-as-judge evals**: pairwise comparison, score rubrics, calibration against human labels (3 hrs); **agent-trajectory evaluation** — did the agent take the right path, not just produce the right answer (3 hrs); **task-success evals** — end-to-end goal completion rate (2 hrs); evaluation dataset versioning (1 hr); 🔬 **research-grade rigor**: confidence intervals on LLM-judge scores, inter-annotator agreement, ablation discipline (2 hrs).
- **Hands-on:** Build 4 eval suites for the capstone — deterministic schema, LLM-judge faithfulness, trajectory correctness, task success. Run them in LangSmith.
- **Resources:** *AI Engineering* Chapters 3–4 · Hamel Husain's eval blog posts · LangSmith eval docs.

### Day 24 — Guardrails + Red-Teaming (15 hrs)
- **Topics:** Guardrail categories — input/output/tool/topical (1 hr); **NeMo Guardrails** — Colang programmable rails, moderation/fact-checking/hallucination rails (3 hrs); **Guardrails AI** — Pydantic + RAIL specs, validators, re-prompting (2 hrs); **Azure AI Content Safety** + Foundry Prompt Shields (2 hrs); **Lakera Guard** for adversarial defense; **Bedrock Guardrails** managed alternative (1 hr); 🔬 **red-teaming agents** — direct + indirect prompt injection, jailbreaks (many-shot, per Anthropic 2024 research), tool misuse, data exfiltration, OWASP LLM Top 10 (4 hrs); `garak` open-source scanner (1 hr); defense-in-depth pattern (1 hr).
- **Hands-on:** Run garak + your own 30-prompt injection suite against the Day 14 capstone. Add NeMo Guardrails + Lakera in front. Re-test.
- **Resources:** docs.nvidia.com/nemo/guardrails · github.com/guardrails-ai/guardrails · OWASP LLM Top 10 · Anthropic *Many-shot jailbreaking* paper.

### Day 25 — Production-grade Eval & Observability Capstone (15 hrs)
- **Build day:** Add eval-on-PR (GitHub Actions: RAGAS faithfulness + LLM-judge trajectory + 30-case red-team must pass) to the capstone repo; wire production-tier LangSmith online evals with sampling; add a dashboard (Langfuse) for cost/latency/quality; ship to a staging environment.
- **Role focus:** 🔬 leads eval design; 🔧 leads GH Actions + dashboards; 🤖 verifies prompt/agent changes.

**Phase 4 milestone:** No PR merges without passing the eval gate. Production dashboards live.

---

## Phase 5 — Cloud, Integration, CI/CD & Production (Days 26–30, 75 hrs)

### Day 26 — Microsoft Foundry (rebranded from Azure AI Foundry at Ignite Nov 18, 2025) Deep Dive (15 hrs) **[MUST — the 80/20 cloud]**
- **Topics:** Microsoft Learn (*What is Microsoft Foundry?*) describes it as "a unified Azure platform-as-a-service offering for enterprise AI operations, model builders, and application development … Microsoft Foundry unifies agents, models, and tools under a single management grouping with built-in enterprise-readiness capabilities including tracing, monitoring, evaluations, and customizable enterprise setup configurations." Natively integrated services: **Foundry Models, Foundry Agent Service, Foundry Tools, Foundry IQ (the evolution of Azure AI Search), Azure Machine Learning, Foundry Control Plane, Foundry Local** (2 hrs).
- **Topics:** Auto-upgrade from Azure OpenAI resources to Foundry resources — per *Upgrade Azure OpenAI to Microsoft Foundry*: "The Microsoft Foundry resource type provides a superset of capabilities compared to the Azure OpenAI resource type. It gives you access to a broader model catalog, agents service, and evaluation capabilities. You can upgrade your Azure OpenAI resource to a Foundry resource. You keep your existing Azure OpenAI API endpoint, state of work, and security configurations" (1 hr).
- **Topics: Foundry Agent Service** — per Microsoft Learn (*What is Microsoft Foundry Agent Service?*): "Foundry Agent Service is a fully managed platform for building, deploying, and scaling AI agents. Use any framework and many models from the Foundry model catalog. Create no-code prompt agents in the Foundry portal, or use the available SDKs and REST API to deploy them and code-based hosted agents built with Agent Framework, LangGraph, or your own code." Protocols: "the OpenResponses and Activity Protocols for Microsoft 365 publishing, an Invocations protocol for flexible endpoint integration with custom apps and services, and the A2A protocol (preview) for agent-to-agent communication" (5 hrs).
- **Topics: Hosted agents** — per *Hosted agents in Foundry Agent Service (preview)*: "Hosted agents are containerized agentic AI applications that run on Agent Service … your own code packaged as a container image." Per *Deep dive into Foundry Agent Service networking*: "Hosted agents run on Azure Container Apps and give you control over CPU and memory configuration. You deploy them through your own Azure Container Registry." (2 hrs).
- **Topics: Azure AI Search / Foundry IQ** — per Microsoft Learn (*Hybrid Search Overview*): "Hybrid search combines results from both full-text and vector queries, which use different ranking functions such as BM25 for text, and Hierarchical Navigable Small World (HNSW) and exhaustive K Nearest Neighbors (eKNN) for vectors." Per *Semantic Ranking Overview*: "In Azure AI Search, semantic ranker is a feature that measurably improves search relevance by using Microsoft's language understanding models to rerank search results. Semantic ranker is also built into agentic retrieval" (3 hrs); **Azure Container Apps** — "a serverless container platform that simplifies the deployment and scaling of microservices and AI-powered applications. With native support for GPU workloads, seamless integration with Foundry Tools" (1 hr); AKS for heavier orchestration; Bicep/Terraform for IaC; Key Vault + managed identities (1 hr).
- **Hands-on:** Deploy the Day 14 capstone agent as a Foundry Hosted Agent on Container Apps; index a 10K-doc corpus in Foundry IQ with semantic ranker; wire managed identity for Key Vault secrets.
- **Resources:** learn.microsoft.com/azure/foundry · *What is Microsoft Foundry?* · *Upgrade Azure OpenAI to Microsoft Foundry* · *Hybrid Search Overview* · *Semantic Ranking Overview*.

### Day 27 — AWS Bedrock + AgentCore, GCP Vertex AI Agent Engine (15 hrs)
- **Topics [NICE — comparative]:** **Amazon Bedrock AgentCore** (preview, 2025) — per AWS Docs (*What is Amazon Bedrock AgentCore?*): "an agentic platform for building, deploying, and operating highly effective agents securely at scale using any framework and foundation model … AgentCore services work together or independently with any open-source framework such as CrewAI, LangGraph, LlamaIndex, and Strands Agents." Per the AgentCore FAQ, the 8 primitives are "Runtime for secure serverless deployment, Memory for custom capabilities, Identity for access control, Policy for comprehensive control over agent actions, Observability for comprehensive monitoring, and Evaluations for continuous quality monitoring." Runtime details: "each user session runs in a dedicated microVM with isolated CPU, memory, and filesystem resources … supports both real-time interactions and long-running workloads up to 8 hours" and "lets agents communicate with other agents and tools via Model Context Protocol (MCP) or Agent to Agent (A2A)." Gateway: "acts as a managed Model Context Protocol (MCP) server that converts APIs and Lambda functions into MCP tools that agents can use" (4 hrs); **classic Bedrock** — Knowledge Bases, Guardrails, Flows, Agents (1 hr); **SageMaker Unified Studio** for end-to-end ML + GenAI (1 hr); **AWS Lambda** as serverless agent host + as Bedrock AgentCore Gateway tool target (1 hr); **ECS Fargate / EKS** for long-running container agents when AgentCore Runtime's 8-hr cap isn't enough; bridge to existing K8s workloads (1 hr); **OpenSearch Serverless** for vector + hybrid (BM25 + k-NN with FAISS) (1 hr).
- **Topics:** **Vertex AI Agent Engine** — per Google Cloud Docs: "Vertex AI Agent Engine (formerly known as LangChain on Vertex AI or Vertex AI Reasoning Engine) is a fully managed Google Cloud service enabling developers to deploy, manage, and scale AI agents in production"; deployment via `adk deploy agent_engine` wraps the agent in `reasoning_engines.AdkApp` and "automatically uses a VertexAiSessionService for persistent, managed session state" (3 hrs); **Cloud Run** for serverless containerized agents; **GKE** for production-scale K8s; **AlloyDB** with pgvector for managed Postgres vector search (1 hr); transition to **Gemini Enterprise Agent Platform** umbrella (1 hr); **cloud selection rubric** — Foundry default, Bedrock for AWS-native, Vertex for GCP-native; cross-cloud egress and data-residency considerations (1 hr).
- **Hands-on:** Deploy a "hello world" agent to all three — Foundry Hosted Agent (already done Day 26), Bedrock AgentCore Runtime, and Vertex AI Agent Engine. Build a cost/latency/cold-start comparison sheet.
- **Resources:** docs.aws.amazon.com/bedrock-agentcore · docs.aws.amazon.com/bedrock · docs.aws.amazon.com/sagemaker · cloud.google.com/products/agent-builder · cloud.google.com/run · cloud.google.com/alloydb.

### Day 28 — Integration: REST/gRPC/Kafka + Enterprise Connectors (15 hrs)
- **Topics:** **REST API design for agents** — idempotency, async via webhook callbacks, run/resume endpoints, SSE for streaming (3 hrs); **gRPC** for internal agent-to-agent calls (low latency, strict typing via protobuf) (2 hrs); **Apache Kafka** for event-driven agent architectures — agents as consumers, "agent topics", exactly-once semantics, schema registry (3 hrs); webhook patterns + replay (1 hr); enterprise integration — **Salesforce** (REST/SOAP + Agentforce + ForcedLeak lessons), **Jira** (REST + webhooks), **Slack** (Bolt SDK + Events API), **databases** (read-only roles, query MCP server pattern), **internal APIs** via OpenAPI → MCP Gateway (5 hrs); **A2A protocol** for cross-vendor multi-agent (1 hr).
- **Hands-on:** Add a Kafka consumer to the capstone agent that processes Jira-issue events; expose its results back via webhook.
- **Role focus:** 🔧 (heavy); 🤖 (heavy); 🔬 (light).

### Day 29 — CI/CD + DevOps for GenAI (15 hrs) **[MUST for 🔧]**
- **Topics:** **GitHub Actions for agent deployment** — matrix for multi-env, OIDC to Azure/AWS/GCP, secret management (2 hrs); **eval-on-PR gates** — RAGAS + trajectory + cost threshold; auto-comment results to PR (3 hrs); **prompt versioning** in LangSmith / Langfuse + git (2 hrs); model versioning + canary routing via LiteLLM or Vercel AI Gateway (2 hrs); **Docker** multi-stage builds for Python/TS agents (1 hr); **Kubernetes** — Deployments, HPA, ConfigMap, Secrets; **AKS** specifics (2 hrs); **Terraform** for Foundry + Container Apps + Key Vault; **Bicep** alternative (2 hrs); secrets management (Key Vault, Secrets Manager, GCP Secret Manager) (1 hr).
- **Hands-on:** Wire the capstone repo with GitHub Actions: lint → test → RAGAS eval → red-team → build container → deploy to Foundry staging. Add Terraform for the full infra.
- **Resources:** *TechWorld with Nana* K8s & Terraform tutorials · learn.microsoft.com/azure/architecture/ai-ml · github.com/Azure/bicep-registry-modules.

### Day 30 — Production Patterns + Capstone Hardening (15 hrs)
- **Topics:** **Blue/green deployments for agents** — version IDs, prompt-version pinning, dual-write evaluation (2 hrs); **canary releases** with traffic-split (1% → 5% → 25%) on metrics gates (2 hrs); **rollback strategies** — prompt rollback (LangSmith prompt hub), model rollback, full-stack rollback (2 hrs); **monitoring + alerting** — latency p95/p99, cost per session, hallucination rate, tool error rate, guardrail trip rate (3 hrs); **incident response runbook** for agent failures (1 hr); cost optimization — prompt caching, batch API (50% off Anthropic/OpenAI), model tiering (Haiku/Flash for cheap, Opus/Pro for hard) (2 hrs); **final capstone hardening + Loom demo + write-up** (3 hrs).
- **Deliverable:** Public GitHub repo with infra-as-code, eval suite, observability dashboards, deployment instructions, and a 10-minute video tour. This is the portfolio artifact.

**Phase 5 milestone:** Production-shape agent system on Microsoft Foundry, blue/green-deployable, observable, evaluatable, and cost-bounded.

---

## Role-Specific Tracks (Cumulative Time Reallocation)

| Topic | 🤖 GenAI Eng | 🔧 ML Eng | 🔬 Senior DS |
|---|---|---|---|
| LangGraph deep dive (Days 7–10) | MUST | MUST | MUST |
| Vercel AI SDK + Next.js (Day 2) | MUST | NICE | NICE |
| Microsoft Foundry deployment (Day 26) | MUST | MUST | NICE |
| Docker/K8s/Terraform/CI-CD (Days 29–30) | NICE | MUST | NICE |
| RAGAS + LLM-judge + trajectory evals (Days 18, 23) | MUST | NICE | MUST |
| Red-teaming + guardrails (Day 24) | MUST | NICE | MUST |
| MCP server building (Day 21) | MUST | MUST | NICE |
| Beads (Day 20) | NICE | NICE | NICE |
| Kafka + gRPC + enterprise connectors (Day 28) | MUST | MUST | NICE |
| OpenAI Agents SDK / ADK / AutoGen (Days 12–13) | NICE | NICE | NICE |

---

## Exhaustive Resource Library

### Official docs (bookmark all)
- **OpenAI:** developers.openai.com/api/docs · developers.openai.com/blog/responses-api · github.com/openai/openai-cookbook · openai.github.io/openai-agents-python · github.com/openai/openai-agents-python · github.com/openai/openai-agents-js
- **Anthropic:** docs.anthropic.com · platform.claude.com/docs · github.com/anthropics/anthropic-cookbook · docs.claude.com · code.claude.com/docs
- **Google:** ai.google.dev/gemini-api/docs · google.github.io/adk-docs · github.com/google/adk-python · cloud.google.com/vertex-ai/generative-ai/docs · cloud.google.com/products/agent-builder
- **LangChain/LangGraph:** docs.langchain.com · reference.langchain.com · github.com/langchain-ai/langchain · github.com/langchain-ai/langgraph · github.com/langchain-ai/langchain-academy
- **CrewAI:** docs.crewai.com · github.com/crewAIInc/crewAI · learn.crewai.com
- **MCP:** modelcontextprotocol.io · github.com/modelcontextprotocol/servers · github.com/modelcontextprotocol/python-sdk · github.com/modelcontextprotocol/typescript-sdk · blog.modelcontextprotocol.io
- **Observability:** docs.smith.langchain.com · langfuse.com/docs · arize.com/docs/phoenix · github.com/pydantic/logfire
- **Microsoft Foundry / Azure:** learn.microsoft.com/azure/foundry · learn.microsoft.com/azure/search · learn.microsoft.com/agent-framework · learn.microsoft.com/azure/container-apps · learn.microsoft.com/azure/aks
- **AWS:** docs.aws.amazon.com/bedrock · docs.aws.amazon.com/bedrock-agentcore · docs.aws.amazon.com/sagemaker · docs.aws.amazon.com/lambda · docs.aws.amazon.com/eks · docs.aws.amazon.com/opensearch-service
- **GCP:** cloud.google.com/products/agent-builder · cloud.google.com/vertex-ai · cloud.google.com/run · cloud.google.com/kubernetes-engine · cloud.google.com/alloydb
- **Vector stores:** docs.trychroma.com · weaviate.io/developers · docs.pinecone.io · github.com/pgvector/pgvector · qdrant.tech/documentation
- **Eval:** docs.ragas.io · github.com/confident-ai/deepeval · github.com/explodinggradients/ragas
- **Coding agents:** docs.claude.com · code.claude.com/docs · docs.cursor.com · github.com/openai/codex · docs.github.com/copilot

### Courses
- **DeepLearning.AI (free short courses):** *ChatGPT Prompt Engineering for Developers* · *LangChain for LLM Application Development* · *Functions, Tools and Agents with LangChain* · *Multi AI Agent Systems with crewAI* (Joao Moura) · *AI Agents in LangGraph* (Harrison Chase) · *Building Agentic RAG with LlamaIndex* · *MCP: Build Rich-Context AI Apps with Anthropic* · *Building AI Agents with Microsoft AutoGen*
- **LangChain Academy (free):** *Intro to LangGraph (Python)* · *LangGraph Essentials* · *Agent Observability with LangSmith* · *Ambient Agents with LangGraph* · *Deep Agents*
- **Coursera:** IBM *Agentic AI with LangChain and LangGraph* · DeepLearning.AI *Generative AI for Software Development* specialization
- **Udemy:** *Agentic AI Engineering with LangChain & LangGraph* (LangChain v1)
- **Google Skills / Codelabs:** *Get Started with ADK* · *Deploy ADK to Agent Engine*

### YouTube channels
- **AI Jason** — practical agent builds
- **Sam Witteveen** — LangGraph deep dives
- **James Briggs** — RAG, vector DBs, embeddings
- **TechWorld with Nana** — Kubernetes, Terraform, DevOps
- **Yannic Kilcher** — research paper deep dives (for 🔬)
- **Anthropic / OpenAI / LangChain / CrewAI** official channels for release videos

### Books (purchase week 1)
- **Chip Huyen — *AI Engineering: Building Applications with Foundation Models*** (O'Reilly, 2025). Read Chapter 1 (Day 1), Chapter 5 (Day 6), Chapter 6 (Day 15), Chapters 3–4 (Day 18 and Day 23), Chapter 10 (Day 30).
- **Chip Huyen — *Designing Machine Learning Systems*** (O'Reilly) for 🔧 readers as MLOps foundation.
- **Lewis Tunstall et al. — *Natural Language Processing with Transformers*** (O'Reilly) for 🔬 foundations.
- **Sahar Mor / Eugene Yan / Hamel Husain blog posts** (free, no book — but read all eval-related posts).

### GitHub repos to clone and study
- langchain-ai/langchain · langchain-ai/langgraph · langchain-ai/langchain-academy
- crewAIInc/crewAI
- openai/openai-agents-python · openai/openai-agents-js · openai/openai-cookbook
- anthropics/anthropic-cookbook · anthropics/claude-code
- google/adk-python
- modelcontextprotocol/servers · modelcontextprotocol/python-sdk · modelcontextprotocol/typescript-sdk
- explodinggradients/ragas
- chiphuyen/aie-book
- steveyegge/beads
- disler/claude-code-hooks-mastery (Claude Code patterns)
- sanjeed5/awesome-cursor-rules-mdc (Cursor templates)

---

## Recommendations (Staged + Decision Thresholds)

### Stage 1 — Days 1–14 (foundations + orchestration)
Ship the Day-14 mini-capstone (multi-agent LangGraph app with HITL + LangSmith + 20-case eval). **Stop and reassess if:**
- Your Day-14 demo's median cost > $0.10/call (likely prompt-caching not used → revisit Day 4).
- Median latency > 8s (likely too many sequential LLM calls → consider parallel tool use, faster models, or a smaller graph).
- Your eval suite is < 20 cases by Day 14 → you cannot reliably iterate. Do **not** proceed to Phase 3 until you have one.

### Stage 2 — Days 15–25 (RAG + observability + evals)
By Day 25 you should have: hybrid-search RAG with reranking, all three observability tools instrumented, an eval-on-PR gate, and a passing red-team suite. **Stop and reassess if:**
- RAGAS faithfulness < 0.85 on your golden set → revisit chunking + reranker (Day 17).
- Cannot trace a HITL interrupt + resume end-to-end → revisit Day 22.
- Red-team injection success > 10% → add second guardrail layer (Day 24) before production.

### Stage 3 — Days 26–30 (cloud + production)
Default to **Microsoft Foundry**. **Switch your cloud only if:**
- Your employer is AWS-native with no Azure footprint → AWS Bedrock + AgentCore (+ Lambda for short tasks, ECS/EKS for long-running, OpenSearch Serverless for vectors).
- Your employer is GCP-native or your stack depends on Gemini grounding → Vertex AI Agent Engine + ADK (+ Cloud Run / GKE / AlloyDB).
- Otherwise, Foundry's Agent Service on Container Apps + Foundry IQ + Azure OpenAI/Foundry Models gives the lowest friction to production.

### Post-Day-30 — Specialization track (next 30 days)
- 🤖 **GenAI Eng:** ship a public MCP server registry contribution; build 2 more production agent apps; learn one new framework deeply each quarter (CrewAI → OpenAI Agents SDK → ADK).
- 🔧 **ML Eng:** add fine-tuning track (LoRA/QLoRA on a 7B model, dataset engineering Chapter 8 of *AI Engineering*); pursue one cloud's certification (Azure AI Engineer Associate **AI-102** is the closest match; AWS Machine Learning - Specialty if AWS-native; Google Professional Machine Learning Engineer if GCP-native).
- 🔬 **Senior DS:** publish an internal eval framework; read 2 papers per day; double LangGraph practice.
- If you cannot build the Day-14 capstone solo, repeat Days 7–9 before Phase 3.
- If you skip the eval suite (Day 18) you will *guess* in production. Don't.

---

## Caveats

1. **Model and version churn.** Names quoted here (GPT-5.x, Claude Opus 4.7/Sonnet 4.6, Gemini 3 Pro/Flash) reflect the May 2026 landscape; expect changes every 60–90 days. Anchor learning on **APIs and patterns** (Responses API, Messages API, MCP), not specific model strings.
2. **Frameworks overlap heavily.** Pick one orchestration framework (recommend LangGraph) and master it. Resist the urge to chase every new agent framework — the marginal value drops fast after the third.
3. **Pricing volatility.** Per-token prices, free-tier limits, and especially grounding/web-search charges change frequently. Always verify against the vendor pricing page the week you build.
4. **Preview vs. GA status.** AWS Bedrock AgentCore, Foundry Hosted Agents, A2A protocol versions, and several Vertex AI services were in **preview** as of May 2026. Do not bet production SLAs on preview features without enterprise-tier support agreements.
5. **15 hrs/day is unsustainable for most humans for 30 days.** A more realistic schedule is 8–10 hrs/day for 45–60 days, with the same content. Treat the 450-hour budget as the *content*, not necessarily the calendar.
6. **The 80/20 stack is not neutral.** It biases toward Python, Anthropic/OpenAI, Microsoft/Azure, and LangChain. Engineers in JS-heavy, GCP-native, or open-source-only shops should swap the relevant days (Vercel AI SDK → Day 2; Vertex/ADK → Day 13 + Day 27; Phoenix/Langfuse → Day 22).
7. **Beads, agentic-coding hooks, and several newer tools have small user bases.** Treat them as productivity multipliers, not load-bearing dependencies for your production stack.
8. **This roadmap intentionally underweights classical ML.** If your team still does heavy tabular ML, fine-tuning, or model training, supplement with *Designing Machine Learning Systems* (Huyen) and a separate MLOps track.

---

## Completion Table

| Item from prompt | Covered? | Where |
|---|---|---|
| 30-day, 15 hrs/day, 450 hrs total | ✅ | All 30 days |
| Three role perspectives marked | ✅ | Conventions + per-day + summary table |
| Phase 1: Python (asyncio, typing, Pydantic, FastAPI, Poetry/uv) | ✅ | Day 1 |
| Phase 1: TypeScript (Node, Bun, Vercel AI SDK, Server Actions, tRPC) | ✅ | Day 2 |
| Phase 1: OpenAI (Chat Completions, Responses, Assistants, function calling, structured outputs, JSON mode) | ✅ | Day 3 |
| Phase 1: Anthropic (Messages, tool use, prompt caching, extended thinking, citations) | ✅ | Day 4 |
| Phase 1: Gemini (2.5 Pro/Flash, context caching, function calling, grounding) | ✅ | Day 5 |
| Prompt engineering (system, few-shot, CoT, ToT, ReAct, structured outputs, tool use) | ✅ | Day 6 |
| SDK comparison (OpenAI/Anthropic/google-genai/LiteLLM) | ✅ | Day 5 |
| API best practices (rate limit, retry, streaming, tokens, cost) | ✅ | Day 3 |
| Phase 2: LangChain (chains, LCEL, parsers, retrievers, tools, memory) | ✅ | Day 7 |
| Phase 2: LangGraph (StateGraph, conditional edges, checkpointers, interrupt/Command, HITL, streaming, subgraphs) | ✅ | Days 8–10 |
| Phase 2: LangGraph patterns (ReAct, tool-calling, supervisor/swarm/handoff, plan-and-execute) | ✅ | Day 9 |
| Phase 2: CrewAI (crews, processes, tools, memory, knowledge, custom LLM) | ✅ | Day 11 |
| Phase 2: CrewAI Flows (@start, @listen, @router) | ✅ | Day 11 |
| Phase 2: AutoGen / Agent Framework / OpenAI Agents SDK / Google ADK | ✅ | Days 12–13 |
| Phase 2: Comparison + production multi-agent | ✅ | Days 13–14 |
| Phase 3: RAG architecture naive → advanced → agentic | ✅ | Day 15 |
| Phase 3: Vector stores (Chroma, Weaviate, Pinecone, pgvector, Qdrant, Azure AI Search) | ✅ | Day 16 |
| Phase 3: Where RAG breaks | ✅ | Day 15 |
| Phase 3: Hybrid search + rerankers (Cohere/Jina/Voyage) | ✅ | Day 17 |
| Phase 3: Agentic coding tools (Claude Code, Cursor, Codex, Copilot) + comparison | ✅ | Day 19 |
| Phase 3: RAGAS + eval-driven RAG | ✅ | Days 18, 20 |
| Phase 4: MCP architecture, primitives, transports | ✅ | Day 21 |
| Phase 4: Building MCP servers Python + TS | ✅ | Day 21 |
| Phase 4: CLAUDE.md | ✅ | Day 19 |
| Phase 4: Beads | ✅ | Day 20 |
| Phase 4: LangSmith / Langfuse / Phoenix | ✅ | Day 22 |
| Phase 4: Agent evaluation (deterministic vs LLM-judge) | ✅ | Day 23 |
| Phase 4: Guardrails (NeMo, Guardrails AI, Azure Content Safety, Lakera) | ✅ | Day 24 |
| Phase 4: Red-teaming (injection, tool misuse, exfil) | ✅ | Day 24 |
| Phase 5: Azure (Foundry, AI Search, Container Apps, AKS) | ✅ | Day 26 |
| Phase 5: AWS (Bedrock, AgentCore, SageMaker, Lambda, ECS/EKS, OpenSearch) | ✅ | Day 27 |
| Phase 5: GCP (Vertex AI, Cloud Run, GKE, AlloyDB) | ✅ | Day 27 |
| Phase 5: API & integration (REST, gRPC, Kafka, webhooks) | ✅ | Day 28 |
| Phase 5: Enterprise (Salesforce, Jira, Slack, DBs, internal APIs) | ✅ | Day 28 |
| Phase 5: CI/CD (GH Actions, eval-on-PR, prompt + model versioning) | ✅ | Day 29 |
| Phase 5: DevOps (Docker, K8s, Terraform/Bicep, secrets) | ✅ | Day 29 |
| Phase 5: Production (blue/green, canary, rollback, monitor/alert) | ✅ | Day 30 |
| Per-day: topics, time, role markers, hands-on, resources | ✅ | Every day |
| 80/20 + must-master vs nice-to-know | ✅ | TL;DR + tier markers + role table |
| Chip Huyen *AI Engineering* book references | ✅ | Books + per-day refs |
| Required GitHub repos (langchain, langgraph, crewAI, MCP servers, anthropic-cookbook) | ✅ | GitHub repo list |
| DeepLearning.AI / LangChain Academy / Coursera courses | ✅ | Courses section |
| YouTube channels (AI Jason, Sam Witteveen, James Briggs, TechWorld with Nana) | ✅ | YouTube section |
| Tools docs URLs (docs.claude.com, docs.cursor.com, Codex, Copilot) | ✅ | Resources section |
| Observability docs (LangSmith, Langfuse, Phoenix) | ✅ | Resources section |
| Cloud docs (Azure, AWS, GCP) | ✅ | Resources section |
