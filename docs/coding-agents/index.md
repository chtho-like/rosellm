# Coding-Agent Systems: Architecture and Evidence Map

**Verified through:** 2026-07-22. This section is based on revision-pinned
source trees, repository history, and first-party product documentation. It
does not treat popularity, a product demonstration, or a model-generated
description as evidence of implementation architecture.

A coding agent is not just a model. It is a system that repeatedly constructs a
model request, exposes tools, interprets tool calls, changes an environment,
persists state, and decides when to ask a person or stop. The same model can
behave very differently inside two harnesses; the same harness can expose
different behavior when its model, prompt, permission policy, or tools change.

## Keep four objects separate

| Object | Examples | What it determines |
|---|---|---|
| **base or post-trained model** | DeepSeek, Claude, GPT, Kimi | token prediction, reasoning and tool-call priors, context limits, supported modalities |
| **agent harness/runtime** | Reasonix kernel, Pi Agent, Codex core | request loop, tool dispatch, context policy, retries, state, cancellation |
| **interaction surface** | terminal user interface (TUI), desktop app, IDE extension, web client | how a person starts, observes, steers, approves, and resumes work |
| **hosted product/control plane** | account service, cloud task runner, remote control, collaboration service | identity, remote execution, policy, synchronization, telemetry, managed deployment |

“Open-source coding agent” may refer only to the local runtime. It does not imply
that the model weights, hosted inference service, account system, or cloud
orchestrator are open. Conversely, a repository that contains a changelog,
plugins, and issue templates may not contain the product's core loop at all.

## The generic loop

```text
user or client input
  -> controller admits a turn
  -> assemble stable instructions, history, context, and tool schemas
  -> provider/model streaming request
  -> text/reasoning/tool-call events
  -> permission decision and optional human approval
  -> sandboxed or ordinary tool execution
  -> append observations and update durable state
  -> repeat until completion, cancellation, or a step limit
```

Every implementation must make choices at each arrow:

- Does state live in memory, JSON Lines, SQLite, or a service?
- Is the provider abstraction broad, OpenAI-compatible only, or tied to one
  vendor's protocol and model behavior?
- Are tool results appended exactly, shortened deterministically, summarized by
  a model, or retrieved later?
- Does “permission” mean a policy prompt, an operating-system sandbox, project
  trust, or all three?
- Are planner and executor roles prompts in one session, independent model
  sessions, subprocesses, or remote workers?
- Is the terminal the product boundary, or one client of a reusable server?

## Comparison axes used in this section

| Axis | Questions to ask |
|---|---|
| **origin and lineage** | Was the current code evolved in place, imported, rewritten, or kept closed? |
| **model coupling** | Which provider protocols and model-specific fields are first-class? |
| **control topology** | Is it a single process, client/server platform, embeddable library, or hosted system? |
| **tool runtime** | How are schemas registered, calls streamed, results paired, and parallelism controlled? |
| **context and persistence** | What remains verbatim, what is compacted, and what can be resumed or branched? |
| **extension boundary** | Config, in-process code, Model Context Protocol (MCP), plugins, hooks, or an application programming interface (API)? |
| **safety boundary** | Policy decision, user approval, project trust, process sandbox, filesystem/network confinement? |
| **auditability** | Is the core implementation present under an open-source license, or only documented behavior visible? |

## Snapshot of the five systems

| System | Public lineage at the verified revision | Main implementation surface | Model posture | Core-source visibility |
|---|---|---|---|---|
| **Reasonix** | TypeScript 0.x experiment -> orphan-root Go 1.x rewrite | Go kernel; TUI, desktop, VS Code/Agent Client Protocol, web/remote workbench | DeepSeek-optimized, now configurable through OpenAI-compatible providers | MIT-licensed core |
| **OpenCode** | early Go/Bubble Tea line -> composite repository -> TypeScript/Bun platform | HTTP server plus generated SDK and TUI/desktop/web clients | broad multi-provider support is a primary design goal | MIT-licensed core |
| **Pi Coding Agent** | general `pi` monorepo -> coding-agent package -> Earendil Works agent harness | small TypeScript package stack, CLI, Remote Procedure Call (RPC), and software development kit (SDK) modes | broad multi-provider model layer | MIT-licensed core |
| **OpenAI Codex** | TypeScript CLI -> early Rust parallel implementation -> Rust replacement | Rust agent/core and app-server protocol; CLI, IDE, app, and cloud surfaces | OpenAI/Codex-native | Apache-2.0 local core; hosted services/models separate |
| **Claude Code** | Anthropic research preview -> vertically integrated product | terminal, IDE, GitHub, remote and plugin surfaces | Claude/Anthropic-native | public repository excludes the proprietary core |

The table is a topology map, not a quality ranking. Repository age, commit
count, stars, and exposed file count are not controlled measurements of coding
ability or reliability.

## Reading path

1. Read the [Reasonix history and architecture reconstruction](reasonix.md).
   It follows both Git roots, the discarded claims, the Go import boundary, and
   the present cache-oriented agent loop.
2. Use its comparison sections to contrast Reasonix with OpenCode, Pi Coding
   Agent, OpenAI Codex, and Claude Code along the axes above.
3. Continue to [Agentic Reinforcement Learning](../agentic-rl/index.md) when the
   question changes from “how does the runtime execute a model?” to “how is a
   model trained from interactive trajectories?”
4. Use the [multimodal architecture map](../multimodal/index.md) when image,
   video, audio, or media generation enters the model input/output path.

## Evidence boundary

- **D — Disclosed:** a maintainer's commit, documentation, or product page says
  it directly.
- **C — Confirmed artifact:** the checked-out source or license directly
  contains it.
- **I — Inferred:** a conclusion combines disclosed facts and source structure;
  assumptions are stated.
- **U — Unknown:** public evidence cannot establish the claim.

The detailed study pins source claims to the revisions inspected on
2026-07-22. OpenCode's default development branch and fast-moving product
changelogs may change immediately afterward. Claude Code's closed core limits
its comparison to documented behavior and public artifacts; it cannot receive
the same source-level architectural confidence as the other four.
