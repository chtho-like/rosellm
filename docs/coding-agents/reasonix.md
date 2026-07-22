# Reasonix: History, Architecture, and Coding-Agent Comparison

**Verified through:** 2026-07-22.

## Executive finding

Reasonix is an independent, MIT-licensed coding-agent project owned by
`esengine`; it is not a DeepSeek-AI repository or an official DeepSeek client.
It began on 2026-04-21 as a TypeScript/Ink experiment whose differentiator was
the economics and protocol behavior of DeepSeek models. On 2026-05-29, its
maintainers created a second, unrelated Git root and imported a completed Go
rewrite that had temporarily been developed as `duo`. The current default
branch is this Go generation, released as Reasonix 1.x.

The shortest fair characterization is:

> Reasonix started as a DeepSeek-specific cache experiment and evolved into a
> configurable Go agent platform that retains DeepSeek-oriented defaults,
> reasoning-field handling, and cache-stable session design.

That is more precise than either “DeepSeek-only wrapper” or “fully neutral
multi-provider framework.” Its provider registry can use OpenAI-compatible
endpoints without code changes, but its identity and several architectural
choices still come from DeepSeek's prefix-cache economics.

## Revisions inspected locally

The repositories were cloned under `/Users/mac/projects/github` and inspected
at these revisions. The path records the local audit snapshot; it is not a
portable installation requirement.

| Project | Local directory | Revision inspected | Default branch at inspection |
|---|---|---|---|
| Reasonix | `deepseek-reasonix` | `c53c13c178ba9d60b8ce600c24810cda0563282a` | `main-v2` |
| OpenCode | `opencode` | `0a601cf334b9a83cc2854108a2b860f25e6e7e8e` | `dev` |
| Pi Coding Agent | `pi` | `dd6bea41efa8caa7a10fe5a6401676dc5699f83f` | `main` |
| OpenAI Codex | `openai-codex` | `963cda85aa2a4cfb85e52d771d22d9f3069951fa` | `main` |
| Claude Code | `claude-code` | `ac062f33ab0ca7c62b9df648d0f2027fa9b969f0` | `main` |

“Picode” is ambiguous. This study interprets it as **Pi Coding Agent**, whose
former `badlogic/pi-mono` GitHub location redirects to the current
`earendil-works/pi` repository and whose executable remains `pi`. A different
project named Picode would require a separate comparison.

The existing `/Users/mac/projects/github/codex` directory was preserved because
it was a source snapshot rather than a Git checkout. The fresh OpenAI repository
therefore uses `openai-codex`.

## Part I — How Reasonix began

### 1. The repository contains two histories, not one continuous rewrite

`git rev-list --max-parents=0 --all` returns two roots:

| Root | Date | Meaning |
|---|---|---|
| `e9e7ccc62237e0b6dca70849c3f5647ad639edb7` | 2026-04-21 | TypeScript v0.0.1 lineage |
| `32a4c02e5446bb38eaf8efb7592bef3d249ea195` | 2026-05-29 | clean-slate Go generation |

This matters. A normal rewrite retains parentage, so blame/log can connect new
code to earlier decisions. Reasonix used an **orphan branch**: the Go root has
no parent. The next commit says that the already-finished implementation was a
“code-only import” and that its upstream `duo` history was not included. The
public repository can therefore establish the import event and the resulting
architecture, but it cannot reconstruct every design iteration made while
`duo` was being developed. That interval has evidence class **U — Unknown**.

### 2. v0.0.1 was a thesis, not a generic product

The first commit described three pillars:

1. **Cache-First Loop.** Keep an immutable prompt prefix, append new turns and
   tool results, and isolate volatile state. DeepSeek's automatic prompt cache
   rewards exact repeated prefixes; changing earlier bytes turns what could be
   cached input into newly processed input.
2. **R1 Thought Harvesting.** Parse DeepSeek's separate `reasoning_content`
   stream and carry selected planning signals into agent state.
3. **Tool-call repair.** Recover malformed or truncated tool calls, flatten
   nested arguments, scavenge partial calls, and stop repeated call storms.

The initial stack was Node.js 20+, TypeScript, ECMAScript modules, React, and
Ink. Its README explicitly listed multi-agent orchestration, retrieval-augmented
generation, multiple providers, and a web service as non-goals. The original
identity was therefore unusually narrow: make DeepSeek inexpensive and robust
inside long tool-using sessions, not build a universal coding-agent platform.

### 3. The first benchmark caused a public thesis correction

On the launch day, an 18-run “harvest” experiment compared a baseline,
DeepSeek reasoner, and reasoner-plus-harvesting across six math/logic tasks. The
maintainer reported that the baseline and ordinary reasoner passed all six,
while the harvesting mode had one timeout. The commit explicitly rejected the
claim that harvesting improved answer accuracy and reframed its possible value
as introspection or tool-sequence steering.

This is historically important for two reasons:

- the project did not preserve all three original pillars as equally validated
  product claims; and
- cache reuse, not harvested reasoning, remained the strongest measured part of
  the initial thesis.

The experiment is a project-authored artifact, not an independent reproduction.
Its tiny task set cannot establish general coding quality, but it can establish
what claim the project itself decided **not** to make.

### 4. The TypeScript line rapidly expanded beyond its original non-goals

The legacy log shows a compressed product-discovery cycle:

| Date | Public-history milestone | Interpretation |
|---|---|---|
| 2026-04-21 | transcript replay/diff, MCP client, `reasonix code` edit workflow | observability and tools arrived immediately |
| 2026-04-22 | project memory and Plan Mode | long-lived project state and role separation began |
| 2026-04-23 | subagents through skills | the initial multi-agent non-goal was relaxed |
| 2026-04-26 | feature catch-up against Codex, Claude Code, and Cursor; semantic search | competitive product breadth became explicit |
| 2026-04-28 | web dashboard | the initial “no web UI” boundary was relaxed |
| May 2026 | many 0.x releases, a Rust-TUI experiment, then return to Ink | the team explored packaging and interaction architecture quickly |
| 2026-05-29 | v0.54.x era and Go orphan root | incremental evolution ended in a replacement kernel |

This is not hypocrisy; non-goals often describe a release boundary rather than
a permanent constitution. But it means the v0.0.1 README must not be used as a
description of current Reasonix.

### 5. The Go rewrite reset implementation lineage

The Go root says:

- kernel rewritten from TypeScript to Go;
- web client removed at the reset;
- desktop client rebuilt;
- legacy TypeScript retained separately.

Twenty minutes later, commit `7de6a247...` imported 92 files and about 13,480
inserted lines from `duo`, renamed every `duo` identifier to Reasonix, and
described a registry core, OpenAI-compatible provider, built-in tools, MCP over
standard input/output and HTTP, permissions, sandboxing, custom commands,
references, and two-model collaboration. Both rewrite commits include a
Claude-Opus co-author trailer. That trailer is evidence of credited assistance,
not evidence that the model independently designed or authored the entire
rewrite.

Reasonix 1.0.0 was marked released on 2026-06-02. In the migration guide,
“v1” means the legacy **codebase generation** even though it used 0.x semantic
versions, while “v2” means the Go generation that uses 1.x releases. Branch
labels and semantic versions therefore cross:

```text
legacy generation: branch v1    -> releases 0.x
Go generation:     branch main-v2 -> releases 1.x
```

That naming is a frequent source of confusion.

## Part II — The current Reasonix architecture

### 1. System topology

The present system is better pictured as a shared controller and event model
behind several interfaces, not as a terminal script wrapped around one API:

```text
CLI/TUI       Wails desktop       VS Code / ACP       HTTP/SSE workbench
    \               |                  |                    /
     +-------- transport-neutral controller and typed events --------+
                                  |
                         Agent or Coordinator
                         /                  \
                 Provider registry       Tool registry
                        |                 /     |      \
              model endpoint       built-ins  MCP   subagents/skills
                                            |
                                  permissions -> sandbox -> host OS

             session JSONL / archives / history / memory / compaction
```

The architecture document states an acyclic dependency direction from command
surfaces into agent/plugin/config, then into tools/providers. This makes the
agent loop reusable by terminal, HTTP streaming, Agent Client Protocol (ACP),
desktop, and remote-workbench surfaces. The visible clients do not each
implement an independent agent.

### 2. Provider abstraction: extensible, but deliberately small

At the core is a Go `Provider` interface with a streaming method. Factories are
registered by provider **kind**, while a configuration instance supplies name,
base URL, model, API key, and extra options. The built-in `openai` kind speaks
the OpenAI-compatible chat-completions protocol. A new compatible vendor is a
configuration entry rather than a source-code branch.

This creates two limits:

- it is genuinely no longer hard-locked to DeepSeek; but
- compatibility is initially protocol-shaped. A provider with semantics that
  cannot be represented through this protocol or its extra fields needs a new
  implementation, transformations, or a gateway.

Reasonix also round-trips model reasoning fields where supported. This is more
than displaying hidden thoughts: DeepSeek-style `reasoning_content` may need to
be serialized back with the assistant turn so a following request remains
valid and cache-compatible. Model-specific behavior therefore survives beneath
the generic provider interface.

### 3. The agent loop

The engineering specification describes the loop as:

```text
build request with tool schemas
  -> Provider.Stream
  -> emit text/reasoning deltas and accumulate tool-call deltas by index
  -> if there is no complete tool call, finish
  -> otherwise permission-check and execute each complete call
  -> append assistant call plus matching tool result
  -> repeat up to maxSteps
```

Incomplete streaming fragments are not exposed to the tool executor. Tool
errors become observations for model self-correction rather than fatal process
errors. A `context.Context` carries cancellation into provider and tool work.

This loop is conceptually conventional; Reasonix's differentiation lies in the
state invariants around it.

### 4. Prefix-cache stability as a state invariant

Autoregressive providers compute key/value states for the request prefix. A
server-side prefix cache can reuse that work only when a sufficiently long
prefix is identical under the provider's cache-key rules. Agent scaffolds often
damage reuse by rewriting system prompts, reordering tool schemas, inserting
volatile clocks/status, or repeatedly summarizing early history.

Reasonix's response is architectural:

- keep standing instructions and environment information stable and
  deterministic;
- canonicalize tool schemas and avoid needless ordering changes;
- append messages and tool results rather than editing old turns;
- shorten old tool results deterministically before asking a model to summarize;
- make summary compaction infrequent and observable as a cache-reset point;
- keep planner and executor in separate sessions so switching models does not
  mutate one shared prefix.

This does **not** mean a conversation can grow forever without cache misses.
Model context windows are finite, provider caches can evict entries, and a
summary necessarily changes the subsequent request. It means the runtime tries
to make invalidation low-frequency and intentional.

### 5. Tiered context management

The inspected specification exposes three approximate pressure stages:

1. At the tool-result snip ratio (default 0.6 of configured context), old large
   results are replaced by deterministic head/tail forms while message pairing
   remains intact.
2. At the compaction ratio (default 0.8), stale tool results are archived and
   pruned to placeholders; only if the request is still too large does a model
   produce a digest.
3. At the force ratio (default 0.9), a fold may proceed even when ordinary fold
   economics would skip it.

Recent results and errors are preferentially retained. User briefs below the
pin budget and previous digests survive the fold verbatim; dropped originals
are archived as JSON Lines. A BM25-based `history` tool can search saved session
transcripts and archives, while a separate `memory` tool searches durable facts.

There are three different notions here:

- **active context:** bytes sent on the next model request;
- **durable transcript/archive:** data retained for traceability or resumption;
- **retrievable memory/history:** data that can be searched back into a later
  turn.

Conflating them leads to false claims such as “nothing is ever forgotten” or
“the whole transcript is always in context.”

### 6. Two-model collaboration is session separation, not expert routing

When `planner_model` differs from the executor, `Coordinator` maintains two
independent histories:

- a lower-frequency planner receives standing memory and a filtered read-only
  research tool set, then emits a concise plan;
- the full executor receives that plan as structured text and performs writes
  and commands through the normal agent loop.

This is application-level orchestration. It is unrelated to Mixture-of-Experts
routing inside a neural network and does not jointly train the models. Its
specific cache advantage comes from never swapping two providers inside one
conversation prefix.

### 7. Tools and MCP

Built-in tools implement a small interface: name, description, JSON Schema,
and execute. A per-run registry combines enabled built-ins with external tools.
External plugins are MCP servers reached through:

- a persistent local subprocess over JSON-RPC standard input/output;
- Streamable HTTP, including Server-Sent Events responses; or
- the legacy HTTP-plus-SSE transport.

MCP tools are namespaced and their `readOnlyHint` annotations influence
parallel dispatch and default permission classification. Reasonix explicitly
warns that those annotations are assertions by an installed server, not a
containment boundary against a malicious server. Project-discovered MCP
configuration requires an identity-specific launch confirmation, whereas an
explicit user installation is itself an authorization decision.

This distinction is sound: **tool metadata**, **permission policy**, and
**process containment** are separate trust layers.

### 8. Permission is not sandboxing

Reasonix uses both:

- a per-call policy returns allow, ask, or deny, with deny taking precedence;
  and
- a lower enforcement layer confines filesystem and shell effects where the
  operating system provides a backend.

At the inspected revision, documentation describes macOS Seatbelt
(`sandbox-exec`) and Linux bubblewrap for shell confinement; Windows lacks an
equivalent built-in operating-system shell sandbox. In enforcement mode, a
missing backend causes shell execution to fail closed rather than silently run
unconfined. File tools also apply workspace/read/write root rules.

Approval protects intent; sandboxing limits effects after approval or model
error. Neither completely neutralizes prompt injection, compromised build
scripts, malicious MCP servers, or credentials deliberately placed inside an
allowed boundary.

### 9. Why the project still calls itself DeepSeek-native

“Native” here is a runtime-optimization claim, not ownership or model lineage.
It refers mainly to:

- design around DeepSeek prefix-cache behavior and cached-input economics;
- explicit reasoning-stream handling and round-tripping;
- model presets, pricing, and defaults oriented toward DeepSeek;
- cache observability and separate-session orchestration.

The current configuration-driven provider layer means “DeepSeek-native” no
longer means “DeepSeek-only.” A more exact label would be **DeepSeek-first,
OpenAI-compatible multi-provider harness**. Whether another provider receives
equal quality depends on protocol fidelity, tool-call behavior, reasoning-field
semantics, cache rules, and maintainer testing—not merely whether its endpoint
accepts the request.

## Part III — Comparison with OpenCode, Pi, Codex, and Claude Code

### High-level matrix

| Dimension | Reasonix | OpenCode | Pi Coding Agent | OpenAI Codex | Claude Code |
|---|---|---|---|---|---|
| primary design center | DeepSeek-efficient long sessions | broad provider platform and clients | minimal, user-modifiable harness | OpenAI-native local/cloud coding system | Anthropic-native integrated product |
| current core | Go | TypeScript/Bun with Effect-based components | TypeScript packages | Rust workspace | proprietary core; public plugins/artifacts |
| provider posture | OpenAI-compatible registry, DeepSeek-first | broad provider/model normalization | 15+ providers and custom providers | OpenAI/Codex service and Responses semantics | Claude models and Anthropic service |
| topology | shared controller behind CLI/desktop/ACP/web | HTTP server + OpenAPI/SDK + clients | library/CLI/RPC/SDK in one small stack | local core + app-server + IDE/app/cloud | terminal/IDE/GitHub/remote product |
| default product policy | plan/executor, permissions, sandbox, memory | build/plan agents, subagents, permissions | intentionally omits built-in plan/subagent policy | approvals, native OS sandbox, skills/MCP/multi-agent | permissions, sandbox, subagents, hooks/plugins |
| extension center | TOML, built-ins, MCP, skills/hooks | plugins, MCP, skills, agents, server API | trusted in-process TypeScript extensions | skills, MCP, plugins/connectors, app-server protocol | skills, agents, hooks, MCP, plugins/marketplaces |
| state emphasis | append-oriented cache stability, archives, retrieval | durable server/session state; V2 moving toward SQLite admission | tree-structured JSONL sessions and branching | typed threads/turns/events across clients | resumable/checkpointed sessions by documented behavior |
| built-in containment | policy + macOS/Linux shell sandbox; Windows limitation | permission rules; implementation-specific process controls | no built-in sandbox by design | deep cross-platform system sandbox and approval model | documented permission/sandbox system; core not auditable |
| open-source boundary | core open under MIT | core open under MIT | core open under MIT | local runtime open under Apache-2.0 | public repository all-rights-reserved, core absent |

### Public-origin chronology

The earliest reachable Git commit and the creation time of the current GitHub
repository are different facts. A repository can import or merge history older
than the repository object itself, as OpenCode demonstrates.

| System | Current GitHub repository created | Earliest relevant public root/milestone | Caution |
|---|---|---|---|
| Claude Code | 2025-02-22 | research-preview repository root, 2025-02-22 | root is a product page, not core source |
| OpenAI Codex | 2025-04-13 | TypeScript CLI root, 2025-04-16 | Rust arrived in parallel eight days later |
| OpenCode | 2025-04-30 | reachable Go root, 2025-03-21 | older and multiple roots reflect synchronized/merged history |
| Pi | 2025-08-09 | general `pi` monorepo root, 2025-08-09 | coding-agent package appeared in October |
| Reasonix | 2026-04-21 | TypeScript v0.0.1 root, 2026-04-21 | Go generation adds a second root on May 29 |

These dates establish public artifact lineage, not the first private prototype,
company decision, or idea. Those earlier events are unknown unless the
maintainers disclose them.

### 1. OpenCode: a platform first, not a model-economics experiment

OpenCode's visible history begins with a Go codebase on 2025-03-21, before the
current GitHub repository's creation date. Its history includes synchronizations
and merges from other roots, a phase combining a Go Bubble Tea TUI with a
JavaScript/TypeScript server, and a 2025-11-02 commit that removed the Go TUI.
The current default development branch is a TypeScript/Bun monorepo.

Current OpenCode starts a server even when the user launches the TUI. The TUI
is a client, the HTTP endpoint is described by OpenAPI, and generated SDKs let
desktop, web, IDE, and programmatic clients share the same backend. That
client/server boundary is more central than in Reasonix, whose specification
starts from a compact Go kernel and transport-neutral controller.

OpenCode also treats provider breadth as a first-class product problem. Its
source contains extensive model-specific normalization around reasoning,
caching, schemas, and provider options. This breadth is not “free neutrality”:
the project carries compatibility code precisely because vendors differ.

Its default agent concepts include full build and read-only plan roles, general
subagents, fine-grained permission rules, MCP, skills, Language Server Protocol
integration, and plugin-defined tools. At the inspected `dev` revision, an
ongoing V2 session architecture separates durable prompt admission from model
execution, uses a process-global per-session execution coordinator, and scopes
model/tool/permission services to a workspace location. These are current
development-branch contracts, not guaranteed stable-release behavior.

**Practical difference from Reasonix:** choose OpenCode when broad provider
choice, multiple clients, server automation, and ecosystem scale are primary.
Choose Reasonix when a small static Go distribution and explicit
DeepSeek/cache-oriented behavior are primary. OpenCode can optimize caching,
but cache stability is not its founding invariant.

### 2. Pi Coding Agent: a substrate with intentionally less built-in policy

Pi's history did not start with a coding agent. The 2025-08-09 root created a
three-package monorepo containing terminal UI, generic agent, and GPU-pod
functionality. A coding-agent work-in-progress appeared on 2025-10-17; the
package became `@mariozechner/pi-coding-agent` in November. In 2026 the project
moved to Earendil Works, shed unrelated pod/messaging packages, and renamed its
published package scope to `@earendil-works`.

The current stack is deliberately layered:

- `pi-ai` normalizes model providers;
- `pi-agent-core` exposes a generic event-driven loop and state;
- `pi-tui` supplies terminal components; and
- `pi-coding-agent` assembles the coding product.

Its loop distinguishes an outer queue of user follow-ups from an inner
assistant/tool loop and exposes event streams, steering, optional parallel tool
execution, and a `convertToLlm` boundary. The product supports interactive,
print/JSON, RPC, and SDK modes. JSONL session records form a tree using message
and parent identifiers, so branching does not require copying a linear chat.

Pi's strongest difference is philosophical. It intentionally skips built-in
subagents and plan mode, expecting extensions or packages to implement the
workflow a user wants. Extensions are trusted TypeScript modules running in the
same process with the user's permissions. Official security documentation is
explicit that project trust only controls whether project-local settings and
extensions load; it is **not a sandbox**. Real isolation must come from a
container, virtual machine, micro-VM, or an extension that redirects execution.

**Practical difference from Reasonix:** Pi is the cleaner choice for embedding,
teaching, or inventing a bespoke agent policy. Reasonix supplies more policy
and containment out of the box. Pi's smaller core is flexibility, not an
unfinished attempt to replicate every Claude Code feature.

### 3. OpenAI Codex: a systems-engineered OpenAI agent stack

Codex CLI began as TypeScript on 2025-04-16. Only eight days later OpenAI
imported an initial Rust implementation, explicitly citing standalone binaries,
lower overhead, and direct access to Linux sandbox APIs. The implementations
coexisted while Rust caught up; the TypeScript implementation was removed on
2025-08-08.

The current Rust workspace has a typed session/turn protocol, an asynchronous
turn loop, and a `ToolRouter` that separates model-visible specifications from
runtime handlers. Tools can declare parallel-call support. A large app-server
JSON-RPC boundary exposes threads, turns, streaming events, configuration,
skills, MCP, plugins/connectors, feedback, and other product operations to
clients. This makes Codex both a local agent and a platform component behind
IDE and desktop experiences.

Codex's most visible architectural specialization is system safety. The open
runtime contains native sandbox paths for macOS and Linux and Windows-specific
isolation work, combined with approval and permission profiles. OpenAI's
first-party architecture material describes the CLI as a cross-platform local
agent and the app as reusing configurable system-level sandboxing.

Its model layer is vertically integrated with OpenAI account/API authentication,
Responses API items, Codex model behavior, and optional hosted/cloud services.
That coupling can support model/runtime co-design, but it is not the provider
neutrality offered by OpenCode or Pi.

**Practical difference from Reasonix:** Codex has a deeper open local systems
surface, stronger cross-platform sandbox investment, and tighter model/product
co-design. Reasonix is easier to point at arbitrary OpenAI-compatible providers
and uniquely foregrounds cached-input economics. Neither repository opens the
frontier model weights or every hosted backend component.

### 4. Claude Code: compare behavior, not unavailable internals

Anthropic's public repository began on 2025-02-22 as a research-preview landing
page. Its initial README said the preview was intended to learn how developers
collaborate with Claude and listed reliability, long-running commands, and
terminal rendering as improvement areas. The repository did not contain the
CLI implementation then, and it does not contain the proprietary core now.
Its current license states “all rights reserved” and points to Anthropic's
commercial terms.

The public tree is still useful: it contains plugins, agents, commands, skills,
hooks examples, a development container, issue tooling, and a detailed
changelog. First-party documentation describes:

- separate-context subagents with selectable models, tools, permission modes,
  skills, MCP servers, and lifecycle hooks;
- foreground and background execution;
- declarative allow/ask/deny rules and several permission modes;
- hooks around tool, session, and subagent lifecycle events;
- MCP, skills, plugins, marketplaces, checkpoints/rewind, IDE and remote
  surfaces.

Those are **D — Disclosed product contracts**. They are not **C — Confirmed core
implementation artifacts**. Claims about Claude Code's private scheduler,
prompt assembly, exact compaction algorithm, internal persistence schema, or
sandbox implementation must remain unknown unless Anthropic documents them.

**Practical difference from Reasonix:** Claude Code is a polished,
Claude-specific, vertically integrated product with a broader disclosed
workflow surface. Reasonix is source-auditable, model-configurable, and exposes
its cache/context mechanics. A source-level claim about Reasonix can often be
traced to Go; the analogous Claude Code claim usually stops at documentation or
observed behavior.

## Part IV — What the comparison does and does not prove

### Common but incorrect conclusions

| Incorrect shortcut | Correct interpretation |
|---|---|
| “DeepSeek-native means DeepSeek made it.” | It describes optimization posture; repository ownership is independent. |
| “OpenAI-compatible means all providers behave identically.” | Request shape may match while reasoning fields, tool calls, caching, limits, and errors differ. |
| “Two models means multi-agent intelligence.” | Reasonix's coordinator is two isolated sessions with a text handoff; capabilities depend on roles, tools, and models. |
| “Project trust is a sandbox.” | Trust controls loading of local configuration/code; sandboxing constrains runtime effects. |
| “Permission prompts make execution safe.” | Approval and containment cover different failure modes and both have gaps. |
| “The project with more files or stars is better.” | Community size and code surface are not controlled task-quality evaluations. |
| “Claude Code's GitHub repository is its source.” | It is a public product/plugin repository; the core remains proprietary. |
| “An open CLI means the whole product is open.” | Models, inference, account services, cloud runners, and control planes can remain closed. |

### No task-quality winner is established here

This study did not run the five agents on an identical commit, model, prompt,
tool policy, budget, and sandbox. It therefore does not claim that one produces
better patches. A defensible evaluation would control at least:

- exact model checkpoint and provider;
- context/tool schemas and repository instructions;
- token, wall-clock, and monetary budgets;
- permission and network policy;
- number of retries/subagents and human interventions;
- clean starting repository and deterministic tests;
- patch correctness, regression rate, security, latency, and cost.

Provider-neutral harnesses make model control easier but not perfect because
their prompts and protocol transforms still differ. Vendor-native agents may
use private model features unavailable through a common endpoint.

## Decision guide

| If the dominant requirement is... | First system to examine | Why |
|---|---|---|
| minimize DeepSeek long-session cached-input cost and inspect that behavior | **Reasonix** | cache stability is a named invariant with separate planner/executor sessions |
| use many vendors through one product with TUI, desktop, web, and server APIs | **OpenCode** | provider normalization and client/server architecture are central |
| embed a small agent loop or invent a custom workflow in TypeScript | **Pi** | layered SDK and trusted extensions, intentionally minimal policy |
| use OpenAI models with strong local sandboxing and connected local/cloud surfaces | **Codex** | model/runtime co-design, Rust core, typed app-server, native isolation |
| use Claude with Anthropic's full integrated workflow and accept a closed core | **Claude Code** | deep Claude product integration, subagents/hooks/plugins/remote features |

For high-risk unattended work, the deciding question should often be the
containment boundary rather than the user interface. Pi's own documentation
recommends external isolation; Reasonix documents platform gaps; Codex and
Claude Code still require careful filesystem, network, secret, and approval
configuration.

## Primary sources

### Reasonix

1. GitHub API, [repository metadata](https://api.github.com/repos/esengine/deepseek-reasonix), verified 2026-07-22.
2. esengine, [initial TypeScript commit](https://github.com/esengine/deepseek-reasonix/commit/e9e7ccc62237e0b6dca70849c3f5647ad639edb7), 2026-04-21.
3. esengine, [harvest benchmark correction](https://github.com/esengine/deepseek-reasonix/commit/fb5977743c3c1e75e3e42fc5887b474c0b8cb20b), 2026-04-21.
4. esengine, [Go orphan-root rewrite](https://github.com/esengine/deepseek-reasonix/commit/32a4c02e5446bb38eaf8efb7592bef3d249ea195) and [`duo` code import](https://github.com/esengine/deepseek-reasonix/commit/7de6a2474fb913f1b94480e671e85276deb46270), 2026-05-29.
5. Reasonix, revision-pinned [README](https://github.com/esengine/deepseek-reasonix/blob/c53c13c178ba9d60b8ce600c24810cda0563282a/README.md), [engineering specification](https://github.com/esengine/deepseek-reasonix/blob/c53c13c178ba9d60b8ce600c24810cda0563282a/docs/SPEC.md), and [migration guide](https://github.com/esengine/deepseek-reasonix/blob/c53c13c178ba9d60b8ce600c24810cda0563282a/docs/MIGRATING.md).

### OpenCode and Pi

6. GitHub API, [OpenCode metadata](https://api.github.com/repos/anomalyco/opencode); OpenCode, [server architecture](https://opencode.ai/docs/server/) and [agent/permission documentation](https://opencode.ai/docs/agents/), verified 2026-07-22.
7. OpenCode history, [initial visible root](https://github.com/anomalyco/opencode/commit/4b0ea68d7af9a6031a7ffda7ad66e0cb83315750), [server/TUI synchronization](https://github.com/anomalyco/opencode/commit/f3da73553c45f17e04b1e77cb13eb0fca714d1bd), and [Go TUI removal](https://github.com/anomalyco/opencode/commit/f68374ad2223ddc213bdea9519ca6a699819ee0e).
8. OpenCode, revision-pinned [session source](https://github.com/anomalyco/opencode/tree/0a601cf334b9a83cc2854108a2b860f25e6e7e8e/packages/opencode/src/session), [provider transforms](https://github.com/anomalyco/opencode/blob/0a601cf334b9a83cc2854108a2b860f25e6e7e8e/packages/opencode/src/provider/transform.ts), and [tool source](https://github.com/anomalyco/opencode/tree/0a601cf334b9a83cc2854108a2b860f25e6e7e8e/packages/opencode/src/tool).
9. GitHub API, [Pi metadata](https://api.github.com/repos/earendil-works/pi); Pi, [product and philosophy](https://pi.dev/), [security boundary](https://pi.dev/docs/latest/security), and [extension model](https://pi.dev/docs/latest/extensions), verified 2026-07-22.
10. Pi history, [initial monorepo](https://github.com/earendil-works/pi/commit/a74c5da112c29466f182a03108337a488c785d76), [coding-agent start](https://github.com/earendil-works/pi/commit/ffc9be88679426fff067eae3831e399c4b0b7651), [package naming](https://github.com/earendil-works/pi/commit/79ee33c3fc8d23aa75695dfc6d9826991b065d77), and [Earendil Works migration](https://github.com/earendil-works/pi/commit/551385e40946d5531e823edea533410c816926ce).
11. Pi, revision-pinned [agent loop](https://github.com/earendil-works/pi/blob/dd6bea41efa8caa7a10fe5a6401676dc5699f83f/packages/agent/src/agent-loop.ts) and [coding-agent package](https://github.com/earendil-works/pi/tree/dd6bea41efa8caa7a10fe5a6401676dc5699f83f/packages/coding-agent).

### Codex and Claude Code

12. GitHub API, [OpenAI Codex metadata](https://api.github.com/repos/openai/codex); OpenAI, [Codex agent-loop engineering article](https://openai.com/index/unrolling-the-codex-agent-loop/) and [Codex app sandbox description](https://openai.com/index/introducing-the-codex-app/), verified 2026-07-22.
13. OpenAI Codex history, [initial TypeScript commit](https://github.com/openai/codex/commit/59a180ddec4adaf9760972cdb1eb89f06a81be8b), [Rust import and rationale](https://github.com/openai/codex/commit/31d0d7a305305ad557035a2edcab60b6be5018d8), and [TypeScript removal](https://github.com/openai/codex/commit/408c7ca142689136d887676def1bf41ea80bb2a9).
14. OpenAI Codex, revision-pinned [turn loop](https://github.com/openai/codex/blob/963cda85aa2a4cfb85e52d771d22d9f3069951fa/codex-rs/core/src/session/turn.rs), [tool router](https://github.com/openai/codex/blob/963cda85aa2a4cfb85e52d771d22d9f3069951fa/codex-rs/core/src/tools/router.rs), and [app-server protocol](https://github.com/openai/codex/blob/963cda85aa2a4cfb85e52d771d22d9f3069951fa/codex-rs/app-server/README.md).
15. GitHub API, [Claude Code metadata](https://api.github.com/repos/anthropics/claude-code); Anthropic, [initial public commit](https://github.com/anthropics/claude-code/commit/bd5ca708adf82c4b81857abf40fe36d9d9cc3d1c) and revision-pinned [license](https://github.com/anthropics/claude-code/blob/ac062f33ab0ca7c62b9df648d0f2027fa9b969f0/LICENSE.md).
16. Anthropic, [how Claude Code works](https://code.claude.com/docs/en/how-claude-code-works), [subagents](https://code.claude.com/docs/en/sub-agents), [permission modes](https://code.claude.com/docs/en/permission-modes), and [extension features](https://code.claude.com/docs/en/features-overview), verified 2026-07-22.

## Unresolved questions

- The omitted `duo` history prevents a complete causal reconstruction of the Go
  rewrite. Only the imported result and commit explanation are public.
- Reasonix's project-authored cache case studies need independent reproduction
  under fixed providers, cache retention, workloads, and prices.
- OpenCode's current `dev` branch is actively migrating session architecture;
  release behavior must be checked against a tagged version before operational
  deployment.
- Cross-harness quality and cost remain unevaluated under controlled models and
  permissions.
- Claude Code's internal loop, prompt assembly, persistence, scheduler, and
  exact sandbox implementation remain unavailable for source audit.
