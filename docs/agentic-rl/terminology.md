# Terminology and Boundaries

The term **Agentic Reinforcement Learning (Agentic RL)** is used inconsistently. This chapter fixes the vocabulary
used throughout RoseLLM and gives operational tests for ambiguous cases.

## A minimal definition

**Agentic reinforcement learning** is policy optimization from experience in
which a **Large Language Model (LLM)**-controlled agent makes temporally extended decisions in a stateful
environment and receives feedback causally downstream of those decisions.

Four properties matter:

1. **Policy dependence:** collected trajectories depend on the policy being
   optimized, even if demonstrations or replay are also used.
2. **Interaction:** an action can change subsequent observations or available
   actions.
3. **Temporal extension:** the horizon contains more than one consequential
   decision.
4. **Reward-linked update:** feedback from the episode influences the probability
   of the actions that produced it.

This definition excludes ordinary tool-call imitation and fixed preference
training, while including online and off-policy agent RL. A **Markov Decision
Process (MDP)** exposes a complete decision state; a **Partially Observable
Markov Decision Process (POMDP)** exposes observations from which the agent must
infer the hidden state.

## The neighboring concepts

| Concept | Data | Interaction during optimization | Typical feedback | Canonical mathematical view |
|---|---|---:|---|---|
| Pretraining | fixed corpus | no | next-token target | maximum likelihood |
| Supervised Fine-Tuning (SFT) / behavioral cloning | fixed demonstrations | no | demonstrated token | conditional maximum likelihood |
| Reward-model training | fixed comparisons/ratings | no | human or AI label | ranking/regression |
| Direct Preference Optimization (DPO) family | fixed or refreshed preferences | usually no environment | preferred vs rejected response | implicit-reward classification objective |
| One-response Reinforcement Learning from Human Feedback (RLHF) | policy samples | prompt then response | learned scalar reward | contextual bandit or one-step MDP approximation |
| Reinforcement Learning with Verifiable Rewards (RLVR) / reasoning RL | policy samples | often one response; may include executable calls | verifiable correctness | bandit, MDP, or search process depending on boundary |
| Agentic RL | interactive trajectories | yes | outcome, process, preference, constraint, cost | finite-horizon MDP/POMDP |
| Test-time search | samples from fixed model | possibly | verifier/judge/search score | inference algorithm; no parameter update required |
| In-context “learning” | context changes | possibly | observations/demonstrations in context | stateful inference; not gradient-based RL by itself |

## Why “multi-turn” is necessary but insufficient

A transcript can have many messages without forming an agentic training problem.
Suppose a fixed dataset contains ten-turn conversations and SFT minimizes token
cross-entropy over assistant spans. The model never changes which observation
comes next; this is still imitation from fixed data.

Conversely, one generated program can initiate many hidden simulator steps. If
the program controls a robot for one minute and reward depends on the resulting
state, the semantic action has long-horizon consequences even though the LLM
was called once. The correct horizon depends on where the policy boundary is
drawn.

## POMDP elements for an LLM agent

| Symbol | General meaning | Agent example | Common implementation mistake |
|---|---|---|---|
| $s_t$ | latent environment state | filesystem, browser DOM, user intent, permissions | treating transcript text as the complete state |
| $o_t$ | observation | tool result, screenshot, error, user reply | training on hidden evaluator state unavailable at inference |
| $h_t$ | policy information state | prompt, memory, retrieved history | silently changing summarization between train and test |
| $a_t$ | action | text, function call, shell command, click | mixing environment text with policy-generated tokens |
| $P$ | transition kernel | API execution and state change | ignoring retries, nondeterminism, or irreversible side effects |
| $R$ | reward | task success minus cost and violations | rewarding proxy compliance instead of verified outcome |
| $\rho_0$ | initial-state distribution | sampled task and sandbox snapshot | leaking evaluation task templates into training |
| $T$ | horizon | maximum turns/actions/time | allowing different budgets for compared models |
| $\gamma$ | discount | preference for earlier progress | using discounting without a time-scale interpretation |

In partial observability, the optimal action may depend on the complete action–
observation history. A transcript is one hand-engineered information state, not
proof of the Markov property. Learned memory, retrieval, and state estimators
are attempts to construct a useful approximation to a belief state.

## Four time scales

### Token scale

The policy factorizes a semantic action $a_t=(x_{t,1},\ldots,x_{t,L_t})$:

$$
\pi_\theta(a_t\mid h_t)=
\prod_{j=1}^{L_t}\pi_\theta(x_{t,j}\mid h_t,x_{t,<j}).
$$

Log-probability ratios and most gradients are computed here. Padding, loss
masks, response length, and token normalization directly affect the update.

### Turn or action scale

The environment observes a parsed semantic action. Invalid JSON, a tool call,
plain text, or an abstention can have different transition semantics even when
their token likelihoods are similar.

### Episode scale

Success, safety, and cost are often known only at termination. Credit assignment
must map this feedback to earlier turns and then to their generated tokens.

### Policy-version scale

Distributed collectors may generate different parts of the replay buffer with
different checkpoints. “On-policy” is therefore an engineering property that
requires explicit policy IDs and bounded lag, not merely the name of an
algorithm.

## Agent, scaffold, and environment

Separate these components before attributing a result:

- **Model policy:** learned conditional distribution over generated tokens.
- **Agent scaffold:** prompt assembly, planning loop, parsers, memory, routing,
  retry logic, and deterministic code around the model.
- **Tools:** functions or external services exposed as actions.
- **Environment:** state and transition logic, including users and external
  systems.
- **Evaluator:** observes some trajectory information and computes metrics or
  rewards.

An “agent improvement” can come from any component. Only a controlled ablation
can attribute it to policy training.

## Reasoning RL versus agentic RL

The categories overlap but neither contains the other.

- A model generating a long mathematical proof and receiving exact-answer reward
  is reasoning RL but can be modeled as a one-step bandit if no intermediate
  environment interaction occurs.
- A calendar assistant choosing among tools over several turns is agentic RL even
  if each choice requires little abstract reasoning.
- A code agent that edits, runs tests, reads failures, and revises is both.

The DeepSeek-R1 report is central to reasoning RL, but its published core recipe
does not by itself disclose a general production agent-training loop. Kimi k1.5
includes long-context and multimodal reasoning with RL, but each evaluated setup
must still be inspected for genuine state transitions. Case studies preserve
these boundaries rather than using “agentic” as a synonym for “capable.”

## RLVR, reward models, and verifiers

**RL with verifiable rewards (RLVR)** uses feedback computed by an executable,
formal, or rule-based checking procedure: exact math answer, unit tests, proof
kernel, game score, schema validation, or environment success state.

A **learned reward model** predicts a target such as human preference. It can
generalize beyond exact rules but introduces approximation error, distribution
shift, and exploitability.

An **LLM judge** is a learned evaluator expressed through prompting or
fine-tuning. It is still a model, not ground truth. Position, verbosity,
self-preference, prompt injection, and correlated-error controls are required.

A **process reward model** evaluates intermediate reasoning or actions. Dense
feedback can reduce variance, but only if the process label correlates causally
with robust success. Otherwise the policy learns to display rewarded-looking
steps.

## Online, off-policy, and asynchronous

- **Strict on-policy:** each update uses samples from the current policy and
  discards them after the update.
- **Near on-policy:** collectors use a recent checkpoint within a defined lag;
  importance ratios and clipping limit mismatch.
- **Off-policy:** data may come from older policies, other models, humans, or a
  replay buffer; correction or an explicitly off-policy objective is required.
- **Asynchronous:** generation and optimization overlap. This describes system
  scheduling, not automatically the statistical validity of the update.

An asynchronous PPO/GRPO system must state how it handles policy lag, version
mixing, stale log-probabilities, duplicate delivery, and samples generated while
weights change.

## Credit-assignment levels

1. **Trajectory:** every generated token receives a function of total return.
2. **Turn:** each environment action receives a separate return or advantage.
3. **Segment/subgoal:** a parser or critic assigns feedback to meaningful spans.
4. **Token:** a value estimator or dense supervision varies within an action.
5. **Component:** planner, executor, memory writer, and other policies receive
   distinct credit.
6. **Agent:** in multi-agent settings, team reward must be decomposed among
   participants.

Finer credit is not automatically better. It can introduce biased labels,
leak privileged evaluator information, and increase reward-hacking surface.

## A classification procedure

When a paper or system claims “Agentic RL,” ask in order:

1. What parameters are optimized?
2. Which distribution produced the training actions?
3. What is the environment state, and can actions change it?
4. What new observation follows each action?
5. How many consequential decisions occur before termination?
6. Where is reward computed, and what information can it access?
7. How is reward assigned across decisions and tokens?
8. How old was the behavior policy relative to the update policy?
9. Which behaviors come from learned weights versus scaffold code?
10. Does evaluation use unseen tasks, fresh state, equal budgets, and repeated
    stochastic trials?

If these questions cannot be answered, the correct classification is
“insufficiently specified,” not the most fashionable category.

## References

1. Guibin Zhang et al.,
   [“The Landscape of Agentic Reinforcement Learning for LLMs”](https://arxiv.org/abs/2509.02547),
   TMLR, 2026, especially Sections 2–3.
2. Richard S. Sutton and Andrew G. Barto,
   [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html),
   2nd ed., 2018, Chapters 3 and 13.
3. Long Ouyang et al.,
   [“Training language models to follow instructions with human feedback”](https://arxiv.org/abs/2203.02155),
   2022.
4. Rafael Rafailov et al.,
   [“Direct Preference Optimization: Your Language Model is Secretly a Reward Model”](https://arxiv.org/abs/2305.18290),
   2023.
5. Daya Guo et al.,
   [“DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”](https://arxiv.org/abs/2501.12948),
   2025.
