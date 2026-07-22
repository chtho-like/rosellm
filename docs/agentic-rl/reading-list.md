# Agentic RL Must-Read Syllabus

**Verified through:** 2026-07-22. This page is a dependency-ordered syllabus,
not a citation-count leaderboard. It identifies the public sources a serious
Agentic Reinforcement Learning (Agentic RL) practitioner should read, the exact
sections that carry the technical content, what may be deferred, and the
artifact the reader should be able to produce afterward.

The extended discovery catalog remains the
[annotated bibliography](bibliography.md). The distinction matters:

- this page is the **course spine**;
- the bibliography is the **reference shelf**;
- the [case studies](case-studies/index.md) compare what frontier laboratories
  actually disclose;
- the [source-level lab](source-lab.md) turns the reading into code and traces.

## 1. What “all must-read sources” means

No honest static page can claim that every Agentic RL paper is mandatory. The
field changes weekly, the 2026 TMLR survey alone synthesizes more than five
hundred works, and many vendor training details remain private. Here,
**coverage-complete** means that the syllabus includes at least one strong
primary source for every dependency needed to reason about a modern system:

1. sequential-decision formalism and policy gradients;
2. Large Language Model (LLM) post-training and preference learning;
3. Reinforcement Learning with Verifiable Rewards (RLVR) and reasoning RL;
4. multi-turn agent state, actions, observations, reward, and credit;
5. tool, search, code, browser, Graphical User Interface (GUI), and multimodal
   environments;
6. synchronous and asynchronous rollout/training systems;
7. evaluation, statistical uncertainty, reward hacking, and safety;
8. industrial model recipes and their disclosure boundaries.

The list deliberately separates **Agent architecture without weight updates**
from **Agentic RL**. ReAct, Toolformer, Reflexion, and Voyager are important
because they define action and memory patterns, but none should be cited as
evidence that an online RL update occurred.

## 2. Priority and reading contract

| Priority | Meaning | Default commitment |
|---|---|---:|
| **P0 — irreducible spine** | Read before designing or reviewing an Agentic RL run. Missing it creates category or implementation errors. | 35–45 hours |
| **P1 — builder core** | Read before implementing, scaling, or diagnosing the corresponding subsystem. | 45–70 additional hours |
| **P2 — track required** | Mandatory for the selected task, laboratory, or architecture track; selective for everyone else. | 50–100 additional hours |
| **P3 — context/watch** | Historical, explanatory, product, or rapidly moving material. Read after its primary technical source. | as needed |

Priority is not quality. A P2 model report may be excellent but still be less
transferable than a P0 policy-gradient paper. A source marked **P1 for one
track** and **P2 overall** should be promoted only when that track is the
reader's active work.

For every P0/P1 source, produce a one-page reading record with six fields:

1. **learning object:** policy, critic, reward model, verifier, environment, or
   orchestration policy;
2. **sampling distribution:** which policy version generated each action;
3. **objective:** exact loss, normalization unit, clipping, and regularization;
4. **trajectory contract:** state, observation, action, termination, masks, and
   token/turn/episode clocks;
5. **evidence:** controlled ablations versus production disclosure versus
   author inference;
6. **unknowns:** missing data mixture, hyperparameters, compute, and deployment
   details.

If the record cannot be completed, the source has not yet been read deeply
enough to support implementation claims.

## 3. Dependency order

Read in this order; publication chronology is less useful:

```text
MDP/POMDP and policy gradient
    -> advantage estimation and trust-region updates
    -> LLM RLHF / RLVR data and reward pipelines
    -> GRPO-family estimator and normalization diagnostics
    -> multi-turn agent trajectory and environment contract
    -> rollout/training system and policy-version provenance
    -> benchmark reliability, reward hacking, and safety
    -> frontier-lab recipes and new algorithm variants
```

The critical conceptual break occurs between a one-response objective and a
stateful episode. A response can contain many tokens without being agentic; an
agentic episode contains actions that alter later observations. The formal
starting point is Zhang et al.,
[*The Landscape of Agentic Reinforcement Learning for LLMs*](https://arxiv.org/abs/2509.02547),
Sections 2–3.

## 4. P0 — the irreducible spine { #p0-spine }

### 4.1 Formalism, estimator, and update geometry

| Order | Source | Read exactly | Defer on first pass | Exit criterion |
|---:|---|---|---|---|
| 1 | Sutton and Barto, [*Reinforcement Learning: An Introduction*, 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html) | Chapter 3, especially 3.1–3.5; Chapters 5–7 for Monte Carlo, temporal-difference, and multi-step returns; Chapter 13, especially 13.1–13.5 | most tabular-control proofs outside Chapters 5–7 | Derive return, value, advantage, policy-gradient theorem, REINFORCE with a baseline, and actor–critic without LLM notation. |
| 2 | Williams, [“Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning”](https://doi.org/10.1007/BF00992696) | pages 229–233: the likelihood-ratio derivation, REINFORCE algorithms, and baseline discussion | associative-reward-penalty experiments | Explain why multiplying a sampled log-probability by a return gives an unbiased gradient estimator and why an action-independent baseline preserves expectation. |
| 3 | Schulman et al., [“High-Dimensional Continuous Control Using Generalized Advantage Estimation”](https://arxiv.org/abs/1506.02438) | Sections 2–4; Section 5 for value fitting | locomotion-task detail in Section 6 | Derive the Generalized Advantage Estimation (GAE) recursion and state how $\gamma$ and $\lambda$ trade bias for variance. |
| 4 | Schulman et al., [“Trust Region Policy Optimization”](https://arxiv.org/abs/1502.05477) | Sections 3–6; Appendix C for conjugate-gradient/Fisher-vector implementation | most benchmark curves and Appendix D–F | Explain the surrogate objective, average Kullback–Leibler (KL) constraint, and what PPO later approximates. |
| 5 | Schulman et al., [“Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347) | Sections 2–5, Equations 6–11, Algorithm 1; Appendix A | showcase curves in 6.3–6.4 | Implement PPO-Clip with value loss, entropy term, GAE, minibatch epochs, approximate-KL monitoring, and sign-correct clipping. |
| 6 | OpenAI, [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | “Background,” “Pseudocode,” key equations, and PyTorch parameter semantics | TensorFlow API detail | Trace every tensor in a small PPO update. This is a teaching implementation, not an LLM-RL system recipe. |

Do not begin with GRPO. GRPO removes the learned critic in a specific grouped
sampling setting; it does not remove the need to understand baselines,
importance ratios, clipping, or on-policy assumptions.

### 4.2 From RLHF to reasoning RL

| Order | Source | Read exactly | Defer on first pass | Exit criterion |
|---:|---|---|---|---|
| 7 | Ouyang et al., [“Training language models to follow instructions with human feedback”](https://arxiv.org/abs/2203.02155) | Sections 3.1–3.6; Appendices B.1–B.2 and C.1–C.4; then Section 4.1 | broad public Natural Language Processing benchmark detail in 4.2 | Draw the demonstration → comparison → reward model → PPO-ptx pipeline and identify policy, value, reward, and reference models. |
| 8 | Shao et al., [“DeepSeekMath”](https://arxiv.org/abs/2402.03300) | Sections 4.1.1–4.1.4 and 4.2; scan Sections 2–3 to preserve data lineage | most benchmark tables | Derive the published Group Relative Policy Optimization (GRPO) objective, group-relative baseline, outcome/process variants, and iterative loop. |
| 9 | Guo et al., [“DeepSeek-R1”](https://arxiv.org/abs/2501.12948) | Sections 2.1–2.4 and 4.1–4.2 | leaderboard detail in Section 3 | Separate R1-Zero from R1; reconstruct cold start, reasoning RL, rejection-sampled Supervised Fine-Tuning (SFT), all-scenario RL, and distillation. |
| 10 | Yu et al., [“DAPO”](https://arxiv.org/abs/2503.14476) | Sections 2–4, especially 3.1–3.4 and 4.1–4.3 | case examples and Appendix B | Explain decoupled clipping, dynamic sampling, token-level loss aggregation, and overlong reward shaping as distinct interventions. |
| 11 | Liu et al., [“Understanding R1-Zero-Like Training”](https://arxiv.org/abs/2503.20783) | Sections 3.1–3.4; Appendix A derivations and G experimental settings | base-model survey in Section 2 after the first reading | Reproduce the length-normalization and group-normalization biases, then derive the Dr. GRPO correction. |

The P0 output is not “GRPO is better than PPO.” It is a comparison sheet that
holds sampling budget, reward, prompt set, aggregation unit, policy freshness,
and optimizer constant while changing one estimator property at a time.

### 4.3 The agent boundary, trajectory, and system

| Order | Source | Read exactly | Defer on first pass | Exit criterion |
|---:|---|---|---|---|
| 12 | Zhang et al., [Agentic RL survey](https://arxiv.org/abs/2509.02547) | Section 2 in full; Section 3 overview; Section 5; Sections 6.1–6.5 | task-specific catalog subsections in Section 4 until choosing a track | Specify a one-step LLM-RL task and a multi-step Agentic RL task as different Markov Decision Process (MDP)/Partially Observable Markov Decision Process (POMDP) objects. |
| 13 | Luo et al., [“Agent Lightning”](https://arxiv.org/abs/2508.03680) | Sections 3.1–3.4 and Appendix A | all three application result sections on first pass | Convert an arbitrary agent trace into states, calls, rewards, trainable transitions, and hierarchical credit without coupling the agent runtime to the trainer. |
| 14 | Jin et al., [“Search-R1”](https://arxiv.org/abs/2503.09516) | Sections 3.1–3.4, 5.1, 5.4; Appendices A–B | remaining benchmark tables and long case appendices | Mark policy tokens versus retrieved observation tokens, explain why observation loss masking matters, and compare PPO with GRPO under the same search environment. |
| 15 | Sheng et al., [“HybridFlow”](https://arxiv.org/abs/2409.19256) | Sections 2.3–2.4, 4–6, and model-placement analysis in 8.3; Appendices A–C | full throughput sweep | Draw the policy/reference/critic/reward dataflow, device placement, generation/training transitions, and communication boundaries. |
| 16 | veRL, [“Agentic RL Training”](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst) | Overview; “Server-based Asynchronous Rollout”; token-based `generate` interface; multi-turn/tool-call trace flow | copy-paste usage until the concepts are clear | Explain why decode-then-retokenize can corrupt log-probabilities and why exact sampled tokens, action masks, and tool observations must survive the rollout boundary. Pin a commit before treating behavior as stable evidence. |
| 17 | Yao et al., [$\tau$-bench](https://arxiv.org/abs/2406.12045) | Sections 3–5 and Appendix B.1–B.2 | long dialogue examples in Appendices C–D | Define hidden database/user state, tool and user actions, binary validator, `pass@k` versus reliability-oriented `pass^k`, and policy-compliance gaps. |

At the end of P0, the reader should be able to audit a trajectory record with
at least these fields:

```text
episode_id, task_version, environment_version, policy_version,
prompt_tokens, sampled_action_tokens, observation_tokens, action_mask,
old_logprobs, current_logprobs, reward_components, termination_reason,
cost, latency, retry lineage, and validator output
```

## 5. P1 — algorithm and credit-assignment core

### 5.1 Preference objectives and the online/offline boundary

| Source | Read exactly | Why it is P1 | Required output |
|---|---|---|---|
| Rafailov et al., [“Direct Preference Optimization”](https://arxiv.org/abs/2305.18290) | Sections 3–5; Appendices A.1–A.4 and B | DPO is the cleanest way to understand how a KL-regularized reward objective can become an offline classification loss—and what online exploration it gives up. | Derive the DPO log-sigmoid loss and list the fixed-dataset/support assumptions that block it from replacing interactive Agentic RL. |
| Christiano et al., [“Deep Reinforcement Learning from Human Preferences”](https://arxiv.org/abs/1706.03741) | Sections 2–3 and preference-predictor experiments | Shows the segment-comparison loop before LLM RLHF. | Compare segment-level human feedback with outcome reward and process reward. |
| Stiennon et al., [“Learning to summarize from human feedback”](https://arxiv.org/abs/2009.01325) | Sections 3.1–3.4, 4.1, and 4.3; Appendices B, C.1–C.2, G.1, G.3, and G.6 | More operational reward-model and policy-training evidence than a high-level alignment diagram. | Track dataset splits, reward calibration, KL, value-function ablations, best-of-N, and evaluation without mixing them. |

### 5.2 Verifiers, process supervision, and AI feedback

| Source | Read exactly | Central question | Required output |
|---|---|---|---|
| Cobbe et al., [“Training Verifiers to Solve Math Word Problems”](https://arxiv.org/abs/2110.14168) | Sections 4.1–4.3 and 5.1–5.2; Appendices A–B | When does sampling many solutions and ranking them with a learned verifier outperform generating once? | Separate generator training, verifier training, verification score, test-time sampling, and selection bias. |
| Lightman et al., [“Let's Verify Step by Step”](https://arxiv.org/abs/2305.20050) | Sections 2.1–2.6, 3–6; Appendices B, E, and F | How do outcome-supervised reward models and process-supervised reward models differ in credit, labeling cost, and search? | Define every step boundary and label, compare ORM/PRM/majority vote, and list process-label failure modes. |
| Bai et al., [“Constitutional AI”](https://arxiv.org/abs/2212.08073) | Sections 3.1–3.5 and 4.1–4.5; Appendices C.1–C.2 | How can written principles generate critique/revision SFT data and AI preference labels for Reinforcement Learning from AI Feedback (RLAIF)? | Draw the Supervised Learning Constitutional AI (SL-CAI) and RL-CAI pipelines separately and identify where a model judgment replaces a human label. |

### 5.3 Reasoning-RL variants: read as controlled differences

| Source | Read exactly | Central delta | Do not overclaim |
|---|---|---|---|
| Kimi Team, [“Kimi k1.5”](https://arxiv.org/abs/2501.12599) | Sections 2.1–2.6; 3.3 and 3.5 | prompt curation, long-Chain-of-Thought SFT, online RL, length penalty, curriculum/sampling, partial rollouts, hybrid deployment | Its long-context and multimodal results do not prove that each technique transfers unchanged to a multi-step tool environment. |
| Cui et al., [“PRIME”](https://arxiv.org/abs/2502.01456) | Sections 2.2, 3.1–3.3, 4, and 5.1–5.4; Appendix C for reward interpretation | online implicit process reward model plus outcome reward and leave-one-out advantage | A learned dense reward is not automatically causal, calibrated, or reward-hack resistant. |
| Zhao et al., [“VAPO”](https://arxiv.org/abs/2504.05118) | Sections 2–5 | actor–critic corrections for long responses, biased value estimates, heterogeneous lengths, and sparse reward | Results belong to the paper's long-reasoning setup, not all agent horizons. |
| Zheng et al., [“GSPO”](https://arxiv.org/abs/2507.18071) | Sections 2–5, especially 4.1–4.3 | length-normalized sequence importance ratio, response-level clipping, GSPO-token stop-gradient variant | Sequence clipping and token credit are separate design choices. |
| Gao et al., [“SAPO”](https://arxiv.org/abs/2511.20347) | Sections 2–5, especially Section 3 and 4.1–4.3 | smooth sigmoid ratio gate and asymmetric temperatures for positive/negative advantages | Qwen3/Qwen3-VL evidence is author-reported transfer, not a universal replacement for clipping. |
| He et al., [“Skywork Open Reasoner 1 / MAGIC”](https://arxiv.org/abs/2505.22312) | Sections 3–7; Sections 4–5 for ablations | entropy schedule, truncation masking, sampling temperature, data difficulty, token-global aggregation, compute allocation | “No KL” is one coupled recipe result, not a standalone theorem. |
| NVIDIA, [“ProRL”](https://arxiv.org/abs/2505.24864) | Sections 2–4 and Appendix E | prolonged RL, entropy-collapse mitigation, curriculum, KL and reference/optimizer resets | Long training gains are conditional on changing the data and control schedule. |
| Wang et al., [“GiGPO”](https://arxiv.org/abs/2505.10978) | Sections 3–5; Appendices A, C–E | episode-relative plus repeated-state step-relative advantages for multi-turn agents | Reused visited states are not newly sampled counterfactual actions. |
| Hou et al., [“SAO”](https://arxiv.org/abs/2607.07508) | Sections 2–4, especially 3.1–3.2; Appendix A.1 | single-rollout asynchronous actor–critic, Direct Double-Sided Importance Sampling (DIS), value-model design, skip-observation GAE | Controlled experiments use Qwen3-30B-A3B; reported GLM-5.2 deployment does not reveal every GLM-5.2 stage or coefficient. |

Read these papers with a single comparison grid:

| Axis | Values to record |
|---|---|
| baseline | critic, group mean/std, leave-one-out, sequence group, repeated-state group |
| optimization unit | token, response, turn, segment, trajectory |
| ratio | token, length-normalized sequence, direct old/current, reference/current |
| clipping/gating | hard symmetric, hard asymmetric, sequence, smooth sigmoid, mask |
| reward | outcome, process, rule, learned judge, cost, length, language, safety |
| freshness | synchronous, minibatch reuse, bounded staleness, single-rollout asynchronous |
| normalization | per token, per sequence, per prompt group, global active token count |

Without this grid, acronym comparison easily becomes name comparison rather
than estimator comparison.

## 6. P1 — genuine multi-step Agentic RL

| Source | Read exactly | What it contributes |
|---|---|---|
| Nakano et al., [“WebGPT”](https://arxiv.org/abs/2112.09332) | Sections 2–3 and 5.1; Appendices A, C, E, and I | A complete early browser-agent loop: action space, demonstrations, comparisons, reward model, behavior cloning, rejection sampling, and RL. |
| Feng et al., [“ReTool”](https://arxiv.org/abs/2504.11536) | Sections 2.1–2.3 and 3.1–3.3 | Cold start plus RL for strategic code-interpreter calls; a compact example of tool observations embedded inside a reasoning trajectory. |
| Microsoft Research, [“rStar2-Agent”](https://arxiv.org/abs/2508.20722) | Sections 2–4; 5.3–5.4 | Concrete math-plus-Python action protocol, GRPO with resampling-on-correct, high-throughput code environment, rollout scheduler, and staged recipe including failed attempts. |
| Moonshot AI, [“Kimi K2”](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) | Sections 3.1.1, 3.2.1–3.2.3, and 3.3.1–3.3.4; Appendices B, F, G | Large-scale tool-task synthesis, real/synthetic tools, verifiable and rubric rewards, agentic rollout, colocated training, and engine switching. |
| DeepSeek-AI, [“DeepSeek-V3.2”](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf) | Section 3, especially 3.1 and 3.2.1–3.2.3; Section 4 evaluation protocol | Specialist distillation plus mixed RL, GRPO stability controls, thinking retention across tool results, real and synthetic agent tasks, and verifiers. |
| GLM-5 Team, [“GLM-5”](https://arxiv.org/abs/2602.15763) | Sections 3.2–3.6 and 4.1–4.2; 6.1.3 and 6.2 | Sequential reasoning/agent/general RL, on-policy cross-stage distillation, asynchronous infrastructure, environment scaling, context management, and real-world agentic evaluation. |
| Kimi Team, [“Kimi K2.5”](https://arxiv.org/abs/2602.02276) | Sections 2.3, 3, 4.4, 4.5; Appendix D and E.6–E.8 | Joint text/vision RL, learned parallel-agent orchestrator with frozen subagents, decoupled visual encoding, unified environment, and computer-use/swarm evaluation. |

For these reports, reconstruct four separate graphs rather than one vague
“training pipeline”:

1. **task factory:** source → synthesis → environment build → verifier → filter;
2. **runtime:** observation → policy action → tool/environment transition → new
   observation;
3. **learning:** sampled token/action → reward/return/advantage → loss → weight
   update;
4. **systems:** GPU placement → inference engine → environment workers → data
   transport → learner → weight broadcast.

## 7. P1 — systems, asynchrony, and reproducibility

| Source | Read exactly | Required systems question |
|---|---|---|
| Hu et al., [“OpenRLHF”](https://arxiv.org/abs/2405.11143) | Sections 3–4 and Appendix C | Which Ray actors own policy, reference, reward, critic, and vLLM generation; when is hybrid placement worth transition overhead? |
| Fu et al., [“AReaL”](https://arxiv.org/abs/2505.24298) | Sections 3–7; Appendix B.1–B.4 | How is staleness measured, bounded, and corrected; what throughput gain is bought with what policy lag? |
| THUDM, [slime](https://github.com/THUDM/slime) | README architecture, examples, Megatron/SGLang rollout path, data-buffer customization, and weight-sync path at a pinned commit | Where can custom agent generation and rewards run, and what state must the buffer preserve across synchronous/asynchronous modes? |
| veRL, [Agent Loop documentation](https://github.com/verl-project/verl/tree/main/docs) | Agentic RL, Agent Loop, multi-turn rollout, rollout trace, and relevant GRPO example at a pinned commit | Can one replay a trajectory token-for-token and identify policy-generated versus environment-generated positions? |
| Kimi K2, GLM-5, and SAO | Kimi K2 Sections 3.3/G; GLM-5 Sections 3.6/4.1; SAO Sections 3–4 | Compare colocated engine switching, disaggregated asynchronous generation, tail latency, fault tolerance, and off-policy correction. |

The minimum reproducibility bundle for an Agentic RL run is:

- immutable model, tokenizer, chat template, code, environment, task, and
  verifier revisions;
- exact sampled tokens and sampling log-probabilities;
- policy version and routing/sampling masks where applicable;
- action/observation/loss masks;
- retry, timeout, side-effect, and termination lineage;
- resolved hyperparameters, seeds, hardware topology, and placement;
- per-reward-component scores before scalarization;
- raw evaluation episodes plus confidence intervals.

## 8. P1 — reward failure and evaluation science

| Source | Read exactly | Why it changes practice |
|---|---|---|
| Gao et al., [“Scaling Laws for Reward Model Overoptimization”](https://arxiv.org/abs/2210.10760) | Sections 2–4, especially 3.5–3.6 and 4.1–4.3 | Proxy reward can improve while gold reward degrades; KL is an optimization-distance indicator, not a proof of alignment. |
| Denison et al., [“Sycophancy to Subterfuge”](https://arxiv.org/abs/2406.10162) | Sections 2–5; Appendix A and D | A curriculum of gameable environments can generalize from mild specification gaming to reward tampering; compare expert iteration with PPO carefully. |
| Google DeepMind, [“Specification gaming: the flip side of AI ingenuity”](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) | full illustrated essay after the two papers above | Builds reward-design intuition through concrete failures; use as a case catalog, not as algorithmic evidence. |
| Miller, [“Adding Error Bars to Evals”](https://arxiv.org/abs/2411.00640) | Sections 2–5; Appendix A for clustered errors | Agent results are stochastic and tasks can be clustered; paired evaluation and power analysis change whether a claimed gain is credible. |
| $\tau$-bench | Sections 3–5 | Mean success can hide catastrophic inconsistency; report repeated-trial reliability and policy compliance. |

An evaluation is incomplete if it reports only mean success. At minimum record
task-level paired differences, repeat count, seed/sampling configuration,
confidence interval, failure taxonomy, cost, latency, and timeout policy.

## 9. P2 — frontier technical reports by the question they answer

These reports are mandatory only for the indicated track. They are not complete
production manifests.

| Report | Priority lift | Read exactly | Question answered |
|---|---|---|---|
| Meta, [“Llama 2”](https://arxiv.org/abs/2307.09288) | P1 for classic RLHF | Sections 3.1–3.4, 4.2; Appendices A.3.1–A.3.4 and A.4.1 | How do iterative preference collection, two reward models, rejection sampling/PPO, safety reward routing, and helpfulness–safety tension interact? |
| Qwen, [“Qwen2.5-Math”](https://arxiv.org/abs/2409.12122) | P1 for math RLVR | Sections 3.1–3.3 and 4 | How are Chain-of-Thought/tool data, reward-model data, rejection sampling, online RL, and decontamination joined? |
| Qwen, [“Qwen3”](https://arxiv.org/abs/2505.09388) | P1 for hybrid thinking | Sections 4.1–4.5 and 4.7 | Why separate long-CoT cold start, reasoning RL, thinking-mode fusion, general RL, and strong-to-weak distillation? |
| Mistral AI, [“Magistral”](https://arxiv.org/abs/2506.10910) | P1 for transparent ablations | Sections 2–4, 6, and 7.4; Section 3 for asynchronous operation | What do the modified GRPO objective, compositional reward, prompt filtering, exact small-model settings, and unsuccessful methods reveal? |
| GLM Team, [“GLM-4.5”](https://arxiv.org/abs/2508.06471) | P1 for coding/search agents | Sections 3.1–3.5, especially 3.3.1–3.3.2 | How are reasoning, coding, browsing, and general specialists iterated and consolidated? |
| ByteDance Seed, [“Seed1.5-Thinking”](https://arxiv.org/abs/2504.13914) | P1 for value-based reasoning | Sections 2–5; Appendix A verifier cases | How do verifiable/non-verifiable tasks, learned/rule rewards, critic-based RL, and streaming rollouts fit together? |
| NVIDIA, [“Llama-Nemotron”](https://arxiv.org/abs/2505.00949) | P2 industrial cascade | Sections 3–6, especially 5.1–5.2 and 6.1–6.2; Section 7.1 for evaluation protocol | How can synthetic reasoning data, SFT, reasoning RL, instruction-following RL, RLHF, and serving-aware infrastructure form one product pipeline? |
| NVIDIA, [“Nemotron-Cascade”](https://arxiv.org/abs/2512.13607) | P2 sequential-domain curriculum | Sections 3–4 in full; Sections 5–7 for coding/RLHF/software-engineering analyses; Appendix D for stage hyperparameters | When does a staged RLHF → instruction → math → code → software-engineering cascade resist forgetting, and what changes at each stage? |
| Google, [“Gemini 2.5 Technical Report”](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) | P2 disclosure audit | Sections 2.4–2.6, 3.1, and 4.1; evaluation appendices relevant to the chosen task | What is disclosed about increased RL compute, verifiable/generative rewards, multi-action environments, thinking, and agentic evaluation—and what optimizer/data details remain absent? |

Cross-lab readers should compare Kimi K2/K2.5, DeepSeek-V3.2, GLM-4.5/5,
Qwen3, Magistral, Seed1.5, and Gemini 2.5 using the same stage table. Do not
compare only benchmark endpoints.

## 10. P2 — environment and benchmark protocols

Read the benchmark paper before training on or reporting its score. The reward
and reset contract is part of the learning problem.

| Track | Source | Read exactly | Audit target |
|---|---|---|---|
| broad agents | Liu et al., [“AgentBench”](https://arxiv.org/abs/2308.03688) | Sections 2–5; Appendix A evaluation framework; the appendix for the environment actually used | action validity, termination, per-environment scorer, prompt and framework effects |
| web | Zhou et al., [“WebArena”](https://arxiv.org/abs/2307.13854) | Sections 2–5; Appendices A.1–A.3, A.6, A.10 | reproducible sites, observation/action spaces, reset semantics, functional correctness, error taxonomy |
| software engineering | Jimenez et al., [“SWE-bench”](https://arxiv.org/abs/2310.06770) | Sections 2–4 and task-construction/evaluation appendices | issue-to-repository mapping, environment build, test patch, false-to-pass/pass-to-fail semantics, contamination |
| agent–computer interface | Yang et al., [“SWE-agent”](https://arxiv.org/abs/2405.15793) | Sections 2–5; Appendices A, B.3–B.5 | interface design, observation compression, commands, trajectory phases, variance, failure modes |
| general assistants | Mialon et al., [“GAIA”](https://arxiv.org/abs/2311.12983) | Sections 3–5; data card and question-design appendices | answer validation, tool/multimodal dependence, level split, private test integrity |
| desktop/GUI | Xie et al., [“OSWorld”](https://arxiv.org/abs/2404.07972) | Sections 2–5; Appendices A and B.5–B.6 | real OS state, screenshot/accessibility observations, primitive actions, initial-state setup, execution-based evaluation |
| user–tool interaction | Yao et al., [$\tau$-bench](https://arxiv.org/abs/2406.12045) | Sections 3–5; Appendix B | database state, policy document, user simulator, authorization, exact state transition, repeated-trial consistency |

For training use, add a separate environment card containing reset cost,
hidden mutable state, network/data dependencies, concurrency limit, validator
false-positive/false-negative risks, irreversible actions, and train/test
contamination barriers.

## 11. P2 — agent architecture sources that are not RL training

These are mandatory context for designing action spaces and trajectories. Their
priority must not erase their method boundary.

| Source | Read exactly | Transferable object | Boundary |
|---|---|---|---|
| Yao et al., [“ReAct”](https://arxiv.org/abs/2210.03629) | Sections 2–4; Appendix C prompts and D trajectories | interleaved thought/action/observation format and failure recovery | prompting and limited fine-tuning; not an online RL algorithm |
| Schick et al., [“Toolformer”](https://arxiv.org/abs/2302.04761) | Section 2 method/data generation, Section 3 tools, and Appendix B training procedure | self-labeling candidate API calls and filtering useful calls | language-model training on filtered calls, not environment-interactive RL |
| Shinn et al., [“Reflexion”](https://arxiv.org/abs/2303.11366) | Section 3, task Sections 4.1–4.3, relevant prompt appendices | verbal feedback and episodic memory across attempts | no policy-weight update in the core method |
| Wang et al., [“Voyager”](https://arxiv.org/abs/2305.16291) | Sections 2–3; Appendix A.1–A.6 | automatic curriculum, executable feedback, skill library, self-verification | frozen foundation model; improvement occurs through prompts/memory/skills |

## 12. P3 — blogs, documentation, and living code

### 12.1 High-value explanatory companions

| Resource | Read when | Use it for | Limitation |
|---|---|---|---|
| Hugging Face, [“The N Implementation Details of RLHF with PPO”](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) | immediately after InstructGPT and PPO | response generation, padding/position IDs, reward/value extraction, whitening, rejection sampling, optimizer differences, learning-curve reproduction | reproduces an early OpenAI-style RLHF codebase, not modern multi-step Agentic RL |
| Hugging Face, [“Open R1: Update #3”](https://huggingface.co/blog/open-r1/update-3) | after DeepSeek-R1/DAPO | code-verifiability problems, dataset construction, sample packing, learning rate, long-CoT training, open reproduction lessons | primarily distillation/SFT and project updates; do not label every result RL |
| Google DeepMind, [specification-gaming essay](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) | before reward design | concrete loophole taxonomy and the difference between maximizing specified reward and achieving intended outcome | explanatory blog, not a complete mitigation study |
| OpenAI, [“Learning to reason with LLMs”](https://openai.com/index/learning-to-reason-with-llms/) | historical/product context only | public statement that more RL and test-time computation improved o1-family reasoning | no reproducible optimizer, data mixture, reward, or system recipe |

### 12.2 Living repositories

The repositories below are P1 implementation references but P3 as *moving
documents*. Pin a revision, record it in the reading note, and distinguish the
repository's recipe from the paper or vendor's private run.

- [veRL](https://github.com/verl-project/verl): HybridFlow implementation,
  GRPO/PPO recipes, asynchronous server rollout, multi-turn agent loop, traces;
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): Ray, vLLM, DeepSpeed, hybrid
  placement, PPO-family workflows;
- [AReaL](https://github.com/inclusionAI/AReaL): asynchronous reasoning RL and
  bounded-staleness machinery;
- [Agent Lightning](https://github.com/microsoft/agent-lightning): decoupled
  agent execution, transition extraction, hierarchical credit;
- [Open-R1](https://github.com/huggingface/open-r1): public data, SFT, GRPO,
  evaluation, and code-verification integration;
- [slime](https://github.com/THUDM/slime): programmable data buffer, Megatron
  learner, SGLang rollouts, custom agent/reward generation, weight sync.

## 13. Role-specific routes

### 13.1 Algorithm researcher

1. P0 Sections 4.1–4.2.
2. DPO only to establish the online/offline boundary.
3. DAPO → Dr. GRPO → GSPO → SAPO.
4. Choose critic track: GAE/PPO → VAPO → SAO.
5. Choose critic-free/dense-credit track: PRIME → MAGIC → GiGPO.
6. Reproduce one estimator on the same frozen prompts, rollouts, rewards, and
   active-token denominator before proposing another acronym.

### 13.2 Tool/search/code Agent trainer

1. P0 in full.
2. ReAct → WebGPT → Search-R1 → ReTool.
3. Agent Lightning → GiGPO → rStar2-Agent.
4. Kimi K2 → DeepSeek-V3.2 → GLM-5.
5. Select WebArena, SWE-bench/SWE-agent, or $\tau$-bench and document the full
   environment contract before training.

### 13.3 Distributed RL systems engineer

1. PPO/GAE and exact-token provenance from P0.
2. HybridFlow → OpenRLHF → veRL Agent Loop.
3. AReaL → Kimi K2 engine switching → GLM-5 asynchronous stack → SAO.
4. Build one end-to-end trace joining generation request, policy version,
   environment calls, reward, learner minibatch, and weight broadcast.

### 13.4 Multimodal or multi-agent researcher

1. P0 plus Agent Lightning.
2. OSWorld and the chosen visual environment protocol.
3. Kimi K2.5 Sections 2.3, 3, Appendix D, E.7–E.8.
4. GiGPO Appendix E.3 for vision-language-agent transfer.
5. Separate the learned orchestrator's action/reward from frozen subagent
   outputs; “many agents ran” does not itself define multi-agent RL.

### 13.5 Evaluator, safety, or reward engineer

1. P0 environment and reward sections.
2. Reward overoptimization → specification gaming → reward tampering.
3. Adding Error Bars → $\tau$-bench → task-specific benchmark protocol.
4. Red-team both reward and validator; report exploit success separately from
   intended task success.

## 14. Twelve-week deep-reading plan

| Week | Reading | Deliverable |
|---:|---|---|
| 1 | Sutton/Barto Chapter 3 and 13; Williams | policy-gradient derivation from trajectory probability |
| 2 | GAE, TRPO, PPO, Spinning Up | tested small PPO implementation and tensor-shape note |
| 3 | InstructGPT, DPO | four-model RLHF dataflow and online/offline comparison |
| 4 | DeepSeekMath, DeepSeek-R1 | GRPO derivation and staged-R1 pipeline reconstruction |
| 5 | DAPO, Dr. GRPO | length/group/token normalization ablation plan |
| 6 | Agentic RL survey, ReAct, Agent Lightning | explicit POMDP and trace-to-transition schema |
| 7 | Search-R1, WebGPT, ReTool | tool-observation masking and reward/termination specification |
| 8 | HybridFlow, veRL, OpenRLHF | policy/reference/critic/reward placement diagram |
| 9 | AReaL, SAO, Kimi K2/GLM-5 systems sections | staleness budget and end-to-end provenance trace |
| 10 | PRIME, VAPO, GSPO/SAPO, GiGPO | one controlled algorithm-comparison matrix |
| 11 | selected benchmark plus error-bars paper | paired evaluation with intervals, cost, latency, and failure taxonomy |
| 12 | reward-overoptimization and tampering sources; one frontier report | pre-mortem, exploit tests, disclosure/unknown ledger, research proposal |

## 15. Reading tests: what competence looks like

After the syllabus, the reader should answer all of the following without
hand-waving:

1. Which random variable is the action: token, tool call, turn, or macro-step?
2. Which tokens were sampled by the policy, and which arrived from the
   environment?
3. Is the reward attached to a token, turn, subgoal, or terminal trajectory?
4. What is the baseline and over what population is it normalized?
5. What exactly is in the numerator and denominator of every importance ratio?
6. How stale can a rollout be, and how is policy version recorded?
7. What happens to truncated, timed-out, retried, malformed, or hacked episodes?
8. Which denominator is used for loss aggregation: sequences, response tokens,
   active tokens, prompts, or groups?
9. Can the validator be satisfied while violating user intent or policy?
10. Are evaluation gains paired, repeated, statistically uncertain, and net of
    cost/latency changes?
11. Is a claim disclosed, artifact-confirmed, independently reproduced,
    inferred, or unknown?
12. Would the result survive a new task distribution, environment version, and
    policy checkpoint?

## 16. Current disclosure gaps and watchlist

As of the verification date:

- **Kimi K3:** the [launch blog](https://www.kimi.com/de/blog/kimi-k3) is an
  architecture/product announcement, not an
  Agentic RL technical report. It does not disclose a reproducible RL objective,
  environment mixture, reward composition, rollout system, or post-training
  ablation. Read K2/K2.5 for the current Moonshot RL evidence, use the separate
  [K3 disclosure audit](../multimodal/kimi-k3.md), and keep K3 on the watchlist
  until a report or artifacts appear.
- **GLM-5.2:** the SAO paper states that SAO was deployed in the GLM-5.2 agentic
  training pipeline, but the controlled experiments use Qwen3-30B-A3B and the
  source does not identify every GLM-5.2 stage that used it. Read SAO for the
  algorithm and GLM-5/GLM-5.2 sources for lineage; do not fuse their details.
- **OpenAI reasoning models:** public product posts establish broad claims about
  reinforcement learning and test-time computation but do not disclose a
  reproducible production recipe. They are P3 context, not substitutes for
  DeepSeekMath/R1, DAPO, or open system papers.
- **Anthropic and Gemini production training:** public reports disclose parts
  of reward, safety, tool, and evaluation strategy, but exact agentic optimizer,
  mixtures, coefficients, and infrastructure are often unknown.
- **Repositories:** default-branch documentation changes. Any implementation
  claim must name a commit; a moving README is not timeless evidence.

The update trigger is not “a new model scored higher.” Promote a new source
into P0/P1 only when it adds a transferable formalism, controlled mechanism,
reproducible system path, or materially stronger environment/evaluation
contract.

## 17. Compact minimum route

If only thirty hours are available, read in this exact order:

1. Sutton/Barto Chapters 3 and 13;
2. GAE Sections 2–4;
3. PPO Sections 2–5 and Algorithm 1;
4. InstructGPT Sections 3.1–3.6 and Appendix C.4;
5. DeepSeekMath Section 4;
6. DeepSeek-R1 Section 2 and Section 4;
7. DAPO Sections 2–4;
8. Dr. GRPO Section 3 and Appendix A;
9. Agentic RL survey Sections 2, 5, and 6;
10. Agent Lightning Section 3;
11. Search-R1 Sections 3 and 5.4;
12. HybridFlow Sections 2.3–2.4 and 4–6;
13. veRL Agentic RL documentation;
14. $\tau$-bench Sections 3–5;
15. the reward-overoptimization and error-bars papers.

Then choose exactly one task track and one industrial report. Breadth without a
trajectory schema, estimator derivation, and evaluation artifact is not yet
working knowledge.
