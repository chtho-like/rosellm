# Annotated Agentic RL Bibliography

This bibliography is a reading map, not a popularity ranking. Primary sources
are grouped by the question they answer. Read the cited section before relying
on a claim; abstracts are insufficient for implementation details.

## Foundations

1. **Sutton and Barto — *Reinforcement Learning: An Introduction*, 2nd ed.
   (2018).** The canonical route through MDPs, value functions, Monte Carlo,
   temporal difference learning, policy gradients, and function approximation.
   Read Chapters 3, 5–7, and 13 first.
   [Book and official draft](http://incompleteideas.net/book/the-book-2nd.html)
2. **Williams — “Simple Statistical Gradient-Following Algorithms for
   Connectionist Reinforcement Learning” (1992).** The score-function policy
   gradient estimator underlying modern LLM RL.
   [DOI](https://doi.org/10.1007/BF00992696)
3. **Schulman et al. — “High-Dimensional Continuous Control Using Generalized
   Advantage Estimation” (2015).** Bias–variance control for actor–critic
   advantage estimates. [arXiv](https://arxiv.org/abs/1506.02438)
4. **Schulman et al. — “Trust Region Policy Optimization” (2015).** Motivation
   for constrained policy updates and the predecessor to PPO.
   [arXiv](https://arxiv.org/abs/1502.05477)
5. **Schulman et al. — “Proximal Policy Optimization Algorithms” (2017).**
   Clipped and KL-penalized surrogate objectives. Read Equations 6–9 and the
   implementation details. [arXiv](https://arxiv.org/abs/1707.06347)

## Human feedback and preference optimization

1. **Christiano et al. — “Deep Reinforcement Learning from Human Preferences”
   (2017).** The sample–compare–reward-model–RL loop before its widespread use
   in language models. [arXiv](https://arxiv.org/abs/1706.03741)
2. **Ziegler et al. — “Fine-Tuning Language Models from Human Preferences”
   (2019).** Early preference-model and PPO experiments for language generation.
   [arXiv](https://arxiv.org/abs/1909.08593)
3. **Stiennon et al. — “Learning to summarize from human feedback” (2020).** A
   detailed large-scale language RLHF pipeline with comparison data, reward
   modeling, and policy optimization.
   [arXiv](https://arxiv.org/abs/2009.01325)
4. **Ouyang et al. — “Training language models to follow instructions with
   human feedback” (2022).** SFT → reward model → PPO, data collection, labeler
   filtering, and evaluation. [arXiv](https://arxiv.org/abs/2203.02155)
5. **Bai et al. — “Constitutional AI: Harmlessness from AI Feedback” (2022).**
   Critique/revision SFT and preference labels generated with written principles.
   [arXiv](https://arxiv.org/abs/2212.08073)
6. **Rafailov et al. — “Direct Preference Optimization” (2023).** Derives an
   offline preference loss from a KL-regularized reward maximization problem.
   [arXiv](https://arxiv.org/abs/2305.18290)

## Reasoning, process supervision, and verifiable rewards

1. **Cobbe et al. — “Training Verifiers to Solve Math Word Problems” (2021).**
   Generate many solutions and learn a verifier to select them; an important
   precursor to verifier-guided reasoning.
   [arXiv](https://arxiv.org/abs/2110.14168)
2. **Lightman et al. — “Let's Verify Step by Step” (2023).** Outcome versus
   process supervision and the PRM800K step-label dataset.
   [arXiv](https://arxiv.org/abs/2305.20050)
3. **Shao et al. — “DeepSeekMath” (2024).** Data construction for mathematical
   continued pretraining plus the original published GRPO formulation and
   training recipe. Read Sections 2–4.
   [arXiv](https://arxiv.org/abs/2402.03300)
4. **Guo et al. — “DeepSeek-R1” (2025).** R1-Zero, cold-start data, reasoning
   RL, rejection-sampled SFT, final all-scenario RL, and distillation.
   [arXiv](https://arxiv.org/abs/2501.12948)
5. **Kimi Team et al. — “Kimi k1.5” (2025).** Long-context and multimodal
   reasoning RL, rollout and curriculum choices, and short-CoT derivation.
   [arXiv](https://arxiv.org/abs/2501.12599)
6. **Yu et al. — “DAPO” (2025).** An open large-scale RL recipe with decoupled
   clipping, dynamic sampling, token-level loss, and overlong reward shaping.
   [arXiv](https://arxiv.org/abs/2503.14476)
7. **Liu et al. — “Understanding R1-Zero-Like Training” (2025).** Analysis of
   GRPO biases, response-length effects, and the Dr. GRPO correction.
   [arXiv](https://arxiv.org/abs/2503.20783)

## Agents, tools, memory, and environment interaction

1. **Nakano et al. — “WebGPT” (2021).** Browser actions, human demonstrations,
   comparison data, reward modeling, rejection sampling, and cited answers.
   [arXiv](https://arxiv.org/abs/2112.09332)
2. **Yao et al. — “ReAct” (2022).** Interleaved reasoning, action, and
   observation trajectories. Distinguish prompting/fine-tuning from online RL.
   [arXiv](https://arxiv.org/abs/2210.03629)
3. **Schick et al. — “Toolformer” (2023).** Self-labeling useful API calls and
   learning them with language-model training; not itself agentic RL.
   [arXiv](https://arxiv.org/abs/2302.04761)
4. **Shinn et al. — “Reflexion” (2023).** Verbal feedback stored in episodic
   memory without weight updates.
   [arXiv](https://arxiv.org/abs/2303.11366)
5. **Wang et al. — “Voyager” (2023).** Automatic curriculum, iterative
   executable feedback, and a skill library in Minecraft with a frozen model.
   [arXiv](https://arxiv.org/abs/2305.16291)
6. **Jin et al. — “Search-R1” (2025).** RL for interleaved reasoning and search
   with retriever observations and outcome reward.
   [arXiv](https://arxiv.org/abs/2503.09516)
7. **Feng et al. — “ReTool” (2025).** Reinforcement learning for strategic
   code-interpreter calls in mathematical reasoning.
   [arXiv](https://arxiv.org/abs/2504.11536)

## Agentic RL formalization and surveys

1. **Zhang et al. — “The Landscape of Agentic Reinforcement Learning for LLMs”
   (TMLR 2026).** Formal one-step versus POMDP distinction; capability and task
   taxonomies; environment, benchmark, and framework catalog. Use it for
   discovery, then cite original works for detailed claims.
   [arXiv](https://arxiv.org/abs/2509.02547)

## Training systems

1. **Sheng et al. — “HybridFlow: A Flexible and Efficient RLHF Framework”
   (2024/2025).** Dataflow abstraction and resource placement for distributed
   post-training; the paper corresponding to veRL.
   [arXiv](https://arxiv.org/abs/2409.19256) ·
   [repository](https://github.com/verl-project/verl)
2. **Hu et al. — “OpenRLHF” (2024/2025).** Ray, vLLM, DeepSpeed, and hybrid
   placement for scalable language-model RL.
   [arXiv](https://arxiv.org/abs/2405.11143) ·
   [repository](https://github.com/OpenRLHF/OpenRLHF)
3. **Fu et al. — “AReaL” (2025).** Asynchronous RL system design and policy-lag
   handling for large reasoning models.
   [arXiv](https://arxiv.org/abs/2505.24298) ·
   [repository](https://github.com/inclusionAI/AReaL)
4. **Luo et al. — “Agent Lightning” (2025).** Decouples agent execution from
   optimization and decomposes arbitrary trajectories for hierarchical credit.
   [arXiv](https://arxiv.org/abs/2508.03680) ·
   [repository](https://github.com/microsoft/agent-lightning)
5. **veRL Agentic RL documentation.** Exact-token generation APIs, asynchronous
   rollout servers, agent loops, and trace support. Pin a repository revision
   before using source behavior as evidence.
   [documentation](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst)

## Environments and evaluation

1. **Liu et al. — “AgentBench” (2023).** Multi-environment evaluation of LLMs as
   agents. [arXiv](https://arxiv.org/abs/2308.03688)
2. **Zhou et al. — “WebArena” (2023).** Reproducible, realistic web environments
   with functional correctness evaluation.
   [arXiv](https://arxiv.org/abs/2307.13854)
3. **Jimenez et al. — “SWE-bench” (2024).** Real repository issues and
   test-based software-engineering evaluation.
   [arXiv](https://arxiv.org/abs/2310.06770)
4. **Zhou et al. — “SWE-agent” (2024).** Agent–computer interface design for
   repository-level software engineering.
   [arXiv](https://arxiv.org/abs/2405.15793)
5. **Mialon et al. — “GAIA” (2023).** Real-world questions requiring reasoning,
   tools, browsing, and multimodal information.
   [arXiv](https://arxiv.org/abs/2311.12983)

## How to add a paper

An entry should state the precise reason to read the source, not repeat its
abstract. Prefer the author's paper, official repository, model card, or dataset
card. Record version-specific quantitative claims in the relevant chapter, not
in this discovery list. Apply the repository's
[research standard](../research-method.md) to every addition.
