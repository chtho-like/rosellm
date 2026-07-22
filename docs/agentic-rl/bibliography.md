# Annotated Agentic RL Bibliography

This bibliography is a reading map, not a popularity ranking. Primary sources
are grouped by the question they answer. Read the cited section before relying
on a claim; abstracts are insufficient for implementation details.

For a dependency-ordered course with P0–P3 priorities, exact sections,
deferrable material, role-specific routes, time budgets, and required reading
artifacts, use the [Agentic RL must-read syllabus](reading-list.md). This page
remains the wider reference shelf.

The scope is **Agentic Reinforcement Learning (Agentic RL)** for **Large
Language Models (LLMs)**.

## Foundations

1. **Sutton and Barto — *Reinforcement Learning: An Introduction*, 2nd ed.
   (2018).** The canonical route through Markov Decision Processes (MDPs),
   value functions, Monte Carlo, temporal difference learning, policy
   gradients, and function approximation.
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
   for constrained policy updates and the predecessor to Proximal Policy
   Optimization (PPO).
   [arXiv](https://arxiv.org/abs/1502.05477)
5. **Schulman et al. — “Proximal Policy Optimization Algorithms” (2017).**
   Clipped and Kullback-Leibler (KL)-penalized surrogate objectives. Read
   Equations 6–9 and the implementation details.
   [arXiv](https://arxiv.org/abs/1707.06347)

## Human feedback and preference optimization

1. **Christiano et al. — “Deep Reinforcement Learning from Human Preferences”
   (2017).** The sample–compare–reward-model–RL loop before its widespread use
   in language models. [arXiv](https://arxiv.org/abs/1706.03741)
2. **Ziegler et al. — “Fine-Tuning Language Models from Human Preferences”
   (2019).** Early preference-model and PPO experiments for language generation.
   [arXiv](https://arxiv.org/abs/1909.08593)
3. **Stiennon et al. — “Learning to summarize from human feedback” (2020).** A
   detailed large-scale language Reinforcement Learning from Human Feedback
   (RLHF) pipeline with comparison data, reward modeling, and policy
   optimization.
   [arXiv](https://arxiv.org/abs/2009.01325)
4. **Ouyang et al. — “Training language models to follow instructions with
   human feedback” (2022).** Supervised Fine-Tuning (SFT) → reward model → PPO,
   data collection, labeler filtering, and evaluation.
   [arXiv](https://arxiv.org/abs/2203.02155)
5. **Bai et al. — “Constitutional AI: Harmlessness from AI Feedback” (2022).**
   Critique/revision SFT and preference labels generated with written principles.
   [arXiv](https://arxiv.org/abs/2212.08073)
6. **Rafailov et al. — “Direct Preference Optimization” (2023).** Direct
   Preference Optimization (DPO) derives an offline preference loss from a
   KL-regularized reward maximization problem.
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
   continued pretraining plus the original published Group Relative Policy
   Optimization (GRPO) formulation and training recipe. Read Sections 2–4.
   [arXiv](https://arxiv.org/abs/2402.03300)
4. **Guo et al. — “DeepSeek-R1” (2025).** R1-Zero, cold-start data, reasoning
   RL, rejection-sampled SFT, final all-scenario RL, and distillation.
   [arXiv](https://arxiv.org/abs/2501.12948)
5. **Kimi Team et al. — “Kimi k1.5” (2025).** Long-context and multimodal
   reasoning RL, rollout and curriculum choices, and short
   Chain-of-Thought (CoT) derivation.
   [arXiv](https://arxiv.org/abs/2501.12599)
6. **Yu et al. — “DAPO: An Open-Source LLM Reinforcement Learning System at
   Scale” (2025).** Decoupled Clip and Dynamic sAmpling Policy Optimization
   (DAPO) is an open large-scale RL recipe with decoupled clipping, dynamic
   sampling, token-level loss, and overlong reward shaping.
   [arXiv](https://arxiv.org/abs/2503.14476)
7. **Liu et al. — “Understanding R1-Zero-Like Training” (2025).** Analysis of
   GRPO biases, response-length effects, and the Dr. GRPO correction.
   [arXiv](https://arxiv.org/abs/2503.20783)
8. **Hou et al. — “Single-Rollout Asynchronous Optimization for Agentic
   Reinforcement Learning” (2026).** Single-rollout asynchronous actor–critic,
   direct double-sided importance masking, faster/frozen-attention critic
   updates, and skip-observation Generalized Advantage Estimation (GAE). The
   authors report deployment in Zhipu AI's General Language Model 5.2
   (GLM-5.2), while controlled experiments use Qwen3-30B-A3B.
   [arXiv](https://arxiv.org/abs/2607.07508)
9. **Zheng et al. — “Group Sequence Policy Optimization” (2025).** Group
   Sequence Policy Optimization (GSPO) replaces token importance ratios with a
   length-normalized sequence ratio, clips at response level, derives equal
   token-gradient weighting, and introduces the GSPO-token stop-gradient
   construction for customized token credit.
   [arXiv](https://arxiv.org/abs/2507.18071)
10. **Gao et al. — “Soft Adaptive Policy Optimization” (2025).** Soft Adaptive
    Policy Optimization (SAPO) uses a sigmoid ratio surrogate with a smooth
    gradient gate and asymmetric positive/negative-advantage temperatures; the
    paper reports controlled Qwen3 experiments and use in Qwen3-VL training.
    [arXiv](https://arxiv.org/abs/2511.20347)

## Frontier model-family and industrial recipes

These sources are especially useful because they expose more than a benchmark
table. None is a complete production run manifest; use the case-study evidence
labels before transferring a hyperparameter from one model or stage to another.

1. **Touvron et al. — “Llama 2” (2023).** A detailed dense pretraining recipe
   followed by SFT, reward-model collection, rejection sampling, and PPO. Read
   it to understand how safety and helpfulness reward models, iterative online
   preference data, reward routing, and the safety reward model's auxiliary
   loss interact.
   [arXiv](https://arxiv.org/abs/2307.09288)
2. **Yang et al. — “Qwen2.5 Technical Report” (2024).** Documents a broad
   general/specialist synthetic-data engine and the model-family transition
   from pretraining through offline and online post-training.
   [arXiv](https://arxiv.org/abs/2412.15115)
3. **Yang et al. — “Qwen2.5-Math Technical Report” (2024).** Gives an
   unusually operational math-data pipeline, reward-model construction,
   rejection-sampling stage, and GRPO-based online RL stage.
   [arXiv](https://arxiv.org/abs/2409.12122)
4. **Qwen Team — “Qwen3 Technical Report” (2025).** Separates long-CoT cold
   start, reasoning RL, thinking-mode fusion, and general RL, making it useful
   for studying staged capability acquisition rather than one undifferentiated
   post-training job. [arXiv](https://arxiv.org/abs/2505.09388)
5. **Mistral AI — “Magistral” (2025).** Discloses a modified GRPO objective,
   compositional reward, prompt filtering, asynchronous trainer/generator/
   verifier operation, exact small-model settings, and negative ablations.
   [arXiv](https://arxiv.org/abs/2506.10910)
6. **GLM Team — “GLM-4.5” (2025).** Connects a 23T-token
   mixture-of-experts (MoE) base to reasoning, coding, browsing, and agent
   training, with task/environment construction and specialist consolidation
   details.
   [arXiv](https://arxiv.org/abs/2508.06471)
7. **GLM Team — “GLM-5” (2026).** Describes the Token-In, Token-Out transport,
   generated software environments, sequential reasoning/agent/general RL,
   anti-hacking, and parallel on-policy distillation used in a large sparse
   model. [arXiv](https://arxiv.org/abs/2602.15763)
8. **Moonshot AI — “Kimi K2” (2025).** Couples a trillion-parameter MoE,
   MuonClip pretraining, factories for real and synthetic tools, executable
   environments, multi-source rewards, and high-concurrency code sandboxes.
   [technical report](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)
9. **Moonshot AI — “Kimi K2.5” (2026).** Covers continual multimodal
   pretraining, text-only agent SFT, joint text/vision RL, efficiency curricula,
   and Parallel-Agent Reinforcement Learning for a learned orchestrator with
   frozen subagents. [arXiv](https://arxiv.org/abs/2602.02276)
10. **DeepSeek-AI — “DeepSeek-V3.2” (2025).** Reports tens of thousands of
    agent tasks, more than 1,800 environments, specialist RL, mixed
    consolidation, and a post-training budget exceeding ten percent of the
    stated pretraining cost. [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)
11. **ByteDance Seed — “Seed1.5-Thinking” (2025).** Combines value-based
    long-chain-of-thought RL, rule and learned verifiers, mixed objective
    domains, a streaming rollout system, and concrete cold-start/data
    operations. [arXiv](https://arxiv.org/abs/2504.13914)
12. **Zhao et al. — “VAPO” (2025).** A value-based long-reasoning recipe that
    targets critic bias, heterogeneous sequence lengths, sparse reward, and
    stable actor-critic optimization. [arXiv](https://arxiv.org/abs/2504.05118)
13. **NVIDIA — “Llama-Nemotron” (2025).** Shows a multi-stage industrial route
    through architecture transformation, distillation, synthetic reasoning
    data, SFT, preference learning, instruction RL, and reasoning RL.
    [arXiv](https://arxiv.org/abs/2505.00949)
14. **NVIDIA — “ProRL” (2025).** Studies more than two thousand verifiable-RL
    steps and the role of KL control, curricula, reference resets, and optimizer
    resets after apparent convergence. [arXiv](https://arxiv.org/abs/2505.24864)
15. **NVIDIA — “Nemotron-Cascade” (2025).** Trains domains sequentially through
    SFT, preference, instruction, math, code, and software-engineering stages;
    useful for comparing cascades with one unified task mixture.
    [arXiv](https://arxiv.org/abs/2512.13607)
16. **Microsoft Research — “rStar2-Agent” (2025).** Exposes a concrete
    math-plus-Python action protocol, SFT mixture, online difficulty filtering,
    resample-on-correct group construction, three RL stages, hardware, and
    paper-to-code differences. [arXiv](https://arxiv.org/abs/2508.20722)
17. **Google DeepMind — “Gemini 2.5 Technical Report” (2025).** Publicly names
    verifiable and generative rewards, complex multi-action environments,
    increased RL compute, and RL from human and critic feedback, while leaving
    the optimizer and exact mixture undisclosed.
    [technical report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)

## Open reproduction and diagnostic artifacts

1. **Hugging Face — Open-R1.** A moving, inspectable integration of public
   reasoning data, distillation, SFT, GRPO, evaluation, and code sandboxes.
   Pin a revision and call the result an Open-R1 recipe, not a reproduction of
   DeepSeek's private run. [repository](https://github.com/huggingface/open-r1)
2. **Guha et al. — “OpenThoughts” (2025).** More than one thousand data
   experiments over math, code, and science sources show how question choice,
   teacher sampling, verification, and mixture design affect a distilled
   reasoner. It is supervised distillation, not online RL.
   [arXiv](https://arxiv.org/abs/2506.04178)
3. **Cui et al. — “Process Reinforcement through Implicit Rewards” (2025).**
   Process Reinforcement through IMplicit rEwards (PRIME) learns a causal
   token-level reward model online from trajectory-level outcomes, then combines
   its leave-one-out process advantage with the outcome advantage.
   [arXiv](https://arxiv.org/abs/2502.01456)
4. **He et al. — “Skywork Open Reasoner 1” (2025).** The Multi-stage Adaptive
   entropy scheduling for GRPO In Convergence (MAGIC) recipe treats entropy,
   sequence budget, truncation, data difficulty, policy freshness, and
   token-global aggregation as jointly controlled training state.
   [arXiv](https://arxiv.org/abs/2505.22312)
5. **Wang et al. — “Group-in-Group Policy Optimization” (2025).** Reuses
   repeated environment states across sampled trajectories to form a second,
   step-relative comparison group without drawing new counterfactual actions.
   [arXiv](https://arxiv.org/abs/2505.10978)
6. **THUDM slime.** A programmable data-buffer architecture joining a Megatron
   learner with SGLang rollout servers, custom agent/reward generation, weight
   synchronization, and synchronous/asynchronous placement choices.
   [repository](https://github.com/THUDM/slime)

Reusable distributed trainers, including Agent Lightning and OpenRLHF, are
cataloged separately under [Training systems](#training-systems).

## Agents, tools, memory, and environment interaction

1. **Nakano et al. — “WebGPT” (2021).** Browser actions, human demonstrations,
   comparison data, reward modeling, rejection sampling, and cited answers.
   [arXiv](https://arxiv.org/abs/2112.09332)
2. **Yao et al. — “ReAct” (2022).** Interleaved reasoning, action, and
   observation trajectories. Distinguish prompting/fine-tuning from online RL.
   [arXiv](https://arxiv.org/abs/2210.03629)
3. **Schick et al. — “Toolformer” (2023).** Self-labeling useful Application
   Programming Interface (API) calls and learning them with language-model
   training; not itself agentic RL.
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
   (*Transactions on Machine Learning Research*, TMLR 2026).** Formal one-step
   versus Partially Observable Markov Decision Process (POMDP) distinction;
   capability and task taxonomies; environment, benchmark, and framework
   catalog. Use it for discovery, then cite original works for detailed claims.
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
6. **Xie et al. — “OSWorld” (2024).** A real computer environment with
   screenshot/accessibility observations, primitive computer actions,
   reproducible initial states, and execution-based evaluators.
   [arXiv](https://arxiv.org/abs/2404.07972)
7. **Yao et al. — “$\tau$-bench” (2024).** Stateful tool–agent–user interaction
   with hidden database state, policies, a user simulator, rule-based terminal
   evaluation, and the repeated-trial reliability metric `pass^k`.
   [arXiv](https://arxiv.org/abs/2406.12045)

## Reward failure and evaluation science

1. **Gao et al. — “Scaling Laws for Reward Model Overoptimization” (2022).**
   Measures proxy-reward optimization against a synthetic gold reward and
   separates Reinforcement Learning (RL), best-of-N, KL distance, and several
   Goodhart mechanisms. Read Sections 2–4.
   [arXiv](https://arxiv.org/abs/2210.10760)
2. **Denison et al. — “Sycophancy to Subterfuge” (2024).** Trains on a
   curriculum of increasingly gameable environments and tests generalization
   toward reward tampering. Read Sections 2–5 and Appendix D.
   [arXiv](https://arxiv.org/abs/2406.10162)
3. **Miller — “Adding Error Bars to Evals” (2024).** Independent versus
   clustered questions, variance reduction, paired model comparison, and
   prospective power analysis. Read Sections 2–5.
   [arXiv](https://arxiv.org/abs/2411.00640)

## Explanatory and implementation companions

These sources can make a primary paper operational, but they do not supersede
its evidence.

1. **OpenAI — Spinning Up PPO documentation.** A compact derivation,
   pseudocode, implementation, and hyperparameter interface for classical
   actor–critic PPO. It is an educational continuous-control implementation,
   not an LLM Agentic RL recipe.
   [documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
2. **Hugging Face — “The N Implementation Details of RLHF with PPO” (2023).**
   A reproduction-oriented audit of response generation, padding, position
   identifiers, reward/value extraction, normalization, rejection sampling,
   and optimizer differences in an early language-model PPO stack.
   [blog](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
3. **Hugging Face — “Open R1: Update #3” (2025).** Code-data construction,
   verifiability failures, sample packing, learning-rate and long-reasoning
   lessons from an open reproduction effort. Much of the reported work is
   distillation or SFT rather than online RL.
   [blog](https://huggingface.co/blog/open-r1/update-3)
4. **Google DeepMind — “Specification gaming: the flip side of AI ingenuity”
   (2020).** An illustrated case catalog for reward loopholes and the gap
   between a specified proxy and intended outcome.
   [blog](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)

## How to add a paper

An entry should state the precise reason to read the source, not repeat its
abstract. Prefer the author's paper, official repository, model card, or dataset
card. Record version-specific quantitative claims in the relevant chapter, not
in this discovery list. Apply the repository's
[research standard](../research-method.md) to every addition.
