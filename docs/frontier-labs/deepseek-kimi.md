# DeepSeek 与 Moonshot AI / Kimi：全研究谱系、技术路线与证据边界

**核验截止：2026-07-19（Asia/Shanghai）。** 本章以仓库在该日冻结的
DeepSeek 与 Moonshot AI / Kimi 文献清单、发现日志、已归档全文和一手研究页面为证据。
它覆盖清单中的 **69 条书目记录（inventory records）**：DeepSeek 37 条，Moonshot/Kimi 32 条；其中
核心模型/技术报告 35 条、直接方法/系统/评测研究 20 条、明确带机构署名的广义研究
14 条。这里的“全”只表示**相对于这 69 条、按本文纳入规则形成的来源宇宙逐条有交代**，
不表示互联网上再无漏检论文，也不把只有权重、代码或简短更新说明的仓库伪装成论文。

## 1. 怎么读：证据标签、范围和五个先行结论

本章沿用 [Research, Evidence, and Citation Standard](../research-method.md)：

- **[D] Disclosed / 已披露**：论文、技术报告、官方模型卡或一手研究页直接陈述。
- **[C] Confirmed artifact / 已由公开制品确认**：发布的配置、权重、代码或清单可直接观察。
- **[R] Reproduced / 已复现**：本仓库按可审计协议复跑；本章**没有**把任何结果标成 [R]。
- **[I] Inferred / 推断**：由列明的证据和假设推出，来源本身没有这样宣称。
- **[U] Unknown / 未知**：公开证据缺失、含混、冲突，或不足以支持结论。

清单的三种归属层级也不可混淆：`core` 是以实验室旗舰或品牌模型为中心的一手报告；
`direct` 是实验室直接产出的架构、优化、系统、评测研究；`affiliated` 只要求至少一位作者
有明确机构署名，或由实验室官方托管，但**不能据此推断该工作进入过旗舰模型**。

五个先行结论：

1. **[D] DeepSeek 的主线不是“R1 突然出现”，而是数据工程、细粒度 MoE、压缩式注意力、
   低精度/通信共设计、可验证领域训练逐步汇合。** DeepSeek LLM/Coder/Math 建立数据与
   领域能力，V2 把 DeepSeekMoE 与 Multi-head Latent Attention（MLA）组合，V3 把
   FP8、无辅助损失负载均衡、Multi-Token Prediction（MTP）与 DualPipe 组合，R1 再把
   长推理强化学习推到中心；V3.2/V4 才把工具环境、稀疏长上下文和部署等价 rollout 变成
   同一条生产链。主要证据见 [DeepSeek LLM](https://arxiv.org/abs/2401.02954)、
   [DeepSeekMoE](https://arxiv.org/abs/2401.06066)、
   [DeepSeek-V2](https://arxiv.org/abs/2405.04434)、
   [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、
   [DeepSeek-R1](https://arxiv.org/abs/2501.12948)、
   [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf) 与
   [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)。

2. **[D] Moonshot 的主线同样不是单一 Agentic RL。** Mooncake 先从 KV cache 与
   prefill/decode 解耦解决长上下文服务；k1.5 公开长思维链 RL 与部分 rollout；Moonlight
   把 Muon 扩到大模型；K2 用 MuonClip 训练万亿参数 MoE 并合成工具环境；Kimi Linear
   与 Attention Residuals 分别改写序列记忆和深度残差；K2.5 将视觉与 Parallel-Agent
   Reinforcement Learning（PARL）纳入统一模型；K3 在截止日只公布高层架构而未交付完整
   报告。证据见 [Mooncake](https://arxiv.org/abs/2407.00079)、
   [Kimi k1.5](https://arxiv.org/abs/2501.12599)、
   [Muon is Scalable](https://arxiv.org/abs/2502.16982)、
   [Kimi K2](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)、
   [Kimi Linear](https://arxiv.org/abs/2510.26692)、
   [Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) 与
   [Kimi K3](https://www.kimi.com/en/blog/kimi-k3)。

3. **[D] 两家分别公开了不止 MoE 的多轴稀疏机制。** DeepSeek 依次探索计算稀疏（MoE）、
   KV 压缩（MLA）、token 选择/压缩（DSA、CSA/HCA）和静态条件记忆（Engram）；Moonshot
   则探索专家稀疏、块路由（MoBA）、递归状态更新（KDA）与深度残差路由（AttnRes）。
   **[I]** 它们可被综合为把“容量、序列、深度、存储”拆成可独立分配的资源轴；但把这些
   方法统称为一套相同机制是错误的。

4. **[D] 可验证奖励是两家扩展 RL 的共同支点，但最终整合方法不同。** DeepSeek 从
   DeepSeekMath 的 Group Relative Policy Optimization（GRPO）走到 R1、V3.2 的领域
   专家，再在 V4 以多教师 On-Policy Distillation（OPD）整合；Moonshot k1.5/K2 采用
   无价值网络的序列级目标，K2.5 又引入 token-ratio clipping、Toggle RL 与 PARL。
   “都用了 RL”远不足以描述这些目标、采样分布与系统约束的差异。

5. **[I] 现有公开材料不足以端到端复现两家旗舰训练与生产系统。** 公开权重、论文、参考
   推理、部分内核/系统很有价值；但原始语料与许可混合、完整训练代码、全部失败试验、奖励
   数据/模型、生产路由、线上安全策略、总能耗与总成本分别是 **[U]**。因此，任何“完全
   开源”或“训练总成本就是某个单一美元数”的概括都越过了证据。

### 1.1 69 条记录实际覆盖哪些研究域

各域会重叠，所以下表不是互斥计数、也不能相加成 69；它用于证明本章没有把“实验室研究”
缩成旗舰 LLM 或 agentic RL。

| 研究域 | DeepSeek 库存覆盖 | Moonshot/Kimi 库存覆盖 |
|---|---|---|
| 基础模型、scaling、MoE | LLM、DeepSeekMoE、V2/V3/V4、ESFT | Moonlight、K2/K2.5/K3 |
| 数据工程与预训练 | global dedup、repo dependency/FIM、math recovery、formal synthesis | multimodal curation、semantic rephrasing、audio 13M+ hours、SWE mid-training |
| 优化、数值稳定与低精度 | FP8/DualPipe、mHC、V4 Muon/FP4 | distributed Muon、MuonClip、K3 Per-Head Muon/MXFP4/8 |
| 后训练、RL、奖励与蒸馏 | DPO、GRPO、R1、DeepSeek-GRM、mixed GRPO、OPD | k1.5 loss、K2 Gym/self-critique、Toggle、PARL、long-to-short |
| 长上下文、attention、memory | MLA、NSA、DSA、CSA/HCA、Engram、1M V4 | Mooncake、MoBA、KDA+MLA、AttnRes、1M Kimi Linear/K3 |
| Code 与 software engineering | Coder/Coder-V2、V3.2 coding env、CCAgent、GUI test migration | Kimi-Dev、K2/K2.6 coding agent |
| 数学、formal proof 与 benchmark | Math/Math-V2、Prover V1/V1.5/V2 | Kimina-Prover、CombiBench |
| 视觉语言与文档 | VL/VL2、OCR/OCR2 | Kimi-VL/K2.5、SimpleSeg、WorldVQA、PerceptionBench |
| 视觉生成、3D、video、image restoration | Janus/JanusFlow/Janus-Pro、DreamCraft3D | VisionLLaMA、LanDiff、RS-Diffusion、machine IQA、skeleton understanding |
| 音频 | 本库存无核心音频报告 | Kimi-Audio 的理解、生成、对话与 serving/evaluation |
| 服务、存储、检索与硬件 | Fire-Flyer、V3 co-design、Faiss、DualPath、DSpark | Mooncake、ANSMET、K2/K2.5 rollout/checkpoint infra |
| Agents、tools 与 multi-agent | R1→V3.2/V4、DSec、CCAgent | K2、Kimi-Researcher、K2 Thinking、K2.5/PARL、Agent Swarm、K2.6 |
| Evaluation、safety、governance | safety RL、GRM、AI-regulation perspective | Vendor Verifier、WorldVQA/PerceptionBench、ExtendAttack、safety rewards |
| 非 LLM 的机构署名应用 | EV charging probabilistic forecasting | SAGraph、rolling shutter、machine IQA、ACLNet 等 |

完整逐条对应关系见第 8、9 节；任何 `affiliated` 行都不会仅凭机构署名被倒推出旗舰配方。

## 2. 双实验室时间轴：模型与研究如何接起来

### 2.1 DeepSeek 主干与分支

| 日期 | 节点 | 研究转折 | 证据边界 |
|---|---|---|---|
| 2024-01 | [DeepSeek LLM](https://arxiv.org/abs/2401.02954) | 7B/67B 稠密基线、2T processed tokens、全局去重、SFT+DPO | 方法事实 [D]；语料源比例与总开发成本 [U] |
| 2024-01 | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | 细粒度 routed experts + shared experts | [D] 后续 V2/V3 的架构祖先 |
| 2024-01 | [DeepSeek Coder](https://arxiv.org/abs/2401.14196) | 仓库级代码、依赖排序、FIM、16K | [D] 形成代码数据工程支线 |
| 2024-02 | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 迭代恢复数学网页、领域继续预训练、首次公开 GRPO | [D] 不应把 GRPO 的起点误记为 R1 |
| 2024-03 | [DeepSeek-VL](https://arxiv.org/abs/2403.05525) | 固定 token 预算的混合视觉编码器、现实场景 VLM 数据 | [D] 多模态理解支线起点 |
| 2024-05 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 236B/21B MoE + MLA；128K；在线 GRPO | [D] 连接架构、服务效率与后训练 |
| 2024-05–08 | [Prover](https://arxiv.org/abs/2405.14333) → [Prover-V1.5](https://arxiv.org/abs/2408.08152) | 8M Lean 合成证明 → proof-assistant feedback RL + RMaxTS | [D] 形式证明独立于一般聊天谱系 |
| 2024-06 | [Coder-V2](https://arxiv.org/abs/2406.11931) | 从 V2 中间点继续 6T，扩到 338 种语言与 128K | [D] 不是在完成版 V2 上简单微调 |
| 2024-10–2025-01 | [Janus](https://arxiv.org/abs/2410.13848) → [JanusFlow](https://arxiv.org/abs/2411.07975) → [Janus-Pro](https://arxiv.org/abs/2501.17811) | 解耦理解/生成视觉编码；自回归与 rectified flow；数据/模型放大 | [D] 统一理解-生成支线，不等同于 VL/VL2 |
| 2024-12 | [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) | 动态切片高分辨率 + MoE 解码器 | [D] 1.0B/2.8B/4.5B active 三档 |
| 2024-12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B/37B；FP8、MTP、DualPipe、无辅助损失平衡 | [D] 14.8T 是 processed exposure，不是 unique corpus |
| 2025-01 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | R1-Zero 无 SFT 实验；生产 R1 为四阶段 SFT/RL 管线 | [D] 两者必须分开叙述 |
| 2025-02 | [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | 压缩、选择、滑窗三路的原生可训练稀疏注意力 | [D] 是架构研究；不是后来的 DSA 同义词 |
| 2025-04 | [DeepSeek-GRM](https://arxiv.org/abs/2504.02495) | 自生成原则/批评与推理时奖励模型放大 | [D] 通用奖励系统支线 |
| 2025-04 | [Prover-V2](https://arxiv.org/abs/2504.21801) | V3 分解子目标、7B 搜索、Lean 验证、二元奖励 RL | [D] 桥接非形式推理与形式证明 |
| 2025-09–12 | [V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) → [V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf) | DSA 转换；专家 RL、工具保留推理、85,267 tasks / >1,800 envs | [D] tasks 不等于 environments |
| 2025-10–2026-01 | [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) → [OCR 2](https://arxiv.org/abs/2601.20552) | 光学上下文压缩 → visual causal flow token 重排 | [D] 文档理解也是上下文效率研究 |
| 2025-11 | [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) | 生成器与验证器共同提升、自验证数学证明 | [D] 以 proof-level 评分扩展可验证性 |
| 2025-12–2026-01 | [mHC](https://arxiv.org/abs/2512.24880) + [Engram](https://arxiv.org/abs/2601.07372) | 约束残差路由；O(1) n-gram 条件记忆 | [D] 后者是 MoE 之外的“记忆稀疏” |
| 2026-02 | [DualPath](https://arxiv.org/abs/2602.21548) | agentic 多轮 KV 存储带宽双路径加载 | [D] 服务/存储研究，不是训练算法 |
| 2026-04 | [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | Flash 284B/13B、Pro 1.6T/49B；mHC、CSA/HCA、Muon、1M、OPD | 架构与训练阶段 [D]；硬件规模与总成本 [U] |
| 2026-07 | [DSpark](https://arxiv.org/abs/2607.05147) | 半自回归 draft + 置信度/负载感知验证 | V4 线上流量部署 [D]；具体流量分布 [U] |

### 2.2 Moonshot AI / Kimi 主干与分支

| 日期 | 节点 | 研究转折 | 证据边界 |
|---|---|---|---|
| 2024-06 | [Mooncake](https://arxiv.org/abs/2407.00079) | Kimi 服务的 prefill/decode 解耦、分布式 KV cache 与 SLO 调度 | [D] 证明服务架构，不披露当时模型参数 |
| 2025-01 | [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 128K 多模态长 CoT；无价值网络 RL；部分 rollout 延续 | 训练方法 [D]；参数量、集群、总成本 [U] |
| 2025-02 | [MoBA](https://arxiv.org/abs/2502.13189) | query 对上下文块路由、选中块内运行标准 attention | [D] 研究到 1M；并不证明 K2/K2.5 使用 MoBA |
| 2025-02 | [Muon is Scalable / Moonlight](https://arxiv.org/abs/2502.16982) | 分布式 Muon、15.29B/2.24B active 研究模型、5.7T | [D] 形成 K2 优化器前史 |
| 2025-04 | [Kimi-VL](https://arxiv.org/abs/2504.07491) | 16B/2.8B active VLM、MoonViT、128K、视觉 agent/RL | [D] 后续 K2.5 视觉路线的先行节点 |
| 2025-04 | [Kimina-Prover](https://arxiv.org/abs/2504.11354) | 72B Lean 模型、形式推理 pattern、20K cold start + RL | [D] 以内部长推理代替外部树搜索为主 |
| 2025-04 | [Kimi-Audio](https://arxiv.org/abs/2504.18425) | 12.5Hz 音频 token、连续感知/离散生成、flow-matching 解码 | [D] 本来源宇宙中两家唯一核心音频报告 |
| 2025-05 | [G1 / VLM-Gym](https://arxiv.org/abs/2505.13426) | 视觉游戏并行 RL，感知冷启动后感知/推理互相 bootstrap | [D] 直接研究，不等同于 Kimi-VL 产品配方 |
| 2025-06 | [Kimi-Dev](https://arxiv.org/abs/2509.23045) | agentless 定位/修复/测试技能先验迁移到 SWE agent | [D] 官方发布早于 arXiv 编号月份 |
| 2025-06 | [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) | 搜索、浏览、代码工具的端到端 REINFORCE；异步/部分 rollout | [D] 无独立 arXiv/PDF，属于一手项目报告 |
| 2025-07 | [Kimi K2](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) | 1.04T/32.6B active、MuonClip、合成工具/agent/Gym、15.5T | 模型与训练阶段 [D]；总 GPU 数、能耗、成本 [U] |
| 2025-09–11 | [K2-0905](https://platform.kimi.com/blog/posts/kimi-k2-0905) → [K2 Thinking](https://www.kimi.com/en/blog/kimi-k2-thinking) | 256K agent coding；思考与 200–300 次工具调用交错、INT4 QAT | 研究/模型页所述更新 [D]；完整更新配方 [U] |
| 2025-10 | [Kimi Linear](https://arxiv.org/abs/2510.26692) | Kimi Delta Attention（KDA）与 MLA 3:1 混合、1M | [D] 是 K3 的已验证前驱，不证明 K3 层比例相同 |
| 2026-01 | [Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) | K2 + ~400M MoonViT、约 15T 继续预训练、视觉 RL、Toggle、PARL | （来源事实 [D]；综合判断 [I]） 累积约 30T processed exposure 是 [I]，不是 unique data |
| 2026-01–02 | [SimpleSeg](https://arxiv.org/abs/2601.19228) + [WorldVQA](https://arxiv.org/abs/2602.02537) | 文本坐标序列的像素级分割；隔离“原子视觉知识”评测 | [D] 方法与评测补齐精细感知边界 |
| 2026-02 | [Agent Swarm](https://www.kimi.com/en/blog/agent-swarm) | PARL 训练 orchestrator 并行派生冻结 subagents | [D] 与 K2.5 报告有内容重叠但为独立一手披露 |
| 2026-03 | [Attention Residuals](https://arxiv.org/abs/2603.15031) | 对早期残差流做学习式注意力，而非等权累加 | K3 采用 AttnRes [D]；具体参数化 [U] |
| 2026-04 | [K2.6](https://www.kimi.com/en/blog/kimi-k2-6) | 长时 coding/design 与更大 agent swarm 产品范围 | [D] 没有 K2.6 专属完整技术报告 |
| 2026-07 | [PerceptionBench](https://www.kimi.com/en/blog/perception-bench) | 原子视觉感知与失败类型评测 | [D] 截止日无链接数据仓库/arXiv |
| 2026-07 | [Kimi K3](https://www.kimi.com/en/blog/kimi-k3) | 2.8T、896 experts/16 selected、KDA、AttnRes、LatentMoE、1M | （已披露部分 [D]；未知部分 [U]） active 参数、数据、训练/RL、总成本仍 [U] |

### 2.3 重大定量结论的一手定位表

为避免“链接到整篇、却找不到结论”，下表给出正文中反复使用的数量级、训练阶段和系统结论
在一手材料中的定位。`§` 指报告章节；页码按截止日归档/发布版本，后续 arXiv 修订可能平移。
逐篇地图中的短摘要级结论则直接对应相应论文摘要/引言和 inventory primary URL。

| 一手来源 | 本章使用的重大结论 | 原文定位 |
|---|---|---|
| [DeepSeek LLM](https://arxiv.org/abs/2401.02954) | 2T、global dedup 89.8% vs 22.2%、7B/67B、1.5M SFT/DPO | §§2、4，pp. 4–6、12–13 |
| [DeepSeek Coder](https://arxiv.org/abs/2401.14196) | 1.8T@4K + 200B@16K、87/10/3 mixture、FIM、32.8% retained | §§2–3，pp. 3–9，Tables 1–2 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 120B underlying corpus、500B exposure、776K SFT、144K RL/64 samples | §§2–4，尤其 Tables 1、3 与 §4.1 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 236B/21B、MLA/MoE、8.1T、128K、训练 GPU-hours、两阶段 GRPO | §§2–4，Tables 1–3；training-cost discussion in §3 |
| [DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931) | V2 4.2T checkpoint + 6T、1.17T code corpus、338 languages、RL prompts | §§2–4，Tables 1–3 |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B/37B、14.8T、FP8/MTP/DualPipe、2.788M H800-hours/\$5.576M | §§2–5；cost Table 1；infrastructure §3；post-training §5 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | R1-Zero vs R1、四阶段、804,745 SFT、rollout/RM/cost/distillation | §§2–3；revised report Appendices B.1–B.6 |
| [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | indexer 1,000 steps/约 2.1B，joint 15,000 steps/约 943.7B，top-2,048 | report §§1–3，architecture/training tables |
| [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf) | 85,267 tasks、>1,800 envs、specialists/mixed GRPO、Keep Routing/Mask | §§2–4，Table 1，agent-data tables |
| [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | Flash/Pro 参数与 32T/33T、1M、CSA/HCA/mHC、Muon、OPD、DSec | §§2、4–5，model-specification/optimizer tables |
| [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) | 380M DeepEncoder、3B/570M active decoder、token compression/precision | §§3–5，Figures 1、3，Table 1 |
| [DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552) | 500M LLM-style encoder、256–1,120 visual tokens、causal-flow query | §§3–5，Figure 1 |
| [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) | 3B/16B/27B total 与 1.0B/2.8B/4.5B active、dynamic tiling | §§2–4，Figure 1，model-architecture table |
| [DeepSeek-Prover](https://arxiv.org/abs/2405.14333) | 8M Lean statement-proof pairs 与 iterative synthesis/verification | §§2–3，Abstract、data-generation figures |
| [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152) | RLPAF、truncate-resume、RMaxTS 与 proof-search protocols | §§2–4，Figures 2–3 |
| [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) | 7B/671B、subgoal cold start、Lean-verification RL | §§2–3，Figure 2、training-data tables |
| [DualPath](https://arxiv.org/abs/2602.21548) | ≥95% cache-hit workload observation、1.87× offline/1.96× online | Abstract、§§1、5–6，Figures 1、9–12 |
| [DSpark](https://arxiv.org/abs/2607.05147) | V4 live-traffic deployment、matched-throughput speed ranges | Abstract、§§1、5，online-serving figures |
| [Mooncake](https://arxiv.org/abs/2407.00079) | KV disaggregation、525% simulation、75% real-workload requests | Abstract、§§2–5，evaluation figures/tables |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | data stages、~2M SFT、RL objective/rewards、partial rollout/infra | §§2–4，pp. 4–11；Appendix B, pp. 21–24 |
| [Muon is Scalable](https://arxiv.org/abs/2502.16982) | ~16B/3B Moonlight、5.7T、distributed Muon、52% fitted FLOPs | §§2–4，model/schedule/scaling tables |
| [MoBA](https://arxiv.org/abs/2502.13189) | block 4,096/top-12、29 sparse + 3 full layers、~100B activation tokens | §§2–4，1M-context experiment table |
| [Kimi K2](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) | 1.04T/32.6B、384/8+1 experts、15.5T、MuonClip、tool/Gym/RL | §§2–5，architecture/training/post-training tables |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) | KDA equation、48B/3B、3:1 KDA/MLA、5.7T/1M、efficiency results | §§2–5，architecture/training/efficiency tables |
| [Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) | ~400M vision、~15T continuation、visual RL/Toggle/PARL、100K tasks | §§2–5，architecture/data/RL/agent tables |
| [Kimi-VL](https://arxiv.org/abs/2504.07491) | 16B/2.8B active、128K、MoonViT 与 Thinking-2506 RL | §§2–5，Figure 1，architecture/training tables |
| [Kimi-Audio](https://arxiv.org/abs/2504.18425) | 12.5Hz、13M+ hours、continuous+discrete I/O、serving/evaluation | §§2–6，Figure 2 |
| [Kimina-Prover](https://arxiv.org/abs/2504.11354) | ~20K cold start、1,000 prompts × 8 rollouts/iteration、Lean reward | §2.2–2.3，Figure 2 |
| [Kimi-Dev](https://arxiv.org/abs/2509.23045) | ~150B mid-training、5K agent trajectories 与 workflow→agent transfer | §§3.2–3.5、§4，Figure 1 |
| [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) | 23 steps/200+ URLs、70+ searches、REINFORCE、async/partial rollout | “Training data”、“RL training”、“Large-scale agent RL infra” |
| [WorldVQA](https://arxiv.org/abs/2602.02537) | 3,500 pairs、九类、atomic isolation 与 dual-gate curation | Abstract、§§1、3，Table 1 |
| [Kimi K2.6](https://www.kimi.com/en/blog/kimi-k2-6) | 300 subagents、4,000 coordinated steps 与长时案例边界 | official “Agent Swarm”/capability/evaluation sections |
| [Kimi K3](https://www.kimi.com/en/blog/kimi-k3) | 2.8T、896/16 experts、1M、precision/architecture 名词表 | official architecture/efficiency/release sections |

这一定位表只提高可追踪性，不把 vendor measurement 升格为 [R]；正文凡作跨论文综合，仍单独
标为 [I]，凡缺少关键 protocol/denominator 则保留 [U]。

## 3. DeepSeek：从数据与 MoE 到百万 token 模型—系统共设计

### 3.1 架构与 scaling：四次资源重分配

#### 3.1.1 稠密 scaling 与细粒度 MoE [D]

[DeepSeek LLM](https://arxiv.org/abs/2401.02954) 用 7B/67B 两档校准 scaling，并公开
4,096 context、RMSNorm、SwiGLU、RoPE 等稠密基线；67B 使用 Grouped-Query
Attention（GQA）。[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 随后指出传统
top-$K$ MoE 中专家粒度过粗、知识重叠，给出两项结构性改动：把 routed experts 切得更细，
让每个 token 从更多细粒度专家中组合；另设 shared experts 吸收通用知识，减少 routed
experts 的冗余。这不是“总参数越大”的朴素 scaling，而是把每 token 的固定计算预算拆成
更丰富的条件组合。

[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 将这条路线放大到 236B total / 21B
active：首层 FFN 稠密，后续每层有 2 个 shared、160 个 routed、激活 6 个 routed experts。
[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 继续到 671B/37B，改成前三层稠密，
后续每层 1 shared + 256 routed / 8 active，并用 routing bias 做无辅助损失负载均衡，
只保留很小的 sequence-level balance loss。这里的关键不是删除一切平衡约束，而是避免
大 auxiliary loss 扭曲语言建模目标。

#### 3.1.2 MLA：把 KV cache 当成架构一等公民 [D]

V2 引入 MLA，把每 token 的 keys/values 压到低维 latent，仅为 RoPE 保留解耦的位置分量；
V2 报告相对其 DeepSeek-67B 基线显著压缩 KV cache。V3 保留 MLA，并把它与 MTP、FP8、
expert parallel 一起共设计。这说明 MLA 不只是一个 attention 公式：它改变了 cache 容量、
HBM/带宽、长序列 batch 和服务吞吐的约束面。

#### 3.1.3 从 token sparsity 到 compressed sparsity [D]

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) 以三路结构处理长上下文：
压缩 token 提供全局粗粒度信息、选择分支保留关键细粒度块、滑动窗口保住局部连续性；论文
强调训练 forward/backward 与推理都要硬件对齐。它是方法研究，不能与后续产品机制画等号。

[V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) 的 DeepSeek Sparse
Attention（DSA）则用轻量 lightning indexer 给历史位置打分，在 MLA 表示上只选择 top-2,048
KV entries。转换分两段：冻结主模型训练 indexer 约 2.1B tokens，再联合稀疏继续训练约
943.7B tokens。**[U]** 这不证明 DSA 与 NSA 共享同一参数化或训练代码。

[DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
进一步把压缩加入选择单元：Compressed Sparse Attention（CSA）先每四个 KV 压成一个条目，
再作 DSA 式选择并保留 128-token local window；Heavily Compressed Attention（HCA）每
128 个 KV 压成一个条目并在短序列上做 dense attention，两类层交错。报告估算 1M context
的 KV cache 约为指定 BF16 GQA8 基线的 2%；这是报告的模型/假设下估算，不是任意部署的
通用比例。

#### 3.1.4 残差与条件记忆：横向扩展 Transformer 原语 （来源事实 [D]；综合判断 [I]）

[mHC](https://arxiv.org/abs/2512.24880) 指出普通 Hyper-Connections 扩展 residual
stream 后会破坏 identity mapping 并放大不稳定性；它将混合矩阵投影到约束流形，并配合
kernel fusion、重计算和 DualPipe 通信重叠。V4 采用 expansion 4 和 Sinkhorn 归一化。

[Conditional Memory / Engram](https://arxiv.org/abs/2601.07372) 把静态局部模式从
神经计算中拆出：经 tokenizer compression、multi-head hashing、contextual gate 的
$n$-gram embedding 以 $O(1)$ lookup 接入 backbone。论文在固定参数/FLOPs 下研究 MoE
与 lookup memory 的分配，并报告 host memory 预取使 100B table 的额外开销小于 3%。
**[I]** 它提出的深层含义是：模型容量可以分成“动态组合计算”和“静态寻址记忆”，而非全部
塞入 experts；但库存证据不表明 Engram 已进入 V4。

### 3.2 数据与预训练：从“抓网页”到可迭代的数据运算

#### 3.2.1 全局去重与领域恢复 [D]

[DeepSeek LLM](https://arxiv.org/abs/2401.02954) 的 2T processed-token 数据中，跨
91 个 Common Crawl dump 的全局 dedup 删除 89.8%，而逐 dump 去重仅删除 22.2%。这个
数字的意义不是某个固定去重率，而是：重复 web snapshots 会让“原始字节/抓取量”严重高估
独特训练信息。

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 给出更可迁移的领域恢复闭环：以
OpenWebMath 为正种子训练 fastText，给约 40B 个去重 HTML pages 打分；从高召回 domain
中让人标注有用 URL path，再恢复页面并迭代四轮。最终 underlying corpus 是 35.5M pages /
120B tokens，但模型把它按 56% 比例与 AlgebraicStack、arXiv、GitHub code、普通中英文
web 混合，实际继续预训练 exposure 为 500B。**underlying unique-ish corpus 与 sampled
training exposure 不可互换。**

#### 3.2.2 仓库级代码与可执行过滤 [D]

[DeepSeek Coder](https://arxiv.org/abs/2401.14196) 不把文件视作无关联文本：它解析
`import`/`include`/`using` 依赖，在可行时拓扑排序，带路径拼接成 repository examples；再做
repository near-dedup、编译/质量/启发式过滤以及 benchmark 10-gram decontamination。
初始代码中 32.8% 留到最终物料；预训练为 1.8T@4K + 200B@16K，其中 50% 文档用
prefix–suffix–middle FIM。

[DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931) 从 V2 **训练到 4.2T 时的中间
checkpoint** 分叉，再吃 6T code-heavy tokens，而不是从完成版 8.1T V2 开始；该支线总
exposure 因而为 10.2T。其 underlying code 约 1.17T tokens，覆盖 338 种语言；大模型用
next-token，小模型保留 50% FIM。报告的区分再次说明 lineage 不能只看模型名称。

#### 3.2.3 V3/V4 的规模与未知边界 （已披露部分 [D]；未知部分 [U]）

V3 报告 14.8T processed tokens，增加 math、code、多语言和高质量内容，128K BBPE
vocabulary，cross-document packing 与 0.1 FIM；精确来源比例、原始数据集、许可、unique
token count 是 **[U]**。V4 从 V3 pipeline 出发，过滤批量自动生成/模板 web，增加多语言、
长尾文化/科技/科学/长文，并在 mid-training 加 agent material；Flash/Pro 分别报告
32T/33T processed exposure。**[U]** 没有公开证据可以把这些数字解释为不重复 token，
也没有可审计材料证明训练集完全无 benchmark contamination。

### 3.3 优化与训练系统：算法、数值和网络拓扑共同决定可训练性

#### 3.3.1 从 AdamW/BF16 到 FP8 + DualPipe [D]

DeepSeek LLM 公开 AdamW、BF16 forward/backward、FP32 gradients、FlashAttention、
ZeRO-1 以及 data/tensor/sequence/pipeline parallel。V2 在 H800 上采用 16-way zero-bubble
pipeline、跨八节点 expert parallel、ZeRO-1、无 tensor parallel，并报告每 1T tokens
172.8K H800 GPU-hours；按 8.1T 相乘约 1.40M hours 是 **[I]（按报告值计算）**，不含长上下文、
后训练、探索和数据处理。

V3 把 2,048 张 H800、8-node expert parallel、DualPipe 双向流水、FP8 mixed precision、
MTP 放在一个系统中。报告列出 base pretraining 2.664M H800-hours、context extension
119K、post-training 5K，总计 2.788M hours，并按 \$2/H800-hour 估算 \$5.576M。
**[D]** 上述数值是报告列明 runs 的租赁等价估算。**[U]** 企业财务意义上的研发总成本
未披露；数据、工资、失败 runs、早期架构研究和基础设施均不在这个数字里。

[Fire-Flyer AI-HPC](https://arxiv.org/abs/2408.14158) 从更早的 10,000 PCIe A100 集群
总结软硬件共设计：HFReduce、计算-存储一体网络、HaiScale、3FS 与平台层重叠通信。
[Insights into DeepSeek-V3](https://arxiv.org/abs/2505.09343) 再从 MLA、MoE、FP8、
multi-plane network 反推未来硬件需要精确低精度单元、scale-up/scale-out 融合和低时延
fabric。两文都说明，V3 的经济性结论不能脱离其网络、路由和精度假设。

#### 3.3.2 V4 改用 Muon，但不是“抄一个优化器”即可 [D]

V4 对多数矩阵使用 Muon，对 embeddings、output head、RMSNorm 保留 AdamW；报告给出
Muon momentum .95、weight decay .1、update RMS .18。训练还包括 4K→16K→64K→1M
context curriculum、SwiGLU clamp、loss-spike detector、rollback 后才短时开启的
Anticipatory Routing。**[I]** Moonshot 的 Moonlight 先公开 Muon scaling，而 V4 后来采用
Muon，说明开放研究发生了跨实验室技术扩散；但没有证据证明两边实现、shape scaling、
分布式状态或稳定化细节相同。

### 3.4 后训练、RL、蒸馏与奖励系统

#### 3.4.1 从 DPO 到 GRPO [D]

DeepSeek LLM 的对齐是约 1.5M SFT examples（1.2M helpful + 300K safety）再做 DPO；
没有报告 online RL。[DeepSeekMath](https://arxiv.org/abs/2402.03300) 才首次公开
DeepSeek 的 GRPO：每个问题采样一组回答，以组内相对回报替代 PPO critic；报告约 144K
math prompts、每题 64 outputs、outcome/process reward 两个版本，并在 iterative RL 中
刷新 reward model、保留 10% historical replay。

V2 把 online GRPO 分成两段：先 code/math reasoning，使用规则或 learned reasoning reward；
再联合 helpfulness、safety 与 rule rewards。Coder-V2 对约 40K code/math prompts 训练：
数学直接用 ground truth；裸 compiler 0/1 被认为噪声过大，改用 compiler outcomes 监督
learned code reward model，再由 GRPO 优化。这是“可执行”不自动等于“可直接作标量奖励”
的典型反例。

#### 3.4.2 R1-Zero 与生产 R1 必须拆开 [D]

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 的 R1-Zero 从 V3-Base 开始，**没有
cold-start SFT**，以 accuracy + format rule rewards 做长推理 GRPO。修订报告披露每题
16 responses、32 unique prompts/update、最大长度先 32,768 后 65,536、约 10,400
policy steps；它出现更长推理、自我反思，也出现可读性和语言混杂问题。

生产 R1 则是四阶段：

1. 几千条经过正确性、可读性、重复和语言过滤的 cold-start long-CoT SFT；
2. math/code/STEM/logic 的 reasoning RL，并加语言一致性 reward；
3. rejection-sampled SFT，共 804,745 条（约 600K reasoning、200K non-reasoning）；
4. rule reasoning rewards 与 helpfulness/safety reward models 的 mixed RL。

因此“R1 全程无 SFT”是错误命题。报告还给出约 147K H800-hours / \$294K 的 R1-Zero、
data generation、R1 post-training 列明开销；**[U]** 它不含 V3 base、数据/RM 研发和失败
实验，不能当作 R1 从零到交付的总成本。

#### 3.4.3 专家、混合 GRPO 与 V4 OPD [D]

V3 用 domain reasoning experts 生成 long-CoT 并 rejection-sample；V3.2-Exp/V3.2 把
writing、general QA、math、programming、logic、general agent、agentic code/search 以及
thinking/non-thinking specialists 分开做大规模 RL，再蒸馏到一个 checkpoint，最后用 mixed
GRPO 减少遗忘。V3.2 还纠正 KL/importance sampling，并以 **Keep Routing** 和
**Keep Sampling Mask** 让 learner replay 遵守 rollout 时的 MoE route 与 top-p/top-k
action support，直接处理训练/推理 numerical mismatch。

V4 保留十多个 GRPO specialists，却把最终整合改为 multi-teacher OPD：student 在自己的
state distribution 上生成轨迹，对相应 experts 做 full-vocabulary reverse KL。系统缓存
teacher final hidden states，而非十万词表 logits；按 teacher 排 minibatch，并在需要时加载
teacher head 重建 logits。**[U]** teacher weights、冲突处理、完整样本量和 reward mixture
未公开；“V4 最终模型由 mixed GRPO 合并”不成立。

#### 3.4.4 Generative reward model （已披露部分 [D]；未知部分 [U]）

[Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)
把开放域 reward 变成 pointwise generative critique，而非固定 scalar：Self-Principled
Critique Tuning（SPCT）通过 online RL 学会按输入自适应生成原则和批评；推理时并行采样，
再由 meta reward model 引导 voting。它为 V3.2/V4 报告中的 rubric-guided generative
judges 提供方法背景，但 **[U]** 清单证据不能证明线上 judge 就是论文发布的具体
DeepSeek-GRM checkpoint。

### 3.5 长上下文、推理服务、存储与容错是一条连续链

#### 3.5.1 Context extension 与 attention efficiency [D]

Coder 从 4K 到 16K；V2/Coder-V2 通过 YaRN 到 128K；V3/V3.1/V3.2 继续在 32K/128K
阶段训练并以 DSA 降低长序列成本；V4 把 curriculum 推到 1M。应区分：训练长度、验证
长度、模型卡/API 宣传长度和单次评测 harness 上限不是同一个量。

#### 3.5.2 Agentic workload 把瓶颈从算力推向 KV I/O [D]

[DualPath](https://arxiv.org/abs/2602.21548) 观察多轮 agent session 往往是“短追加、
长历史、高 cache hit”，外部存储到 prefill engine 的 KV load 成为瓶颈，而 decode engine
的 storage NIC 空闲。它增加 storage→decode→RDMA→prefill 路径，并由 global scheduler
在传统路径与新路径之间分流。论文在自有系统/指定 workload 报告吞吐提升；这些数值不可
外推到不同网络或 cache hit 分布。

V4 又把百万 token rollout 的训练偏差纳入容错：token-granular write-ahead log、preemption
时保存未完成 KV、硬件故障后由 token log 重建，避免每次故障从头生成而系统性偏向短轨迹；
DeepSeek Elastic Compute（DSec）描述在 3FS 上提供 container/microVM/QEMU、全序
trajectory logs、preemption-safe resume 和 non-idempotent tool replay。**[U]** DSec 的完整
生产实现并未随报告开源。

#### 3.5.3 Speculative decoding 也必须感知在线负载 [D]

[DSpark](https://arxiv.org/abs/2607.05147) 用 parallel backbone + lightweight sequential
module 的 semi-autoregressive draft 缓解后缀 acceptance decay；confidence head 估计前缀
存活概率，scheduler 再结合引擎实时吞吐决定每个请求验证多长。论文报告已在 V4 live traffic
中替换/比较 MTP-1 baseline，并在 matched throughput 下提升 per-user generation speed；
**[U]** 用户流量分布、完整 SLO 和运维成本未公开，不能把百分比视作通用硬件加速常数。

### 3.6 多模态、OCR、统一理解与生成

#### 3.6.1 VL → VL2：现实场景理解 [D]

[DeepSeek-VL](https://arxiv.org/abs/2403.05525) 以 web screenshot、PDF、OCR、chart、
知识材料与真实 use-case taxonomy 构造数据；混合 vision encoder 同时取语义与高分辨率
细节，在固定视觉 token 预算内处理最高 1024×1024。训练先 language-heavy，再逐步提高
vision-language 比例，目的是避免视觉预训练抹掉 LLM 能力。

[DeepSeek-VL2](https://arxiv.org/abs/2412.10302) 用 dynamic tiling 适应不同 aspect ratio
和超高分辨率，替代 VL 的固定 384/1024 双分辨率；语言端换成 3B/16B/27B total 的 MoE，
分别约 1.0B/2.8B/4.5B active，覆盖 VQA、OCR、document/table/chart、grounding 与
visual reasoning。

#### 3.6.2 Janus 系列：解耦编码、统一 Transformer [D]

[Janus](https://arxiv.org/abs/2410.13848) 的核心判断是：理解要高层语义，生成要局部细节，
强迫两者共用同一个 vision encoder 会冲突；所以理解与生成走不同视觉编码路径，再进入一个
统一 autoregressive Transformer。[JanusFlow](https://arxiv.org/abs/2411.07975) 保留编码
解耦，却以 rectified flow 做图像生成，并对齐理解/生成 representations；
[Janus-Pro](https://arxiv.org/abs/2501.17811) 调整训练顺序、扩大数据并由 1B 放大到 7B。
这条线证明“统一”不必等于“所有 modality 强制同 tokenizer/encoder”。

#### 3.6.3 OCR → OCR 2：把文档图像当作上下文压缩介质 （来源事实 [D]；综合判断 [I]）

[DeepSeek-OCR](https://arxiv.org/abs/2510.18234) 的 DeepEncoder 由约 80M SAM-style
window-attention perception、16× convolution token compressor、约 300M CLIP-style
global component 串联，接 3B total / 570M active MoE decoder。它把数字文本排成 2D 图像，
研究视觉 token 与原文本 token 的压缩：报告在小于 10× 压缩时给出约 97% OCR precision，
20× 时约 60%；这只是指定数据/metric 的实验曲线，不是任意内容的无损压缩保证。

[DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552) 把 CLIP component 换成约 500M
LLM-style encoder：原 visual tokens 之间双向注意，learnable causal-flow queries 可看全部
视觉 token 和之前 queries，最终只把 queries 送给 decoder，从而按文档语义学习 reading
order。**[I]** 这把 OCR 从“识字任务”扩成一个更一般问题：怎样把二维观察压成适合一维
causal model 消费的顺序；报告只在文档上验证，尚未证明能统一任意视觉/音频编码。

### 3.7 Code、数学与形式推理：同一“验证器”思想的三种落地

#### 3.7.1 Code [D]

Coder/Coder-V2 的数据工程提供 repository context 与 FIM，后训练则把 compiler/tests 变成
训练信号。R1 代码 prompt 来自 contest 与 GitHub issue；V3.2 agentic coding 从数百万
issue/PR 挖任务，setup agent 安装依赖并运行 tests，只有 gold patch 让至少一个 failing test
转 pass 且不使既有 passing tests 回归时才保留。V3.2 报告 24,667 coding-agent tasks 与
5,908 code-interpreter tasks；这些是任务数，不是独立仓库或环境数。

#### 3.7.2 Informal math 与 self-verification [D]

DeepSeekMath 建立领域数据 + GRPO；R1 让规则奖励驱动长推理；
[DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) 从 V3.2-Exp-Base 出发，训练 verifier
分析证明并给出分数，再用更大 verification compute 审核超过固定 verifier 能力的候选，
让自动验证的证明反过来提升 generator 与 verifier。**[U]** 自验证不是逻辑证明器：开放域
judge 仍可能同源错误，只有 Lean/compiler/tests 一类外部可执行 oracle 才提供严格通过/失败。

#### 3.7.3 Formal theorem proving [D]

[DeepSeek-Prover](https://arxiv.org/abs/2405.14333) 把自然语言竞赛题转 Lean statements，
过滤简单/不良 statements，以并行尝试原命题与否定命题加速排除不可证数据，再迭代生成并
Lean 验证，形成 8M statement-proof pairs。

[Prover-V1.5](https://arxiv.org/abs/2408.08152) 从 DeepSeekMath-Base 继续形式语言预训练，
SFT 后用 Reinforcement Learning from Proof Assistant Feedback（RLPAF）；推理的
truncate-and-resume 把第一个 Lean error 之前的有效代码留下，并加入 tactic state，RMaxTS
以 intrinsic reward 探索不同 proof paths。

[Prover-V2](https://arxiv.org/abs/2504.21801) 再用 V3 把复杂 theorem 拆成 subgoals，小型
7B prover 搜索，成功子证明重组为 cold-start informal+formal CoT，之后用 Lean binary
verification 做 RL。这里真正连续的不是某个 benchmark 分数，而是数据闭环：
**autoformalize/decompose → search/generate → machine verify → retain → retrain**。

### 3.8 Agents、tool use 与训练环境

#### 3.8.1 从“R1 不会工具”到 V3.2 environment factory [D]

R1 原报告明确承认 tool use、structured output 与软件工程 RL 不足。V3.1（未作为独立论文
入清单）之后，V3.2 才公开把隐藏 reasoning 在 tool call/result 间保留，并构造：24,667
coding-agent、50,275 search-agent、4,417 general-agent、5,908 code-interpreter，合计
85,267 tasks，覆盖 **>1,800 environments**。

Search 数据从 web long-tail entities 出发，由 question agent 搜索提题、不同 answer agents
尝试、verifier 多重检查，只保留 verified ground truth 且存在可验证失败的替代尝试。General
agent 则合成 tool schemas/databases、reference solution 与 verifier，并要求 reference 必须
实际调用工具；RL prompts 保留 non-zero pass@100，避免全易/全难造成无梯度。

#### 3.8.2 Rollout fidelity 是 agent RL 的隐藏核心 （来源事实 [D]；综合判断 [I]）

Keep Routing/Mask、V4 直接用真实 FP4 deployment weights rollout、token-level WAL、
preemption resume 与 deterministic replay，表面上是工程细节，实际共同解决同一问题：训练
更新必须针对**真实 actor 所访问的状态/动作支持与数值路径**。若 learner 用不同 quantization、
router、top-p mask 或在失败后重启不同轨迹，所谓 on-policy 会被系统层悄悄破坏。这个统一
解释是 [I]，组成事实均由 V3.2/V4 报告 [D]。

### 3.9 评测、安全与披露边界

- **[D]** V3/R1/V3.2 把 helpfulness 与 safety reward、规则、generative rubric judge 混合；
  R1 的 safety RM 使用 point labels，V3.2/V4 对难验证任务用生成式 judge。
- **[D]** [DeepSeek-GRM](https://arxiv.org/abs/2504.02495) 自身承认通用 reward 在部分任务
  仍有困难；因此 reward score 不是事实性或安全性的地面真值。
- **[D]** DeepSeek 报告通常给出多套 benchmark，但 prompt、采样、tools、judge、context
  budget 并不统一。本章不据 vendor 表格做跨实验室“谁全面领先”的排行。
- **[U]** 清单中没有覆盖全部旗舰版本的一套持续、统一、外部审计的 system card；生产 API
  routing、moderation、incident response、user-data retention 与 live policy updates 未充分披露。
- **[U]** 开放 weights/部分系统代码不等于开放数据、reward models、训练编排和生产栈；
  可检查性与可复现性必须逐 artifact 判断。

## 4. Moonshot AI / Kimi：长上下文、Muon、全模态与可训练 agent orchestration

### 4.1 架构与 scaling：从服务先行到万亿参数稀疏模型

#### 4.1.1 Mooncake 先暴露了真正的产品瓶颈 （已披露部分 [D]；未知部分 [U]）

[Mooncake](https://arxiv.org/abs/2407.00079) 是清单中最早的 Kimi 直接研究。它不是模型
报告，而是生产 serving platform：prefill 与 decode cluster 分离；GPU HBM、CPU DRAM、
SSD 形成分布式 KV cache；KV chunks 通过 Remote Direct Memory Access（RDMA）搬移；
Conductor 综合 prefix locality、Time to First Token（TTFT）、inter-token latency 与 Service
Level Objective（SLO）调度，并在过载时预测性拒绝请求。论文报告在指定 simulation 中最高
525% throughput gain、实际 workload 可多处理 75% requests。**[U]** 这些数字受 topology、
arrival distribution、cache locality 与 SLO 约束，不能当成模型或任意集群的固定加速。

这篇论文在研究谱系中的位置很重要：Moonshot 在公开完整 base architecture 之前，先把
long-context 产品的核心对象定义为“可搬移、可分层、可调度的 KV state”。后来的 k1.5
partial rollout、K2 checkpoint streaming、Researcher/K2.5 长轨迹系统都沿用这一系统观。

#### 4.1.2 Moonlight：Muon 从小模型算法走到可分布式训练 [D]

[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) 的 Moonlight
研究模型约 15.29B total / 2.24B active（不计 embedding；计入约 16B/3B），采用类似
DeepSeek-V3-small 的 MoE，8K context，训练 5.7T tokens。Muon 对矩阵梯度先做 momentum，
再用五次 Newton–Schulz iteration 近似正交化；Moonlight 增加 weight decay 与按矩阵形状的
update scale，对 embeddings、norms、output head 保留 AdamW。

分布式版本以 ZeRO-1 shard optimizer state，reduce-scatter gradients，在本地更新 momentum，
临时 gather full matrix 做 orthogonalization，再丢弃非本地 update piece 并 all-gather 参数。
论文估算其通信约为 AdamW optimizer 的 1–1.25 倍、forward/backward latency 的 1–3%；
scaling fit 推测同 loss FLOPs 约为 AdamW 的 52%。最后一项是模型族/拟合范围内的 [D]
vendor result，不是“Muon 对所有模型都墙钟快 2 倍”的定律。

#### 4.1.3 K2：把 MuonClip、MLA 与 384 experts 合在一起 （已披露部分 [D]；未知部分 [U]）

[Kimi K2](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) 报告
1.04T total / 32.6B active、61 layers、hidden 7,168、384 routed experts、每 token 8 routed
+ 1 shared active、64 attention heads、MLA、160K vocabulary、128K context。它与 V3
共享许多 MoE/MLA 设计语言，但 experts/heads/dense-layer 布局不同；不能把 K2 叫成 V3
checkpoint 或只靠参数表证明代码继承。

MuonClip 的 QK-Clip 监测每个 head 的最大 pre-softmax attention logit；超过阈值 100 时，
分别按 rotary/non-rotary key 结构缩放相关 Q/K components。K2 报告一个 53B/9B precursor
出现 >1,000 logits，而正式 15.5T run 无 loss spike。**[D]** 这只证明报告中的训练 run；
**[U]** MuonClip 对任意规模、数据和精度的普适稳定性并未得到证明。

#### 4.1.4 Kimi Linear：把一部分 attention 变成递归状态 [D]

[Kimi Linear](https://arxiv.org/abs/2510.26692) 的 Kimi Delta Attention（KDA）维护
矩阵状态 $S_t$：

$$
S_t=\operatorname{diag}(\alpha_t)S_{t-1}
+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
$$

per-channel decay $\alpha_t$ 决定各通道遗忘，delta write $\beta_t$ 只写入当前 memory 尚未
预测的 value；short convolution、normalized Q/K、low-rank gates、RMSNorm 与 sigmoid
output gate 提供局部/输出控制。发布的 48B/3B active MoE 用 KDA:MLA=3:1：KDA 降低长
序列 recurrent cost，MLA 层保留精确 retrieval。

该 checkpoint 先训练 1.4T@4K，公开版本到 5.7T，并延长到 1M；报告给出指定配置下 KV
cache 最多降 75%、1M decode 约 6×、prefill 约 2.9×。这些是研究模型的 [D] 报告值；
**[U]** K3 虽采用 KDA，却未披露完全相同的 3:1 layer ratio 或 training recipe。

#### 4.1.5 Attention Residuals：稀疏还可以发生在深度轴 [D]

[Attention Residuals](https://arxiv.org/abs/2603.15031) 不再让第 $\ell$ 层输入只是所有早期
residual outputs 的等权和，而以学习到的 query/key 对早期 streams 做 normalized attention：

$$
h_\ell=\sum_{i<\ell}\alpha_{i\to\ell}v_i,
\qquad
\alpha=\operatorname{softmax}(q_\ell^\top k_i).
$$

blockwise 版本对 block representatives 做路由，把 residual-attention memory 从随 layer 数
线性增长的细粒度存储降到按 blocks。论文在约八个 blocks 时保留多数收益，并在 Kimi Linear
48B/3B 上做 scaling/下游实验。**[I]** 与 DeepSeek mHC 对照看，两者都不再把 residual
connectivity 当作固定背景：mHC 扩展路径后施加流形约束，AttnRes 则学习选择历史路径；
机制和稳定性假设完全不同。

#### 4.1.6 K3：披露了“名词表”，尚未披露可复现配方 （已披露部分 [D]；未知部分 [U]）

[Kimi K3](https://www.kimi.com/en/blog/kimi-k3) 在截止日前两天公布：2.8T total、896
experts / 16 selected、1M context、native vision，采用 KDA、Attention Residuals、Stable
LatentMoE、Quantile Balancing、Per-Head Muon、SiTU、Gated MLA，从 SFT 起进行
quantization-aware training，MXFP4 weights/MXFP8 activations。官方称 scaling efficiency
约为 K2 的 2.5×，但没有定义这是 loss/FLOP、throughput、cost 还是别的指标。

**[U]** active parameters 不能用 $2.8T\times16/896$ 反推，因为 attention、embedding、
shared components 与 LatentMoE 内部均未给出；pretraining tokens/mixture、优化器完整参数、
post-training/RL、环境、硬件、能耗、时长和成本也未披露。截止日 full weights 尚在承诺日期
之前，完整架构/训练/评测报告仍待后续；因此 K3 是谱系端点，不是本章可以自行补全的报告。

### 4.2 数据与预训练：多模态分阶段、语义改写和“不要把 synthetic 填满一切”

#### 4.2.1 k1.5 的多模态数据管线 （已披露部分 [D]；未知部分 [U]）

[Kimi k1.5](https://arxiv.org/abs/2501.12599) 披露英文、中文、code、math/reasoning、
general knowledge；rule cleaning → fastText classification → embedding near-dedup → LLM
quality。代码按 BigCode 风格清洗并有意 upsample 32 种语言；数学 web/PDF 使用专门 OCR
和 learned filtering；知识材料来自 exercises、textbooks、papers；多模态含 caption、
interleaved document、OCR、knowledge、VQA，并对 synthetic caption 设上限。

训练顺序是 language-first；冻结语言模型单训 vision tower；联合解冻、vision-text 提升到
30%；高质量 cooldown；context 4K→32K→131,072。最后长上下文阶段 40% full attention
使用 natural long docs 与 synthetic long QA/summarization，60% partial attention 使用均匀
cooldown data。**[U]** 参数、总 pretraining tokens、各来源比例、cluster 与成本未公开。

#### 4.2.2 K2 的 semantic rephrasing （已披露部分 [D]；未知部分 [U]）

K2 报告 15.5T processed tokens，来自 web/code/math/knowledge。其显著数据操作是语义改写：
改变 style、perspective、exposition；长文 chunk-by-chunk autoregressive rewrite；数学改成
learning notes，选定材料翻成英文，并用 semantic fidelity filter。大 corpus 最多改写两次。
报告 ablation 显示同一知识的多种独立改写优于机械重复，支持“数据表面多样性可改善固定知识
exposure”的局部结论。**[U]** 原始/合成比例、exact corpus、license、dedup threshold、
unique tokens 仍未知，不能把 15.5T 等同 15.5T 新信息。

#### 4.2.3 K2.5 的 early fusion 与视觉数据 （来源事实 [D]；综合判断 [I]）

[Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) 从接近
最终 K2 的语言 backbone 开始，加入约 400M MoonViT-3D（由 SigLIP-SO-400M 初始化）和
short projector，进行约 15T mixed visual/text continual pretraining。native-resolution NaViT
处理图像；四帧分组和 temporal pooling 把 video-token rate 降 4×。固定 token ablation 中，
early 10:90 visual:text fusion 优于 mid 20:80 与 late 50:50。

数据含 captions、interleaved image-text、OCR、visual knowledge、grounding、video、
computer-use trajectories、screenshot-to-HTML/React/SVG、桌面/移动/web action traces、
小时级视频、points/boxes/contours/segmentation，并对 media filter/dedup。**[I]** K2 的
15.5T 加这约 15T 表示 lineage 上约 30T cumulative processed exposure；不是 30T unique
corpus，也不证明两个阶段无重复。

### 4.3 Post-training 与 RL：一条不同于 GRPO 的目标演化

#### 4.3.1 k1.5：value-model-free long-CoT RL [D]

k1.5 的管线是 vanilla SFT → small long-CoT SFT warmup → RL。SFT 约 1M text + 1M
text-vision examples，先 32K 再 128K。RL prompts 覆盖 STEM/code/general，按从 SFT policy
高温采样十次的 pass rate 估难度；剔除 multiple-choice、true/false、易 hack proof format，
并用 **no-CoT guessing test**：要求不推理直接猜，八次能猜对者剔除，减少 memorization/
leakage/shallow path。

terminal reward 是 binary outcome：code tests、structured math exact/rule verifier、free-form
learned answer matcher。对同题 $K$ 个 old-policy responses，以组内平均回报构造 advantage；
优化 sequence log-ratio 对解析 reward-tilted target 的平方误差，无 value network、无 MCTS，
每轮重置 optimizer，并可通过 log-ratio 使用 older-policy responses。**[U]** 这不是 GRPO，
也没有证据可把后续 K2/K2.5 所有阶段都简化成完全不变的同一 loss。

#### 4.3.2 长输出不应被“一刀切” （来源事实 [D]；综合判断 [I]）

k1.5 使用 gradual length reward：正确答案越短越好、错误答案越长惩罚越大；也用
shortest-correct rejection sampling、DPO 与第二次更强长度惩罚 RL 做 long-to-short。
Partial rollout continuation 给每 iteration 固定新输出 token budget，未完成状态保存到下一轮，
旧 tokens 只作 context，新 segment 才 on-policy/trainable，并以 repetition detector 阻断循环。

**[I]** 这说明 reasoning efficiency 的正确对象不是粗暴固定全局 max tokens，而是“任务难度、
正确性、已完成进度”条件下的预算。K2.5 的 Toggle RL 后来把这一点显式化：交替无约束
reasoning-growth 与 budget-constrained efficiency，预算取每任务正确 rollout length percentile。

#### 4.3.3 K2：工具合成、Gym 与 self-critique [D]

K2 的 agentic SFT factory 先建立 >3,000 real Model Context Protocol（MCP）tools 与
>20,000 synthetic tools，再生成 agent identities、rubric-scored tasks、tool trajectories；
stateful simulator 注入随机 failure，coding 用真实 sandbox/tests，候选由 k1.5/内部 experts
生成并经 LLM/human 过滤。

RL Gym 的 reward 按 domain 分工：math/STEM/logic 用 rule/expert verification；code 用
open/synthetic/pretraining-derived tests 与 GitHub issue/PR；instruction following 用
deterministic verifier + judge + hack detector；grounded factuality 用 sentence-level judge；
safety 用 evolved attacks/targets/rubrics；主观任务用 self-critique reward model。K2 critic 以
core/prescriptive/human rubrics 排序 candidates，再用 verifiable-domain on-policy rollouts 防止
judge 脱离可校验事实。

#### 4.3.4 K2.5：视觉 RL、token-ratio clipping 与 PARL [D]

K2.5 有一个反直觉步骤：主 SFT 使用 text-only examples；图片通过 IPython tool 操作，报告称
human visual trajectories 的泛化较差，随后用 multimodal RL 激活视觉工具。视觉奖励包括
IoU/soft-F1 localization、Gaussian point matching、segmentation IoU、OCR normalized edit
distance、count absolute-difference 与 puzzle verification。

其 policy objective 改为 token-level clipped ratios：ratio 超出区间的 token 不论 advantage
正负都置零梯度；同时保留 general reward models 与 MuonClip。**[U]** 详细 clip interval、
全部 reward weights、rollout counts、SFT/RL token totals 未公开。

PARL 只更新 orchestrator，subagents 冻结；orchestrator 学习是否 spawn、任务如何切分，
subagent tokens 不进 orchestrator gradient。奖励为：

$$
r=\lambda_1r_{\text{parallel}}+
\lambda_2r_{\text{finish}}+r_{\text{performance}}.
$$

parallel shaping 防止退化成串行，finish shaping 防止无意义 spawn/未完成分支；两项逐步 anneal
到零，让最终 task outcome 主导。训练从小 subagents 到大 subagents，并动态分配 inference。
独立 contexts 最后只把相关结果返回 orchestrator，因此它也是 context sharding 方法，而不只是
“多开几个模型”。

### 4.4 长上下文与 attention：MoBA 和 KDA 是两条不同答案

[MoBA](https://arxiv.org/abs/2502.13189) 把 keys/values 切 blocks，以 mean key 表示每块；
每个 query 对所有 block representatives 打分，选 top-$k$ 加当前 causal block，再在选中块内
运行普通 attention。实现把 queries 按所选 blocks 排序，使用 variable-length FlashAttention
与 online softmax。论文的 1M 实验用 block 4,096、top-12、前 29 层 MoBA/末 3 层 full
attention，并约 100B activation tokens 分阶段从 128K 走到 1M。

MoBA 是 query-dependent block sparsity，KDA 是固定大小 recurrent state 的 linear attention，
Mooncake 是 cache placement/serving；三者分别优化 attention 计算、序列状态和系统存储。
**[U]** K2/K2.5 明确披露 MLA，不能因同属 Moonshot 就断言使用 MoBA；K3 明确说 KDA，
却仍未披露其 exact layer schedule。

### 4.5 多模态、音频与精细视觉

#### 4.5.1 Kimi-VL：高分辨率视觉、长 context 与 agent capability [D]

[Kimi-VL](https://arxiv.org/abs/2504.07491) 是 16B total / 2.8B active 的 MoE VLM，
MoonViT native-resolution encoder、128K context，覆盖 OCR、多图、长视频、大学级图文理解、
math reasoning 与 OSWorld 式 agent tasks。Thinking-2506 由 long-CoT SFT + RL 得到。
报告的各 benchmark 是版本/协议绑定的 vendor results；不能与不同 context/tool budget 的
DeepSeek-VL2 或闭源 API 做无条件排行。

#### 4.5.2 Kimi-Audio：连续输入、离散输出的统一音频模型 [D]

[Kimi-Audio](https://arxiv.org/abs/2504.18425) 由三部分组成：audio tokenizer 以 12.5Hz
产生离散 semantic tokens 并保留连续 acoustic vectors；audio LLM 的 shared layers 后分
text/audio heads；chunk-wise streaming detokenizer 用 flow matching 还原 waveform。预训练
数据超过 13M 小时，覆盖 speech、sound、music；处理含 enhancement、diarization、
transcription、filtering，并以 text-only/audio-only、audio↔text mapping、interleaving 任务
从预训练 LLM 继续训练，再做多任务 SFT。

报告还描述在 Kimi App 的部署和开源 evaluation toolkit。**[U]** “13M hours” 是处理后训练
数据总小时量还是含重复/切分的 exposure，不能从摘要 alone 推断 unique audio；原始版权/
许可混合和完整生产语音安全策略未公开。

#### 4.5.3 G1 与 SimpleSeg：可验证视觉奖励的两种粒度 [D]

[G1](https://arxiv.org/abs/2505.13426) 构造 VLM-Gym：2048、Shisen-Sho、CIFAR variant、
Swap 等视觉游戏，共用 observation/action interface，支持多 game/scenario 与同状态多 actions
并行采样。G0 从弱 VLM 纯 RL self-evolve；G1 先做 perception-enhanced cold start/teacher
distillation，再 RL。论文分析感知与推理准确率相互 bootstrap。它说明 environment score 可
同时训练“看懂”和“行动”，但限定在这些可程序化游戏。

[SimpleSeg](https://arxiv.org/abs/2601.19228) 则不加 segmentation decoder，把 mask 边界
序列化为约几十个文本坐标点。SFT 学格式，随后以 sequence-level IoU reward 做 RL，直接
优化 contour closure/fidelity，并把 text/point/box/mask 组合成统一 localization interface。
这是像素级可验证奖励；**[U]** 它不能证明所有 dense vision task 都无需专用 decoder。

#### 4.5.4 WorldVQA 与 PerceptionBench：把错误源拆开 （已披露部分 [D]；未知部分 [U]）

[WorldVQA](https://arxiv.org/abs/2602.02537) 用 3,500 bilingual VQA pairs、九类 taxonomy，
只问视觉 stimulus 的 proper name/taxonomic name，排除 OCR、算术和 multi-hop retrieval，
目的是隔离“视觉原子知识”而非把 perception、memory、reasoning 混成一个总分。数据用 corpus
dedup、自动 consistency checks 与 human-in-the-loop dual gate。

[PerceptionBench](https://www.kimi.com/en/blog/perception-bench) 继续聚焦 atomic visual
perception 与 failure taxonomy，但截止日只有官方 web report，未链接 arXiv 或 dataset repo。
所以可记录其 [D] 评测主张，**[U]** 尚不能审计全部样本、污染、标注一致性与重跑工具。

### 4.6 Code、数学与 formal reasoning

#### 4.6.1 Kimina-Prover：把长 CoT 引入 Lean whole-proof generation [D]

[Kimina-Prover Preview](https://arxiv.org/abs/2504.11354) 从 Qwen2.5-72B 出发，约 20K
cold-start examples 把 informal reasoning 与 Lean snippets 交错在 `<think>` 中，最终 proof
放专用输出区；再每轮抽 1,000 problems、每题 8 rollouts，由 Lean compiler 给 binary reward，
沿 k1.5-family loss 做 RL。其定位是让模型内部长推理隐式探索 proof space，而不是依赖 BFS/
MCTS 外部搜索；蒸馏 1.5B/7B 版本公开。

[CombiBench](https://arxiv.org/abs/2505.03171) 补齐评测：100 个 Lean 4 combinatorics
problems，覆盖中学、IMO、大学与十余主题，并以 Fine-Eval 同时评估 proof 与 fill-in-the-blank。
发布时各模型在该域仍很弱，说明 miniF2F 总分不能代表组合数学。

#### 4.6.2 Kimi-Dev：agentless 不是 agent 的反面，而是技能先验 [D]

[Kimi-Dev](https://arxiv.org/abs/2509.23045) 从 Qwen2.5-72B-Base 做约 150B high-quality
real-world mid-training，把 SWE issue 解成 BugFixer 与 TestWriter，两者都训练 file localization
和 code edit；再做 cold start、verifiable RL、test-time self-play。仅用 5K public agent
trajectories SFT 即迁移到多轮 SWE agent。论文的核心可迁移结论是：在模块化、单轮、可验证
workflow 中先学定位/修复/反思，能降低端到端长 horizon agent 初始化难度；这不证明 fixed
workflow 本身等价于 agent。

#### 4.6.3 Kimi-Researcher：全轨迹结果奖励与异步系统 （已披露部分 [D]；未知部分 [U]）

[Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) 使用 parallel search、
text browser、code execution 三类工具；报告称平均 23 reasoning steps、每题探索 200+ URLs，
单 trajectory 可有 70+ searches、数十轮和数十万 context。训练数据一类强制特定工具依赖，
另一类为 math/code reasoning 与 hard search；主要用 REINFORCE，严格 on-policy 时关闭
tool-call format enforcer，并丢弃部分 negative samples 防 entropy collapse。

reward 包括 invalid tool/context/iteration 的 format penalty 与 against ground truth 的 correctness；
正确轨迹用 $r\gamma^{T-i}$ 鼓励更短探索。context manager 丢弃不必要 documents、保留关键信息；
fully asynchronous rollout、turn-level partial rollout/replay、Kubernetes hybrid cloud、stateful MCP
共同解决长尾环境等待，报告 partial rollout 至少 1.5× 加速。**[U]** 没有独立 PDF/arXiv，
base/RL weights 在报告时仍是未来开源计划；dataset、reward matcher、完整超参数与成本未知。

### 4.7 Agentic SFT、单 agent 与 agent swarm 的层级关系

可以把公开证据重建成五层，而不是笼统说“Kimi 会用工具”：

1. **[D] Tool vocabulary/data：** K2 的 3K+ real MCP 与 20K+ synthetic tools、agents、users、
   tasks、rubrics、trajectories。
2. **[D] Environment：** coding sandboxes、stateful simulator、随机 tool failures、async Gym、
   rule/test/judge/hack detector。
3. **[D] Single-agent policy：** k1.5/K2 的 long-horizon outcome RL、partial continuation 与
   high-quality pretraining loss 防 drift。
4. **[D] Interleaved thinking/tools：** [K2 Thinking](https://www.kimi.com/en/blog/kimi-k2-thinking)
   披露 256K、思考与 function call 交错、某些 search 任务 200–300 sequential calls、INT4 QAT；
   **[U]** 完整训练 recipe 未公开。
5. **[D] Learned orchestration：** K2.5/PARL 训练主 orchestrator spawn/coordinate 冻结 subagents；
   [Agent Swarm](https://www.kimi.com/en/blog/agent-swarm) 把产品 scale-out 与 shaping 机制单独披露。

[K2.6](https://www.kimi.com/en/blog/kimi-k2-6) 把产品上限扩到最多 300 subagents、4,000
coordinated steps，并展示 multi-hour/multi-day coding/design cases。**[U]** 这些 demonstrations
不是训练环境证明；K2.6 没有专属完整报告，new data、RL、optimizer、compute 仍未知。

### 4.8 评测、供应链验证、安全与披露边界

- **[D]** [Kimi Vendor Verifier](https://www.kimi.com/en/blog/kimi-vendor-verifier) 试图验证
  第三方 inference endpoint 是否按宣称模型/精度/功能服务，并开源评测制品；它是 deployment
  fidelity 研究，不是模型全面安全审计。
- **[D]** K2/K2.5 报告给出 safety evolved attacks 与 rubric rewards；Kimi-Researcher 的工具
  错误、iteration/context limit 进入 format reward；但这些都不等于生产 policy 全披露。
- **[D]** [ExtendAttack](https://doi.org/10.1609/aaai.v40i41.40833) 是 Moonshot-affiliated
  安全研究：通过 poly-base ASCII 等方式诱导 reasoning server 延长推理，暴露长度/时延型
  resource-exhaustion 攻击面。它不能被写成 Kimi 产品已被攻破，也不证明论文防御已部署。
- **[D]** K2 Thinking、K2.6、K3 与 web-only benchmarks 的 disclosure 明显少于 k1.5、
  K2、K2.5 论文；高层 model card/博客可支持其中写明的版本能力和协议。**[U]** 未写出的
  训练配方不能由模型名或相邻版本补全。
- **[U]** 跨模型 vendor benchmark 的 search steps、tools、reasoning caps、context overflow、
  judge 与样本版本差异极大。本章保留论文中的方法/协议事实，不用分数做“实验室总排名”。

## 5. 跨实验室综合：真正值得迁移的九个结论

### 5.1 不要只比较 total parameters；要比较每 token 激活、状态量与通信 （来源事实 [D]；综合判断 [I]）

| 轴 | DeepSeek | Moonshot/Kimi | 可迁移结论 |
|---|---|---|---|
| 条件计算 | V2 236B/21B；V3 671B/37B；V4 Flash 284B/13B、Pro 1.6T/49B | Moonlight ~16B/3B；K2/K2.5 1.04T/32.6B；K3 total 2.8T、active [U] | [I] total/active、experts selected、shared/dense 部分和路由拓扑必须一起报告 |
| 序列状态 | MLA → DSA → CSA/HCA；Engram 外置静态 memory | Mooncake cache hierarchy；MoBA block route；KDA recurrent state + MLA | [I] 长 context 成本至少含 attention FLOPs、KV/state bytes、selection 和 storage I/O 四项 |
| 深度连接 | mHC 扩 residual paths 后做 doubly-stochastic/Sinkhorn 约束 | AttnRes 学习对 earlier residual streams 加权 | [I] “更深”之外，信息如何跨层流动已成为独立 scaling 轴 |
| 训练通信 | DualPipe、EP8-node、FP8、DeepEP/3FS 生态 | PP16/EP16/ZeRO-1、Muon distributed state、CPU offload | [I] 参数表相似不代表训练效率相似，通信/精度实现决定可行区间 |

**[U]** 不能用 total/active 的简单比值推算任一模型单 token FLOPs：attention、embedding、
shared experts、MTP、vision tower、routing/indexer、sequence length 和 precision 都会改变结果。

### 5.2 两条长上下文路线不是互斥，而是位于不同层 （来源事实 [D]；综合判断 [I]）

DeepSeek 更连续地从 MLA 的**每 token KV 压缩**，走向 DSA 的**重要 token 选择**，再走向
CSA/HCA 的**先压缩再选择/密集处理**。Moonshot 同时研究 Mooncake 的**KV placement**、
MoBA 的**块级 query routing**、KDA 的**固定递归状态**。这些机制可以概念上叠加，却各有
信息丢失、训练可塑性和 kernel irregularity。

**[I]** 设计百万 token 系统时，应该先定位 workload：长 prefill 一次性阅读、长 decode、
多轮 agent short-append、高重复 prefix、还是训练 rollout。为 long-document prefill 最优的
sparse attention，不必是 multi-turn agent KV reload 最优；DualPath/Mooncake 正是后者。

### 5.3 优化器研究已经和 attention 稳定性绑定 （来源事实 [D]；综合判断 [I]）

Moonlight 解决 Muon update 的尺度、weight decay 与 distributed orthogonalization；K2 的
MuonClip 再单独约束 attention QK logits；Kimi Linear 继续使用 MuonClip；K3 只高层披露
Per-Head Muon。DeepSeek V3 以 AdamW + FP8 成功训练，V4 改用 Muon，同时加入 mHC、
SwiGLU clamp、routing rollback 与 anticipatory routing。

**[I]** “哪个优化器更好”不是脱离模型的单变量问题。matrix update geometry、attention-logit
growth、MoE route balance、low precision range 与 residual topology 形成耦合稳定系统。

### 5.4 数据 scaling 的共同方向是“运算”，不是静态清单 （来源事实 [D]；综合判断 [I]）

DeepSeek 的代表操作是跨 dump global dedup、domain classifier + URL-path iterative recovery、
repository dependency ordering、compiler/test filtering、autoformalization + verifier loop；Moonshot
是 embedding near-dedup、OCR/document filtering、semantic rephrasing、tool/task/user/rubric
synthesis、game/sandbox interaction 与 visual grounding generation。

**[I]** 高质量数据管线越来越像一个可运行程序：输入原始语料，调用 classifier、retriever、
LLM、compiler、theorem prover、sandbox、human gate，输出带 provenance 和难度估计的样本。
因此论文只报“多少 tokens”已不足以说明数据价值；至少还需 unique/processed、来源、重复、
合成次数、过滤器版本、验证率、污染控制与许可。

### 5.5 可验证奖励扩大能力边界，也重新定义了“任务可训练性” （来源事实 [D]；综合判断 [I]）

两家都优先扩展可机器判定任务：exact math、compiler/tests、Lean、IoU/coordinates、OCR edit
distance、game score、database/tool state。难验证任务则转向 learned matcher、self-critique、
rubric-guided generative judge。DeepSeek-GRM/SPCT 与 K2 self-critique 都试图让 judge 生成
理由/原则，而不只输出 scalar。

**[I]** Agent/RL 数据工程的核心问题由“能否写 prompt”变为：能否构造一个既难、可重置、
可并发、抗 hack、能区分部分策略但又不泄露答案的环境。V3.2 的 non-zero pass@100、
k1.5 的 no-CoT guessing test 与 K2 的 stateful failure simulator 都是这个问题的不同回答。

### 5.6 两家的最终 consolidation 揭示了两种抗遗忘路线 （来源事实 [D]；综合判断 [I]）

DeepSeek V3.2 在 specialists 蒸馏后做 mixed GRPO，使 reasoning、agent、alignment 同时更新；
V4 改为 student-on-policy、multi-teacher full-vocabulary reverse-KL OPD。Moonshot K2/K2.5
除了 task RL，还混入 high-quality pretraining loss 防 drift，并用 joint text-vision RL、Toggle
budget 和 PARL 逐层增加能力。

**[I]** 专家化不是终点：部署模型必须解决不同 teacher/reward/effort mode 的冲突。关键
评测应从“单领域峰值”转向 consolidation 后的 retention、calibration、style/length、tools 与
safety trade-off；清单中两家均未把完整权重/冲突矩阵公开。

### 5.7 Agent 能力的上限同时受 policy、environment 与 state system 限制 （来源事实 [D]；综合判断 [I]）

DeepSeek V3.2/V4 强调可重放 environment、deployment-equivalent precision、route/mask fidelity、
preemption-safe million-token trajectory；Moonshot K2/K2.5/Researcher 强调 tool synthesis、
async Gym、partial rollout、context manager、checkpoint streaming 与 learned orchestration。

**[I]** 一个 agent benchmark 的结果至少是

$$
f(\text{policy},\text{tool set},\text{environment version},\text{context manager},
\text{budget},\text{judge},\text{serving reliability}),
$$

而非只属于 model weights。没有这些元数据，跨实验室分数比较无法归因。

### 5.8 多模态路线显示“统一”有至少三种含义 （来源事实 [D]；综合判断 [I]）

1. **共享 decoder、分开 encoder：** Janus 让理解/生成各用适合的视觉编码，却进同一
   autoregressive Transformer。
2. **共享 language space/output：** SimpleSeg 用文本 coordinates 输出 mask；Kimi-Audio
   同时生成 text 与 semantic audio tokens。
3. **共享 foundation model 与 agent loop：** K2.5 把 vision、reasoning、IPython/tools、
   computer-use 与 PARL 放在一个后训练/服务体系。

**[I]** “原生多模态”不能只看是否能接图像；还要问视觉 tokenization、时空分辨率、output
space、RL reward、工具观察和长期 context 是否共同训练。DeepSeek OCR 的 optical compression
与 Moonshot KDA 的 sequence memory 进一步说明 modality 与 context efficiency 正在合流。

### 5.9 披露质量本身应该成为评估维度 （来源事实 [D]；综合判断 [I]）

DeepSeek V3/R1/V3.2/V4 和 Moonshot k1.5/K2/K2.5 提供相对丰富的 architecture/data/system/
post-training 细节；V3.2-Exp、Kimi-Researcher 是 repo/web-hosted report；K2 Thinking、K2.6、
K3 与 PerceptionBench 在截止日主要是 web disclosure。**[I]** 能力主张的可置信度不只看来源
是否官方，还看它是否给出 checkpoint、prompt、sampling、tools、judge、denominator、日期和
可下载 artifact。官方博客仍是一手来源，但不能替代缺失的实验协议。

## 6. 公开程度与关键未知：逐阶段审计

| 阶段 | DeepSeek | Moonshot/Kimi | 结论 |
|---|---|---|---|
| Base architecture | V2/V3/V4 参数表、MLA/MoE/attention 较详 | K2/K2.5 详；k1.5 size [U]；K3 active [U] | （已披露部分 [D]；未知部分 [U]） 版本差异必须逐 checkpoint 记录 |
| Pretraining data | token exposure 与部分操作；exact corpus/许可/unique [U] | k1.5/K2/K2.5/Kimi-Audio 类别与操作；比例/许可/unique 多 [U] | [U] 无法端到端重建任一旗舰语料 |
| Optimization | V3/V4 schedule、精度、并行较详 | Moonlight/K2 schedule/并行较详；K3 仅高层 | （已披露部分 [D]；未知部分 [U]） failed runs、完整 optimizer state 未公开 |
| SFT | 多数旗舰给类别/部分样本量 | k1.5 较详；K2/K2.5 分类详但总量缺 | （已披露部分 [D]；未知部分 [U]） prompts、mixture weights、human process 不完整 |
| Rewards/RM | rule、help/safety RM、GRM/rubric；weights 多缺 | rule/test/matcher/self-critique/safety；weights 多缺 | [U] 不能重建完整 reward function |
| Online RL | GRPO/R1/V3.2、V4 specialists 细节不均 | k1.5 objective、K2 extensions、K2.5 clipping/PARL | （已披露部分 [D]；未知部分 [U]） rollout token 总量与多数超参缺失 |
| Distillation/merge | R1 rejection/distillation；V3.2 mixed；V4 OPD | long-to-short、K2/K2.5 teacher candidates；merge 细节有限 | （已披露部分 [D]；未知部分 [U]） teacher conflicts/selection bias 未公开 |
| Long-context training | 各阶段长度、DSA/V4 curriculum | k1.5/Kimi Linear/K2.5 部分 curriculum | （已披露部分 [D]；未知部分 [U]） advertised API limit 不自动等于 training length |
| Serving/system | Fire-Flyer、V3、DualPath、V4、DSpark，选定代码 | Mooncake、k1.5/K2/K2.5 infra，选定代码 | （已披露部分 [D]；未知部分 [U]） live topology/routing/monitoring 不公开 |
| Evaluation | 大量 vendor tables，协议细节不均 | 大量 vendor tables + Vendor Verifier/atomic benchmarks | （已披露部分 [D]；未知部分 [U]） 非统一 harness，不能直接总排名 |
| Safety/governance | mixed safety RL；一篇机构署名治理观点 | safety rewards；ExtendAttack；endpoint verification | [U] 生产 safety stack 与事故数据不足 |
| Reproduction | weights/参考 inference/部分 kernels | weights/code/toolkits 依版本开放 | [U] 本章无 [R] 结果；公开 artifact ≠ 完整训练复现 |

## 7. 按研究域重建的“因果图”，而非品牌年表

```text
DeepSeek
global dedup / repo-code / math recovery
  ├─> DeepSeekMoE ─> V2 (MoE + MLA) ─> V3 (FP8 + MTP + DualPipe)
  │                                      ├─> R1 long-reasoning RL
  │                                      └─> V3.2 DSA + agent environments
  │                                            └─> V4 mHC + CSA/HCA + Muon + OPD
  ├─> Math ─> Prover ─> Prover-V1.5 ─> Prover-V2 / Math-V2
  ├─> Coder ─> Coder-V2 ─> V3.2 agentic coding
  ├─> VL ─> VL2
  ├─> Janus ─> JanusFlow / Janus-Pro
  └─> OCR optical compression ─> OCR2 visual causal flow

Moonshot / Kimi
Mooncake long-context serving
  ├─> k1.5 multimodal long-CoT RL ─> K2 agent RL ─> K2 Thinking
  │                                      └─> K2.5 visual RL + PARL ─> K2.6
  ├─> Moonlight distributed Muon ─> K2 MuonClip ─> Kimi Linear / KDA
  │                                             └─> K3 KDA + Per-Head Muon
  ├─> MoBA block sparse attention
  ├─> Kimi-VL / G1 / SimpleSeg / WorldVQA ─> K2.5 native vision
  ├─> Kimi-Audio
  ├─> Kimina-Prover ─> CombiBench
  ├─> Kimi-Dev agentless skill prior
  └─> Kimi-Researcher async end-to-end agent RL ─> K2.5/PARL systems line
```

箭头只表示报告能支持的技术/时间继承或清晰的方法前驱；**不表示未披露的 weight initialization、
代码复制或组织因果**。例如 MoBA 与 KDA 都是 Moonshot 研究，却没有证据把 MoBA 画进 K2；
Engram 是 DeepSeek 研究，却没有证据把它画进 V4。

## 8. 逐条记录注释地图：DeepSeek 37 条

下表是对库存逐项的审计索引。`core/direct/affiliated` 是归属强度，不是论文质量等级；每一行
的链接均为清单保存的 primary URL。短注释只概括该记录最独特的研究贡献和边界，不能替代
原文实验协议。

### 8.1 Core：旗舰模型、专用模型与正式技术报告（22）

| 日期 / 记录 | 论文或报告 | 注释与证据边界 |
|---|---|---|
| 2024-01-05 `deepseek-2401.02954` | [DeepSeek LLM](https://arxiv.org/abs/2401.02954) | [D] 7B/67B、2T exposure、scaling law、全局 web dedup、1.5M SFT 与 DPO；[U] exact corpus/总成本。 |
| 2024-01-11 `deepseek-2401.06066` | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | [D] 细粒度 routed experts 与 shared experts，建立 V2 以后 MoE 的结构祖先。 |
| 2024-01-25 `deepseek-2401.14196` | [DeepSeek-Coder](https://arxiv.org/abs/2401.14196) | [D] 仓库依赖排序、87% code mixture、FIM、4K→16K 和 executable/quality filtering。 |
| 2024-02-05 `deepseek-2402.03300` | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | [D] 迭代恢复 120B-token math corpus、500B sampled exposure、776K SFT 与 GRPO 首次公开。 |
| 2024-03-08 `deepseek-2403.05525` | [DeepSeek-VL](https://arxiv.org/abs/2403.05525) | [D] 现实场景视觉语料、混合 encoder、高分辨率固定 token budget、language-first 联合训练。 |
| 2024-05-07 `deepseek-2405.04434` | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | [D] 236B/21B、MLA + DeepSeekMoE、8.1T、128K、两阶段 online GRPO。 |
| 2024-05-23 `deepseek-2405.14333` | [DeepSeek-Prover](https://arxiv.org/abs/2405.14333) | [D] 自然语言竞赛题到 8M Lean statement-proof pairs 的合成/验证闭环。 |
| 2024-06-17 `deepseek-2406.11931` | [DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931) | [D] 从 V2 4.2T 中间点继续 6T，338 languages、128K，code/math verifier/RM 后训练。 |
| 2024-08-15 `deepseek-2408.08152` | [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152) | [D] RLPAF、truncate-and-resume 与 intrinsic-reward RMaxTS，把 Lean feedback 接回 whole-proof generation。 |
| 2024-10-17 `deepseek-2410.13848` | [Janus](https://arxiv.org/abs/2410.13848) | [D] 理解/生成视觉编码解耦、共享 autoregressive Transformer 的统一多模态框架。 |
| 2024-11-12 `deepseek-2411.07975` | [JanusFlow](https://arxiv.org/abs/2411.07975) | [D] 将 rectified flow 接入统一 LLM 框架并对齐两种视觉 representation。 |
| 2024-12-13 `deepseek-2412.10302` | [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) | [D] dynamic tiling 与 MoE VLM，1.0B/2.8B/4.5B active 三档，强化 OCR/grounding/document。 |
| 2024-12-27 `deepseek-2412.19437` | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [D] 671B/37B、14.8T、FP8、MTP、loss-free balance、DualPipe、specialist post-training。 |
| 2025-01-22 `deepseek-2501.12948` | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | [D] 分开记录无 SFT 的 R1-Zero 与 cold-start/两段 SFT/两段 RL 的生产 R1，含 distillation。 |
| 2025-01-29 `deepseek-2501.17811` | [Janus-Pro](https://arxiv.org/abs/2501.17811) | [D] 调整 Janus 训练策略、扩数据并由 1B 放大至 7B；不等同 JanusFlow。 |
| 2025-04-30 `deepseek-2504.21801` | [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) | [D] V3 subgoal decomposition + 7B proof search + Lean binary-RL，统一 informal/formal reasoning。 |
| 2025-09-29 `deepseek-v3.2-exp-report` | [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | [D] repo-hosted 独立报告；DSA indexer + top-k selected attention，经约 946B tokens 转换。 |
| 2025-10-21 `deepseek-2510.18234` | [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) | [D] DeepEncoder + 570M-active decoder，以视觉 token 对文档文本做 2D optical compression。 |
| 2025-11-27 `deepseek-2511.22570` | [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) | [D] proof generator/verifier 共演化与扩展 verification compute；self-verifiable 不等于形式证明。 |
| 2025-12-01 `deepseek-2512.02556` | [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf) | [D] specialist RL + mixed GRPO、thinking-with-tools、85,267 tasks、>1,800 envs 与 rollout fidelity。 |
| 2026-01-28 `deepseek-2601.20552` | [DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552) | [D] LLM-style DeepEncoder V2 与 causal-flow queries 学习视觉 reading order。 |
| 2026-04-24 `deepseek-2606.19348` | [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | [D] Flash/Pro、1M、mHC、CSA/HCA、Muon、FP4、specialist GRPO + multi-teacher OPD；[U] 硬件/成本。 |

### 8.2 Direct：架构、优化、系统与评测方法（9）

| 日期 / 记录 | 论文 | 注释与证据边界 |
|---|---|---|
| 2024-07-02 `deepseek-2407.01906` | [Expert-Specialized Fine-Tuning](https://arxiv.org/abs/2407.01906) | [D] 观察不同任务路由集中到不同 experts，仅微调任务相关专家；是 MoE PEFT 研究，不证明旗舰默认使用 ESFT。 |
| 2024-08-26 `deepseek-2408.14158` | [Fire-Flyer AI-HPC](https://arxiv.org/abs/2408.14158) | [D] 10K PCIe A100、HFReduce、3FS/HaiScale 与软硬件协同的 SC24 系统论文。 |
| 2025-02-16 `deepseek-2502.11089` | [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | [D] compression/selection/sliding 三路、原生 trainable、硬件对齐；[U] 不是 DSA 的同义词。 |
| 2025-04-03 `deepseek-2504.02495` | [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) | [D] SPCT、生成原则/批评、parallel reward sampling 与 meta-RM voting。 |
| 2025-05-14 `deepseek-2505.09343` | [Insights into DeepSeek-V3](https://arxiv.org/abs/2505.09343) | [D] ISCA 产业论文，从 MLA/MoE/FP8/network bottleneck 提出下一代硬件方向。 |
| 2025-12-31 `deepseek-2512.24880` | [mHC](https://arxiv.org/abs/2512.24880) | [D] 以流形/Sinkhorn 约束 Hyper-Connections，恢复 identity/stability 并优化 memory access。 |
| 2026-01-12 `deepseek-2601.07372` | [Conditional Memory / Engram](https://arxiv.org/abs/2601.07372) | [D] $O(1)$ hashed n-gram lookup 作为 MoE 之外的 sparsity axis，研究 compute-memory allocation。 |
| 2026-02-25 `deepseek-2602.21548` | [DualPath](https://arxiv.org/abs/2602.21548) | [D] 利用 decode-side storage NIC，经 RDMA 双路径加载 agentic session KV。 |
| 2026-07-06 `deepseek-2607.05147` | [DSpark](https://arxiv.org/abs/2607.05147) | [D] semi-autoregressive draft 与 confidence/load-aware verification，报告部署于 V4 serving。 |

### 8.3 Affiliated：明确机构关联、但不并入旗舰主线（6）

| 日期 / 记录 | 论文 | 注释与归属边界 |
|---|---|---|
| 2023-10-25 `deepseek-2310.16818` | [DreamCraft3D](https://arxiv.org/abs/2310.16818) | [D] hierarchical 3D generation 与 bootstrapped diffusion prior；官方托管 ICLR implementation，但不是 DeepSeek LLM 报告。 |
| 2024-01-16 `deepseek-doi-10.1109-tbdata.2025.3618474` | [The Faiss Library](https://arxiv.org/abs/2401.08281) | [D] 向量搜索、聚类、压缩与索引权衡；一位作者有 DeepSeek 署名，不能推断 Faiss 是旗舰训练栈。 |
| 2024-09-11 `deepseek-doi-10.1145-3650212.3680327` | [Synthesis-Based Enhancement for GUI Test Case Migration](https://doi.org/10.1145/3650212.3680327) | [D] program synthesis 辅助 GUI test migration；机构署名软件工程研究，当前 inventory 另存作者公开 PDF。 |
| 2024-11-01 `deepseek-doi-10.1109-tia.2023.3344544` | [Coherent Hierarchical Probabilistic Forecasting of EV Charging Demand](https://arxiv.org/abs/2411.00337) | [D] PICNN conditional distribution + differentiable reconciliation；非 LLM 机构署名研究。 |
| 2025-10-09 `deepseek-doi-10.1126-science.ady7922` | [China's emerging regulation toward an open future for AI](https://doi.org/10.1126/science.ady7922) | [D] AI 治理/开放生态观点；policy perspective 不是技术报告，且无公开 PDF。 |
| 2025-11-10 `deepseek-doi-10.1145-3746252.3761392` | [CCAgent](https://doi.org/10.1145/3746252.3761392) | [D] Web3 协作式数据扩展的 OS-agent 研究；单一机构署名关联，不能投射到 V3.2/V4。 |

## 9. 逐条记录注释地图：Moonshot AI / Kimi 32 条

### 9.1 Core：旗舰、专用模型与正式/实质研究报告（13）

| 日期 / 记录 | 论文或报告 | 注释与证据边界 |
|---|---|---|
| 2025-01-20 `kimi-2501.12599` | [Kimi k1.5](https://arxiv.org/abs/2501.12599) | [D] 多模态 128K、~2M SFT、长 CoT value-model-free RL、partial rollout、long-to-short；[U] 参数/成本。 |
| 2025-04-10 `kimi-2504.07491` | [Kimi-VL](https://arxiv.org/abs/2504.07491) | [D] 16B/2.8B active MoE、MoonViT、128K、高分辨率/长视频/agent 与 long-thinking RL。 |
| 2025-04-15 `kimi-2504.11354` | [Kimina-Prover Preview](https://arxiv.org/abs/2504.11354) | [D] 72B、formal reasoning pattern、20K cold start、Lean binary reward RL，蒸馏 1.5B/7B。 |
| 2025-04-26 `kimi-2504.18425` | [Kimi-Audio](https://arxiv.org/abs/2504.18425) | [D] 12.5Hz hybrid audio tokenizer、13M+ hours、text/audio shared LLM、flow-matching streaming detokenizer。 |
| 2025-06-17 `kimi-2509.23045` | [Kimi-Dev](https://arxiv.org/abs/2509.23045) | [D] agentless SWE mid-training/RL 形成 localization/edit/reflection 技能先验，再以 5K trajectories 转 agent。 |
| 2025-06-20 `kimi-researcher-report` | [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) | [D] search/browser/code 的 end-to-end REINFORCE、context management、async/turn-partial rollout；[U] 无独立 PDF/arXiv。 |
| 2025-07-11 `kimi-2507.20534` | [Kimi K2](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) | [D] 1.04T/32.6B、384 experts、MuonClip、15.5T、agentic SFT/Gym/RL；[U] 总硬件/成本。 |
| 2025-09-05 `kimi-k2-instruct-0905-note` | [Kimi-K2-Instruct-0905](https://platform.kimi.com/blog/posts/kimi-k2-0905) | [D] 256K、agent coding/frontend 版本更新；[U] 无独立完整报告，更新训练配方未知。 |
| 2025-10-30 `kimi-2510.26692` | [Kimi Linear](https://arxiv.org/abs/2510.26692) | [D] KDA+MLA hybrid、48B/3B、5.7T、1M、MuonClip 与 k1.5-family post-training。 |
| 2025-11-06 `kimi-k2-thinking-note` | [Kimi K2 Thinking](https://www.kimi.com/en/blog/kimi-k2-thinking) | [D] 1T/32B、256K、reasoning/tool interleave、INT4 QAT；[U] SFT/RL counts/compute。 |
| 2026-01-27 `kimi-2602.02276` | [Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) | [D] K2+~400M vision、~15T continuation、joint visual RL、Toggle、PARL、100K concurrent tasks。 |
| 2026-04-20 `kimi-k2.6-note` | [Kimi K2.6](https://www.kimi.com/en/blog/kimi-k2-6) | [D] long-horizon coding/design 与 300 subagents/4,000 steps 产品范围；[U] K2.6-specific recipe/report。 |
| 2026-07-17 `kimi-k3-note` | [Kimi K3](https://www.kimi.com/en/blog/kimi-k3) | [D] 2.8T、896/16 experts、1M、KDA/AttnRes/LatentMoE、MXFP4/8；[U] active、data、训练/RL。 |

### 9.2 Direct：优化、attention、系统、agent、视觉方法与 benchmark（11）

| 日期 / 记录 | 论文或报告 | 注释与证据边界 |
|---|---|---|
| 2024-06-24 `kimi-2407.00079` | [Mooncake](https://arxiv.org/abs/2407.00079) | [D] prefill/decode disaggregation、GPU/CPU/SSD KV cache、RDMA、locality/SLO scheduler 与 overload rejection。 |
| 2025-02-18 `kimi-2502.13189` | [MoBA](https://arxiv.org/abs/2502.13189) | [D] block-mean route + selected-block FlashAttention，研究延至 1M；[U] K2/K2.5 未披露使用。 |
| 2025-02-23 `kimi-2502.16982` | [Muon is Scalable / Moonlight](https://arxiv.org/abs/2502.16982) | [D] Muon scale/decay、distributed orthogonalization、~16B/3B MoE 训练 5.7T。 |
| 2025-05-06 `kimi-2505.03171` | [CombiBench](https://arxiv.org/abs/2505.03171) | [D] 100 个 Lean 4 combinatorics problems 与 Fine-Eval proof/fill-blank protocol。 |
| 2025-05-19 `kimi-2505.13426` | [G1 / VLM-Gym](https://arxiv.org/abs/2505.13426) | [D] 多视觉游戏 parallel RL、perception cold start，研究感知与推理互相 bootstrap。 |
| 2026-01-22 `kimi-vendor-verifier-note` | [Kimi Vendor Verifier](https://www.kimi.com/en/blog/kimi-vendor-verifier) | [D] endpoint/model-deployment fidelity 的开放评测 artifact；不是全面 model safety report。 |
| 2026-01-27 `kimi-2601.19228` | [SimpleSeg](https://arxiv.org/abs/2601.19228) | [D] decoder-free coordinate sequence segmentation，以 SFT→IoU-RL 解锁 pixel-level perception。 |
| 2026-02-03 `kimi-2602.02537` | [WorldVQA](https://arxiv.org/abs/2602.02537) | [D] 3,500 atomic visual-knowledge pairs，隔离 entity naming、排除 OCR/推理 confounder。 |
| 2026-02-09 `kimi-agent-swarm-note` | [Agent Swarm](https://www.kimi.com/en/blog/agent-swarm) | [D] K2.5 PARL 的独立研究预览：冻结 subagents、训练 orchestrator、annealed shaping。 |
| 2026-03-16 `kimi-2603.15031` | [Attention Residuals](https://arxiv.org/abs/2603.15031) | [D] 对 earlier residual streams 做 learned normalized routing，blockwise 降内存；K3 参数化 [U]。 |
| 2026-07-16 `kimi-perceptionbench-note` | [PerceptionBench](https://www.kimi.com/en/blog/perception-bench) | [D] atomic visual perception/failure taxonomy web report；[U] 截止日无 arXiv/dataset repo。 |

### 9.3 Affiliated：明确机构署名的广义研究（8）

| 日期 / 记录 | 论文 | 注释与归属边界 |
|---|---|---|
| 2024-03-01 `kimi-2403.00522` | [VisionLLaMA](https://arxiv.org/abs/2403.00522) | [D] plain/pyramid LLaMA-style vision Transformer，覆盖 perception/generation；一位作者 Moonshot 署名，非 Kimi 模型报告。 |
| 2024-03-22 `kimi-doi-10.1145-3726302.3730334` | [SAGraph](https://arxiv.org/abs/2403.15105) | [D] 345,039 Weibo users、interaction/content context 的 influencer-selection dataset；机构署名 data-mining 研究。 |
| 2025-03-06 `kimi-2503.04606` | [LanDiff](https://arxiv.org/abs/2503.04606) | [D] semantic tokenizer + autoregressive LM + streaming diffusion 的 coarse-to-fine video generation。 |
| 2025-03-13 `kimi-2503.10078` | [Image Quality Assessment: From Human to Machine Preference](https://arxiv.org/abs/2503.10078) | [D] 定义 machine-vision IQA，发布含 2.25M annotations 的 MPD；不能推断进入 Kimi 数据。 |
| 2025-04-11 `kimi-2407.02906` | [Single Image Rolling Shutter Removal with Diffusion Models](https://doi.org/10.1609/aaai.v39i9.33015) | [D] RS-Diffusion、patch attention 与真实 RS/GS+IMU dataset；出版版新增 Moonshot 署名作者。 |
| 2025-06-20 `kimi-doi-10.1145-3695053.3731013` | [ANSMET](https://doi.org/10.1145/3695053.3731013) | [D] near-memory approximate-nearest-neighbor search 与 hybrid early termination；机构署名系统研究，当前 inventory 另存作者公开 PDF。 |
| 2026-01-23 `kimi-2601.16694` | [Affinity Contrastive Learning for Skeleton-based Human Activity Understanding](https://arxiv.org/abs/2601.16694) | [D] ACLNet 的 class-affinity、dynamic temperature、margin contrast；广义 CV 署名研究。 |
| 2026-03-14 `kimi-2506.13737` | [ExtendAttack](https://doi.org/10.1609/aaai.v40i41.40833) | [D] 通过复杂编码延长 reasoning 造成 resource exhaustion；安全研究，不代表 Kimi 产品漏洞。 |

## 10. 研究议程：从这两条谱系还能严格推出什么问题

以下问题由公开证据综合提出，属于 **[I]**；在本来源宇宙中的答案仍为 **[U]**，不是对
生产系统的事实断言。

1. **Sparsity allocation：** MoE experts、Engram/static memory、KDA recurrent state、selected
   KV、residual paths 应如何在固定 FLOPs/HBM/network/capacity 下联合分配？现有论文多一次
   只比较一到两轴。
2. **百万 token 学习而非仅检索：** needle/retrieval 可通过 sparse/full hybrid 提升，但模型
   是否能在 1M trajectory 中稳定 credit assignment、更新计划并从 tool failures 学习，需要
   与 retrieval 分离的评测。
3. **On-policy 的系统定义：** route、sampling support、quantization、tool version、context
   compaction、partial-rollout policy lag 哪些偏差可用 importance correction，哪些会改变 MDP？
4. **Generative judge 的闭环风险：** 当 policy、rubric generator、critic、meta-RM 来自同一
   backbone 家族，如何测 correlated blind spot、reward hacking 和置信度校准？
5. **Specialist consolidation：** mixed RL、reverse-KL OPD、pretraining-loss regularization、
   weight averaging 在相同 teacher/data/compute 下怎样比较 retention 与 mode interference？
6. **视觉 agent 的坐标/像素接口：** SimpleSeg、K2.5 grounding rewards、OCR causal flow 能否
   在真实 GUI/tool loop 中减少定位错误，需 end-to-end task success 而非静态 IoU alone。
7. **音频 agent 与长上下文：** Kimi-Audio 已给统一 speech/sound/music 模型，但当前库存未把
   它与 K2.5/PARL 或 KDA/Mooncake 的长期交互系统连接；DeepSeek 则无核心音频报告。
8. **形式证明到开放数学：** Lean binary oracle 很强，却依赖 formalization 与 mathlib；
   Math-V2/general judge 更开放却不严格。如何把形式验证嵌入自然语言研究流程仍未解决。
9. **成本核算：** 论文常报 final run GPU-hours 或不报。真正可比的总成本应含数据、失败 runs、
   post-training rollouts、sandboxes、storage/network、评测、工程人力和能源。
10. **持续安全披露：** resource-exhaustion、tool injection、sandbox escape、data exfiltration、
    subagent amplification、judge manipulation 都需要版本化 threat model 与 incident evidence；
    当前公开资料不足以证明任一体系“已解决安全”。

## 11. 使用本章时应避免的具体误述

- “GRPO 是 R1 发明的。”——[DeepSeekMath](https://arxiv.org/abs/2402.03300) 已先公开。
- “DeepSeek-R1 完全没 SFT。”——只有 R1-Zero 无 SFT；生产 R1 是四阶段管线。
- “DeepSeek V3 的全部研发只花 \$5.6M。”——这是报告列明 runs 的租赁等价估算。
- “DeepSeek V3.2 有 85K 个环境。”——是 85,267 tasks、超过 1,800 environments。
- “V4 最终模型由 mixed GRPO 合并。”——specialists 用 GRPO，最终 consolidation 是 OPD。
- “NSA 就是 DSA。”——前者是三路原生可训练研究架构，后者是 V3.2-Exp lightning indexer
  选择机制；公开证据不支持等同。
- “K2/K2.5 使用 MoBA。”——报告披露 MLA；MoBA 是独立研究。
- “K3 active params 等于 2.8T×16/896。”——忽略 attention/shared/embedding/LatentMoE，
  active 数在截止日 [U]。
- “K2.5 共训练了 30T unique tokens。”——约 30T 只是 K2 + continuation 的 [I]
  cumulative processed exposure。
- “Agent Swarm 的收益只来自多花 token。”——PARL 训练了 orchestration policy，但公平比较
  仍必须控制总 compute、wall time、tools 与 judge；公开数据尚不足以给出单因果归因。
- “官方 benchmark 分数可以直接横向排序。”——不同 prompt、sampling、tools、context、
  overflow、judge 与 API snapshot 使其不可无条件比较。
- “core/direct/affiliated 都代表同一实验室主线。”——`affiliated` 只证明机构署名/官方托管；
  本章没有把 EV forecast、rolling-shutter、policy perspective 等倒灌进旗舰训练配方。
- “本章已经证明穷尽两家所有论文。”——它只对截止日冻结、按发现规则收录的 69 条记录
  逐项覆盖；affiliation index、未来版本和未公开工作仍可能造成遗漏。

## 12. 完整性声明与更新规则

**[D]** 本章已对当前 inventory 的 DeepSeek 37 条、Moonshot/Kimi 32 条逐条给出链接与短注；
其中当前 inventory 有 59 条非空公开 PDF URL，10 条未找到公开 PDF。发现过程同时检查两家一手研究
索引、官方 GitHub/Hugging Face、arXiv、Crossref 结构化 affiliation 与 OpenAlex raw
affiliation，并明确排除只“提到/评测 DeepSeek 或 Kimi”的论文、把模型列作 AI coauthor 的
记录、Alphabet X 的 “Moonshot” false positives、重复的 conference/journal 版本，以及只有
软件/权重/短 release note 而无独立出版型内容的仓库。

**[U]** affiliation discovery 天生不完备：publisher 可能不存 affiliation，arXiv 没有统一结构化
机构字段，官方 index 也可能漏旧工作。后续维护时应：

1. 保留 **verified-through date**，不静默把新 checkpoint 混入旧报告；
2. 先更新 inventory 与 provenance，再改本章数字；
3. arXiv/conference/Nature 等同一工作合并，除非是实质不同报告；
4. web-only K3/PerceptionBench/K2.6 若出现正式 paper，应保留旧披露日期并显式记录 supersede；
5. 任何跨模型数值比较都重新核对 variant、prompt、sampling、tools、context、judge 与日期；
6. 只有保留代码 revision、环境、数据 hash、命令、seed、raw logs 和 artifacts 的实测才标 [R]。

本章的目标不是用大量链接营造“全面感”，而是让读者能区分：**来源真的说了什么、公开制品
能确认什么、我们依据什么作了推断，以及在哪些地方应当诚实地停在未知。**
