# Zhipu AI / Z.ai 与 GLM：从空白填充到多模态 Agentic Engineering 的全研究谱系

**核验截止：2026-07-19（Asia/Shanghai）。** 本章以该日冻结的
`research/literature/inventory/glm.jsonl`、一手论文与技术报告、
官方研究页和仓库制品为证据。当前清单共 **95 条**：`core` 26 条、`direct` 39 条、
`affiliated` 30 条；按 inventory `type` 计，`research_paper` 54 条、`model_card` 21 条、
`technical_report` 10 条、`benchmark` 6 条、带实质报告的官方文章 2 条，另有其他研究制品 2 条。清单记录了
58 个公开 PDF 路由，当前可读归档与全文抽取各 57 份。全文数不是纳入边界：没有本地
PDF 的官方模型卡仍可是一手证据，
而只在论文中提到 GLM 的工作不会因此成为 Zhipu 研究。

这里的“全”有严格含义：**下文末尾对冻结清单 95/95 逐条给出 inventory ID 与
`primary_url`，且正文覆盖旗舰、架构、数据、优化、后训练、系统、多模态、代码、agents、
评测、安全和广义应用。** 它不声称能枚举私有、撤下、未索引或未披露的工作，也不把
所有 THUDM 论文等同于 Zhipu AI。

## 1. 怎么读：证据标签、归属边界和七个结论

本章沿用 [Research, Evidence, and Citation Standard](../research-method.md)：

- **[D] Disclosed / 已披露**：论文、技术报告、官方模型卡或一手研究页直接陈述。
- **[C] Confirmed artifact / 公开制品确认**：发布的配置、代码、权重或数据清单可直接观察。
- **[R] Reproduced / 已复现**：按可审计协议复跑并保留命令、配置、日志与制品；本章没有
  把任何模型能力或训练结果标为 [R]。
- **[I] Inferred / 推断**：从列明事实与假设推出，但来源没有这样宣称。
- **[U] Unknown / 未知**：公开证据缺失、冲突或不足以支持结论。

清单中的 `core` 是 GLM 旗舰或专用品牌模型的一手报告；`direct` 是 Zhipu/Z.ai 当前
官方仓库直接关联的方法、系统与评测；`affiliated` 只证明论文作者行中有明确 Zhipu/Z.ai
机构署名，或仓库明确写明受其支持。`affiliated` **不能**证明方法进入过 GLM 旗舰训练。

七个先行结论：

1. **[D] “GLM”不是一种从 2021 年原封不动延续至今的架构。** 初代 GLM 用自回归
   blank infilling、双向 Part A 与自回归 Part B；GLM-130B 把它扩到 130B dense；ChatGLM
   逐步转向对话、Multi-Query Attention（MQA）和原生工具协议；GLM-4.5 首次公开
   Mixture-of-Experts（MoE）旗舰；GLM-5 再改用 Multi-head Latent Attention（MLA）、
   DeepSeek Sparse Attention（DSA）与 Multi-Token Prediction（MTP）。产品家族名保持，
   计算图、训练目标与系统约束却已多次改变。

2. **[D] 扩展不只发生在参数量。** 公开主线依次披露约 400B、约 10T、23T、28.5T
   processed training tokens；上下文从 2K、8K/32K、128K、200K 到 1M；服务端又引入
   稀疏 token selection、跨层 index reuse、KV cache 管理和异步长轨迹。把进展归因于
   “模型更大”会漏掉数据、优化、稀疏性与系统共设计。

3. **[D] Zhipu 的后训练主线从高层 RLHF 披露走向可执行环境。** ChatGLM/GLM-4 只给
   SFT、human preference、RLHF 的轮廓；GLM-4.5 才详细描述 reasoning/general/agent
   experts、SWE/web 环境、自蒸馏与 Slime；GLM-5 披露顺序 RL、异步 rollout、跨阶段
   on-policy distillation；GLM-5.2 又针对 compaction 后不等长子轨迹改用 single-rollout、
   critic-based PPO。精确奖励权重和生产规模仍多为 [U]。

4. **[D] 多模态不是语言旗舰的附录，而是并行演化的研究树。** CogView→CogView2→
   Relay Diffusion/CogView3→CogView4/GLM-Image 是图像生成线；CogVideo→CogVideoX→
   RealVideo/Kaleido/SCAIL 是视频与角色动画线；CogVLM→CogVLM2/GLM-4V→GLM-4.5V/
   4.6V/5V 是理解与工具线；GLM-4-Voice、GLM-ASR、GLM-TTS、GLM-OCR 则把音频与
   文档结构纳入统一 token/decoder 或专用小模型系统。

5. **[D] 代码、数学与 agent 能力共享“可验证反馈”支点，但不是同一算法。** CodeGeeX
   依赖编译/测试和 HumanEval-X；ReST-RL 将 self-training、GRPO、value-model-guided
   decoding 组合；GLM-4.5/5 在真实仓库、terminal、search 和 GUI 环境训练；GLM-TTS
   与 GLM-OCR 也使用多奖励或规则奖励。奖励可验证不等于奖励无漏洞，GLM-5/5.2
   专门讨论了 sandbox、test leakage 与 anti-hacking。

6. **[D/I] 该研究谱系比 LLM 更宽，但归属必须克制。** CogDL、GE-SpMM、LPS-GNN、
   OAG-BERT/OAG-Bench、推荐、知识图谱、医疗分期、水利工程与 social-abuse detection
   都出现在 95 条里；其中不少只是工作级机构署名。它们说明组织研究宽度 [D]，却不
   证明相应数据或方法进入 GLM [U]。

7. **[U] 公开材料不足以端到端复现任一最新旗舰，也不足以给出跨供应商总排行榜。**
   原始语料清单与许可、完整训练代码、失败 runs、SFT/RM 数据、reward composition、
   生产路由、安全运营、总硬件时长与总成本均不完整。官方 benchmark 数字只能在各自
   checkpoint、prompt、sampling、tools、context、judge 和日期下解读，不能无条件横排。

### 1.1 95 条实际覆盖哪些研究域

下表各域重叠，不能相加成 95；它用于防止把“实验室研究”缩成旗舰 LLM 或 Agentic RL。

| 研究域 | 清单中的主要一手节点 |
|---|---|
| 基础模型、架构与 scaling | GLM、GLM-130B、ChatGLM、GLM-4、GLM-4.5、GLM-5/5.1/5.2、GLM-Edge |
| 数据、预训练与优化 | blank infilling、bilingual corpus、Muon/Muon Split、repo/issue/PR 数据、MEPipe |
| 后训练、RL 与蒸馏 | ImageReward/VisionReward、ReST-RL、MobileRL、GLM-4.5/5/5.2、Slime |
| 长上下文与系统 | 128K/200K/1M、DSA、IndexCache、ZCube、coding-agent serving、GE-SpMM/LPS-GNN |
| 图像/视频生成 | CogView 1-4、Relay Diffusion、Inf-DiT、CogVideo/X、RealVideo、Kaleido、SCAIL 1/2、GLM-Image |
| 视觉理解与 GUI | VisualGLM、CogVLM/2、CogCoM、CogAgent、GLM-4V/4.5V/4.6V/5V、Vision2Web |
| 语音与文档 | GLM-4-Voice、GLM-ASR-Nano、GLM-TTS、GLM-OCR |
| 代码、数学与工具 | CodeGeeX 1/2/4、HumanEval-X、ComplexFuncBench、UI2Code、AutoGLM、AutoWebGLM |
| benchmark/evaluation | GLM-SIMPLE-EVALS、LVBench、MotionBench、RPC-Bench、OAG-Bench、HoWToBench |
| 安全、隐私与鲁棒 | GLM-130B ethics、GLM-4.5 safety eval、anti-hacking、JPS、GuidedLatent、ICT |
| 广义应用 | 学术图谱、推荐、hydro-science、肿瘤分期、表格推理、知识图谱、social-abuse detection |

## 2. 时间轴与谱系：至少六条线，不是一根品牌箭头

| 时间 | 一手节点 | 披露的结构性变化 | 不能推出什么 |
|---|---|---|---|
| 2020-2021 | [GE-SpMM](https://arxiv.org/abs/2007.03179)、[CogDL](https://arxiv.org/abs/2103.00959)、[GLM](https://arxiv.org/abs/2103.10360)、[CogView](https://arxiv.org/abs/2105.13290) | 图系统、统一生成式预训练与视觉 token generation 并行起步 [D] | 不能把所有 THUDM 工作倒溯成 Zhipu 生产线 [U] |
| 2022 | [CogView2](https://arxiv.org/abs/2204.14217)、[CogVideo](https://arxiv.org/abs/2205.15868)、[GLM-130B](https://arxiv.org/abs/2210.02414) | hierarchical image generation、视频 Transformer、130B bilingual dense scaling [D] | 视觉模型与语言模型是否共享训练数据/权重 [U] |
| 2023 | ChatGLM-6B/2/3、CodeGeeX、ImageReward、WebGLM、CogVLM、CogAgent | 低成本对话、MQA/长上下文、原生 function call、偏好模型、web/GUI agents [D/C] | 原生工具接口不等于已经做 environment RL [U] |
| 2024 | GLM-4 All Tools、CogVLM2/GLM-4V、CogVideoX、AutoGLM、GLM-4-Voice | 128K + 工具闭环，视觉 expert/共训，diffusion Transformer，GUI curriculum RL，端到端语音 [D] | 每个分支都进入同一个 GLM-4 checkpoint [U] |
| 2025 | GLM-4.5/4.5V、ReST-RL、MobileRL、VisionReward、GLM-4.6V/4.7、GLM-TTS | 355B/32B MoE、23T、agentic RL factory、多域视觉 RL、代码 self-training、speech RL [D] | vendor 表中的跨模型平均分构成统一排行榜 [U] |
| 2026 | GLM-5/5.1/5.2、GLM-OCR、GLM-Image、GLM-5V、IndexCache、Vision2Web、ZCube | 744B/40B、28.5T、DSA/MLA/MTP、1M、compaction-aware RL、0.9B OCR、跨层 index reuse [D/C] | 最新产品更新具备与正式技术报告相同的配方透明度 [U] |

更准确的谱系是：

```text
language foundation: GLM -> GLM-130B -> ChatGLM 1/2/3 -> GLM-4
                     -> GLM-4.5 MoE -> GLM-4.6/4.7 -> GLM-5 -> 5.1/5.2

agent/tool: WebGLM -> ChatGLM3 native tools -> GLM-4 All Tools
            -> AutoWebGLM / AutoGLM -> GLM-4.5 agentic RL
            -> MobileRL / GLM-5 agentic engineering -> GLM-5.2 long-horizon

visual generation: CogView -> CogView2 -> Relay Diffusion -> CogView3/4 -> GLM-Image
                  CogVideo -> CogVideoX -> RealVideo/Kaleido -> SCAIL/SCAIL-2

visual understanding/action: VisualGLM -> CogVLM -> CogCoM/CogAgent
                             -> CogVLM2/GLM-4V -> GLM-4.5V/4.6V -> GLM-5V

speech/document: GLM-4-Voice -> GLM-ASR-Nano / GLM-TTS
                 GLM-V-like encoder + compact decoder -> GLM-OCR

graph/knowledge: GE-SpMM -> CogDL -> OAG-BERT/AMinerGNN -> OAG-Bench/LPS-GNN
```

箭头只表示来源支持的时间/方法前驱，不表示未披露的 weight initialization、代码复制或
组织因果。比如 CogView3 明确建立在 Relay Diffusion 上；但没有证据把 CogView3 的生成器
训练并入 GLM-4 的语言预训练。

## 3. 基础模型：架构与 scaling 的五次换轨

### 3.1 初代 GLM：空白填充统一理解与生成 [D]

[GLM](https://arxiv.org/abs/2103.10360) 用连续 span corruption 与随机 span 顺序，将
bidirectional context 和 autoregressive generation 放进同一 Transformer。给定被遮盖后的
Part A 与缺失 spans $B_1,\ldots,B_m$，目标为：

$$
p(B\mid A)
=\prod_{i=1}^{m}\prod_{t=1}^{\lvert B_{\pi_i}\rvert}
p\!\left(b_{\pi_i,t}\mid A,B_{\pi_{<i}},b_{\pi_i,<t}\right).
$$

Part A 内部双向可见；Part B 看见 A、已完成 spans 与当前 span prefix；二维位置分别编码
原 mask 位置和 span 内偏移。不同 mask 长度把 NLU、conditional generation 与 left-to-right
language modeling 统一为生成任务。论文用 158GB 语料，在同规模/算力对照中报告
SuperGLUE 平均提升 4.6-5.0 个百分点；这是作者实验，不是跨时代模型排名。证据：
Section 2，pp. 2-4；Section 3.1 与 Table 1，pp. 5-6；语料见 Appendix A.1，p. 11。

### 3.2 GLM-130B：dense bilingual scaling 与稳定性工程 [D]

[GLM-130B](https://arxiv.org/abs/2210.02414) 是 70 层、hidden 12,288、96 heads、
FFN 32,768、2,048 context 的 130B dense 模型。它在 96 台 8×40GB A100 DGX 节点上，
从 2022-05-06 训练至 2022-07-03，共处理 400B tokens，约中英各 200B；数据由大规模英文
Pile/WuDao pools 过滤而来，95% 为 `[MASK]`/`[gMASK]` 自监督目标，5% 为 74 个 prompted
datasets 的 multi-task instruction pretraining。证据：abstract 与 Introduction，pp. 1-2；
Sections 2-3，pp. 3-6；完整配置见 Appendix Table 11，p. 48。

训练系统采用 data/tensor/pipeline parallel 的 24×4×8 组合、FP16、DeepNorm post-LN、
GeGLU、RoPE 与 embedding-gradient shrink。报告给出 hardware FLOP utilization 43.3%、
model FLOP utilization 32.5%，并记录 30 多次 100B-scale preliminary failures。这些数字的
分母和硬件在报告中明确，不能改写成“43.3% 训练效率”而省略 HFU 定义。证据：Section 3，
p. 5；Figure 3，p. 3；Appendix Table 11，p. 48。

### 3.3 ChatGLM 到 GLM-4：对话、MQA、长上下文与工具协议 [D/C/U]

[ChatGLM family report](https://arxiv.org/abs/2406.12793) 与三个公开模型卡显示：

- **ChatGLM-6B [C/D]**：6.2B、28 layers、hidden 4,096、32 heads、约 1T bilingual tokens、
  约 2K context；SFT、feedback bootstrap 与 RLHF 只有高层描述。
- **ChatGLM2-6B [C/D]**：1.4T bilingual tokens；32 query heads、2 MQA KV groups；base
  context 32K，dialogue alignment 8K。MQA 降低 KV cache，不等于稀疏 attention。
- **ChatGLM3-6B [C]**：8K/32K checkpoints，公开 native function calls、code interpreter
  与结构化 tool protocol。这证明接口能力，不证明端到端 environment RL。
- **GLM-4 [D]**：约 10T tokens，主要中英并含较小的 24-language collection；约 150K
  tokenizer、RMSNorm、SwiGLU、2D RoPE、Grouped-Query Attention（GQA）、128K；
  All Tools 进一步对 browser、Python、retrieval、CogView3 与 custom functions 对齐。
- **GLM-4-9B [C/D]**：40 layers、hidden 4,096、32 query/2 KV heads、native 8K；chat 128K，
  另有实验性 1M checkpoint。

以上结构与数据见 report Sections 2-3，pp. 4-12；function call、AgentBench 与 All Tools
数字见 Tables 7-9，pp. 11-12。表中 78.08 对 67.12 的 browser information score 是该报告
自己的 tool、prompt 与 judge 设置，不能脱离 Table 9 协议横比。**[U]** GLM-4 参数量、
层数、硬件、精确 data mixture、SFT/preference 数、reward architecture 与 PPO 超参数。

### 3.4 GLM-4.5：首次公开 MoE 旗舰 [D/C]

[GLM-4.5 report](https://arxiv.org/abs/2508.06471) 的 Table 1（p. 3）区分 total 与
activated parameters：

| 模型 | 总参数 / 每 token 激活 | Transformer 主体 | routed / active experts | attention |
|---|---:|---|---:|---|
| GLM-4.5 | 355B / 32B | 3 dense + 89 MoE + 1 MTP | 160 / 8，另 1 shared | GQA，96 Q / 8 KV |
| GLM-4.5-Air | 106B / 12B | 1 dense + 45 MoE + 1 MTP | 128 / 8，另 1 shared | GQA，96 Q / 8 KV |

两者使用 QK normalization、partial RoPE、sigmoid gate 与 loss-free balancing；公开配置
确认 131,072 context [C]。参数表排除 embeddings/output layer，hosting 侧总数略有差异时，
应保留计数 convention，而不是断言某一方“错了”。

### 3.5 GLM-5 与 5.2：MLA、DSA、MTP、1M 和计数冲突 [D/C/U]

[GLM-5 report](https://arxiv.org/abs/2602.15763) 披露 744B total/40B active、256 routed
experts、约 200K context；架构将 GLM-4.5 的 GQA 换成 MLA，并在 dense attention 完成
mid-training 后转换为 DSA，配合 shared-parameter MTP。公开配置可确认 hidden 6,144、
3 个 initial dense layers、8 routed + 1 shared expert、64 MLA heads、top-k 2,048 与一个
MTP layer [C]。报告正文写 80 layers，公开配置写 78；报告计 744B，某些 hosting 工具计
约 754B。**[U]** 差异究竟来自层、embedding/output 或 tensor-count convention。

核心数字见 report Sections 2.1-2.4，pp. 4-9；架构对照见 Appendix Table 10，p. 36。
DSA 先冻结 base、训练 indexer 1,000 steps，再 joint sparse adaptation 20B tokens；RL 时
冻结 indexer，并要求 deterministic `torch.topk`，因为非确定 top-k 在作者实验中导致
entropy collapse。这里“约 1.5-2× attention-compute saving”是 report 对长序列实验的
系统/算力口径，不是端到端吞吐保证。

[GLM-5.2 official research page](https://z.ai/blog/glm-5.2) 与公开 config 显示 context
1,048,576、RoPE theta 8M、仍为 78 hidden layers [D/C]。同一家族被分别写成 744B/40B、
约 750B-A40B 和 hosting 计数 753B；本章保留三种口径。5.2 的 IndexShare 每四层共享一组
DSA indices，并让 MTP steps 共享 indices/KV cache；官方 ablation 给出 7-step acceptance
length 从 4.56 到 5.47。该表只能说明该版本自己的 speculative decoding ablation，不能
直接换算为任意硬件上的 20% wall-clock speedup。

## 4. 预训练数据与优化：公开的是操作骨架，不是可重建语料

### 4.1 processed-token 时间线 [D]

| 代际 | 公开 token exposure | 可核验的数据操作 | 主要未知 |
|---|---:|---|---|
| GLM-130B | 400B | 约中英各半；Pile/WuDao pools；95% blank infilling + 5% 74-dataset MIP | 文档级清单、许可与 exact mixture |
| ChatGLM-6B / 2 | 约 1T / 1.4T | bilingual pretraining；模型卡给结构与 token 总量 | 去重、质量阈值、重复 exposure |
| GLM-4 / 4-9B | 约 10T | web、Wikipedia、books、code、papers；exact/fuzzy dedup；质量过滤；24-language tail | 参数、各域比例、硬件和 duration |
| GLM-4.5 | headline 23T | 15T general、7T code/reasoning、500B repo code、500B synthetic reasoning、100B long/agent；MinHash/semantic dedup、FIM、code classifiers | 数字约计后为 23.1T；unique tokens、许可和 source weights |
| GLM-5 | headline 28.5T | 27T main + 1T 32K + 500B 128K + 50B 200K；约 10M issue-PR pairs、约 160B unique filtered repo tokens | private/public 比例、完整 curriculum、污染审计 |

GLM-4.5 数字与操作见 report Sections 2.2-2.4，pp. 3-5；GLM-5 见 Sections 2.2-2.3，
pp. 8-9。这里的 `23T`/`28.5T` 是训练过程处理量，不是可以相加后声称的 unique corpus；
重复、阶段继续训练与 synthetic data 都会改变 exposure。

GLM-4.5 的 mid-training 加入完整 repository files、issues、pull requests、commits、
teacher-generated reasoning、long documents 与 synthetic agent trajectories，并将 context
4K→32K→128K。GLM-5 继续强调 Software Heritage metadata repair、language-specific code
filter、math/science LLM scoring 和 repo/issue/PR 结构；报告特意说 math/science slice 排除
synthetic/AI/template-generated 内容，但这不能外推到全语料。

### 4.2 从 DeepNorm 到 Muon Split 的稳定性路线 [D/U]

- **GLM-130B：** FP16 + DeepNorm post-LN + embedding-gradient shrink；报告把 loss spikes
  和 divergence 作为核心工程问题，并公开多次失败试验。
- **GLM-4.5：** 除 embedding、bias、RMSNorm 外使用 Muon；5 次 Newton-Schulz、momentum
  0.95、update RMS 0.2；学习率 $2.5\times10^{-4}\to2.5\times10^{-5}$；token batch 在前
  500B exposure 内由 16M 增至 64M；MTP loss weight 前 15T 为 0.3、之后为 0.1。证据：
  Section 2.4，p. 5。
- **GLM-5：** Muon Split 分头正交化 projection，减少不同 attention heads 的 scale
  interference；训练三个 shared-parameter MTP prediction layers，部署时保留一个公开 MTP
  head。证据：Sections 2.1-2.4，pp. 4-9。

**[U]** GLM-4.5/5 总 GPU 数、训练时长、总 FLOPs、能耗、失败 runs 与完整 optimizer state。
公开 optimizer 公式和几个超参数不足以复现训练，也不能据 token count 反推美元成本。

## 5. 后训练、RL 与蒸馏：从“有 RLHF”到执行环境工厂

### 5.1 ChatGLM/GLM-4：披露停在阶段名 [D/U]

ChatGLM-130B/6B 披露 manually constructed prompt-response data、SFT、feedback bootstrap
和 RLHF；GLM-4 披露 human/third-party prompts、SFT、RLHF 与安全对齐，偏好维度包含
safety、factuality、relevance、helpfulness 和 overall preference。**[U]** 样本数、标注
流程、reward model、PPO/其他 actor objective、KL、batch、steps、硬件和过滤淘汰率。

GLM-4-32B-0414 与 GLM-Z1 的公开仓库进一步写到 human preference、rejection sampling、
math/code/logic RL、general-domain pairwise-ranking feedback；但版本名和高层 recipe 不能
填补算法细节。GLM-Z1-Air 的参数量也未由一手资料完整披露，不能依据第三方命名猜测。

### 5.2 GLM-4.5：experts、轨迹工厂与迭代自蒸馏 [D/I/U]

GLM-4.5 先训练 reasoning、agent、general experts，再以 self-distillation 合为同时支持
thinking/direct modes 的模型。report Section 3.1（pp. 5-7）写明“millions” SFT samples、
最长 128K；过滤 repetition、truncation、reasoning correctness、reward-model score、tool
protocol 与 terminal state。作者报告去除约最容易的 50% prompts 后提升 2-4%，对难题
每题生成四个 candidates 再过滤提升 1-2%；这些是同一内部 pipeline 的 ablation，不是
通用定律。

agent trajectory factory 的已披露流程是：收集真实 frameworks/APIs/MCP servers/tools →
合成 tool sets 与单/多步任务 → 生成轨迹与模拟用户 → 多 judge 验证 → 保留成功轨迹 →
校验 XML-like calls、参数与终态。XML-like protocol 的理由是减少嵌入代码时的 escaping
失败，不说明 XML 本身提升 planning。

web RL 使用 multihop knowledge-graph questions 与 final-answer reward；software-engineering
RL 使用真实 issues/PRs、沙箱测试与格式终止。报告写出的 centered outcome advantage 为：

$$
A_i=r(x,y_i)-\frac{1}{K}\sum_{j=1}^{K}r(x,y_j).
$$

然而 report Section 3.3.2（pp. 10-11）印刷的 objective 只剩 centered rewards 求和，
其值恒为零，缺少必要的 score-function/log-probability 项。**[I]** 合理的梯度应含
$A_i\nabla_\theta\log\pi_\theta(y_i\mid x)$；**[U]** 实际实现的 clipping、importance
correction 与 normalization。不能原样复制论文排版错误并称为可复现算法。

### 5.3 GLM-5：顺序 RL、异步 rollout 与 on-policy consolidation [D]

GLM-5 report Sections 3-4（pp. 10-21）给出更完整的顺序：

```text
reasoning RL -> agentic RL -> general RL
             -> on-policy cross-stage distillation
```

Reasoning RL 使用 Group Relative Policy Optimization（GRPO）与 train/inference mismatch
抑制；报告给出 group 32、batch 32、lower/upper PPO clips 0.2/0.28、mismatch bound 2，
并说明无 KL regularization。agentic 环境包括：

- 超过 10,000 个可验证 SWE/terminal 场景，覆盖 Python、Java、Go、C/C++、JS/TS、PHP、Ruby；
- search 侧超过 2M pages 的 world-knowledge graph，并用“无需 tool 八次中一次可解”等规则
  排除过易问题；
- slides/artifacts 采用 markup、rendered image、content 多层奖励；
- context manager 在作者 BrowseComp 设置中从 55.3 到保留最近五次交互的 62.0，再到
  32K hierarchical discard 的 75.9。证据：Sections 4.2.3-4.2.5，pp. 18-21；这不是与
  其他模型无协议差异的排行榜。

异步系统分开 inference/training GPU pools，中央 orchestrator 支持超过 1,000 concurrent
rollouts；Token-In, Token-Out（TITO）保留 rollout token IDs/log-probs，记录 weight version，
过滤过旧或环境崩溃轨迹，并让同一 rollout 的 turns 固定到同一 data-parallel rank 以复用
KV cache。On-Policy Distillation（OPD）让 teacher 在 student 自己采样的 states/tokens 上
提供 log-probability difference，因此不是纯 offline imitation。**[U]** policy-lag 阈值、
同步周期、各域 rollout 数、reward weights 与总硬件。

### 5.4 GLM-5.2：compaction-aware critic PPO 与 SAO 边界 [D/I/U]

[GLM-5.2 official page](https://z.ai/blog/glm-5.2) 解释了问题：超长 agent trace 经 context
compaction 后产生数量和长度差异很大的 sub-traces，prompt-group relative optimization
难以利用所有片段。官方披露改为 critic-based PPO、individual-rollout learning、token-level
advantage 与 token-level loss，并保留全部 compacted sub-traces [D]。**[U]** critic
architecture、Generalized Advantage Estimation 的 $\gamma/\lambda$、value/entropy coefficients、
clipping、batch、steps 与 optimizer。

冻结 95 条 inventory 之外，旧 case study 还核对了临近截止日的一手论文
[Single-Rollout Asynchronous Optimization](https://arxiv.org/abs/2607.07508)。其 abstract
直接说 SAO 被部署在 open GLM-5.2 的 agentic-RL pipeline [D]；single rollout 用 learned
critic 替代 same-prompt group baseline，并以 rollout log-probability 对当前 policy 做严格
双侧 importance masking。**[I]** 这与官方“individual rollout + critic”描述吻合；但论文的
controlled experiments 使用 Qwen3-30B-A3B，不是 GLM-5.2。故不能把实验里的学习率、
critic 更新比或收益转抄成 753B checkpoint 的生产配方 [U]。

5.2 还披露 parallel OPD 同时合并十余个 experts，完整合并约两天；Slime 支持白盒/黑盒
rollout、compact traces 与 sub-agent workflows。两天缺少硬件分母，不能换算成总 compute。

### 5.5 奖励攻击不是旁枝 [D]

GLM-5/5.2 明确列出 coding agent 可能读 evaluation artifacts、复制答案、恢复 upstream
commits、下载 target source 或经 chained tools 泄漏。系统先用高召回规则筛可疑 actions，
再用 LLM intent judge；命中后阻断调用、返回 dummy information，但让 rollout 继续。继续
并不等于放行：外部 hard block 阻止动作，保留后续纠错数据，并避免每次怀疑都 terminal
zero 导致策略坍缩。这是已披露设计 [D]；规则、false-positive rate、生产 incident 数据 [U]。

## 6. 长上下文与推理系统：context window 不是一个数字

### 6.1 四种长度必须区分

1. **训练长度：** GLM-4.5 的 4K→32K→128K；GLM-5 的 32K/1T、128K/500B、200K/50B。
2. **公开 config 上限：** GLM-4.5 131,072；GLM-5 202,752；GLM-5.2 1,048,576 [C]。
3. **API/产品上限：** 可能另有 truncation、routing、output budget，不能从 config 自动推出。
4. **agent 有效记忆：** compaction、tool observations、KV eviction 与 state summary 会改变
   实际 decision history；needle retrieval 也不等于长轨迹 credit assignment。

### 6.2 MLA、DSA、IndexShare 与 IndexCache 不是同义词 [D]

- **MLA** 将 key/value state 压到 latent representation，减少 KV cache。
- **DSA** 由 learned indexer 为每个 query 选择 top-k historical tokens，再执行 sparse
  attention；GLM-5 top-k 2,048。
- **IndexShare** 是 GLM-5.2 在模型训练/架构中让每四层共享 selected indices。
- **IndexCache** 是独立方法，利用相邻层 token-selection 相似性，可 training-free 搜索
  share pattern，也可 multi-layer distillation 训练 sharing。

[IndexCache](https://arxiv.org/abs/2603.12201) 在 30B DSA 模型上报告最多移除 75% indexer
computations；end-to-end speed/memory 见 Section 4.2 与 Table 1，p. 7。对 744B GLM-5 的
结果只是 preliminary scaling experiment，见 Section 4.5 与 Table 4，p. 10。不能把
“少 75% indexer computation”写成“整个模型快 4×”。

### 6.3 服务层研究 [D/U]

[ZCube report](https://www.zhipuai.cn/zh/research/160) 聚焦下一代 LLM inference network；
[Scaling Pain](https://z.ai/blog/scaling-pain) 记录 GLM-5 coding-agent serving 的调试经验。
它们是官方 web-native engineering reports [D]，但公开程度不同于技术报告：完整 topology、
traffic distribution、SLO traces、failure logs 与复现实验 [U]。Slime、Transformers、vLLM
和 SGLang 的公开实现可确认部分 tensor shape/rollout interface [C]，不能自动代表线上版本。

`affiliated` 系统论文扩展了背景：MEPipe 研究低成本 accelerator 上的 slice-level pipeline；
GE-SpMM 与 LPS-GNN 研究 GPU sparse matrix multiplication 和 100-billion-edge graphs。
它们是工作级机构关联或当前官方 repo 线索，不应画入 GLM-5 serving critical path [U]。

## 7. 图像与视频生成：从 autoregressive pixels 到 diffusion 与 in-context control

### 7.1 CogView 1/2：视觉 token 的早期 scaling [D]

[CogView](https://arxiv.org/abs/2105.13290) 是 4B Transformer + VQ-VAE tokenizer，使用
约 30M 中英文 text-image pairs；Appendix A（p. 15）说明约 2.5TB、约 50% text 为英文，
正文报告 96B trained tokens。模型用 PB-relax 与 Sandwich-LN 处理 FP16 overflow，并以
caption loss 自重排候选。证据：Sections 2-4，pp. 2-10。早期 paper 还在 Appendix D
评估 gender bias；这说明作者检查过一个 bias slice，不等于完整生成安全审计。

[CogView2](https://arxiv.org/abs/2204.14217) 把 6B CogLM、bidirectional masked patch
prediction 与三阶段 hierarchy 组合：20×20 low-resolution tokens → direct 60×60 super-
resolution → local-parallel iterative refinement。结构见 Sections 3.1-3.3，pp. 4-7；
MS-COCO machine evaluation 见 Table 1，p. 8。它同时支持 inpainting/outpainting/editing，
不能只按一张 FID 表概括。

### 7.2 Relay Diffusion、CogView3 与 Inf-DiT [D]

[Relay Diffusion](https://arxiv.org/abs/2309.03350) 用 forward diffusion 把低分辨率 stage
平滑接到高分辨率 stage，避免传统 cascade 的独立 stage mismatch；方法见 Section 3，
pp. 5-7，结果见 Tables 1-3，pp. 8-9。

[CogView3](https://arxiv.org/abs/2403.05121) 把 relay diffusion 用于 text-to-image：3B
backbone、base + super-resolution 两 stage，并做 progressive distillation。作者表中原版
50+10 steps 用 10.33 秒，4+1-step distilled 版用 1.47 秒；对应质量指标同时列在 Table 2，
p. 10。只引用 7× latency ratio 而省略硬件、steps 和 quality columns 会误导。

[Inf-DiT](https://arxiv.org/abs/2405.04312) 用 block-wise autoregression 与可丢弃 KV
state，把最低 hidden-state memory 从图像边长 $N$ 的 $O(N^2)$ 降至 $O(N)$；报告在
4096×4096 上称节省超过 5× memory。证据：Sections 3-4，pp. 3-8，Figure 2，p. 2。
这是 upsampler 的特定 memory curve，不是所有 diffusion Transformers 的普遍复杂度。

### 7.3 CogVideo、CogVideoX 与后续控制 [D/C]

[CogVideo](https://arxiv.org/abs/2205.15868) 是 9.4B、5.4M captioned-video pairs 的两阶段
Transformer：先按 frame-rate token 生成 key frames，再递归插帧；公开样例为 4 秒/32 帧、
最高 480×480。证据：Introduction 与 Section 3，pp. 1-6；训练数据和参数见 Appendix，
p. 12；Tables 1-2，pp. 7-8 给出 UCF/Kinetics protocol。

[CogVideoX](https://arxiv.org/abs/2408.06072) 转向 3D causal VAE + Expert Transformer +
progressive diffusion training，公开 2B/5B 两档；训练数据在过滤后保留视频 clips，并加入
2B aesthetics-filtered images。架构见 Section 2，pp. 3-6；数据见 Section 3，pp. 6-9；
结果与完整 hyperparameters 见 Table 3（p. 10）和 Tables 5-9（pp. 16-17）。

后续研究把控制目标分解：Kaleido 做 multi-subject reference video 与 subject/background
disentanglement；SCAIL 用 3D-consistent pose 和 pose-shifted RoPE；SCAIL-2 绕开 skeleton/
mask 中间表示，以 end-to-end driving unifying animation/replacement。SCAIL 的同 backbone
14B comparison 见 Table 1，p. 7；SCAIL-2 的 59,376 end-to-end motion-transfer pairs 与
DPO+SFT anchor 见 Appendix，pp. 11-12。它们不是 GLM language flagship 的 benchmark。

### 7.4 偏好、cache 与 VAE 表征 [D/I]

[ImageReward](https://arxiv.org/abs/2304.05977) 收集 137K expert pairwise comparisons，
训练通用 text-to-image preference model；数据/标注见 Sections 3-4，结果见 Tables 1-3。
[VisionReward](https://arxiv.org/abs/2412.21059) 把偏好拆为多维 image/video criteria，
用于细粒度评分与生成优化。二者说明视觉后训练需要超越单一 CLIP/FID [D]；把 reward
model 得分当作“人类偏好真值”仍是 [I]。

OmniCache 研究 diffusion Transformer 的 trajectory-oriented cache reuse；VPO 用 prompt
optimization 对齐 text-to-video；Concat-ID 研究 identity preservation；SSVAE 分析 video
VAE latent 的 spectral bias。它们在 inventory 中属 `affiliated` 或 `direct` 方法研究，
不能据标题推断已进入 CogVideoX/GLM-Image [U]。

## 8. 视觉理解、语音与 OCR：perception 正在变成 action interface

### 8.1 CogVLM/CogVLM2：visual expert 与 text-image co-training 两条路 [D]

[CogVLM](https://arxiv.org/abs/2311.03079) 在冻结的 pretrained language layers 旁为每层
加入 visual expert（visual QKV 与 FFN），让视觉 token 深入 language transformer，同时尽量
保留纯文本能力。CogVLM-17B 由约 6.6B language parameters、约 4.4B visual expert 和
1.5B vision encoder 等组成；caption/VQA/grounding 与 ablation 见 Tables 1-4，pp. 6-8，
training hyperparameters 见 Tables 5-6，p. 13。参数总名 `17B` 不等于每个 token 都激活
全部模块。

[CogVLM2](https://arxiv.org/abs/2408.16500) 明确区分：CogVLM2-LLaMA3-19B 继续用 visual
expert，以保语言能力；GLM-4V-9B 采用 text-image co-training；CogVLM2-Video-12B 加入
temporal position IDs 与 video data。结构对照见 Table 1，p. 4；CLAY-1B recaption data 和
VQA/video mixtures 见 Section 3，pp. 5-8；结果见 Tables 4-5，p. 10。三者共享“family”
不等于同一 checkpoint。

### 8.2 Chain-of-Manipulations 与 GUI grounding [D]

[CogCoM](https://arxiv.org/abs/2402.04236) 将 crop、zoom、grounding 等 visual manipulations
写进 reasoning chain。17B model 加入 70K CoM training examples 后，作者 ablation 在一个
VQA aggregate 从 64.5 到 71.1；见 Section 4 与 Table 4，p. 10。该提升依赖论文自己的
tasks/metric，不是“视觉推理普遍提升 6.6%”。

[CogAgent](https://arxiv.org/abs/2312.08914) 是 18B dual-resolution VLM，输入最高
1120×1120，针对 text-rich GUI、grounding 和 navigation；pretraining 60,000 steps，配置/
训练见 Sections 3-4，pp. 4-5 与 Tables 7-8，p. 11。静态 Mind2Web/Android 数据上的 action
prediction 不能替代 live environment success。

[GLM-4.5V/4.1V-Thinking report](https://arxiv.org/abs/2507.01006) 将视觉 backbone、
GLM language MoE 与多域 RL 结合。pretraining data 包含 220M OCR images、40M natural
grounding annotations 和超过 140M GUI referring-expression QA pairs；sequence 8,192、
global batch 1,536、120,000 steps，之后 32,768 context 继续 10,000 steps。证据：
Sections 3.1-3.2，pp. 5-7。数字是 dataset instances/steps，不是 unique real photographs。

其 Reinforcement Learning with Curriculum Sampling（RLCS）按 domain 维护 dynamic sampling
ratio；GUI reward 组合 action correctness 与 intersection-over-union（IoU），grounding 用 IoU，
QA 用 exact/semantic matching。完整 reward 表见 Table 1，p. 12；curriculum/stability 见
Sections 5.2-5.4，pp. 10-15。报告警告单域弱 verifier 会拖累其他 domains；这比只引用
Table 2 vendor scores 更重要。

GLM-4.6V 官方卡增加 native tool use 和多模态 URL values；GLM-5V-Turbo 强调视觉 coding。
这些是 web/model-card 级一手披露 [D]；参数、训练 token、RL task count、reward weights 与
相对 4.5V 的严格 ablation [U]。

### 8.3 GLM-4-Voice：低帧率 speech tokens 与 streaming thought [D]

[GLM-4-Voice](https://arxiv.org/abs/2412.02612) 使用 supervised single-codebook speech
tokenizer，12.5Hz frame rate、175-token codebook，并用 flow-matching decoder 生成 waveform；
streaming-thought template 交替输出 text/speech tokens，使输入、语言思考与语音输出可流式
交错。tokenizer/decoder 见 Sections 3.1-3.2 与 Table 1，pp. 3-5；data/training 见 Section 4
与 Table 2，pp. 6-7；ASR/TTS/chat 结果见 Tables 3-6，pp. 7-9。speech token frame rate
不是音频采样率，也不能单独决定端到端 latency。

### 8.4 GLM-TTS：多奖励 RL 不只属于文本 [D]

[GLM-TTS](https://arxiv.org/abs/2512.14291) 是 1.5B autoregressive speech LM + diffusion
decoder，使用约 100K hours training data。它在 GRPO 中组合 pronunciation、speaker
similarity、naturalness/emotion 等 rewards，并设计 dynamic sampling、asymmetric clipping、
phoneme-in 与 LoRA voice customization。method 见 Sections 2.4-2.6，pp. 4-7；Seed-TTS 与
RL ablations 见 Tables 3-7，pp. 9-10。作者表中 GLM-TTS 到 GLM-TTS-RL 的 test-zh CER
1.03→0.89，同时 similarity 76.1→76.4；必须连 metric/dataset 一起引用，不能写成
“RL 提升 14% TTS 质量”。

GLM-ASR-Nano model card 披露 real-world robustness 取向 [D]，但完整 speech corpus、
accent/noise mixture、training compute、置信度校准与生产错误分布 [U]。

### 8.5 GLM-OCR：0.9B specialist、MTP 与两阶段文档系统 [D]

[GLM-OCR report](https://arxiv.org/abs/2603.10910) 的 0.9B model 组合 vision encoder、
cross-modal connector 与 compact GLM decoder；训练时预测 10 tokens/step，作者报告平均
5.2 generated tokens/decoding step。SDK 先做 layout analysis，再并行识别 regions；因此
benchmark 既测 core model，也可能包含 pipeline。架构见 Sections 2.1-2.2，pp. 4-6。

公开 benchmark 中 OmniDocBench v1.5 94.62、OCRBench Text 94.0，见 Tables 3-4，
pp. 7-8；作者环境吞吐为 PDF 1.86 pages/s、standalone image 0.67 images/s，见 Table 6，
p. 9。供应商/API、GPU、resolution、batch 与 pipeline 不同，不能据这些值宣布全球 OCR
吞吐排名。报告 Section 6（pp. 13-14）还明确承认 two-stage constraints、data coverage 与
structured-output stochasticity；这类 limitation 应与 headline 同时保留。

## 9. 代码与数学：从 multilingual completion 到 agentic software engineering

### 9.1 CodeGeeX 1/2/4 [D/C/U]

[CodeGeeX](https://arxiv.org/abs/2303.17568) 是 13B、39-layer autoregressive code model；
158B-token corpus 覆盖 23 programming languages，训练在 1,536 Ascend 910 processors 上
处理 850B tokens、213,000 steps、约两个月。数据/训练见 Sections 2.2-2.3，pp. 5-7 与
Table 2，p. 6。HumanEval-X 把 164 problems 扩为 Python/C++/Java/JavaScript/Go 的
820 problem-solution pairs，见 Section 3，pp. 8-10；Tables 5-7 的 pass@k/translation
数字必须保留 temperature、top-p 与 sample budget。

CodeGeeX2 基于 GLM code lineage，CodeGeeX4-ALL-9B model card 披露 completion、chat、
interpreter/function calling 等一体化能力 [C/D]。二者有公开仓库/weights，但 model-card
更新未给出与 CodeGeeX 论文同等完整的 corpus、optimizer、steps 与污染审计 [U]。

### 9.2 ReST-RL 与 UI2Code：方法论文不自动等于旗舰 recipe [D]

[ReST-RL](https://arxiv.org/abs/2508.19576) 组合 reinforced self-training、ReST-GRPO、
value-model-guided Monte Carlo Tree Search（VM-MCTS）与 verifier，在多种 7B/8B code
policies 上实验；method 见 Sections 3-4，pp. 3-7，policy/decoding results 见 Tables 1-2，
pp. 8-10。它证明该方法在所测 base policies/benchmarks 上有效 [D]，不证明 GLM-4.5/5
生产 RL 采用同一 VM-MCTS [U]。

[UI2Code^N](https://arxiv.org/abs/2511.08195) 把 UI-to-code 写成 rendered-feedback 的
interactive visual optimization，并以 Relative Visual Policy Optimization（RVPO）训练 9B
模型。Table 1（p. 6）区分 SFT/RL；Table 3（p. 7）显示 UI2Code-Real 随 1→5 iterations
由 66.0→74.0，而 synthetic set 在三轮后饱和。它是 test-time interaction scaling 的一个
controlled setting，不等于任意 coding agent 多迭代都会提高。

### 9.3 数学能力的证据边界 [D/U]

GLM-4.5/5 的 reasoning data/RL 明确包含 math、science、code；GLM-Z1 公开页也说明
math/code/logic cold start 与 RL [D]。但是当前 95 条没有一个等同于 DeepSeekMath 那样
公开完整 math corpus recovery、dedup、verifier contamination 与 formal-proof pipeline 的
Zhipu core report。AIME、MATH-500、GPQA 等 vendor scores 可描述 checkpoint 在特定
sampling 下的结果，不能代替训练数据披露；数学语料身份、可验证题占比、reward mix 与
proof checking 仍为 [U]。

## 10. Agents 与 tool use：接口、导航、环境 RL 和 serving 必须分层

### 10.1 WebGLM：retrieval + bootstrapped QA + preference scorer [D]

[WebGLM](https://arxiv.org/abs/2306.07906) 用 LLM-augmented retriever 取 top-5 references，
GLM-10B/2B generator 在 bootstrapped WebGLM-QA 上训练，再由 human-preference-aware scorer
选答案。系统见 Section 3 与 Figure 3，pp. 3-6；22,000 WebGLM-QA examples 的过滤
ablation 见 Table 5，p. 9；human evaluation 与 scorer ablations 见 Tables 2、8-9，
pp. 8-11。retrieval QA 不是浏览器 action policy，别把 WebGLM 与 AutoWebGLM 合并。

### 10.2 AutoWebGLM：从 imitation 到 preference/rejection tuning [D]

[AutoWebGLM](https://doi.org/10.1145/3637528.3671620) 基于 ChatGLM3-6B，将 HTML/GUI
observations 映射到浏览器 action space。训练从 atomic→complex curriculum SFT；DPO
每个 complex prompt 采 20 responses，形成约 13K preferences；rejection fine-tuning 使用
约 15K MiniWoB++ trajectories/66K steps 与 240 WebArena trajectories/2K steps。证据：
Section 4，pp. 5-7；Appendix A，p. 11。Table 4（p. 8）中的 MiniWoB++/WebArena success
必须保留静态 benchmark 与实现版本。

### 10.3 AutoGLM 与 MobileRL：online GUI environment [D]

[AutoGLM](https://arxiv.org/abs/2411.00820) 分离 planner 与 grounder，以 GLM-4-9B 做
progressive weak-to-strong/self-evolving online curriculum RL。作者在 AndroidLab 报告
36.2% success，见 Introduction 与 Section 3.2，pp. 1、7；Web/Android 结果见 Figures 5-6。
这不是商业 AutoGLM 所有后续版本的恒定分数。

[MobileRL](https://arxiv.org/abs/2509.18119) 又把 mobile RL 分为 reasoning-free SFT、
reasoning SFT 和 Difficulty-Adaptive GRPO：Shortest-Path Adjustment 重塑 binary terminal
reward，Adaptive Positive Replay 复用成功轨迹，Failure Curriculum Filtering 调整难度。
method 见 Section 2，pp. 3-6；作者训练用 2,000 AndroidWorld 与 1,103 AndroidLab tasks，
见 Appendix data section，p. 20；full 9B model 的 AndroidWorld/AndroidLab 结果见 Table 1
与 ablation，pp. 6-9。AutoGLM 与 MobileRL 是相邻研究，不应被写成同一 optimizer。

### 10.4 Function calling、视觉 coding 与 agent evaluation [D]

- [ComplexFuncBench](https://arxiv.org/abs/2501.10132)：43 real-time APIs、五个日常领域、
  1,000 samples，覆盖 multi-step、constraints、parameter reasoning 与 128K response context；
  统计见 Table 3，p. 10，主结果见 Table 2，p. 7。
- [Vision2Web](https://arxiv.org/abs/2603.26648)：193 tasks、16 categories、918 prototype
  images、1,255 test cases，分 static UI、interactive frontend、full-stack；Table 1，p. 2；
  workflow verifier 与结果见 Sections 3-4、Tables 2-5，pp. 4-7。
- GLM-4 All Tools、4.6V native tools、GLM-5 agentic engineering 和 GLM-5V visual coding
  分别处于 model alignment、multimodal tool protocol、environment RL 与产品 model-card
  层级；“会调用函数”不自动等于“通过工具环境强化学习”。

AtomR、SoAy、XDAI、social-abuse multimodal agent 与 MQE 也进入 inventory，但属于
`affiliated`/supported research。它们证明原子知识操作、academic API use、grounded dialogue、
meme analysis 或 quadruped multi-agent environment 的工作级贡献，不证明进入 GLM 主线。

## 11. Benchmark、evaluation、安全与 interpretability

### 11.1 自建 benchmark 应被当作测量工具，不是奖牌 [D]

| Benchmark | 公开规模与目标 | 一手定位 |
|---|---|---|
| HumanEval-X | 164 problems × 5 languages = 820 pairs；generation/translation | CodeGeeX Section 3，pp. 8-10 |
| LVBench | 103 videos、117 hours、1,549 QA；长视频理解 | Table 1，p. 3；dataset section，pp. 4-5、11 |
| MotionBench | 5,385 videos、8,052 QA；六类 fine-grained motion | Table 1，p. 3；Figure 3，pp. 4-5 |
| ComplexFuncBench | 1,000 cases、43 APIs；约束/多步/长参数 | Tables 2-3，pp. 7、10 |
| RPC-Bench | 4,150 papers、61.3K QA；review/rebuttal-derived taxonomy | Table 1，p. 2；Sections 3-4，pp. 3-8 |
| Vision2Web | 193 tasks、918 prototypes、1,255 tests；三级网站开发 | Table 1，p. 2；Tables 2-5，pp. 4-7 |
| OAG-Bench | human-curated academic graph mining | publisher primary record；`affiliated` |
| HoWToBench | Tree-of-Writing 的 human-level writing evaluation | ACL primary record；`affiliated` |

GLM-SIMPLE-EVALS 是公开 evaluation toolkit [C]，有助于固定 prompts/harness；但 vendor
report 仍常混合 public/private sets、API snapshots、不同 output budgets、tools、judges 与
sampling。正确比较对象是：

$$
f(\text{checkpoint},\text{prompt},\text{sampling},\text{tools},
\text{context},\text{budget},\text{judge},\text{date}),
$$

而不是一个脱离协议的分数。故本章不把 GLM-4.5/5 的 vendor benchmark 横比成总排行榜。

### 11.2 Safety：已有 slices，但没有连续 system-card 体系 [D/U]

GLM-130B 在 Appendix A（pp. 21-23）评估 CrowS-Pairs、StereoSet、ETHOS 与 prompt-conditioned
toxicity，且在 Ethics Statement（p. 10）讨论开放研究风险 [D]。这些是 2022 checkpoint 的
bias/toxicity slices，不证明后续模型安全。

GLM-4.5 report Section 4.2.5（pp. 17-18）用 SafetyBench 11,435 multiple-choice questions、
七类 safety concerns；GLM-5/5.2 又披露 agent anti-hacking [D]。库存中的 JPS 研究 multimodal
jailbreak；GuidedLatent 研究 VAE membership inference；ICT 干预 VLM object hallucination；
GLM-ASR-Nano model card 强调 noisy real-world robustness。它们覆盖攻击、隐私、幻觉与鲁棒，
但不能拼成生产 safety stack。

**[U]** 最新 GLM 的统一 system card、deployment threat model、red-team denominator、
cyber/CBRN/autonomy thresholds、事故/near-miss、false refusal、区域政策、monitoring 与
更新机制。没有公开证据时，既不能说“没有做安全”，也不能说“已系统解决”。

### 11.3 Interpretability：可观察机制多，因果解释少 [D/U]

早期 GLM 的 mask/position ablations、CogVideo attention visualization、GLM-130B loss-spike
failures、GLM-5 deterministic top-k collapse、SSVAE latent spectral analysis 都提供局部机制
证据 [D]。但当前 inventory 没有针对 GLM-4.5/5 内部 circuits、feature attribution、
deception/goal representation 或 causal intervention 的核心 interpretability program。
把 router load、attention map 或 reward trace 直接称为“理解模型思想”是 [I]，不能升级为 [D]。

### 11.4 广义应用研究：宽度不等于旗舰组成 [D/U]

`affiliated` 30 条展示了组织关联研究的宽度：

- graph/academic intelligence：AI 2000、OAG-BERT、AMinerGNN、text-attributed graph
  pretraining、OAG-Bench、graph Transformer、research-interest mining、LPS-GNN；
- domain models：hydro-science Hammer、non-small-cell lung-cancer staging、table CoRe；
- knowledge/retrieval：Genre、knowledge graph + LLM review、AtomR、SoAy、WebGLM journal；
- multimodal/application safety：CogCartoon、JPS、VPO、Concat-ID、ICT、GuidedLatent、
  social-abuse memes；
- systems/interaction：MEPipe、OmniCache、MQE、XDAI。

工作级 raw affiliation 是论文的机构声明 [D]，不是独立 employment audit，也不是 data-flow
evidence。**[U]** 这些项目与 GLM weights、私有 data lake、评测平台或生产 API 的共享程度。

## 12. 公开制品与未知项：开放不是一个布尔值

### 12.1 当前可确认的公开面 [C]

- 95 条中 51 条有 arXiv ID、35 条有 DOI、58 条有公开 PDF route、37 条没有单独 PDF；
- ChatGLM、GLM-4/4.5/5、GLM-OCR、GLM-TTS、CodeGeeX、CogView/CogVLM/CogVideo、
  AutoGLM 等提供不同程度的 weights/config/code/model cards；
- Slime 提供 Megatron + SGLang rollout、buffer、rewards、PPO/GRPO utilities、TITO traces、
  OPD、coding/search examples 与 async queue；
- GLM-SIMPLE-EVALS、HumanEval-X、LVBench、MotionBench、Vision2Web 等提供评测制品或
  明确的公开入口。

这能支持配置检查和局部实现研究，却不满足完整训练复现。当前 Slime 还会持续演进，不能
自动视为某个已发布 checkpoint 的冻结 internal trainer。

### 12.2 阶段审计

| 阶段 | 已公开 [D/C] | 关键未知 [U] |
|---|---|---|
| Architecture | 旗舰参数表、部分 config、MoE/MLA/DSA/MTP 设计 | 5/5.2 layer/parameter accounting 冲突、production kernels |
| Pretraining data | processed tokens、domains、过滤/去重操作、部分 repo/long data | document manifest、许可、mixture weights、重复 exposure、污染 |
| Optimization | DeepNorm、Muon/Muon Split、部分 schedule/batch/precision | 总 hardware、duration/FLOPs、failed runs、optimizer state |
| SFT | task families、millions-level 4.5 样本、loss masks | prompts、annotator process、mixture、淘汰样本 |
| Rewards/RM | rule/test/IoU/judge、人类偏好类别、anti-hack | reward models、weights、calibration、false positives |
| RL | GRPO/PPO/SAO 轮廓、groups/clips 的部分版本、环境类别 | GLM-5.2 coefficients、rollout volume、policy lag、各域配比 |
| Distillation | self/cross-stage/parallel OPD | teacher selection/conflicts、accepted/rejected corpus、完整 objective |
| Serving | TITO、async pools、IndexCache、部分框架与 web reports | live topology、router、traffic、SLO、incident logs、安全边界 |
| Evaluation | benchmark datasets、若干公开 harness | private sets、prompt/sampling/tools/judge/API snapshot |
| Safety | 早期 ethics slices、SafetyBench、anti-hacking、相关安全论文 | continuous system cards、threat model、incidents、production controls |
| Reproduction | weights/config/code/PDFs | 本章没有 [R]；无法端到端复现旗舰 |

### 12.3 一个实用的证据停止规则

当来源只说“improved”“SOTA”“RL-enhanced”时，应追问：model variant、training stage、data
denominator、reward、evaluation protocol、hardware 和 comparator 是否公开。答不出就停在
[U]，而不是用相邻论文或 model name 补空白。尤其不能把：

- API context window 当作训练长度；
- total parameters 当作 active parameters；
- processed tokens 当作 unique/licensed corpus；
- official model card 当作完整 technical report；
- affiliated paper 当作 flagship recipe；
- public weights 当作 full open-source training；
- vendor benchmark 当作统一排行榜。

## 13. 跨谱系综合：哪些关系可以推断，哪些仍不能

### 13.1 四条可辩护的综合判断 [I]

1. **稀疏性从“专家”扩到“序列与时间”。** MoE 稀疏计算 capacity；MLA 压 KV state；
   DSA 选历史 tokens；IndexShare/IndexCache 复用 selection；MTP 复用未来预测。联合目标
   是把 HBM、compute、network 与 latency 分成可独立优化的资源轴。
2. **agent training 正在变成系统问题。** 长尾 rollout、sandbox、tool drift、KV locality、
   context compaction、policy lag 和 reward leakage 会改变可训练样本分布；只写 policy
   objective 已不足以描述 GLM-5/5.2。
3. **多模态 reward 正在趋向结构化。** ImageReward/VisionReward 从总体偏好走向多维 QA；
   GLM-V 用 IoU/action/exact/semantic rewards；GLM-TTS 用 pronunciation/timbre/naturalness；
   GLM-OCR 用结构匹配。结构化并不消除 verifier bias。
4. **生成与 agent 的边界正在收敛。** UI2Code/Vision2Web 把 code 变成可渲染 action；
   GLM-4.6V/5V 把视觉 observation 接到 tools；slides/artifacts RL 又把生成物放回可执行
   feedback loop。评价单位因此应从单次回答转向完整 trajectory + artifact + environment。

### 13.2 仍然未知的因果链 [U]

- CogView/CogVideo 的数据或 weights 是否进入 GLM-4V/5V；
- ReST-RL、MobileRL、UI2Code 的 exact algorithms 是否用于 GLM-5/5.2；
- affiliated graph/medical/knowledge work与旗舰 data/architecture 是否共享；
- GLM-5.1/5.2 相对 5 的新增 pretraining exposure 与 domain mixture；
- 1M context 的 train/eval/serve 路径如何统一，compaction 何时触发；
- production safety filters、human escalation 与 incident response；
- 所有旗舰的总研发成本、失败实验、能源与供应链。

## 14. 逐条记录紧凑地图：95/95 inventory records

这是冻结清单的审计索引，不是引用次数排行榜。每一行都保留 inventory ID 与清单保存的
`primary_url`；短注只说明该记录在研究谱系中的最小定位。正式论文与 web model card 的
披露深度不同，`core/direct/affiliated` 是归属强度而非质量评分。

### 14.1 Core：旗舰、专用模型与正式技术报告（26）

| 日期 / inventory ID | 一手记录 | 最小定位与边界 |
|---|---|---|
| 2026-06-16 `official-research-161` | [GLM-5.2: Built for Long-Horizon Tasks](https://z.ai/blog/glm-5.2) | [D] 1M、IndexShare/MTP sharing、compaction-aware single-rollout PPO、parallel OPD；[U] 完整超参/硬件。 |
| 2026-04-07 `official-research-157` | [GLM-5.1: Towards Long-Horizon Tasks](https://z.ai/blog/glm-5.1) | [D] 长任务与 coding/system case 的版本更新；[U] 新增 token、RL 配方与 compute。 |
| 2026-04-01 `official-research-156` | [GLM-5V-Turbo: A Multimodal Coding Foundation Model](https://www.zhipuai.cn/zh/research/156) | [D] visual coding/product model card；无同等深度独立技术报告 [U]。 |
| 2026-03-15 `official-research-155` | [GLM-5-Turbo: A Foundation Model Enhanced for OpenClaw](https://www.zhipuai.cn/zh/research/155) | [D] OpenClaw-oriented agent model card；不能代替 GLM-5 report 的训练披露。 |
| 2026-03-11 `arxiv-2603.10910` | [GLM-OCR Technical Report](https://arxiv.org/abs/2603.10910) | [D] 0.9B、MTP、layout→parallel recognition、RL 与部署/limitations。 |
| 2026-02-17 `arxiv-2602.15763` | [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) | [D] 744B/40B、28.5T、MLA/DSA/MTP、顺序/异步 RL 与 OPD；主旗舰报告。 |
| 2026-02-02 `official-research-150` | [GLM-OCR: SOTA Performance, Mastering Complex Document Recognition](https://www.zhipuai.cn/zh/research/150) | [D] OCR 发布/模型页；正式算法与限制以 technical report 为准。 |
| 2026-01-19 `official-research-148` | [GLM-4.7-Flash, open source and free](https://www.zhipuai.cn/zh/research/148) | [D/C] compact open checkpoint/model card；[U] 完整训练配方。 |
| 2026-01-13 `official-research-158` | [GLM-Image: Auto-regressive for Dense-knowledge and High-fidelity Image Generation](https://z.ai/blog/glm-image) | [D] autoregressive image-generation model card；与 CogView/diffusion 线不能未经证据合并。 |
| 2025-12-21 `official-research-143` | [GLM-4.7: Your New Coding Partner](https://www.zhipuai.cn/zh/research/143) | [D] coding、interleaved/preserved thinking 的版本发布；vendor results 非统一榜单。 |
| 2025-12-16 `arxiv-2512.14291` | [GLM-TTS Technical Report](https://arxiv.org/abs/2512.14291) | [D] 1.5B、100K hours、multi-reward GRPO、phoneme-in 与 diffusion decoder。 |
| 2025-12-10 `official-research-147` | [GLM-TTS: Controllable & Emotion-Expressive Zero-shot TTS with Multi-Reward Reinforcement Learning](https://www.zhipuai.cn/zh/research/147) | [D] TTS 模型发布；详细数字由相邻 report 支撑。 |
| 2025-12-09 `official-research-149` | [GLM-ASR-Nano: Robust Speech Recognition for the Real World](https://www.zhipuai.cn/zh/research/149) | [D] small ASR/robustness model card；corpus、calibration 与 production error [U]。 |
| 2025-12-08 `official-research-145` | [AutoGLM Goes Open Source](https://autoglm.z.ai/blog/) | [D/C] AutoGLM artifacts/发布边界；不把产品更新反投射到 2024 paper。 |
| 2025-12-07 `official-research-144` | [GLM-4.6V: Open Source Multimodal Models with Native Tool Use](https://www.zhipuai.cn/zh/research/144) | [D] native multimodal tool protocol；[U] 相对 4.5V 的完整 data/RL ablation。 |
| 2025-08-08 `arxiv-2508.06471` | [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471) | [D] 355B/32B、23T、Muon、expert→distillation、agentic RL factory。 |
| 2025-07-01 `arxiv-2507.01006` | [GLM-4.5V and GLM-4.1V-Thinking](https://arxiv.org/abs/2507.01006) | [D] 视觉 data scale、RLCS、多域 rewards、GUI/grounding/video reasoning。 |
| 2024-12-03 `arxiv-2412.02612` | [GLM-4-Voice](https://arxiv.org/abs/2412.02612) | [D] 12.5Hz supervised speech tokens、streaming thought、flow-matching decoder。 |
| 2024-10-28 `arxiv-2411.00820` | [AutoGLM: Autonomous Foundation Agents for GUIs](https://arxiv.org/abs/2411.00820) | [D] planner/grounder 分工与 self-evolving online curriculum RL。 |
| 2024-06-18 `arxiv-2406.12793` | [ChatGLM: From GLM-130B to GLM-4 All Tools](https://arxiv.org/abs/2406.12793) | [D] ChatGLM generations、GLM-4 10T/128K、All Tools 与评测协议。 |
| 2023-10-27 `chatglm3-model-card` | [ChatGLM3 model card](https://github.com/zai-org/ChatGLM3) | [C/D] 8K/32K、native function call/code interpreter；接口不等于 agent RL。 |
| 2023-06-25 `chatglm2-6b-model-card` | [ChatGLM2-6B model card](https://github.com/zai-org/ChatGLM2-6B) | [C/D] 1.4T、MQA、32K base/8K chat；alignment 细节有限。 |
| 2023-03-14 `chatglm-6b-model-card` | [ChatGLM-6B model card](https://github.com/zai-org/ChatGLM-6B) | [C/D] 6.2B、约 1T bilingual、低内存量化与高层 RLHF。 |
| 2022-10-05 `arxiv-2210.02414` | [GLM-130B](https://arxiv.org/abs/2210.02414) | [D] 130B bilingual dense、400B、768 A100、稳定性/量化/ethics 详报。 |
| 2021-03-18 `arxiv-2103.10360` | [GLM: Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) | [D] Part A/B attention、2D positions、多任务生成式预训练的结构祖先。 |
| 未标日期 `glm-edge-model-cards` | [GLM-Edge model cards](https://github.com/zai-org/GLM-Edge) | [C/D] 端侧 model family artifacts；日期和完整 training recipe [U]。 |

### 14.2 Direct：方法、系统、评测与相邻模型研究（39）

| 日期 / inventory ID | 一手记录 | 最小定位与边界 |
|---|---|---|
| 2026-06-09 `arxiv-2606.10804` | [SCAIL-2](https://arxiv.org/abs/2606.10804) | [D] end-to-end in-context character animation/replacement，绕开易失真 intermediates。 |
| 2026-05-20 `official-research-160` | [How ZCube Alleviates LLM Inference Network Bottlenecks](https://www.zhipuai.cn/zh/research/160) | [D] inference-network engineering report；live topology/traces 不完整 [U]。 |
| 2026-04-29 `official-research-159` | [Scaling Pain of Coding Agent Serving](https://z.ai/blog/scaling-pain) | [D] GLM-5 coding-agent serving 调试经验；不是训练 technical report。 |
| 2026-03-27 `arxiv-2603.26648` | [Vision2Web](https://arxiv.org/abs/2603.26648) | [D] 193-task hierarchical visual web-development benchmark + workflow verifier。 |
| 2026-03-12 `arxiv-2603.12201` | [IndexCache](https://arxiv.org/abs/2603.12201) | [D] DSA cross-layer index reuse；indexer-compute saving 不等于端到端同倍率。 |
| 2026-01-14 `arxiv-2601.14289` | [RPC-Bench](https://arxiv.org/abs/2601.14289) | [D] 4,150 papers/61.3K QA 的细粒度论文理解 taxonomy。 |
| 2025-12-05 `arxiv-2512.05905` | [SCAIL](https://arxiv.org/abs/2512.05905) | [D] 3D-consistent pose、pose-shifted RoPE、studio-oriented character animation。 |
| 2025-12-05 `arxiv-2512.05394` | [Latent Spectral Biasing of Video VAEs](https://arxiv.org/abs/2512.05394) | [D] VAE latent spectrum/diffusability 分析；不是旗舰 safety report。 |
| 2025-11-11 `arxiv-2511.08195` | [UI2Code^N](https://arxiv.org/abs/2511.08195) | [D] rendered-feedback interactive visual optimization 与 RVPO。 |
| 2025-10-21 `arxiv-2510.18573` | [Kaleido](https://arxiv.org/abs/2510.18573) | [D] multi-subject reference video、R-RoPE 与 background disentanglement。 |
| 2025-09-10 `arxiv-2509.18119` | [MobileRL](https://arxiv.org/abs/2509.18119) | [D] difficulty-adaptive mobile GUI RL、positive replay 与 shortest-path reward。 |
| 2025-08-27 `arxiv-2508.19576` | [ReST-RL](https://arxiv.org/abs/2508.19576) | [D] code self-training + GRPO + VM-MCTS；不证明旗舰采用同一 recipe。 |
| 2025-01-17 `arxiv-2501.10132` | [ComplexFuncBench](https://arxiv.org/abs/2501.10132) | [D] multi-step/constrained/long-context function calling，1,000 cases/43 APIs。 |
| 2025-01-06 `arxiv-2501.02955` | [MotionBench](https://arxiv.org/abs/2501.02955) | [D] 5,385 videos/8,052 QA 的 fine-grained motion benchmark。 |
| 2025 `glm-simple-evals` | [GLM-SIMPLE-EVALS](https://github.com/zai-org/glm-simple-evals) | [C] 公开 evaluation harness；不能消除 API/model-version 差异。 |
| 2024-12-30 `arxiv-2412.21059` | [VisionReward](https://arxiv.org/abs/2412.21059) | [D] image/video 多维偏好、可解释线性组合与 consistent optimization。 |
| 2024-08-29 `arxiv-2408.16500` | [CogVLM2](https://arxiv.org/abs/2408.16500) | [D] CogVLM2/Video/GLM-4V 结构和数据分叉；family 不等于单模型。 |
| 2024-08-12 `arxiv-2408.06072` | [CogVideoX](https://arxiv.org/abs/2408.06072) | [D] 3D causal VAE、Expert Transformer、2B/5B diffusion models。 |
| 2024-07-04 `codegeex4-all-9b-model-card` | [CodeGeeX4-ALL-9B model card](https://github.com/zai-org/CodeGeeX4) | [C/D] completion/chat/interpreter/function-call code model；完整 corpus [U]。 |
| 2024-06-12 `arxiv-2406.08035` | [LVBench](https://arxiv.org/abs/2406.08035) | [D] 103 videos/117 hours/1,549 QA 的 extreme-long-video benchmark。 |
| 2024-05-07 `arxiv-2405.04312` | [Inf-DiT](https://arxiv.org/abs/2405.04312) | [D] block autoregression 把最低 image hidden-state memory 降为 linear-in-width。 |
| 2024-03-08 `arxiv-2403.05121` | [CogView3](https://arxiv.org/abs/2403.05121) | [D] 3B two-stage relay diffusion 与 progressive distillation。 |
| 2024-02-06 `arxiv-2402.04236` | [CogCoM](https://arxiv.org/abs/2402.04236) | [D] crop/zoom/grounding 的 Chain-of-Manipulations visual reasoning。 |
| 2023-12-14 `arxiv-2312.08914` | [CogAgent](https://arxiv.org/abs/2312.08914) | [D] 18B dual-resolution GUI grounding/navigation VLM。 |
| 2023-11-06 `arxiv-2311.03079` | [CogVLM](https://arxiv.org/abs/2311.03079) | [D] layerwise visual expert，在保语言能力同时深融合视觉。 |
| 2023-09-04 `arxiv-2309.03350` | [Relay Diffusion](https://arxiv.org/abs/2309.03350) | [D] 以 forward process 连接 cascaded resolutions，CogView3 方法前驱。 |
| 2023-06-13 `arxiv-2306.07906` | [WebGLM](https://arxiv.org/abs/2306.07906) | [D] retrieval + bootstrapped QA + preference scorer；不是 browser policy。 |
| 2023-04-12 `arxiv-2304.05977` | [ImageReward](https://arxiv.org/abs/2304.05977) | [D] 136,892 expert comparisons 的 text-to-image reward model 与 ReFL。 |
| 2023-03-30 `arxiv-2303.17568` | [CodeGeeX](https://arxiv.org/abs/2303.17568) | [D] 13B/23 languages/850B exposure 与 HumanEval-X。 |
| 2022-05-29 `arxiv-2205.15868` | [CogVideo](https://arxiv.org/abs/2205.15868) | [D] 9.4B、5.4M video pairs、key-frame→recursive interpolation。 |
| 2022-04-28 `arxiv-2204.14217` | [CogView2](https://arxiv.org/abs/2204.14217) | [D] 6B hierarchical low-res/direct/iterative super-resolution。 |
| 2021-05-26 `arxiv-2105.13290` | [CogView](https://arxiv.org/abs/2105.13290) | [D] 4B autoregressive image tokens、30M pairs、PB-relax/Sandwich-LN。 |
| 2021-03-01 `arxiv-2103.00959` | [CogDL](https://arxiv.org/abs/2103.00959) | [D/C] graph deep-learning library；研究宽度证据，不是 GLM architecture。 |
| 2020-07-07 `arxiv-2007.03179` | [GE-SpMM](https://arxiv.org/abs/2007.03179) | [D] GPU general-purpose sparse matrix multiplication；图系统前史。 |
| 未标日期 `visualglm-6b-model-card` | [VisualGLM-6B model card](https://github.com/zai-org/VisualGLM-6B) | [C/D] early bilingual image-dialogue model card；精确日期/recipe [U]。 |
| 未标日期 `slime-rl-framework` | [Slime](https://github.com/THUDM/slime) | [C] GLM post-training line使用的开放 RL scaling framework；持续演进。 |
| 未标日期 `realvideo-model-card` | [RealVideo model card](https://github.com/zai-org/RealVideo) | [C/D] video generation artifact；无独立 paper 时不虚构训练细节。 |
| 未标日期 `cogview4-model-card` | [CogView4 model card](https://github.com/zai-org/CogView4) | [C/D] CogView 后续 open artifact；不能由仓库名推断完整谱系。 |
| 未标日期 `codegeex2-model-card` | [CodeGeeX2 model card](https://github.com/zai-org/CodeGeeX2) | [C/D] GLM-based code model artifact；训练 corpus/steps 未完整公开。 |

### 14.3 Affiliated：明确工作级机构关联，不并入旗舰主线（30）

| 日期 / inventory ID | 一手记录 | 最小定位与归属边界 |
|---|---|---|
| 2026-03-31 `doi-63e381c54067` | [LPS-GNN](https://doi.org/10.1145/3801100) | [D] 100-billion-edge graph deployment；机构署名系统研究，非 GLM serving 报告。 |
| 2026-03-13 `doi-6adcf54f32c8` | [Hammer](https://doi.org/10.5194/egusphere-egu26-2906) | [D] hydro-science/engineering domain LLM；不证明进入 general GLM data。 |
| 2026-01-29 `doi-76a72c50d5eb` | [LLM for Non-Small Cell Lung Cancer TNM Staging](https://doi.org/10.2196/77988) | [D] prompt engineering + SFT 的临床分期验证；医疗用途需独立风险边界。 |
| 2026-01-01 `doi-b5ac1784f162` | [HoWToBench](https://doi.org/10.18653/v1/2026.acl-long.317) | [D] Tree-of-Writing holistic writing benchmark；affiliation 不等于 GLM 官方 benchmark。 |
| 2025-11-13 `doi-ff4744af38f0` | [Generalizing Graph Transformers Across Diverse Graphs and Tasks](https://doi.org/10.1109/tkde.2025.3632394) | [D] graph pretraining/generalization；与 language Transformer 主线分开。 |
| 2025-11-07 `doi-63483ce3c898` | [Dual Denoising Diffusion for Session-based Social Recommendation](https://doi.org/10.1145/3746252.3761031) | [D] recommendation diffusion；单篇机构署名不证明产品采用。 |
| 2025-10-25 `doi-63b3172b76d4` | [JPS](https://doi.org/10.1145/3746027.3754561) | [D] visual perturbation + textual steering 的 multimodal jailbreak 研究；非产品漏洞公告。 |
| 2025-10-19 `doi-fe9d52d319a1` | [VPO](https://doi.org/10.1109/iccv51701.2025.01451) | [D] prompt optimization 对齐 text-to-video；不自动属于 CogVideoX recipe。 |
| 2025-10-19 `doi-0161327cde57` | [OmniCache](https://doi.org/10.1109/iccv51701.2025.01513) | [D] diffusion Transformer training-free cache reuse；区别于 LLM KV/IndexCache。 |
| 2025-10-19 `doi-40d913e90679` | [Concat-ID](https://doi.org/10.1109/iccvw69036.2025.00202) | [D] identity-preserving video synthesis；广义视觉生成署名研究。 |
| 2025-09-06 `doi-c014f0f61392` | [Two-stage Large-scale Research-interest Mining](https://doi.org/10.1016/j.future.2025.108117) | [D] research-interest mining systems；不等同 agent retrieval training。 |
| 2025-08-03 `doi-19fbadf7fb9f` | [AtomR](https://doi.org/10.1145/3711896.3736849) | [D] atomic operators for heterogeneous knowledge reasoning；与 flagship tools 关系 [U]。 |
| 2025-06-30 `doi-c91d475ed0ae` | [GuidedLatent](https://doi.org/10.1109/ijcnn64981.2025.11227775) | [D] VAE membership-inference defense；不是 GLM privacy system card。 |
| 2025-06-18 `doi-2e5932d9e76d` | [CoRe](https://doi.org/10.48448/xy3g-n136) | [D] zero-shot table understanding/reasoning framework；广义应用署名。 |
| 2025-06-10 `doi-59d01c6b6c43` | [ICT](https://doi.org/10.1109/cvpr52734.2025.00398) | [D] image-object intervention 缓解 VLM hallucination；非 GLM-V 专属报告。 |
| 2025-06-01 `doi-1bb8b4c383d4` | [Knowledge Graphs and LLMs Review](https://doi.org/10.1007/s00607-025-01499-8) | [D] survey/research synthesis；不作生产数据证据。 |
| 2025-04-22 `doi-d54afeed9c74` | [Ask, Acquire, Understand](https://doi.org/10.1145/3696410.3714895) | [D] multimodal agent for social-abuse meme detection；独立应用框架。 |
| 2025-04-18 `doi-358aa28557a0` | [WebGLM Journal Version](https://doi.org/10.1145/3729421) | [D] efficient/reliable web-enhanced QA 的出版扩展；与 arXiv/KDD 工作相关但清单保留此版本。 |
| 2025-04-04 `doi-b115fec8d4fd` | [SoAy](https://doi.org/10.1145/3690624.3709412) | [D] solution-based LLM API use for academic information seeking；非 GLM agent card。 |
| 2025-03-26 `doi-89ee533f4cf7` | [MEPipe](https://doi.org/10.1145/3689031.3717469) | [D] memory-efficient slice-level pipeline scheduling；与旗舰 training stack 关系 [U]。 |
| 2024-10-21 `doi-310ff5abd04e` | [CogCartoon](https://doi.org/10.1007/s11263-024-02267-5) | [D] practical story visualization；Cog prefix 不证明共享 CogView weights。 |
| 2024-10-14 `doi-3698f3db7db6` | [MQE](https://doi.org/10.1109/iros58592.2024.10801682) | [D] multi-agent quadruped environment；robotics research 不并入 LLM agent RL。 |
| 2024-08-24 `doi-28922e27054e` | [Few-shot Node Classification on Text-attributed Graphs](https://doi.org/10.1145/3637528.3671952) | [D] graph pretraining/prompting；工作级关联。 |
| 2024-08-24 `doi-62639c9df9dc` | [OAG-Bench](https://doi.org/10.1145/3637528.3672354) | [D] human-curated academic graph mining benchmark；不等于 GLM-SIMPLE-EVALS。 |
| 2024-08-24 `doi-772911da4c84` | [AutoWebGLM](https://doi.org/10.1145/3637528.3671620) | [D] ChatGLM3-based web navigation、DPO+SFT、rejection fine-tuning；研究实验非商业配方。 |
| 2024-02-08 `doi-2789d0a381ef` | [Genre](https://doi.org/10.1007/s40747-023-01321-y) | [D] multi-turn QA for entity-relation extraction；广义 NLP 应用。 |
| 2022-10-16 `doi-777e742aac92` | [AMinerGNN](https://doi.org/10.1145/3511808.3557544) | [D] academic graph click-through prediction；与 GLM pretraining 分开。 |
| 2022-08-12 `doi-73d34dccba15` | [OAG-BERT](https://doi.org/10.1145/3534678.3539210) | [D] academic knowledge-service backbone；BERT/graph line而非生成式 GLM。 |
| 2020-06-23 `doi-00e509eb2b7a` | [AI 2000](https://doi.org/10.1145/3394231.3397925) | [D] decade-scale AI landscape analysis；最早机构署名记录之一。 |
| 未标日期 `xdai-framework` | [XDAI](https://github.com/THUDM/XDAI) | [C/D] Zhipu.AI-supported tuning-free grounded-dialogue framework；无独立 paper/date [U]。 |

## 15. 研究议程：从现有证据还能严格提出什么问题

以下是由一手资料综合出的研究问题 [I]，答案在当前来源宇宙中仍为 [U]：

1. **三轴 sparsity 联合分配：** experts、selected history tokens、cross-layer reused indices
   如何在固定 FLOPs/HBM/network/latency 下共同优化？现有 ablation 多分开研究。
2. **百万 token 的学习而非检索：** 1M needle/retrieval 成功不证明 policy 能在 compacted
   million-token trajectory 中稳定 credit assignment、更新计划并从 tool failure 学习。
3. **异步 on-policy 的定义：** quantization、serving kernels、tool version、context manager、
   policy lag 和 partial rollout 分别引入何种 distribution shift？哪些能 importance-correct？
4. **专家整合：** self-distillation、cross-stage OPD、parallel OPD 在 teacher 冲突、mode
   coverage 与 catastrophic forgetting 上如何公平比较？
5. **多域 verifier interference：** math exact check、code tests、IoU、speech rewards、OCR
   structure 和 LLM judges 的误差如何跨 domain 传播？RLCS 已观察到弱 verifier 可伤及他域。
6. **视觉生成到行动：** CogView/Video 的 generative priors 是否能提高 GUI grounding、
   UI2Code 或 visual-agent planning，需共享 checkpoint 的 controlled evidence，而非品牌推断。
7. **reward hacking 与 evaluation security：** test leakage、repository history、network access、
   hidden artifacts、judge manipulation 应有版本化 threat model 与 incident denominator。
8. **解释性：** router/indexer/attention/critic 的 causal intervention 能否预测长期 agent
   failure，而不是只在事后可视化？
9. **端侧与 specialist routing：** GLM-Edge、ASR/OCR/TTS/5V specialists 与通用旗舰之间的
   routing、distillation、privacy 和 update cadence 如何审计？
10. **总成本核算：** 数据、失败 runs、post-training rollouts、sandboxes、storage/network、
    评测、工程人力与能源怎样进入统一 denominator？

## 16. 使用本章时应避免的具体误述

- “GLM-5 就是放大的 2021 GLM。”——attention、MoE、MTP、objective/system 已多次换轨。
- “GLM-5 有 744B 参数，所以每 token 算 744B。”——报告明确是 40B active，且计数 convention
  与 config/hosting 有冲突。
- “GLM-4.5 训练集有 23T unique tokens。”——23T 是 processed exposure，components 约 23.1T。
- “1M context 表示训练和每次 API 调用都能有效使用 1M。”——训练/config/API/agent memory
  是四种不同长度。
- “IndexCache 让 GLM-5 快 4×。”——最多 75% 指 indexer computations；744B 结果还是
  preliminary Table 4。
- “ChatGLM3 会 function call，所以它经过 agentic RL。”——公开证据只确认 interface [C]。
- “GLM-4.5 论文给出了完整 policy objective。”——印刷式漏掉 score-function 项，不能复现。
- “SAO 的 Qwen3 实验超参就是 GLM-5.2 配方。”——论文只披露 deployed linkage，实验模型不同。
- “CogView/CogVideo 是 GLM-5V 的直接训练祖先。”——时间/组织相关不等于 weight/data lineage。
- “VisionReward/ImageReward 等于人类偏好。”——它们是有限 annotation/protocol 学到的 proxy。
- “官方 benchmark 表能做跨供应商排行榜。”——variant、prompt、sampling、tools、judge、日期不同。
- “affiliated 论文都进入 GLM。”——它只证明 work-level affiliation/support。
- “公开 weights 就是全开源。”——data、training code、rewards、compute 与 production stack 不完整。
- “95 条证明互联网再无遗漏。”——它只对截止日按发现规则冻结的公开来源宇宙逐条覆盖。

## 17. 完整性声明与更新规则

**[D]** 本章已对当前 inventory 的 95 条逐项给出 ID、primary URL 与边界短注：core 26、
direct 39、affiliated 30。发现流程检查 Zhipu 官方 Research index、`zai-org`/`THUDM`
repositories、primary arXiv/DOI pages 与 work-level raw affiliation；THUDM 只有在显式 Zhipu/
Z.ai evidence、当前 first-party repo linkage 或 foundational GLM lineage 时才纳入。Hugging
Face API 在 discovery 环境超时，因此 checkpoint-by-checkpoint hosting revisions 不作穷尽声明。

冻结 95 条之外，Section 5.4 透明列出了临近截止日、由旧 case study 核过的一手 SAO
deployment linkage；它不冒充第 96 条 inventory row。后续若 inventory 正式接纳该论文，
应先更新 provenance/count，再把本文数字由 95 改为新总数。

维护本章时应：

1. 保留 verified-through date，不把新 checkpoint 静默混入旧报告；
2. 先更新 inventory 与 discovery evidence，再改本章计数/逐篇地图；
3. DOI/arXiv/conference 同一工作按内容去重，实质不同 model card/report 分开；
4. 所有重大数字保留 section/page/table、单位、variant 与 denominator；
5. 跨模型比较重新核对 prompt、sampling、tools、context、judge 与 API date；
6. 只有保留 revision、environment、data hash、command、seed、logs 与 artifacts 的实测才标 [R]；
7. 任何无法由一手来源或公开制品确认的生产细节停在 [U]。

本章的目标不是用 95 个链接营造“全面感”，而是让读者逐条分清：**来源真正披露了什么、
公开制品确认了什么、综合判断依赖什么假设，以及证据在哪些地方要求我们停下来。**
