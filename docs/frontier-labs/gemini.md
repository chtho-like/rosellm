# Google DeepMind 与 Gemini：从通用学习、Alpha 系列到多模态基础模型的全研究谱系

> **审计口径（截至 2026-07-19）**：本章以冻结的
> `research/literature/inventory/gemini.jsonl` 为唯一逐条清单，共 **1,985** 条：
> **core 24 / direct 18 / affiliated 1,943**；类型为 **research paper 1,926 / model card 41 /
> technical report 17 / dataset 1**。其中 1,983 条有日期、2 条未标日期；1,985 条都有
> `primary_url`，共有 1,984 个不同 URL。审计时本地已有 **1,543** 份可检索 TXT 全文，正文只把能在
> 一手材料中定位的数字写成确定事实；其余记录仍由元数据、官方页面或论文入口支撑。

这不是一篇“Gemini 型号排行榜”。Google DeepMind 的研究主线至少同时包含通用强化学习、搜索与规划、
记忆与世界模型、语言与多模态基础模型、具身智能、Alpha 系列的算法发现与科学发现、安全治理，以及与
机构成员明确关联的数学、神经科学、生命科学和社会影响研究。Gemini 是这些路线在通用模型产品层的一次
汇流，不等于此前全部工作，也不能反向证明每篇 affiliated 论文进入了 Gemini 的训练配方。

## 1. 阅读规则：D/C/R/I/U、三层归属与可证伪性

本章逐项使用以下标签：

- **[D] Disclosed**：论文、技术报告、模型卡或第一方研究页明确披露；引用会给出 inventory ID，能获得
  全文时再给页码、表号、图号或章节。
- **[C] Confirmed artifact**：公开代码、权重、数据、模型卡文件或可检查配置确认。
- **[R] Reproduced**：在本仓库留下命令、配置、日志和结果制品的复现。本章目前没有把厂商 benchmark
  当作 [R]；没有本地复现实验就明确写“无 [R]”。
- **[I] Inference**：由若干 [D]/[C] 事实推出的综合判断；必须说明前提，并给出可以推翻它的观察。
- **[U] Unknown**：公开证据缺失、不同来源冲突或只能看到结果而看不到过程。

归属也分三层：

1. **core（24）**：Gemini 正式模型卡及直接对应的旗舰安全报告；回答“发布了什么、怎样评估和限制”。
2. **direct（18）**：标题、摘要或报告明确把 Gemini 当作方法、基座或被测对象；回答“Gemini 被怎样训练、
   扩展、评估或用于研究”。
3. **affiliated（1,943）**：官方 Google DeepMind 研究出版物，或有逐工作机构署名证据的广义论文/报告；
   只说明“这是该研究组织的知识与方法环境”，**不说明它进入任何 Gemini 版本**。

!!! warning "最重要的边界"
    “作者在 Google DeepMind”“论文出现在官方 publications 页”“主题与 Gemini 相近”都不足以证明该方法
    用于 Gemini。要把一项 affiliated 方法升级为旗舰配方，至少需要模型报告、模型卡、代码/config 或作者
    的第一方技术说明建立直接连接。

本章的综合判断都设计成可证伪：例如“Gemini 是多条研究线汇流”若被未来完整训练清单证明其只采用单一路线，
该判断应撤回；“模型卡公开面宽于训练 recipe”若未来卡片披露逐阶段数据、优化器与 RL 配方，也应更新。

## 2. 全量语料画像：1,985 条记录实际覆盖什么

清单不是均匀的“1,985 篇 Gemini 论文”。其中 1,943 条 affiliated 长尾占 97.88%，正是本章必须把
“组织研究史”和“旗舰模型事实”拆开的原因。[I] 主题标签是多标签，计数不能相加成 1,985：

| 研究面 | 标签命中数 | 本章怎样使用 |
|---|---:|---|
| 效率与系统 | 874 | 训练、推理、优化、分布式系统的组织能力背景；除非有 direct/core 桥接，不写进 Gemini recipe |
| 训练与 scaling | 678 | 基础模型、数据与优化的长期方法库 |
| 评测与 benchmark | 604 | 能力、安全、科学和社会评价；区分自报与独立评价 |
| 数学与科学 | 573 | Alpha 系列、算法发现、天气、物理与通用科学方法 |
| 多模态 | 503 | 视觉、语言、音频、视频与 action 的汇流路线 |
| interpretability | 495 | 表征、机制与审计；不自动等于部署安全 |
| code / software | 395 | AlphaCode、程序合成、算法发现与 agentic coding |
| foundation models | 394 | 表征学习、序列模型、语言与通用模型主干 |
| reinforcement learning | 380 | DQN—AlphaGo—world model—agent/robotics 的方法谱系 |
| society / governance | 376 | 治理、经济、教育、人机协作与影响研究 |
| agents / tool use | 372 | 环境、规划、工具、计算机使用与通用 agent |
| safety / alignment | 244 | dangerous capability、red teaming、watermarking、alignment audit |
| science / biology | 233 | AlphaFold、基因组、医学与生物发现 |
| reasoning | 158 | 数学、程序、搜索与语言推理 |
| robotics / embodied | 137 | 机器人操作、视觉—语言—动作与 on-device 路线 |
| long context / memory | 134 | 外部记忆、检索、长窗口和持续交互 |

按年份，记录从 2013 年起连续覆盖：2013: 6、2014: 17、2015: 35、2016: 77、2017: 73、
2018: 107、2019: 158、2020: 217、2021: 201、2022: 129、2023: 258、2024: 301、
2025: 286、2026: 118，另有 2 条未标日期。该分布是“清单收录量”，不是组织产出强度或模型贡献权重。[D/U]

## 3. 一张总图：八条长期路线怎样在 Gemini 汇流

| 路线 | 可观察的研究问题 | 代表性里程碑 | 与 Gemini 的关系边界 |
|---|---|---|---|
| 通用 RL、搜索与规划 | 怎样从像素、规则和反馈学策略 | DQN、AlphaGo/AlphaZero、MuZero、Dreamer | 长期方法祖先；不能仅凭相似性写成 Gemini 后训练 recipe |
| 记忆、序列与 scaling | 怎样扩大容量、上下文与训练效率 | DNC、Transformer 研究、Chinchilla、检索增强 | 构成基础模型背景；具体 Gemini 架构仍以报告/卡片为准 |
| 多模态生成与理解 | 怎样统一文本、图像、音频、视频 | Perceiver、Flamingo、Imagen/Veo、Gemini | Gemini core/direct 证据最强的汇流面之一 |
| reasoning、code 与 agents | 怎样把搜索、验证、工具和环境接入模型 | AlphaCode、AlphaGeometry、SIMA、AlphaEvolve | 一部分明确以 Gemini 为基座，其余仍属相邻谱系 |
| 科学与 Alpha 系列 | 怎样把学习系统变成发现器 | AlphaFold、AlphaTensor、GNoME、GraphCast/GenCast | 展示方法迁移，不应统称为 Gemini 产品能力 |
| robotics / embodied | 怎样从 perception 走到 action | RT-1/RT-2、Gemini Robotics、on-device | 2025 后出现 core/direct 桥接 |
| safety / governance | 怎样测危险能力、滥用、失控与社会风险 | dangerous-capability eval、FSF、watermarking | 模型卡给部署切片，FSF 给治理门槛；两者不是完整训练披露 |
| 社会影响与 human-AI | 怎样测教育、劳动、协作、文化与公平 | 教育现场研究、collective reasoning、经济与治理论文 | direct 个案与 affiliated 广谱证据并存，不能用单个结果概括社会净效应 |

**综合判断 [I]**：最稳妥的因果图不是“AlphaGo → Gemini”的单箭头，而是多条研究流在基础模型层汇合，
再向 agent、science 和 robotics 分叉。可推翻证据是：官方披露一份逐组件 genealogy，显示这些路线之间没有
技术、人员或评测接口的连续性。现有公开材料没有提供这样一份完整 genealogy。[U]

## 4. 基础模型：从记忆、序列建模与 compute-optimal training 到 Gemini

### 4.1 早期原语：表征、注意力、外部记忆与生成模型 [D]

早期路线的核心不是“参数越大越好”，而是**怎样让一个可微系统保存、检索、组合并执行状态**。
[DNC](https://doi.org/10.1038/nature20101)（`gdm-doi-10.1038-nature20101`）把神经网络控制器与
可读写外部矩阵记忆连接起来；`write/read weighting`、content lookup 与 temporal link 让模型能学习
图遍历、复制和问答式算法。它是 affiliated 方法祖先，不是 Gemini 采用 DNC 的证据。[D/U]

同一时期，PixelRNN/PixelCNN、条件图像生成、WaveNet 系列、Matching Networks、Neural
Programmer-Interpreters、synthetic gradients、learned optimizers 与 memory-augmented meta-learning 在清单中
形成多条可组合原语：自回归离散建模、条件生成、少样本适应、程序状态、跨模块 credit assignment 与优化器学习。
例如 [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)
（`gdm-arxiv-1606.05328`）和 [Learned Optimizers that Scale and Generalize](https://arxiv.org/abs/1703.04813)
（`gdm-arxiv-1703.04813`）分别代表生成建模与训练算法支线。[D] 它们解释了组织方法库的宽度；
没有 core/direct 桥接时，不能写成 Gemini 组件。[U]

两个常被忽略的连接是：

1. **记忆不是上下文窗口的同义词。** 固定状态 recurrent model、显式外部记忆、检索库和全注意力窗口有不同
   写入规则、误差传播、容量与延迟；“支持长文本”不说明使用哪一种。
2. **多模态不是把独立 encoder 的输出拼接起来就结束。** 视觉/音频 token 的时间采样、位置编码、分辨率、
   交错顺序、生成头和损失权重都会改变模型学到的跨模态条件分布。

**[I] 可证伪综合。** 这些工作提供了“状态、生成、优化、组合”的原语库。若未来 Gemini 完整 recipe 显示其
没有使用、继承或重新实现任何同类机制，则应撤回“方法环境连续”的判断；当前证据只支持概念谱系，不支持
逐权重 genealogy。

### 4.2 scaling 与数据—算力配置：能知道什么，不能知道什么 [D/I/U]

Gemini 1.0 报告只披露到一个重要但不充分的层级：最大模型的 token 预算“followed the approach” of
compute-optimal scaling，较小模型为部署质量训练更多 token；最终 mixture/weights 由小模型 ablation 决定，
训练后段提高 domain-relevant data 权重（`gdm-url-0453ae158a21`，PDF p.5）。报告**没有**给出模型参数量、
总 token 数、每种来源比例、去重阈值或完整许可清单。[D/U] 因而不能从“compute-optimal”四个字反推出
公开文献中的某个固定 token/parameter 比就是生产配方。

系统层证据更具体：Gemini 1.0 在 TPUv4/v5e 上训练；Ultra 使用 TPUv4 大型 fleet，而 SuperPod 单元为
4,096 chips；JAX、Pathways 与 MegaScale XLA 组织分布式计算，报告还讨论 silent data corruption 与
goodput（同一 ID，PDF pp.4–5）。这些披露证明硬件/编译/容错与模型共同扩展，但不披露 fleet 总芯片数、
总 TPU-hours、能耗和总成本。[D/U]

affiliated 系统论文提供了更可检查的对照，而非 Gemini recipe：

- [DiLoCo](https://deepmind.google/research/publications/57039/)（`gdm-arxiv-2311.08105`）让多个 worker
  做多步 local update 后才同步 outer optimizer；PDF Figure 1（p.3）把 worker replication、inner training、
  averaging 与 outer update画成一个循环，Table 1（p.3）给出实验模型配置。它研究低通信训练，不证明
  Gemini 采用相同同步周期。[D/U]
- [RecurrentGemma](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf)
  （`gdm-url-1334f9ac0ca9`）使用 Griffin 的 linear recurrence + local attention，固定状态降低生成期内存；
  Table 1（PDF p.1）给出 2.7B 总参数等开放模型配置，Figure 1（p.4）报告 TPUv5e throughput。它是一个
  可观察的架构反例：基础模型不必把全局 KV cache 随序列线性扩大。[D/C]
- [Gemma](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)
  （`gdm-url-d2b992c527cc`）和 [Gemma 3](https://deepmind.google/models/gemma/gemma-3/)
  （`gdm-arxiv-2503.19786`）提供权重/报告层的开放窗口；它们可用于独立实验，但不能当作 Gemini 参数量或
  data mixture 的代理。[C/U]

**[I]** 可迁移结论是“scaling 是 compute、data、communication、state、compiler 与 serving 的联合约束”；
不可迁移结论是“某个开放 Gemma 配方就是闭源 Gemini 配方”。若未来完整训练日志显示 Gemini 性能变化可由
单一参数轴解释，联合约束判断才被推翻。

### 4.3 Gemini 1.0 / 1.5：原生多模态与长上下文的正式公开面 [D/C/U]

[Gemini 1.0](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf#page=71)
（`gdm-url-0453ae158a21`）公开的可核验主干是：

- Ultra / Pro / Nano 三档，构建于 Transformer decoder；32K context，并举 multi-query attention 为高效
  attention 例子（PDF p.3，Table 1）。
- 文本、图像、音频、视频联合训练；图像/视频可与文本、音频交错，视觉编码明确说受 Flamingo、CoCa、PaLI
  启发，同时强调从训练开始就多模态（PDF pp.3–4，Figure 2）。
- pretraining 后产生面向 conversational app 与 developer API 的不同 post-trained variants；PDF p.20
  披露 SFT + reward model + RLHF 的轮廓，p.21 说明 human-feedback utility 依赖 prompt selection 与
  candidate sampling。[D]
- 公开 benchmark 声称 Ultra 在 32 项中 30 项达到当时 SOTA、MMLU >90%、MMMU 62.4%；这些是报告协议下
  的 vendor results。报告自己的 leaked-data analysis 又指出，加入与 HellaSwag 相关网页提取会改变验证表现，
  这是“benchmark 不是纯能力常数”的一手警告（PDF pp.2、8）。[D]

[Gemini 1.5](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf#page=105)
（`gdm-url-49ab4305f7af`）把主矛盾转到**长上下文的训练、服务和评价**：Pro 与 Flash；摘要声称
text needle recall 在 1M 内 >99.7%（Figure 1，PDF p.2），并把极限实验扩到约 10M text tokens、9.7M
audio tokens 与长视频。报告说加入 sparse/dense scaling、distillation 与 serving improvements，但没有给出可复现
MoE routing、专家数、optimizer 或完整训练 mixture。[D/U]

长上下文结论必须拆成四层：

1. **needle retrieval**：在可控位置找回人工插入信息；可测 reachability，不等于理解。[D]
2. **next-token likelihood**：报告称上下文增加仍改善预测；这是平均统计，不保证关键事件推理。[D]
3. **结构性任务**：长文 QA、长视频 QA、ASR、代码仓库等更接近真实使用，但仍受 benchmark construction 影响。[D]
4. **工作流效用**：10 类职业任务的参与者估计节省 26–75% 时间（PDF p.37，Figure 19）；这是特定参与者、
   任务与自报/评价设置，不是宏观生产率因果估计。[D/U]

**[I]** 1.5 的技术转折不是“窗口从 32K 变成 1M”这么简单，而是把 training curriculum、state/attention、
distillation、serving 和 multimodal temporal sampling 同时推向长序列。可推翻证据是逐模块 ablation 显示只改
position extrapolation 就能复现全部增益；公开报告没有这样的证据。[U]

### 4.4 Gemini 2.x / 3.x：model-card 谱系、thinking 与产品分化 [D/C/U]

2.x 以后，冻结清单的核心证据从长 technical report 转向版本化 model card：

| 时间 | 核心记录 | 可确认 | 仍未知 |
|---|---|---|---|
| 2025-04 | [Gemini 2.0 Flash](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-0-Flash-Model-Card.pdf) (`gdm-url-0e2ab16c7b9c`) / [Flash-Lite](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-0-Flash-Lite-Model-Card.pdf) (`gdm-url-2604411a6472`) | 用途、数据类别、限制、安全评价与部署切片 [D] | 参数、token 总量、optimizer、全部后训练数据 [U] |
| 2025-06–08 | [2.5 Pro](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Pro-Model-Card.pdf) (`gdm-url-082786c15ca1`) / [Deep Think](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf) (`gdm-url-f9179a0c9b65`) | thinking/reasoning 用途；卡片列出 conditional pretraining、SFT、human and critic feedback RL；FSF 评价 [D] | RL optimizer、reward mixture、rollout 数、test-time budget [U] |
| 2025-09–10 | Flash-Lite、Flash/Image 与 Computer Use cards | 产品分型、多模态/GUI action 风险与相应 eval [D] | capability 增量来自 checkpoint、harness、tools 还是 policy training 的分解 [U] |
| 2025-11–12 | [Gemini 3 Pro](https://deepmind.google/models/model-cards/gemini-3-pro/) (`gdm-url-d94bc4796e34`) / [3 Flash](https://deepmind.google/models/model-cards/gemini-3-flash/) (`gdm-url-78c6fa8eafe5`) / 3 Pro Image | 数据大类、用途、限制、安全结果；Pro card PDF p.9 称未达到当时 FSF CCL [D] | 等价 1.0/1.5 深度的训练报告 [U] |
| 2026-02–03 | [3.1 Pro](https://deepmind.google/models/model-cards/gemini-3-1-pro/) (`gdm-url-d2aed2990254`) / [3.1 Flash-Lite](https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/) (`gdm-url-c6d5b2d93460`) / 3.1 Image | 官方卡片存在、版本与 intended-use/safety 页面 [D/C] | 系列内部蒸馏、路由和数据差异 [U] |
| 2026-04–06 | Robotics-ER 1.6、3.1 Flash Audio、3.5 Flash、Omni Flash、3.5 Audio、3.1 Flash-Lite Image | 官方 model-card 索引与 HTML/PDF 链接已归档 [C] | 若只凭标题，不能推断具体训练方法或相对能力 [U] |

2.5 Pro 卡片 PDF p.10 的“reinforcement learning from human and critic feedback”是重要披露，但其粒度仍只有
mechanism class；同卡 Table 1（p.12）总结 FSF critical capability evaluation。Deep Think 卡片 p.7 重复同类
后训练层，Figure 3（p.14）还明确说明 cyber challenge 的 attempt budgets 与旧版本套件变化，使不同模型数字
不能无条件横排。[D]

**[I] 配方透明度结论。** 型号越新，公开的用途/限制/危险能力评价更体系化；但架构、数据与优化披露没有随
版本号单调增加。若未来 2.x/3.x 正式报告给出模型 size、mixture、optimizer、RL trajectories 与 ablation，
应把相应 [U] 升级为 [D]，而不是把当前空白永久化。

## 5. 多模态：理解、生成、音频、视频与 action 不是一个任务

### 5.1 表征统一：Perceiver、Flamingo 与通用视觉语言接口 [D]

Gemini 1.0 的“native multimodal”有明确、有限的技术含义：多种模态在预训练中联合出现，视觉/音频 token
可以与文本交错，视频按 frame sequence 进入大上下文；它**不等于**所有生成模态共用同一个输出 decoder，
也不证明 Imagen、Veo 或 Lyria 权重被合并。[D/U]

清单中的广义视觉/语言工作覆盖 slot/scene decomposition、contrastive representation、video-language、
open-vocabulary detection、tracking、3D、prosody 与 universal audio representation。它们说明 organization
长期研究接口丰富。[I] 一个更有用的分解是：

$$
p(y\mid x)=p(y\mid z_{1:T}),\qquad
z_{1:T}=E_m(x; r, f, \tau, \pi),
$$

其中 $E_m$ 是模态 encoder，而 resolution $r$、frame sampling $f$、timestamp/temporal order $\tau$、
interleaving policy $\pi$ 都会改变 token stream。这个式子是本章的分析框架 [I]，不是 Google 披露的
Gemini 方程。[U]

[Gemini 1.0](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf#page=71)
（`gdm-url-0453ae158a21`）Figure 1（PDF p.3）以手写物理题展示 image+text reasoning，Figure 5（p.15）展示
多模态到 code 的交互，Table 13（p.19）展示多轮音频/图像序列。视觉抽样确认 Figure 1 与架构段同页：报告
的证据是输入、输出和定性/定量评价，而不是完整 visual tokenizer 配方。[D/U]

[Image Generators are Generalist Vision Learners](https://deepmind.google/research/publications/240658/)
（`gdm-arxiv-2604.20329`）则把生成模型的内部表示作为通用视觉学习对象；这是 direct Gemini-adjacent 研究，
仍不能反推某一 Gemini image card 的训练 recipe。[D/U]

### 5.2 图像、视频与音频生成：Imagen、Veo、Lyria 与 Gemini media cards [D/C/U]

生成路线必须至少分为四类：

| 输出 | 冻结清单代表记录 | 可审计边界 |
|---|---|---|
| image | [Imagen 4 Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Imagen-4-Model-Card.pdf?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=) (`gdm-url-1d5924e59d3f`) | affiliated card；确认用途、限制与安全切片，不等于 Gemini core |
| video | [Veo technical report](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf) (`gdm-url-93a933910a46`) 与 [Veo 3 card](https://storage.googleapis.com/deepmind-media/Model-Cards/Veo-3-Model-Card.pdf) (`gdm-url-07dc6591b390`) | generation system 与部署 card 分开；版本、prompt、duration 必须绑定 |
| music/audio | [Lyria 3 Model Card](https://deepmind.google/models/model-cards/lyria-3/) (`gdm-web-f9c42b10bb60`) | 官方 HTML card；不能从品牌共现推出与 Gemini 共享 decoder |
| unified/image variants | Gemini 2.5 Flash Image、3 Pro Image、3.1 Flash Image、Omni Flash 等 core cards | 证明 Gemini 品牌下存在 image/media 分型；具体 cross-modal parameter sharing 多为 [U] |

生成与理解也不能用同一 benchmark 衡量：文本到图像涉及 prompt adherence、aesthetics、spatial relations、
identity consistency 与 safety；视频再加 temporal coherence、camera motion 与物理一致性；音频再加韵律、
speaker/timbre、音乐版权与语言覆盖。平均一个“多模态分数”会丢掉这些可操作失败模式。[I]

### 5.3 multimodal evaluation：平均分掩盖了哪些失败模式 [D/I/U]

三种常见评价错位：

1. **协议错位。** Gemini 1.0 MMLU 使用 uncertainty-routed chain-of-thought 答案选择，而不同外部模型可能用
   不同 few-shot、CoT、tool 与 sampling。报告 Table 2（PDF pp.8–9）是版本化快照，不是永恒名次。[D]
2. **检索错位。** Gemini 1.5 needle >99% 证明定位人工标记的能力；Michelangelo 等 latent-structure long-context
   评价（`gdm-arxiv-2409.12640`）专门强调“能找针”不等于能整合隐藏结构。[D/I]
3. **judge 错位。** 开放式图像、视频、长回答常依赖自动或人类偏好 judge；若 judge 与被测模型共享训练分布，
   相关误差可能被平均数掩盖。[U]

因此，本章只接受形如

$$
M=(\text{checkpoint},\text{prompt},\text{shots},\text{tools},
\text{context},\text{sampling},\text{judge},\text{date})
$$

的版本化测量对象 [I]。若两个数字的 $M$ 不同，就不做无条件横排。可推翻这一规则的证据是：评价者证明
这些维度对结果无影响；当前一手材料反而显示它们会影响结果。[D]

## 6. Reasoning、code、agents 与 RL：四层系统必须分开

### 6.1 从 DQN 到 AlphaZero/MuZero：搜索、价值与模型学习 [D]

[DQN](https://doi.org/10.1038/nature14236)（`gdm-doi-10.1038-nature14236`）把卷积网络、Q-learning、
experience replay 与 target network 结合，从像素学习 Atari policy；[AlphaGo](https://doi.org/10.1038/nature16961)
（`gdm-doi-10.1038-nature16961`）把 human-game policy、self-play policy、value network 与 Monte Carlo tree
search 组合；[AlphaGo Zero](https://doi.org/10.1038/nature24270)（`gdm-doi-10.1038-nature24270`）去掉人类棋谱，
以自博弈统一 policy/value learning；[AlphaZero](https://doi.org/10.1126/science.aar6404)
（`gdm-doi-10.1126-science.aar6404`）再把规则接口扩到 chess、shogi 与 Go。[D]

这条线的核心变化不是“用了 RL”而是 model/search/target 的重构：

| 系统 | 学什么 | 搜索依赖什么 | 监督/奖励来源 |
|---|---|---|---|
| DQN | $Q(s,a)$ | 无 tree search | environment return + bootstrap |
| AlphaGo | policy + value | known game rules + MCTS | human games + self-play outcomes |
| AlphaZero | policy + value | known game rules + MCTS | self-play outcomes |
| MuZero | representation + dynamics + reward + policy/value | learned latent dynamics + MCTS | environment observations/rewards |

[MuZero](https://doi.org/10.1038/s41586-020-03051-4)（`gdm-arxiv-1911.08265`）PDF Figure 1（p.3）把
representation、dynamics、prediction 三个函数及 planning/training loop 画在一起；Table 1（p.6）分大/小
Atari data regimes。它只预测规划相关的 reward/value/policy，不要求重建全部 observation。[D]

**[I]** 从这条谱系能迁移到 LLM agent 的是“proposal、value/judge、environment transition 与 search 可以分工”；
不能迁移的是“Gemini post-training 就是 AlphaZero/MuZero”。语言工具环境有开放 action space、不可逆外部
副作用、非平稳网页与 learned judges，公开卡片也没有这样披露。[U]

### 6.2 reasoning 与可验证任务：AlphaCode、AlphaGeometry、AlphaProof [D/I]

[AlphaCode](https://doi.org/10.1126/science.abq1158)（`gdm-arxiv-2203.07814`）展示了一个必须与
policy-gradient RL 区分的系统：在 Codeforces 问题上大规模 sampling，经 behavior-based filtering 与 clustering，
再受限提交；PDF Figure 1（p.3）给出十场竞赛排名，Table 1（p.8）给出 CodeContests 数据统计，filtering
细节在 p.13。它的能力来自生成模型 + 搜索/选择/执行反馈，不能简写成“RL 代码模型”。[D]

随后几个 Alpha 节点把**可执行验证器**逐步放到中心：

- [AlphaTensor](https://doi.org/10.1038/s41586-022-05172-4)（`gdm-doi-10.1038-s41586-022-05172-4`）把
  matrix multiplication decomposition 变成有限因子分解游戏；PDF Figure 2（p.3）是 network/search 概览，
  Extended Data Table 1（p.16）列出组合分解结果。[D]
- [AlphaDev](https://doi.org/10.1038/s41586-023-06004-9)（`gdm-doi-10.1038-s41586-023-06004-9`）在 assembly
  指令空间用 deep RL 搜索 sorting routines；Table 1（PDF p.3）比较指令数/性能。正确性测试、latency 与
  真实库集成是不同 gate。[D/C]
- [FunSearch](https://doi.org/10.1038/s41586-023-06924-6)（`gdm-doi-10.1038-s41586-023-06924-6`）是预训练
  LLM + evaluator + evolutionary program database；Figure 1（PDF p.14）给流程，Table 1（p.8）给 bin-packing
  结果。它是 search/evolution，不是 proposal LLM 的在线 policy RL。[D]
- [AlphaProof](https://doi.org/10.1038/s41586-025-09833-y)（`gdm-doi-10.1038-s41586-025-09833-y`）明确是
  AlphaZero-inspired RL：Lean tactic environment、auto-formalized problems、tree search 与 test-time RL；
  Figure 1（PDF p.4）给核心 reasoning components，Table 1（p.8）分 search budget/TTRL 报告。[D]

**[I] 统一视角。** 这些系统共同把候选空间 $\mathcal A$、生成器 $q(a)$、可执行评分 $V(a)$ 与搜索策略
$S(q,V,B)$ 分开；差异在于 $q$ 是否被 RL 更新、$V$ 是否完备、预算 $B$ 与结果能否形式验证。若没有证据
显示 proposal model 参数更新，就只能写 search/selection，不能写 RL。

### 6.3 Gemini agent：tool use、computer use、SIMA 与 AlphaEvolve [D/C/U]

[SIMA 2](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/SIMA_Tech_Report_2025.pdf)
（`gdm-url-d41003b9db7f`）是一个明确的 Gemini bridge：基于 Gemini foundation model，在多种 3D virtual worlds
中 reason、dialogue、act；Figure 1（PDF p.1）还显示由 Gemini 生成 tasks/rewards 的 self-improvement loop。
但“自我改进”必须绑定虚拟环境、任务生成器、reward 与安全边界，不能外推成生产环境无限递归改写。[D/U]

[AlphaEvolve](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=)
（`gdm-url-119a83ec2a50`）把 Gemini 2.0 Flash/Pro ensemble、prompt sampler、program database、automated
evaluators 与 evolutionary loop 组合；Figure 1（PDF p.3）给高层架构，p.7 披露所用 Gemini ensemble。
它适用于候选能自动执行/评分的算法问题；judge 不充分时仍可能优化 proxy。[D/U]

[Gemini 2.5 Computer Use card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Computer-Use-Model-Card.pdf)
（`gdm-url-98dcfe57eb6c`）证明 GUI/computer-use 被作为单独部署风险面建卡，而不是只给通用聊天卡。[D]
其端到端表现仍是 checkpoint、system prompt、browser/runtime、action schema、confirmation policy、credential
access、budget 与 monitor 的联合函数：[I]

$$
P(\text{success})=F(\theta,H,E,A,B,G,S),
$$

其中 $\theta$ 是模型、$H$ harness、$E$ environment、$A$ authorization、$B$ budget、$G$ grader、$S$ safeguards。
这个分解是可审计分析式，不是厂商训练公式。

### 6.4 post-training / RL 的公开边界 [D/U]

公开材料支持三个强度不同的层级：

1. **Gemini 1.0 [D]**：SFT、reward modeling、RLHF；报告 PDF pp.20–21 讨论 feedback collection，但不给 PPO/
   REINFORCE、KL coefficient、batch、rollout 数或 reward weights。
2. **Gemini 2.5 [D]**：card 明列 conditional pretraining、SFT、RL from human and critic feedback、safety policies
   与 product mitigations（Pro p.10；Deep Think p.7）。这证明 AI critic 进入至少一类 reward loop，但不披露
   critic identity、calibration 或 optimizer。[U]
3. **agent/science systems [D]**：AlphaProof 有形式验证 RL；SIMA 2 用 Gemini 生成任务/奖励；AlphaEvolve 是
   evolutionary selection。它们不能互相替代，也不能合并成“Gemini 都用 agentic RL”。

没有 [R] 的原因很具体：缺少生产 prompts/trajectories、sampling distribution、reward services、policy/value
checkpoints、optimizer/hyperparameters、环境镜像、失败 run、总 rollout compute 与安全过滤规则。未来若只发布
更多 benchmark，而不发布这些变量，复现级别仍不会从 [D] 自动升为 [R]。

## 7. Alpha 系列：不是一个模型家族，而是“学习 + 搜索 + 验证器”的方法族

### 7.1 游戏与通用决策：AlphaGo、AlphaZero、MuZero [D]

Alpha 标签在游戏线上至少经历四次目标变化：

1. AlphaGo 把 imitation、self-play、value estimation 与 rule-based search 组合；
2. AlphaGo Zero 去掉人类棋谱，以棋局胜负和规则建立闭环；
3. AlphaZero 把同一抽象扩到三种 perfect-information board games；
4. MuZero 不再要求搜索器拿到规则转移，而学习只保留 planning-relevant information 的 latent dynamics。

这里“通用”的含义是**跨特定规则环境复用算法结构**，不是 unrestricted general intelligence。[D/U]
[AlphaStar](https://doi.org/10.1038/s41586-019-1724-z)（`gdm-doi-10.1038-s41586-019-1724-z`）又引入
multi-agent league/population 与更复杂 partial observation/action；[Agent57](https://deepmind.google/research/publications/41580/)
（`gdm-web-e373853f5bf8`）则把 exploration、episodic/meta controller 与多策略价值学习组合，针对 Atari 全套
任务的不同 horizon 与 reward scales。[D]

这些系统共同表明 aggregate score 后面有不同 failure surface：

- search-heavy system 会受 simulation fidelity、branching factor 与 evaluator bias 限制；
- population system 会受 opponent distribution、league update 与 exploitability 限制；
- learned world model 会受 model error 与 planning exploitation 限制；
- sparse-reward exploration 会受 intrinsic signal 与 horizon 限制。

**[I] 可证伪结论。** “Alpha 方法族”的稳定核心是把 experience generator、policy/value/model、search/selection
与 verifier/environment loop 组合，而不是某一 network architecture。若未来逐系统消融显示这些组件互不影响，
该方法族归纳就应拆除；现有一手工作反而以组件接口为主要创新对象。[D]

### 7.2 code、数学与算法发现：AlphaCode、AlphaTensor、AlphaDev、AlphaGeometry、AlphaEvolve [D]

在算法发现线上，“验证器质量”决定结论强度：

| 系统 | candidate | evaluator | 参数是否因 evaluator 更新 | 最强可说结论 |
|---|---|---|---|---|
| AlphaCode | source program | compile/test + filtering/clustering | 论文系统以 sampling/selection 为中心 | 竞赛代码生成与搜索 [D] |
| AlphaTensor | tensor decomposition action | exact multiplication identity/rank | self-play RL 更新 policy/value | 在定义空间发现可验证分解 [D] |
| AlphaDev | assembly program | correctness tests + latency/instruction metrics | deep RL | 发现并部署部分 sorting routines [D/C] |
| FunSearch | Python function | executable problem score | evolutionary database；LLM 可固定 | LLM-guided program search [D] |
| AlphaProof | Lean tactics/proof | Lean kernel | RL + tree search + TTRL | 形式化命题的机器核验 proof [D/C] |
| AlphaEvolve | code diff/program | user-defined automated evaluators | evolutionary selection；Gemini ensemble proposal | 有 evaluator 的算法/系统优化 [D] |

这张表也说明“结果可验证”不是“研究结论无误”。形式证明验证器能确认给定 theorem formalization；不能自动确认
自然语言题是否形式化正确。benchmark tests 能确认样例；不能自动确认对全部输入的 correctness。latency benchmark
能测特定硬件/compiler；不能证明普遍最快。[U]

AlphaEvolve 报告还给出一个重要反身案例：它用于改善与 Gemini 训练相关的 kernel/系统问题（PDF p.15），但
这不意味着 agent 可无约束改写自身生产系统；evaluator、review、deployment gate 与 rollback 仍是系统的一部分。[D/U]

### 7.3 科学发现：AlphaFold、GNoME、天气与物理 [D/C/U]

科学线与游戏线共享“结构化候选 + 专业评价”，但候选、观测噪声与证伪周期完全不同：

- AlphaFold 的候选是 biomolecular coordinates，训练/评价来自结构数据库与实验 benchmark；
- GraphCast/GenCast 的候选是天气状态/分布，评价来自再分析与 operational forecast；
- materials 路线把 crystal stability prediction 与 automated synthesis 串联；
- AlphaGenome 的候选是序列变异效应，评价依赖 genomic assays 与独立生物学验证；
- tokamak control 把 learned policy 接到 safety-critical physical dynamics。

[An autonomous laboratory for inorganic materials](https://doi.org/10.1038/s41586-023-06734-w)
（`gdm-doi-10.1038-s41586-023-06734-w`）把计算候选数据库与 A-Lab synthesis/characterization 闭环连接；论文
Supplementary Table 1（本地 PDF p.1 可见引用）还检查未获得目标的 synthesis 与 computational failure modes。
这类工作比“模型预测一个分数”多一个物理验证层，但仍受 experiment selection、measurement 与 negative-result
reporting 影响。[D/U]

**[I]** Alpha 系列最有迁移价值的不是品牌，而是逐层收紧 claim：生成候选 → 自动评价 → 独立/物理验证 →
数据库/库/流程部署。任何跳过中间层的“发现了”都应降级为 hypothesis generation。若后续独立复验持续失败，
相应发现声明必须撤回。

## 8. Science 与 biology：从预测 benchmark 到可用科学基础设施

### 8.1 AlphaFold 及数据库：模型、数据库与影响研究要拆开 [D/C]

[AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2)
（`gdm-doi-10.1038-s41586-021-03819-2`）报告在 CASP14 中显著超过竞争方法，并给出 median backbone accuracy
0.96 Å r.m.s.d.95（PDF p.2）；这是一组 blind benchmark 任务下的结构预测结果，不是所有 protein、complex、
conformation 或 biological function 的普遍保证。[D/U]

[AlphaFold Protein Structure Database](https://doi.org/10.1093/nar/gkad1011)
（`gdm-doi-10.1093-nar-gkad1011`）到 2024 版本报告覆盖超过 214 million protein sequences。数据库规模、
confidence、version、taxonomy 与 access 是 [C]；每条结构都经过实验确认则是错误结论。[U]

[AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
（`gdm-doi-10.1038-s41586-024-07487-w`）把对象扩到 protein、nucleic acid、small molecule、ion 与 modified
residue complexes，并以 diffusion module 取代 AF2 的 structure module（PDF pp.2–3）；Extended Data Table 1
用于分任务比较。它是目标空间与架构的变化，不是 AF2 单纯放大。[D]

应始终拆开三个实体：

1. **model claim**：在固定 dataset/split/metric 上的预测准确度；
2. **database artifact**：版本化预测、置信度、接口与许可；
3. **impact claim**：研究者是否改变实验、节省时间、发现机制或开发 therapeutic。

第三类需要用户研究、实验 validation、counterfactual 或长期追踪；不能由第一类 benchmark 自动推出。[I]

### 8.2 genomics、medicine、materials、weather 与 fusion [D/U]

### 天气与气候

[GraphCast](https://deepmind.google/research/publications/22598/)（`gdm-doi-10.1126-science.adi2336`）以 graph
neural network 做 deterministic global medium-range forecast；[GenCast](https://deepmind.google/research/publications/68149/)
（`gdm-arxiv-2312.15796`）转向 diffusion-based ensemble。GenCast Figure 1（PDF p.2）给 conditional diffusion
forecast loop，报告 p.3 称单个 15-day forecast 约 8 minutes on Cloud TPUv5、并与 ECMWF ENS 比较。[D]
deterministic point forecast 与 calibrated ensemble distribution 回答不同问题；不能只取一个 aggregate skill score
判定极端事件风险。[I]

### 基因组、医学与药物

[Med-Gemini](https://deepmind.google/research/publications/87645/)（`gdm-arxiv-2405.03162`）是 direct bridge：
以 Gemini 为基础，针对 radiology、histopathology、ophthalmology、dermatology 与 genomic data 调优；Figure 1
（PDF p.3）给 family/evaluation 流程，Table 1（p.8）列出超过 7 million samples、3.7 million medical images/cases
的 fine-tuning datasets。[D] 这些是研究 benchmark；临床部署还需要 prospective validation、workflow、calibration、
subgroup drift、human factors 与监管证据。[U]

[AlphaGenome](https://doi.org/10.1038/s41586-025-10014-0)
（`gdm-doi-10.1038-s41586-025-10014-0`）统一 long DNA sequence 的 regulatory variant effect prediction；
[TxGemma](https://deepmind.google/research/publications/153799/)（`gdm-arxiv-2504.06196`）则探索 therapeutics 的
efficient/agentic LLM。二者一个更接近 sequence-to-assay prediction，一个更接近 language/tool workflow；不能因
都服务生命科学就合并评价。[D/I]

### 材料、物理与控制

清单还覆盖 neural-network wavefunctions、lattice field sampling、quantum circuits、tokamak magnetic control、
fluid/physics simulation、materials diffusion 与 autonomous microscopy。代表性的
[magnetic control of tokamak plasmas](https://doi.org/10.1038/s41586-021-04301-9)
（`gdm-doi-10.1038-s41586-021-04301-9`）说明 RL 可进入真实控制；但 hardware interlock、sim-to-real gap、
operator policy 与故障安全仍不能由 policy score 吸收。[D/U]

### 8.3 “AI for science”强结论的证据门槛 [I/U]

本章采用五级证据梯：

| 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|
| S0：离线 benchmark | 在固定 split/metric 上相对性能 | 现实 utility、机制发现、外部分布安全 |
| S1：blind/temporal external test | 对选择偏差更稳健的预测证据 | 使用流程中的净收益 |
| S2：prospective/physical validation | 候选在新实验或现实时间序列中成立 | 大规模部署成本收益 |
| S3：workflow/controlled field study | 在规定人群/流程中的因果效应 | 跨地区、跨制度普遍效应 |
| S4：replicated deployment + monitoring | 多场景效果、漂移与失败率 | 永久安全或无未知风险 |

AlphaFold CASP blind benchmark 主要在 S1；A-Lab 的部分 synthesized targets进入 S2；Sierra Leone 教育 RCT
进入特定场景 S3；模型卡/benchmark 本身通常只到 S0/S1。[I] “AI for science 已解决某领域”必须同时说明
对象、证据等级、失败集与外推范围。可推翻这个保守结论的证据是独立、多场景、长期复验与机制确认。

## 9. Robotics 与 embodied intelligence：从控制到视觉—语言—动作

### 9.1 机器人 RL、simulation 与通用控制 [D]

早期具身路线同时包含：continuous control、distributed off-policy manipulation、sim-to-real locomotion、tactile
typing、table tennis、multi-task visuomotor imitation、representation/exploration 与 real-world challenge taxonomies。
例如 [deep RL for robotic manipulation](https://doi.org/10.1109/icra.2017.7989385)
（`gdm-doi-10.1109-icra.2017.7989385`）与 [sim-to-real quadruped locomotion](https://doi.org/10.15607/rss.2018.xiv.010)
（`gdm-doi-10.15607-rss.2018.xiv.010`）分别代表 asynchronous off-policy control 与 randomized simulation transfer。
[D]

三个不可合并的训练问题是：

- **skill acquisition**：已知 embodiment/task 上学动作；
- **generalization**：新 object、instruction、scene 或 embodiment；
- **adaptation**：少量 demonstration/experience 后更新；
- **orchestration**：高层规划何时调用哪个低层 policy。

把 imitation dataset 增大可能改善前两项，却不自动解决在线 recovery；RL 改善 dexterity 也可能破坏 generality；
LLM planner 生成语言计划仍需 ground 到 contact-rich action。[I]

### 9.2 RT 系列到 Gemini Robotics / ER / on-device [D/C/U]

[RT-2](https://robotics-transformer2.github.io/assets/rt2.pdf)（`gdm-url-2a645302dbbd`）把 robot actions 表成
token，与 web-scale vision-language data co-fine-tune；报告 PDF p.6 明确比较 co-fine-tuning 与仅 robot data，
Table 1（p.9）给 simulated Language-Table 结果。它的桥接是“web knowledge → action token”，不证明可靠掌握
所有物理因果。[D/U]

[RoboCat](https://deepmind.google/blog/robocat-a-self-improving-robotic-agent/)（`gdm-arxiv-2306.11706`）是
multi-task/multi-embodiment visual goal-conditioned agent；Figure 1（PDF p.3）给 demo→fine-tune→data collection→
retrain 的 self-improvement loop，Table 1（p.10）逐任务报告 final performance。它说明新 task data 可回流，
不等于无监督开放世界自我提升。[D/U]

[Gemini Robotics](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/)
（`gdm-arxiv-2503.20020`）把 Gemini 2.0 分成 Gemini Robotics-ER（embodied-reasoning VLM）与 Gemini Robotics
（VLA）；Figure 1（PDF p.2）清楚画出 base→robotics-specific training→adaptation/specialization，Table 1（p.5）
比较 embodied reasoning benchmarks。[D] 这是真正的 core/direct 桥，强于仅凭作者 affiliation 推断。

[Gemini Robotics On-Device card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-Robotics-On-Device-Model-Card.pdf?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=)
（`gdm-url-4a5f0122d44e`）把 local latency/availability 与部署限制单列；[Gemini Robotics 1.5](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1-5-Tech-Report.pdf#page=30)
（`gdm-url-40b6bdf31038`）又把 VLA 与 ER VLM 配对，Figure 1（PDF p.2）展示 reasoning/planning/action/motion
transfer，Table 1（p.19）把 long-horizon failures 分成 planning、execution、perception 等类别；p.22 说明另用 RL
增强 dexterity 而尽量保持 generality。[D]

2026 的 Robotics-ER 1.6 card (`gdm-web-12e01e8a6622`) 是 HTML/core 记录：可确认官方版本和卡片边界，
但没有完整 PDF recipe 时不从标题扩写新增算法。[D-meta/U]

### 9.3 physical-world evaluation 的未知项 [U]

现实评测至少需要以下分母，而多数发布只给其中一部分：[U]

| 维度 | 最小披露 |
|---|---|
| task distribution | train/test task、object、scene 与 instruction overlap |
| embodiment | robot family、sensor、action rate、controller 与 calibration |
| trial protocol | attempts、reset、human intervention、timeout、random seed |
| failure taxonomy | perception、planning、grounding、execution、safety stop、hardware fault |
| latency | cloud/on-device path、network jitter、closed-loop frequency |
| safety | force/velocity envelope、collision monitor、semantic policy、human proximity |
| adaptation | demonstrations/rollouts 数、是否 weight update、是否污染 test |

因此，“成功率 80%”若没有 attempts 与 intervention 不能与另一系统比较；“zero-shot”若允许先验 robot data、
prompt tuning 或 scene calibration，也不是“从未见过机器人”。Gemini Robotics 1.5 的 Table 1 分 failure type
是向可诊断评价前进，但仍不足以证明任意开放世界安全。[D/U]

## 10. Safety、alignment 与 governance：模型卡、评测和政策框架三种证据

### 10.1 dangerous capability、red teaming 与 scheming audit [D]

[Evaluating Frontier Models for Dangerous Capabilities](https://deepmind.google/research/publications/78150/)
（`gdm-arxiv-2403.13793`）把 model + scaffolding 当作 agent（PDF Figure 1，p.4），在 persuasion/deception、cyber、
self-proliferation 与 self-reasoning 上设计任务；Table 1（p.2）列 task families。作者当时未发现被测模型具有强
dangerous capabilities，同时明确列出需增强的 autonomy、long-term deception、self-exfiltration 与 AI R&D eval。
[D] “未在该套件发现”不是“不存在”。

[Holistic Safety and Responsibility Evaluations](https://deepmind.google/research/publications/78149/)
（`gdm-arxiv-2404.14068`）把 foresight、risk prioritization、multimodal evaluation、external participation 与
launch process放到同一生命周期；PDF pp.4–6 以 Gemini launch 为例讨论 impact assessment 与 multimodal
evaluation。[D] 它补足了 capability eval 不覆盖的社会/产品风险，但也不是独立认证。[U]

2026 direct 研究进一步测更接近 agent deployment 的行为：

- [Scheming honeypots](https://deepmind.google/research/publications/253391/)（`gdm-arxiv-2605.29729`）Figure 1
  （PDF p.2）给 code-review/agentic-coding honeypot；报告称 current Gemini 在真实 internal setting 未出现
  unprompted scheming，而 prompted variants 用于 sensitivity validation。[D] 结论受 scenario 与 elicitation 覆盖限制。[U]
- [Gram](https://deepmind.google/research/publications/252981/)（`gdm-arxiv-2605.30322`）用 investigator agent
  自动生成/迭代 alignment audits；Figure 1（PDF p.2）给 17 seed scenarios 的流程，Table 1（p.11）汇总关键
  experiments。自动 auditor 扩大搜索，却可能共享模型盲点，因此不是独立完备性证明。[D/U]

### 10.2 Frontier Safety Framework：门槛框架不是零风险证明 [D/U]

[FSF 1.0](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/introducing-the-frontier-safety-framework/fsf-technical-report.pdf)
（`gdm-url-25f473162418`）建立 Critical Capability Levels（CCLs）与评估/mitigation 触发逻辑；
[FSF 2.0](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/updating-the-frontier-safety-framework/Frontier%20Safety%20Framework%202.0.pdf)
（`gdm-url-f018b70353a1`）更新框架；[Gemini 3 Pro FSF report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf)
（`gdm-url-30f975dda414`）把框架应用到 bio/nuclear、cyber、ML R&D、harmful manipulation 与 misalignment。
后者 PDF p.2 定义 CCL 为“若无 mitigation 会带来 severe harm 风险”的 capability level，p.6 汇总未达到的
instrumental-reasoning thresholds，Figure 1（pp.7–8）展示 external biology benchmarks。[D]

正确逻辑是：

$$
\text{eval below threshold}\ne \text{proof of zero risk},\qquad
\text{threshold reached}\longrightarrow \text{specified mitigation/governance action}.
$$

第一式来自 coverage/elicitation/measurement uncertainty；第二式是框架承诺的触发关系，不证明 mitigation
一定有效。[I/U] 需要另外记录 evaluator access、attempt budget、scaffolding、version、external testers、
false-negative analysis 与 deployment controls。

### 10.3 watermarking、provenance、privacy 与 misuse [D/C/U]

[SynthID-Text](https://doi.org/10.1038/s41586-024-08025-4)
（`gdm-doi-10.1038-s41586-024-08025-4`）提出 tournament sampling watermark，并评估 quality/detection；
PDF p.10 以 fixed FPR=1% 报 TPR，Extended Data Table 1（p.14）给 human preference。它证明特定生成与检测
协议下的统计信号，不证明短文本、改写、翻译、混合来源或对抗攻击下永远可检测。[D/U]

provenance、防滥用与 privacy 是不同控制：watermark 表示“可能由某 detector-compatible generator 产生”；
content policy 约束生成；credential/access control 限制 action；privacy 处理数据与泄漏；incident response
处理部署后损害。把 watermark 当作 truth detector 或 copyright ownership proof 都越界。[I]

Gemma Scope (`gdm-url-4fce1bd2d654`) 与 Gemma Scope 2 (`gdm-url-76b573eca27e`) 发布 sparse autoencoder
研究制品，用于观察开放 Gemma 表示；这是 interpretability artifact [C]，不是 safety guarantee，也不能无证据
外推到闭源 Gemini 3.x 的内部 feature。[U]

### 10.4 模型卡 41/41 审计：23 个 Gemini core 与 18 个广义 GDM cards [D/C]

冻结 inventory 的 41 个 `model_card` 可完整分成：

- **23 个 Gemini core cards**：Gemini 1.0/1.5、2.0 Flash/Flash-Lite、2.5 Pro/Deep Think/Flash/Flash-Lite/
  Computer Use、Robotics On-Device/1.5/ER 1.6、3 Pro/Flash/Pro Image、3.1 Pro/Flash-Lite/Image/Audio/
  Flash-Lite Image、3.5 Flash/Audio、Omni Flash。
- **18 个 affiliated cards**：Gemma 1/2/3/3n/4、CodeGemma、PaliGemma 1/2、RecurrentGemma、ShieldGemma 1/2、
  EmbeddingGemma、FunctionGemma、DiffusionGemma、Imagen 4、Veo 3/3.1 Lite、Lyria 3。

core 的 23 而不是 24，是因为第 24 条 core 是 `technical_report` 类型的 Gemini 3 Pro FSF report。卡片审计的
四个状态不可混写：

| 状态 | 数量/范围 | 可支持的结论 |
|---|---|---|
| PDF + local text | 1.0/1.5、2.0/2.5 多卡、3/3.1 多卡、Robotics 多卡等 | 可给页码、限制、安全表与数据类别 [D] |
| official HTML + archived hash | 2026 新 cards 与部分 media/Gemma cards | 可确认页面/版本/HTML 内容与 PDF link [C] |
| official index metadata | 卡片索引发现 | 只确认 card identity/date/path [D-meta] |
| no equivalent training report | 绝大多数新卡 | 不能重建参数、token、optimizer、data mixture、RL loop [U] |

任何模型卡的 benchmark/safety 数字都要绑定 card revision。卡片会更新；“last viewed”不是模型发布日期，
HTML 页面与 PDF 可能异步。全量 ID/URL 见第 17 节，coverage ledger 保留本地 artifact 状态。

## 11. 社会影响：教育、劳动、文化、人机协作与全球公平

### 11.1 direct 现场与交互研究 [D]

[Teaching with Gemini](https://storage.googleapis.com/deepmind-media/LearnLM/learnLM_sierraleone_may26.pdf)
（`gdm-url-b28a37bc78c1`）是清单里最强的 direct field evidence：预注册、两臂 randomized controlled trial，
覆盖 Sierra Leone Port Loko 的 12 所政府支持初中；报告 PDF p.1 给 intent-to-treat effect +0.258 SD
（$p=0.029$），69.0% 学生至少使用一次 Guided Learning；Figure 1（p.3）显示 uptake/hours。[D]

这项结果的严格解释是“在指定学校、学科、实施伙伴、教师支持、时间段与 outcome 下的平均因果效应”，不是
“Gemini 在所有国家/年级提高 0.258 SD”。必须继续报告 attrition、cluster assignment、teacher adaptation、
usage heterogeneity、cost、language、baseline resources 与 long-run retention。[U]

[To Mask or to Mirror](https://deepmind.google/research/publications/180362/)（`gdm-web-e36538e67c47`）研究
collective reasoning 中 human-AI alignment；[Cultural Evolution of Cooperation among LLM Agents](https://doi.org/10.65109/jnmb7739)
（`gdm-doi-10.65109-jnmb7739`）研究多 agent cooperation 文化演化。它们测试 interaction dynamics，不直接测
社会部署净福利。agent society 的合作率依赖 population、memory、identity、payoff、communication 与 judge。[D/U]

### 11.2 affiliated 长尾说明了研究宽度，不说明产品净影响 [D/I/U]

376 条带 `society_governance` 主题的广义记录覆盖 fairness、health equity、democracy、labor/economics、AI
governance、social dilemmas、machine culture、human-AI relationships、clinical collaboration 与 global participation。
代表节点包括：

- [AI Governance](https://doi.org/10.1093/oxfordhb/9780197579329.013.2)
  （`gdm-doi-10.1093-oxfordhb-9780197579329.013.2`）；
- [impact of advanced AI systems on democracy](https://doi.org/10.1038/s41562-025-02309-z)
  （`gdm-doi-10.1038-s41562-025-02309-z`）；
- [deep mechanism design](https://doi.org/10.1073/pnas.2319949121)
  （`gdm-doi-10.1073-pnas.2319949121`）；
- [Concordia generative agent-based modeling](https://deepmind.google/research/publications/64717/)
  （`gdm-arxiv-2312.03664`）；
- [pluralistic geo-cultural safety alignment](https://deepmind.google/research/publications/225819/)
  （`gdm-arxiv-2606.00369`）。

这批 affiliated 工作说明研究问题很宽，不说明产品策略已经采纳其结论。[D/U] 社会净影响至少是

$$
\Delta W=\Delta Q-\Delta C-\Delta H+\Delta A,
$$

其中 $Q$ 是任务质量/收益、$C$ 使用与机会成本、$H$ harms/externalities、$A$ access/equity 变化。该式是
核算框架 [I]，不是把异质社会价值压成一个已知标量；各项权重本身有规范性争议。[U]

## 12. 公开制品与可复现性：开放不是布尔值

截至冻结日，1,985 条中 1,625 条有公开 PDF 路由，1,543 份 PDF 已通过 magic/pdfinfo 并抽取全文；360 条没有
公开 PDF 路由，82 条有路由但未形成可读本地全文。**纳入完整性**与**artifact 可得性**是两个分母：无 PDF
不删除研究记录，有 PDF 也不自动成为可复现实验。[D/C]

| 开放层 | 代表例 | 本章标签 | 仍缺什么 |
|---|---|---|---|
| identity/metadata | 1,985 JSONL records、official sitemap/publications/model-card indexes | [C] | 不等于正文可读 |
| report/PDF | 1,543 validated local PDFs/TXT，manifest 含 hash/bytes/resolved URL | [C] | 不等于 code/data |
| code/data | CodeContests、AlphaFold DB、Gemma/Gemma Scope、部分 benchmark/repository | [C] | 环境、依赖、训练资源、licenses 仍逐项核对 |
| weights | Gemma/RecurrentGemma/PaliGemma 等开放模型 | [C] | 不等于 Gemini core weights |
| claimed result | papers/cards 的 benchmark 与 deployment statement | [D] | 非本仓库独立重跑 |
| reproduced result | 固定代码/数据/seed/hardware 后的本地结果 | 本章 **0 项 [R]** | 需要专门实验协议 |

本地 PDF/TXT 是研究副本，`archive/` 被 Git 忽略；inventory、discovery snapshot、hash manifest 与生成报告才是
可重建入口。一次最低限度复现还需锁定 source revision、data split、preprocessing、checkpoint、prompt/tools、
seed、hardware/software、metric implementation、raw outputs 与 failure logs。没有这些，不使用 [R]。

## 13. 关键定量结论的一手定位表

下表把最容易被二手转述扭曲的结论绑定到冻结 inventory ID 与本地 PDF 定位；没有本地全文的记录只给入口，
不伪造页码。

| 结论 | 一手记录与定位 | 标签/限制 |
|---|---|---|
| Gemini 1.0 是 Transformer decoder、32K、举 MQA 为高效 attention | `gdm-url-0453ae158a21`, PDF p.3, Table 1 | [D] 未披露 size/层数 |
| 1.0 用 TPUv4/v5e；Ultra 用 TPUv4 fleet | 同 ID, pp.4–5 | [D] 总 fleet/TPU-hours [U] |
| 1.0 mixture 后段提高 domain-relevant data；token budget follow compute-optimal approach | 同 ID, p.5 | [D] 具体 mixture/token 数 [U] |
| 1.0 post-training 有 SFT/RM/RLHF | 同 ID, pp.20–21 | [D] optimizer/rollouts [U] |
| 1.0 Figure 1/5 与 Table 13 展示跨模态交互 | 同 ID, pp.3/15/19 | [D] 定性示例不等于成功率 |
| 1.5 needle recall >99.7% 到 1M，多模态极限实验到约 10M token | `gdm-url-49ab4305f7af`, pp.1–2, Figure 1 | [D] retrieval≠reasoning |
| 1.5 职业任务报告 26–75% time saving | 同 ID, p.37, Figure 19 | [D] 特定设计，不是宏观生产率 |
| 2.5 Pro 写明 human/critic feedback RL | `gdm-url-082786c15ca1`, p.10 | [D] reward/optimizer [U] |
| 2.5 Pro FSF summary | 同 ID, p.12, Table 1 | [D] 套件/attempt/version 限制 |
| 2.5 Deep Think cyber attempt budgets/套件变更 | `gdm-url-f9179a0c9b65`, p.14, Figure 3 | [D] 跨版不可直接比 |
| 3 Pro card 当时未达到列明 CCL | `gdm-url-d94bc4796e34`, p.9 | [D] 不等于零风险 |
| Gemini 3 Pro FSF 的 CCL 定义与 domain | `gdm-url-30f975dda414`, pp.2–3 | [D] 框架承诺不是独立认证 |
| MuZero 的 latent model/search/training loop | `gdm-arxiv-1911.08265`, p.3, Figure 1 | [D] 不等于 Gemini recipe |
| MuZero 大/小 Atari regimes | 同 ID, p.6, Table 1 | [D] 训练预算不可混 |
| AlphaCode 数据/竞赛/过滤 | `gdm-arxiv-2203.07814`, pp.3/8/13, Figure 1/Table 1 | [D] search，不是 proposal policy RL |
| AlphaTensor 网络与分解结果 | `gdm-doi-10.1038-s41586-022-05172-4`, pp.3/16, Figure 2/Extended Table 1 | [D] 定义空间内可验证 |
| AlphaDev 指令空间排序结果 | `gdm-doi-10.1038-s41586-023-06004-9`, p.3, Table 1 | [D/C] 硬件/compiler 依赖 |
| FunSearch evolutionary program loop | `gdm-doi-10.1038-s41586-023-06924-6`, p.14, Figure 1 | [D] evolutionary selection |
| AlphaProof Lean/RL/search/TTRL | `gdm-doi-10.1038-s41586-025-09833-y`, pp.4/8, Figure 1/Table 1 | [D/C] formalization 仍需审计 |
| AlphaEvolve 用 Gemini 2.0 Flash/Pro ensemble | `gdm-url-119a83ec2a50`, pp.3/7, Figure 1 | [D] evaluator coverage [U] |
| SIMA 2 是 Gemini-based virtual-world agent/self-improvement loop | `gdm-url-d41003b9db7f`, pp.1–2, Figure 1 | [D] 不外推无限自治 |
| AlphaFold 2 CASP14 median backbone 0.96 Å r.m.s.d.95 | `gdm-doi-10.1038-s41586-021-03819-2`, p.2 | [D] 任务分布限定 |
| AlphaFold 3 diffusion module 与 complex scope | `gdm-doi-10.1038-s41586-024-07487-w`, pp.2–3 | [D] 不等于 biological function |
| GenCast conditional diffusion ensemble 与 runtime | `gdm-arxiv-2312.15796`, pp.2–3, Figure 1 | [D] hardware/ensemble size 绑定 |
| Med-Gemini 数据概览 | `gdm-arxiv-2405.03162`, pp.3/8, Figure 1/Table 1 | [D] 临床部署 [U] |
| RT-2 web+robot co-fine-tuning | `gdm-url-2a645302dbbd`, pp.6/9, Table 1 | [D] physical generality 有限 |
| RoboCat self-improvement loop/逐任务结果 | `gdm-arxiv-2306.11706`, pp.3/10, Figure 1/Table 1 | [D] 依赖 demonstrations/data loop |
| Gemini Robotics 的 ER/VLA 分叉 | `gdm-arxiv-2503.20020`, pp.2/5, Figure 1/Table 1 | [D] 真实 core/direct bridge |
| Robotics 1.5 的 planning/execution failure taxonomy | `gdm-url-40b6bdf31038`, p.19, Table 1 | [D] 开放世界安全 [U] |
| dangerous-capability eval 测 model+scaffolding | `gdm-arxiv-2403.13793`, pp.2/4, Table 1/Figure 1 | [D] coverage 不完备 |
| SynthID 固定 FPR=1% 的 detection 与人类偏好 | `gdm-doi-10.1038-s41586-024-08025-4`, pp.10/14, Extended Table 1 | [D] 改写鲁棒性非无限 |
| scheming honeypot 设计/结论边界 | `gdm-arxiv-2605.29729`, p.2, Figure 1 | [D] 未见≠不存在 |
| Gram automated auditing | `gdm-arxiv-2605.30322`, pp.2/11, Figure 1/Table 1 | [D] auditor 共享盲点 [U] |
| Sierra Leone 教育 RCT | `gdm-url-b28a37bc78c1`, pp.1/3, Figure 1 | [D] 场景外推 [U] |

页码均按 PDF 文件页序，不是印刷卷页；HTML cards 没有稳定页码时引用 URL/archived hash，而不是伪造页号。

## 14. 跨谱系综合：能迁移的结论与不能越界的推断

### 14.1 十个可证伪综合结论

1. **[I] Gemini 是多条研究流汇流，不是 AlphaGo 的线性后继。** 证据：native multimodal report、RL/search
   长线、Gemini-based SIMA/AlphaEvolve、Robotics bridge、science 分支并存。反证：官方 genealogy 显示没有
   任何方法/人员/系统接口连续性。
2. **[D/I] 模型能力与系统能力不可分，但必须可拆。** Long context 依赖 training+serving；agent 依赖
   harness/tools/environment；robotics 依赖 controller/sensor。反证：在固定系统下单 checkpoint 能解释全部增益。
3. **[D] 公开透明度从 1.0/1.5 的技术报告转向 2.x/3.x 的版本化卡片。** 反证：新报告公开等价的 architecture、
   data、optimizer、RL 与 ablation 细节。
4. **[D/I] “thinking”不是可识别算法名。** 2.5 卡片确认 reasoning 与 feedback RL，却不公开 optimizer/reward。
   反证：正式 recipe 证明 thinking 对应唯一、稳定、可复现机制。
5. **[D/I] Alpha 系列的稳态是 proposal + evaluator + search/learning，不是统一 network。** 反证：组件消融
   显示 evaluator/search 不影响发现。
6. **[D/I] long-context reachability、integration 与 action 是三种能力。** 1.5 needle、Michelangelo-style structure、
   agent/robot trajectories分别测不同层。反证：同一检索指标在独立研究中充分预测后两者。
7. **[D/I] science impact 必须经过 prediction→validation→workflow 梯级。** AlphaFold/A-Lab/教育 RCT处于
   不同等级。反证：离线 benchmark 在多领域稳定等价于现实因果效应。
8. **[D/I] safety 是 model behavior、scaffolding、access、monitoring、provenance 与 governance 的控制栈。**
   反证：独立研究显示单一 model score 足以预测部署伤害。
9. **[D/U] 1,943 条 affiliated 是组织知识环境，不是 Gemini BOM。** 反证：官方逐项材料清单把这些工作连接到
   Gemini training/serving。
10. **[U] 没有公开证据支持最新 Gemini 端到端复现。** 反证：发布完整数据 lineage、代码、checkpoints、
    optimizer、reward/trajectory、environment、hardware 与失败日志，并由独立团队复跑。

### 14.2 横跨主题的资源分配图

更深的共同结构是“在哪里花额外计算和验证预算”：

| 轴 | 代表系统 | 额外预算花在哪里 | 主要风险 |
|---|---|---|---|
| parameter/data scaling | Gemini/Gemma | pretraining tokens/weights | opaque mixture、memorization |
| context scaling | Gemini 1.5 | attention/state/serving | retrieval≠reasoning、latency |
| search scaling | AlphaGo/MuZero/AlphaCode/AlphaProof | branches/candidates | evaluator exploitation、cost |
| interaction scaling | SIMA/computer use/robotics | environment steps | non-stationarity、side effects |
| experiment scaling | AlphaFold/weather/A-Lab | candidate/physical validation | measurement bias、negative results |
| safety scaling | cards/FSF/Gram | elicitation/audit/monitor | false negatives、shared blind spots |

**[I]** 所谓 frontier advance 经常是把预算从单次 forward pass 外移到 search、tools、environment 或 evaluator。
因此只报 parameter count 会越来越失真；但这也不意味着 runtime scaling 永远优于训练 scaling，二者受 latency、
energy、privacy 与 error correlation 约束。

### 14.3 不能越界迁移的四类结论

- 游戏中 exact rules/verifier 的成功，不能直接迁移到开放网页与社会环境；
- scientific prediction 的高 accuracy，不能直接迁移为 intervention efficacy；
- affiliated 开放模型的可解释性结果，不能直接迁移到闭源 Gemini checkpoint；
- model card “threshold not reached”，不能直接迁移为 deployment “safe”。

这些边界不是修辞保守，而是变量不同：environment closure、ground truth、checkpoint access、threat model 与
distribution shift 都不同。[I]

## 15. 研究议程：从现有证据还能严格问什么

下列问题可直接转成可证伪实验，而不是继续做型号宣传：

1. **长上下文机制消融**：固定 token budget，分别改变 attention/state、position、retrieval、training curriculum
   与 serving；同时测 needle、latent structure、multi-document contradiction、long-horizon tool use 和 latency。
2. **thinking compute curve**：固定 checkpoint/prompts/tools，公开 token/branch/time budget 与 calibration，画
   accuracy、cost、error correlation、abstention 随 compute 的 Pareto frontier。
3. **critic-feedback 审计**：公开 critic families、inter-rater agreement、reward calibration、adversarial prompts、
   policy exploitation 与 offline→online shift；证明 human/critic reward 何时互补。
4. **agent harness factorial**：checkpoint × system prompt × tool schema × memory × budget × confirmation policy，
   把 model gain 与 harness gain 分离。
5. **search/verifier stress**：给 AlphaEvolve/FunSearch/AlphaProof 类系统故意不完备或可利用 evaluator，测 proxy
   hacking、diversity collapse、false proof/formalization error 与 human review load。
6. **robotics intervention accounting**：逐 trial 保存 observation/action/latency/human override/force/safety stop，
   按 failure taxonomy 报 confidence interval，而不是只报 success mean。
7. **science external validation**：time-split、lab/site-split 与 prospective replication；登记 negative candidates，
   计算每个 validated discovery 的实验/compute 成本。
8. **safety coverage estimate**：用独立 red teams、不同 base auditor、adaptive elicitation 与 capture-recapture 思路
   估算未发现失败，而不是把 zero observed 当作 zero rate。
9. **社会效应多场景复验**：教育 RCT 跨语言/地区/资源水平复验，记录 teacher behavior、usage、cost、retention、
   subgroup 与 spillover。
10. **谱系审计**：发布 component/data/eval version graph，使“继承 Gemini 2.0”“based on Gemini”可追溯到
    checkpoint、adapter、distillation、tooling 或仅 branding。

优先级应按“结论被广泛使用 × 当前证据弱 × 错误成本高”排序：医疗/机器人/agent safety/教育外推优先于再做
一个平均 benchmark 表。[I]

## 16. 使用本章时应避免的具体误述

| 不应写 | 严格写法 |
|---|---|
| “1,985 篇 Gemini 论文” | 24 core、18 direct、1,943 affiliated；后者只是组织归属 |
| “Gemini 用了所有 DeepMind 技术” | 只有 core/direct 桥接能支持组件关系；其余是方法环境 |
| “Gemini 1.5 能理解 10M tokens” | 报告在特定 retrieval/NLL/任务上测试到该规模；理解需另测 |
| “Gemini 2.5 用 PPO/GRPO” | 卡片只披露 human/critic feedback RL，optimizer [U] |
| “Deep Think 就是更长 CoT” | 它是版本/产品能力描述；内部训练与 runtime budget 未完整披露 |
| “AlphaCode/AlphaEvolve/FunSearch 是 RL” | 分别说明 sampling/filtering、evolutionary coding、LLM-guided program search；无 policy update 证据就不写 RL |
| “AlphaProof 证明自然语言题绝对正确” | Lean kernel 验证 formal statement 的 proof；formalization 仍需审计 |
| “AlphaFold 已解决蛋白质/药物发现” | 报固定任务、版本、confidence、实验 validation 与未知失败面 |
| “GraphCast/GenCast 取代数值天气” | 说明 deterministic/ensemble、输入、lead time、metric、operational baseline 与极端事件覆盖 |
| “Gemini Robotics 是聊天模型加机械臂” | 区分 ER VLM、VLA、robot-specific training、controller 与 specialization |
| “SIMA 2 能无限自我提升” | 报任务生成、reward、虚拟环境、更新 gate 与测试边界 |
| “FSF 说模型安全” | 报指定版本在指定 elicitation/eval 下是否达到 CCL，以及 mitigation 触发逻辑 |
| “水印能识别所有 AI 文本” | 报 generator/detector、length、FPR/TPR、transformation 与 adversary |
| “Sierra Leone 证明 AI 普遍提升教育” | 报 12 校 RCT、ITT +0.258 SD、使用率、场景与外推未知 |
| “下载了 PDF 就复现了论文” | 下载/抽取是 [C]；只有固定协议独立重跑结果才是 [R] |
| “模型卡是完整技术报告” | 卡片主要覆盖用途、限制、安全和选择性模型/数据信息；缺失项写 [U] |

## 17. 全量审计附录：1,985/1,985 inventory IDs 与 primary URLs

下面的链接文字就是 inventory ID，目标就是对应 `primary_url`。分组只为压缩版面，不改变 tier；同一 URL
若对应两个不同记录，会保留两次。自动检查必须同时满足：清单 ID 集合与附录 ID 集合完全相等；每个 ID 的
链接目标与 JSONL 完全相等；无缺项、无额外项、无错误映射。

### 17.1 2026（118 条）

- [gdm-doi-10.1038-s41586-026-10781-4](<https://doi.org/10.1038/s41586-026-10781-4>) - 2026-07-15 · `affiliated` · An encyclopedia of human enhancer–gene regulatory interactions
- [gdm-arxiv-2506.21511](<https://doi.org/10.1080/01621459.2026.2702653>) - 2026-07-15 · `affiliated` · Gaussian Invariant Markov Chain Monte Carlo
- [gdm-arxiv-2607.09197](<https://arxiv.org/abs/2607.09197>) - 2026-07-10 · `affiliated` · When is Routing Meaningful? Diversity and Robustness in Language Model Societies
- [gdm-arxiv-2606.00369](<https://deepmind.google/research/publications/225819/>) - 2026-07-10 · `affiliated` · Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment
- [gdm-arxiv-2511.08493](<https://doi.org/10.1038/s41586-026-10759-2>) - 2026-07-08 · `affiliated` · Reinforcement learning control of quantum error correction
- [gdm-arxiv-2607.03906](<https://deepmind.google/research/publications/260960/>) - 2026-07-06 · `affiliated` · The Case for Globally Beneficial Technology
- [gdm-web-a6cd2bca27cb](<https://deepmind.google/research/publications/203490/>) - 2026-07-02 · `affiliated` · Towards Structural Understanding of LLM Overthinking
- [gdm-doi-10.1371-journal.pbio.3003839](<https://doi.org/10.1371/journal.pbio.3003839>) - 2026-07-01 · `affiliated` · Towards globally equitable bioinformatics adoption
- [gdm-web-125c1d0964e1](<https://deepmind.google/models/model-cards/gemini-3-1-flash-lite-image/>) - 2026-06-30 · `core` · Gemini 3.1 Flash-Lite Image Model Card
- [gdm-doi-10.1145-3805689.3812253](<https://deepmind.google/research/publications/224297/>) - 2026-06-26 · `affiliated` · Real-Time Group Dynamics with LLM Facilitation: Evidence from a Charity Allocation Task
- [gdm-doi-10.1145-3805689.3812210](<https://deepmind.google/research/publications/149262/>) - 2026-06-26 · `affiliated` · Bridging the Scale Gap: Augmenting Human Red-Teaming to Uncover Latent Risks in T2I Models
- [gdm-doi-10.1145-3805689.3812308](<https://doi.org/10.1145/3805689.3812308>) - 2026-06-25 · `affiliated` · Human-AI Complementarity: A Goal for Amplified Oversight
- [gdm-doi-10.1145-3805689.3812296](<https://deepmind.google/research/publications/224397/>) - 2026-06-25 · `affiliated` · Going PLACES: Participatory Localized Red Teaming forText-to-Image Safety in the Global South
- [gdm-doi-10.7717-peerj-cs.3906](<https://doi.org/10.7717/peerj-cs.3906>) - 2026-06-22 · `affiliated` · Modeling polarization in public opinion through LLM-synthesized arguments and stance trees
- [gdm-doi-10.21203-rs.3.rs-10063234-v1](<https://doi.org/10.21203/rs.3.rs-10063234/v1>) - 2026-06-22 · `affiliated` · An AI Co-Data-Scientist for Prioritizing Candidate Biomarkers from Wearable Sensor Data
- [gdm-web-7168db335b3e](<https://deepmind.google/research/publications/248131/>) - 2026-06-15 · `affiliated` · Artificial Minds, Human Disagreement: The Politics of AI Consciousness
- [gdm-doi-10.1007-s11023-026-09786-9](<https://doi.org/10.1007/s11023-026-09786-9>) - 2026-06-15 · `affiliated` · Agents, Alignment, and the Many Faces of Autonomy
- [gdm-arxiv-2606.12683](<https://deepmind.google/research/publications/239142/>) - 2026-06-12 · `affiliated` · From AGI to ASI
- [gdm-web-75b7cd55cf18](<https://ai.google.dev/gemma/docs/diffusiongemma/model_card?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content>) - 2026-06-10 · `affiliated` · DiffusionGemma Model Card
- [gdm-web-7420e4780dde](<https://deepmind.google/models/model-cards/gemini-3-5-audio/>) - 2026-06-09 · `core` · Gemini 3.5 Audio (Live Translate) Model Card
- [gdm-doi-10.1038-d41586-026-01820-1](<https://doi.org/10.1038/d41586-026-01820-1>) - 2026-06-08 · `affiliated` · How AI is reshaping discovery in maths and physics
- [gdm-doi-10.48550-arxiv.2606.03237](<https://deepmind.google/research/publications/231466/>) - 2026-06-04 · `affiliated` · Solipsistic superintelligence is unlikely to be cooperative
- [gdm-doi-10.21203-rs.3.rs-9684240-v1](<https://doi.org/10.21203/rs.3.rs-9684240/v1>) - 2026-06-03 · `affiliated` · SymptomAI: Toward a Conversational AI Agent for Everyday Symptom Assessment
- [gdm-arxiv-2605.30322](<https://deepmind.google/research/publications/252981/>) - 2026-05-28 · `direct` · Gram: Assessing sabotage propensities via automated alignment auditing
- [gdm-arxiv-2605.29729](<https://deepmind.google/research/publications/253391/>) - 2026-05-28 · `direct` · Realistic honeypot evaluations for scheming propensity
- [gdm-doi-10.3390-e28060596](<https://doi.org/10.3390/e28060596>) - 2026-05-27 · `affiliated` · Algorithmic Compression via Pretrained Neural Networks
- [gdm-doi-10.1609-icwsm.v20i1.42689](<https://doi.org/10.1609/icwsm.v20i1.42689>) - 2026-05-25 · `affiliated` · Real-World Challenges in Fake News Detection: Dealing with Posts by Cold Users
- [gdm-doi-10.65109-pubn4692](<https://doi.org/10.65109/pubn4692>) - 2026-05-24 · `affiliated` · Active Evaluation of General Agents: Problem Definition and Comparison of Baseline Algorithms
- [gdm-doi-10.21203-rs.3.rs-9784163-v1](<https://doi.org/10.21203/rs.3.rs-9784163/v1>) - 2026-05-22 · `affiliated` · On the Limits of End-to-End Foundation Models: Coordination as a Missing Primitive for Artificial General Intelligence
- [gdm-doi-10.64898-2026.05.18.725921](<https://doi.org/10.64898/2026.05.18.725921>) - 2026-05-21 · `affiliated` · AI-Discovered Cognitive Models Reveal Novel Insights into Human and Animal Learning
- [gdm-web-5cbf6a0a48be](<https://deepmind.google/models/model-cards/gemini-3-5-flash/>) - 2026-05-19 · `core` · Gemini 3.5 Flash Model Card
- [gdm-web-3e34f88c613d](<https://deepmind.google/models/model-cards/gemini-omni-flash/>) - 2026-05-19 · `core` · Gemini Omni Flash Model Card
- [gdm-doi-10.1609-aaaiss.v8i1.42589](<https://doi.org/10.1609/aaaiss.v8i1.42589>) - 2026-05-18 · `affiliated` · Echoes of Citations: Automated Extraction of Claims from Full Scientific Papers
- [gdm-url-b28a37bc78c1](<https://storage.googleapis.com/deepmind-media/LearnLM/learnLM_sierraleone_may26.pdf>) - 2026-05-15 · `direct` · Teaching with Gemini: Measuring the Impact of Guided Learning on Student Mathematics Progress in Sierra Leone
- [gdm-doi-10.1007-s12532-026-00309-2](<https://doi.org/10.1007/s12532-026-00309-2>) - 2026-05-15 · `affiliated` · PDLP: a practical first-order method for large-scale linear programming
- [gdm-doi-10.1002-wps.70067](<https://doi.org/10.1002/wps.70067>) - 2026-05-15 · `affiliated` · A framework for clinical validation of generative artificial intelligence therapeutics
- [gdm-doi-10.3390-ai7050168](<https://doi.org/10.3390/ai7050168>) - 2026-05-13 · `affiliated` · What You Read Is What You Classify: Highlighting Attributions to Text and Text-like Inputs
- [gdm-doi-10.21203-rs.3.rs-9683321-v1](<https://doi.org/10.21203/rs.3.rs-9683321/v1>) - 2026-05-13 · `affiliated` · Digital Volume Correlation Challenge 2.0: A Comprehensive Dataset for Digital Volume Correlation Benchmarking
- [gdm-doi-10.1088-1741-4326-ae6d12](<https://doi.org/10.1088/1741-4326/ae6d12>) - 2026-05-13 · `affiliated` · Progress and innovations in the TCV tokamak research programme
- [gdm-arxiv-2605.03767](<https://deepmind.google/research/publications/239849/>) - 2026-05-06 · `affiliated` · Did US Worker Retraining Reduce Participant Automation Exposure?
- [gdm-arxiv-2605.02462](<https://arxiv.org/abs/2605.02462>) - 2026-05-04 · `affiliated` · Black-box optimization of noisy functions with unknown smoothness
- [gdm-arxiv-2605.02458](<https://arxiv.org/abs/2605.02458>) - 2026-05-04 · `affiliated` · Active multiple matrix completion with adaptive confidence sets
- [gdm-arxiv-2505.04653](<https://doi.org/10.1038/s41591-026-04371-0>) - 2026-05-01 · `affiliated` · Advancing conversational diagnostic AI with multimodal reasoning
- [gdm-arxiv-2604.23099](<https://deepmind.google/research/publications/238239/>) - 2026-04-25 · `affiliated` · ProEval: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation
- [gdm-web-c96e5f5f1f8e](<https://deepmind.google/research/publications/193694/>) - 2026-04-23 · `affiliated` · Dynamic Reflections: Probing Video Representations with Text Alignment
- [gdm-arxiv-2604.21432](<https://arxiv.org/abs/2604.21432>) - 2026-04-23 · `affiliated` · A single algorithm for both restless and rested rotting bandits
- [gdm-arxiv-2604.21428](<https://deepmind.google/blog/decoupled-diloco/>) - 2026-04-23 · `affiliated` · Decoupled DiLoCo for Resilient Distributed Pre-training
- [gdm-doi-10.1038-s42256-026-01217-9](<https://doi.org/10.1038/s42256-026-01217-9>) - 2026-04-22 · `affiliated` · Competing Biases underlie Overconfidence and Underconfidence in LLMs
- [gdm-arxiv-2604.20329](<https://deepmind.google/research/publications/240658/>) - 2026-04-22 · `direct` · Image Generators are Generalist Vision Learners
- [gdm-doi-10.1109-icassp55912.2026.11461962](<https://doi.org/10.1109/icassp55912.2026.11461962>) - 2026-04-21 · `affiliated` · CoVA: Text-Guided Composed Video Retrieval for Audio-Visual Content
- [gdm-arxiv-2604.19698](<https://arxiv.org/abs/2604.19698>) - 2026-04-21 · `affiliated` · On two ways to use determinantal point processes for Monte Carlo integration
- [gdm-arxiv-2604.19695](<https://arxiv.org/abs/2604.19695>) - 2026-04-21 · `affiliated` · Planning in entropy-regularized Markov decision processes and games
- [gdm-arxiv-2604.19672](<https://arxiv.org/abs/2604.19672>) - 2026-04-21 · `affiliated` · Budgeted Online Influence Maximization
- [gdm-web-12e01e8a6622](<https://deepmind.google/models/model-cards/gemini-robotics-er-1-6/>) - 2026-04-20 · `core` · Gemini Robotics-ER 1.6 Model Card
- [gdm-arxiv-2604.16111](<https://arxiv.org/abs/2604.16111>) - 2026-04-17 · `affiliated` · Sample Complexity Bounds for Stochastic Shortest Path with a Generative Model
- [gdm-web-50d0fe5038d4](<https://deepmind.google/models/model-cards/gemini-3-1-flash-audio/>) - 2026-04-15 · `core` · Gemini 3.1 Flash Audio (Flash Live, TTS) Model Card
- [gdm-doi-10.1145-3772363.3779005](<https://doi.org/10.1145/3772363.3779005>) - 2026-04-13 · `affiliated` · RAI@CHI: Responsible and Human Centred AI Across Borders
- [gdm-web-da4efcf3f092](<https://deepmind.google/models/model-cards/veo-3-1-lite/>) - 2026-04-08 · `affiliated` · Veo 3.1 Lite Model Card
- [gdm-web-827722ac3ea0](<https://hal.science/hal-05578059>) - 2026-04-02 · `affiliated` · Accelerating Nash learning from human feedback via Mirror Prox
- [gdm-web-6d1b200270eb](<https://ai.google.dev/gemma/docs/core/model_card_4?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content>) - 2026-04-02 · `affiliated` · Gemma 4 Model Card
- [gdm-arxiv-2412.09988](<https://doi.org/10.1177/26339137261459441>) - 2026-04-01 · `affiliated` · AI and the future of digital public squares
- [gdm-doi-10.64898-2026.03.27.714458](<https://doi.org/10.64898/2026.03.27.714458>) - 2026-03-29 · `affiliated` · AlphaFold Database expands to proteome-scale quaternary structures
- [gdm-doi-10.48550-arxiv.2603.26310](<https://doi.org/10.48550/arxiv.2603.26310>) - 2026-03-27 · `affiliated` · Applications of a novel model-based real-time observer for electron density profile control experiments in TCV
- [gdm-doi-10.1088-1361-6587-ae56b7](<https://doi.org/10.1088/1361-6587/ae56b7>) - 2026-03-24 · `affiliated` · FGE: a fast free-boundary Grad–Shafranov evolutive solver
- [gdm-doi-10.1038-s44168-026-00362-6](<https://doi.org/10.1038/s44168-026-00362-6>) - 2026-03-23 · `affiliated` · Generative AI for climate governance and acceptability-constrained policy design
- [gdm-doi-10.1145-3742413.3789078](<https://deepmind.google/research/publications/146950/>) - 2026-03-22 · `affiliated` · Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining
- [gdm-doi-10.1016-j.ecoinf.2026.103699](<https://doi.org/10.1016/j.ecoinf.2026.103699>) - 2026-03-17 · `direct` · Large language models possess some ecological knowledge, but how much?
- [gdm-web-64211f9bdd98](<https://deepmind.google/research/publications/231971/>) - 2026-03-10 · `affiliated` · The Abstraction Fallacy: Why AI Can Simulate But Not Instantiate Consciousness
- [gdm-doi-10.5281-zenodo.18923885](<https://doi.org/10.5281/zenodo.18923885>) - 2026-03-09 · `affiliated` · Tree crop mapping of South America reveals links to deforestation and conservation
- [gdm-doi-10.1017-9781009607551.015](<https://doi.org/10.1017/9781009607551.015>) - 2026-03-07 · `affiliated` · Outlook
- [gdm-doi-10.1017-9781009607551.014](<https://doi.org/10.1017/9781009607551.014>) - 2026-03-07 · `affiliated` · Submodular Minimisation
- [gdm-doi-10.1017-9781009607551.013](<https://doi.org/10.1017/9781009607551.013>) - 2026-03-07 · `affiliated` · Gaussian Optimistic Smoothing
- [gdm-doi-10.1017-9781009607551.012](<https://doi.org/10.1017/9781009607551.012>) - 2026-03-07 · `affiliated` · Online Newton Step for Adversarial Losses
- [gdm-doi-10.1017-9781009607551.011](<https://doi.org/10.1017/9781009607551.011>) - 2026-03-07 · `affiliated` · Online Newton Step
- [gdm-doi-10.1017-9781009607551.010](<https://doi.org/10.1017/9781009607551.010>) - 2026-03-07 · `affiliated` · Cutting Plane Methods
- [gdm-doi-10.1017-9781009607551.009](<https://doi.org/10.1017/9781009607551.009>) - 2026-03-07 · `affiliated` · Exponential Weights
- [gdm-doi-10.1017-9781009607551.008](<https://doi.org/10.1017/9781009607551.008>) - 2026-03-07 · `affiliated` · Linear and Quadratic Bandits
- [gdm-doi-10.1017-9781009607551.007](<https://doi.org/10.1017/9781009607551.007>) - 2026-03-07 · `affiliated` · Self-Concordant Regularisation
- [gdm-doi-10.1017-9781009607551.006](<https://doi.org/10.1017/9781009607551.006>) - 2026-03-07 · `affiliated` · Online Gradient Descent
- [gdm-doi-10.1017-9781009607551.005](<https://doi.org/10.1017/9781009607551.005>) - 2026-03-07 · `affiliated` · Bisection in One Dimension
- [gdm-doi-10.1017-9781009607551.004](<https://doi.org/10.1017/9781009607551.004>) - 2026-03-07 · `affiliated` · Mathematical Tools
- [gdm-doi-10.1017-9781009607551.003](<https://doi.org/10.1017/9781009607551.003>) - 2026-03-07 · `affiliated` · Overview of Methods and History
- [gdm-doi-10.1017-9781009607551.002](<https://doi.org/10.1017/9781009607551.002>) - 2026-03-07 · `affiliated` · Introduction and Problem Statement
- [gdm-doi-10.1017-9781009607551](<https://doi.org/10.1017/9781009607551>) - 2026-03-07 · `affiliated` · Bandit Convex Optimisation
- [gdm-doi-10.1109-wacv61042.2026.00666](<https://doi.org/10.1109/wacv61042.2026.00666>) - 2026-03-06 · `affiliated` · Test-Time Adaptation for Video Highlight Detection Using Meta-Auxiliary Learning and Cross-Modality Hallucinations
- [gdm-doi-10.1109-wacv61042.2026.00424](<https://doi.org/10.1109/wacv61042.2026.00424>) - 2026-03-06 · `affiliated` · X-JEPA: A Novel Joint Learning Cross-Modal Predictive Alignment Framework for Remote Sensing Image Retrieval
- [gdm-doi-10.1109-wacv61042.2026.00186](<https://doi.org/10.1109/wacv61042.2026.00186>) - 2026-03-06 · `affiliated` · Segmentation-Aware Latent Diffusion for Satellite Image Super-Resolution: Enabling Smallholder Farm Boundary Delineation
- [gdm-doi-10.1109-wacv61042.2026.00099](<https://doi.org/10.1109/wacv61042.2026.00099>) - 2026-03-06 · `affiliated` · Do generative video models understand physical principles?
- [gdm-doi-10.1109-wacv61042.2026.00081](<https://doi.org/10.1109/wacv61042.2026.00081>) - 2026-03-06 · `affiliated` · DICE: Discrete Inversion Enabling Controllable Editing for Masked Generative Models
- [gdm-doi-10.64898-2026.03.03.709199](<https://doi.org/10.64898/2026.03.03.709199>) - 2026-03-05 · `affiliated` · Functional reorganization of motor cortex connectivity during learning
- [gdm-url-c6d5b2d93460](<https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/>) - 2026-03-03 · `core` · Gemini 3.1 Flash-Lite Model Card
- [gdm-web-c731c29759e8](<https://deepmind.google/models/model-cards/gemini-3-1-flash-image/>) - 2026-02-26 · `core` · Gemini 3.1 Flash Image Model Card
- [gdm-url-d2aed2990254](<https://deepmind.google/models/model-cards/gemini-3-1-pro/>) - 2026-02-19 · `core` · Gemini 3.1 Pro Model Card
- [gdm-doi-10.1088-2632-2153-ae484b](<https://doi.org/10.1088/2632-2153/ae484b>) - 2026-02-19 · `affiliated` · Roadmap on fast machine learning for science
- [gdm-web-f9c42b10bb60](<https://deepmind.google/models/model-cards/lyria-3/>) - 2026-02-18 · `affiliated` · Lyria 3 Model Card
- [gdm-doi-10.1038-s41586-025-10021-1](<https://doi.org/10.1038/s41586-025-10021-1>) - 2026-02-18 · `affiliated` · A roadmap for evaluating moral competence in large language models
- [gdm-doi-10.3390-e28020226](<https://deepmind.google/research/publications/225507/>) - 2026-02-15 · `affiliated` · Simplicity and Complexity in Combinatorial Optimization
- [gdm-web-e9f511defe0f](<https://deepmind.google/research/publications/137741/>) - 2026-02-12 · `affiliated` · Decoding Safety Feedback from Diverse Raters: A Data-driven Lens on Responsiveness to Severity
- [gdm-doi-10.48550-arxiv.2602.06355](<https://doi.org/10.48550/arxiv.2602.06355>) - 2026-02-06 · `affiliated` · Di3PO - Diptych Diffusion DPO for Targeted Improvements in Image Generation
- [gdm-doi-10.1038-s41562-025-02324-0](<https://deepmind.google/research/publications/94006/>) - 2026-02-05 · `affiliated` · Hybrid neural–cognitive models reveal how memory shapes human reward learning
- [gdm-doi-10.1109-tpami.2026.3660066](<https://doi.org/10.1109/tpami.2026.3660066>) - 2026-02-02 · `affiliated` · DiffusionLight-Turbo: Accelerated Light Probes for Free via Single-Pass Chrome Ball Inpainting
- [gdm-doi-10.1016-j.tics.2026.01.002](<https://doi.org/10.1016/j.tics.2026.01.002>) - 2026-02-01 · `affiliated` · Imagining and building wise machines: the centrality of AI metacognition
- [gdm-doi-10.1038-s41586-025-10014-0](<https://doi.org/10.1038/s41586-025-10014-0>) - 2026-01-28 · `affiliated` · Advancing regulatory variant effect prediction with AlphaGenome
- [gdm-doi-10.21203-rs.3.rs-8414536-v1](<https://doi.org/10.21203/rs.3.rs-8414536/v1>) - 2026-01-23 · `affiliated` · Towards a Science of Scaling Agent Systems
- [gdm-doi-10.1056-aip2501266](<https://doi.org/10.1056/aip2501266>) - 2026-01-22 · `affiliated` · The Missing Dimension in Clinical AI: Making Hidden Values Visible
- [gdm-doi-10.1007-jhep01-2026-109](<https://doi.org/10.1007/jhep01(2026)109>) - 2026-01-16 · `affiliated` · The evaporation of charged black holes
- [gdm-web-78ae7fdb5a23](<https://ai.google.dev/gemma/docs/functiongemma/model_card>) - 2026-01-14 · `affiliated` · FunctionGemma Model Card
- [gdm-url-07dc6591b390](<https://storage.googleapis.com/deepmind-media/Model-Cards/Veo-3-Model-Card.pdf>) - 2026-01-13 · `affiliated` · Veo 3 Model Card
- [gdm-web-e538aaecf217](<https://deepmind.google/research/publications/122591/>) - 2026-01-09 · `affiliated` · TRecViT: A Recurrent Video Transformer
- [gdm-doi-10.64898-2026.01.07.698044](<https://doi.org/10.64898/2026.01.07.698044>) - 2026-01-08 · `affiliated` · Neural representations supporting generalization under continual learning
- [gdm-doi-10.3389-fnhum.2025.1633272](<https://doi.org/10.3389/fnhum.2025.1633272>) - 2026-01-02 · `affiliated` · LLMs achieve adult human performance on higher-order theory of mind tasks
- [gdm-doi-10.2139-ssrn.7118018](<https://doi.org/10.2139/ssrn.7118018>) - 2026-01-01 · `affiliated` · Extractable Memorization From First Principles
- [gdm-doi-10.2139-ssrn.7095164](<https://doi.org/10.2139/ssrn.7095164>) - 2026-01-01 · `affiliated` · Anthropomorphism and Trust in Human-Large Language Model interactions
- [gdm-doi-10.2139-ssrn.6372438](<https://doi.org/10.2139/ssrn.6372438>) - 2026-01-01 · `affiliated` · AI Agent Traps
- [gdm-doi-10.2139-ssrn.6169446](<https://doi.org/10.2139/ssrn.6169446>) - 2026-01-01 · `affiliated` · What kind of Generative world Model can Get us to Human-like General Intelligence?
- [gdm-doi-10.1587-bplus.20.36](<https://doi.org/10.1587/bplus.20.36>) - 2026-01-01 · `affiliated` · 大規模言語モデルの現在地 ――世界の潮流と日本の動向――
- [gdm-doi-10.1017-s0140525x26104609](<https://doi.org/10.1017/s0140525x26104609>) - 2026-01-01 · `affiliated` · Rich data drive generalization: Lessons from machine learning for linguistics and cognitive science
- [gdm-doi-10.1017-s0140525x25101714](<https://doi.org/10.1017/s0140525x25101714>) - 2026-01-01 · `affiliated` · Fair contracts for artificial intelligence?

### 17.2 2025（286 条）

- [gdm-url-76b573eca27e](<https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf>) - 2025-12-19 · `affiliated` · Gemma Scope 2 Technical Paper
- [gdm-doi-10.1073-pnas.2521089122](<https://doi.org/10.1073/pnas.2521089122>) - 2025-12-19 · `affiliated` · High-speed X-ray tomography for 4D imaging
- [gdm-doi-10.1038-s42256-025-01115-6](<https://doi.org/10.1038/s42256-025-01115-6>) - 2025-12-18 · `affiliated` · A psychometric framework for evaluating and shaping personality traits in large language models
- [gdm-url-78c6fa8eafe5](<https://deepmind.google/models/model-cards/gemini-3-flash/>) - 2025-12-17 · `core` · Gemini 3 Flash Model Card
- [gdm-arxiv-2512.14856](<https://deepmind.google/models/gemma/t5gemma/>) - 2025-12-16 · `affiliated` · T5Gemma 2: Seeing, Reading, and Understanding Longer
- [gdm-doi-10.1073-pnas.2523997122](<https://doi.org/10.1073/pnas.2523997122>) - 2025-12-15 · `affiliated` · Advancing evaluation of AI systems when humans make the decisions
- [gdm-doi-10.5281-zenodo.17914353](<https://doi.org/10.5281/zenodo.17914353>) - 2025-12-12 · `affiliated` · The AlphaFold educational summit provides insights into advancing global equity in computational structural biology
- [gdm-url-c80b82e1af5e](<https://storage.googleapis.com/deepmind-media/FACTS/FACTS_benchmark_suite_paper.pdf>) - 2025-12-09 · `affiliated` · FACTS benchmark suite paper
- [gdm-doi-10.1109-cdc57313.2025.11312561](<https://doi.org/10.1109/cdc57313.2025.11312561>) - 2025-12-09 · `affiliated` · Passivity, No-Regret, and Convergent Learning in Contractive Games
- [gdm-doi-10.1109-cdc57313.2025.11312282](<https://doi.org/10.1109/cdc57313.2025.11312282>) - 2025-12-09 · `affiliated` · Buffer Centering for bittide Synchronization via Frame Rotation
- [gdm-arxiv-2512.08924](<https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/>) - 2025-12-09 · `affiliated` · Efficiently Reconstructing Dynamic Scenes One D4RT at a Time
- [gdm-doi-10.1109-asru65441.2025.11434707](<https://doi.org/10.1109/asru65441.2025.11434707>) - 2025-12-06 · `direct` · mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks
- [gdm-web-6202d44636d4](<https://deepmind.google/research/publications/141313/>) - 2025-12-03 · `affiliated` · Capturing Human Preferences with Reward Features
- [gdm-arxiv-2107.02266](<https://doi.org/10.1214/24-aos2450>) - 2025-12-01 · `affiliated` · Near-optimal inference in adaptive linear regression
- [gdm-arxiv-2511.21989](<http://arxiv.org/abs/2511.21989>) - 2025-11-27 · `affiliated` · Selecting User Histories to Generate LLM Users for Cold-Start Item Recommendation
- [gdm-doi-10.1093-nar-gkaf1226](<https://doi.org/10.1093/nar/gkaf1226>) - 2025-11-22 · `affiliated` · AlphaFold Protein Structure Database 2025: a redesigned interface and updated structural coverage
- [gdm-doi-10.1002-aaai.70040](<https://deepmind.google/research/publications/42697/>) - 2025-11-21 · `affiliated` · Imitation Learning is Probably Existentially Safe
- [gdm-web-7ba25bd90503](<https://deepmind.google/models/model-cards/gemini-3-pro-image/>) - 2025-11-20 · `core` · Gemini 3 Pro Image Model Card
- [gdm-doi-10.1145-3719027.3765122](<https://doi.org/10.1145/3719027.3765122>) - 2025-11-19 · `affiliated` · Cascading Adversarial Bias from Injection to Distillation in Language Models
- [gdm-doi-10.1001-jamapsychiatry.2025.3258](<https://doi.org/10.1001/jamapsychiatry.2025.3258>) - 2025-11-19 · `affiliated` · Generative Psychometrics—An Emerging Frontier in Mental Health Measurement
- [gdm-url-d94bc4796e34](<https://deepmind.google/models/model-cards/gemini-3-pro/>) - 2025-11-18 · `core` · Gemini 3 Pro Model Card
- [gdm-doi-10.1007-978-981-95-4381-6_5](<https://doi.org/10.1007/978-981-95-4381-6_5>) - 2025-11-15 · `affiliated` · Excitatory-Inhibitory Dynamics in Adaptive Decision-Making
- [gdm-url-d41003b9db7f](<https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/SIMA_Tech_Report_2025.pdf>) - 2025-11-13 · `direct` · SIMA Tech Report 2025
- [gdm-doi-10.1101-2025.11.12.688086](<https://doi.org/10.1101/2025.11.12.688086>) - 2025-11-13 · `affiliated` · AI-discovered tuning laws explain neuronal population code geometry
- [gdm-doi-10.1103-nbvq-gykq](<https://doi.org/10.1103/nbvq-gykq>) - 2025-11-12 · `affiliated` · Computational search for materials having a giant anomalous Hall effect in the pyrochlore and spinel crystal structures
- [gdm-doi-10.1038-s41586-025-09833-y](<https://doi.org/10.1038/s41586-025-09833-y>) - 2025-11-12 · `affiliated` · Olympiad-level formal mathematical reasoning with reinforcement learning
- [gdm-doi-10.1016-j.aei.2025.104021](<https://doi.org/10.1016/j.aei.2025.104021>) - 2025-11-08 · `affiliated` · Recurrent U-Net-based Graph Neural Network (RUGNN) for accurate deformation predictions in sheet material forming
- [gdm-doi-10.1038-s44387-025-00041-7](<https://doi.org/10.1038/s44387-025-00041-7>) - 2025-11-05 · `affiliated` · We need accountability in human–AI agent relationships
- [gdm-web-e36538e67c47](<https://deepmind.google/research/publications/180362/>) - 2025-11-04 · `direct` · To Mask or to Mirror: Human-AI Alignment in Collective Reasoning
- [gdm-url-30f975dda414](<https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf>) - 2025-11-01 · `core` · Frontier Safety Framework Report – Gemini 3 Pro
- [gdm-arxiv-2202.09848](<https://doi.org/10.1007/s10489-025-06989-y>) - 2025-11-01 · `affiliated` · Personalized federated learning with exact stochastic gradient descent
- [gdm-arxiv-2402.10236](<https://doi.org/10.1126/sciadv.adp0834>) - 2025-10-31 · `affiliated` · Discovering sensorimotor agency in cellular automata using diversity search
- [gdm-arxiv-2510.26396](<https://deepmind.google/research/publications/210560/>) - 2025-10-30 · `affiliated` · A Pragmatic View of AI Personhood
- [gdm-doi-10.1145-3773291](<https://doi.org/10.1145/3773291>) - 2025-10-27 · `affiliated` · Dynamics of Ethereum’s EIP-1559 Transaction Fee Mechanism
- [gdm-doi-10.1109-tpami.2025.3625728](<https://doi.org/10.1109/tpami.2025.3625728>) - 2025-10-27 · `affiliated` · Neural Eigenfunctions are Structured Representation Learners
- [gdm-doi-10.1038-s41586-025-09761-x](<https://doi.org/10.1038/s41586-025-09761-x>) - 2025-10-22 · `affiliated` · Discovering state-of-the-art reinforcement learning algorithms
- [gdm-doi-10.3233-faia251454](<https://doi.org/10.3233/faia251454>) - 2025-10-21 · `affiliated` · Beyond Listenership: AI-Predicted Interventions Drive Improvements in Maternal Health Behaviours
- [gdm-doi-10.1109-iros60139.2025.11247678](<https://doi.org/10.1109/iros60139.2025.11247678>) - 2025-10-19 · `affiliated` · Drive&amp;Gen: Co-Evaluating End-to-End Driving and Video Generation Models
- [gdm-doi-10.1109-iros60139.2025.11247210](<https://doi.org/10.1109/iros60139.2025.11247210>) - 2025-10-19 · `affiliated` · AugInsert: Learning Robust Visual-Force Policies via Data Augmentation for Object Assembly Tasks
- [gdm-doi-10.1109-iros60139.2025.11246267](<https://doi.org/10.1109/iros60139.2025.11246267>) - 2025-10-19 · `affiliated` · QuietPaw: Learning Quadrupedal Locomotion with Versatile Noise Preference Alignment
- [gdm-doi-10.1109-iros60139.2025.11246124](<https://doi.org/10.1109/iros60139.2025.11246124>) - 2025-10-19 · `affiliated` · Exploiting Policy Idling for Dexterous Manipulation
- [gdm-doi-10.1109-iccvw69036.2025.00445](<https://doi.org/10.1109/iccvw69036.2025.00445>) - 2025-10-19 · `affiliated` · Infusing Fine-Grained Visual Knowledge to Vision-Language Models
- [gdm-doi-10.1109-iccv51701.2025.02304](<https://doi.org/10.1109/iccv51701.2025.02304>) - 2025-10-19 · `affiliated` · Bolt3D: Generating 3D Scenes in Seconds
- [gdm-doi-10.1109-iccv51701.2025.02222](<https://doi.org/10.1109/iccv51701.2025.02222>) - 2025-10-19 · `affiliated` · Minerva: Evaluating Complex Video Reasoning
- [gdm-doi-10.1109-iccv51701.2025.01809](<https://doi.org/10.1109/iccv51701.2025.01809>) - 2025-10-19 · `affiliated` · LayerLock: Non-Collapsing Representation Learning with Progressive Freezing
- [gdm-doi-10.1109-iccv51701.2025.01733](<https://doi.org/10.1109/iccv51701.2025.01733>) - 2025-10-19 · `affiliated` · From Prompt to Progression: Taming Video Diffusion Models for Seamless Attribute Transition
- [gdm-doi-10.1109-iccv51701.2025.01186](<https://doi.org/10.1109/iccv51701.2025.01186>) - 2025-10-19 · `affiliated` · Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images
- [gdm-doi-10.1109-iccv51701.2025.00934](<https://doi.org/10.1109/iccv51701.2025.00934>) - 2025-10-19 · `affiliated` · MoMaps: Semantics-Aware Scene Motion Generation with Motion Maps
- [gdm-doi-10.1109-iccv51701.2025.00904](<https://doi.org/10.1109/iccv51701.2025.00904>) - 2025-10-19 · `affiliated` · Tapnext: Tracking Any Point (Tap) as Next Token Prediction
- [gdm-arxiv-2507.03578](<https://doi.org/10.1109/iccv51701.2025.02024>) - 2025-10-19 · `affiliated` · SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications
- [gdm-arxiv-2506.21117](<https://doi.org/10.1109/iccv51701.2025.00732>) - 2025-10-19 · `affiliated` · CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization
- [gdm-arxiv-2503.24366](<https://doi.org/10.1109/iccv51701.2025.02443>) - 2025-10-19 · `affiliated` · StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting
- [gdm-arxiv-2503.21581](<https://doi.org/10.1109/iccv51701.2025.02497>) - 2025-10-19 · `affiliated` · AlignDiff: Learning Physically-Grounded Camera Alignment via Diffusion
- [gdm-arxiv-2503.06271](<https://doi.org/10.1109/iccv51701.2025.00448>) - 2025-10-19 · `affiliated` · Splattalk: 3D VQA with Gaussian Splatting
- [gdm-arxiv-2502.07001](<https://doi.org/10.1109/iccv51701.2025.01574>) - 2025-10-19 · `affiliated` · From Image to Video: An Empirical Study of Diffusion Representations
- [gdm-arxiv-2501.13087](<https://doi.org/10.1109/iccv51701.2025.02620>) - 2025-10-19 · `affiliated` · Orchid: Image Latent Diffusion for Joint Appearance and Geometry Generation
- [gdm-arxiv-2411.18650](<https://doi.org/10.1109/iccv51701.2025.00581>) - 2025-10-19 · `affiliated` · RoMo: Robust Motion Segmentation Improves Structure from Motion
- [gdm-arxiv-2411.13626](<https://doi.org/10.1109/iccv51701.2025.01974>) - 2025-10-19 · `affiliated` · Principles of Visual Tokens for Efficient Video Understanding
- [gdm-arxiv-2407.02489](<https://doi.org/10.1109/iccv51701.2025.01482>) - 2025-10-19 · `affiliated` · Magic Insert: Style-Aware Drag-And-Drop
- [gdm-arxiv-2405.14715](<https://doi.org/10.1109/iccv51701.2025.00174>) - 2025-10-19 · `affiliated` · Towards Cross-Modal Backward-Compatible Representation Learning for Vision-Language Models
- [gdm-arxiv-2510.15001](<https://deepmind.google/models/gemma/vaultgemma/>) - 2025-10-15 · `affiliated` · VaultGemma: A Differentially Private Gemma Model
- [gdm-doi-10.1109-waspaa66052.2025.11230963](<https://doi.org/10.1109/waspaa66052.2025.11230963>) - 2025-10-12 · `affiliated` · Source Separation by Flow Matching
- [gdm-arxiv-2510.08409](<http://arxiv.org/abs/2510.08409>) - 2025-10-09 · `affiliated` · Optimal Stopping in Latent Diffusion Models
- [gdm-url-98dcfe57eb6c](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Computer-Use-Model-Card.pdf>) - 2025-10-07 · `core` · Gemini 2.5 Computer Use Model Card
- [gdm-doi-10.1021-acsnano.5c09057](<https://doi.org/10.1021/acsnano.5c09057>) - 2025-10-02 · `affiliated` · Zero-Shot Autonomous Microscopy for Scalable and Intelligent Characterization of 2D Materials
- [gdm-doi-10.1038-s41562-025-02309-z](<https://doi.org/10.1038/s41562-025-02309-z>) - 2025-10-01 · `affiliated` · The impact of advanced AI systems on democracy
- [gdm-arxiv-2507.00583](<https://deepmind.google/research/publications/160567/>) - 2025-09-29 · `affiliated` · AI-Generated Video Detection via Perceptual Straightening
- [gdm-url-f14dd44296f9](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Flash-Model-Card.pdf>) - 2025-09-26 · `core` · Gemini 2.5 Flash and Gemini 2.5 Flash Image Model Card
- [gdm-url-00b8c518adaf](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Flash-Lite-Model-Card.pdf>) - 2025-09-26 · `core` · Gemini 2.5 Flash-Lite Model Card
- [gdm-web-bdb639b496cc](<https://ai.google.dev/gemma/docs/embeddinggemma/model_card>) - 2025-09-25 · `affiliated` · EmbeddingGemma Model Card
- [gdm-url-40b6bdf31038](<https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1-5-Tech-Report.pdf#page=30>) - 2025-09-25 · `core` · Gemini Robotics 1.5 Model Card
- [gdm-arxiv-2509.20354](<https://deepmind.google/research/publications/194199/>) - 2025-09-24 · `affiliated` · EmbeddingGemma: Powerful and Lightweight Text Representations
- [gdm-arxiv-2509.20328](<https://deepmind.google/research/publications/203190/>) - 2025-09-24 · `affiliated` · Video models are zero-shot learners and reasoners
- [gdm-doi-10.21203-rs.3.rs-7330548-v1](<https://doi.org/10.21203/rs.3.rs-7330548/v1>) - 2025-09-23 · `affiliated` · Humanizing the dehumanized: A test of strategies
- [gdm-doi-10.1007-978-3-032-06004-4_32](<https://doi.org/10.1007/978-3-032-06004-4_32>) - 2025-09-22 · `affiliated` · On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI
- [gdm-arxiv-2507.13383](<https://deepmind.google/research/publications/118251/>) - 2025-09-18 · `affiliated` · Whose View of Safety? A Deep DIVE Dataset for Pluralistic Alignment of Text-to-Image Models
- [gdm-doi-10.1126-scirobotics.adt1497](<https://doi.org/10.1126/scirobotics.adt1497>) - 2025-09-17 · `affiliated` · A review of learning-based dynamics models for robotic manipulation
- [gdm-arxiv-2509.14185](<https://deepmind.google/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/>) - 2025-09-17 · `affiliated` · Discovery of Unstable Singularities
- [gdm-doi-10.1038-s41591-025-03953-8](<https://doi.org/10.1038/s41591-025-03953-8>) - 2025-09-15 · `affiliated` · The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence
- [gdm-arxiv-2509.14016](<https://deepmind.google/research/publications/145314/>) - 2025-09-04 · `affiliated` · Improving cosmological reach of LIGO usingDeep Loop Shaping
- [gdm-arxiv-2509.05397](<https://deepmind.google/research/publications/111579/>) - 2025-09-03 · `affiliated` · RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning
- [gdm-doi-10.24963-ijcai.2025-33](<https://doi.org/10.24963/ijcai.2025/33>) - 2025-09-01 · `affiliated` · Quantifying the Self-Interest Level of Markov Social Dilemmas
- [gdm-doi-10.1088-1742-6596-3104-1-012057](<https://doi.org/10.1088/1742-6596/3104/1/012057>) - 2025-09-01 · `affiliated` · Rapid prediction of material deformation in hot stamping of battery box geometries using graph neural network
- [gdm-doi-10.1021-jacs.5c09558](<https://doi.org/10.1021/jacs.5c09558>) - 2025-08-31 · `affiliated` · Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields
- [gdm-doi-10.1145-3763794](<https://doi.org/10.1145/3763794>) - 2025-08-26 · `affiliated` · Timetide: A Programming Model for Logically Synchronous Distributed Systems
- [gdm-doi-10.1101-2025.08.21.671544](<https://doi.org/10.1101/2025.08.21.671544>) - 2025-08-26 · `affiliated` · Evaluating the Effectiveness of Parameter-Efficient Fine-Tuning in Genomic Classification Tasks
- [gdm-doi-10.1109-hoti66940.2025.00019](<https://doi.org/10.1109/hoti66940.2025.00019>) - 2025-08-20 · `affiliated` · bittide: Control Time, Not Flows
- [gdm-doi-10.1080-10586458.2025.2542174](<https://doi.org/10.1080/10586458.2025.2542174>) - 2025-08-18 · `affiliated` · The Unknotting Number, Hard Unknot Diagrams, and Reinforcement Learning
- [gdm-web-df14dba6bd3c](<https://ai.google.dev/gemma/docs/core/model_card_3>) - 2025-08-14 · `affiliated` · Gemma 3 Model Card
- [gdm-doi-10.1101-2025.08.08.669381](<https://doi.org/10.1101/2025.08.08.669381>) - 2025-08-12 · `affiliated` · Low dimensional latent structure underlying the choices of mice
- [gdm-doi-10.1109-tit.2025.3597092](<https://deepmind.google/research/publications/148245/>) - 2025-08-08 · `affiliated` · Properties of Algorithmic Information Distance
- [gdm-doi-10.1007-978-3-032-00800-8_30](<https://doi.org/10.1007/978-3-032-00800-8_30>) - 2025-08-06 · `affiliated` · Value Under Ignorance in Universal Artificial Intelligence
- [gdm-arxiv-2508.04665](<https://deepmind.google/blog/how-ai-is-helping-advance-the-science-of-bioacoustics-to-save-endangered-species/>) - 2025-08-06 · `affiliated` · Perch 2.0: The Bittern Lesson for Bioacoustics
- [gdm-arxiv-2509.10289](<https://doi.org/10.1038/d41586-025-02454-5>) - 2025-08-04 · `affiliated` · We need a new ethics for a world of AI agents
- [gdm-doi-10.1109-igarss55030.2025.11243780](<https://doi.org/10.1109/igarss55030.2025.11243780>) - 2025-08-03 · `affiliated` · Not Every Tree is A Forest: Benchmarking Forest Types from Satellite Remote Sensing
- [gdm-url-f9179a0c9b65](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf>) - 2025-08-01 · `core` · Gemini 2.5 Deep Think Model Card
- [gdm-arxiv-2504.13621](<https://deepmind.google/research/publications/192581/>) - 2025-08-01 · `affiliated` · Visual Intention Grounding for Egocentric Assistants
- [gdm-doi-10.1101-2025.07.31.667923](<https://doi.org/10.1101/2025.07.31.667923>) - 2025-07-31 · `affiliated` · Hybrid Neural-Cognitive Models Reveal Flexible Context-Dependent Information Processing in Reversal Learning
- [gdm-arxiv-2507.22291](<https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/>) - 2025-07-29 · `affiliated` · AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data
- [gdm-doi-10.1016-j.compbiomed.2025.110824](<https://doi.org/10.1016/j.compbiomed.2025.110824>) - 2025-07-28 · `affiliated` · Segmentation of the human tongue musculature using MRI: Field guide and validation in motor neuron disease
- [gdm-doi-10.1145-3730843](<https://doi.org/10.1145/3730843>) - 2025-07-27 · `affiliated` · TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space
- [gdm-doi-10.1038-s41586-025-09292-5](<https://doi.org/10.1038/s41586-025-09292-5>) - 2025-07-23 · `affiliated` · Contextualizing ancient texts with generative neural networks
- [gdm-doi-10.1609-aies.v8i3.36742](<https://deepmind.google/research/publications/139779/>) - 2025-07-21 · `affiliated` · "Just a Strange Pic": Rethinking 'Safety' in GenAI Image Safety Annotation Tasks from Diverse Annotators’ Perspectives
- [gdm-web-373531579414](<https://deepmind.google/research/publications/181976/>) - 2025-07-16 · `affiliated` · Dialogues Between Technologists and the Art Worlds
- [gdm-web-0bfcb6bbb2a4](<https://deepmind.google/research/publications/126936/>) - 2025-07-13 · `affiliated` · Long-Form Speech Generation with Spoken Language Models
- [gdm-doi-10.1145-3726302.3730348](<https://deepmind.google/research/publications/147939/>) - 2025-07-13 · `affiliated` · Large Language Models as Rankers, Judges, and Assistants: A Perspective on the Potential Over-Reliance on LLMs in IR
- [gdm-arxiv-2410.09615](<https://deepmind.google/research/publications/148040/>) - 2025-07-13 · `affiliated` · SLIM: ONE-SHOT QUANTIZED SPARSE PLUS LOW-RANK APPROXIMATION OF LLMS
- [gdm-doi-10.23919-acc63710.2025.11107803](<https://doi.org/10.23919/acc63710.2025.11107803>) - 2025-07-08 · `affiliated` · Modeling Buffer Occupancy in bittide Systems
- [gdm-doi-10.1371-journal.pcbi.1013226](<https://doi.org/10.1371/journal.pcbi.1013226>) - 2025-07-02 · `affiliated` · Nucleus accumbens dopamine release reflects Bayesian inference during instrumental learning
- [gdm-doi-10.1038-s41586-025-09215-4](<https://doi.org/10.1038/s41586-025-09215-4>) - 2025-07-02 · `affiliated` · A foundation model to predict and capture human cognition
- [gdm-url-4a5f0122d44e](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-Robotics-On-Device-Model-Card.pdf?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=>) - 2025-07-01 · `core` · Gemini Robotics On-Device Model Card
- [gdm-doi-10.1111-cogs.70083](<https://doi.org/10.1111/cogs.70083>) - 2025-07-01 · `affiliated` · Statistical or Embodied? Comparing Colorseeing, Colorblind, Painters, and Large Language Models in Their Processing of Color Metaphors
- [gdm-arxiv-2506.12346](<https://deepmind.google/research/publications/102792/>) - 2025-07-01 · `direct` · Rethinking Example Selection in the Era of Million-Token Models
- [gdm-url-082786c15ca1](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Pro-Model-Card.pdf>) - 2025-06-27 · `core` · Gemini 2.5 Pro Model Card
- [gdm-doi-10.1101-2025.06.25.661532](<https://doi.org/10.1101/2025.06.25.661532>) - 2025-06-27 · `affiliated` · AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model
- [gdm-doi-10.21203-rs.3.rs-6869504-v1](<https://doi.org/10.21203/rs.3.rs-6869504/v1>) - 2025-06-26 · `affiliated` · Association of centenarian polygenic score with disability-free survival and its modification effects on aging outcomes
- [gdm-arxiv-2506.21718](<https://deepmind.google/research/publications/187733/>) - 2025-06-26 · `affiliated` · Performance Prediction for Large Systems via Text-to-Text Regression
- [gdm-doi-10.1145-3695053.3731092](<https://deepmind.google/research/publications/81986/>) - 2025-06-23 · `affiliated` · LIA: Cost-efficient LLM Inference Acceleration with Intel Advanced Matrix Extensions and CXL
- [gdm-doi-10.1109-isit63088.2025.11195239](<https://doi.org/10.1109/isit63088.2025.11195239>) - 2025-06-22 · `affiliated` · Discretized Approximate Ancestral Sampling
- [gdm-web-f4609afb142f](<https://deepmind.google/research/publications/122089/>) - 2025-06-20 · `affiliated` · AuPair: Golden Example Pairs for Code Repair
- [gdm-web-2def4ae7889e](<https://ai.google.dev/gemma/docs/shieldgemma/model_card_2>) - 2025-06-17 · `affiliated` · ShieldGemma 2 Model Card
- [gdm-web-222477dc41a8](<https://ai.google.dev/gemma/docs/gemma-3n/model_card>) - 2025-06-17 · `affiliated` · Gemma 3n Model Card
- [gdm-doi-10.1073-pnas.2319949121](<https://doi.org/10.1073/pnas.2319949121>) - 2025-06-16 · `affiliated` · Deep mechanism design: Learning social and economic policies for human benefit
- [gdm-doi-10.1073-pnas.2319948121](<https://doi.org/10.1073/pnas.2319948121>) - 2025-06-16 · `affiliated` · Collective cooperative intelligence
- [gdm-doi-10.1073-pnas.2319947121](<https://doi.org/10.1073/pnas.2319947121>) - 2025-06-16 · `affiliated` · Tabula rasa agents display emergent in-group behavior
- [gdm-doi-10.1073-pnas.2319929121](<https://doi.org/10.1073/pnas.2319929121>) - 2025-06-16 · `affiliated` · Heterogeneity, reinforcement learning, and chaos in population games
- [gdm-arxiv-2503.24340](<https://doi.org/10.1145/3717823.3718242>) - 2025-06-15 · `affiliated` · Faster Rates for No-Regret Learning in General Games via Cautious Optimism
- [gdm-url-acfd552b57ab](<https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/how-we-re-supporting-better-tropical-cyclone-prediction-with-ai/skillful-joint-probabilistic-weather-forecasting-from-marginals.pdf>) - 2025-06-12 · `affiliated` · Skillful Joint Probabilistic Weather Forecasting from Marginals
- [gdm-doi-10.1098-rstb.2024.0280](<https://doi.org/10.1098/rstb.2024.0280>) - 2025-06-12 · `affiliated` · Using tropical reef, bird and unrelated sounds for superior transfer learning in marine bioacoustics
- [gdm-doi-10.1145-3729321](<https://doi.org/10.1145/3729321>) - 2025-06-10 · `affiliated` · Handling the Selection Monad
- [gdm-doi-10.1109-cvpr52734.2025.02793](<https://doi.org/10.1109/cvpr52734.2025.02793>) - 2025-06-10 · `affiliated` · Cropper: Vision-Language Model for Image Cropping through In-Context Learning
- [gdm-doi-10.1109-cvpr52734.2025.02707](<https://doi.org/10.1109/cvpr52734.2025.02707>) - 2025-06-10 · `affiliated` · Flexible Frame Selection for Efficient Video Reasoning
- [gdm-doi-10.1109-cvpr52734.2025.02706](<https://doi.org/10.1109/cvpr52734.2025.02706>) - 2025-06-10 · `affiliated` · VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models
- [gdm-doi-10.1109-cvpr52734.2025.02427](<https://doi.org/10.1109/cvpr52734.2025.02427>) - 2025-06-10 · `affiliated` · CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models
- [gdm-doi-10.1109-cvpr52734.2025.02166](<https://doi.org/10.1109/cvpr52734.2025.02166>) - 2025-06-10 · `affiliated` · Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion
- [gdm-doi-10.1109-cvpr52734.2025.01838](<https://doi.org/10.1109/cvpr52734.2025.01838>) - 2025-06-10 · `affiliated` · Visual Lexicon: Rich Image Features in Language Space
- [gdm-doi-10.1109-cvpr52734.2025.01741](<https://doi.org/10.1109/cvpr52734.2025.01741>) - 2025-06-10 · `affiliated` · A Bias-Free Training Paradigm for More General AI-generated Image Detection
- [gdm-doi-10.1109-cvpr52734.2025.01723](<https://doi.org/10.1109/cvpr52734.2025.01723>) - 2025-06-10 · `affiliated` · Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation
- [gdm-doi-10.1109-cvpr52734.2025.01721](<https://doi.org/10.1109/cvpr52734.2025.01721>) - 2025-06-10 · `affiliated` · Calibrated Multi-Preference Optimization for Aligning Diffusion Models
- [gdm-doi-10.1109-cvpr52734.2025.01535](<https://doi.org/10.1109/cvpr52734.2025.01535>) - 2025-06-10 · `affiliated` · SimVS: Simulating World Inconsistencies for Robust View Synthesis
- [gdm-doi-10.1109-cvpr52734.2025.01482](<https://doi.org/10.1109/cvpr52734.2025.01482>) - 2025-06-10 · `affiliated` · VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis
- [gdm-doi-10.1109-cvpr52734.2025.01465](<https://doi.org/10.1109/cvpr52734.2025.01465>) - 2025-06-10 · `affiliated` · Language-Guided Image Tokenization for Generation
- [gdm-doi-10.1109-cvpr52734.2025.01345](<https://doi.org/10.1109/cvpr52734.2025.01345>) - 2025-06-10 · `affiliated` · Active Data Curation Effectively Distills Large-Scale Multimodal Models
- [gdm-doi-10.1109-cvpr52734.2025.01274](<https://doi.org/10.1109/cvpr52734.2025.01274>) - 2025-06-10 · `affiliated` · Learning from Streaming Video with Orthogonal Gradients
- [gdm-doi-10.1109-cvpr52734.2025.01257](<https://doi.org/10.1109/cvpr52734.2025.01257>) - 2025-06-10 · `affiliated` · FirePlace: Geometric Refinements of LLM Common Sense Reasoning for 3D Object Placement
- [gdm-doi-10.1109-cvpr52734.2025.01168](<https://doi.org/10.1109/cvpr52734.2025.01168>) - 2025-06-10 · `affiliated` · Generative Omnimatte: Learning to Decompose Video into Layers
- [gdm-doi-10.1109-cvpr52734.2025.00982](<https://doi.org/10.1109/cvpr52734.2025.00982>) - 2025-06-10 · `affiliated` · Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos
- [gdm-doi-10.1109-cvpr52734.2025.00981](<https://doi.org/10.1109/cvpr52734.2025.00981>) - 2025-06-10 · `affiliated` · MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos
- [gdm-doi-10.1109-cvpr52734.2025.00910](<https://doi.org/10.1109/cvpr52734.2025.00910>) - 2025-06-10 · `affiliated` · Token Cropr: Faster ViTs for Quite a Few Tasks
- [gdm-doi-10.1109-cvpr52734.2025.00639](<https://doi.org/10.1109/cvpr52734.2025.00639>) - 2025-06-10 · `affiliated` · SceneCrafter: Controllable Multi-View Driving Scene Editing
- [gdm-doi-10.1109-cvpr52734.2025.00576](<https://doi.org/10.1109/cvpr52734.2025.00576>) - 2025-06-10 · `affiliated` · DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
- [gdm-doi-10.1109-cvpr52734.2025.00557](<https://doi.org/10.1109/cvpr52734.2025.00557>) - 2025-06-10 · `affiliated` · 3D-GSW: 3D Gaussian Splatting for Robust Watermarking
- [gdm-doi-10.1109-cvpr52734.2025.00403](<https://doi.org/10.1109/cvpr52734.2025.00403>) - 2025-06-10 · `affiliated` · Context-Aware Multimodal Pretraining
- [gdm-doi-10.1109-cvpr52734.2025.00354](<https://doi.org/10.1109/cvpr52734.2025.00354>) - 2025-06-10 · `affiliated` · Learning Visual Composition through Improved Semantic Guidance
- [gdm-doi-10.1109-cvpr52734.2025.00292](<https://doi.org/10.1109/cvpr52734.2025.00292>) - 2025-06-10 · `affiliated` · Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks
- [gdm-doi-10.1109-cvpr52734.2025.00010](<https://doi.org/10.1109/cvpr52734.2025.00010>) - 2025-06-10 · `affiliated` · Motion Prompting: Controlling Video Generation with Motion Trajectories
- [gdm-arxiv-2504.00072](<https://doi.org/10.1109/cvpr52734.2025.01765>) - 2025-06-10 · `affiliated` · Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs
- [gdm-arxiv-2506.08065](<http://arxiv.org/abs/2506.08065>) - 2025-06-09 · `affiliated` · Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions
- [gdm-arxiv-2506.06248](<https://arxiv.org/abs/2506.06248>) - 2025-06-06 · `affiliated` · Lagrangian-based Equilibrium Propagation: generalisation to arbitrary boundary conditions & equivalence with Hamiltonian Echo Learning
- [gdm-doi-10.1103-physrevlett.134.221601](<https://doi.org/10.1103/physrevlett.134.221601>) - 2025-06-05 · `affiliated` · Channel Capacity of a Relativistic String
- [gdm-doi-10.1021-acs.jctc.4c01784](<https://doi.org/10.1021/acs.jctc.4c01784>) - 2025-06-05 · `affiliated` · Force-Field Optimization by End-to-End Differentiable Atomistic Simulation
- [gdm-doi-10.1109-mipro65660.2025.11131790](<https://doi.org/10.1109/mipro65660.2025.11131790>) - 2025-06-02 · `affiliated` · Automating Prompt Leakage Attacks on Large Language Models Using Agentic Approach
- [gdm-doi-10.1016-j.physd.2025.134669](<https://deepmind.google/research/publications/148243/>) - 2025-06-01 · `affiliated` · Bridging Algorithmic Information Theory and Machine Learning, Part II: Clustering, Density Estimation, Kolmogorov Complexity-Based Kernels, and Kernel Learning in Unsupervised Learning
- [gdm-doi-10.65109-zohx8824](<https://doi.org/10.65109/zohx8824>) - 2025-05-28 · `affiliated` · Will Systems of LLM Agents Lead to Cooperation: An Investigation into a Social Dilemma
- [gdm-doi-10.65109-jnmb7739](<https://doi.org/10.65109/jnmb7739>) - 2025-05-28 · `direct` · Cultural Evolution of Cooperation among LLM Agents
- [gdm-doi-10.65109-gzfu8152](<https://doi.org/10.65109/gzfu8152>) - 2025-05-28 · `affiliated` · Game of Thoughts: Iterative Reasoning in Game-Theoretic Domains with Large Language Models
- [gdm-doi-10.65109-adut2666](<https://doi.org/10.65109/adut2666>) - 2025-05-28 · `affiliated` · Resolving Social Dilemmas with Minimal Reward Transfer - Extended Abstract
- [gdm-doi-10.1609-aaaiss.v5i1.35585](<https://doi.org/10.1609/aaaiss.v5i1.35585>) - 2025-05-28 · `affiliated` · DyESP: Accelerating Hyperparameter-Architecture Search via Dynamic Exploration and Space Pruning
- [gdm-doi-10.1057-s41599-025-04532-5](<https://doi.org/10.1057/s41599-025-04532-5>) - 2025-05-28 · `affiliated` · Why human–AI relationships need socioaffective alignment
- [gdm-doi-10.1016-j.cnsns.2025.108963](<https://doi.org/10.1016/j.cnsns.2025.108963>) - 2025-05-27 · `affiliated` · Interval maps mimicking circle rotations
- [gdm-doi-10.1038-s41586-025-09061-4](<https://doi.org/10.1038/s41586-025-09061-4>) - 2025-05-26 · `affiliated` · Scaling and logic in the colour code on a superconducting quantum processor
- [gdm-arxiv-2505.19731](<http://arxiv.org/abs/2505.19731>) - 2025-05-26 · `affiliated` · Proximal Point Nash Learning from Human Feedback
- [gdm-doi-10.1093-oxfordhb-9780198940272.013.0023](<https://doi.org/10.1093/oxfordhb/9780198940272.013.0023>) - 2025-05-22 · `affiliated` · How Risky Are Open Frontier Models?
- [gdm-doi-10.1093-oxfordhb-9780192886491.013.18](<https://doi.org/10.1093/oxfordhb/9780192886491.013.18>) - 2025-05-22 · `affiliated` · Language evolution with deep learning
- [gdm-doi-10.1016-j.tics.2025.05.007](<https://doi.org/10.1016/j.tics.2025.05.007>) - 2025-05-22 · `affiliated` · Defending the foundation model view of infant development
- [gdm-url-1d5924e59d3f](<https://storage.googleapis.com/deepmind-media/Model-Cards/Imagen-4-Model-Card.pdf?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=>) - 2025-05-20 · `affiliated` · Imagen 4 Model Card
- [gdm-doi-10.1016-j.aei.2025.103458](<https://doi.org/10.1016/j.aei.2025.103458>) - 2025-05-20 · `affiliated` · A multi-level graph-based surrogate model for real-time high-fidelity sheet forming simulations
- [gdm-doi-10.1109-icra55743.2025.11128270](<https://doi.org/10.1109/icra55743.2025.11128270>) - 2025-05-19 · `affiliated` · Chain-of-Modality: Learning Manipulation Programs from Multimodal Human Videos with Vision-Language-Models
- [gdm-doi-10.1109-icra55743.2025.11127882](<https://doi.org/10.1109/icra55743.2025.11127882>) - 2025-05-19 · `affiliated` · SAS-Prompt: Large Language Models as Numerical Optimizers for Robot Self-Improvement
- [gdm-doi-10.1109-icra55743.2025.11127813](<https://doi.org/10.1109/icra55743.2025.11127813>) - 2025-05-19 · `affiliated` · DemoStart: Demonstration-Led Auto-Curriculum Applied to Sim-to-Real with Multi-Fingered Robots
- [gdm-doi-10.1109-icra55743.2025.11127525](<https://doi.org/10.1109/icra55743.2025.11127525>) - 2025-05-19 · `affiliated` · RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulation
- [gdm-doi-10.1109-icra55743.2025.11127404](<https://doi.org/10.1109/icra55743.2025.11127404>) - 2025-05-19 · `affiliated` · STEER: Flexible Robotic Manipulation via Dense Language Grounding
- [gdm-url-119a83ec2a50](<https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=>) - 2025-05-14 · `direct` · AlphaEvolve
- [gdm-doi-10.1145-3735633](<https://doi.org/10.1145/3735633>) - 2025-05-14 · `affiliated` · Continual Learning of Large Language Models: A Comprehensive Survey
- [gdm-doi-10.1109-sp61157.2025.00186](<https://doi.org/10.1109/sp61157.2025.00186>) - 2025-05-12 · `affiliated` · Hash-Prune-Invert: Improved Differentially Private Heavy-Hitter Detection in the Two-Server Model
- [gdm-doi-10.1109-sp61157.2025.00178](<https://doi.org/10.1109/sp61157.2025.00178>) - 2025-05-12 · `affiliated` · SoK: Watermarking for AI-Generated Content
- [gdm-doi-10.1109-sp61157.2025.00082](<https://doi.org/10.1109/sp61157.2025.00082>) - 2025-05-12 · `affiliated` · Supporting Human Raters with the Detection of Harmful Content Using Large Language Models
- [gdm-doi-10.22331-q-2025-05-06-1737](<https://doi.org/10.22331/q-2025-05-06-1737>) - 2025-05-06 · `affiliated` · A Quadratic Speedup in Finding Nash Equilibria of Quantum Zero-Sum Games
- [gdm-arxiv-2505.03071](<https://deepmind.google/blog/how-ai-is-helping-advance-the-science-of-bioacoustics-to-save-endangered-species/>) - 2025-05-05 · `affiliated` · The Search for Squawk: Agile Modeling in Bioacoustics
- [gdm-doi-10.1109-deeptest66595.2025.00011](<https://doi.org/10.1109/deeptest66595.2025.00011>) - 2025-05-03 · `affiliated` · Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation
- [gdm-doi-10.36227-techrxiv.174612962.26131807-v1](<https://doi.org/10.36227/techrxiv.174612962.26131807/v1>) - 2025-05-01 · `affiliated` · A Survey on Large Language Model based Human-Agent Systems
- [gdm-arxiv-2412.06771](<https://deepmind.google/research/publications/121578/>) - 2025-05-01 · `affiliated` · Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty
- [gdm-web-e3c57c3b3a02](<https://deepmind.google/research/publications/114003/>) - 2025-04-29 · `affiliated` · Prompting with Phonemes: Enhancing LLM Multilinguality for non-Latin Scripts
- [gdm-arxiv-2506.08569](<https://deepmind.google/research/publications/106327/>) - 2025-04-28 · `affiliated` · Flow-Lenia: Emergent evolutionary dynamics in mass conservative continuous cellular automata
- [gdm-web-525f891425ac](<https://deepmind.google/research/publications/122290/>) - 2025-04-26 · `affiliated` · Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA
- [gdm-web-15b92561f9ef](<https://deepmind.google/research/publications/122088/>) - 2025-04-26 · `affiliated` · Toward Understanding In-context vs. In-weight Learning
- [gdm-doi-10.1109-icse55347.2025.00038](<https://doi.org/10.1109/icse55347.2025.00038>) - 2025-04-26 · `affiliated` · Vulnerability Detection with Code Language Models: How Far are We?
- [gdm-arxiv-2402.01662](<https://deepmind.google/research/publications/65827/>) - 2025-04-26 · `affiliated` · Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives
- [gdm-doi-10.1145-3706599.3719677](<https://doi.org/10.1145/3706599.3719677>) - 2025-04-25 · `affiliated` · LLM Adoption in Data Curation Workflows: Industry Practices and Insights
- [gdm-web-2d272dc76ddd](<https://deepmind.google/research/publications/121073/>) - 2025-04-24 · `affiliated` · MELODI: Exploring Memory Compression for Long Contexts
- [gdm-doi-10.1145-3706598.3714049](<https://doi.org/10.1145/3706598.3714049>) - 2025-04-23 · `affiliated` · AI and Non-Western Art Worlds: Reimagining Critical AI Futures through Artistic Inquiry and Situated Dialogue
- [gdm-doi-10.1038-s41586-025-09029-4](<https://doi.org/10.1038/s41586-025-09029-4>) - 2025-04-23 · `affiliated` · Whole-body physics simulation of fruit fly locomotion
- [gdm-web-8219522662e7](<https://deepmind.google/research/publications/155313/>) - 2025-04-22 · `affiliated` · Societal and technological progress as sewing an ever-growing, ever-changing, patchy, and polychrome quilt
- [gdm-doi-10.1145-3696410.3714692](<https://doi.org/10.1145/3696410.3714692>) - 2025-04-22 · `affiliated` · A Scalable Crawling Algorithm Utilizing Noisy Change-Indicating Signals
- [gdm-doi-10.1101-2025.04.14.648850](<https://doi.org/10.1101/2025.04.14.648850>) - 2025-04-17 · `affiliated` · Scaling Large Language Models for Next-Generation Single-Cell Analysis
- [gdm-url-0e2ab16c7b9c](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-0-Flash-Model-Card.pdf>) - 2025-04-15 · `core` · Gemini 2.0 Flash Model Card
- [gdm-doi-10.1109-tpami.2025.3560423](<https://doi.org/10.1109/tpami.2025.3560423>) - 2025-04-15 · `affiliated` · Hadamard Product in Deep Learning: Introduction, Advances and Challenges
- [gdm-doi-10.1609-aaai.v39i26.34942](<https://doi.org/10.1609/aaai.v39i26.34942>) - 2025-04-11 · `affiliated` · Robust Multi-Objective Preference Alignment with Online DPO
- [gdm-doi-10.1609-aaai.v39i24.34738](<https://doi.org/10.1609/aaai.v39i24.34738>) - 2025-04-11 · `affiliated` · RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs
- [gdm-doi-10.1609-aaai.v39i20.35395](<https://doi.org/10.1609/aaai.v39i20.35395>) - 2025-04-11 · `affiliated` · Differentially Private Prototypes for Imbalanced Transfer Learning
- [gdm-doi-10.1609-aaai.v39i19.34243](<https://doi.org/10.1609/aaai.v39i19.34243>) - 2025-04-11 · `affiliated` · Offline-to-Online Hyperparameter Transfer for Stochastic Bandits
- [gdm-doi-10.1609-aaai.v39i19.34238](<https://doi.org/10.1609/aaai.v39i19.34238>) - 2025-04-11 · `affiliated` · General Uncertainty Estimation with Delta Variances
- [gdm-url-2604411a6472](<https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-0-Flash-Lite-Model-Card.pdf>) - 2025-04-10 · `core` · Gemini 2.0 Flash-Lite Model Card
- [gdm-doi-10.1109-satml64287.2025.00060](<https://doi.org/10.1109/satml64287.2025.00060>) - 2025-04-09 · `affiliated` · Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms Under Composition
- [gdm-doi-10.1109-satml64287.2025.00034](<https://doi.org/10.1109/satml64287.2025.00034>) - 2025-04-09 · `affiliated` · Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy
- [gdm-doi-10.1016-j.media.2025.103556](<https://doi.org/10.1016/j.media.2025.103556>) - 2025-04-09 · `affiliated` · Evaluating medical AI systems in dermatology under uncertain ground truth
- [gdm-arxiv-2504.06196](<https://deepmind.google/research/publications/153799/>) - 2025-04-08 · `affiliated` · TxGemma: Efficient and Agentic LLMs for Therapeutics
- [gdm-doi-10.1017-s0956792525000099](<https://doi.org/10.1017/s0956792525000099>) - 2025-04-02 · `affiliated` · NLP verification: towards a general methodology for certifying robustness
- [gdm-web-cd100313ee99](<https://deepmind.google/research/publications/127036/>) - 2025-04-01 · `affiliated` · Effective Kernel Fuzzing with Learned White-box Test Mutators
- [gdm-arxiv-2504.01081](<https://deepmind.google/models/gemma/shieldgemma-2/>) - 2025-04-01 · `affiliated` · ShieldGemma 2: Robust and Tractable Image Content Moderation
- [gdm-doi-10.1007-s11098-025-02300-4](<https://doi.org/10.1007/s11098-025-02300-4>) - 2025-03-30 · `affiliated` · A matter of principle? AI alignment as the fair treatment of claims
- [gdm-arxiv-2503.22674](<https://deepmind.google/research/publications/121987/>) - 2025-03-28 · `affiliated` · QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks?
- [gdm-doi-10.1073-pnas.2406675122](<https://doi.org/10.1073/pnas.2406675122>) - 2025-03-26 · `affiliated` · Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero
- [gdm-doi-10.1109-3dv66043.2025.00065](<https://doi.org/10.1109/3dv66043.2025.00065>) - 2025-03-25 · `affiliated` · CamCtrl3D: Single-Image Scene Exploration with Precise 3D Camera Control
- [gdm-arxiv-2503.20020](<https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/>) - 2025-03-25 · `direct` · Gemini Robotics: Bringing AI into the Physical World
- [gdm-arxiv-2503.19786](<https://deepmind.google/models/gemma/gemma-3/>) - 2025-03-25 · `affiliated` · Gemma 3 Technical Report
- [gdm-doi-10.1145-3708359.3712085](<https://deepmind.google/research/publications/124002/>) - 2025-03-24 · `affiliated` · Gensors: Authoring Personalized Visual Sensors with Multimodal Foundation Models and Reasoning
- [gdm-doi-10.1038-s41467-025-58043-7](<https://doi.org/10.1038/s41467-025-58043-7>) - 2025-03-22 · `affiliated` · Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem
- [gdm-doi-10.1038-s42256-025-01001-1](<https://doi.org/10.1038/s42256-025-01001-1>) - 2025-03-20 · `affiliated` · Quantum circuit optimization with AlphaTensor
- [gdm-doi-10.1038-s41586-025-08897-0](<https://doi.org/10.1038/s41586-025-08897-0>) - 2025-03-20 · `affiliated` · End-to-end data-driven weather prediction
- [gdm-doi-10.1109-icassp49660.2025.10889875](<https://doi.org/10.1109/icassp49660.2025.10889875>) - 2025-03-12 · `affiliated` · Towards Sub-millisecond Latency Real-Time Speech Enhancement Models on Hearables
- [gdm-doi-10.1109-icassp49660.2025.10889296](<https://doi.org/10.1109/icassp49660.2025.10889296>) - 2025-03-12 · `affiliated` · SimulTron: On-Device Simultaneous Speech to Speech Translation
- [gdm-arxiv-2503.07891](<https://deepmind.google/research/publications/157741/>) - 2025-03-11 · `direct` · Gemini Embedding: Generalizable Embeddings from Gemini
- [gdm-doi-10.1037-abn0000944](<https://doi.org/10.1037/abn0000944>) - 2025-03-10 · `affiliated` · Computational modeling of reversal learning impairments in schizophrenia and bipolar disorder reveals shared failure to exploit rewards.
- [gdm-arxiv-2410.16512](<https://deepmind.google/research/publications/121982/>) - 2025-03-10 · `affiliated` · TIPS: Text-Image Pretraining with Spatial awareness
- [gdm-doi-10.7554-elife.101841.2](<https://doi.org/10.7554/elife.101841.2>) - 2025-03-06 · `affiliated` · Neural mechanisms of credit assignment for delayed outcomes during contingent learning
- [gdm-arxiv-2407.17773](<https://deepmind.google/research/publications/166018/>) - 2025-03-05 · `affiliated` · KiVA: Kid-Inspired Visual Analogies for Testing Large Multimodal Models
- [gdm-doi-10.1112-plms.70031](<https://doi.org/10.1112/plms.70031>) - 2025-03-01 · `affiliated` · Drums of high width
- [gdm-doi-10.1145-3708815](<https://deepmind.google/research/publications/106025/>) - 2025-02-27 · `affiliated` · HCI for AGI
- [gdm-doi-10.1109-wacv61041.2025.00843](<https://doi.org/10.1109/wacv61041.2025.00843>) - 2025-02-26 · `affiliated` · Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence
- [gdm-doi-10.1109-wacv61041.2025.00782](<https://doi.org/10.1109/wacv61041.2025.00782>) - 2025-02-26 · `affiliated` · Learning Visual Grounding from Generative Vision and Language Model
- [gdm-doi-10.1109-wacv61041.2025.00569](<https://doi.org/10.1109/wacv61041.2025.00569>) - 2025-02-26 · `affiliated` · EgoCast: Forecasting Egocentric Human Pose in the Wild
- [gdm-doi-10.1109-wacv61041.2025.00365](<https://doi.org/10.1109/wacv61041.2025.00365>) - 2025-02-26 · `affiliated` · Generating Long-Take Videos via Effective Keyframes and Guidance
- [gdm-doi-10.1109-wacv61041.2025.00364](<https://doi.org/10.1109/wacv61041.2025.00364>) - 2025-02-26 · `affiliated` · Fine-grained Controllable Video Generation via Object Appearance and Context
- [gdm-arxiv-2502.19325](<https://deepmind.google/research/publications/134306/>) - 2025-02-26 · `affiliated` · Partition Tree Weighting for Non-Stationary Stochastic Bandits
- [gdm-web-e13e87b56a04](<https://ai.google.dev/gemma/docs/shieldgemma/model_card>) - 2025-02-25 · `affiliated` · ShieldGemma 1 Model Card
- [gdm-web-9924a3250eac](<https://ai.google.dev/gemma/docs/model_card>) - 2025-02-25 · `affiliated` · Gemma 1 Model Card
- [gdm-web-8be0e9e8fa78](<https://ai.google.dev/gemma/docs/codegemma/model_card>) - 2025-02-25 · `affiliated` · CodeGemma Model Card
- [gdm-web-8a685fd8c581](<https://ai.google.dev/gemma/docs/paligemma/model-card-2>) - 2025-02-25 · `affiliated` · PaliGemma 2 Model Card
- [gdm-web-5397596c781c](<https://ai.google.dev/gemma/docs/paligemma/model-card>) - 2025-02-25 · `affiliated` · PaliGemma 1 Model Card
- [gdm-web-3d9b9b4b4133](<https://ai.google.dev/gemma/docs/model_card_2>) - 2025-02-25 · `affiliated` · Gemma 2 Model Card
- [gdm-web-17877b2e2ac9](<https://ai.google.dev/gemma/docs/recurrentgemma/model_card>) - 2025-02-25 · `affiliated` · RecurrentGemma Model Card
- [gdm-doi-10.1109-tpami.2025.3543846](<https://doi.org/10.1109/tpami.2025.3543846>) - 2025-02-20 · `affiliated` · Optimization of Rank Losses for Image Retrieval
- [gdm-arxiv-2502.14698](<https://deepmind.google/research/publications/112791/>) - 2025-02-20 · `affiliated` · Delta Variances
- [gdm-doi-10.1145-3711897](<https://doi.org/10.1145/3711897>) - 2025-02-15 · `affiliated` · Turing.jl: A General-Purpose Probabilistic Programming Language
- [gdm-arxiv-2502.08646](<https://deepmind.google/research/publications/122892/>) - 2025-02-12 · `affiliated` · Poly-Autoregressive Prediction for Interaction Modeling
- [gdm-arxiv-2502.07617](<https://deepmind.google/research/publications/132991/>) - 2025-02-11 · `affiliated` · Scaling Pre-training to One Hundred Billion Data for Vision Language Models
- [gdm-doi-10.1145-3669940.3707284](<https://doi.org/10.1145/3669940.3707284>) - 2025-02-06 · `affiliated` · PartIR: Composing SPMD Partitioning Strategies for Machine Learning
- [gdm-doi-10.1101-2025.02.05.636732v1](<https://deepmind.google/research/publications/130468/>) - 2025-02-06 · `affiliated` · Automated Discovery of Interpretable Cognitive Programs underlying Reward-guided behavior
- [gdm-doi-10.1101-2025.02.05.636732](<https://doi.org/10.1101/2025.02.05.636732>) - 2025-02-06 · `affiliated` · Discovering Symbolic Cognitive Models from Human and Animal Behavior
- [gdm-doi-10.1126-sciadv.adr6698](<https://doi.org/10.1126/sciadv.adr6698>) - 2025-02-05 · `affiliated` · A detailed theory of thalamic and cortical microcircuits for predictive visual inference
- [gdm-arxiv-2502.02996](<http://arxiv.org/abs/2502.02996>) - 2025-02-05 · `affiliated` · Building Bridges between Regression, Clustering, and Classification
- [gdm-url-f018b70353a1](<https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/updating-the-frontier-safety-framework/Frontier%20Safety%20Framework%202.0.pdf>) - 2025-02-04 · `affiliated` · Frontier Safety Framework 2.0
- [gdm-url-5cd686826d41](<https://hal.science/hal-04925309>) - 2025-02-04 · `affiliated` · Strategic Foundation Models
- [gdm-arxiv-2501.19383](<https://deepmind.google/research/publications/141785/>) - 2025-02-02 · `affiliated` · Decoding-based Regression
- [gdm-doi-10.1016-j.xgen.2025.100762](<https://doi.org/10.1016/j.xgen.2025.100762>) - 2025-01-29 · `affiliated` · A multi-modal transformer for cell type-agnostic regulatory predictions
- [gdm-doi-10.1016-j.jmb.2025.168967](<https://doi.org/10.1016/j.jmb.2025.168967>) - 2025-01-29 · `affiliated` · AlphaFold Protein Structure Database and 3D-Beacons: New Data and Capabilities
- [gdm-doi-10.1101-2025.01.27.634873](<https://doi.org/10.1101/2025.01.27.634873>) - 2025-01-28 · `affiliated` · Self-Contemplating In-Context Learning Enhances T Cell Receptor Generation for Novel Epitopes
- [gdm-doi-10.1073-pnas.2401227121](<https://doi.org/10.1073/pnas.2401227121>) - 2025-01-27 · `affiliated` · How should the advancement of large language models affect the practice of science?
- [gdm-web-8b4ce3a1475b](<https://deepmind.google/research/publications/83299/>) - 2025-01-22 · `affiliated` · Are vision-language models shape or texture biased and can we steer them?
- [gdm-arxiv-2501.13011](<https://deepmind.google/research/publications/148850/>) - 2025-01-22 · `affiliated` · MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking
- [gdm-arxiv-2501.09891](<https://deepmind.google/research/publications/122391/>) - 2025-01-17 · `affiliated` · Evolving Deeper LLM Thinking
- [gdm-doi-10.1103-physreve.111.014118](<https://deepmind.google/research/publications/82794/>) - 2025-01-08 · `affiliated` · Foundations of Algorithmic Thermodynamics
- [gdm-arxiv-2311.14752](<https://doi.org/10.1038/s42254-024-00791-4>) - 2025-01-06 · `affiliated` · Roadmap on machine learning glassy dynamics
- [gdm-web-2fd3b2862cb5](<http://agritrop.cirad.fr/616719/>) - 2025-01-01 · `affiliated` · Overview of BirdCLEF+ 2025: Multi-Taxonomic Sound Identification in the Middle Magdalena, Colombia
- [gdm-doi-10.2139-ssrn.5705186](<https://doi.org/10.2139/ssrn.5705186>) - 2025-01-01 · `affiliated` · Open Technical Problems in Open-Weight AI Model Risk Management
- [gdm-doi-10.2139-ssrn.5403981](<https://doi.org/10.2139/ssrn.5403981>) - 2025-01-01 · `affiliated` · Robust AI Personalization Will Require a Human Context Protocol
- [gdm-doi-10.2139-ssrn.5288768](<https://doi.org/10.2139/ssrn.5288768>) - 2025-01-01 · `affiliated` · Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research
- [gdm-doi-10.2139-ssrn.5183663](<https://doi.org/10.2139/ssrn.5183663>) - 2025-01-01 · `affiliated` · Leveraging Large Language Models for Collective Decision-Making
- [gdm-doi-10.18653-v1-2025.emnlp-main.52](<https://doi.org/10.18653/v1/2025.emnlp-main.52>) - 2025-01-01 · `affiliated` · On Relation-Specific Neurons in Large Language Models
- [gdm-doi-10.18653-v1-2025.acl-long.822](<https://doi.org/10.18653/v1/2025.acl-long.822>) - 2025-01-01 · `affiliated` · FRACTAL: Fine-Grained Scoring from Aggregate Text Labels
- [gdm-doi-10.1109-taslpro.2025.3594300](<https://doi.org/10.1109/taslpro.2025.3594300>) - 2025-01-01 · `affiliated` · MT2KD: Toward a General-Purpose Encoder for Speech, Speaker, and Audio Events
- [gdm-doi-10.1109-lca.2025.3541961](<https://doi.org/10.1109/lca.2025.3541961>) - 2025-01-01 · `affiliated` · QuArch: A Question-Answering Dataset for AI Agents in Computer Architecture
- [gdm-doi-10.1093-rssdat-udaf001](<https://doi.org/10.1093/rssdat/udaf001>) - 2025-01-01 · `affiliated` · Introducing <i>RSS: Data Science and Artificial Intelligence</i>
- [gdm-doi-10.1016-j.ifacol.2025.09.547](<https://doi.org/10.1016/j.ifacol.2025.09.547>) - 2025-01-01 · `affiliated` · A Simulink-based platform for model-based design and deployment of controllers for tokamak nuclear fusion devices
- [gdm-doi-10.1007-978-3-031-92808-6_5](<https://doi.org/10.1007/978-3-031-92808-6_5>) - 2025-01-01 · `affiliated` · Making Images from Images: Tightly Constrained Parallel Denoising

### 17.3 2024（301 条）

- [gdm-web-862be77cdee8](<https://deepmind.google/research/publications/46840/>) - 2024-12-31 · `affiliated` · Exposing Limitations of Language Model Agents in Sequential-Task Compositions on the Web
- [gdm-arxiv-2412.20203](<http://arxiv.org/abs/2412.20203>) - 2024-12-28 · `affiliated` · No-regret learning in harmonic games: Extrapolation in the face of conflicting interests
- [gdm-doi-10.26434-chemrxiv-2024-1dw4q](<https://doi.org/10.26434/chemrxiv-2024-1dw4q>) - 2024-12-26 · `affiliated` · Inverse Design of Complex Nanoparticle Heterostructures via Deep Learning on Heterogeneous Graphs
- [gdm-arxiv-2412.19010](<https://deepmind.google/research/publications/126226/>) - 2024-12-26 · `affiliated` · A theory of appropriateness with applications to generative artificial intelligence
- [gdm-arxiv-2412.17747](<https://deepmind.google/research/publications/141788/>) - 2024-12-23 · `affiliated` · Deliberation in Latent Space via Differentiable Cache Augmentation
- [gdm-doi-10.7554-elife.97612.2](<https://doi.org/10.7554/elife.97612.2>) - 2024-12-13 · `affiliated` · Dynamic reinforcement learning reveals time-dependent shifts in strategy during reward learning
- [gdm-doi-10.1021-acs.jpclett.4c02376](<https://doi.org/10.1021/acs.jpclett.4c02376>) - 2024-12-13 · `affiliated` · Complete and Efficient Covariants for Three-Dimensional Point Configurations with Application to Learning Molecular Quantum Properties
- [gdm-web-c3ef931eff7e](<https://deepmind.google/research/publications/92499/>) - 2024-12-10 · `affiliated` · What type of inference is planning?
- [gdm-url-a696f52baaf0](<https://hal.science/hal-05413270>) - 2024-12-10 · `affiliated` · Metacognitive capabilities of LLMs: An exploration in mathematical problem solving
- [gdm-arxiv-2412.06966](<https://deepmind.google/research/publications/101479/>) - 2024-12-10 · `affiliated` · Machine Unlearning Doesn’t Do What You Think: Lessons for Generative AI Policy, Research, and Practice
- [gdm-doi-10.1038-s41586-024-08449-y](<https://doi.org/10.1038/s41586-024-08449-y>) - 2024-12-09 · `affiliated` · Quantum error correction below the surface code threshold
- [gdm-doi-10.1007-978-981-96-0901-7_28](<https://doi.org/10.1007/978-981-96-0901-7_28>) - 2024-12-07 · `affiliated` · BootsTAP: Bootstrapped Training for Tracking-Any-Point
- [gdm-arxiv-2412.05196](<https://deepmind.google/research/publications/120972/>) - 2024-12-06 · `affiliated` · Exponential Speedups by Rerooting Levin Tree Search
- [gdm-web-3dec117bddf9](<https://deepmind.google/research/publications/139455/>) - 2024-12-04 · `affiliated` · Mastering Board Games by External and Internal Planning with Language Models
- [gdm-doi-10.1038-s41586-024-08252-9](<https://doi.org/10.1038/s41586-024-08252-9>) - 2024-12-04 · `affiliated` · Probabilistic weather forecasting with machine learning
- [gdm-arxiv-2412.03555](<https://deepmind.google/models/gemma/paligemma-2/>) - 2024-12-04 · `affiliated` · PaliGemma 2: A Family of Versatile VLMs for Transfer
- [gdm-doi-10.1007-978-3-031-72848-8_11](<https://doi.org/10.1007/978-3-031-72848-8_11>) - 2024-11-28 · `affiliated` · MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space
- [gdm-doi-10.1038-s41586-024-08416-7](<https://doi.org/10.1038/s41586-024-08416-7>) - 2024-11-27 · `affiliated` · Addendum: Accurate structure prediction of biomolecular interactions with AlphaFold 3
- [gdm-arxiv-2411.16679](<https://deepmind.google/research/publications/133302/>) - 2024-11-26 · `affiliated` · How Well Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?
- [gdm-arxiv-2411.14708](<https://deepmind.google/research/publications/135718/>) - 2024-11-22 · `affiliated` · Understanding LLM Embeddings for Regression
- [gdm-doi-10.1093-nar-gkae1082](<https://doi.org/10.1093/nar/gkae1082>) - 2024-11-20 · `affiliated` · InterPro: the protein sequence classification resource in 2025
- [gdm-doi-10.1038-s41586-024-08148-8](<https://doi.org/10.1038/s41586-024-08148-8>) - 2024-11-20 · `affiliated` · Learning high-accuracy error decoding for quantum processors
- [gdm-doi-10.1007-978-3-031-73036-8_16](<https://doi.org/10.1007/978-3-031-73036-8_16>) - 2024-11-20 · `affiliated` · Volumetric Rendering with Baked Quadrature Fields
- [gdm-doi-10.1007-978-3-031-73021-4_23](<https://doi.org/10.1007/978-3-031-73021-4_23>) - 2024-11-20 · `affiliated` · UniIR: Training and Benchmarking Universal Multimodal Information Retrievers
- [gdm-doi-10.1093-nar-gkae997](<https://doi.org/10.1093/nar/gkae997>) - 2024-11-14 · `affiliated` · The Pfam protein families database: embracing AI/ML
- [gdm-doi-10.1101-2024.11.11.623030](<https://doi.org/10.1101/2024.11.11.623030>) - 2024-11-12 · `affiliated` · SIMPL: Scalable and hassle-free optimisation of neural representations from behaviour
- [gdm-doi-10.1073-pnas.2408134121](<https://doi.org/10.1073/pnas.2408134121>) - 2024-11-08 · `affiliated` · Decomposing dynamical subprocesses for compositional generalization
- [gdm-doi-10.1038-s41591-024-03302-1](<https://doi.org/10.1038/s41591-024-03302-1>) - 2024-11-07 · `affiliated` · Collaboration between clinicians and vision–language models in radiology report generation
- [gdm-doi-10.1007-978-3-031-73383-3_11](<https://doi.org/10.1007/978-3-031-73383-3_11>) - 2024-11-02 · `affiliated` · Lagrangian Hashing for Compressed Neural Field Representations
- [gdm-doi-10.1111-epic.12194](<https://doi.org/10.1111/epic.12194>) - 2024-11-01 · `affiliated` · AI Mental Models &amp; Trust: The Promises and Perils of Interaction Design
- [gdm-doi-10.1007-978-3-031-73226-3_2](<https://doi.org/10.1007/978-3-031-73226-3_2>) - 2024-10-31 · `affiliated` · Affective Visual Dialog: A Large-Scale Benchmark for Emotional Reasoning Based on Visually Grounded Conversations
- [gdm-doi-10.1007-978-3-031-73039-9_27](<https://doi.org/10.1007/978-3-031-73039-9_27>) - 2024-10-30 · `affiliated` · IG Captioner: Information Gain Captioners Are Strong Zero-Shot Classifiers
- [gdm-doi-10.1007-978-3-031-72907-2_19](<https://doi.org/10.1007/978-3-031-72907-2_19>) - 2024-10-30 · `affiliated` · Learned Neural Physics Simulation for Articulated 3D Human Pose Reconstruction
- [gdm-doi-10.1007-978-3-031-72907-2_12](<https://doi.org/10.1007/978-3-031-72907-2_12>) - 2024-10-30 · `affiliated` · Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection
- [gdm-doi-10.1145-3699690](<https://doi.org/10.1145/3699690>) - 2024-10-29 · `affiliated` · Adventures in AI Wonderland: How Children Are Shaping the Future of AI
- [gdm-doi-10.1145-3678717.3691246](<https://doi.org/10.1145/3678717.3691246>) - 2024-10-29 · `affiliated` · SRL: Towards a General-Purpose Framework for Spatial Representation Learning
- [gdm-doi-10.1109-focs61266.2024.00135](<https://doi.org/10.1109/focs61266.2024.00135>) - 2024-10-27 · `affiliated` · Efficient and Near-Optimal Noise Generation for Streaming Differential Privacy
- [gdm-doi-10.1007-978-3-031-73016-0_19](<https://doi.org/10.1007/978-3-031-73016-0_19>) - 2024-10-25 · `affiliated` · LookupViT: Compressing Visual Information to a Limited Number of Tokens
- [gdm-doi-10.1007-978-3-031-73016-0_16](<https://doi.org/10.1007/978-3-031-73016-0_16>) - 2024-10-25 · `affiliated` · Text-Conditioned Resampler For Long Form Video Understanding
- [gdm-doi-10.1016-j.cognition.2024.105993](<https://doi.org/10.1016/j.cognition.2024.105993>) - 2024-10-24 · `affiliated` · Beyond the matrix: Experimental approaches to studying cognitive agents in social-ecological systems
- [gdm-doi-10.1145-3689904.3694708](<https://doi.org/10.1145/3689904.3694708>) - 2024-10-23 · `affiliated` · The Case for Globalizing Fairness: A Mixed Methods Study on Colonialism, AI, and Health in Africa
- [gdm-doi-10.1038-s41586-024-08025-4](<https://doi.org/10.1038/s41586-024-08025-4>) - 2024-10-23 · `direct` · Scalable watermarking for identifying large language model outputs
- [gdm-doi-10.1371-journal.pcbi.1012383](<https://doi.org/10.1371/journal.pcbi.1012383>) - 2024-10-18 · `affiliated` · Understanding dual process cognition via the minimum description length principle
- [gdm-doi-10.1126-science.adq2852](<https://deepmind.google/research/publications/65220/>) - 2024-10-18 · `affiliated` · AI can help humans find common ground in democratic deliberation
- [gdm-web-e483e3ac8428](<https://deepmind.google/research/publications/90773/>) - 2024-10-17 · `affiliated` · Prompting Considered Harmful
- [gdm-doi-10.3233-faia240938](<https://doi.org/10.3233/faia240938>) - 2024-10-16 · `affiliated` · Translation and Transliteration Based Data Augmentation for Multilingual Semantic Parsing
- [gdm-doi-10.1609-aies.v7i1.31717](<https://doi.org/10.1609/aies.v7i1.31717>) - 2024-10-16 · `affiliated` · Gaps in the Safety Evaluation of Generative AI
- [gdm-doi-10.1609-aies.v7i1.31694](<https://doi.org/10.1609/aies.v7i1.31694>) - 2024-10-16 · `affiliated` · The Code That Binds Us: Navigating the Appropriateness of Human-AI Assistant Relationships
- [gdm-doi-10.1609-aies.v7i1.31678](<https://doi.org/10.1609/aies.v7i1.31678>) - 2024-10-16 · `affiliated` · Responsible Reporting for Frontier AI Development
- [gdm-doi-10.1609-aies.v7i1.31671](<https://doi.org/10.1609/aies.v7i1.31671>) - 2024-10-16 · `affiliated` · Epistemic Injustice in Generative AI
- [gdm-doi-10.1609-aies.v7i1.31637](<http://dx.doi.org/10.1609/aies.v7i1.31637>) - 2024-10-16 · `affiliated` · Beyond Thumbs Up/Down: Untangling Challenges of Fine-Grained Feedback for Text-to-Image Generation
- [gdm-doi-10.1609-aies.v7i1.31613](<https://doi.org/10.1609/aies.v7i1.31613>) - 2024-10-16 · `affiliated` · All Too Human? Mapping and Mitigating the Risk from Anthropomorphic AI
- [gdm-doi-10.1021-jacs.4c10294](<https://doi.org/10.1021/jacs.4c10294>) - 2024-10-16 · `affiliated` · Efficient Exploratory Synthesis of Quaternary Cesium Chlorides Guided by In Silico Predictions
- [gdm-doi-10.1109-iros58592.2024.10802716](<https://doi.org/10.1109/iros58592.2024.10802716>) - 2024-10-14 · `affiliated` · CoNVOI: Context-aware Navigation using Vision Language Models in Outdoor and Indoor Environments
- [gdm-doi-10.1109-iros58592.2024.10801714](<https://doi.org/10.1109/iros58592.2024.10801714>) - 2024-10-14 · `affiliated` · Versatile Locomotion Skills for Hexapod Robots
- [gdm-doi-10.1109-iros58592.2024.10801525](<https://doi.org/10.1109/iros58592.2024.10801525>) - 2024-10-14 · `affiliated` · GenCHiP: Generating Robot Policy Code for High-Precision and Contact-Rich Manipulation Tasks
- [gdm-doi-10.1109-iros58592.2024.10801377](<https://doi.org/10.1109/iros58592.2024.10801377>) - 2024-10-14 · `affiliated` · The Design of the Barkour Benchmark for Robot Agility
- [gdm-arxiv-2410.10190](<https://deepmind.google/research/publications/122292/>) - 2024-10-14 · `affiliated` · Predicting from Strings: Language Model Embeddings for Bayesian Optimization
- [gdm-doi-10.1007-s10458-024-09675-4](<https://doi.org/10.1007/s10458-024-09675-4>) - 2024-10-12 · `affiliated` · Resolving social dilemmas with minimal reward transfer
- [gdm-arxiv-2410.05364](<https://deepmind.google/research/publications/93097/>) - 2024-10-09 · `affiliated` · Diffusion model predictive control
- [gdm-doi-10.1007-s10462-024-10931-y](<https://doi.org/10.1007/s10462-024-10931-y>) - 2024-10-04 · `affiliated` · A review of graph neural network applications in mechanics-related domains
- [gdm-arxiv-2410.03011](<http://arxiv.org/abs/2410.03011>) - 2024-10-03 · `affiliated` · Towards Understanding the Universality of Transformers for Next-Token Prediction
- [gdm-doi-10.2172-2475542](<https://doi.org/10.2172/2475542>) - 2024-10-01 · `affiliated` · Safety in Artificial Intelligence: Challenges and Opportunities for the U.S. National Labs and Beyond
- [gdm-doi-10.1007-978-3-031-72952-2_4](<https://doi.org/10.1007/978-3-031-72952-2_4>) - 2024-09-30 · `affiliated` · TC4D: Trajectory-Conditioned Text-to-4D Generation
- [gdm-doi-10.1007-978-3-031-72920-1_26](<https://doi.org/10.1007/978-3-031-72920-1_26>) - 2024-09-30 · `affiliated` · Parrot: Pareto-Optimal Multi-reward Reinforcement Learning Framework for Text-to-Image Generation
- [gdm-doi-10.1007-978-3-031-72920-1_13](<https://doi.org/10.1007/978-3-031-72920-1_13>) - 2024-09-30 · `affiliated` · PointNeRF++: A Multi-scale, Point-Based Neural Radiance Field
- [gdm-doi-10.1007-978-3-031-73235-5_25](<https://doi.org/10.1007/978-3-031-73235-5_25>) - 2024-09-29 · `affiliated` · ViC-MAE: Self-supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders
- [gdm-doi-10.1007-978-3-031-73232-4_22](<https://doi.org/10.1007/978-3-031-73232-4_22>) - 2024-09-29 · `affiliated` · 3D Congealing: 3D-Aware Image Alignment in the Wild
- [gdm-web-8856adfb18eb](<https://deepmind.google/research/publications/92798/>) - 2024-09-26 · `affiliated` · Geometric-Averaged Preference Optimization for Soft Preference Labels
- [gdm-doi-10.1371-journal.pcbi.1012436](<https://doi.org/10.1371/journal.pcbi.1012436>) - 2024-09-26 · `affiliated` · The refinement paradox and cumulative cultural evolution: Complex products of collective improvement favor conformist outcomes, blind copying, and hyper-credulity
- [gdm-web-964bcd5da6cd](<https://deepmind.google/research/publications/90369/>) - 2024-09-21 · `affiliated` · Learned feature representations are biased by complexity, learning order, position, and more
- [gdm-doi-10.1101-2024.09.20.614119](<https://doi.org/10.1101/2024.09.20.614119>) - 2024-09-20 · `affiliated` · A neural mechanism for compositional generalization of structure in humans
- [gdm-doi-10.1038-s41562-024-01959-9](<https://doi.org/10.1038/s41562-024-01959-9>) - 2024-09-20 · `affiliated` · How large language models can reshape collective intelligence
- [gdm-arxiv-2409.12640](<https://deepmind.google/research/publications/117639/>) - 2024-09-19 · `affiliated` · Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries
- [gdm-doi-10.1167-jov.24.10.132](<http://dx.doi.org/10.1167/jov.24.10.132>) - 2024-09-15 · `affiliated` · Learning to discriminate by learning to generate: zero-shot generative models increase human object recognition alignment
- [gdm-arxiv-2409.13741](<https://deepmind.google/models/gemma/datagemma/>) - 2024-09-10 · `affiliated` · Knowing When to Ask -- Bridging Large Language Models and Data
- [gdm-doi-10.3390-e26090776](<https://deepmind.google/research/publications/98247/>) - 2024-09-07 · `affiliated` · Modeling the Arrows of Time with Causal Multibaker Maps
- [gdm-doi-10.1016-j.physd.2024.134335](<http://dx.doi.org/10.1016/j.physd.2024.134335>) - 2024-09-07 · `affiliated` · Hierarchical Partitioning Forecaster
- [gdm-arxiv-2409.03875](<http://arxiv.org/abs/2409.03875>) - 2024-09-05 · `affiliated` · Learning in Games with Progressive Hiding
- [gdm-doi-10.1080-01621459.2024.2393466](<https://doi.org/10.1080/01621459.2024.2393466>) - 2024-09-03 · `affiliated` · Evaluating Treatment Prioritization Rules via Rank-Weighted Average Treatment Effects
- [gdm-doi-10.1109-tts.2024.3446183](<https://doi.org/10.1109/tts.2024.3446183>) - 2024-09-01 · `affiliated` · Human Participants in AI Research: Ethics and Transparency in Practice
- [gdm-doi-10.1017-s0960129524000276](<https://doi.org/10.1017/s0960129524000276>) - 2024-09-01 · `affiliated` · Optimal approximate minimization of one-letter weighted finite automata
- [gdm-arxiv-2111.15323](<http://dx.doi.org/10.2140/gt.2024.28.2313>) - 2024-08-24 · `affiliated` · The signature and cusp geometry of hyperbolic knots
- [gdm-doi-10.1126-science.adn0137](<https://doi.org/10.1126/science.adn0137>) - 2024-08-22 · `affiliated` · Accurate computation of quantum excited states with neural networks
- [gdm-doi-10.1038-d41586-024-02525-z](<http://dx.doi.org/10.1038/d41586-024-02525-z>) - 2024-08-21 · `affiliated` · Switching between tasks can cause AI to lose the ability to learn
- [gdm-arxiv-2408.11527](<https://deepmind.google/research/publications/108347/>) - 2024-08-21 · `affiliated` · The Vizier Gaussian Process Bandit Algorithm
- [gdm-arxiv-2408.11146](<https://deepmind.google/research/publications/107338/>) - 2024-08-20 · `affiliated` · Swim till you sink: Computing the limit of a game
- [gdm-web-ecfbaabeefce](<https://deepmind.google/research/publications/78755/>) - 2024-08-13 · `affiliated` · The Probabilities Also Matter: A More Faithful Metric for Faithfulness of Free-Text Explanations in Large Language Models
- [gdm-doi-10.1073-pnas.2315002121](<https://doi.org/10.1073/pnas.2315002121>) - 2024-08-12 · `affiliated` · AlphaFold two years on: Validation and impact
- [gdm-arxiv-2408.03906](<https://deepmind.google/research/publications/107741/>) - 2024-08-07 · `affiliated` · Achieving Human Level Competitive Robot Table Tennis
- [gdm-doi-10.1093-isq-sqae111](<https://doi.org/10.1093/isq/sqae111>) - 2024-08-05 · `affiliated` · Anarchy as Architect: Competitive Pressure, Technology, and the Internal Structure of States
- [gdm-doi-10.14778-3685800.3685833](<https://doi.org/10.14778/3685800.3685833>) - 2024-08-01 · `affiliated` · Differentially Private Stream Processing at Scale
- [gdm-web-c2ade0591f5b](<https://deepmind.google/research/publications/70169/>) - 2024-07-31 · `affiliated` · Pre-trained Gaussian processes for Bayesian optimization
- [gdm-url-4fce1bd2d654](<https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf>) - 2024-07-31 · `affiliated` · Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2
- [gdm-doi-10.1101-2024.07.30.605655](<https://doi.org/10.1101/2024.07.30.605655>) - 2024-07-31 · `affiliated` · Video Foundation Models for Animal Behavior Analysis
- [gdm-doi-10.1093-scipol-scae043](<https://doi.org/10.1093/scipol/scae043>) - 2024-07-30 · `affiliated` · Risk-sensitive innovation: leveraging interactions between technologies to navigate technology risks
- [gdm-arxiv-2407.19985](<https://deepmind.google/research/publications/108549/>) - 2024-07-30 · `affiliated` · Mixture of Nested Experts: Adaptive Processing of Visual Tokens
- [gdm-doi-10.24963-ijcai.2024-772](<https://doi.org/10.24963/ijcai.2024/772>) - 2024-07-26 · `affiliated` · Collaborative Multi-LoRA Experts with Achievement-based Multi-Tasks Loss for Unified Multimodal Information Extraction
- [gdm-doi-10.24963-ijcai.2024-339](<https://doi.org/10.24963/ijcai.2024/339>) - 2024-07-26 · `affiliated` · App2Exa: Accelerating Exact kNN Search via Dynamic Cache-Guided Approximation
- [gdm-doi-10.24963-ijcai.2024-19](<https://doi.org/10.24963/ijcai.2024/19>) - 2024-07-26 · `affiliated` · Combining Deep Reinforcement Learning and Search with Generative Models for Game-Theoretic Opponent Modeling
- [gdm-doi-10.24963-ijcai.2024-17](<https://doi.org/10.24963/ijcai.2024/17>) - 2024-07-26 · `affiliated` · Approximate Verification of Strategic Abilities under Imperfect Information Using Local Models
- [gdm-doi-10.1038-s41586-024-07744-y](<https://doi.org/10.1038/s41586-024-07744-y>) - 2024-07-22 · `affiliated` · Neural general circulation models for weather and climate
- [gdm-arxiv-2311.02462](<https://deepmind.google/research/publications/66938/>) - 2024-07-21 · `affiliated` · Levels of AGI for Operationalizing Progress on the Path to AGI
- [gdm-doi-10.1038-s41593-024-01713-4](<https://doi.org/10.1038/s41593-024-01713-4>) - 2024-07-19 · `affiliated` · Higher-order interactions between hippocampal CA1 neurons are disrupted in amnestic mice
- [gdm-doi-10.1093-pnasnexus-pgae233](<https://deepmind.google/research/publications/9266/>) - 2024-07-16 · `affiliated` · Language models, like humans, show content effects on reasoning tasks
- [gdm-arxiv-2407.11666](<https://deepmind.google/research/publications/105317/>) - 2024-07-16 · `affiliated` · Neural Climate Data Compression
- [gdm-doi-10.1145-3641519.3657435](<https://doi.org/10.1145/3641519.3657435>) - 2024-07-12 · `affiliated` · Blue noise for diffusion models
- [gdm-doi-10.1038-s42003-024-06465-2](<https://doi.org/10.1038/s42003-024-06465-2>) - 2024-07-09 · `affiliated` · A foundational large language model for edible plant genomes
- [gdm-doi-10.1145-3670865.3673551](<https://doi.org/10.1145/3670865.3673551>) - 2024-07-08 · `affiliated` · Complex Dynamics in Autobidding Systems
- [gdm-arxiv-2407.06076](<http://arxiv.org/abs/2407.06076>) - 2024-07-08 · `affiliated` · Understanding Visual Feature Reliance through the Lens of Complexity
- [gdm-doi-10.1109-isit57864.2024.10619428](<https://doi.org/10.1109/isit57864.2024.10619428>) - 2024-07-07 · `affiliated` · Gaussian Channel Simulation with Rotated Dithered Quantization
- [gdm-doi-10.1109-igarss53475.2024.10641578](<https://doi.org/10.1109/igarss53475.2024.10641578>) - 2024-07-07 · `affiliated` · PLANTED: A Dataset for Planted Forest Identification from Multi-Satellite Time Series
- [gdm-doi-10.3389-fbirs.2024.1369756](<https://doi.org/10.3389/fbirs.2024.1369756>) - 2024-07-01 · `affiliated` · Birds, bats and beyond: evaluating generalization in bioacoustics models
- [gdm-doi-10.1093-bioinformatics-btae405](<https://doi.org/10.1093/bioinformatics/btae405>) - 2024-07-01 · `affiliated` · Geometric epitope and paratope prediction
- [gdm-doi-10.1063-5.0201401](<https://doi.org/10.1063/5.0201401>) - 2024-07-01 · `affiliated` · X-point radiator and power exhaust control in configurations with multiple X-points in TCV
- [gdm-doi-10.1145-3643834.3660692](<https://doi.org/10.1145/3643834.3660692>) - 2024-06-29 · `affiliated` · Do LLMs Meet the Needs of Software Tutorial Writers? Opportunities and Design Implications
- [gdm-doi-10.1109-isca59077.2024.00093](<https://doi.org/10.1109/isca59077.2024.00093>) - 2024-06-29 · `affiliated` · DACAPO: Accelerating Continuous Learning in Autonomous Systems for Video Analytics
- [gdm-doi-10.1177-10597123241262534](<https://doi.org/10.1177/10597123241262534>) - 2024-06-21 · `affiliated` · The origin and function of external representations
- [gdm-doi-10.1038-s41562-024-01878-9](<https://doi.org/10.1038/s41562-024-01878-9>) - 2024-06-21 · `affiliated` · Using games to understand the mind
- [gdm-arxiv-2406.13843](<https://deepmind.google/blog/mapping-the-misuse-of-generative-ai/>) - 2024-06-19 · `affiliated` · Generative AI Misuse: A Taxonomy of Tactics and Insights from Real-World Data
- [gdm-doi-10.1038-s41467-024-49290-1](<https://doi.org/10.1038/s41467-024-49290-1>) - 2024-06-18 · `affiliated` · Neural network variational Monte Carlo for positronic chemistry
- [gdm-web-3555e192354b](<https://deepmind.google/research/publications/60877/>) - 2024-06-17 · `affiliated` · Bayes' Rays: uncertainty quantification for neural radiance fields
- [gdm-doi-10.1167-jov.24.6.12](<https://doi.org/10.1167/jov.24.6.12>) - 2024-06-17 · `affiliated` · How does V1 population activity inform perceptual certainty?
- [gdm-doi-10.1109-cvpr52733.2024.00409](<https://deepmind.google/research/publications/61382/>) - 2024-06-17 · `affiliated` · Neural Fields as Distributions: Signal Processing Beyond Euclidean Space
- [gdm-arxiv-2406.11409](<https://deepmind.google/models/gemma/codegemma/>) - 2024-06-17 · `affiliated` · CodeGemma: Open Code Models Based on Gemma
- [gdm-arxiv-2311.05698](<https://deepmind.google/research/publications/50070/>) - 2024-06-17 · `affiliated` · Mirasol3B: A Multimodal Autoregressive Model for Time-Aligned and Contextual Modalities
- [gdm-doi-10.1109-cvpr52733.2024.02153](<https://doi.org/10.1109/cvpr52733.2024.02153>) - 2024-06-16 · `affiliated` · Unsupervised Keypoints from Pretrained Diffusion Models
- [gdm-doi-10.1109-cvpr52733.2024.02036](<https://doi.org/10.1109/cvpr52733.2024.02036>) - 2024-06-16 · `affiliated` · ReconFusion: 3D Reconstruction with Diffusion Priors
- [gdm-doi-10.1109-cvpr52733.2024.01897](<https://doi.org/10.1109/cvpr52733.2024.01897>) - 2024-06-16 · `affiliated` · Accelerating Neural Field Training via Soft Mining
- [gdm-doi-10.1109-cvpr52733.2024.01519](<https://doi.org/10.1109/cvpr52733.2024.01519>) - 2024-06-16 · `affiliated` · Frozen Feature Augmentation for Few-Shot Image Classification
- [gdm-doi-10.1109-cvpr52733.2024.01370](<https://doi.org/10.1109/cvpr52733.2024.01370>) - 2024-06-16 · `affiliated` · SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
- [gdm-doi-10.1109-cvpr52733.2024.01281](<https://doi.org/10.1109/cvpr52733.2024.01281>) - 2024-06-16 · `affiliated` · De-Diffusion Makes Text a Strong Cross-Modal Interface
- [gdm-doi-10.1109-cvpr52733.2024.00905](<https://doi.org/10.1109/cvpr52733.2024.00905>) - 2024-06-16 · `affiliated` · Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion
- [gdm-doi-10.1109-cvpr52733.2024.00893](<https://doi.org/10.1109/cvpr52733.2024.00893>) - 2024-06-16 · `affiliated` · C3: High-Performance and Low-Complexity Neural Compression from a Single Image or Video
- [gdm-doi-10.1109-cvpr52733.2024.00701](<https://doi.org/10.1109/cvpr52733.2024.00701>) - 2024-06-16 · `affiliated` · Video Interpolation with Diffusion Models
- [gdm-doi-10.1109-cvpr52733.2024.00455](<https://doi.org/10.1109/cvpr52733.2024.00455>) - 2024-06-16 · `affiliated` · Instruct-Imagen: Image Generation with Multi-modal Instruction
- [gdm-doi-10.1038-s41586-024-07633-4](<https://doi.org/10.1038/s41586-024-07633-4>) - 2024-06-11 · `affiliated` · A virtual rodent predicts the structure of neural activity across behaviours
- [gdm-arxiv-2406.06316](<https://deepmind.google/research/publications/88248/>) - 2024-06-10 · `affiliated` · Tx-LLM: A Large Language Model for Therapeutics
- [gdm-web-0cc1d8b1791c](<https://deepmind.google/research/publications/49869/>) - 2024-06-07 · `affiliated` · Don't trust your eyes: on the (un)reliability of feature visualizations
- [gdm-doi-10.1145-3630106.3658993](<https://deepmind.google/research/publications/70876/>) - 2024-06-05 · `affiliated` · A Robot Walks into a Bar: Can Language Models Serve as Creativity Support Tools for Comedy? An Evaluation of LLMs' Humour Alignment with Comedians
- [gdm-doi-10.1016-j.tics.2024.05.001](<https://doi.org/10.1016/j.tics.2024.05.001>) - 2024-06-05 · `affiliated` · Helpless infants are learning a foundation model
- [gdm-doi-10.1145-3630106.3659007](<https://doi.org/10.1145/3630106.3659007>) - 2024-06-03 · `affiliated` · Beyond Use-Cases: A Participatory Approach to Envisioning Data Science in Law Enforcement
- [gdm-doi-10.1145-3630106.3658964](<https://doi.org/10.1145/3630106.3658964>) - 2024-06-03 · `affiliated` · Should Users Trust Advanced AI Assistants? Justified Trust As a Function of Competence and Alignment
- [gdm-doi-10.1101-2024.05.30.596539](<https://doi.org/10.1101/2024.05.30.596539>) - 2024-06-02 · `affiliated` · ProtEx: A Retrieval-Augmented Approach for Protein Function Prediction
- [gdm-doi-10.1029-2023ms004019](<https://doi.org/10.1029/2023ms004019>) - 2024-06-01 · `affiliated` · WeatherBench 2: A Benchmark for the Next Generation of Data‐Driven Global Weather Models
- [gdm-web-c392b4ed4d6e](<https://deepmind.google/research/publications/33304/>) - 2024-05-28 · `affiliated` · An Introduction to Universal Artificial Intelligence
- [gdm-arxiv-2405.16021](<http://arxiv.org/abs/2405.16021>) - 2024-05-25 · `affiliated` · VADER: Visual Affordance Detection and Error Recovery for Multi Robot Human Collaboration
- [gdm-arxiv-2401.10874](<https://doi.org/10.1103/physrevd.109.094514>) - 2024-05-24 · `affiliated` · Applications of flow models to the generation of correlated lattice QCD ensembles
- [gdm-arxiv-2305.06989](<https://doi.org/10.1103/physrevx.14.021030>) - 2024-05-22 · `affiliated` · Neural Wave Functions for Superfluids
- [gdm-arxiv-2201.13448](<https://doi.org/10.1007/s10458-024-09649-6>) - 2024-05-22 · `affiliated` · Warmth and competence in human-agent cooperation
- [gdm-doi-10.1109-sp54263.2024.00179](<https://doi.org/10.1109/sp54263.2024.00179>) - 2024-05-19 · `affiliated` · Poisoning Web-Scale Training Datasets is Practical
- [gdm-url-25f473162418](<https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/introducing-the-frontier-safety-framework/fsf-technical-report.pdf>) - 2024-05-17 · `affiliated` · Frontier Safety Framework, Version 1.0
- [gdm-doi-10.1090-bull-1843](<https://doi.org/10.1090/bull/1843>) - 2024-05-15 · `affiliated` · Working with machines in mathematics
- [gdm-doi-10.1109-icra57147.2024.10611575](<https://doi.org/10.1109/icra57147.2024.10611575>) - 2024-05-13 · `affiliated` · Robotic Offline RL from Internet Videos via Value-Function Learning
- [gdm-doi-10.1109-icra57147.2024.10611477](<https://doi.org/10.1109/icra57147.2024.10611477>) - 2024-05-13 · `affiliated` · Open X-Embodiment: Robotic Learning Datasets and RT-X Models : Open X-Embodiment Collaboration<sup>0</sup>
- [gdm-doi-10.1109-icra57147.2024.10611331](<https://doi.org/10.1109/icra57147.2024.10611331>) - 2024-05-13 · `affiliated` · Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation
- [gdm-doi-10.1109-icra57147.2024.10611275](<http://dx.doi.org/10.1109/icra57147.2024.10611275>) - 2024-05-13 · `affiliated` · Conditionally Combining Robot Skills using Large Language Models
- [gdm-doi-10.1109-icra57147.2024.10610784](<https://doi.org/10.1109/icra57147.2024.10610784>) - 2024-05-13 · `affiliated` · How to Prompt Your Robot: A PromptBook for Manipulation Skills with Code as Policies
- [gdm-doi-10.1109-icra57147.2024.10610297](<http://dx.doi.org/10.1109/icra57147.2024.10610297>) - 2024-05-13 · `affiliated` · Mastering Stacking of Diverse Shapes with Large-Scale Iterative Reinforcement Learning on Real Robots
- [gdm-doi-10.1109-icra57147.2024.10610090](<https://doi.org/10.1109/icra57147.2024.10610090>) - 2024-05-13 · `affiliated` · Physically Grounded Vision-Language Models for Robotic Manipulation
- [gdm-arxiv-2403.14117](<https://doi.org/10.1145/3613904.3642697>) - 2024-05-11 · `affiliated` · A Design Space for Intelligent and Interactive Writing Assistants
- [gdm-arxiv-2401.08572](<https://doi.org/10.1145/3613904.3642703>) - 2024-05-11 · `affiliated` · The Illusion of Artificial Inclusion
- [gdm-doi-10.1038-s41586-024-07487-w](<https://doi.org/10.1038/s41586-024-07487-w>) - 2024-05-08 · `affiliated` · Accurate structure prediction of biomolecular interactions with AlphaFold 3
- [gdm-arxiv-2405.04407](<https://deepmind.google/research/publications/90066/>) - 2024-05-08 · `affiliated` · Super-Exponential Regret for UCT, AlphaGo and Variants
- [gdm-web-d1acf4b4a8aa](<https://deepmind.google/research/publications/50878/>) - 2024-05-07 · `affiliated` · Learning 3D Particle-based Simulators from RGB-D Videos
- [gdm-web-57798b49e69b](<https://deepmind.google/research/publications/33405/>) - 2024-05-07 · `affiliated` · Kalman Filter for Online Classification of Non-Stationary Data
- [gdm-web-44d27be06a2c](<https://deepmind.google/research/publications/36940/>) - 2024-05-07 · `affiliated` · Deep SE(3)-Equivariant Geometric Reasoning for Precise Placement Tasks
- [gdm-web-18c624f4872e](<https://deepmind.google/research/publications/25628/>) - 2024-05-07 · `affiliated` · π2vec: Policy Representations with SuccessorFeatures
- [gdm-web-1490c2840a0a](<https://deepmind.google/research/publications/43000/>) - 2024-05-07 · `affiliated` · Teach LLMs to Phish: Stealing Private Information from Language Models
- [gdm-web-1212b63f27a9](<https://deepmind.google/research/publications/39768/>) - 2024-05-07 · `affiliated` · Language Modeling Is Compression
- [gdm-doi-10.5281-zenodo.10991321](<https://deepmind.google/research/publications/81077/>) - 2024-05-07 · `affiliated` · Adaptive Hashing: Faster Hash Functions Perhaps with Fewer Collisions
- [gdm-arxiv-2310.15526](<https://deepmind.google/research/publications/42798/>) - 2024-05-07 · `affiliated` · Privacy Amplification by Sampling for the Matrix Mechanism.
- [gdm-arxiv-2310.06771](<https://deepmind.google/research/publications/50273/>) - 2024-05-07 · `affiliated` · CORRELATED NOISE PROVABLY BEATS INDEPENDENT NOISE FOR DIFFERENTIALLY PRIVATE LEARNING
- [gdm-arxiv-2308.00951](<https://deepmind.google/research/publications/49566/>) - 2024-05-07 · `affiliated` · From Sparse to Soft Mixture of Experts
- [gdm-web-ed11e4a45f3b](<https://deepmind.google/research/publications/49061/>) - 2024-05-06 · `affiliated` · ExeDec: Execution Decomposition for Compositional Generalization in Neural Program Synthesis
- [gdm-doi-10.65109-wpam8724](<https://doi.org/10.65109/wpam8724>) - 2024-05-06 · `affiliated` · Pruning Neural Networks Using Cooperative Game Theory
- [gdm-doi-10.65109-vzce5163](<https://doi.org/10.65109/vzce5163>) - 2024-05-06 · `affiliated` · The Reasons that Agents Act: Intention and Instrumental Goals
- [gdm-doi-10.65109-tpmy7673](<https://doi.org/10.65109/tpmy7673>) - 2024-05-06 · `affiliated` · Scaling Opponent Shaping to High Dimensional Games
- [gdm-doi-10.65109-fkii5381](<https://doi.org/10.65109/fkii5381>) - 2024-05-06 · `affiliated` · Approximating the Core via Iterative Coalition Sampling
- [gdm-arxiv-2405.03689](<https://deepmind.google/research/publications/164806/>) - 2024-05-06 · `affiliated` · Pose Priors from Language Models
- [gdm-arxiv-2405.03547](<https://deepmind.google/research/publications/77643/>) - 2024-05-06 · `affiliated` · Position: Leverage Foundational Models for Black-Box Optimization
- [gdm-arxiv-2405.03162](<https://deepmind.google/research/publications/87645/>) - 2024-05-06 · `direct` · Advancing Biomedical Understanding with Multimodal Gemini
- [gdm-url-20e08cf23950](<https://hal.science/hal-05413277>) - 2024-05-02 · `affiliated` · A general theoretical paradigm to understand learning from human preferences
- [gdm-arxiv-2403.14467](<https://doi.org/10.1145/3613905.3650999>) - 2024-05-02 · `affiliated` · Recourse for Reclamation: Chatting with Generative Language Models
- [gdm-arxiv-2404.16014](<https://deepmind.google/research/publications/88147/>) - 2024-04-25 · `affiliated` · Improving Dictionary Learning with Gated Sparse Autoencoders
- [gdm-arxiv-2404.16244](<https://deepmind.google/blog/the-ethics-of-advanced-ai-assistants/>) - 2024-04-24 · `affiliated` · The Ethics of Advanced AI Assistants
- [gdm-arxiv-2404.14068](<https://deepmind.google/research/publications/78149/>) - 2024-04-22 · `direct` · Holistic Safety and Responsibility Evaluations of Advanced AI Models
- [gdm-doi-10.5281-zenodo.18504270](<https://doi.org/10.5281/zenodo.18504270>) - 2024-04-18 · `affiliated` · Adaptive Hashing
- [gdm-arxiv-2404.11018](<https://deepmind.google/research/publications/88349/>) - 2024-04-17 · `affiliated` · Many-Shot In-Context Learning
- [gdm-doi-10.1126-scirobotics.adi8022](<https://deepmind.google/research/publications/31284/>) - 2024-04-10 · `affiliated` · Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning
- [gdm-url-1334f9ac0ca9](<https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf>) - 2024-04-09 · `affiliated` · RecurrentGemma: Moving Past Transformers for Efficient Open Language Models
- [gdm-doi-10.21203-rs.3.rs-3916561-v1](<https://doi.org/10.21203/rs.3.rs-3916561/v1>) - 2024-04-09 · `affiliated` · Genetic determinants of centenarian longevity, as quantified by the 'CentPGS' score, are associated with a lower risk of multiple age-related diseases and a longer healthspan.
- [gdm-doi-10.1109-satml59370.2024.00027](<https://doi.org/10.1109/satml59370.2024.00027>) - 2024-04-09 · `affiliated` · Evading Black-box Classifiers Without Breaking Eggs
- [gdm-doi-10.1016-j.physd.2024.134153](<https://doi.org/10.1016/j.physd.2024.134153>) - 2024-04-06 · `affiliated` · Bridging Algorithmic Information Theory and Machine Learning: A new approach to kernel learning
- [gdm-doi-10.1126-sciadv.adn4397](<https://deepmind.google/research/publications/88551/>) - 2024-04-05 · `affiliated` · Biomolecular dynamics with machine-learned quantum-mechanical force fields trained on diverse chemical fragments
- [gdm-doi-10.21203-rs.3.rs-3913308-v1](<https://doi.org/10.21203/rs.3.rs-3913308/v1>) - 2024-04-03 · `affiliated` · Multimodality and Attention Increase Alignment in NaturalLanguage Prediction Between Humans and ComputationalModels
- [gdm-doi-10.1137-22m1532548](<https://doi.org/10.1137/22m1532548>) - 2024-04-01 · `affiliated` · Accelerated Forward-Backward Optimization Using Deep Learning
- [gdm-doi-10.1038-s41591-024-02838-6](<https://doi.org/10.1038/s41591-024-02838-6>) - 2024-04-01 · `affiliated` · Generative models improve fairness of medical classifiers under distribution shifts
- [gdm-arxiv-2404.00411](<http://arxiv.org/abs/2404.00411>) - 2024-03-30 · `affiliated` · Aardvark weather: end-to-end data-driven weather forecasting
- [gdm-arxiv-2403.20327](<https://deepmind.google/research/publications/85521/>) - 2024-03-29 · `affiliated` · Gecko: Versatile Text Embeddings Distilled from Large Language Models
- [gdm-arxiv-2312.00598](<https://deepmind.google/research/publications/59160/>) - 2024-03-28 · `affiliated` · Learning from One Continuous Video Stream
- [gdm-arxiv-2403.18802](<https://deepmind.google/research/publications/85420/>) - 2024-03-27 · `direct` · Long-form factuality in large language models
- [gdm-arxiv-2403.18286](<https://deepmind.google/research/publications/47848/>) - 2024-03-27 · `affiliated` · Few-Shot Recalibration of Language Models
- [gdm-doi-10.1145-3652591](<https://doi.org/10.1145/3652591>) - 2024-03-26 · `affiliated` · Optimising Human-Machine Collaboration for Efficient High-Precision Information Extraction from Text Documents
- [gdm-doi-10.1101-2024.03.22.586239](<http://dx.doi.org/10.1101/2024.03.22.586239>) - 2024-03-25 · `affiliated` · The refinement paradox and cumulative cultural evolution: collective improvement in knowledge favors conformity, blind copying and hyper-credulity
- [gdm-doi-10.1609-aaai.v38i9.28818](<https://doi.org/10.1609/aaai.v38i9.28818>) - 2024-03-24 · `affiliated` · Learning Discrete-Time Major-Minor Mean Field Games
- [gdm-doi-10.1609-aaai.v38i5.28299](<https://doi.org/10.1609/aaai.v38i5.28299>) - 2024-03-24 · `affiliated` · V2Meow: Meowing to the Visual Beat via Video-to-Music Generation
- [gdm-doi-10.1609-aaai.v38i20.30601](<http://dx.doi.org/10.1609/aaai.v38i20.30601>) - 2024-03-24 · `affiliated` · Discovering Agents (Abstract Reprint)
- [gdm-doi-10.1609-aaai.v38i20.30597](<http://dx.doi.org/10.1609/aaai.v38i20.30597>) - 2024-03-24 · `affiliated` · Reasoning about Causality in Games (Abstract Reprint)
- [gdm-doi-10.1609-aaai.v38i15.29575](<http://dx.doi.org/10.1609/aaai.v38i15.29575>) - 2024-03-24 · `affiliated` · Dynamic Knowledge Injection for AIXI Agents
- [gdm-doi-10.1609-aaai.v38i14.29443](<http://dx.doi.org/10.1609/aaai.v38i14.29443>) - 2024-03-24 · `affiliated` · Learning Not to Regret
- [gdm-doi-10.1609-aaai.v38i12.29241](<http://dx.doi.org/10.1609/aaai.v38i12.29241>) - 2024-03-24 · `affiliated` · Learning Uncertainty-Aware Temporally-Extended Actions
- [gdm-doi-10.1609-aaai.v38i10.28980](<http://dx.doi.org/10.1609/aaai.v38i10.28980>) - 2024-03-24 · `affiliated` · Scores for Learning Discrete Causal Graphs with Unobserved Confounders
- [gdm-doi-10.1609-aaai.v38i10.28957](<http://dx.doi.org/10.1609/aaai.v38i10.28957>) - 2024-03-24 · `affiliated` · Optimal Transport with Tempered Exponential Measures
- [gdm-arxiv-2403.13793](<https://deepmind.google/research/publications/78150/>) - 2024-03-21 · `direct` · Evaluating Frontier Models for Dangerous Capabilities
- [gdm-doi-10.1038-s41598-024-56648-4](<https://doi.org/10.1038/s41598-024-56648-4>) - 2024-03-19 · `affiliated` · STELA: a community-centred approach to norm elicitation for AI alignment
- [gdm-doi-10.1038-s41467-024-45965-x](<https://doi.org/10.1038/s41467-024-45965-x>) - 2024-03-19 · `affiliated` · TacticAI: an AI assistant for football tactics
- [gdm-arxiv-2403.10616](<https://deepmind.google/research/publications/84915/>) - 2024-03-19 · `affiliated` · DiPaCo: Distributed Path Composition
- [gdm-doi-10.1109-icassp48485.2024.10448292](<https://doi.org/10.1109/icassp48485.2024.10448292>) - 2024-03-18 · `affiliated` · CryCeleb: A Speaker Verification Dataset Based on Infant Cry Sounds
- [gdm-doi-10.1109-icassp48485.2024.10448217](<https://doi.org/10.1109/icassp48485.2024.10448217>) - 2024-03-18 · `affiliated` · USM-Lite: Quantization and Sparsity Aware Fine-Tuning for Speech Recognition with Universal Speech Models
- [gdm-doi-10.1109-icassp48485.2024.10447448](<https://doi.org/10.1109/icassp48485.2024.10447448>) - 2024-03-18 · `affiliated` · Retrieval Augmented End-to-End Spoken Dialog Models
- [gdm-doi-10.1101-2024.03.11.584515](<https://doi.org/10.1101/2024.03.11.584515>) - 2024-03-14 · `affiliated` · Whole-body simulation of realistic fruit fly locomotion with deep reinforcement learning
- [gdm-doi-10.1109-ciss59072.2024.10480168](<https://doi.org/10.1109/ciss59072.2024.10480168>) - 2024-03-13 · `affiliated` · Wasserstein Distortion: Unifying Fidelity and Realism
- [gdm-arxiv-2404.10179](<https://deepmind.google/blog/sima-generalist-ai-agent-for-3d-virtual-environments/>) - 2024-03-13 · `affiliated` · Scaling Instructable Agents Across Many Simulated Worlds
- [gdm-doi-10.4171-jems-1442](<https://doi.org/10.4171/jems/1442>) - 2024-03-12 · `affiliated` · An asymmetric container lemma and the structure of graphs with no induced &#36;4&#36;-cycle
- [gdm-doi-10.1371-journal.pone.0300024](<https://doi.org/10.1371/journal.pone.0300024>) - 2024-03-12 · `affiliated` · Framework-based qualitative analysis of free responses of Large Language Models: Algorithmic fidelity
- [gdm-doi-10.1145-3639304](<https://doi.org/10.1145/3639304>) - 2024-03-12 · `affiliated` · Machine Unlearning in Learned Databases: An Experimental Analysis
- [gdm-arxiv-2403.08144](<https://deepmind.google/research/publications/74310/>) - 2024-03-11 · `affiliated` · Prosody for Intuitive Robotic Interface Design: It's Not What You Said, It's How You Said It
- [gdm-arxiv-2310.18186](<https://deepmind.google/research/publications/32193/>) - 2024-03-11 · `affiliated` · Model-free Posterior Sampling via Learning Rate Randomization
- [gdm-arxiv-2310.17303](<https://deepmind.google/research/publications/41182/>) - 2024-03-11 · `affiliated` · Demonstration-Regularized RL
- [gdm-arxiv-2310.12036](<https://deepmind.google/research/publications/54918/>) - 2024-03-11 · `affiliated` · Understanding Learning from Human Preferences
- [gdm-arxiv-2305.01521](<https://deepmind.google/research/publications/15530/>) - 2024-03-11 · `affiliated` · Robust Exploration via Clustering-based Density Estimation
- [gdm-doi-10.1101-2024.03.05.24303805](<https://doi.org/10.1101/2024.03.05.24303805>) - 2024-03-06 · `affiliated` · Development and Evaluation of Deep Learning Models for Cardiotocography Interpretation
- [gdm-doi-10.1101-cshperspect.a041472](<https://doi.org/10.1101/cshperspect.a041472>) - 2024-03-04 · `affiliated` · Protein Design Using Structure-Prediction Networks: AlphaFold and RoseTTAFold as Protein Structure Foundation Models
- [gdm-arxiv-2403.00745](<https://deepmind.google/research/publications/68553/>) - 2024-03-04 · `affiliated` · AtP*: Efficient and scalable methods for localizing LLM behaviour to components
- [gdm-web-9f4422ad62db](<https://deepmind.google/research/publications/75635/>) - 2024-03-02 · `affiliated` · How aligned are different alignment metrics?
- [gdm-doi-10.1136-bmjopen-2023-079105](<https://doi.org/10.1136/bmjopen-2023-079105>) - 2024-03-01 · `affiliated` · Defining acceptable data collection and reuse standards for queer artificial intelligence research in mental health: protocol for the online PARQAIR-MH Delphi study
- [gdm-doi-10.1016-j.fusengdes.2024.114161](<https://deepmind.google/research/publications/30578/>) - 2024-03-01 · `affiliated` · Towards Practical Reinforcement Learning for Tokamak Magnetic Control
- [gdm-arxiv-2402.03928](<https://deepmind.google/research/publications/52090/>) - 2024-03-01 · `affiliated` · Approximating the Core of Cooperative Games
- [gdm-arxiv-2312.05328](<https://deepmind.google/research/publications/62998/>) - 2024-02-29 · `affiliated` · Bad Students Make Great Teachers: Active Learning Accelerates Large Scale Visual Understanding
- [gdm-arxiv-2210.06433](<https://deepmind.google/research/publications/15533/>) - 2024-02-29 · `affiliated` · Self-supervised video pretraining yields strong image representations
- [gdm-arxiv-2307.02245](<https://deepmind.google/research/publications/46131/>) - 2024-02-27 · `affiliated` · Set Learning for Accurate and Calibrated Models
- [gdm-web-9ae4f3350b99](<https://deepmind.google/research/publications/45424/>) - 2024-02-26 · `affiliated` · Intriguing Properties of Generative Classifers
- [gdm-web-412b91d72ef1](<https://deepmind.google/research/publications/64513/>) - 2024-02-26 · `affiliated` · A density estimation perspective on learning from pairwise human preferences
- [gdm-arxiv-2403.10519](<https://deepmind.google/research/publications/63200/>) - 2024-02-26 · `affiliated` · Frozen Feature Augmentation
- [gdm-arxiv-2302.01925](<https://deepmind.google/research/publications/49969/>) - 2024-02-25 · `affiliated` · Learning a Fourier Transform for Linear Relative Positional Encodings in Transformers
- [gdm-arxiv-2402.08164](<https://deepmind.google/research/publications/77946/>) - 2024-02-24 · `affiliated` · On Limitations of the Transformer Architecture
- [gdm-arxiv-2402.15391](<https://deepmind.google/research/publications/60474/>) - 2024-02-23 · `affiliated` · Genie: Generative Interactive Environments
- [gdm-arxiv-2402.14396](<https://deepmind.google/research/publications/77240/>) - 2024-02-23 · `affiliated` · AlphaTensor for Optimizing Quantum Computations
- [gdm-web-1852d3c217fd](<https://deepmind.google/research/publications/49667/>) - 2024-02-22 · `affiliated` · When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method
- [gdm-doi-10.1056-aioa2300138](<https://doi.org/10.1056/aioa2300138>) - 2024-02-22 · `affiliated` · Towards Generalist Biomedical AI
- [gdm-arxiv-2402.14547](<https://deepmind.google/research/publications/78451/>) - 2024-02-22 · `affiliated` · OmniPred: Language Models as Universal Regressors
- [gdm-url-d2b992c527cc](<https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf>) - 2024-02-21 · `affiliated` · Gemma: Open Models Based on Gemini Research and Technology
- [gdm-doi-10.1145-3640537.3641580](<https://deepmind.google/research/publications/57746/>) - 2024-02-20 · `affiliated` · The Next 700 ML-Enabled Compiler Optimizations
- [gdm-arxiv-2402.13196](<http://arxiv.org/abs/2402.13196>) - 2024-02-20 · `affiliated` · Practical Kernel Tests of Conditional Independence
- [gdm-arxiv-2402.12422](<https://deepmind.google/research/publications/79663/>) - 2024-02-19 · `affiliated` · Simulacra as Conscious Exotica
- [gdm-arxiv-2402.11450](<https://deepmind.google/research/publications/74007/>) - 2024-02-18 · `affiliated` · Learning to Learn Faster from Human Feedback with Language Model Predictive Control
- [gdm-arxiv-2402.09727](<https://deepmind.google/research/publications/74917/>) - 2024-02-15 · `affiliated` · A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts
- [gdm-arxiv-2402.08733](<https://deepmind.google/research/publications/73709/>) - 2024-02-15 · `affiliated` · Experts Don't Cheat: Learning What You Don't Know by Predicting Pairs
- [gdm-arxiv-2402.08939](<https://deepmind.google/research/publications/75421/>) - 2024-02-14 · `affiliated` · Premise Order Matters in Reasoning with Large Language Models
- [gdm-arxiv-2402.08530](<https://deepmind.google/research/publications/44717/>) - 2024-02-13 · `affiliated` · A Distributional Analogue to the Successor Representation
- [gdm-doi-10.21203-rs.3.rs-3940387-v1](<https://doi.org/10.21203/rs.3.rs-3940387/v1>) - 2024-02-12 · `affiliated` · Consensus, dissensus and synergy between clinicians and specialist foundation models in radiology report generation
- [gdm-arxiv-2402.07872](<https://deepmind.google/research/publications/72495/>) - 2024-02-12 · `affiliated` · PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs
- [gdm-arxiv-2402.07598](<https://deepmind.google/research/publications/70372/>) - 2024-02-12 · `affiliated` · Near-Minimax-Optimal Distributional RL with a Generative Model
- [gdm-web-da672a31934c](<https://deepmind.google/research/publications/47444/>) - 2024-02-11 · `affiliated` · Chain-of-Table: Evolves Tables in the LLM Reasoning Chain for Table Understanding
- [gdm-arxiv-2402.05878](<https://deepmind.google/research/publications/52797/>) - 2024-02-08 · `affiliated` · Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits
- [gdm-arxiv-2402.05861](<https://deepmind.google/research/publications/79057/>) - 2024-02-08 · `affiliated` · Memory Consolidation Enables Long-Context Video Understanding
- [gdm-arxiv-2402.05787](<http://arxiv.org/abs/2402.05787>) - 2024-02-08 · `affiliated` · How do Transformers perform In-Context Autoregressive Learning?
- [gdm-arxiv-2402.03620](<https://deepmind.google/research/publications/64816/>) - 2024-02-06 · `affiliated` · Large Language Models Self-Discover Reasoning Structures
- [gdm-arxiv-2402.01704](<https://deepmind.google/research/publications/67342/>) - 2024-02-06 · `affiliated` · States as Strings as Strategies: Steering Language Models with Game-Theoretic Solvers
- [gdm-doi-10.1109-tpami.2024.3362288](<https://doi.org/10.1109/tpami.2024.3362288>) - 2024-02-05 · `affiliated` · Multi-Task Learning of Object States and State-Modifying Actions From Web Videos
- [gdm-web-a9e79738e688](<https://deepmind.google/research/publications/45020/>) - 2024-02-02 · `affiliated` · Transfer Learning for Bayesian Optimization on Heterogeneous Search Spaces
- [gdm-arxiv-2402.01825](<https://deepmind.google/research/publications/48253/>) - 2024-02-02 · `affiliated` · Fractal Patterns May Unravel the Intelligence in Next-Token Prediction
- [gdm-web-39f227ce5010](<https://deepmind.google/research/publications/49666/>) - 2024-02-01 · `affiliated` · Robust agents learn causal world models
- [gdm-doi-10.1038-s41592-023-02151-z](<https://doi.org/10.1038/s41592-023-02151-z>) - 2024-02-01 · `affiliated` · Metrics reloaded: recommendations for image analysis validation
- [gdm-doi-10.1038-s41587-024-02133-2](<https://doi.org/10.1038/s41587-024-02133-2>) - 2024-02-01 · `affiliated` · Sparks of function by de novo protein design
- [gdm-arxiv-2402.00396](<https://deepmind.google/research/publications/73001/>) - 2024-02-01 · `affiliated` · Exploration at Scale using Epistemic Neural Networks
- [gdm-doi-10.1080-01621459.2024.2311364](<https://doi.org/10.1080/01621459.2024.2311364>) - 2024-01-30 · `affiliated` · Stochastic Low-Rank Tensor Bandits for Multi-Dimensional Online Decision Making
- [gdm-doi-10.1109-tnnls.2024.3352657](<https://doi.org/10.1109/tnnls.2024.3352657>) - 2024-01-29 · `affiliated` · Linear Deconfounded Score Method: Scoring DAGs With Dense Unobserved Confounding
- [gdm-arxiv-2401.14953](<https://deepmind.google/research/publications/42394/>) - 2024-01-26 · `affiliated` · Learning Universal Predictors
- [gdm-arxiv-2401.05133](<https://deepmind.google/research/publications/24820/>) - 2024-01-20 · `affiliated` · Neural Population Learning beyond Symmetric Zero-Sum Games
- [gdm-arxiv-2401.09135](<https://deepmind.google/research/publications/66535/>) - 2024-01-17 · `affiliated` · Asynchronous Local-SGD Training forLanguage Modeling
- [gdm-arxiv-2401.07595](<https://deepmind.google/research/publications/68048/>) - 2024-01-17 · `affiliated` · E3x: E(3)-Equivariant Deep Learning Made Easy
- [gdm-web-b5c7593f9313](<https://deepmind.google/research/publications/24821/>) - 2024-01-16 · `affiliated` · Generative Adversarial Equilibrium Solvers
- [gdm-web-4e9b812f45ae](<https://deepmind.google/research/publications/40173/>) - 2024-01-16 · `affiliated` · NfgTransformer: Equivariant Representation Learning for Normal-form Games
- [gdm-web-42c2a6025502](<https://deepmind.google/research/publications/51081/>) - 2024-01-16 · `affiliated` · Directly Fine-Tuning Diffusion Models on Differentiable Rewards
- [gdm-web-42c03506345c](<https://deepmind.google/research/publications/34213/>) - 2024-01-16 · `affiliated` · Approximating Nash Equilibria in Normal-Form Games via Stochastic Optimization
- [gdm-arxiv-2401.08525](<https://deepmind.google/research/publications/67846/>) - 2024-01-16 · `affiliated` · GATS: Gather-Attend-Scatter
- [gdm-arxiv-2306.13649](<https://deepmind.google/research/publications/48050/>) - 2024-01-16 · `affiliated` · On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
- [gdm-arxiv-2401.06005](<https://pubmed.ncbi.nlm.nih.gov/38259351>) - 2024-01-11 · `affiliated` · How does the primate brain combine generative and discriminative computations in vision?
- [gdm-arxiv-2401.05946](<https://deepmind.google/research/publications/63907/>) - 2024-01-11 · `affiliated` · Learning Planning-compatible Cognitive Maps with Transformers in PartiallyObserved Environments
- [gdm-doi-10.1038-s41593-023-01535-w](<https://deepmind.google/research/publications/2504/>) - 2024-01-10 · `affiliated` · Distributional reinforcement learning in prefrontal cortex
- [gdm-doi-10.1162-tacl_a_00727](<https://doi.org/10.1162/tacl_a_00727>) - 2024-01-07 · `affiliated` · <scp>Dolomites</scp>: Domain-Specific Long-Form Methodical Tasks
- [gdm-arxiv-2401.03529](<http://arxiv.org/abs/2401.03529>) - 2024-01-07 · `affiliated` · Quantifying stability of non-power-seeking in artificial agents
- [gdm-doi-10.7554-elife.85274](<https://doi.org/10.7554/elife.85274>) - 2024-01-04 · `affiliated` · RatInABox, a toolkit for modelling locomotion and neuronal activity in continuous environments
- [gdm-arxiv-2401.12963](<https://deepmind.google/research/publications/48151/>) - 2024-01-04 · `affiliated` · AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents
- [gdm-doi-10.2302-kjm.abstract_73_4-1](<https://doi.org/10.2302/kjm.abstract_73_4-1>) - 2024-01-01 · `affiliated` · Accelerating Scientific Discovery with AI
- [gdm-doi-10.2139-ssrn.4808800](<http://dx.doi.org/10.2139/ssrn.4808800>) - 2024-01-01 · `affiliated` · A Game-Theoretic Model of Misinformation Spread on Social Networks
- [gdm-doi-10.1017-s0140525x23002923](<http://dx.doi.org/10.1017/s0140525x23002923>) - 2024-01-01 · `affiliated` · Dynamic diversity is the answer to proxy failure
- [gdm-doi-10.1007-978-3-031-53468-3_5](<https://doi.org/10.1007/978-3-031-53468-3_5>) - 2024-01-01 · `affiliated` · Training Matters: Unlocking Potentials of Deeper Graph Convolutional Neural Networks
- [gdm-doi-10.1007-978-3-031-53468-3_4](<https://doi.org/10.1007/978-3-031-53468-3_4>) - 2024-01-01 · `affiliated` · When Do We Need Graph Neural Networks for Node Classification?

### 17.4 2023（258 条）

- [gdm-arxiv-2312.15796](<https://deepmind.google/research/publications/68149/>) - 2023-12-25 · `affiliated` · GenCast: learning skillful ensemble forecasting of medium-range weather
- [gdm-url-0453ae158a21](<https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf#page=71>) - 2023-12-22 · `core` · Gemini 1.0 Model Card
- [gdm-arxiv-2312.13252](<https://deepmind.google/research/publications/63604/>) - 2023-12-20 · `affiliated` · Zero-Shot Metric Depth with a Field-of-View Conditioned Diffusion Model
- [gdm-web-f9b970421119](<https://deepmind.google/research/publications/26234/>) - 2023-12-19 · `affiliated` · Equivariant MuZero
- [gdm-doi-10.1109-asru57964.2023.10389789](<http://dx.doi.org/10.1109/asru57964.2023.10389789>) - 2023-12-16 · `affiliated` · Detecting Speech Abnormalities With a Perceiver-Based Sequence Classifier that Leverages a Universal Speech Model
- [gdm-doi-10.4171-icm2022-180](<http://dx.doi.org/10.4171/icm2022/180>) - 2023-12-15 · `affiliated` · Simulation-based search
- [gdm-doi-10.1145-3631119](<https://doi.org/10.1145/3631119>) - 2023-12-15 · `affiliated` · Optimal Inapproximability with Universal Factor Graphs
- [gdm-arxiv-2312.10029](<https://deepmind.google/research/publications/66937/>) - 2023-12-15 · `affiliated` · Challenges with unsupervised LLM knowledge discovery
- [gdm-arxiv-2311.17894](<https://deepmind.google/research/publications/48254/>) - 2023-12-15 · `affiliated` · Learning Silicon Dopant Transitions in Graphene using Scanning Transmission Electron Microscopy
- [gdm-web-ecdd6752f7a5](<https://deepmind.google/research/publications/33809/>) - 2023-12-14 · `affiliated` · Meta-in-context learning in large language models
- [gdm-doi-10.1038-s41586-023-06924-6](<https://doi.org/10.1038/s41586-023-06924-6>) - 2023-12-14 · `affiliated` · Mathematical discoveries from program search with large language models
- [gdm-doi-10.1101-2023.12.12.571268](<https://doi.org/10.1101/2023.12.12.571268>) - 2023-12-13 · `affiliated` · A generative model of the hippocampal formation trained with theta driven local learning rules
- [gdm-web-eceef3973b88](<https://deepmind.google/research/publications/34921/>) - 2023-12-12 · `affiliated` · Online RL in Linearly &#36;q^&#92;&#92;pi&#36;-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore
- [gdm-web-769a52666d60](<https://deepmind.google/research/publications/35122/>) - 2023-12-12 · `affiliated` · Schema-learning and rebinding as mechanisms of in-context learning and emergence
- [gdm-web-6e3657de35c3](<https://deepmind.google/research/publications/33910/>) - 2023-12-12 · `affiliated` · A Definition of Continual Reinforcement Learning
- [gdm-web-4e44e6d4c183](<https://deepmind.google/research/publications/84309/>) - 2023-12-12 · `affiliated` · Rethinking the Role of Token Retrieval in Multi-Vector Retrieval
- [gdm-arxiv-2312.07395](<https://deepmind.google/research/publications/60675/>) - 2023-12-12 · `affiliated` · A Simple Recipe for Contrastively Pre-training Video-First Encoders Beyond 16 Frames
- [gdm-web-d4259f3e86ba](<https://deepmind.google/research/publications/18457/>) - 2023-12-10 · `affiliated` · Benchmarking Robustness to Adversarial Image Obfuscations
- [gdm-web-9d82558c556c](<https://deepmind.google/research/publications/37041/>) - 2023-12-10 · `affiliated` · Optimization and Evaluation of Fine-grained Jaccard Indexes for Semantic Segmentation
- [gdm-web-93dc8862cdf5](<https://deepmind.google/research/publications/34617/>) - 2023-12-10 · `affiliated` · Optimal Preconditioning and Fisher Adaptive Langevin Sampling
- [gdm-web-8d0448119c76](<https://deepmind.google/research/publications/31486/>) - 2023-12-10 · `affiliated` · Feature Likelihood Divergence: Evaluating the Generalization of Generative Models Using Samples
- [gdm-web-88da360fdb86](<https://deepmind.google/research/publications/33709/>) - 2023-12-10 · `affiliated` · Passive learning of active causal strategies in agents and language models
- [gdm-web-86345e30ba8b](<https://deepmind.google/research/publications/32698/>) - 2023-12-10 · `affiliated` · Towards In-context Scene Understanding
- [gdm-web-747e2e88b4b4](<https://deepmind.google/research/publications/34719/>) - 2023-12-10 · `affiliated` · Probabilistic Inference in Reinforcement Learning Done Right
- [gdm-web-7215988af4cf](<https://deepmind.google/research/publications/82693/>) - 2023-12-10 · `affiliated` · LambdaBeam: Neural Program Search with Higher-Order Functions and Lambdas
- [gdm-web-2618a3b26b8d](<https://deepmind.google/research/publications/33708/>) - 2023-12-10 · `affiliated` · Improving neural network representations using human similarity judgments
- [gdm-arxiv-2312.07358](<https://deepmind.google/research/publications/42395/>) - 2023-12-09 · `affiliated` · Distributional Bellman Operators over Mean-embeddings
- [gdm-web-0cec1552ff74](<https://deepmind.google/research/publications/53605/>) - 2023-12-08 · `affiliated` · POMRL: No-Regret Learning-to-Plan with IncreasingHorizons
- [gdm-web-f51feeab41f8](<https://deepmind.google/research/publications/49871/>) - 2023-12-07 · `affiliated` · Revisiting Dynamic Evaluation:Online Adaptation for LLMs
- [gdm-web-4bc0eb129da3](<https://deepmind.google/research/publications/82087/>) - 2023-12-06 · `affiliated` · A Benchmark for Reasoning with Spatial Prepositions
- [gdm-web-4a9bf851f1b4](<https://deepmind.google/research/publications/82492/>) - 2023-12-06 · `affiliated` · SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation
- [gdm-doi-10.18653-v1-2023.emnlp-main.266](<https://deepmind.google/research/publications/81988/>) - 2023-12-06 · `affiliated` · MingOfficial: A Ming Official Career Dataset and a Historical Context-Aware Representation Learning Framework
- [gdm-doi-10.1080-10400419.2023.2289757](<https://doi.org/10.1080/10400419.2023.2289757>) - 2023-12-06 · `affiliated` · The Receptive Brain: Up-Regulated Right Temporal Alpha Oscillation Boosting Aha!
- [gdm-arxiv-2312.03664](<https://deepmind.google/research/publications/64717/>) - 2023-12-06 · `affiliated` · Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia
- [gdm-web-93afc90c257e](<https://deepmind.google/research/publications/83506/>) - 2023-12-05 · `affiliated` · Gaussian Process Probes (GPP) for Uncertainty-Aware Probing
- [gdm-web-869c1af3a59c](<https://deepmind.google/research/publications/35829/>) - 2023-12-05 · `affiliated` · RoboCat: A Self-Improving Foundation Agent for Robotic Manipulation
- [gdm-web-344dd7abc6f8](<https://deepmind.google/research/publications/82494/>) - 2023-12-04 · `affiliated` · Small batch deep reinforcement learning
- [gdm-arxiv-2312.01057](<https://deepmind.google/research/publications/63806/>) - 2023-12-02 · `affiliated` · RLHF and IIA: Perverse Incentives
- [gdm-web-6a800cb32c4e](<https://deepmind.google/research/publications/61281/>) - 2023-11-29 · `affiliated` · Unsupervised Keypoints with Stable Diffusion
- [gdm-web-146dc096f7c0](<https://deepmind.google/research/publications/61180/>) - 2023-11-29 · `affiliated` · Accelerating Neural Field Training via Langevin Monte-Carlo Sampling
- [gdm-doi-10.1109-cvpr52733.2024.02181](<https://deepmind.google/research/publications/44213/>) - 2023-11-29 · `affiliated` · SODA: Bottleneck Diffusion Models for Representation Learning
- [gdm-doi-10.1038-s41586-023-06734-w](<https://doi.org/10.1038/s41586-023-06734-w>) - 2023-11-29 · `affiliated` · An autonomous laboratory for the accelerated synthesis of inorganic materials
- [gdm-arxiv-2311.17311](<https://deepmind.google/research/publications/50879/>) - 2023-11-29 · `affiliated` · Universal Self-Consistency with Large Language Models
- [gdm-doi-10.1038-s41467-023-42875-2](<https://doi.org/10.1038/s41467-023-42875-2>) - 2023-11-28 · `affiliated` · Learning few-shot imitation as cultural transmission
- [gdm-arxiv-2311.15951](<https://deepmind.google/research/publications/50575/>) - 2023-11-27 · `affiliated` · Replay Across Experiments
- [gdm-doi-10.1017-s0140525x23003266](<https://doi.org/10.1017/s0140525x23003266>) - 2023-11-23 · `affiliated` · Meta-learned models of cognition
- [gdm-arxiv-2311.14125](<https://deepmind.google/research/publications/34920/>) - 2023-11-23 · `affiliated` · Scalable AI Safety via Doubly-Efficient Debate
- [gdm-doi-10.1523-jneurosci.1009-23.2023](<https://doi.org/10.1523/jneurosci.1009-23.2023>) - 2023-11-21 · `affiliated` · Responses to Pattern-Violating Visual Stimuli Evolve Differently Over Days in Somata and Distal Apical Dendrites
- [gdm-doi-10.1038-s42256-023-00748-9](<https://doi.org/10.1038/s42256-023-00748-9>) - 2023-11-20 · `affiliated` · Spatially embedded recurrent neural networks reveal widespread links between structural and functional neuroscience findings
- [gdm-arxiv-2311.11388](<https://doi.org/10.1038/s41562-023-01742-2>) - 2023-11-20 · `affiliated` · Machine culture
- [gdm-web-afd0db86c927](<https://deepmind.google/research/publications/25830/>) - 2023-11-17 · `affiliated` · No agent is an island: A social path to human-like artificial intelligence
- [gdm-arxiv-2405.15815](<https://doi.org/10.1038/s42256-023-00754-x>) - 2023-11-17 · `affiliated` · A social path to human-like artificial intelligence
- [gdm-doi-10.1126-science.adi2336](<https://deepmind.google/research/publications/22598/>) - 2023-11-14 · `affiliated` · GraphCast: Learned Global Weather Forecasting
- [gdm-arxiv-2311.08105](<https://deepmind.google/research/publications/57039/>) - 2023-11-14 · `affiliated` · DiLoCo: Distributed Low-Communication Training of Language Models
- [gdm-arxiv-2311.06477](<https://deepmind.google/research/publications/58352/>) - 2023-11-14 · `affiliated` · Report of the 1st Workshop on Generative AI and Law
- [gdm-doi-10.1101-2023.11.09.563812](<https://doi.org/10.1101/2023.11.09.563812>) - 2023-11-13 · `affiliated` · An encyclopedia of enhancer-gene regulatory interactions in the human genome
- [gdm-doi-10.1073-pnas.2308911120](<https://deepmind.google/research/publications/35526/>) - 2023-11-10 · `affiliated` · Emotions and courtship help bonded pairs cooperate, but emotional agents are vulnerable to deceit
- [gdm-doi-10.1038-s41586-023-06647-8](<https://deepmind.google/research/publications/35223/>) - 2023-11-08 · `affiliated` · Role Play with Large Language Models
- [gdm-arxiv-2311.03583](<https://deepmind.google/research/publications/44214/>) - 2023-11-06 · `affiliated` · Finding Increasingly Large Extremal Graphs with AlphaZero and Tabu Search
- [gdm-doi-10.1140-epja-s10050-023-01154-w](<https://doi.org/10.1140/epja/s10050-023-01154-w>) - 2023-11-04 · `affiliated` · Aspects of scaling and scalability for flow-based sampling of lattice QCD
- [gdm-web-b9c20adf439d](<https://deepmind.google/research/publications/48757/>) - 2023-11-03 · `affiliated` · RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches
- [gdm-web-96a684d7bd6d](<https://deepmind.google/research/publications/83400/>) - 2023-11-03 · `affiliated` · Grammar Prompting for Domain-Specific Language Generation with Large Language Models
- [gdm-web-4e87fd67f9c8](<https://deepmind.google/research/publications/24720/>) - 2023-11-02 · `affiliated` · Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework for Online RL
- [gdm-web-474388e2dde5](<https://deepmind.google/research/publications/5642/>) - 2023-11-02 · `affiliated` · Optimistic Meta-Gradients
- [gdm-doi-10.1093-nar-gkad1011](<https://doi.org/10.1093/nar/gkad1011>) - 2023-11-02 · `affiliated` · AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences
- [gdm-arxiv-2311.00899](<https://deepmind.google/research/publications/63605/>) - 2023-11-01 · `affiliated` · RoboVQA: Multimodal Long-Horizon Reasoning for Robotics
- [gdm-web-fe6707dc12df](<https://deepmind.google/research/publications/22497/>) - 2023-10-30 · `affiliated` · Population-based Evaluation in Repeated Rock-Paper-Scissors as a Benchmark for Multiagent Reinforcement Learning
- [gdm-doi-10.1109-iccad57390.2023.10323681](<https://doi.org/10.1109/iccad57390.2023.10323681>) - 2023-10-28 · `affiliated` · LFPS: Learned Formal Proof Strengthening for Efficient Hardware Verification
- [gdm-doi-10.1038-s42254-023-00670-4](<https://doi.org/10.1038/s42254-023-00670-4>) - 2023-10-27 · `affiliated` · Can a dual mandate be a model for the global governance of AI?
- [gdm-doi-10.1101-2021.06.06.447249](<https://deepmind.google/research/publications/5630/>) - 2023-10-26 · `affiliated` · Generative replay for compositional visual understanding in the prefrontal-hippocampal circuit
- [gdm-doi-10.1098-rspb.2023.1716](<https://doi.org/10.1098/rspb.2023.1716>) - 2023-10-25 · `affiliated` · The emergence of division of labour through decentralized social sanctioning
- [gdm-arxiv-2310.13225](<https://deepmind.google/research/publications/50474/>) - 2023-10-20 · `affiliated` · Scalable Neural Network Kernels
- [gdm-arxiv-2311.09235](<https://deepmind.google/research/publications/51282/>) - 2023-10-18 · `affiliated` · Scalable Diffusion for Materials Generation
- [gdm-arxiv-2310.11986](<https://deepmind.google/research/publications/45425/>) - 2023-10-18 · `affiliated` · Sociotechnical Safety Evaluation of generative AI systems
- [gdm-doi-10.3389-frym.2023.1241472](<http://dx.doi.org/10.3389/frym.2023.1241472>) - 2023-10-17 · `affiliated` · AI Helping Science: The ‘Shape’ of Things to Come
- [gdm-arxiv-2310.09213](<http://arxiv.org/abs/2310.09213>) - 2023-10-13 · `affiliated` · A Sampling-Based Domain Generalization Study with Diffusion Generative Models
- [gdm-arxiv-2310.08864](<https://deepmind.google/blog/scaling-up-learning-across-many-different-robot-types/>) - 2023-10-13 · `affiliated` · Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- [gdm-arxiv-2310.08584](<http://arxiv.org/abs/2310.08584>) - 2023-10-12 · `affiliated` · Is ImageNet worth 1 video? Learning strong image encoders from 1 long unlabelled video
- [gdm-web-37a769a93f40](<https://deepmind.google/research/publications/48657/>) - 2023-10-09 · `affiliated` · DyST: Towards Dynamic Neural Scene Representations on Real-World Videos
- [gdm-arxiv-2310.06117](<https://deepmind.google/research/publications/50274/>) - 2023-10-09 · `affiliated` · Step-Back Prompting Enables Reasoning via Abstraction in Large Language Models
- [gdm-arxiv-2310.06114](<https://deepmind.google/research/publications/47545/>) - 2023-10-09 · `affiliated` · Learning Interactive Real-World Simulator
- [gdm-arxiv-2310.04859](<https://deepmind.google/research/publications/48555/>) - 2023-10-07 · `affiliated` · Universal Graph Random Features
- [gdm-arxiv-2310.04854](<https://deepmind.google/research/publications/48455/>) - 2023-10-07 · `affiliated` · Repelling Random Walks
- [gdm-doi-10.1073-pnas.2305349120](<https://deepmind.google/research/publications/29062/>) - 2023-10-05 · `affiliated` · An Impossibility Theorem in Game Dynamics
- [gdm-doi-10.1016-j.physd.2023.133940](<https://doi.org/10.1016/j.physd.2023.133940>) - 2023-10-05 · `affiliated` · A stochastic variant of replicator dynamics in zero-sum games and its invariant measures
- [gdm-arxiv-2310.02932](<https://deepmind.google/research/publications/43202/>) - 2023-10-05 · `affiliated` · Assessing LLMs on Climate Information
- [gdm-arxiv-2310.01798](<https://deepmind.google/research/publications/48252/>) - 2023-10-03 · `affiliated` · Large Language Models Cannot Self-Correct Reasoning Yet
- [gdm-arxiv-2310.01714](<https://deepmind.google/research/publications/51283/>) - 2023-10-03 · `affiliated` · Large Language Models as Analogical Reasoners
- [gdm-doi-10.1109-iccv51070.2023.01977](<https://deepmind.google/research/publications/14720/>) - 2023-10-02 · `affiliated` · 3D Neural Embedding Likelihood: Probabilistic Inverse Graphics for Robust 6D Pose Estimation
- [gdm-doi-10.1109-iccv51070.2023.00923](<https://deepmind.google/research/publications/26336/>) - 2023-10-02 · `affiliated` · TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement
- [gdm-arxiv-2301.08173](<https://doi.org/10.1088/2632-2153/acff39>) - 2023-10-02 · `affiliated` · Time-warping invariant quantum recurrent neural networks via quantum-classical adaptive gating
- [gdm-doi-10.1109-iros55552.2023.10341979](<http://dx.doi.org/10.1109/iros55552.2023.10341979>) - 2023-10-01 · `affiliated` · Discovering Adaptable Symbolic Algorithms from Scratch
- [gdm-doi-10.1109-iccv51070.2023.01443](<https://doi.org/10.1109/iccv51070.2023.01443>) - 2023-10-01 · `affiliated` · Waffling around for Performance: Visual Classification with Random Words and Broad Concepts
- [gdm-doi-10.1109-iccv51070.2023.01438](<https://doi.org/10.1109/iccv51070.2023.01438>) - 2023-10-01 · `affiliated` · What does a platypus look like? Generating customized prompts for zero-shot image classification
- [gdm-doi-10.1109-iccv51070.2023.01430](<https://doi.org/10.1109/iccv51070.2023.01430>) - 2023-10-01 · `affiliated` · Contrastive Feature Masking Open-Vocabulary Vision Transformer
- [gdm-doi-10.1109-iccv51070.2023.01400](<https://doi.org/10.1109/iccv51070.2023.01400>) - 2023-10-01 · `affiliated` · Guiding image captioning models toward more specific captions
- [gdm-doi-10.1109-iccv51070.2023.01278](<https://doi.org/10.1109/iccv51070.2023.01278>) - 2023-10-01 · `affiliated` · Helping Hands: An Object-Aware Ego-Centric Video Recognition Model
- [gdm-doi-10.1109-iccv51070.2023.01269](<https://doi.org/10.1109/iccv51070.2023.01269>) - 2023-10-01 · `affiliated` · Video OWL-ViT: Temporally-consistent open-world localization in video
- [gdm-doi-10.1109-iccv51070.2023.00492](<https://doi.org/10.1109/iccv51070.2023.00492>) - 2023-10-01 · `affiliated` · M2T: Masking Transformers Twice for Faster Decoding
- [gdm-doi-10.1109-iccv51070.2023.00257](<https://doi.org/10.1109/iccv51070.2023.00257>) - 2023-10-01 · `affiliated` · SuS-X: Training-Free Name-Only Transfer of Vision-Language Models
- [gdm-doi-10.1016-j.cell.2023.09.004](<https://doi.org/10.1016/j.cell.2023.09.004>) - 2023-10-01 · `affiliated` · Generative replay underlies compositional inference in the hippocampal-prefrontal circuit
- [gdm-doi-10.1007-978-3-031-19568-6_10](<https://doi.org/10.1007/978-3-031-19568-6_10>) - 2023-09-30 · `affiliated` · Using Approximate DRAM for Enabling Energy-Efficient, High-Performance Deep Neural Network Inference
- [gdm-doi-10.1371-journal.pbio.3002306](<http://dx.doi.org/10.1371/journal.pbio.3002306>) - 2023-09-26 · `affiliated` · Computational and systems neuroscience: The next 20 years
- [gdm-doi-10.1016-j.isci.2023.108047](<https://doi.org/10.1016/j.isci.2023.108047>) - 2023-09-26 · `affiliated` · Pretrial predictors of conflict response efficacy in the human prefrontal cortex
- [gdm-web-88c458d39d42](<https://deepmind.google/research/publications/56635/>) - 2023-09-23 · `affiliated` · Theoretical and Practical Perspectives on what Influence Functions Do
- [gdm-web-04375b6248a5](<https://deepmind.google/research/publications/34416/>) - 2023-09-21 · `affiliated` · Self-Predictive Universal AI
- [gdm-arxiv-2301.05062](<https://deepmind.google/research/publications/22295/>) - 2023-09-21 · `affiliated` · Tracr: Compiled Transformers as a Laboratory for Interpretability
- [gdm-doi-10.5281-zenodo.8208688](<https://zenodo.org/record/8208688>) - 2023-09-19 · `affiliated` · Predictions for AlphaMissense
- [gdm-doi-10.1126-science.adg7492](<https://deepmind.google/research/publications/21083/>) - 2023-09-19 · `affiliated` · Accurate proteome-wide missense variant effect prediction with AlphaMissense
- [gdm-doi-10.1016-j.neuron.2023.08.021](<https://doi.org/10.1016/j.neuron.2023.08.021>) - 2023-09-18 · `affiliated` · Goal-seeking compresses neural codes for space in the human hippocampus and orbitofrontal cortex
- [gdm-doi-10.1088-2632-2153-acfa63](<https://doi.org/10.1088/2632-2153/acfa63>) - 2023-09-15 · `affiliated` · Rediscovering orbital mechanics with machine learning
- [gdm-doi-10.1109-tmc.2023.3315120](<https://doi.org/10.1109/tmc.2023.3315120>) - 2023-09-13 · `affiliated` · Teamwork Reinforcement Learning With Concave Utilities
- [gdm-arxiv-2309.05803](<https://deepmind.google/research/publications/82491/>) - 2023-09-11 · `affiliated` · Revisiting Energy Based Models as Policies: Ranking Noise Contrastive Estimation and Interpolating Energy Models
- [gdm-arxiv-2305.11290](<https://deepmind.google/research/publications/34011/>) - 2023-09-10 · `affiliated` · Massively Scalable Inverse Reinforcement Learning for Route Optimization
- [gdm-doi-10.3758-s13428-023-02203-4](<https://doi.org/10.3758/s13428-023-02203-4>) - 2023-09-08 · `affiliated` · Test–retest reliability of reinforcement learning parameters
- [gdm-doi-10.1038-s41562-023-01686-7](<https://doi.org/10.1038/s41562-023-01686-7>) - 2023-09-07 · `affiliated` · Scaffolding cooperation in human groups with deep reinforcement learning
- [gdm-doi-10.31219-osf.io-6fw42](<http://dx.doi.org/10.31219/osf.io/6fw42>) - 2023-09-06 · `affiliated` · Beyond the Matrix: Using Multi-Agent-Reinforcement Learning and Behavioral Experiments to Study Social-Ecological Systems
- [gdm-doi-10.1088-2632-2153-acefa8](<https://deepmind.google/research/publications/16942/>) - 2023-09-04 · `affiliated` · Estimating Gibbs free energies via isobaric-isothermal flows
- [gdm-arxiv-2309.00656](<http://arxiv.org/abs/2309.00656>) - 2023-09-01 · `affiliated` · Local and adaptive mirror descents in extensive-form games
- [gdm-arxiv-2308.16848](<https://deepmind.google/research/publications/41485/>) - 2023-08-31 · `affiliated` · Natural Quantum Monte Carlo Computation of Excited States
- [gdm-arxiv-2308.15975](<https://deepmind.google/research/publications/38759/>) - 2023-08-31 · `affiliated` · RoboTAP: Tracking Arbitrary Points for Few-Shot Visual Imitation
- [gdm-doi-10.21203-rs.3.rs-3296728-v1](<https://doi.org/10.21203/rs.3.rs-3296728/v1>) - 2023-08-28 · `affiliated` · Personality Traits in Large Language Models
- [gdm-doi-10.1371-journal.pcbi.1011316](<https://doi.org/10.1371/journal.pcbi.1011316>) - 2023-08-25 · `affiliated` · Disentangling Abstraction from Statistical Pattern Matching in Human and Machine Learning
- [gdm-doi-10.1523-jneurosci.1014-23.2023](<https://doi.org/10.1523/jneurosci.1014-23.2023>) - 2023-08-23 · `affiliated` · Neuroscience Needs Network Science
- [gdm-web-e479ac06252a](<https://deepmind.google/research/publications/32195/>) - 2023-08-22 · `affiliated` · Nevis’22: A Stream of 100 Tasks Sampled from 30 Years of Computer Vision Research
- [gdm-doi-10.24963-ijcai.2023-624](<https://deepmind.google/research/publications/21589/>) - 2023-08-19 · `affiliated` · Levin Tree Search with Context Models
- [gdm-doi-10.1111-ajps.12818](<https://doi.org/10.1111/ajps.12818>) - 2023-08-17 · `affiliated` · Placebo Tests for Causal Inference
- [gdm-arxiv-2308.09198](<http://arxiv.org/abs/2308.09198>) - 2023-08-17 · `affiliated` · Half-Hop: A graph upsampling approach for slowing down message passing
- [gdm-arxiv-2208.12590](<https://doi.org/10.1038/s41570-023-00516-8>) - 2023-08-09 · `affiliated` · Ab initio quantum chemistry with neural-network wavefunctions
- [gdm-doi-10.1145-3600211.3604693](<https://doi.org/10.1145/3600211.3604693>) - 2023-08-08 · `affiliated` · Democratising AI: Multiple Meanings, Goals, and Methods
- [gdm-doi-10.1101-2023.08.07.23293764](<http://dx.doi.org/10.1101/2023.08.07.23293764>) - 2023-08-07 · `affiliated` · Protocol for a Delphi consensus process for PARticipatory Queer AI Research in Mental Health (PARQAIR-MH)
- [gdm-web-a99644389189](<https://deepmind.google/research/publications/8258/>) - 2023-08-04 · `affiliated` · Advances in ML-based sampling for Lattice-QCD
- [gdm-arxiv-2309.01156](<https://doi.org/10.1038/s42254-023-00616-w>) - 2023-08-04 · `affiliated` · Advances in machine-learning-based sampling motivated by lattice quantum chromodynamics
- [gdm-doi-10.5281-zenodo.8208697](<https://zenodo.org/record/8208697>) - 2023-08-02 · `affiliated` · Source code for AlphaMissense
- [gdm-doi-10.1038-s41586-023-06221-2](<https://doi.org/10.1038/s41586-023-06221-2>) - 2023-08-02 · `affiliated` · Scientific discovery in the age of artificial intelligence
- [gdm-doi-10.24963-ijcai.2023-783](<http://dx.doi.org/10.24963/ijcai.2023/783>) - 2023-08-01 · `affiliated` · Rethinking Formal Models of Partially Observable Multiagent Decision Making (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2023-384](<http://dx.doi.org/10.24963/ijcai.2023/384>) - 2023-08-01 · `affiliated` · Scaling Goal-based Exploration via Pruning Proto-goals
- [gdm-doi-10.1145-3544548.3581225](<https://deepmind.google/research/publications/13609/>) - 2023-08-01 · `affiliated` · Co-Writing Screenplays and Theatre Scripts with Language Models: An Evaluation by Industry Professionals
- [gdm-doi-10.1111-cogs.13315](<https://doi.org/10.1111/cogs.13315>) - 2023-08-01 · `affiliated` · The Puzzle of Evaluating Moral Cognition in Artificial Agents
- [gdm-arxiv-2307.16560](<https://deepmind.google/research/publications/37648/>) - 2023-08-01 · `affiliated` · Line Search for Convex Minimization
- [gdm-doi-10.1109-tro.2023.3297015](<https://doi.org/10.1109/tro.2023.3297015>) - 2023-07-31 · `affiliated` · VAE-Loco: Versatile Quadruped Locomotion by Learning a Disentangled Gait Representation
- [gdm-url-2a645302dbbd](<https://robotics-transformer2.github.io/assets/rt2.pdf>) - 2023-07-28 · `affiliated` · RT-2: New model translates vision and language into action — ‍ Read our paper
- [gdm-doi-10.3389-frai.2023.804682](<https://doi.org/10.3389/frai.2023.804682>) - 2023-07-20 · `affiliated` · Learning to play against any mixture of opponents
- [gdm-doi-10.1038-s41467-023-39902-7](<https://doi.org/10.1038/s41467-023-39902-7>) - 2023-07-18 · `affiliated` · Detecting shortcut learning for fair medical AI using shortcut testing
- [gdm-arxiv-2307.08874](<http://arxiv.org/abs/2307.08874>) - 2023-07-17 · `affiliated` · Latent Space Representations of Neural Algorithmic Reasoners
- [gdm-doi-10.1145-3583133.3595822](<https://doi.org/10.1145/3583133.3595822>) - 2023-07-15 · `affiliated` · Discovering Evolution Strategies via Meta-Black-Box Optimization
- [gdm-doi-10.1145-3597926.3598141](<https://doi.org/10.1145/3597926.3598141>) - 2023-07-12 · `affiliated` · CodeGrid: A Grid Representation of Code
- [gdm-doi-10.1145-3583131.3590496](<https://doi.org/10.1145/3583131.3590496>) - 2023-07-12 · `affiliated` · Discovering Attention-Based Genetic Algorithms via Meta-Black-Box Optimization
- [gdm-doi-10.1038-s41586-023-06291-2](<https://doi.org/10.1038/s41586-023-06291-2>) - 2023-07-12 · `affiliated` · Large language models encode clinical knowledge
- [gdm-doi-10.1561-2200000097](<https://doi.org/10.1561/2200000097>) - 2023-07-11 · `affiliated` · Reinforcement Learning, Bit by Bit
- [gdm-arxiv-2307.04699](<https://deepmind.google/blog/exploring-institutions-for-global-ai-governance/>) - 2023-07-10 · `affiliated` · International Institutions for Advanced AI
- [gdm-doi-10.1162-coli_a_00490](<https://doi.org/10.1162/coli_a_00490>) - 2023-07-06 · `affiliated` · Measuring Attribution in Natural Language Generation Models
- [gdm-doi-10.1016-j.isci.2023.107256](<https://doi.org/10.1016/j.isci.2023.107256>) - 2023-07-04 · `affiliated` · Humans perceive warmth and competence in artificial intelligence
- [gdm-doi-10.1111-cogs.13312](<https://doi.org/10.1111/cogs.13312>) - 2023-07-01 · `affiliated` · Modeling Structure‐Building in the Brain With CCG Parsing and Large Language Models
- [gdm-doi-10.1038-s41591-023-02437-x](<https://doi.org/10.1038/s41591-023-02437-x>) - 2023-07-01 · `affiliated` · Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians
- [gdm-doi-10.1037-dec0000219](<https://doi.org/10.1037/dec0000219>) - 2023-06-29 · `affiliated` · Discounting future reward in an uncertain world.
- [gdm-doi-10.1609-aaai.v37i9.26227](<https://doi.org/10.1609/aaai.v37i9.26227>) - 2023-06-26 · `affiliated` · Predictive Multiplicity in Probabilistic Classification
- [gdm-doi-10.1609-aaai.v37i8.26164](<https://doi.org/10.1609/aaai.v37i8.26164>) - 2023-06-26 · `affiliated` · Exploration via Epistemic Value Estimation
- [gdm-doi-10.1609-aaai.v37i13.26962](<http://dx.doi.org/10.1609/aaai.v37i13.26962>) - 2023-06-26 · `affiliated` · AlphaSnake: Policy Iteration on a Nondeterministic NP-Hard Markov Decision Process (Student Abstract)
- [gdm-doi-10.1101-2023.06.23.546250](<https://doi.org/10.1101/2023.06.23.546250>) - 2023-06-26 · `affiliated` · Cognitive Model Discovery via Disentangled RNNs
- [gdm-doi-10.1016-j.artint.2023.103963](<https://doi.org/10.1016/j.artint.2023.103963>) - 2023-06-24 · `affiliated` · Discovering agents
- [gdm-doi-10.1093-oxfordhb-9780197579329.013.2](<https://doi.org/10.1093/oxfordhb/9780197579329.013.2>) - 2023-06-20 · `affiliated` · AI Governance
- [gdm-arxiv-2306.11706](<https://deepmind.google/blog/robocat-a-self-improving-robotic-agent/>) - 2023-06-20 · `affiliated` · RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation
- [gdm-doi-10.1145-3593013.3594019](<https://doi.org/10.1145/3593013.3594019>) - 2023-06-12 · `affiliated` · Representation in AI Evaluations
- [gdm-doi-10.1063-5.0147877](<https://doi.org/10.1063/5.0147877>) - 2023-06-08 · `affiliated` · Optimizing Jastrow factors for the transcorrelated method
- [gdm-doi-10.1038-s41551-023-01049-7](<https://doi.org/10.1038/s41551-023-01049-7>) - 2023-06-08 · `affiliated` · Robust and data-efficient generalization of self-supervised machine learning for diagnostic imaging
- [gdm-doi-10.1038-s41586-023-06004-9](<https://doi.org/10.1038/s41586-023-06004-9>) - 2023-06-07 · `affiliated` · Faster sorting algorithms discovered using deep reinforcement learning
- [gdm-doi-10.1109-dsn58367.2023.00049](<http://dx.doi.org/10.1109/dsn58367.2023.00049>) - 2023-06-01 · `affiliated` · Adaptive Webpage Fingerprinting from TLS Traces
- [gdm-doi-10.1109-cvpr52729.2023.02337](<https://doi.org/10.1109/cvpr52729.2023.02337>) - 2023-06-01 · `affiliated` · Understanding Deep Generative Models with Generalized Empirical Likelihoods
- [gdm-doi-10.1109-cvpr52729.2023.01185](<https://doi.org/10.1109/cvpr52729.2023.01185>) - 2023-06-01 · `affiliated` · Seasoning Model Soups for Robustness to Adversarial and Natural Distribution Shifts
- [gdm-doi-10.1109-cvpr52729.2023.01134](<https://doi.org/10.1109/cvpr52729.2023.01134>) - 2023-06-01 · `affiliated` · Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption
- [gdm-arxiv-2306.00662](<https://doi.org/10.1063/5.0136752>) - 2023-06-01 · `affiliated` · Fast transport simulations with higher-fidelity surrogate models for ITER
- [gdm-arxiv-2302.14115](<https://doi.org/10.1109/cvpr52729.2023.01032>) - 2023-06-01 · `affiliated` · Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning
- [gdm-doi-10.65109-ehmm3042](<https://doi.org/10.65109/ehmm3042>) - 2023-05-30 · `affiliated` · Search-Improved Game-Theoretic Multiagent Reinforcement Learning in General and Negotiation Games
- [gdm-doi-10.65109-acih8655](<https://doi.org/10.65109/acih8655>) - 2023-05-30 · `affiliated` · Diversity Through Exclusion (DTE): Niche Identification for Reinforcement Learning through Value-Decomposition
- [gdm-doi-10.1109-icra48891.2023.10161544](<https://doi.org/10.1109/icra48891.2023.10161544>) - 2023-05-29 · `affiliated` · NeRF2Real: Sim2real Transfer of Vision-guided Bipedal Motion Skills using Neural Radiance Fields
- [gdm-arxiv-2305.18501](<http://arxiv.org/abs/2305.18501>) - 2023-05-29 · `affiliated` · DoMo-AC: Doubly Multi-step Off-policy Actor-Critic Algorithm
- [gdm-arxiv-2305.18161](<http://arxiv.org/abs/2305.18161>) - 2023-05-29 · `affiliated` · VA-learning as a more efficient alternative to Q-learning
- [gdm-arxiv-2305.13991](<http://arxiv.org/abs/2305.13991>) - 2023-05-23 · `affiliated` · Expressive Losses for Verified Robustness via Convex Combinations
- [gdm-arxiv-2305.13185](<http://arxiv.org/abs/2305.13185>) - 2023-05-22 · `affiliated` · Regularization and Variance-Weighted Regression Achieves Minimax Optimality in Linear MDPs: Theory and Practice
- [gdm-doi-10.1101-2023.05.17.541226](<https://doi.org/10.1101/2023.05.17.541226>) - 2023-05-17 · `affiliated` · Predictive and Interpretable: Combining Artificial Neural Networks and Classic Cognitive Models to Understand Human Learning and Decision Making
- [gdm-doi-10.1038-s41597-023-02214-y](<https://doi.org/10.1038/s41597-023-02214-y>) - 2023-05-17 · `affiliated` · Responses of pyramidal cell somata and apical dendrites in mouse visual cortex over multiple days
- [gdm-doi-10.1145-3564246.3585161](<https://doi.org/10.1145/3564246.3585161>) - 2023-05-16 · `affiliated` · Optimistic MLE: A Generic Model-Based Algorithm for Partially Observable Sequential Decision Making
- [gdm-doi-10.1037-rev0000414](<https://doi.org/10.1037/rev0000414>) - 2023-05-11 · `affiliated` · A probabilistic successor representation for context-dependent learning.
- [gdm-doi-10.1109-tc.2023.3272282](<https://doi.org/10.1109/tc.2023.3272282>) - 2023-05-05 · `affiliated` · EcoFlow: Efficient Convolutional Dataflows on Low-Power Neural Network Accelerators
- [gdm-doi-10.1073-pnas.2213709120](<https://doi.org/10.1073/pnas.2213709120>) - 2023-04-24 · `affiliated` · Using the Veil of Ignorance to align AI systems with principles of justice
- [gdm-doi-10.1111-desc.13401](<https://doi.org/10.1111/desc.13401>) - 2023-04-23 · `affiliated` · An individual differences perspective on pragmatic abilities in the preschool years
- [gdm-doi-10.1007-s13347-023-00606-x](<https://doi.org/10.1007/s13347-023-00606-x>) - 2023-04-19 · `affiliated` · In Conversation with Artificial Intelligence: Aligning language Models with Human Values
- [gdm-doi-10.1186-s12859-023-05277-1](<https://doi.org/10.1186/s12859-023-05277-1>) - 2023-04-18 · `affiliated` · Benchmarking causal reasoning algorithms for gene expression-based compound mechanism of action analysis
- [gdm-arxiv-2304.13385](<https://doi.org/10.1016/j.media.2023.102807>) - 2023-04-18 · `affiliated` · Low-field magnetic resonance image enhancement via stochastic image quality transfer
- [gdm-doi-10.1038-s42254-023-00569-0](<https://doi.org/10.1038/s42254-023-00569-0>) - 2023-04-17 · `affiliated` · Graph neural networks at the Large Hadron Collider
- [gdm-doi-10.1126-science.adf6369](<https://doi.org/10.1126/science.adf6369>) - 2023-04-13 · `affiliated` · Rethink reporting of evaluation results in AI
- [gdm-doi-10.1101-2023.04.11.536361](<https://doi.org/10.1101/2023.04.11.536361>) - 2023-04-12 · `affiliated` · DiscoGen: Learning to Discover Gene Regulatory Networks
- [gdm-doi-10.1016-j.artint.2023.103919](<https://doi.org/10.1016/j.artint.2023.103919>) - 2023-04-05 · `affiliated` · Reasoning about causality in games
- [gdm-doi-10.1177-26339137231162025](<https://doi.org/10.1177/26339137231162025>) - 2023-04-01 · `affiliated` · A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings
- [gdm-arxiv-2004.01214](<http://dx.doi.org/10.2140/ant.2023.17.93>) - 2023-03-24 · `affiliated` · Constructions of difference sets in nonabelian 2-groups
- [gdm-doi-10.1186-s13195-023-01182-0](<https://doi.org/10.1186/s13195-023-01182-0>) - 2023-03-14 · `affiliated` · Mechanism of action deconvolution of the small-molecule pathological tau aggregation inhibitor Anle138b
- [gdm-arxiv-2301.00810](<https://doi.org/10.1145/3568162.3576989>) - 2023-03-09 · `affiliated` · SIRL
- [gdm-doi-10.7554-elife.80663](<https://doi.org/10.7554/elife.80663>) - 2023-02-27 · `affiliated` · Rapid learning of predictive maps with STDP and theta phase precession
- [gdm-doi-10.1016-j.jmb.2023.168021](<https://doi.org/10.1016/j.jmb.2023.168021>) - 2023-02-22 · `affiliated` · ModelCIF: An Extension of PDBx/mmCIF Data Representation for Computed Structure Models
- [gdm-doi-10.1016-j.sbi.2023.102538](<https://doi.org/10.1016/j.sbi.2023.102538>) - 2023-02-09 · `affiliated` · Everything is connected: Graph neural networks
- [gdm-doi-10.1017-eis.2023.1](<https://doi.org/10.1017/eis.2023.1>) - 2023-02-07 · `affiliated` · Engines of power: Electricity, AI, and general-purpose, military transformations
- [gdm-arxiv-2302.01425](<http://arxiv.org/abs/2302.01425>) - 2023-02-02 · `affiliated` · Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective
- [gdm-doi-10.1016-j.bpj.2022.11.320](<https://doi.org/10.1016/j.bpj.2022.11.320>) - 2023-02-01 · `affiliated` · Applying AI methods to protein biology
- [gdm-doi-10.1038-s41578-022-00513-1](<https://doi.org/10.1038/s41578-022-00513-1>) - 2023-01-24 · `affiliated` · Knowledge-integrated machine learning for materials: lessons from gameplaying and robotics
- [gdm-doi-10.1109-tro.2022.3204509](<https://doi.org/10.1109/tro.2022.3204509>) - 2023-01-20 · `affiliated` · Multimodal Learning of Keypoint Predictive Models for Visual Object Manipulation
- [gdm-doi-10.1103-physrevlett.130.036401](<https://doi.org/10.1103/physrevlett.130.036401>) - 2023-01-20 · `affiliated` · Discovering Quantum Phase Transitions with Fermionic Neural Networks
- [gdm-doi-10.1016-j.neuron.2022.12.028](<https://doi.org/10.1016/j.neuron.2022.12.028>) - 2023-01-13 · `affiliated` · Replay and compositional computation
- [gdm-doi-10.22323-1.430.0036](<http://dx.doi.org/10.22323/1.430.0036>) - 2023-01-09 · `affiliated` · Sampling QCD field configurations with gauge-equivariant flow models
- [gdm-doi-10.32470-ccn.2023.1705-0](<https://doi.org/10.32470/ccn.2023.1705-0>) - 2023-01-01 · `affiliated` · Connecting hippocampal representations to predictive auxiliary tasks in deep RL
- [gdm-doi-10.32470-ccn.2023.1637-0](<https://doi.org/10.32470/ccn.2023.1637-0>) - 2023-01-01 · `affiliated` · Structured Credit Assignment in Mice
- [gdm-doi-10.32470-ccn.2023.1587-0](<https://doi.org/10.32470/ccn.2023.1587-0>) - 2023-01-01 · `affiliated` · The role of subgoals in hierarchical reinforcement learning
- [gdm-doi-10.32470-ccn.2023.1400-0](<https://doi.org/10.32470/ccn.2023.1400-0>) - 2023-01-01 · `affiliated` · Human prefrontal neurons encode economic risk and risk prediction error
- [gdm-doi-10.32470-ccn.2023.1188-0](<https://doi.org/10.32470/ccn.2023.1188-0>) - 2023-01-01 · `affiliated` · Factorization of graphs in the compositional reuse of experience
- [gdm-doi-10.18653-v1-2023.newsum-1.11](<http://dx.doi.org/10.18653/v1/2023.newsum-1.11>) - 2023-01-01 · `affiliated` · Unsupervised Opinion Summarization Using Approximate Geodesics
- [gdm-doi-10.18653-v1-2023.findings-emnlp.926](<https://doi.org/10.18653/v1/2023.findings-emnlp.926>) - 2023-01-01 · `affiliated` · A Comprehensive Evaluation of Tool-Assisted Generation Strategies
- [gdm-doi-10.18653-v1-2023.findings-emnlp.840](<https://doi.org/10.18653/v1/2023.findings-emnlp.840>) - 2023-01-01 · `affiliated` · Pragmatics in Language Grounding: Phenomena, Tasks, and Modeling Approaches
- [gdm-doi-10.18653-v1-2023.findings-emnlp.624](<https://doi.org/10.18653/v1/2023.findings-emnlp.624>) - 2023-01-01 · `affiliated` · In-Context Learning Creates Task Vectors
- [gdm-doi-10.18653-v1-2023.findings-emnlp.538](<http://dx.doi.org/10.18653/v1/2023.findings-emnlp.538>) - 2023-01-01 · `affiliated` · Romanization-based Large-scale Adaptation of Multilingual Language Models
- [gdm-doi-10.18653-v1-2023.findings-emnlp.504](<http://dx.doi.org/10.18653/v1/2023.findings-emnlp.504>) - 2023-01-01 · `affiliated` · POSQA: Probe the World Models of LLMs with Size Comparisons
- [gdm-doi-10.18653-v1-2023.findings-emnlp.39](<https://doi.org/10.18653/v1/2023.findings-emnlp.39>) - 2023-01-01 · `affiliated` · SQLPrompt: In-Context Text-to-SQL with Minimal Labeled Data
- [gdm-doi-10.18653-v1-2023.findings-emnlp.101](<https://doi.org/10.18653/v1/2023.findings-emnlp.101>) - 2023-01-01 · `affiliated` · What Makes Chain-of-Thought Prompting Effective? A Counterfactual Study
- [gdm-doi-10.18653-v1-2023.findings-eacl.90](<https://doi.org/10.18653/v1/2023.findings-eacl.90>) - 2023-01-01 · `affiliated` · Reassessing Evaluation Practices in Visual Question Answering: A Case Study on Out-of-Distribution Generalization
- [gdm-doi-10.18653-v1-2023.findings-acl.62](<https://doi.org/10.18653/v1/2023.findings-acl.62>) - 2023-01-01 · `affiliated` · Triggering Multi-Hop Reasoning for Question Answering in Language Models using Soft Prompts and Random Walks
- [gdm-doi-10.18653-v1-2023.findings-acl.53](<http://dx.doi.org/10.18653/v1/2023.findings-acl.53>) - 2023-01-01 · `affiliated` · Re-appraising the Schema Linking for Text-to-SQL
- [gdm-doi-10.18653-v1-2023.findings-acl.216](<https://doi.org/10.18653/v1/2023.findings-acl.216>) - 2023-01-01 · `affiliated` · Better Zero-Shot Reasoning with Self-Adaptive Prompting
- [gdm-doi-10.18653-v1-2023.emnlp-main.778](<https://doi.org/10.18653/v1/2023.emnlp-main.778>) - 2023-01-01 · `affiliated` · LM vs LM: Detecting Factual Errors via Cross Examination
- [gdm-doi-10.18653-v1-2023.emnlp-main.701](<http://dx.doi.org/10.18653/v1/2023.emnlp-main.701>) - 2023-01-01 · `affiliated` · Don’t Take This Out of Context!: On the Need for Contextual Models and Evaluations for Stylistic Rewriting
- [gdm-doi-10.18653-v1-2023.emnlp-main.607](<https://doi.org/10.18653/v1/2023.emnlp-main.607>) - 2023-01-01 · `affiliated` · CRoW: Benchmarking Commonsense Reasoning in Real-World Tasks
- [gdm-doi-10.18653-v1-2023.emnlp-main.510](<https://doi.org/10.18653/v1/2023.emnlp-main.510>) - 2023-01-01 · `affiliated` · DSI++: Updating Transformer Memory with New Documents
- [gdm-doi-10.18653-v1-2023.emnlp-main.24](<https://doi.org/10.18653/v1/2023.emnlp-main.24>) - 2023-01-01 · `affiliated` · CompoundPiece: Evaluating and Improving Decompounding Performance of Language Models
- [gdm-doi-10.18653-v1-2023.emnlp-main.190](<https://doi.org/10.18653/v1/2023.emnlp-main.190>) - 2023-01-01 · `affiliated` · Evaluating Large Language Models on Controlled Generation Tasks
- [gdm-doi-10.18653-v1-2023.emnlp-main.10](<https://doi.org/10.18653/v1/2023.emnlp-main.10>) - 2023-01-01 · `affiliated` · Evaluating and Modeling Attribution for Cross-Lingual Question Answering
- [gdm-doi-10.18653-v1-2023.emnlp-industry.34](<http://dx.doi.org/10.18653/v1/2023.emnlp-industry.34>) - 2023-01-01 · `affiliated` · Creator Context for Tweet Recommendation
- [gdm-doi-10.18653-v1-2023.emnlp-demo.32](<http://dx.doi.org/10.18653/v1/2023.emnlp-demo.32>) - 2023-01-01 · `affiliated` · SynJax: Structured Probability Distributions for JAX
- [gdm-doi-10.18653-v1-2023.emnlp-demo.13](<https://doi.org/10.18653/v1/2023.emnlp-demo.13>) - 2023-01-01 · `affiliated` · Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning
- [gdm-doi-10.18653-v1-2023.eacl-main.279](<http://dx.doi.org/10.18653/v1/2023.eacl-main.279>) - 2023-01-01 · `affiliated` · Know your audience: specializing grounded language models with listener subtraction
- [gdm-doi-10.18653-v1-2023.cawl-1.8](<https://doi.org/10.18653/v1/2023.cawl-1.8>) - 2023-01-01 · `affiliated` · Lenient Evaluation of Japanese Speech Recognition: Modeling Naturally Occurring Spelling Inconsistency
- [gdm-doi-10.18653-v1-2023.acl-short.22](<http://dx.doi.org/10.18653/v1/2023.acl-short.22>) - 2023-01-01 · `affiliated` · A Natural Bias for Language Generation Models
- [gdm-doi-10.18653-v1-2023.acl-long.807](<https://doi.org/10.18653/v1/2023.acl-long.807>) - 2023-01-01 · `affiliated` · To Adapt or to Annotate: Challenges and Interventions for Domain Adaptation in Open-Domain Question Answering
- [gdm-doi-10.18653-v1-2023.acl-long.784](<https://doi.org/10.18653/v1/2023.acl-long.784>) - 2023-01-01 · `affiliated` · QUEST: A Retrieval Dataset of Entity-Seeking Queries with Implicit Set Operations
- [gdm-doi-10.18653-v1-2023.acl-long.714](<https://doi.org/10.18653/v1/2023.acl-long.714>) - 2023-01-01 · `affiliated` · MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering
- [gdm-doi-10.18653-v1-2023.acl-long.66](<https://doi.org/10.18653/v1/2023.acl-long.66>) - 2023-01-01 · `affiliated` · Improving Pretraining Techniques for Code-Switched NLP
- [gdm-doi-10.18653-v1-2023.acl-long.477](<http://dx.doi.org/10.18653/v1/2023.acl-long.477>) - 2023-01-01 · `affiliated` · On “Scientific Debt” in NLP: A Case for More Rigour in Language Model Pre-Training Research
- [gdm-doi-10.18653-v1-2023.acl-long.262](<https://doi.org/10.18653/v1/2023.acl-long.262>) - 2023-01-01 · `affiliated` · Reward Gaming in Conditional Text Generation
- [gdm-doi-10.18564-jasss.5087](<https://doi.org/10.18564/jasss.5087>) - 2023-01-01 · `affiliated` · Learning Interpretable Logic for Agent-Based Models from Domain Independent Primitives
- [gdm-doi-10.1162-tacl_a_00583](<https://doi.org/10.1162/tacl_a_00583>) - 2023-01-01 · `affiliated` · Conditional Generation with a Question-Answering Blueprint
- [gdm-doi-10.1162-nol_a_00110](<https://doi.org/10.1162/nol_a_00110>) - 2023-01-01 · `affiliated` · Neural Correlates of Object-Extracted Relative Clause Processing Across English and Chinese
- [gdm-doi-10.1162-isal_a_00651](<https://doi.org/10.1162/isal_a_00651>) - 2023-01-01 · `affiliated` · Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization
- [gdm-doi-10.1162-coli_a_00481](<https://doi.org/10.1162/coli_a_00481>) - 2023-01-01 · `affiliated` · Machine Learning for Ancient Languages: A Survey
- [gdm-doi-10.1109-lcsys.2023.3260731](<https://doi.org/10.1109/lcsys.2023.3260731>) - 2023-01-01 · `affiliated` · DRIP: Domain Refinement Iteration With Polytopes for Backward Reachability Analysis of Neural Feedback Loops
- [gdm-doi-10.1038-s41591-022-02137-y](<https://doi.org/10.1038/s41591-022-02137-y>) - 2023-01-01 · `affiliated` · A participatory initiative to include LGBT+ voices in AI for mental health
- [gdm-doi-10.1016-b978-0-323-85280-7.00005-1](<https://doi.org/10.1016/b978-0-323-85280-7.00005-1>) - 2023-01-01 · `affiliated` · Machine learning in connectomics: from representation learning to model fitting
- [gdm-doi-10.1007-978-3-031-30105-6_49](<https://doi.org/10.1007/978-3-031-30105-6_49>) - 2023-01-01 · `affiliated` · Correlation Based Semantic Transfer with Application to Domain Adaptation
- [gdm-doi-10.1007-978-3-031-26316-3_23](<https://doi.org/10.1007/978-3-031-26316-3_23>) - 2023-01-01 · `affiliated` · Is an Object-Centric Video Representation Beneficial for Transfer?
- [gdm-doi-10.1007-978-3-031-26293-7_40](<https://doi.org/10.1007/978-3-031-26293-7_40>) - 2023-01-01 · `affiliated` · Compressed Vision for Efficient Video Understanding
- [gdm-doi-10.1007-978-3-031-19907-3_18](<https://doi.org/10.1007/978-3-031-19907-3_18>) - 2023-01-01 · `affiliated` · Reinforcement Learning with Information-Theoretic Actuation
- [gdm-arxiv-2305.16634](<https://doi.org/10.1007/978-3-031-37196-7_9>) - 2023-01-01 · `affiliated` · Machine Learning for Protein Engineering

### 17.5 2022（129 条）

- [gdm-doi-10.1613-jair.1.13673](<https://doi.org/10.1613/jair.1.13673>) - 2022-12-22 · `affiliated` · Towards Continual Reinforcement Learning: A Review and Perspectives
- [gdm-doi-10.3389-fncom.2022.1060101](<https://doi.org/10.3389/fncom.2022.1060101>) - 2022-12-21 · `affiliated` · Importance of prefrontal meta control in human-like reinforcement learning
- [gdm-doi-10.1021-acs.jctc.2c00934](<https://doi.org/10.1021/acs.jctc.2c00934>) - 2022-12-12 · `affiliated` · <tt>ipie</tt>: A Python-Based Auxiliary-Field Quantum Monte Carlo Program with Flexibility and Efficiency on CPUs and GPUs
- [gdm-doi-10.3390-e24121791](<https://doi.org/10.3390/e24121791>) - 2022-12-08 · `affiliated` · Compositional Sequence Generation in the Entorhinal–Hippocampal System
- [gdm-doi-10.1101-2022.12.08.519453](<https://doi.org/10.1101/2022.12.08.519453>) - 2022-12-08 · `affiliated` · BCI learning phenomena can be explained by gradient-based optimization
- [gdm-arxiv-2203.07814](<https://doi.org/10.1126/science.abq1158>) - 2022-12-08 · `affiliated` · Competition-level code generation with AlphaCode
- [gdm-doi-10.1038-s41467-022-34473-5](<https://doi.org/10.1038/s41467-022-34473-5>) - 2022-12-06 · `affiliated` · Negotiation and honesty in artificial intelligence methods for the board game of Diplomacy
- [gdm-arxiv-2212.03319](<http://arxiv.org/abs/2212.03319>) - 2022-12-06 · `affiliated` · Understanding Self-Predictive Learning for Reinforcement Learning
- [gdm-doi-10.1126-science.add4679](<https://doi.org/10.1126/science.add4679>) - 2022-12-01 · `affiliated` · Mastering the game of Stratego with model-free multiagent reinforcement learning
- [gdm-arxiv-2211.15646](<http://arxiv.org/abs/2211.15646>) - 2022-11-28 · `affiliated` · Beyond Invariance: Test-Time Label-Shift Adaptation for Distributions with "Spurious" Correlations
- [gdm-doi-10.1007-s11222-022-10183-2](<https://doi.org/10.1007/s11222-022-10183-2>) - 2022-11-25 · `affiliated` · Variance reduction for Metropolis–Hastings samplers
- [gdm-arxiv-2211.10515](<http://arxiv.org/abs/2211.10515>) - 2022-11-18 · `affiliated` · Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments
- [gdm-arxiv-2111.15161](<https://doi.org/10.1090/ert/624>) - 2022-11-16 · `affiliated` · Towards combinatorial invariance for Kazhdan-Lusztig polynomials
- [gdm-doi-10.21203-rs.3.rs-2231672-v1](<https://doi.org/10.21203/rs.3.rs-2231672/v1>) - 2022-11-14 · `affiliated` · Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians (CoDoC)
- [gdm-arxiv-2111.09259](<https://doi.org/10.1073/pnas.2206625119>) - 2022-11-14 · `affiliated` · Acquisition of chess knowledge in AlphaZero
- [gdm-doi-10.1073-pnas.2206704119](<https://doi.org/10.1073/pnas.2206704119>) - 2022-11-02 · `affiliated` · Adult neurogenesis acts as a neural regularizer
- [gdm-doi-10.1111-rssa.12976](<https://doi.org/10.1111/rssa.12976>) - 2022-11-01 · `affiliated` · Authors’ Reply to the Discussion of ‘Efficient Bayesian Inference of Instantaneous Reproduction Numbers at Fine Spatial Scales, with an Application to Mapping and Nowcasting the Covid-19 Epidemic in British Local Authorities’ by Teh et al. in Session 2 of the Royal Statistical Society’s Special Topic Meeting on COVID-19 Transmission: 11 June 2021
- [gdm-doi-10.1111-rssa.12971](<https://doi.org/10.1111/rssa.12971>) - 2022-11-01 · `affiliated` · Efficient Bayesian inference of Instantaneous Reproduction Numbers at Fine Spatial Scales, with an Application to Mapping and Nowcasting the Covid-19 Epidemic in British Local Authorities
- [gdm-doi-10.1613-jair.1.13854](<https://doi.org/10.1613/jair.1.13854>) - 2022-10-27 · `affiliated` · Low-Rank Representation of Reinforcement Learning Policies
- [gdm-arxiv-2210.14756](<http://arxiv.org/abs/2210.14756>) - 2022-10-26 · `affiliated` · Maximum Likelihood Learning of Unnormalized Models for Simulation-Based Inference
- [gdm-doi-10.21203-rs.3.rs-2188216-v1](<https://doi.org/10.21203/rs.3.rs-2188216/v1>) - 2022-10-24 · `affiliated` · Non-Chaotic Limit Sets in Multi-Agent Learning
- [gdm-doi-10.1109-iros47612.2022.9981648](<https://doi.org/10.1109/iros47612.2022.9981648>) - 2022-10-23 · `affiliated` · Learning Coordinated Terrain-Adaptive Locomotion by Imitating a Centroidal Dynamics Planner
- [gdm-doi-10.1109-iros47612.2022.9981126](<https://doi.org/10.1109/iros47612.2022.9981126>) - 2022-10-23 · `affiliated` · How to Spend Your Robot Time: Bridging Kickstarting and Offline Reinforcement Learning for Vision-based Robotic Manipulation
- [gdm-doi-10.1103-physrevd.106.074506](<https://doi.org/10.1103/physrevd.106.074506>) - 2022-10-18 · `affiliated` · Gauge-equivariant flow models for sampling in lattice field theories with pseudofermions
- [gdm-doi-10.1038-s41598-022-20234-3](<https://doi.org/10.1038/s41598-022-20234-3>) - 2022-10-08 · `affiliated` · Designing all-pay auctions using deep learning and multi-agent simulation
- [gdm-arxiv-2209.07572](<https://doi.org/10.1145/3551624.3555290>) - 2022-10-06 · `affiliated` · Power to the People? Opportunities and Challenges for Participatory AI
- [gdm-doi-10.1038-s41586-022-05172-4](<https://doi.org/10.1038/s41586-022-05172-4>) - 2022-10-05 · `affiliated` · Discovering faster matrix multiplication algorithms with reinforcement learning
- [gdm-arxiv-2209.14414](<http://arxiv.org/abs/2209.14414>) - 2022-09-28 · `affiliated` · Optimistic Posterior Sampling for Reinforcement Learning with Few Samples and Tight Guarantees
- [gdm-doi-10.1038-s41467-022-33175-2](<https://doi.org/10.1038/s41467-022-33175-2>) - 2022-09-20 · `affiliated` · Structure of the PAPP-ABP5 complex reveals mechanism of substrate recognition
- [gdm-doi-10.1016-j.tics.2022.08.004](<https://doi.org/10.1016/j.tics.2022.08.004>) - 2022-09-20 · `affiliated` · Realizing the promise of AI: a new calling for cognitive science
- [gdm-doi-10.3233-aic-220113](<https://doi.org/10.3233/aic-220113>) - 2022-09-06 · `affiliated` · Developing, evaluating and scaling learning agents in multi-agent environments
- [gdm-doi-10.1126-scirobotics.abo0235](<https://doi.org/10.1126/scirobotics.abo0235>) - 2022-08-31 · `affiliated` · From motor control to team play in simulated humanoid football
- [gdm-doi-10.7554-elife.64575](<https://doi.org/10.7554/elife.64575>) - 2022-08-17 · `affiliated` · Value representations in the rodent orbitofrontal cortex drive learning, not choice
- [gdm-doi-10.31234-osf.io-tnf4e](<https://doi.org/10.31234/osf.io/tnf4e>) - 2022-08-15 · `affiliated` · Artificial moral cognition: Learning from developmental psychology
- [gdm-arxiv-2208.05568](<http://arxiv.org/abs/2208.05568>) - 2022-08-10 · `affiliated` · The emergence of division of labor through decentralized social sanctioning
- [gdm-doi-10.1111-phc3.12865](<https://doi.org/10.1111/phc3.12865>) - 2022-08-07 · `affiliated` · Reinforcement learning: A brief guide for philosophers of mind
- [gdm-doi-10.1126-science.abq4282](<https://doi.org/10.1126/science.abq4282>) - 2022-08-04 · `affiliated` · Response to Comment on “Pushing the frontiers of density functionals by solving the fractional electron problem”
- [gdm-arxiv-2009.10709](<https://doi.org/10.22331/q-2022-08-04-773>) - 2022-08-04 · `affiliated` · Fast Black-Box Quantum State Preparation
- [gdm-doi-10.1103-physrevd.106.014514](<https://doi.org/10.1103/physrevd.106.014514>) - 2022-07-29 · `affiliated` · Flow-based sampling in the lattice Schwinger model at criticality
- [gdm-doi-10.1109-cec55065.2022.9870269](<https://doi.org/10.1109/cec55065.2022.9870269>) - 2022-07-18 · `affiliated` · Quantum Circuit Evolution on NISQ Devices
- [gdm-doi-10.1038-s41467-022-30695-9](<https://doi.org/10.1038/s41467-022-30695-9>) - 2022-07-15 · `affiliated` · The Medical Segmentation Decathlon
- [gdm-doi-10.1038-s41562-022-01394-8](<https://doi.org/10.1038/s41562-022-01394-8>) - 2022-07-11 · `affiliated` · Intuitive physics learning in a deep-learning model inspired by developmental psychology
- [gdm-doi-10.1613-jair.1.13187](<https://doi.org/10.1613/jair.1.13187>) - 2022-07-08 · `affiliated` · Evolutionary Dynamics and Phi-Regret Minimization in Games
- [gdm-doi-10.1038-s41467-022-31564-1](<https://doi.org/10.1038/s41467-022-31564-1>) - 2022-07-06 · `affiliated` · Discovery of archaeal fusexins homologous to eukaryotic HAP2/GCS1 gamete fusion proteins
- [gdm-doi-10.1038-s41562-022-01383-x](<https://doi.org/10.1038/s41562-022-01383-x>) - 2022-07-04 · `affiliated` · Human-centred mechanism design with Democratic AI
- [gdm-doi-10.24963-ijcai.2022-811](<https://doi.org/10.24963/ijcai.2022/811>) - 2022-07-01 · `affiliated` · Ethics and Governance of Artificial Intelligence: A Survey of Machine Learning Researchers (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2022-799](<https://doi.org/10.24963/ijcai.2022/799>) - 2022-07-01 · `affiliated` · Making Sense of Raw Input (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2022-742](<https://doi.org/10.24963/ijcai.2022/742>) - 2022-07-01 · `affiliated` · Detect, Understand, Act: A Neuro-Symbolic Hierarchical Reinforcement Learning Framework (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2022-730](<https://doi.org/10.24963/ijcai.2022/730>) - 2022-07-01 · `affiliated` · On the Expressivity of Markov Reward (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2022-484](<https://doi.org/10.24963/ijcai.2022/484>) - 2022-07-01 · `affiliated` · Approximate Exploitability: Learning a Best Response
- [gdm-doi-10.1609-aaai.v36i9.21242](<https://doi.org/10.1609/aaai.v36i9.21242>) - 2022-06-28 · `affiliated` · A Complete Criterion for Value of Information in Soluble Influence Diagrams
- [gdm-doi-10.1609-aaai.v36i9.21186](<https://doi.org/10.1609/aaai.v36i9.21186>) - 2022-06-28 · `affiliated` · Path-Specific Objectives for Safer Agent Incentives
- [gdm-doi-10.1609-aaai.v36i9.21182](<https://doi.org/10.1609/aaai.v36i9.21182>) - 2022-06-28 · `affiliated` · Why Fair Labels Can Yield Unfair Predictions: Graphical Conditions for Introduced Unfairness
- [gdm-doi-10.1609-aaai.v36i8.20887](<https://doi.org/10.1609/aaai.v36i8.20887>) - 2022-06-28 · `affiliated` · Multi-Agent Reinforcement Learning with General Utilities via Decentralized Shadow Reward Actor-Critic
- [gdm-doi-10.1609-aaai.v36i8.20798](<https://doi.org/10.1609/aaai.v36i8.20798>) - 2022-06-28 · `affiliated` · Online Apprenticeship Learning
- [gdm-doi-10.1609-aaai.v36i8.20792](<https://doi.org/10.1609/aaai.v36i8.20792>) - 2022-06-28 · `affiliated` · Chaining Value Functions for Off-Policy Learning
- [gdm-doi-10.1609-aaai.v36i7.20681](<https://doi.org/10.1609/aaai.v36i7.20681>) - 2022-06-28 · `affiliated` · Introducing Symmetries to Black Box Meta Reinforcement Learning
- [gdm-doi-10.1609-aaai.v36i6.20660](<https://doi.org/10.1609/aaai.v36i6.20660>) - 2022-06-28 · `affiliated` · Learning Expected Emphatic Traces for Deep RL
- [gdm-doi-10.1609-aaai.v36i6.20623](<https://doi.org/10.1609/aaai.v36i6.20623>) - 2022-06-28 · `affiliated` · Algorithmic Concept-Based Explainable Reasoning
- [gdm-arxiv-2112.06751](<https://ojs.aaai.org/index.php/AAAI/article/view/20465>) - 2022-06-28 · `affiliated` · Role of Human-AI Interaction in Selective Prediction
- [gdm-arxiv-2109.09717](<https://doi.org/10.1609/aaai.v36i9.21173>) - 2022-06-28 · `affiliated` · Generalization in Mean Field Games by Learning Master Policies
- [gdm-doi-10.1145-3531146.3533088](<https://doi.org/10.1145/3531146.3533088>) - 2022-06-20 · `affiliated` · Taxonomy of Risks posed by Language Models
- [gdm-doi-10.1038-s41467-022-31214-6](<https://doi.org/10.1038/s41467-022-31214-6>) - 2022-06-20 · `affiliated` · Structural basis of template strand deoxyuridine promoter recognition by a viral RNA polymerase
- [gdm-arxiv-2205.13740](<https://doi.org/10.1145/3531146.3533161>) - 2022-06-20 · `affiliated` · Subverting machines, fluctuating identities: Re-learning human categorization
- [gdm-doi-10.1063-5.0088981](<https://doi.org/10.1063/5.0088981>) - 2022-06-17 · `affiliated` · Performance of a one-parameter correlation factor for transcorrelation: Study on a series of second row atomic and molecular systems
- [gdm-arxiv-2206.08332](<http://arxiv.org/abs/2206.08332>) - 2022-06-16 · `affiliated` · BYOL-Explore: Exploration by Bootstrapped Prediction
- [gdm-arxiv-2206.08155](<http://arxiv.org/abs/2206.08155>) - 2022-06-16 · `affiliated` · Zero-Shot Video Question Answering via Frozen Bidirectional Language Models
- [gdm-doi-10.1038-s41586-022-04798-8](<https://doi.org/10.1038/s41586-022-04798-8>) - 2022-06-08 · `affiliated` · Core control principles of the eukaryotic cell cycle
- [gdm-doi-10.1101-2022.06.03.494671](<https://doi.org/10.1101/2022.06.03.494671>) - 2022-06-04 · `affiliated` · A probabilistic successor representation for context-dependent prediction
- [gdm-doi-10.1111-cogs.13146](<https://doi.org/10.1111/cogs.13146>) - 2022-06-01 · `affiliated` · The Emergence of Gender Associations in Child Language Development
- [gdm-doi-10.1109-cvpr52688.2022.01862](<https://doi.org/10.1109/cvpr52688.2022.01862>) - 2022-06-01 · `affiliated` · Neural Mean Discrepancy for Efficient Out-of-Distribution Detection
- [gdm-doi-10.1109-cvpr52688.2022.01357](<https://doi.org/10.1109/cvpr52688.2022.01357>) - 2022-06-01 · `affiliated` · Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos
- [gdm-doi-10.1109-cvpr52688.2022.01033](<https://doi.org/10.1109/cvpr52688.2022.01033>) - 2022-06-01 · `affiliated` · More than Words: In-the-Wild Visually-Driven Prosody for Text-to-Speech
- [gdm-doi-10.1109-cvpr52688.2022.00727](<https://doi.org/10.1109/cvpr52688.2022.00727>) - 2022-06-01 · `affiliated` · Non-isotropy Regularization for Proxy-based Deep Metric Learning
- [gdm-doi-10.1109-cvpr52688.2022.00608](<https://doi.org/10.1109/cvpr52688.2022.00608>) - 2022-06-01 · `affiliated` · Input-level Inductive Biases for 3D Reconstruction
- [gdm-doi-10.1109-cvpr52688.2022.00373](<https://doi.org/10.1109/cvpr52688.2022.00373>) - 2022-06-01 · `affiliated` · Kubric: A scalable dataset generator
- [gdm-arxiv-2203.16434](<https://doi.org/10.1109/cvpr52688.2022.01595>) - 2022-06-01 · `affiliated` · TubeDETR: Spatio-Temporal Video Grounding with Transformers
- [gdm-arxiv-2203.08543](<https://doi.org/10.1109/cvpr52688.2022.01570>) - 2022-06-01 · `affiliated` · Integrating Language Guidance into Vision-based Deep Metric Learning
- [gdm-doi-10.1109-icra46639.2022.9812312](<https://doi.org/10.1109/icra46639.2022.9812312>) - 2022-05-23 · `affiliated` · Offline Meta-Reinforcement Learning for Industrial Insertion
- [gdm-doi-10.1109-icra46639.2022.9812209](<https://doi.org/10.1109/icra46639.2022.9812209>) - 2022-05-23 · `affiliated` · Few-Shot Keypoint Detection as Task Adaptation via Latent Embeddings
- [gdm-doi-10.1038-s41598-022-12547-0](<https://doi.org/10.1038/s41598-022-12547-0>) - 2022-05-23 · `affiliated` · Multiagent off-screen behavior prediction in football
- [gdm-arxiv-2105.06948](<https://doi.org/10.1038/s41586-022-04743-9>) - 2022-05-19 · `affiliated` · People construct simplified mental representations to plan
- [gdm-arxiv-2205.07704](<http://arxiv.org/abs/2205.07704>) - 2022-05-16 · `affiliated` · From Dirichlet to Rubin: Optimistic Exploration in RL without Bonuses
- [gdm-doi-10.1109-tpami.2022.3173208](<https://doi.org/10.1109/tpami.2022.3173208>) - 2022-05-09 · `affiliated` · Learning to Answer Visual Questions From Web Videos
- [gdm-arxiv-2111.08350](<https://doi.org/10.65109/whor7931>) - 2022-05-09 · `affiliated` · Learning Equilibria in Mean-Field Games: Introducing Mean-Field PSRO
- [gdm-arxiv-2106.01285](<https://doi.org/10.65109/bmei7345>) - 2022-05-09 · `affiliated` · Sample-based Approximation of Nash in Large Many-Player Games via Gradient Descent
- [gdm-arxiv-2010.00575](<https://doi.org/10.65109/ztul8959>) - 2022-05-09 · `affiliated` · D3C: Reducing the Price of Anarchy in Multi-Agent Learning
- [gdm-doi-10.1109-sp46214.2022.9833677](<https://doi.org/10.1109/sp46214.2022.9833677>) - 2022-05-01 · `affiliated` · Reconstructing Training Data with Informed Adversaries
- [gdm-doi-10.1109-icassp43922.2022.9746790](<https://doi.org/10.1109/icassp43922.2022.9746790>) - 2022-04-27 · `affiliated` · Towards Learning Universal Audio Representations
- [gdm-doi-10.1038-s41580-022-00488-5](<https://doi.org/10.1038/s41580-022-00488-5>) - 2022-04-27 · `affiliated` · The prospects and opportunities of protein structure prediction with AI
- [gdm-arxiv-2111.08696](<https://doi.org/10.1088/2632-2153/ac6b16>) - 2022-04-27 · `affiliated` · Normalizing flows for atomic solids
- [gdm-doi-10.1038-s41746-022-00595-9](<https://doi.org/10.1038/s41746-022-00595-9>) - 2022-04-22 · `affiliated` · Machine learning and health need better values
- [gdm-doi-10.1016-j.neunet.2022.03.037](<https://doi.org/10.1016/j.neunet.2022.03.037>) - 2022-04-19 · `affiliated` · Deep learning, reinforcement learning, and world models
- [gdm-doi-10.3389-fncom.2022.836498](<https://doi.org/10.3389/fncom.2022.836498>) - 2022-04-14 · `affiliated` · Symmetry-Based Representations for Artificial and Biological General Intelligence
- [gdm-doi-10.1101-2022.04.07.487582](<https://doi.org/10.1101/2022.04.07.487582>) - 2022-04-10 · `affiliated` · Can neurogenesis act as a neural regularizer?
- [gdm-arxiv-2203.16177](<http://arxiv.org/abs/2203.16177>) - 2022-03-30 · `affiliated` · Marginalized Operators for Off-policy Reinforcement Learning
- [gdm-doi-10.1093-oxfordhb-9780198857815.013.18](<https://doi.org/10.1093/oxfordhb/9780198857815.013.18>) - 2022-03-18 · `affiliated` · The Challenge of Value Alignment
- [gdm-arxiv-2102.08370](<https://doi.org/10.1007/s10458-022-09548-8>) - 2022-03-18 · `affiliated` · Quantifying the effects of environment and population diversity in multi-agent reinforcement learning
- [gdm-doi-10.1007-s11222-022-10088-0](<https://doi.org/10.1007/s11222-022-10088-0>) - 2022-03-13 · `affiliated` · Sequential changepoint detection in neural networks with checkpoints
- [gdm-doi-10.1038-s41586-022-04448-z](<https://doi.org/10.1038/s41586-022-04448-z>) - 2022-03-09 · `affiliated` · Restoring and attributing ancient texts using deep neural networks
- [gdm-doi-10.1038-s41594-022-00729-3](<https://doi.org/10.1038/s41594-022-00729-3>) - 2022-03-01 · `affiliated` · Structure of the decoy module of human glycoprotein 2 and uromodulin and its interaction with bacterial adhesin FimH
- [gdm-arxiv-2106.10015](<https://doi.org/10.1371/journal.pcbi.1009882>) - 2022-02-28 · `affiliated` · Meta-control of social learning strategies
- [gdm-arxiv-2202.08417](<http://arxiv.org/abs/2202.08417>) - 2022-02-17 · `affiliated` · Retrieval-Augmented Reinforcement Learning
- [gdm-doi-10.1038-s41586-021-04301-9](<https://doi.org/10.1038/s41586-021-04301-9>) - 2022-02-16 · `affiliated` · Magnetic control of tokamak plasmas through deep reinforcement learning
- [gdm-doi-10.3389-fcomp.2022.810358](<https://doi.org/10.3389/fcomp.2022.810358>) - 2022-02-08 · `affiliated` · The Brain-Computer Metaphor Debate Is Useless: A Matter of Semantics
- [gdm-arxiv-1906.05433](<https://deepmind.google/blog/the-podcast-episode-5-out-of-the-lab/>) - 2022-02-07 · `affiliated` · Tackling Climate Change with Machine Learning
- [gdm-doi-10.1111-rssb.12459](<https://doi.org/10.1111/rssb.12459>) - 2022-02-01 · `affiliated` · Sebastian Dietz’s Contribution to the Discussion of ‘Gaussian Differential Privacy’ by Dong <i>et al</i>.
- [gdm-arxiv-2201.12909](<http://arxiv.org/abs/2201.12909>) - 2022-01-30 · `affiliated` · Scaling Gaussian Process Optimization by Evaluating a Few Unique Candidates Multiple Times
- [gdm-doi-10.1145-3460349](<https://doi.org/10.1145/3460349>) - 2022-01-24 · `affiliated` · Reimagining chess with AlphaZero
- [gdm-doi-10.1073-pnas.2106028118](<https://doi.org/10.1073/pnas.2106028118>) - 2022-01-12 · `affiliated` · Spurious normativity enhances learning of compliance and enforcement behavior in artificial agents
- [gdm-doi-10.32470-ccn.2022.1229-0](<https://doi.org/10.32470/ccn.2022.1229-0>) - 2022-01-01 · `affiliated` · Continual Reinforcement Learning with Multi-Timescale Successor Features
- [gdm-doi-10.32470-ccn.2022.1032-0](<https://doi.org/10.32470/ccn.2022.1032-0>) - 2022-01-01 · `affiliated` · Spatially-embedded Recurrent Neural Networks: Bridging common structural and functional findings in neuroscience, including small-worldness, functional clustering in space and mixed selectivity
- [gdm-doi-10.18653-v1-2022.gem-1.31](<http://dx.doi.org/10.18653/v1/2022.gem-1.31>) - 2022-01-01 · `affiliated` · Control Prefixes for Parameter-Efficient Text Generation
- [gdm-doi-10.18653-v1-2022.acl-tutorials.7](<https://doi.org/10.18653/v1/2022.acl-tutorials.7>) - 2022-01-01 · `affiliated` · Vision-Language Pretraining: Current Trends and the Future
- [gdm-doi-10.1162-tacl_a_00526](<https://doi.org/10.1162/tacl_a_00526>) - 2022-01-01 · `affiliated` · Transformer Grammars: Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale
- [gdm-doi-10.1162-tacl_a_00476](<https://doi.org/10.1162/tacl_a_00476>) - 2022-01-01 · `affiliated` · Relational Memory-Augmented Language Models
- [gdm-doi-10.1162-opmi_a_00069](<https://doi.org/10.1162/opmi_a_00069>) - 2022-01-01 · `affiliated` · Modeling Individual Differences in Children’s Information Integration During Pragmatic Word Learning
- [gdm-doi-10.1093-gigascience-giac118](<https://doi.org/10.1093/gigascience/giac118>) - 2022-01-01 · `affiliated` · 3D-Beacons: decreasing the gap between protein sequences and structures through a federated network of protein structure data resources
- [gdm-doi-10.1038-s41592-021-01362-6](<https://doi.org/10.1038/s41592-021-01362-6>) - 2022-01-01 · `affiliated` · Protein structure predictions to atomic accuracy with AlphaFold
- [gdm-doi-10.1017-s0140525x22001364](<https://doi.org/10.1017/s0140525x22001364>) - 2022-01-01 · `affiliated` · What is the simplest model that can account for high-fidelity imitation?
- [gdm-doi-10.1017-s0140525x21001357](<https://doi.org/10.1017/s0140525x21001357>) - 2022-01-01 · `affiliated` · Learning agents that acquire representations of social groups
- [gdm-doi-10.1017-s0140525x21000224](<https://doi.org/10.1017/s0140525x21000224>) - 2022-01-01 · `affiliated` · Publishing fast and slow: A path toward generalizability in psychology and AI
- [gdm-doi-10.1007-978-3-031-19778-9_31](<https://doi.org/10.1007/978-3-031-19778-9_31>) - 2022-01-01 · `affiliated` · Latent Space Smoothing for Individually Fair Representations
- [gdm-doi-10.1007-978-3-031-16980-9_7](<https://doi.org/10.1007/978-3-031-16980-9_7>) - 2022-01-01 · `affiliated` · Morphology-Preserving Autoregressive 3D Generative Modelling of the Brain
- [gdm-doi-10.1007-978-3-031-16749-2](<https://doi.org/10.1007/978-3-031-16749-2>) - 2022-01-01 · `affiliated` · Uncertainty for Safe Utilization of Machine Learning in Medical Imaging
- [gdm-doi-10.1007-978-3-031-12053-4_31](<https://doi.org/10.1007/978-3-031-12053-4_31>) - 2022-01-01 · `affiliated` · Utility of Equivariant Message Passing in Cortical Mesh Segmentation
- [gdm-arxiv-2207.12909](<https://doi.org/10.1007/978-3-031-19769-7_14>) - 2022-01-01 · `affiliated` · AlignSDF: Pose-Aligned Signed Distance Fields for Hand-Object Reconstruction
- [gdm-arxiv-2203.08777](<https://doi.org/10.1007/978-3-031-19812-0_8>) - 2022-01-01 · `affiliated` · Object Discovery and Representation Networks
- [gdm-arxiv-2110.02450](<https://doi.org/10.1007/978-3-030-93758-4_1>) - 2022-01-01 · `affiliated` · Reward-Punishment Symmetric Universal Intelligence

### 17.6 2021（201 条）

- [gdm-arxiv-2106.05934](<https://doi.org/10.1103/physrevd.104.114507>) - 2021-12-15 · `affiliated` · Flow-based sampling for fermionic lattice field theories
- [gdm-doi-10.1126-science.abj6511](<https://doi.org/10.1126/science.abj6511>) - 2021-12-09 · `affiliated` · Pushing the frontiers of density functionals by solving the fractional electron problem
- [gdm-doi-10.1038-s41586-021-04086-x](<https://doi.org/10.1038/s41586-021-04086-x>) - 2021-12-01 · `affiliated` · Advancing mathematics by guiding human intuition with AI
- [gdm-doi-10.1007-s11238-021-09850-z](<https://doi.org/10.1007/s11238-021-09850-z>) - 2021-12-01 · `affiliated` · Classification by decomposition: a novel approach to classification of symmetric &#36;&#36;2&#92;&#92;times 2&#36;&#36; games
- [gdm-arxiv-2006.14304](<https://doi.org/10.1038/s41467-021-26751-5>) - 2021-11-09 · `affiliated` · Unsupervised deep learning identifies semantic disentanglement in single inferotemporal face patch neurons
- [gdm-arxiv-2111.02338](<https://pubmed.ncbi.nlm.nih.gov/36467015>) - 2021-11-03 · `affiliated` · Drop, Swap, and Generate: A Self-Supervised Approach for Generating Neural Activity
- [gdm-doi-10.1242-dev.199664](<https://doi.org/10.1242/dev.199664>) - 2021-11-01 · `affiliated` · Deep learning is widely applicable to phenotyping embryonic development and disease
- [gdm-doi-10.1101-2021.10.28.466307](<https://doi.org/10.1101/2021.10.28.466307>) - 2021-10-29 · `affiliated` · Optimal Design of Stochastic DNA Synthesis Protocols based on Generative Sequence Models
- [gdm-arxiv-2108.11482](<https://doi.org/10.1145/3459637.3481916>) - 2021-10-26 · `affiliated` · ETA Prediction with Graph Neural Networks in Google Maps
- [gdm-arxiv-2104.03829](<https://doi.org/10.1016/j.media.2021.102274>) - 2021-10-22 · `affiliated` · Does your dermatology classifier know what it doesn’t know? Detecting the long-tail of unseen conditions
- [gdm-doi-10.1093-nar-gkab1061](<https://doi.org/10.1093/nar/gkab1061>) - 2021-10-19 · `affiliated` · AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models
- [gdm-doi-10.1016-j.neunet.2021.10.004](<https://doi.org/10.1016/j.neunet.2021.10.004>) - 2021-10-19 · `affiliated` · Meta-learning, social cognition and consciousness in brains and machines
- [gdm-doi-10.1002-jeab.721](<https://doi.org/10.1002/jeab.721>) - 2021-10-13 · `affiliated` · Dreading the pain of others? Altruistic responses to others' pain underestimate dread
- [gdm-doi-10.1101-2021.10.04.463034](<https://doi.org/10.1101/2021.10.04.463034>) - 2021-10-04 · `affiliated` · Protein complex prediction with AlphaFold-Multimer
- [gdm-doi-10.1002-prot.26257](<https://doi.org/10.1002/prot.26257>) - 2021-10-04 · `affiliated` · Applying and improving <scp>AlphaFold</scp> at <scp>CASP14</scp>
- [gdm-doi-10.1109-iccv48922.2021.01027](<https://doi.org/10.1109/iccv48922.2021.01027>) - 2021-10-01 · `affiliated` · PARTS: Unsupervised segmentation with slots, attention and independence maximization
- [gdm-doi-10.1109-iccv48922.2021.00991](<https://doi.org/10.1109/iccv48922.2021.00991>) - 2021-10-01 · `affiliated` · Divide and Contrast: Self-supervised Learning from Uncurated Data
- [gdm-doi-10.1109-iccv48922.2021.00945](<https://doi.org/10.1109/iccv48922.2021.00945>) - 2021-10-01 · `affiliated` · With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations
- [gdm-doi-10.1038-s41592-021-01252-x](<https://doi.org/10.1038/s41592-021-01252-x>) - 2021-10-01 · `affiliated` · Effective gene expression prediction from sequence by integrating long-range interactions
- [gdm-arxiv-2103.16559](<https://doi.org/10.1109/iccv48922.2021.00129>) - 2021-10-01 · `affiliated` · Broaden Your Views for Self-Supervised Video Learning
- [gdm-arxiv-2103.10957](<https://doi.org/10.1109/iccv48922.2021.00993>) - 2021-10-01 · `affiliated` · Efficient Visual Pretraining with Contrastive Detection
- [gdm-arxiv-2101.04117](<https://doi.org/10.1073/pnas.2026053118>) - 2021-10-01 · `affiliated` · A Bayesian neural network predicts the dissolution of compact planetary systems
- [gdm-arxiv-2012.00451](<https://doi.org/10.1109/iccv48922.2021.00171>) - 2021-10-01 · `affiliated` · Just Ask: Learning to Answer Questions from Millions of Narrated Videos
- [gdm-doi-10.1038-s41586-021-03854-z](<https://doi.org/10.1038/s41586-021-03854-z>) - 2021-09-29 · `affiliated` · Skilful precipitation nowcasting using deep generative models of radar
- [gdm-doi-10.1007-s00521-021-06259-1](<https://doi.org/10.1007/s00521-021-06259-1>) - 2021-09-28 · `affiliated` · Policy invariant explicit shaping: an efficient alternative to reward shaping
- [gdm-arxiv-1809.00948](<https://doi.org/10.1088/1361-6420/ac28ec>) - 2021-09-21 · `affiliated` · Task adapted reconstruction for inverse problems
- [gdm-doi-10.31234-osf.io-yd9k4](<https://doi.org/10.31234/osf.io/yd9k4>) - 2021-09-14 · `affiliated` · Great ape communication as contextual social inference: a computational modeling perspective
- [gdm-doi-10.1109-tii.2021.3112097](<https://doi.org/10.1109/tii.2021.3112097>) - 2021-09-13 · `affiliated` · Fixed-Point Theorem-Based Voltage Stability Margin Estimation Techniques for Distribution Systems With Renewables
- [gdm-doi-10.1101-2021.08.30.458274](<https://doi.org/10.1101/2021.08.30.458274>) - 2021-08-31 · `affiliated` · Template strand deoxyuridine promoter recognition by a viral RNA polymerase
- [gdm-doi-10.1101-2021.08.29.458114](<https://doi.org/10.1101/2021.08.29.458114>) - 2021-08-31 · `affiliated` · Neural evidence for the successor representation in choice evaluation
- [gdm-arxiv-2104.05372](<https://doi.org/10.1145/3473593>) - 2021-08-19 · `affiliated` · Getting to the point: index sets and parallelism-preserving autodiff for pointful array programming
- [gdm-doi-10.1613-jair.1.12594](<https://doi.org/10.1613/jair.1.12594>) - 2021-08-18 · `affiliated` · Evaluating Strategic Structures in Multi-Agent Inverse Reinforcement Learning
- [gdm-doi-10.1111-rssb.12455](<https://doi.org/10.1111/rssb.12455>) - 2021-08-04 · `affiliated` · Proposer of the Vote of Thanks to Dong <i>et al.</i> and Contribution to the Discussion of ‘Gaussian Differential Privacy’
- [gdm-doi-10.7554-elife.66551](<https://doi.org/10.7554/elife.66551>) - 2021-08-02 · `affiliated` · Interpreting wide-band neural activity using convolutional neural networks
- [gdm-doi-10.24963-ijcai.2021-706](<https://doi.org/10.24963/ijcai.2021/706>) - 2021-08-01 · `affiliated` · A Neural Network Auction For Group Decision Making Over a Continuous Space
- [gdm-arxiv-2105.07933](<https://doi.org/10.24963/ijcai.2021/50>) - 2021-08-01 · `affiliated` · Mean Field Games Flock! The Reinforcement Learning Way
- [gdm-arxiv-2107.12685](<http://arxiv.org/abs/2107.12685>) - 2021-07-27 · `affiliated` · On the Role of Optimization in Double Descent: A Least Squares Study
- [gdm-arxiv-2010.01845](<https://arxiv.org/abs/2010.01845>) - 2021-07-27 · `affiliated` · Unbiased Gradient Estimation for Variational Auto-Encoders using Coupled Markov Chains
- [gdm-arxiv-2006.05145](<https://www.auai.org/uai2021/accepted_papers#26>) - 2021-07-27 · `affiliated` · Matrix games with bandit feedback
- [gdm-doi-10.1038-s41586-021-03828-1](<https://doi.org/10.1038/s41586-021-03828-1>) - 2021-07-22 · `affiliated` · Highly accurate protein structure prediction for the human proteome
- [gdm-arxiv-2102.04257](<https://doi.org/10.1145/3461702.3462540>) - 2021-07-21 · `affiliated` · Fairness for Unobserved Characteristics: Insights from Technological Impacts on Queer Communities
- [gdm-web-fa0ea7d49439](<https://inria.hal.science/hal-03289292>) - 2021-07-18 · `affiliated` · Revisiting Peng's Q(λ) for for modern reinforcement learning
- [gdm-web-f12e3984483e](<http://proceedings.mlr.press/v139/xiao21b/xiao21b.pdf>) - 2021-07-18 · `affiliated` · On the Optimality of Batch Policy Optimization Algorithms
- [gdm-web-da750f792bf1](<https://openalex.org/W3168171623>) - 2021-07-18 · `affiliated` · Catformer: Designing Stable Transformers via Sensitivity Analysis
- [gdm-web-8473116c0ef0](<http://proceedings.mlr.press/v139/hao21a/hao21a.pdf>) - 2021-07-18 · `affiliated` · Sparse Feature Selection Makes Batch Reinforcement Learning More Sample Efficient
- [gdm-web-7c9f2c745a04](<https://openalex.org/W3171658869>) - 2021-07-18 · `affiliated` · Average-Reward Off-Policy Policy Evaluation with Function Approximation
- [gdm-web-64c3750f5add](<http://proceedings.mlr.press/v139/kostrikov21a/kostrikov21a.pdf>) - 2021-07-18 · `affiliated` · Offline Reinforcement Learning with Fisher Divergence Critic Regularization
- [gdm-arxiv-2104.06303](<http://proceedings.mlr.press/v139/hubert21a/hubert21a.pdf>) - 2021-07-18 · `affiliated` · Learning and Planning in Complex Action Spaces
- [gdm-arxiv-2006.16318](<https://openalex.org/W3167096480>) - 2021-07-18 · `affiliated` · Learning and Planning in Average-Reward Markov Decision Processes
- [gdm-doi-10.1038-s41586-021-03819-2](<https://doi.org/10.1038/s41586-021-03819-2>) - 2021-07-15 · `affiliated` · Highly accurate protein structure prediction with AlphaFold
- [gdm-arxiv-2107.06857](<http://arxiv.org/abs/2107.06857>) - 2021-07-14 · `affiliated` · Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot
- [gdm-doi-10.2196-26151](<https://doi.org/10.2196/26151>) - 2021-07-12 · `affiliated` · Clinically Applicable Segmentation of Head and Neck Anatomy for Radiotherapy: Deep Learning Algorithm Development and Validation Study
- [gdm-doi-10.1101-2021.07.07.451322](<https://doi.org/10.1101/2021.07.07.451322>) - 2021-07-09 · `affiliated` · Pre-trial predictors of conflict response efficacy in human dorsolateral prefrontal cortex
- [gdm-arxiv-2107.03851](<http://arxiv.org/abs/2107.03851>) - 2021-07-08 · `affiliated` · Imitation by Predicting Observations
- [gdm-arxiv-2107.00848](<http://arxiv.org/abs/2107.00848>) - 2021-07-02 · `affiliated` · Systematic Evaluation of Causal Discovery in Visual Model Based  Reinforcement Learning
- [gdm-arxiv-2105.02761](<https://doi.org/10.1016/j.patter.2021.100273>) - 2021-07-01 · `affiliated` · Neural algorithmic reasoning
- [gdm-doi-10.1016-j.cell.2021.06.012](<https://doi.org/10.1016/j.cell.2021.06.012>) - 2021-06-30 · `affiliated` · Impaired neural replay of inferred relationships in schizophrenia
- [gdm-arxiv-2106.14693](<http://arxiv.org/abs/2106.14693>) - 2021-06-28 · `affiliated` · Robust Learning-Augmented Caching: An Experimental Study
- [gdm-doi-10.15607-rss.2021.xvii.002](<https://doi.org/10.15607/rss.2021.xvii.002>) - 2021-06-27 · `affiliated` · Manipulator-Independent Representations for Visual Imitation
- [gdm-arxiv-2103.11512](<https://doi.org/10.15607/rss.2021.xvii.088>) - 2021-06-27 · `affiliated` · Robust Multi-Modal Policies for Industrial Assembly via Reinforcement Learning and Demonstrations: A Large-Scale Study
- [gdm-arxiv-2106.13125](<http://arxiv.org/abs/2106.13125>) - 2021-06-24 · `affiliated` · Unifying Gradient Estimators for Meta-Reinforcement Learning via  Off-Policy Evaluation
- [gdm-doi-10.1038-s41593-021-00866-w](<https://doi.org/10.1038/s41593-021-00866-w>) - 2021-06-21 · `affiliated` · Formalizing planning and information search in naturalistic decision-making
- [gdm-arxiv-2106.11779](<http://arxiv.org/abs/2106.11779>) - 2021-06-21 · `affiliated` · Emphatic Algorithms for Deep Reinforcement Learning
- [gdm-doi-10.1101-2021.06.18.448989](<https://doi.org/10.1101/2021.06.18.448989>) - 2021-06-18 · `affiliated` · The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning
- [gdm-doi-10.1016-j.neuron.2021.05.021](<https://doi.org/10.1016/j.neuron.2021.05.021>) - 2021-06-17 · `affiliated` · Promises and challenges of human computational ethology
- [gdm-arxiv-2106.09435](<http://arxiv.org/abs/2106.09435>) - 2021-06-17 · `affiliated` · Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium Meta-Solvers
- [gdm-arxiv-2106.07841](<http://arxiv.org/abs/2106.07841>) - 2021-06-15 · `affiliated` · Randomized Exploration for Reinforcement Learning with General Value Function Approximation
- [gdm-arxiv-2106.06854](<http://arxiv.org/abs/2106.06854>) - 2021-06-12 · `affiliated` · A Deep Reinforcement Learning Approach to Marginalized Importance  Sampling with the Successor Representation
- [gdm-arxiv-2106.06508](<http://arxiv.org/abs/2106.06508>) - 2021-06-11 · `affiliated` · Preferential Temporal Difference Learning
- [gdm-arxiv-2106.06279](<http://arxiv.org/abs/2106.06279>) - 2021-06-11 · `affiliated` · Model-Free Learning for Two-Player Zero-Sum Partially Observable Markov Games with Perfect Recall
- [gdm-arxiv-2106.06189](<http://arxiv.org/abs/2106.06189>) - 2021-06-11 · `affiliated` · Order Matters: Probabilistic Modeling of Node Sequence for Graph Generation
- [gdm-arxiv-2106.06170](<http://arxiv.org/abs/2106.06170>) - 2021-06-11 · `affiliated` · Taylor Expansion of Discount Factors
- [gdm-arxiv-2106.04615](<http://arxiv.org/abs/2106.04615>) - 2021-06-08 · `affiliated` · Vector Quantized Models for Planning
- [gdm-doi-10.7554-elife.66917](<https://doi.org/10.7554/elife.66917>) - 2021-06-07 · `affiliated` · Temporally delayed linear modelling (TDLM) measures replay in both animals and humans
- [gdm-arxiv-2106.02487](<https://doi.org/10.48550/arxiv.2106.02487>) - 2021-06-04 · `affiliated` · Debiasing a First-order Heuristic for Approximate Bi-level Optimization
- [gdm-arxiv-2106.01901](<http://arxiv.org/abs/2106.01901>) - 2021-06-03 · `affiliated` · Iterative Empirical Game Solving via Single Policy Best Response
- [gdm-doi-10.1136-bmjopen-2020-047709](<https://doi.org/10.1136/bmjopen-2020-047709>) - 2021-06-01 · `affiliated` · Developing a reporting guideline for artificial intelligence-centred diagnostic test accuracy studies: the STARD-AI protocol
- [gdm-doi-10.1109-cvpr46437.2021.00446](<https://doi.org/10.1109/cvpr46437.2021.00446>) - 2021-06-01 · `affiliated` · Temporal Query Networks for Fine-grained Video Understanding
- [gdm-doi-10.1109-cvpr46437.2021.00421](<https://doi.org/10.1109/cvpr46437.2021.00421>) - 2021-06-01 · `affiliated` · HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps
- [gdm-doi-10.1038-s42254-021-00330-5](<https://doi.org/10.1038/s42254-021-00330-5>) - 2021-06-01 · `affiliated` · Learning many-electron wavefunctions with deep neural networks
- [gdm-arxiv-2106.08318](<https://doi.org/10.1109/cvpr46437.2021.00913>) - 2021-06-01 · `affiliated` · Gradient Forward-Propagation for Large-Scale Temporal Video Modelling
- [gdm-arxiv-2011.01758](<https://doi.org/10.1109/icra48506.2021.9560733>) - 2021-05-30 · `affiliated` · Representation Matters: Improving Perception and Exploration for Robotics
- [gdm-arxiv-2105.13922](<http://arxiv.org/abs/2105.13922>) - 2021-05-28 · `affiliated` · Discretization Drift in Two-Player Games
- [gdm-arxiv-2105.09992](<https://doi.org/10.24963/ijcai.2021/406>) - 2021-05-20 · `affiliated` · Don't Do What Doesn't Matter: Intrinsic Motivation with Action Usefulness
- [gdm-arxiv-1908.04734](<https://doi.org/10.1007/s11229-021-03141-4>) - 2021-05-19 · `affiliated` · Reward tampering problems and solutions in reinforcement learning: a causal influence diagram perspective
- [gdm-doi-10.1609-aaai.v35i14.17524](<https://doi.org/10.1609/aaai.v35i14.17524>) - 2021-05-18 · `affiliated` · Analogy Training Multilingual Encoders
- [gdm-doi-10.1609-aaai.v35i12.17300](<https://doi.org/10.1609/aaai.v35i12.17300>) - 2021-05-18 · `affiliated` · Sample Efficient Reinforcement Learning with REINFORCE
- [gdm-doi-10.1609-aaai.v35i12.17235](<https://doi.org/10.1609/aaai.v35i12.17235>) - 2021-05-18 · `affiliated` · Self-Supervised Attention-Aware Reinforcement Learning
- [gdm-doi-10.1609-aaai.v35i11.17166](<https://doi.org/10.1609/aaai.v35i11.17166>) - 2021-05-18 · `affiliated` · Solving Common-Payoff Games with Approximate Policy Iteration
- [gdm-arxiv-2103.11505](<https://doi.org/10.1609/aaai.v35i14.17469>) - 2021-05-18 · `affiliated` · Policy-Guided Heuristic Search with Guarantees
- [gdm-arxiv-2102.07716](<https://ojs.aaai.org/index.php/AAAI/article/view/17378>) - 2021-05-18 · `affiliated` · How RL Agents Behave When Their Actions Are Modified
- [gdm-arxiv-2102.01685](<https://ojs.aaai.org/index.php/AAAI/article/view/17368>) - 2021-05-18 · `affiliated` · Agent Incentives: A Causal Perspective
- [gdm-arxiv-2012.10200](<https://doi.org/10.1609/aaai.v35i10.17074>) - 2021-05-18 · `affiliated` · Exact Reduction of Huge Action Spaces in General Reinforcement Learning
- [gdm-arxiv-2012.07827](<https://doi.org/10.1609/aaai.v35i8.16832>) - 2021-05-18 · `affiliated` · Relative Variational Intrinsic Control
- [gdm-arxiv-2012.05874](<https://doi.org/10.1609/aaai.v35i6.16702>) - 2021-05-18 · `affiliated` · Hindsight and Sequential Rationality of Correlated Play
- [gdm-arxiv-2007.01839](<https://ojs.aaai.org/index.php/AAAI/article/view/17200>) - 2021-05-18 · `affiliated` · Expected Eligibility Traces
- [gdm-arxiv-2006.02243](<https://ojs.aaai.org/index.php/AAAI/article/view/16880>) - 2021-05-18 · `affiliated` · The Value-Improvement Path: Towards Better Representations for Reinforcement Learning
- [gdm-arxiv-1910.01526](<https://ojs.aaai.org/index.php/AAAI/article/view/17202>) - 2021-05-18 · `affiliated` · Gated Linear Networks
- [gdm-doi-10.1109-jsait.2021.3079722](<https://doi.org/10.1109/jsait.2021.3079722>) - 2021-05-14 · `affiliated` · Curiosity Killed or Incapacitated the Cat and the Asymptotically Optimal Agent
- [gdm-doi-10.1093-jamia-ocab101](<https://doi.org/10.1093/jamia/ocab101>) - 2021-05-14 · `affiliated` · Multitask prediction of organ dysfunction in the intensive care unit using sequential subnetwork routing
- [gdm-arxiv-2105.06072](<http://arxiv.org/abs/2105.06072>) - 2021-05-13 · `affiliated` · Leveraging Non-uniformity in First-order Non-convex Optimization
- [gdm-arxiv-2009.14280](<https://doi.org/10.1103/physrevfluids.6.050505>) - 2021-05-12 · `affiliated` · Learning to swim in potential flow
- [gdm-arxiv-2105.05246](<http://arxiv.org/abs/2105.05246>) - 2021-05-11 · `affiliated` · Spectral Normalisation for Deep Reinforcement Learning: an Optimisation Perspective
- [gdm-arxiv-2011.09192](<https://doi.org/10.1613/jair.1.12505>) - 2021-05-06 · `affiliated` · Game Plan: What AI can do for Football, and What Football can do for AI
- [gdm-doi-10.1038-s41596-021-00513-5](<https://doi.org/10.1038/s41596-021-00513-5>) - 2021-05-05 · `affiliated` · Use of deep learning to develop continuous-risk models for adverse event prediction from electronic health records
- [gdm-web-e7f6ca7edb15](<https://openreview.net/forum?id=rEaz5uTcL6Q>) - 2021-05-04 · `affiliated` · Neural spatio-temporal reasoning with object-centric self-supervised learning
- [gdm-web-e5366f251c9d](<https://openreview.net/pdf?id=0vO-u0sucRF>) - 2021-05-04 · `affiliated` · Information Theoretic Meta Learning with Gaussian Processes
- [gdm-web-e2562303b4ce](<https://openreview.net/pdf?id=guEuB3FPcd>) - 2021-05-04 · `affiliated` · AlgebraNets
- [gdm-web-78405dbe0dbe](<https://openreview.net/pdf?id=Q9U_H8lQ4yV>) - 2021-05-04 · `affiliated` · Good for Misconceived Reasons: Revisiting Neural Multimodal Machine Translation
- [gdm-web-75618bb8f374](<https://openalex.org/W3126592918>) - 2021-05-04 · `affiliated` · Model-Free Counterfactual Credit Assignment
- [gdm-web-65606e6b4901](<https://openreview.net/pdf?id=Nj8EIrSu5O>) - 2021-05-04 · `affiliated` · Divide-and-Conquer Monte Carlo Tree Search
- [gdm-web-64d9daefa40a](<https://openalex.org/W3132997327>) - 2021-05-04 · `affiliated` · Online Limited Memory Neural-Linear Bandits
- [gdm-web-60ab88c9e8b2](<https://openalex.org/W3128917574>) - 2021-05-04 · `affiliated` · Bayesian Metric Learning for Robust Training of Deep Models under Noisy Labels
- [gdm-web-5679f214f6cd](<https://openalex.org/W3093980962>) - 2021-05-04 · `affiliated` · Incremental Policy Gradients for Online Reinforcement Learning Control
- [gdm-web-3bbb1d87c889](<https://openreview.net/pdf?id=OCRKCul3eKN>) - 2021-05-04 · `affiliated` · Addressing Extrapolation Error in Deep Offline Reinforcement Learning
- [gdm-web-3732bcc64ce3](<https://openreview.net/pdf?id=pOHW7EwFbo9>) - 2021-05-04 · `affiliated` · Explicit Pareto Front Optimization for Constrained Reinforcement Learning
- [gdm-web-1f94d7918a4f](<https://openalex.org/W3127293269>) - 2021-05-04 · `affiliated` · Non-Markovian Predictive Coding For Planning In Latent Space
- [gdm-doi-10.1038-d41586-021-01170-0](<https://doi.org/10.1038/d41586-021-01170-0>) - 2021-05-04 · `affiliated` · Cooperative AI: machines must learn to find common ground
- [gdm-arxiv-2102.13515](<https://arxiv.org/abs/2102.13515>) - 2021-05-04 · `affiliated` · Coverage as a Principle for Discovering Transferable Behavior in Reinforcement Learning
- [gdm-arxiv-2007.15588](<https://arxiv.org/pdf/2007.15588>) - 2021-05-04 · `affiliated` · Data-efficient Hindsight Off-policy Option Learning
- [gdm-web-ee6a12d7e044](<https://openalex.org/W3122771415>) - 2021-05-03 · `affiliated` · Factorizing Declarative and Procedural Knowledge in Structured, Dynamical Environments
- [gdm-web-c92c3e5cbe3c](<https://openreview.net/pdf?id=q3KSThy2GwB>) - 2021-05-03 · `affiliated` · Practical Real Time Recurrent Learning with a Sparse Approximation
- [gdm-web-6fd2a9594221](<https://openreview.net/pdf?id=bgQek2O63w>) - 2021-05-03 · `affiliated` · Self-supervised Adversarial Robustness for the Low-label, High-data Regime
- [gdm-web-2e99d9a8e8f9](<https://openreview.net/pdf?id=NzTU59SYbNq>) - 2021-05-03 · `affiliated` · EigenGame: PCA as a Nash Equilibrium
- [gdm-web-0a1b8315c1cd](<https://openreview.net/pdf?id=Fmg_fQYUejf>) - 2021-05-03 · `affiliated` · Linear Mode Connectivity in Multitask and Continual Learning
- [gdm-doi-10.1016-j.artint.2021.103521](<https://doi.org/10.1016/j.artint.2021.103521>) - 2021-05-03 · `affiliated` · Making sense of raw input
- [gdm-arxiv-2110.04041](<https://doi.org/10.65109/vhlv2397>) - 2021-05-03 · `affiliated` · Pick Your Battles: Interaction Graphs as Population-Level Objectives for Strategic Diversity
- [gdm-arxiv-2102.06911](<https://doi.org/10.65109/sjez3723>) - 2021-05-03 · `affiliated` · Modelling Cooperation in Network Games with Spatio-Temporal Complexity
- [gdm-arxiv-2102.05008](<https://doi.org/10.65109/doif1591>) - 2021-05-03 · `affiliated` · Equilibrium Refinements for Multi-Agent Influence Diagrams: Theory and Practice
- [gdm-arxiv-2101.10276](<https://doi.org/10.65109/ptgw7037>) - 2021-05-03 · `affiliated` · Emergent Communication under Competition
- [gdm-arxiv-2011.04021](<https://arxiv.org/pdf/2011.04021>) - 2021-05-03 · `affiliated` · On the role of planning in model-based deep reinforcement learning
- [gdm-arxiv-2009.01719](<https://arxiv.org/pdf/2009.01719>) - 2021-05-03 · `affiliated` · Grounded Language Learning Fast and Slow
- [gdm-arxiv-2104.13877](<http://arxiv.org/abs/2104.13877>) - 2021-04-28 · `affiliated` · Autoregressive Dynamics Models for Offline Policy Evaluation and Optimization
- [gdm-doi-10.1007-s10994-021-05961-4](<https://doi.org/10.1007/s10994-021-05961-4>) - 2021-04-22 · `affiliated` · Challenges of real-world reinforcement learning: definitions, benchmarks and analysis
- [gdm-arxiv-2104.11186](<http://arxiv.org/abs/2104.11186>) - 2021-04-22 · `affiliated` · Stochastic Shortest Path: Minimax, Parameter-Free and Towards Horizon-Free Regret
- [gdm-arxiv-2008.05456](<https://doi.org/10.1103/physrevd.103.074504>) - 2021-04-20 · `affiliated` · Sampling usingSU&#40;N&#41;gauge equivariant flows
- [gdm-arxiv-2104.06159](<http://arxiv.org/abs/2104.06159>) - 2021-04-13 · `affiliated` · Muesli: Combining Improvements in Policy Optimization
- [gdm-doi-10.1038-s41593-021-00831-7](<https://doi.org/10.1038/s41593-021-00831-7>) - 2021-04-12 · `affiliated` · Flexible modulation of sequence generation in the entorhinal–hippocampal system
- [gdm-arxiv-2104.04975](<http://arxiv.org/abs/2104.04975>) - 2021-04-11 · `affiliated` · Scalable Marginal Likelihood Estimation for Model Selection in Deep  Learning
- [gdm-doi-10.1111-gcb.15630](<https://doi.org/10.1111/gcb.15630>) - 2021-04-08 · `affiliated` · Classifying ecosystem stressor interactions: Theory highlights the data limitations of the additive null model and the difficulty in revealing ecological surprises
- [gdm-arxiv-2104.00587](<http://arxiv.org/abs/2104.00587>) - 2021-04-01 · `affiliated` · NeRF-VAE: A Geometry Aware 3D Scene Generative Model
- [gdm-arxiv-2103.16596](<http://arxiv.org/abs/2103.16596>) - 2021-03-30 · `affiliated` · Benchmarks for Deep Off-Policy Evaluation
- [gdm-arxiv-2012.02308](<https://doi.org/10.1145/3450439.3451858>) - 2021-03-23 · `affiliated` · Concept-based model explanations for electronic health records
- [gdm-doi-10.31234-osf.io-me3xy](<https://doi.org/10.31234/osf.io/me3xy>) - 2021-03-18 · `affiliated` · Direct Human-AI Comparison in the Animal-AI Environment
- [gdm-doi-10.1038-s42256-021-00318-x](<https://doi.org/10.1038/s42256-021-00318-x>) - 2021-03-18 · `affiliated` · Generalizing universal function approximators
- [gdm-doi-10.1101-2021.03.10.434756](<https://doi.org/10.1101/2021.03.10.434756>) - 2021-03-12 · `affiliated` · A rapid and efficient learning rule for biological neural circuits
- [gdm-doi-10.1016-j.adhoc.2021.102475](<https://doi.org/10.1016/j.adhoc.2021.102475>) - 2021-03-12 · `affiliated` · Human tracking and identification through a millimeter wave radar
- [gdm-arxiv-2102.09774](<https://doi.org/10.1109/jsac.2021.3065057>) - 2021-03-11 · `affiliated` · A Reinforcement Learning Approach to Age of Information in Multi-User Networks With HARQ
- [gdm-arxiv-2103.03841](<http://arxiv.org/abs/2103.03841>) - 2021-03-05 · `affiliated` · Generating Images with Sparse Representations
- [gdm-arxiv-2103.01950](<http://arxiv.org/abs/2103.01950>) - 2021-03-02 · `affiliated` · Predicting Video with VQVAE
- [gdm-arxiv-2103.01312](<http://arxiv.org/abs/2103.01312>) - 2021-03-01 · `affiliated` · UCB Momentum Q-learning: Correcting the bias without forgetting
- [gdm-arxiv-2103.00107](<http://arxiv.org/abs/2103.00107>) - 2021-02-27 · `affiliated` · Revisiting Peng's Q&#40;&#36;&#92;&#92;lambda&#36;&#41; for Modern Reinforcement Learning
- [gdm-doi-10.1016-j.sbi.2021.01.008](<https://doi.org/10.1016/j.sbi.2021.01.008>) - 2021-02-26 · `affiliated` · Advances in machine learning for directed evolution
- [gdm-arxiv-2102.12611](<http://arxiv.org/abs/2102.12611>) - 2021-02-25 · `affiliated` · Improved Regret Bound and Experience Replay in Regularized Policy Iteration
- [gdm-doi-10.1145-3446776](<https://doi.org/10.1145/3446776>) - 2021-02-22 · `affiliated` · Understanding deep learning (still) requires rethinking generalization
- [gdm-arxiv-2103.03905](<http://arxiv.org/abs/2103.03905>) - 2021-02-20 · `affiliated` · Kanerva++: extending The Kanerva Machine with differentiable, locally block allocated latent memory
- [gdm-doi-10.1038-s41588-021-00782-6](<https://doi.org/10.1038/s41588-021-00782-6>) - 2021-02-18 · `affiliated` · Base-resolution models of transcription-factor binding reveal soft motif syntax
- [gdm-arxiv-2102.07501](<http://arxiv.org/abs/2102.07501>) - 2021-02-15 · `affiliated` · Annealed Flow Transport Monte Carlo
- [gdm-arxiv-2102.06973](<http://arxiv.org/abs/2102.06973>) - 2021-02-13 · `affiliated` · Efficient Deviation Types and Learning for Hindsight Rationality in Extensive-Form Games
- [gdm-arxiv-2102.06171](<http://arxiv.org/abs/2102.06171>) - 2021-02-11 · `affiliated` · High-Performance Large-Scale Image Recognition Without Normalization
- [gdm-arxiv-2102.06129](<http://arxiv.org/abs/2102.06129>) - 2021-02-11 · `affiliated` · Meta-Thompson Sampling
- [gdm-doi-10.1038-s41598-021-82889-8](<https://doi.org/10.1038/s41598-021-82889-8>) - 2021-02-09 · `affiliated` · Overcoming Pavlovian bias in semantic space
- [gdm-arxiv-2102.04323](<http://arxiv.org/abs/2102.04323>) - 2021-02-08 · `affiliated` · Discovering a set of policies for the worst case reward
- [gdm-arxiv-2102.03799](<http://arxiv.org/abs/2102.03799>) - 2021-02-07 · `affiliated` · Online Limited Memory Neural-Linear Bandits with Likelihood Matching
- [gdm-arxiv-2102.03607](<http://arxiv.org/abs/2102.03607>) - 2021-02-06 · `affiliated` · Bootstrapping Fitted Q-Evaluation for Off-Policy Inference
- [gdm-doi-10.1016-j.neuron.2021.01.021](<https://doi.org/10.1016/j.neuron.2021.01.021>) - 2021-02-01 · `affiliated` · What can classic Atari video games tell us about the human brain?
- [gdm-arxiv-2101.12176](<http://arxiv.org/abs/2101.12176>) - 2021-01-28 · `affiliated` · On the Origin of Implicit Regularization in Stochastic Gradient Descent
- [gdm-arxiv-1807.06763](<https://doi.org/10.1613/jair.1.12105>) - 2021-01-28 · `affiliated` · General Value Function Networks
- [gdm-arxiv-2101.11046](<http://arxiv.org/abs/2101.11046>) - 2021-01-26 · `affiliated` · Generalized Doubly Reparameterized Gradient Estimators
- [gdm-arxiv-2011.13464](<https://doi.org/10.1016/j.cobeha.2021.01.002>) - 2021-01-25 · `affiliated` · Meta-learning in natural and artificial intelligence
- [gdm-arxiv-2101.08692](<http://arxiv.org/abs/2101.08692>) - 2021-01-21 · `affiliated` · Characterizing signal propagation to close the performance gap in unnormalized ResNets
- [gdm-arxiv-2011.11818](<https://doi.org/10.1109/slt48900.2021.9383525>) - 2021-01-19 · `affiliated` · Synth2Aug: Cross-Domain Speaker Recognition with TTS Synthesized Speech
- [gdm-arxiv-2101.07123](<http://arxiv.org/abs/2101.07123>) - 2021-01-18 · `affiliated` · Learning Successor States and Goal-Dependent Values: A Mathematical  Viewpoint
- [gdm-doi-10.1101-2021.01.15.426915](<https://doi.org/10.1101/2021.01.15.426915>) - 2021-01-16 · `affiliated` · Learning from unexpected events in the neocortical microcircuit
- [gdm-arxiv-2101.01631](<http://arxiv.org/abs/2101.01631>) - 2021-01-05 · `affiliated` · On the Approximation Relationship between Optimizing Ratio of Submodular  (RS) and Difference of Submodular (DS) Functions
- [gdm-arxiv-1910.02227](<https://doi.org/10.1016/j.artint.2020.103438>) - 2021-01-05 · `affiliated` · Making sense of sensory input
- [gdm-web-dcbf793682c2](<https://hal.science/hal-05578071>) - 2021-01-01 · `affiliated` · Self-supervised representation learning using bootstrapped latent representations
- [gdm-doi-10.4230-dagrep.11.4.20](<https://drops.dagstuhl.de/entities/document/10.4230/DagRep.11.4.20>) - 2021-01-01 · `affiliated` · Approaches and Applications of Inductive Programming (Dagstuhl Seminar 21192)
- [gdm-doi-10.25080-majora-1b6fd038-008](<https://doi.org/10.25080/majora-1b6fd038-008>) - 2021-01-01 · `affiliated` · PyCID: A Python Library for Causal Influence Diagrams
- [gdm-doi-10.2139-ssrn.3841306](<https://doi.org/10.2139/ssrn.3841306>) - 2021-01-01 · `affiliated` · Optimal Bidding, Allocation, and Budget Spending for a Demand-Side Platform with Generic Auctions
- [gdm-doi-10.18653-v1-2021.textgraphs-1.7](<https://doi.org/10.18653/v1/2021.textgraphs-1.7>) - 2021-01-01 · `affiliated` · WikiGraphs: A Wikipedia Text - Knowledge Graph Paired Dataset
- [gdm-doi-10.18653-v1-2021.naacl-main.18](<https://doi.org/10.18653/v1/2021.naacl-main.18>) - 2021-01-01 · `affiliated` · Counterfactual Data Augmentation for Neural Machine Translation
- [gdm-doi-10.18653-v1-2021.findings-emnlp.63](<https://doi.org/10.18653/v1/2021.findings-emnlp.63>) - 2021-01-01 · `affiliated` · Efficient Test Time Adapter Ensembling for Low-resource Language Varieties
- [gdm-doi-10.18653-v1-2021.findings-emnlp.410](<https://doi.org/10.18653/v1/2021.findings-emnlp.410>) - 2021-01-01 · `affiliated` · MAD-G: Multilingual Adapter Generation for Efficient Cross-Lingual Transfer
- [gdm-doi-10.18653-v1-2021.emnlp-main.699](<https://doi.org/10.18653/v1/2021.emnlp-main.699>) - 2021-01-01 · `affiliated` · IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation
- [gdm-doi-10.18653-v1-2021.emnlp-main.536](<https://doi.org/10.18653/v1/2021.emnlp-main.536>) - 2021-01-01 · `affiliated` · A Generative Framework for Simultaneous Machine Translation
- [gdm-doi-10.1162-tacl_a_00390](<https://doi.org/10.1162/tacl_a_00390>) - 2021-01-01 · `affiliated` · Pretraining the Noisy Channel Model for Task-Oriented Dialogue
- [gdm-doi-10.1162-tacl_a_00371](<https://doi.org/10.1162/tacl_a_00371>) - 2021-01-01 · `affiliated` · Adaptive Semiparametric Language Models
- [gdm-doi-10.1016-j.stem.2020.12.012](<https://doi.org/10.1016/j.stem.2020.12.012>) - 2021-01-01 · `affiliated` · Computational Stem Cell Biology: Open Questions and Guiding Principles
- [gdm-doi-10.1007-978-3-030-93733-1_10](<https://doi.org/10.1007/978-3-030-93733-1_10>) - 2021-01-01 · `affiliated` · IReEn: Reverse-Engineering of Black-Box Functions via Iterative Neural Program Synthesis
- [gdm-doi-10.1007-978-3-030-78601-4_12](<https://doi.org/10.1007/978-3-030-78601-4_12>) - 2021-01-01 · `affiliated` · Google and DeepMind: Deep Learning Systems in Ophthalmology
- [gdm-doi-10.1007-978-3-030-03009-4_67-1](<https://doi.org/10.1007/978-3-030-03009-4_67-1>) - 2021-01-01 · `affiliated` · Learned Iterative Reconstruction
- [gdm-arxiv-2103.11811](<https://doi.org/10.1162/tacl_a_00416>) - 2021-01-01 · `affiliated` · MasakhaNER: Named Entity Recognition for African Languages
- [gdm-arxiv-2103.08490](<https://doi.org/10.18653/v1/2021.naacl-main.40>) - 2021-01-01 · `affiliated` · Multi-view Subword Regularization
- [gdm-arxiv-2102.09544](<https://hdl.handle.net/11585/905136>) - 2021-01-01 · `affiliated` · Combinatorial Optimization and Reasoning with Graph Neural Networks
- [gdm-arxiv-2102.06860](<https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2021.118>) - 2021-01-01 · `affiliated` · Optimal Spectral-Norm Approximate Minimization of Weighted Finite Automata
- [gdm-arxiv-2102.00529](<https://doi.org/10.1162/tacl_a_00385>) - 2021-01-01 · `affiliated` · Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers
- [gdm-arxiv-2012.15562](<https://doi.org/10.18653/v1/2021.emnlp-main.800>) - 2021-01-01 · `affiliated` · UNKs Everywhere: Adapting Multilingual Language Models to New Scripts
- [gdm-arxiv-2004.07625](<https://doi.org/10.1007/978-3-030-72376-7_7>) - 2021-01-01 · `affiliated` · Should I Tear down This Wall? Optimizing Social Metrics by Evaluating Novel Actions
- [gdm-arxiv-2003.02344](<https://doi.org/10.1007/s11222-020-09984-0>) - 2021-01-01 · `affiliated` · Fast sampling from &#36;&#36;&#92;&#92;beta &#36;&#36;-ensembles
- [gdm-arxiv-1309.6129](<https://doi.org/10.1109/access.2021.3070490>) - 2021-01-01 · `affiliated` · Partition-Merge: Distributed Inference and Modularity Optimization

### 17.7 2020（217 条）

- [gdm-arxiv-2012.14755](<http://arxiv.org/abs/2012.14755>) - 2020-12-29 · `affiliated` · Improved Sample Complexity for Incremental Autonomous Exploration in MDPs
- [gdm-doi-10.1016-j.neuron.2020.12.007](<https://doi.org/10.1016/j.neuron.2020.12.007>) - 2020-12-23 · `affiliated` · Replay bursts in humans coincide with activation of the default mode and parietal alpha networks
- [gdm-arxiv-1911.08265](<https://doi.org/10.1038/s41586-020-03051-4>) - 2020-12-23 · `affiliated` · Mastering Atari, Go, chess and shogi by planning with a learned model
- [gdm-arxiv-2012.10885](<http://arxiv.org/abs/2012.10885>) - 2020-12-20 · `affiliated` · LieTransformer: Equivariant self-attention for Lie Groups
- [gdm-doi-10.1613-jair.1.12087](<https://doi.org/10.1613/jair.1.12087>) - 2020-12-14 · `affiliated` · Adapting Behavior via Intrinsic Reward: A Survey and Empirical Study
- [gdm-doi-10.1107-s2052252520014384](<https://doi.org/10.1107/s2052252520014384>) - 2020-12-07 · `affiliated` · Exploiting prior knowledge about biological macromolecules in cryo-EM structure determination
- [gdm-doi-10.1016-j.neuron.2020.11.011](<https://doi.org/10.1016/j.neuron.2020.11.011>) - 2020-12-07 · `affiliated` · Regional, Layer, and Cell-Type-Specific Connectivity of the Mouse Default Mode Network
- [gdm-arxiv-1904.08128](<https://doi.org/10.1038/s41592-020-01008-z>) - 2020-12-07 · `affiliated` · nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
- [gdm-web-2dfa368a77b1](<https://inria.hal.science/hal-03137351>) - 2020-12-06 · `affiliated` · Leverage the Average: an Analysis of KL Regularization in Reinforcement Learning
- [gdm-arxiv-1908.04480](<https://doi.org/10.1103/physreva.102.062405>) - 2020-12-04 · `affiliated` · Quantum adiabatic machine learning by zooming into a region of the energy surface
- [gdm-doi-10.1111-1740-9713.01463](<https://doi.org/10.1111/1740-9713.01463>) - 2020-12-01 · `affiliated` · Decision-Making with Uncertainty
- [gdm-arxiv-2005.13482](<https://doi.org/10.1162/tacl_a_00345>) - 2020-12-01 · `affiliated` · Syntactic Structure Distillation Pretraining for Bidirectional Encoders
- [gdm-doi-10.2196-preprints.26151](<https://doi.org/10.2196/preprints.26151>) - 2020-11-30 · `affiliated` · Clinically Applicable Segmentation of Head and Neck Anatomy for Radiotherapy: Deep Learning Algorithm Development and Validation Study (Preprint)
- [gdm-arxiv-2011.12916](<http://arxiv.org/abs/2011.12916>) - 2020-11-25 · `affiliated` · Equivariant Learning of Stochastic Fields: Gaussian Processes and Steerable Conditional Neural Processes
- [gdm-doi-10.1073-pnas.2007981117](<https://doi.org/10.1073/pnas.2007981117>) - 2020-11-23 · `affiliated` · A general model of hippocampal and dorsal striatal learning and decision making
- [gdm-doi-10.1101-2020.11.17.387043](<https://doi.org/10.1101/2020.11.17.387043>) - 2020-11-20 · `affiliated` · Rotational dynamics in motor cortex are consistent with a feedback controller
- [gdm-arxiv-2001.10912](<https://doi.org/10.1103/physrevlett.125.200604>) - 2020-11-13 · `affiliated` · Observing Localization in a 2D Quasicrystalline Optical Lattice
- [gdm-doi-10.1101-2020.11.11.378141](<https://doi.org/10.1101/2020.11.11.378141>) - 2020-11-12 · `affiliated` · A model of egocentric to allocentric understanding in mammalian brains
- [gdm-arxiv-2011.04020](<http://arxiv.org/abs/2011.04020>) - 2020-11-08 · `affiliated` · High-Dimensional Sparse Linear Bandits
- [gdm-arxiv-2005.01642](<https://doi.org/10.1038/s41467-020-19244-4>) - 2020-11-05 · `affiliated` · Navigating the landscape of multiplayer games
- [gdm-doi-10.1016-j.tics.2020.09.004](<https://doi.org/10.1016/j.tics.2020.09.004>) - 2020-11-03 · `affiliated` · Embracing Change: Continual Learning in Deep Neural Networks
- [gdm-arxiv-2006.16392](<https://doi.org/10.1109/tnse.2020.3035352>) - 2020-11-03 · `affiliated` · Approximating Network Centrality Measures Using Node Embedding and Machine Learning
- [gdm-doi-10.1109-ieeeconf51394.2020.9443419](<https://doi.org/10.1109/ieeeconf51394.2020.9443419>) - 2020-11-01 · `affiliated` · WaveNetEQ — Packet Loss Concealment with WaveRNN
- [gdm-arxiv-2006.10901](<https://doi.org/10.1109/sc41405.2020.00021>) - 2020-11-01 · `affiliated` · Sparse GPU Kernels for Deep Learning
- [gdm-arxiv-2011.00344](<http://arxiv.org/abs/2011.00344>) - 2020-10-31 · `affiliated` · A Distribution Dependent Analysis of Meta-Learning
- [gdm-arxiv-2010.15274](<http://arxiv.org/abs/2010.15274>) - 2020-10-28 · `affiliated` · Representation learning for improved interpretability and classification accuracy of clinical factors from EEG
- [gdm-doi-10.1101-2020.10.24.353409](<https://doi.org/10.1101/2020.10.24.353409>) - 2020-10-25 · `affiliated` · A meta-learning approach to (re)discover plasticity rules that carve a desired function into a neural network
- [gdm-arxiv-2010.13146](<http://arxiv.org/abs/2010.13146>) - 2020-10-25 · `affiliated` · XLVIN: eXecuted Latent Value Iteration Nets
- [gdm-arxiv-2008.03127](<https://doi.org/10.21437/interspeech.2020-2892>) - 2020-10-25 · `affiliated` · A Machine of Few Words: Interactive Speaker Recognition with Reinforcement Learning
- [gdm-arxiv-2007.13971](<https://doi.org/10.1109/iros45743.2020.9341683>) - 2020-10-24 · `affiliated` · Accurate, Low-Latency Visual Perception for Autonomous Racing: Challenges, Mechanisms, and Practical Solutions
- [gdm-arxiv-2003.14398](<https://doi.org/10.1109/iros45743.2020.9341191>) - 2020-10-24 · `affiliated` · Robotic Table Tennis with Model-Free Reinforcement Learning
- [gdm-arxiv-2010.10864](<http://arxiv.org/abs/2010.10864>) - 2020-10-21 · `affiliated` · A Short Note on the Kinetics-700-2020 Human Action Dataset
- [gdm-arxiv-2010.10644](<http://arxiv.org/abs/2010.10644>) - 2020-10-20 · `affiliated` · Robust Constrained Reinforcement Learning for Continuous Control with Model Misspecification
- [gdm-arxiv-2010.10241](<http://arxiv.org/abs/2010.10241>) - 2020-10-20 · `affiliated` · BYOL works even without batch statistics
- [gdm-doi-10.1145-3408294](<https://doi.org/10.1145/3408294>) - 2020-10-17 · `affiliated` · Human-computer Coalition Formation in Weighted Voting Games
- [gdm-arxiv-2010.07922](<http://arxiv.org/abs/2010.07922>) - 2020-10-15 · `affiliated` · Representation Learning via Invariant Causal Mechanisms
- [gdm-doi-10.1038-s41586-020-2679-9](<https://doi.org/10.1038/s41586-020-2679-9>) - 2020-10-14 · `affiliated` · Addendum: International evaluation of an AI system for breast cancer screening
- [gdm-arxiv-2010.07154](<http://arxiv.org/abs/2010.07154>) - 2020-10-14 · `affiliated` · Learning Deep Features in Instrumental Variable Regression
- [gdm-arxiv-2010.06324](<http://arxiv.org/abs/2010.06324>) - 2020-10-13 · `affiliated` · Balancing Constraints and Rewards with Meta-Gradient D4PG
- [gdm-arxiv-2002.04913](<https://doi.org/10.1063/5.0018903>) - 2020-10-13 · `affiliated` · Targeted free energy estimation via learned mappings
- [gdm-doi-10.1016-j.tics.2020.09.002](<https://doi.org/10.1016/j.tics.2020.09.002>) - 2020-10-08 · `affiliated` · Artificial Intelligence and the Common Sense of Animals
- [gdm-arxiv-2007.13681](<https://doi.org/10.1088/2632-2153/abbf9a>) - 2020-10-08 · `affiliated` · Graph neural networks in particle physics
- [gdm-doi-10.1002-jeab.631](<https://doi.org/10.1002/jeab.631>) - 2020-10-07 · `affiliated` · Social discounting of pain
- [gdm-arxiv-2010.03531](<http://arxiv.org/abs/2010.03531>) - 2020-10-07 · `affiliated` · Episodic Reinforcement Learning in Finite MDPs: Minimax Lower Bounds  Revisited
- [gdm-arxiv-2010.02255](<http://arxiv.org/abs/2010.02255>) - 2020-10-05 · `affiliated` · Temporal Difference Uncertainties as a Signal for Exploration
- [gdm-arxiv-2010.01787](<http://arxiv.org/abs/2010.01787>) - 2020-10-05 · `affiliated` · Improving Relational Regularized Autoencoders with Spherical Sliced  Fused Gromov Wasserstein
- [gdm-arxiv-1905.10862](<https://doaj.org/article/e56b837c0845448ca65e693881073aa0>) - 2020-10-01 · `affiliated` · Automatic Discovery of Privacy–Utility Pareto Fronts
- [gdm-doi-10.1371-journal.pone.0234595](<https://doi.org/10.1371/journal.pone.0234595>) - 2020-09-28 · `affiliated` · Modelling parameter uncertainty reveals bushmeat yields versus survival trade-offs in heavily-hunted duiker Cephalophus spp.
- [gdm-doi-10.1073-pnas.1910416117](<https://doi.org/10.1073/pnas.1910416117>) - 2020-09-28 · `affiliated` · Placing language in an integrated understanding system: Next steps toward human-level performance in neural language models
- [gdm-arxiv-2009.12583](<http://arxiv.org/abs/2009.12583>) - 2020-09-26 · `affiliated` · Small Data, Big Decisions: Model Selection in the Small-Data Regime
- [gdm-doi-10.1016-j.cobeha.2020.07.003](<https://doi.org/10.1016/j.cobeha.2020.07.003>) - 2020-09-21 · `affiliated` · Multi-step planning in the brain
- [gdm-arxiv-1902.10674](<https://doi.org/10.1109/tifs.2020.3025441>) - 2020-09-21 · `affiliated` · The Best Defense Is a Good Offense: Adversarial Attacks to Avoid Modulation Detection
- [gdm-arxiv-2009.09153](<http://arxiv.org/abs/2009.09153>) - 2020-09-19 · `affiliated` · Hidden Incentives for Auto-Induced Distributional Shift
- [gdm-arxiv-2009.07476](<http://arxiv.org/abs/2009.07476>) - 2020-09-16 · `affiliated` · Path Planning using Neural A* Search
- [gdm-arxiv-1909.02487](<https://doi.org/10.1103/physrevresearch.2.033429>) - 2020-09-16 · `affiliated` · <i>Ab initio</i> solution of the many-electron Schrödinger equation with deep neural networks
- [gdm-arxiv-2003.06413](<https://doi.org/10.1103/physrevlett.125.121601>) - 2020-09-15 · `affiliated` · Equivariant Flow-Based Sampling for Lattice Gauge Theory
- [gdm-doi-10.1038-s42256-020-00228-4](<https://doi.org/10.1038/s42256-020-00228-4>) - 2020-09-14 · `affiliated` · Domesticating the techno-racial project
- [gdm-doi-10.1145-3372020.3391557](<https://doi.org/10.1145/3372020.3391557>) - 2020-09-12 · `affiliated` · Minimal Assumptions Refinement for Realizable Specifications
- [gdm-arxiv-2001.09768](<https://doi.org/10.1007/s11023-020-09539-2>) - 2020-09-01 · `affiliated` · Artificial Intelligence, Values, and Alignment
- [gdm-arxiv-2008.13773](<http://arxiv.org/abs/2008.13773>) - 2020-08-31 · `affiliated` · Beyond variance reduction: Understanding the true impact of baselines on policy optimization
- [gdm-doi-10.3390-e22090943](<https://doi.org/10.3390/e22090943>) - 2020-08-27 · `affiliated` · Variational Information Bottleneck for Semi-Supervised Classification
- [gdm-arxiv-2008.12234](<http://arxiv.org/abs/2008.12234>) - 2020-08-27 · `affiliated` · The Advantage Regret-Matching Actor-Critic
- [gdm-web-98abe1a9ba01](<https://inria.hal.science/hal-03288939>) - 2020-08-26 · `affiliated` · Derivative-free &amp; order-robust optimisation
- [gdm-doi-10.1145-3394486.3403181](<https://doi.org/10.1145/3394486.3403181>) - 2020-08-20 · `affiliated` · The NodeHopper: Enabling Low Latency Ranking with Constraints via a Fast Dual Solver
- [gdm-doi-10.1038-s41598-020-70960-9](<https://doi.org/10.1038/s41598-020-70960-9>) - 2020-08-20 · `affiliated` · Non-linear changes in modelled terrestrial ecosystems subjected to perturbations
- [gdm-arxiv-2001.02589](<https://doi.org/10.1038/s41467-020-17835-9>) - 2020-08-19 · `affiliated` · Machine learning enables completely automatic tuning of a quantum device faster than human experts
- [gdm-doi-10.1073-pnas.1907370117](<https://doi.org/10.1073/pnas.1907370117>) - 2020-08-17 · `affiliated` · Fast reinforcement learning with generalized policy updates
- [gdm-doi-10.1109-tpami.2020.3016711](<https://doi.org/10.1109/tpami.2020.3016711>) - 2020-08-14 · `affiliated` · NCNet: Neighbourhood Consensus Networks for Estimating Image Correspondences
- [gdm-doi-10.1145-3386569.3392474](<https://doi.org/10.1145/3386569.3392474>) - 2020-08-12 · `affiliated` · Catch &amp; Carry
- [gdm-doi-10.1101-2020.08.10.243972](<https://doi.org/10.1101/2020.08.10.243972>) - 2020-08-10 · `affiliated` · Ecological theory predicts ecosystem stressor interactions in freshwater ecosystems, but highlights the strengths and weaknesses of the additive null model
- [gdm-arxiv-2006.12983](<https://doi.org/10.1016/j.simpa.2020.100022>) - 2020-08-03 · `affiliated` · dm_control: Software and tasks for continuous control
- [gdm-doi-10.1145-3408971](<https://doi.org/10.1145/3408971>) - 2020-08-02 · `affiliated` · A quick look at impredicativity
- [gdm-web-30021a604bc1](<http://hdl.handle.net/1843/54840>) - 2020-07-30 · `affiliated` · Para que serve o subteste de Aritmética do Teste de Desempenho Escolar
- [gdm-arxiv-2007.13442](<http://hdl.handle.net/10230/69310>) - 2020-07-27 · `affiliated` · Fast active learning for pure exploration in reinforcement learning
- [gdm-arxiv-2007.13363](<http://arxiv.org/abs/2007.13363>) - 2020-07-27 · `affiliated` · Learning Compositional Neural Programs for Continuous Control
- [gdm-arxiv-2007.12911](<https://research.manchester.ac.uk/en/publications/1c38b4e0-4a5c-4458-8b7c-3ac648bab141>) - 2020-07-25 · `affiliated` · Tighter risk certificates for neural networks
- [gdm-arxiv-2010.10380](<https://doi.org/10.1016/j.artint.2020.103356>) - 2020-07-24 · `affiliated` · Negotiating team formation using deep reinforcement learning
- [gdm-arxiv-2007.12509](<http://arxiv.org/abs/2007.12509>) - 2020-07-24 · `affiliated` · Monte-Carlo Tree Search as Regularized Policy Optimization
- [gdm-arxiv-1907.04907](<https://doi.org/10.1162/tacl_a_00325>) - 2020-07-23 · `affiliated` · Topic Modeling in Embedding Spaces
- [gdm-arxiv-2008.02646](<https://doi.org/10.1109/lra.2020.3010461>) - 2020-07-20 · `affiliated` · Deep Reinforcement Learning for Tactile Robotics: Learning to Type on a Braille Keyboard
- [gdm-arxiv-2007.06521](<https://doi.org/10.1073/pnas.2001258117>) - 2020-07-16 · `affiliated` · Predicting the long-term stability of compact multiplanet systems
- [gdm-arxiv-2007.08620](<http://arxiv.org/abs/2007.08620>) - 2020-07-15 · `affiliated` · The Monte Carlo Transformer: a stochastic self-attention model for  sequence prediction
- [gdm-arxiv-2007.06700](<http://arxiv.org/abs/2007.06700>) - 2020-07-13 · `affiliated` · Revisiting Fundamentals of Experience Replay
- [gdm-arxiv-2007.06437](<http://arxiv.org/abs/2007.06437>) - 2020-07-13 · `affiliated` · A Provably Efficient Sample Collection Strategy for Reinforcement Learning
- [gdm-arxiv-2007.06202](<http://arxiv.org/abs/2007.06202>) - 2020-07-13 · `affiliated` · Structured Policy Iteration for Linear Quadratic Regulator
- [gdm-arxiv-2007.03750](<https://doi.org/10.1016/j.neuron.2020.06.014>) - 2020-07-13 · `affiliated` · Deep Reinforcement Learning and Its Neuroscientific Implications
- [gdm-web-ff7e6a948a49](<https://inria.hal.science/hal-02950106>) - 2020-07-12 · `affiliated` · Improved sleeping bandits with stochastic action sets and adversarial rewards
- [gdm-web-e373853f5bf8](<http://proceedings.mlr.press/v119/badia20a/badia20a.pdf>) - 2020-07-12 · `affiliated` · Agent57: Outperforming the Atari Human Benchmark
- [gdm-web-dd6a33c3beb6](<http://proceedings.mlr.press/v119/henaff20a/henaff20a.pdf>) - 2020-07-12 · `affiliated` · Data-Efficient Image Recognition with Contrastive Predictive Coding
- [gdm-web-d1506de443ea](<https://openalex.org/W3034932139>) - 2020-07-12 · `affiliated` · Invariant Causal Prediction for Block MDPs
- [gdm-web-cef3e1e29e0c](<http://proceedings.mlr.press/v119/nagarajan20a/nagarajan20a.pdf>) - 2020-07-12 · `affiliated` · From Chaos to Order: Symmetry and Conservation Laws in Game Dynamics
- [gdm-web-bdd12ff1a121](<http://proceedings.mlr.press/v119/joulani20a/joulani20a.pdf>) - 2020-07-12 · `affiliated` · A simpler approach to accelerated optimization: iterative averaging meets optimism
- [gdm-web-88e8961445a0](<https://icml.cc/Conferences/2020/AcceptedPapersInitial#822>) - 2020-07-12 · `affiliated` · CoMic: Co-Training and Mimicry for Reusable Skills
- [gdm-web-87dd79a5210d](<http://proceedings.mlr.press/v119/guo20g/guo20g.pdf>) - 2020-07-12 · `affiliated` · Bootstrap Latent-Predictive Representations for Multitask Reinforcement Learning
- [gdm-web-69b7569e8df7](<http://proceedings.mlr.press/v119/lattimore20a/lattimore20a.pdf>) - 2020-07-12 · `affiliated` · Learning with Good Feature Representations in Bandits and in RL with a Generative Model
- [gdm-web-5ac8b6dd8fbd](<http://proceedings.mlr.press/v119/munos20a/munos20a.pdf>) - 2020-07-12 · `affiliated` · Fast computation of Nash Equilibria in Imperfect Information Games
- [gdm-web-2b50a854e275](<http://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf>) - 2020-07-12 · `affiliated` · Learning to Simulate Complex Physics with Graph Networks
- [gdm-web-2434faf0bb79](<https://icml.cc/Conferences/2020/AcceptedPapersInitial#333>) - 2020-07-12 · `affiliated` · Influence Diagram Bandits
- [gdm-web-1630c325e478](<http://proceedings.mlr.press/v119/hu20b/hu20b.pdf>) - 2020-07-12 · `affiliated` · XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation
- [gdm-web-128bf14ab7fb](<https://openalex.org/W3035234304>) - 2020-07-12 · `affiliated` · Analyzing the effect of neural network architecture on training performance
- [gdm-web-0c3e11394e8b](<http://proceedings.mlr.press/v119/vezhnevets20a/vezhnevets20a.pdf>) - 2020-07-12 · `affiliated` · OPtions as REsponses: Grounding behavioural hierarchies in multi-agent reinforcement learning
- [gdm-web-0bafd84309c3](<https://dblp.uni-trier.de/db/journals/corr/corr2006.html#abs-2006-02119>) - 2020-07-12 · `affiliated` · Non-Stationary Bandits with Intermediate Observations
- [gdm-url-1f56d4901da0](<https://openalex.org/W3035303608>) - 2020-07-12 · `affiliated` · Training Neural Networks for and by Interpolation
- [gdm-arxiv-2007.04068](<https://doi.org/10.1007/s13347-020-00405-8>) - 2020-07-12 · `affiliated` · Decolonial AI: Decolonial Theory as Sociotechnical Foresight in Artificial Intelligence
- [gdm-arxiv-2003.06350](<https://openalex.org/W3034848825>) - 2020-07-12 · `affiliated` · Interference and Generalization in Temporal Difference Learning
- [gdm-arxiv-1912.02738](<https://openalex.org/W3034710639>) - 2020-07-12 · `affiliated` · MetaFun: Meta-Learning with Iterative Functional Updates
- [gdm-arxiv-1910.13324](<https://arxiv.org/pdf/1910.13324.pdf>) - 2020-07-12 · `affiliated` · Divide, Conquer, and Combine: a New Inference Strategy for Probabilistic Programs with Stochastic Support
- [gdm-arxiv-1807.02089](<http://proceedings.mlr.press/v119/vernade20a/vernade20a.pdf>) - 2020-07-12 · `affiliated` · Linear bandits with Stochastic Delayed Feedback
- [gdm-web-79923b766281](<https://hal.archives-ouvertes.fr/hal-02896961>) - 2020-07-11 · `affiliated` · The Monte Carlo Transformer
- [gdm-arxiv-2007.05078](<http://arxiv.org/abs/2007.05078>) - 2020-07-09 · `affiliated` · A Kernel-Based Approach to Non-Stationary Reinforcement Learning in Metric Spaces
- [gdm-arxiv-1910.00553](<https://doi.org/10.1162/tacl_a_00319>) - 2020-07-08 · `affiliated` · Better Document-Level Machine Translation with Bayes’ Rule
- [gdm-arxiv-2007.00953](<http://arxiv.org/abs/2007.00953>) - 2020-07-02 · `affiliated` · Gamification of Pure Exploration for Linear Bandits
- [gdm-arxiv-2004.13654](<https://doi.org/10.24963/ijcai.2020/221>) - 2020-07-01 · `affiliated` · Pitfalls of Learning a Reward Function Online
- [gdm-arxiv-1806.04067](<https://doi.org/10.1109/ijcnn48605.2020.9207690>) - 2020-07-01 · `affiliated` · Adaptive Mechanism Design: Learning to Promote Cooperation
- [gdm-arxiv-2006.16981](<http://arxiv.org/abs/2006.16981>) - 2020-06-30 · `affiliated` · Learning to Combine Top-Down and Bottom-Up Signals in Recurrent Neural Networks with Attention over Modules
- [gdm-arxiv-2006.16947](<http://arxiv.org/abs/2006.16947>) - 2020-06-30 · `affiliated` · Sampling from a &#36;k&#36;-DPP without looking at all items
- [gdm-arxiv-2006.15502](<http://arxiv.org/abs/2006.15502>) - 2020-06-28 · `affiliated` · Scalable Deep Generative Modeling for Sparse Graphs
- [gdm-arxiv-2006.15085](<http://arxiv.org/abs/2006.15085>) - 2020-06-26 · `affiliated` · What can I do here? A Theory of Affordances in Reinforcement Learning
- [gdm-arxiv-2006.15081](<http://arxiv.org/abs/2006.15081>) - 2020-06-26 · `affiliated` · On the Generalization Benefit of Noise in Stochastic Gradient Descent
- [gdm-doi-10.1016-j.neuroimage.2020.117087](<https://doi.org/10.1016/j.neuroimage.2020.117087>) - 2020-06-25 · `affiliated` · The influence of microsatellite polymorphisms in sex steroid receptor genes ESR1, ESR2 and AR on sex differences in brain structure
- [gdm-doi-10.1101-2020.06.23.166645](<https://doi.org/10.1101/2020.06.23.166645>) - 2020-06-24 · `affiliated` · Replay bursts coincide with activation of the default mode and parietal alpha network
- [gdm-arxiv-2006.13900](<http://arxiv.org/abs/2006.13900>) - 2020-06-24 · `affiliated` · Quantifying Differences in Reward Functions
- [gdm-arxiv-2006.10459](<http://arxiv.org/abs/2006.10459>) - 2020-06-18 · `affiliated` · Stochastic bandits with arm-dependent delays
- [gdm-doi-10.1126-sciadv.aba3828](<https://doi.org/10.1126/sciadv.aba3828>) - 2020-06-17 · `affiliated` · The value of what’s to come: Neural mechanisms coupling prediction error and the utility of anticipation
- [gdm-doi-10.31234-osf.io-5bhwe](<https://doi.org/10.31234/osf.io/5bhwe>) - 2020-06-16 · `affiliated` · Misdirected vigor: Differentiating the control of value from the value of control
- [gdm-doi-10.1038-s41467-020-16856-8](<https://doi.org/10.1038/s41467-020-16856-8>) - 2020-06-15 · `affiliated` · Social training reconfigures prediction errors to shape Self-Other boundaries
- [gdm-arxiv-2006.09265](<http://arxiv.org/abs/2006.09265>) - 2020-06-13 · `affiliated` · IsarStep: a Benchmark for High-level Mathematical Reasoning
- [gdm-arxiv-2006.07733](<http://arxiv.org/abs/2006.07733>) - 2020-06-13 · `affiliated` · Bootstrap your own latent: A new approach to self-supervised Learning
- [gdm-arxiv-2006.06613](<http://arxiv.org/abs/2006.06613>) - 2020-06-11 · `affiliated` · Statistical Efficiency of Thompson Sampling for Combinatorial Semi-Bandits
- [gdm-arxiv-2006.06294](<http://arxiv.org/abs/2006.06294>) - 2020-06-11 · `affiliated` · Adaptive Reward-Free Exploration
- [gdm-web-65e05e232a63](<https://iovs.arvojournals.org/article.aspx?articleid=2766896>) - 2020-06-10 · `affiliated` · Quantitative analysis of change in retinal tissues in neovascular age-related macular degeneration using artificial intelligence
- [gdm-arxiv-2006.04710](<http://arxiv.org/abs/2006.04710>) - 2020-06-08 · `affiliated` · The Lipschitz Constant of Self-Attention
- [gdm-doi-10.1007-s10817-020-09559-8](<https://doi.org/10.1007/s10817-020-09559-8>) - 2020-06-06 · `affiliated` · Proof-Producing Synthesis of CakeML from Monadic HOL Functions
- [gdm-arxiv-2006.03662](<http://arxiv.org/abs/2006.03662>) - 2020-06-05 · `affiliated` · Rapid Task-Solving in Novel Environments
- [gdm-arxiv-2001.06944](<http://proceedings.mlr.press/v108/zhang20b/zhang20b.pdf>) - 2020-06-03 · `affiliated` · Nested-Wasserstein Self-Imitation Learning for Sequence Generation
- [gdm-doi-10.48550-arxiv.2006.01782](<https://doi.org/10.48550/arxiv.2006.01782>) - 2020-06-02 · `affiliated` · Temporally-Extended ε-Greedy Exploration
- [gdm-doi-10.1109-cvpr42600.2020.00990](<https://doi.org/10.1109/cvpr42600.2020.00990>) - 2020-06-01 · `affiliated` · End-to-End Learning of Visual Representations From Uncurated Instructional Videos
- [gdm-doi-10.1101-2020.06.01.127522](<https://doi.org/10.1101/2020.06.01.127522>) - 2020-06-01 · `affiliated` · Modelling parameter uncertainty reveals bushmeat yields versus survival trade-offs in heavily-hunted duiker <i>Cephalophus</i> spp.
- [gdm-doi-10.1038-s41591-020-0941-1](<https://doi.org/10.1038/s41591-020-0941-1>) - 2020-06-01 · `affiliated` · Developing specific reporting guidelines for diagnostic accuracy studies assessing AI interventions: The STARD-AI Steering Group
- [gdm-arxiv-2006.15418](<https://doi.org/10.1109/cvpr42600.2020.01040>) - 2020-06-01 · `affiliated` · Counting Out Time: Class Agnostic Video Repetition Counting in the Wild
- [gdm-arxiv-2006.01016](<http://arxiv.org/abs/2006.01016>) - 2020-06-01 · `affiliated` · Probing Emergent Semantics in Predictive Agents via Question Answering
- [gdm-arxiv-2003.13594](<https://doi.org/10.1109/cvpr42600.2020.01033>) - 2020-06-01 · `affiliated` · Speech2Action: Cross-Modal Supervision for Action Recognition
- [gdm-arxiv-2001.06232](<https://doi.org/10.1109/cvpr42600.2020.01185>) - 2020-06-01 · `affiliated` · Sideways: Depth-Parallel Training of Video Models
- [gdm-arxiv-1912.03192](<https://doi.org/10.1109/cvpr42600.2020.00129>) - 2020-06-01 · `affiliated` · Achieving Robustness in the Wild via Adversarial Mixing With Disentangled Representations
- [gdm-arxiv-1912.02184](<https://doi.org/10.1109/cvpr42600.2020.00950>) - 2020-06-01 · `affiliated` · Towards Robust Image Classification Using Sequential Attention Models
- [gdm-arxiv-1911.09723](<https://doi.org/10.1109/cvpr42600.2020.01464>) - 2020-06-01 · `affiliated` · Fast Sparse ConvNets
- [gdm-arxiv-1907.02055](<https://doi.org/10.1109/cvpr42600.2020.00881>) - 2020-06-01 · `affiliated` · Self-Supervised Learning of Interpretable Keypoints From Unlabelled Videos
- [gdm-doi-10.1038-s41467-020-15533-0](<https://doi.org/10.1038/s41467-020-15533-0>) - 2020-05-19 · `affiliated` · Representation of visual uncertainty through neural gain variability
- [gdm-doi-10.1038-s41591-020-0867-7](<https://doi.org/10.1038/s41591-020-0867-7>) - 2020-05-18 · `affiliated` · Predicting conversion to wet age-related macular degeneration using deep learning
- [gdm-doi-10.1038-s41467-020-15871-z](<https://doi.org/10.1038/s41467-020-15871-z>) - 2020-05-18 · `affiliated` · AI for social good: unlocking the opportunity for positive impact
- [gdm-doi-10.1016-j.neuropsychologia.2020.107479](<https://doi.org/10.1016/j.neuropsychologia.2020.107479>) - 2020-05-16 · `affiliated` · Localizing syntactic predictions using recurrent neural network grammars
- [gdm-arxiv-2005.07572](<http://arxiv.org/abs/2005.07572>) - 2020-05-15 · `affiliated` · Participatory Problem Formulation for Fairer Machine Learning Through Community Based System Dynamics
- [gdm-arxiv-2005.07513](<http://arxiv.org/abs/2005.07513>) - 2020-05-15 · `affiliated` · A Distributional View on Multi-Objective Policy Optimization
- [gdm-arxiv-2005.07186](<http://arxiv.org/abs/2005.07186>) - 2020-05-14 · `affiliated` · Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors
- [gdm-arxiv-2005.06392](<http://arxiv.org/abs/2005.06392>) - 2020-05-13 · `affiliated` · On the Global Convergence Rates of Softmax Policy Gradient Methods
- [gdm-doi-10.65109-gjmw6851](<https://doi.org/10.65109/gjmw6851>) - 2020-05-05 · `affiliated` · Neural Replicator Dynamics: Multiagent Learning via Hedging Policy Gradients
- [gdm-doi-10.65109-diew7802](<https://doi.org/10.65109/diew7802>) - 2020-05-05 · `affiliated` · Robust Self-organization in Games: Symmetries, Conservation Laws and Dimensionality Reduction
- [gdm-arxiv-2003.00799](<https://doi.org/10.65109/wuwb6150>) - 2020-05-05 · `affiliated` · Learning to Resolve Alliance Dilemmas in Many-Player Zero-Sum Games
- [gdm-arxiv-2002.02325](<https://doi.org/10.65109/dnuf7773>) - 2020-05-05 · `affiliated` · Social Diversity and Social Preferences in Mixed-Motive Reinforcement Learning
- [gdm-doi-10.1101-2020.04.30.066407](<https://doi.org/10.1101/2020.04.30.066407>) - 2020-05-02 · `affiliated` · Measuring Sequences of Representations with Temporally Delayed Linear Modelling
- [gdm-arxiv-1910.09470](<https://doi.org/10.1109/icra40945.2020.9197326>) - 2020-05-01 · `affiliated` · Self-Supervised Sim-to-Real Adaptation for Visual Robotic Manipulation
- [gdm-arxiv-1909.00025](<https://research.manchester.ac.uk/en/publications/36b0a885-9e2c-4650-a486-2f24e1f74fb2>) - 2020-04-30 · `affiliated` · Meta-Learning with Warped Gradient Descent
- [gdm-doi-10.1038-s41583-020-0277-3](<https://doi.org/10.1038/s41583-020-0277-3>) - 2020-04-17 · `affiliated` · Backpropagation and the brain
- [gdm-arxiv-2004.05599](<http://arxiv.org/abs/2004.05599>) - 2020-04-12 · `affiliated` · Kernel-Based Reinforcement Learning: A Finite-Time Analysis
- [gdm-doi-10.1609-aaai.v34i04.5857](<https://ojs.aaai.org/index.php/AAAI/article/view/5857>) - 2020-04-03 · `affiliated` · Algorithmic Improvements for Deep Reinforcement Learning Applied to Interactive Fiction
- [gdm-doi-10.1609-aaai.v34i04.5771](<https://doi.org/10.1609/aaai.v34i04.5771>) - 2020-04-03 · `affiliated` · A General Approach to Fairness with Optimal Transport
- [gdm-doi-10.1609-aaai.v34i01.5400](<https://doi.org/10.1609/aaai.v34i01.5400>) - 2020-04-03 · `affiliated` · Learning the Graphical Structure of Electronic Health Records with Graph Convolutional Transformer
- [gdm-arxiv-1907.02633](<https://ojs.aaai.org/index.php/AAAI/article/view/6203>) - 2020-04-03 · `affiliated` · On the Convergence of Model Free Learning in Mean Field Games
- [gdm-arxiv-1903.01083](<https://doi.org/10.1609/aaai.v34i04.5899>) - 2020-04-03 · `affiliated` · Stochastic Online Learning with Probabilistic Graph Feedback
- [gdm-arxiv-1903.00401](<https://ojs.aaai.org/index.php/AAAI/article/view/6849>) - 2020-04-03 · `affiliated` · Learning to Follow Directions in Street View
- [gdm-arxiv-1902.03393](<https://ojs.aaai.org/index.php/AAAI/article/view/5963>) - 2020-04-03 · `affiliated` · Improved Knowledge Distillation via Teacher Assistant
- [gdm-doi-10.1038-s41567-020-0842-8](<https://doi.org/10.1038/s41567-020-0842-8>) - 2020-04-02 · `affiliated` · Unveiling the predictive power of static structure in glassy systems
- [gdm-doi-10.2478-popets-2020-0024](<https://doi.org/10.2478/popets-2020-0024>) - 2020-04-01 · `affiliated` · Secure and Scalable Document Similarity on Distributed Databases: Differential Privacy to the Rescue
- [gdm-doi-10.1101-2020.03.14.991745](<https://doi.org/10.1101/2020.03.14.991745>) - 2020-03-15 · `affiliated` · Human dorsal anterior cingulate neurons signal conflict by amplifying task-relevant information
- [gdm-arxiv-2003.06259](<http://arxiv.org/abs/2003.06259>) - 2020-03-13 · `affiliated` · Taylor Expansion Policy Optimization
- [gdm-arxiv-2003.02037](<https://arxiv.org/abs/2003.02037v1>) - 2020-03-04 · `affiliated` · Simple and Scalable Epistemic Uncertainty Estimation Using a Single Deep Deterministic Neural Network
- [gdm-doi-10.1111-oik.07748](<https://doi.org/10.1111/oik.07748>) - 2020-03-02 · `affiliated` · The Madingley general ecosystem model predicts bushmeat yields, species extinction rates and ecosystem‐level impacts of bushmeat harvesting
- [gdm-arxiv-1908.08495](<https://doi.org/10.1088/2632-2153/ab6432>) - 2020-02-25 · `affiliated` · Applying machine learning optimization methods to the production of a quantum gas
- [gdm-arxiv-2002.10880](<http://arxiv.org/abs/2002.10880>) - 2020-02-23 · `affiliated` · PolyGen: An Autoregressive Generative Model of 3D Meshes
- [gdm-arxiv-2002.08797](<http://arxiv.org/abs/2002.08797>) - 2020-02-19 · `affiliated` · Robust Pruning at Initialization
- [gdm-arxiv-2002.07367](<http://arxiv.org/abs/2002.07367>) - 2020-02-18 · `affiliated` · Distributional Sliced-Wasserstein and Applications to Generative Modeling
- [gdm-arxiv-2002.06473](<http://arxiv.org/abs/2002.06473>) - 2020-02-15 · `affiliated` · Universal Value Density Estimation for Imitation Learning and Goal-Conditioned Reinforcement Learning
- [gdm-arxiv-2002.05685](<http://arxiv.org/abs/2002.05685>) - 2020-02-13 · `affiliated` · Fractional Underdamped Langevin Dynamics: Retargeting SGD with Momentum under Heavy-Tailed Gradient Noise
- [gdm-arxiv-2002.03712](<http://arxiv.org/abs/2002.03712>) - 2020-02-10 · `affiliated` · On Contrastive Learning for Likelihood-free Inference
- [gdm-arxiv-2002.02428](<http://arxiv.org/abs/2002.02428>) - 2020-02-06 · `affiliated` · Normalizing Flows on Tori and Spheres
- [gdm-doi-10.2478-jagi-2020-0003](<https://doi.org/10.2478/jagi-2020-0003>) - 2020-02-01 · `affiliated` · Special Issue “On Defining Artificial Intelligence”—Commentaries and Author’s Response
- [gdm-arxiv-2001.09318](<http://arxiv.org/abs/2001.09318>) - 2020-01-25 · `affiliated` · Silly rules improve the capacity of agents to learn stable enforcement  and compliance behaviors
- [gdm-doi-10.1038-s41586-019-1924-6](<https://doi.org/10.1038/s41586-019-1924-6>) - 2020-01-15 · `affiliated` · A distributional code for value in dopamine-based reinforcement learning
- [gdm-doi-10.1038-s41586-019-1923-7](<https://doi.org/10.1038/s41586-019-1923-7>) - 2020-01-15 · `affiliated` · Improved protein structure prediction using potentials from deep learning
- [gdm-arxiv-1810.10342](<https://doi.org/10.1038/s41467-019-13922-8>) - 2020-01-08 · `affiliated` · Predicting optical coherence tomography-derived diabetic macular edema grades from fundus photographs using deep learning
- [gdm-web-8c80e4e0ae98](<http://urn.fi/urn:nbn:fi-fe2020052538879>) - 2020-01-01 · `affiliated` · Multi-scale learned iterative reconstruction
- [gdm-web-8897e8f5260a](<https://inria.hal.science/hal-03288970>) - 2020-01-01 · `affiliated` · Reward-free exploration beyond finite-horizon
- [gdm-doi-10.4230-dagrep.9.12.67](<https://drops.dagstuhl.de/entities/document/10.4230/DagRep.9.12.67>) - 2020-01-01 · `affiliated` · Artificial and Computational Intelligence in Games: Revolutions in Computational Game AI (Dagstuhl Seminar 19511)
- [gdm-doi-10.18653-v1-2020.findings-emnlp.106](<https://doi.org/10.18653/v1/2020.findings-emnlp.106>) - 2020-01-01 · `affiliated` · Learning Robust and Multilingual Speech Representations
- [gdm-doi-10.18653-v1-2020.acl-main.672](<https://doi.org/10.18653/v1/2020.acl-main.672>) - 2020-01-01 · `affiliated` · Do Transformers Need Deep Long-Range Memory?
- [gdm-doi-10.1038-s41586-019-1799-6](<https://doi.org/10.1038/s41586-019-1799-6>) - 2020-01-01 · `affiliated` · International evaluation of an AI system for breast cancer screening
- [gdm-doi-10.1007-978-3-030-58526-6_28](<https://doi.org/10.1007/978-3-030-58526-6_28>) - 2020-01-01 · `affiliated` · Learning Actionness via Long-Range Temporal Order Verification
- [gdm-doi-10.1007-978-3-030-40245-7_10](<https://doi.org/10.1007/978-3-030-40245-7_10>) - 2020-01-01 · `affiliated` · Message Passing Neural Networks
- [gdm-doi-10.1007-978-3-030-06164-7_12](<https://doi.org/10.1007/978-3-030-06164-7_12>) - 2020-01-01 · `affiliated` · Reinforcement Learning
- [gdm-arxiv-2012.15816](<https://doi.org/10.1007/978-3-030-43883-8_7>) - 2020-01-01 · `affiliated` · Fairness in Machine Learning
- [gdm-arxiv-2011.07593](<https://doi.org/10.18653/v1/2020.coling-main.256>) - 2020-01-01 · `affiliated` · Morphologically Aware Word-Level Translation
- [gdm-arxiv-2009.06610](<https://doi.org/10.1007/978-3-030-58517-4_4>) - 2020-01-01 · `affiliated` · Adaptive Text Recognition Through Visual Matching
- [gdm-arxiv-2007.07779](<https://doi.org/10.18653/v1/2020.emnlp-demos.7>) - 2020-01-01 · `affiliated` · AdapterHub: A Framework for Adapting Transformers
- [gdm-arxiv-2006.05879](<http://hdl.handle.net/20.500.12210/29467>) - 2020-01-01 · `affiliated` · Planning in Markov Decision Processes with Gap-Dependent Sample Complexity
- [gdm-arxiv-2005.07064](<https://doi.org/10.18653/v1/2020.acl-main.685>) - 2020-01-01 · `affiliated` · Multi-agent Communication meets Natural Language: Synergies between Functional and Structural Language Learning
- [gdm-arxiv-2005.01279](<https://doi.org/10.18653/v1/2020.acl-main.227>) - 2020-01-01 · `affiliated` · Improving Adversarial Text Generation by Modeling the Distant Future
- [gdm-arxiv-2005.00052](<https://doi.org/10.18653/v1/2020.emnlp-main.617>) - 2020-01-01 · `affiliated` · MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer
- [gdm-arxiv-2004.14958](<https://doi.org/10.18653/v1/2020.acl-main.658>) - 2020-01-01 · `affiliated` · A Call for More Rigor in Unsupervised Cross-lingual Learning
- [gdm-arxiv-2004.10566](<https://doi.org/10.1007/978-3-030-58545-7_35>) - 2020-01-01 · `affiliated` · Efficient Neighbourhood Consensus Networks via Submanifold Sparse Convolutions
- [gdm-arxiv-2004.04070](<https://doi.org/10.18653/v1/2020.emnlp-main.257>) - 2020-01-01 · `affiliated` · Are All Good Word Vector Spaces Isomorphic?
- [gdm-arxiv-2003.05078](<https://ora.ox.ac.uk/objects/uuid:f4cbea7b-4014-49e5-b85a-235b40092c6f>) - 2020-01-01 · `affiliated` · Visual grounding in video for unsupervised word translation
- [gdm-arxiv-2002.09954](<https://inria.hal.science/hal-02950066>) - 2020-01-01 · `affiliated` · Near-linear time Gaussian process optimization with adaptive batching and resparsification
- [gdm-arxiv-1912.03517](<https://inria.hal.science/hal-03287824>) - 2020-01-01 · `affiliated` · No-regret exploration in goal-oriented reinforcement learning
- [gdm-arxiv-1910.03065](<https://doi.org/10.18653/v1/2020.acl-main.382>) - 2020-01-01 · `affiliated` · Make Up Your Mind! Adversarial Generation of Inconsistent Natural Language Explanations
- [gdm-arxiv-1810.11547](<https://doi.org/10.1109/tip.2019.2963389>) - 2020-01-01 · `affiliated` · Unsupervised Multi-Target Domain Adaptation: An Information Theoretic Approach
- [gdm-arxiv-1809.05567](<https://doi.org/10.1137/18m1214123>) - 2020-01-01 · `affiliated` · Multifidelity Dimension Reduction via Active Subspaces
- [gdm-arxiv-1711.05139](<https://doi.org/10.1007/978-3-030-30671-7_3>) - 2020-01-01 · `affiliated` · XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings

### 17.8 2019（158 条）

- [gdm-doi-10.1038-s41598-019-55395-1](<https://doi.org/10.1038/s41598-019-55395-1>) - 2019-12-13 · `affiliated` · Encoding Temporal Regularities and Information Copying in Hippocampal Circuits
- [gdm-arxiv-1912.06430](<http://arxiv.org/abs/1912.06430>) - 2019-12-13 · `affiliated` · End-to-End Learning of Visual Representations from Uncurated  Instructional Videos
- [gdm-web-0d3b0dcd24e5](<https://openalex.org/W2994629834>) - 2019-12-11 · `affiliated` · Deepinsight: a general framework for interpreting wide-band neural activity
- [gdm-arxiv-1912.05500](<http://arxiv.org/abs/1912.05500>) - 2019-12-11 · `affiliated` · What Can Learned Intrinsic Rewards Capture?
- [gdm-arxiv-1912.03074](<http://arxiv.org/abs/1912.03074>) - 2019-12-06 · `affiliated` · Solving Bernoulli Rank-One Bandits with Unimodal Thompson Sampling
- [gdm-arxiv-1912.05652](<http://arxiv.org/abs/1912.05652>) - 2019-12-05 · `affiliated` · Learning Human Objectives by Evaluating Hypothetical Behavior
- [gdm-doi-10.1007-s10458-019-09432-y](<https://doi.org/10.1007/s10458-019-09432-y>) - 2019-12-04 · `affiliated` · Bounds and dynamics for empirical game theoretic analysis
- [gdm-doi-10.1038-s41467-019-13239-6](<https://doi.org/10.1038/s41467-019-13239-6>) - 2019-12-02 · `affiliated` · Hierarchical motor control in mammals and machines
- [gdm-arxiv-1911.04890](<https://doi.org/10.1109/asru46091.2019.9004036>) - 2019-12-01 · `affiliated` · Recurrent Neural Network Transducer for Audio-Visual Speech Recognition
- [gdm-web-2a8e7d5a42c8](<https://hal.science/hal-02386585>) - 2019-11-29 · `affiliated` · Self-Educated Language Agent With Hindsight Experience Replay For Instruction Following
- [gdm-arxiv-1902.00506](<https://doi.org/10.1016/j.artint.2019.103216>) - 2019-11-27 · `affiliated` · The Hanabi challenge: A new frontier for AI research
- [gdm-arxiv-1911.11134](<http://arxiv.org/abs/1911.11134>) - 2019-11-25 · `affiliated` · Rigging the Lottery: Making All Tickets Winners
- [gdm-arxiv-1810.06721](<https://doi.org/10.1038/s41467-019-13073-w>) - 2019-11-19 · `affiliated` · Optimizing agent behavior over long time scales by transporting value
- [gdm-arxiv-1911.06636](<http://arxiv.org/abs/1911.06636>) - 2019-11-15 · `affiliated` · Catch & Carry: Reusable Neural Controllers for Vision-Guided Whole-Body Tasks
- [gdm-doi-10.1016-j.tcs.2019.11.015](<https://doi.org/10.1016/j.tcs.2019.11.015>) - 2019-11-12 · `affiliated` · A modular analysis of adaptive (non-)convex optimization: Optimism, composite objectives, variance reduction, and variational bounds
- [gdm-doi-10.1145-3319535.3353562](<https://doi.org/10.1145/3319535.3353562>) - 2019-11-06 · `affiliated` · PPML '19
- [gdm-doi-10.5281-zenodo.3529714](<https://doi.org/10.5281/zenodo.3529714>) - 2019-11-04 · `affiliated` · ISMIR 2019 tutorial: waveform-based music processing with deep learning
- [gdm-doi-10.1109-globalsip45357.2019.8969541](<https://doi.org/10.1109/globalsip45357.2019.8969541>) - 2019-11-01 · `affiliated` · Communication without Interception: Defense against Modulation Detection
- [gdm-arxiv-1911.06833](<https://doi.org/10.1109/iros40897.2019.8967896>) - 2019-11-01 · `affiliated` · Improved Exploration through Latent Trajectory Optimization in Deep Deterministic Policy Gradient
- [gdm-doi-10.1038-s41586-019-1724-z](<https://doi.org/10.1038/s41586-019-1724-z>) - 2019-10-30 · `affiliated` · Grandmaster level in StarCraft II using multi-agent reinforcement learning
- [gdm-doi-10.1186-s12916-019-1426-2](<https://doi.org/10.1186/s12916-019-1426-2>) - 2019-10-29 · `affiliated` · Key challenges for delivering clinical impact with artificial intelligence
- [gdm-doi-10.1038-s41593-019-0520-2](<https://doi.org/10.1038/s41593-019-0520-2>) - 2019-10-28 · `affiliated` · A deep learning framework for neuroscience
- [gdm-arxiv-1910.10945](<http://arxiv.org/abs/1910.10945>) - 2019-10-24 · `affiliated` · Fixed-Confidence Guarantees for Bayesian Best-Arm Identification
- [gdm-arxiv-1910.09890](<http://arxiv.org/abs/1910.09890>) - 2019-10-22 · `affiliated` · Improving the Gating Mechanism of Recurrent Neural Networks
- [gdm-arxiv-1910.09451](<http://arxiv.org/abs/1910.09451>) - 2019-10-21 · `affiliated` · HIGhER : Improving instruction following with Hindsight Generation for Experience Replay
- [gdm-doi-10.1145-3347450.3357659](<https://doi.org/10.1145/3347450.3357659>) - 2019-10-15 · `affiliated` · Learning to Navigate
- [gdm-arxiv-1910.06764](<http://arxiv.org/abs/1910.06764>) - 2019-10-13 · `affiliated` · Stabilizing Transformers for Reinforcement Learning
- [gdm-doi-10.1002-prot.25834](<https://doi.org/10.1002/prot.25834>) - 2019-10-11 · `affiliated` · Protein structure prediction using multiple deep neural networks in the 13th Critical Assessment of Protein Structure Prediction (CASP13)
- [gdm-arxiv-1910.00760](<http://arxiv.org/abs/1910.00760>) - 2019-10-02 · `affiliated` · Efficient Graph Generation with Graph Recurrent Attention Networks
- [gdm-doi-10.1109-iccv.2019.00819](<https://doi.org/10.1109/iccv.2019.00819>) - 2019-10-01 · `affiliated` · Cross-View Policy Learning for Street Navigation
- [gdm-doi-10.1109-iccv.2019.00494](<https://doi.org/10.1109/iccv.2019.00494>) - 2019-10-01 · `affiliated` · Scalable Verified Training for Provably Robust Image Classification
- [gdm-doi-10.1109-iccv.2019.00135](<https://doi.org/10.1109/iccv.2019.00135>) - 2019-10-01 · `affiliated` · Layout-Induced Video Representation for Recognizing Agent-in-Place Actions
- [gdm-arxiv-1910.11306](<https://doi.org/10.1109/iccv.2019.00583>) - 2019-10-01 · `affiliated` · Controllable Attention for Structured Layered Video Decomposition
- [gdm-arxiv-1907.04927](<https://doi.org/10.1109/waspaa.2019.8937169>) - 2019-10-01 · `affiliated` · Speech Bandwidth Extension with Wavenet
- [gdm-arxiv-1906.03327](<https://doi.org/10.1109/iccv.2019.00272>) - 2019-10-01 · `affiliated` · HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips
- [gdm-doi-10.1038-s41593-019-0494-0](<https://doi.org/10.1038/s41593-019-0494-0>) - 2019-09-30 · `affiliated` · Widespread temporal coding of cognitive control in the human prefrontal cortex
- [gdm-doi-10.1016-s2589-7500-19-30123-2](<https://doi.org/10.1016/s2589-7500(19)30123-2>) - 2019-09-25 · `affiliated` · A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis
- [gdm-arxiv-1909.11646](<http://arxiv.org/abs/1909.11646>) - 2019-09-25 · `affiliated` · High Fidelity Speech Synthesis with Adversarial Networks
- [gdm-arxiv-1909.10893](<http://arxiv.org/abs/1909.10893>) - 2019-09-24 · `affiliated` · Recurrent Independent Mechanisms
- [gdm-doi-10.1101-775957](<https://doi.org/10.1101/775957>) - 2019-09-19 · `affiliated` · Mechanistic macroecology: exploring the drivers of latitudinal variation in terrestrial body size in a General Ecosystem Model
- [gdm-arxiv-1909.09146](<http://arxiv.org/abs/1909.09146>) - 2019-09-19 · `affiliated` · Weighted Linear Bandits for Non-Stationary Environments
- [gdm-arxiv-1910.02532](<https://discovery.ucl.ac.uk/id/eprint/10106621/>) - 2019-09-14 · `affiliated` · Probabilistic Successor Representations with Kalman Temporal Differences
- [gdm-arxiv-1901.08810](<https://doi.org/10.1109/taslp.2019.2938863>) - 2019-09-03 · `affiliated` · Unsupervised Speech Representation Learning Using WaveNet Autoencoders
- [gdm-doi-10.1016-s2589-7500-19-30108-6](<https://doi.org/10.1016/s2589-7500(19)30108-6>) - 2019-09-01 · `affiliated` · Automated deep learning design for medical image classification by health-care professionals with no coding experience: a feasibility study
- [gdm-arxiv-1809.06165](<https://doi.org/10.1007/978-3-030-29513-4_78>) - 2019-08-23 · `affiliated` · Towards Partner-Aware Humanoid Robot Control Under Physical Interactions
- [gdm-arxiv-1808.01639](<https://doi.org/10.1007/978-3-030-29513-4_79>) - 2019-08-23 · `affiliated` · Momentum-Based Topology Estimation of Articulated Objects
- [gdm-doi-10.1162-jocn_a_01454](<https://doi.org/10.1162/jocn_a_01454>) - 2019-08-08 · `affiliated` · Structural and Functional MRI Evidence for Distinct Medial Temporal and Prefrontal Roles in Context-dependent Relational Memory
- [gdm-doi-10.1101-724021](<https://doi.org/10.1101/724021>) - 2019-08-03 · `affiliated` · Representation of uncertainty in macaque visual cortex
- [gdm-arxiv-1811.03805](<https://doi.org/10.1109/pesgm40551.2019.8973654>) - 2019-08-01 · `affiliated` · A Sufficient Condition for Small-Signal Stability and Construction of Robust Stability Region
- [gdm-doi-10.21203-rs.2.10083-v1](<https://doi.org/10.21203/rs.2.10083/v1>) - 2019-07-31 · `affiliated` · Developing Deep Learning Continuous Risk Models for Early Adverse Event Prediction in Electronic Health Records: an AKI Case Study
- [gdm-doi-10.1038-s41746-019-0100-6](<https://doi.org/10.1038/s41746-019-0100-6>) - 2019-07-31 · `affiliated` · Evaluation of a digitally-enabled care pathway for acute kidney injury management in hospital emergency admissions
- [gdm-doi-10.1038-s41586-019-1390-1](<https://doi.org/10.1038/s41586-019-1390-1>) - 2019-07-31 · `affiliated` · A clinically applicable approach to continuous prediction of future acute kidney injury
- [gdm-doi-10.24963-ijcai.2019-854](<https://doi.org/10.24963/ijcai.2019/854>) - 2019-07-28 · `affiliated` · A Dual Approach to Verify and Train Deep Networks
- [gdm-doi-10.24963-ijcai.2019-391](<https://doi.org/10.24963/ijcai.2019/391>) - 2019-07-28 · `affiliated` · Learning Generative Adversarial Networks from Multiple Data Sources
- [gdm-doi-10.24963-ijcai.2019-305](<https://doi.org/10.24963/ijcai.2019/305>) - 2019-07-28 · `affiliated` · Three-Player Wasserstein GAN via Amortised Duality
- [gdm-arxiv-1907.13062](<https://doi.org/10.24963/ijcai.2019/174>) - 2019-07-28 · `affiliated` · Iterative Budgeted Exponential Search
- [gdm-arxiv-1903.05614](<https://doi.org/10.24963/ijcai.2019/66>) - 2019-07-28 · `affiliated` · Computing Approximate Equilibria in Sequential Adversarial Games by Exploitability Descent
- [gdm-arxiv-1902.10089](<https://doi.org/10.24963/ijcai.2019/386>) - 2019-07-28 · `affiliated` · Perturbed-History Exploration in Stochastic Multi-Armed Bandits
- [gdm-arxiv-1806.02136](<https://doi.org/10.1145/3341701>) - 2019-07-26 · `affiliated` · Efficient differentiable programming in a functional array-processing language
- [gdm-doi-10.1145-3292500.3330649](<https://doi.org/10.1145/3292500.3330649>) - 2019-07-25 · `affiliated` · A Generalized Framework for Population Based Training
- [gdm-web-66f6dc2a25c6](<https://iovs.arvojournals.org/article.aspx?articleid=2742174>) - 2019-07-22 · `affiliated` · Diagnostic accuracy and interobserver variability of macular disease evaluation using optical coherence tomography
- [gdm-web-2d4fdda8165c](<https://openalex.org/W3049221462>) - 2019-07-22 · `affiliated` · Automated Development of Deep Learning Models to Diagnose Retinal Disease from Fundus and Optical Coherence Tomography Images
- [gdm-arxiv-1907.09633](<http://arxiv.org/abs/1907.09633>) - 2019-07-22 · `affiliated` · Low-Variance and Zero-Variance Baselines for Extensive-Form Games
- [gdm-arxiv-1907.12906](<https://doi.org/10.1007/s40300-019-00155-4>) - 2019-07-20 · `affiliated` · Unsupervised separation of dynamics from pixels
- [gdm-doi-10.1609-aaai.v33i01.33019751](<https://ojs.aaai.org/index.php/AAAI/article/view/5044>) - 2019-07-17 · `affiliated` · Model AI Assignments 2019
- [gdm-doi-10.1609-aaai.v33i01.3301216](<https://doi.org/10.1609/aaai.v33i01.3301216>) - 2019-07-17 · `affiliated` · SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning
- [gdm-arxiv-1907.07751](<https://ojs.aaai.org/index.php/AAAI/article/view/4284>) - 2019-07-17 · `affiliated` · Meta-Descent for Online, Continual Prediction
- [gdm-arxiv-1809.07893](<https://ojs.aaai.org/index.php/AAAI/article/view/4011>) - 2019-07-17 · `affiliated` · Solving Large Extensive-Form Games with Strategy Constraints
- [gdm-arxiv-1809.04474](<https://doi.org/10.1609/aaai.v33i01.33013796>) - 2019-07-17 · `affiliated` · Multi-Task Deep Reinforcement Learning with PopArt
- [gdm-arxiv-1809.03057](<https://ojs.aaai.org/index.php/AAAI/article/view/4048>) - 2019-07-17 · `affiliated` · Variance Reduction in Monte Carlo Counterfactual Regret Minimization (VR-MCCFR) for Extensive Form Games Using Baselines
- [gdm-arxiv-1802.08139](<https://ojs.aaai.org/index.php/AAAI/article/view/4777>) - 2019-07-17 · `affiliated` · Path-Specific Counterfactual Fairness
- [gdm-doi-10.2196-13147](<https://doi.org/10.2196/13147>) - 2019-07-15 · `affiliated` · Implementation of a Digitally Enabled Care Pathway (Part 1): Impact on Clinical Outcomes and Associated Health Care Costs
- [gdm-web-2ed7f25bfe25](<https://hal.science/hal-02177808>) - 2019-07-09 · `affiliated` · Active Roll-outs in MDP with Irreversible Dynamics
- [gdm-doi-10.1038-s41598-019-45619-9](<https://doi.org/10.1038/s41598-019-45619-9>) - 2019-07-09 · `affiliated` · α-Rank: Multi-Agent Evaluation by Evolution
- [gdm-doi-10.1101-695924](<https://doi.org/10.1101/695924>) - 2019-07-08 · `affiliated` · Modelling variation in bushmeat harvesting among seven African ecosystems using the Madingley Model: yield, survival and ecosystem impacts
- [gdm-doi-10.1016-j.cell.2019.06.012](<https://doi.org/10.1016/j.cell.2019.06.012>) - 2019-07-01 · `affiliated` · Human Replay Spontaneously Reorganizes Experience
- [gdm-arxiv-1705.09189](<https://doi.org/10.1017/s1351324919000184>) - 2019-07-01 · `affiliated` · Jointly learning sentence embeddings and syntax with unsupervised Tree-LSTMs
- [gdm-doi-10.1016-s2589-7500-19-30057-3](<https://doi.org/10.1016/s2589-7500(19)30057-3>) - 2019-06-27 · `affiliated` · The effects and preventability of 2627 patient safety incidents related to health information technology failures: a retrospective analysis of 10 years of incident reporting in England and Wales
- [gdm-doi-10.15607-rss.2019.xv.027](<https://doi.org/10.15607/rss.2019.xv.027>) - 2019-06-22 · `affiliated` · Simultaneously Learning Vision and Feature-Based Control Policies for Real-World Ball-In-A-Cup
- [gdm-doi-10.1177-1460458219854602](<https://doi.org/10.1177/1460458219854602>) - 2019-06-15 · `affiliated` · A regulatory perspective on the influence of health information technology on organisational quality and safety in England
- [gdm-url-a2076299ed75](<http://hdl.handle.net/20.500.12210/22452>) - 2019-06-14 · `affiliated` · A simple dynamic bandit algorithm for hyper-parameter tuning
- [gdm-doi-10.1038-s41593-019-0414-3](<https://doi.org/10.1038/s41593-019-0414-3>) - 2019-06-10 · `affiliated` · Circuit mechanisms for the maintenance and manipulation of information in working memory
- [gdm-doi-10.1145-3314221.3314622](<https://doi.org/10.1145/3314221.3314622>) - 2019-06-07 · `affiliated` · Verified compilation on a verified processor
- [gdm-doi-10.2200-s00920ed2v01y201904hlt042](<https://doi.org/10.2200/s00920ed2v01y201904hlt042>) - 2019-06-04 · `affiliated` · Cross-Lingual Word Embeddings
- [gdm-arxiv-1906.01681](<http://arxiv.org/abs/1906.01681>) - 2019-06-04 · `affiliated` · Learning dynamic polynomial proofs
- [gdm-doi-10.1109-cvpr.2019.01254](<https://doi.org/10.1109/cvpr.2019.01254>) - 2019-06-01 · `affiliated` · Knowing When to Stop: Evaluation and Verification of Conformity to Output-Size Specifications
- [gdm-arxiv-1905.04266](<https://doi.org/10.1109/cvpr.2019.00351>) - 2019-06-01 · `affiliated` · Exploiting Temporal Context for 3D Human Pose Estimation in the Wild
- [gdm-arxiv-1904.07846](<https://doi.org/10.1109/cvpr.2019.00190>) - 2019-06-01 · `affiliated` · Temporal Cycle-Consistency Learning
- [gdm-arxiv-1903.08225](<https://doi.org/10.1109/cvpr.2019.00365>) - 2019-06-01 · `affiliated` · Cross-Task Weakly Supervised Learning From Instructional Videos
- [gdm-arxiv-1812.07252](<https://doi.org/10.1109/cvpr.2019.01291>) - 2019-06-01 · `affiliated` · Sim-To-Real via Sim-To-Sim: Data-Efficient Robotic Grasping via Randomized-To-Canonical Adaptation Networks
- [gdm-arxiv-1812.02707](<https://doi.org/10.1109/cvpr.2019.00033>) - 2019-06-01 · `affiliated` · Video Action Transformer Network
- [gdm-arxiv-1812.01461](<https://doi.org/10.1109/cvpr.2019.00256>) - 2019-06-01 · `affiliated` · The Visual Centrifuge: Model-Free Layered Video Representations
- [gdm-arxiv-1811.09716](<https://doi.org/10.1109/cvpr.2019.00929>) - 2019-06-01 · `affiliated` · Robustness via Curvature Regularization, and Vice Versa
- [gdm-arxiv-1905.12941](<http://arxiv.org/abs/1905.12941>) - 2019-05-30 · `affiliated` · Learning Compositional Neural Programs with Recursive Tree Search and Planning
- [gdm-arxiv-1807.01281](<https://doi.org/10.1126/science.aau6249>) - 2019-05-30 · `affiliated` · Human-level performance in 3D multiplayer games with population-based reinforcement learning
- [gdm-arxiv-1905.10307](<http://arxiv.org/abs/1905.10307>) - 2019-05-24 · `affiliated` · An Explicitly Relational Neural Network Architecture
- [gdm-doi-10.1038-s41562-019-0595-5](<https://doi.org/10.1038/s41562-019-0595-5>) - 2019-05-20 · `affiliated` · Slow escape decisions are swayed by trait anxiety
- [gdm-doi-10.1038-s41746-019-0118-9](<https://doi.org/10.1038/s41746-019-0118-9>) - 2019-05-16 · `affiliated` · Evaluating the impact of organisational digital maturity on clinical outcomes in secondary care in England
- [gdm-doi-10.65109-vfrm7723](<https://doi.org/10.65109/vfrm7723>) - 2019-05-08 · `affiliated` · Computing Stable Solutions in Threshold Network Flow Games With Bounded Treewidth
- [gdm-doi-10.65109-nqzd5621](<https://doi.org/10.65109/nqzd5621>) - 2019-05-08 · `affiliated` · Observational Learning by Reinforcement Learning
- [gdm-doi-10.65109-mfry9850](<https://doi.org/10.65109/mfry9850>) - 2019-05-08 · `affiliated` · The Body is Not a Given: Joint Agent Policy Learning and Morphology Evolution
- [gdm-doi-10.65109-afng4004](<https://doi.org/10.65109/afng4004>) - 2019-05-08 · `affiliated` · The Imitation Game: Learned Reciprocity in Markov games
- [gdm-arxiv-1812.07019](<https://doi.org/10.65109/fdir9565>) - 2019-05-08 · `affiliated` · Malthusian Reinforcement Learning
- [gdm-arxiv-1811.05931](<https://doi.org/10.65109/vtgr9509>) - 2019-05-08 · `affiliated` · Evolving Intrinsic Motivations for Altruistic Behavior
- [gdm-arxiv-1905.00537](<https://papers.nips.cc/paper/8589-superglue-a-stickier-benchmark-for-general-purpose-language-understanding-systems.pdf>) - 2019-05-02 · `affiliated` · SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
- [gdm-doi-10.1109-dcoss.2019.00028](<https://doi.org/10.1109/dcoss.2019.00028>) - 2019-05-01 · `affiliated` · mID: Tracking and Identifying People with Millimeter Wave Radar
- [gdm-arxiv-1810.01531](<https://doi.org/10.1109/icra.2019.8794074>) - 2019-05-01 · `affiliated` · A Practical Approach to Insertion with Variable Socket Position Using Deep Reinforcement Learning
- [gdm-arxiv-1809.07004](<https://doi.org/10.1109/icra.2019.8793733>) - 2019-05-01 · `affiliated` · Leveraging Contact Forces for Learning to Grasp
- [gdm-doi-10.1016-j.tics.2019.02.006](<https://doi.org/10.1016/j.tics.2019.02.006>) - 2019-04-16 · `affiliated` · Reinforcement Learning, Fast and Slow
- [gdm-arxiv-1910.06464](<https://doi.org/10.1109/icassp.2019.8683277>) - 2019-04-16 · `affiliated` · Low Bit-rate Speech Coding with VQ-VAE and a WaveNet Decoder
- [gdm-doi-10.1101-588699](<https://doi.org/10.1101/588699>) - 2019-03-26 · `affiliated` · The value of what’s to come: neural mechanisms coupling prediction error and reward anticipation
- [gdm-doi-10.1101-588665](<https://doi.org/10.1101/588665>) - 2019-03-26 · `affiliated` · Food Webs: Insights from a General Ecosystem Model
- [gdm-doi-10.2196-13143](<https://doi.org/10.2196/13143>) - 2019-03-24 · `affiliated` · Implementation of a Digitally Enabled Care Pathway (Part 2): Qualitative Analysis of Experiences of Health Care Professionals
- [gdm-doi-10.1016-j.radonc.2019.03.004](<https://doi.org/10.1016/j.radonc.2019.03.004>) - 2019-03-22 · `affiliated` · Rapid advances in auto-segmentation of organs at risk and target volumes in head and neck cancer
- [gdm-doi-10.1117-12.2519413](<https://doi.org/10.1117/12.2519413>) - 2019-03-14 · `affiliated` · The U-net and its impact to medical imaging (Conference Presentation)
- [gdm-doi-10.1109-tcyb.2019.2901499](<https://doi.org/10.1109/tcyb.2019.2901499>) - 2019-03-14 · `affiliated` · A Generic Human–Machine Annotation Framework Based on Dynamic Cooperative Learning
- [gdm-doi-10.1016-j.ijmedinf.2019.03.003](<https://doi.org/10.1016/j.ijmedinf.2019.03.003>) - 2019-03-07 · `affiliated` · Exploring mobile working in healthcare: Clinical perspectives on transitioning to a mobile first culture of work
- [gdm-doi-10.1016-j.conb.2019.01.011](<https://doi.org/10.1016/j.conb.2019.01.011>) - 2019-03-06 · `affiliated` · Backpropagation through time and the brain
- [gdm-arxiv-1902.09996](<http://arxiv.org/abs/1902.09996>) - 2019-02-26 · `affiliated` · The Termination Critic
- [gdm-arxiv-1810.11542](<https://doi.org/10.1613/jair.1.11370>) - 2019-02-25 · `affiliated` · Revisiting CFR+ and Alternating Updates
- [gdm-doi-10.1007-s10458-019-09402-4](<https://doi.org/10.1007/s10458-019-09402-4>) - 2019-02-22 · `affiliated` · Strategic behavior and learning in all-pay auctions: an empirical study using crowdsourced data
- [gdm-arxiv-1810.13373](<https://doi.org/10.1016/j.conb.2019.01.007>) - 2019-02-19 · `affiliated` · Analyzing biological and artificial neural networks: challenges with opportunities for synergy?
- [gdm-arxiv-1810.05246](<https://doi.org/10.1145/3301275.3302288>) - 2019-02-19 · `affiliated` · Piano Genie
- [gdm-arxiv-1812.02941](<https://doi.org/10.1109/lra.2019.2899192>) - 2019-02-13 · `affiliated` · From Pixels to Percepts: Highly Robust Edge Perception and Contour Following Using Deep Learning and an Optical Biomimetic Tactile Sensor
- [gdm-doi-10.3389-frym.2019.00009](<https://doi.org/10.3389/frym.2019.00009>) - 2019-02-07 · `affiliated` · Action Recognition Happens Very Quickly in the Human Brain
- [gdm-arxiv-1902.02186](<http://arxiv.org/abs/1902.02186>) - 2019-02-06 · `affiliated` · Distilling Policy Distillation
- [gdm-doi-10.1016-j.neucom.2018.10.084](<https://doi.org/10.1016/j.neucom.2018.10.084>) - 2019-02-04 · `affiliated` · Adaptive long-term control of biological neural networks with Deep Reinforcement Learning
- [gdm-arxiv-1705.06769](<https://doi.org/10.1109/tnnls.2019.2891792>) - 2019-01-29 · `affiliated` · Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning
- [gdm-arxiv-1902.10730](<https://doi.org/10.1145/3306618.3314288>) - 2019-01-27 · `affiliated` · Degenerate Feedback Loops in Recommender Systems
- [gdm-arxiv-1705.09575](<https://doi.org/10.1177/1471082x18817650>) - 2019-01-23 · `affiliated` · Ranking soccer teams on the basis of their current strength: A comparison of maximum likelihood approaches
- [gdm-doi-10.1126-scirobotics.aav2975](<https://doi.org/10.1126/scirobotics.aav2975>) - 2019-01-17 · `affiliated` · Toward high-performance, memory-efficient, and fast reinforcement learning—Lessons from decision neuroscience
- [gdm-doi-10.47102-annals-acadmedsg.v48n1p1](<https://doi.org/10.47102/annals-acadmedsg.v48n1p1>) - 2019-01-15 · `affiliated` · Deep Learning in Medicine. Are We Ready?
- [gdm-arxiv-1901.04884](<http://arxiv.org/abs/1901.04884>) - 2019-01-15 · `affiliated` · Optimistic optimization of a Brownian
- [gdm-doi-10.1038-s41467-018-08194-7](<https://doi.org/10.1038/s41467-018-08194-7>) - 2019-01-11 · `affiliated` · Activity in perceptual classification networks as a basis for human subjective time perception
- [gdm-arxiv-1901.01761](<http://arxiv.org/abs/1901.01761>) - 2019-01-07 · `affiliated` · Credit Assignment Techniques in Stochastic Computation Graphs
- [gdm-doi-10.21105-jose.00032](<https://doi.org/10.21105/jose.00032>) - 2019-01-06 · `affiliated` · nbgrader: A Tool for Creating and Grading Assignments in the Jupyter Notebook
- [gdm-doi-10.1016-j.cobeha.2018.12.011](<https://doi.org/10.1016/j.cobeha.2018.12.011>) - 2019-01-04 · `affiliated` · Analogues of mental simulation and imagination in deep learning
- [gdm-doi-10.1016-j.cobeha.2018.12.010](<https://doi.org/10.1016/j.cobeha.2018.12.010>) - 2019-01-04 · `affiliated` · Reconciling deep learning with symbolic artificial intelligence: representing objects and relations
- [gdm-doi-10.1038-s41593-018-0310-2](<https://doi.org/10.1038/s41593-018-0310-2>) - 2019-01-03 · `affiliated` · Task representations in neural networks trained to perform many cognitive tasks
- [gdm-web-f909ad98901b](<https://hal.science/hal-02277739>) - 2019-01-01 · `affiliated` · On two ways to use determinantal point processes for Monte Carlo integration -- Long version
- [gdm-web-390ae5c8a0e4](<https://inria.hal.science/hal-02387478>) - 2019-01-01 · `affiliated` · Exploiting structure of uncertainty for efficient matroid semi-bandits
- [gdm-doi-10.32470-ccn.2019.1197-0](<https://doi.org/10.32470/ccn.2019.1197-0>) - 2019-01-01 · `affiliated` · Compositional Neural Representations in the Hippocampal Formation and Prefrontal Cortex Underlie Visual Construction and Planning
- [gdm-doi-10.2139-ssrn.3402015](<https://doi.org/10.2139/ssrn.3402015>) - 2019-01-01 · `affiliated` · Feasibility of Automated Deep Learning Design for Medical Image Classification by Healthcare Professionals with Limited Coding Experience
- [gdm-doi-10.2139-ssrn.3384923](<https://doi.org/10.2139/ssrn.3384923>) - 2019-01-01 · `affiliated` · Deep Learning Under Scrutiny: Performance Against Health Care Professionals in Detecting Diseases from Medical Imaging - Systematic Review and Meta-Analysis
- [gdm-doi-10.18653-v1-p19-1645](<https://doi.org/10.18653/v1/p19-1645>) - 2019-01-01 · `affiliated` · Learning to Discover, Ground and Use Words with Segmental Neural Language Models
- [gdm-doi-10.18653-v1-n19-1114](<https://doi.org/10.18653/v1/n19-1114>) - 2019-01-01 · `affiliated` · Unsupervised Recurrent Neural Network Grammars
- [gdm-doi-10.18653-v1-d19-1668](<https://doi.org/10.18653/v1/d19-1668>) - 2019-01-01 · `affiliated` · Restoring ancient text using deep learning: a case study on Greek epigraphy
- [gdm-doi-10.18653-v1-d19-1594](<https://doi.org/10.18653/v1/d19-1594>) - 2019-01-01 · `affiliated` · Text Genre and Training Data Size in Human-like Parsing
- [gdm-doi-10.1162-isal_a_00148](<https://doi.org/10.1162/isal_a_00148>) - 2019-01-01 · `affiliated` · Reinforcement Learning Agents acquire Flocking and Symbiotic Behaviour in Simulated Ecosystems
- [gdm-doi-10.1093-nc-niz004](<https://doi.org/10.1093/nc/niz004>) - 2019-01-01 · `affiliated` · Confidence modulates exploration and exploitation in value-based learning
- [gdm-doi-10.1007-978-3-030-20890-5_3](<https://doi.org/10.1007/978-3-030-20890-5_3>) - 2019-01-01 · `affiliated` · GhostVLAD for Set-Based Face Recognition
- [gdm-doi-10.1007-978-3-030-11018-5_36](<https://doi.org/10.1007/978-3-030-11018-5_36>) - 2019-01-01 · `affiliated` · Compact Deep Aggregation for Set Retrieval
- [gdm-doi-10.1007-978-3-030-01800-9_13](<https://doi.org/10.1007/978-3-030-01800-9_13>) - 2019-01-01 · `affiliated` · A Kantian Cognitive Architecture
- [gdm-arxiv-1908.08025](<https://doi.org/10.18653/v1/d19-1439>) - 2019-01-01 · `affiliated` · WikiCREM: A Large Unsupervised Corpus for Coreference Resolution
- [gdm-arxiv-1907.06430](<https://doi.org/10.1007/978-3-030-16744-8_1>) - 2019-01-01 · `affiliated` · A Causal Bayesian Networks Viewpoint on Fairness
- [gdm-arxiv-1905.13476](<https://inria.hal.science/hal-02387524>) - 2019-01-01 · `affiliated` · Exact sampling of determinantal point processes with sublinear time preprocessing
- [gdm-arxiv-1904.08873](<https://doi.org/10.1017/s0140525x19001407>) - 2019-01-01 · `affiliated` · Codes, functions, and causes: A critique of Brette's conceptual analysis of coding
- [gdm-arxiv-1710.04971](<https://hdl.handle.net/11511/92663>) - 2019-01-01 · `affiliated` · Average Age of Information With Hybrid ARQ Under a Resource Constraint

### 17.9 2018（107 条）

- [gdm-doi-10.1613-jair.1.11270](<https://doi.org/10.1613/jair.1.11270>) - 2018-12-27 · `affiliated` · Bounds on the Cost of Stabilizing a Cooperative Game
- [gdm-arxiv-1809.02108](<https://doi.org/10.1109/tpami.2018.2889052>) - 2018-12-21 · `affiliated` · Deep Audio-Visual Speech Recognition
- [gdm-doi-10.1007-s10994-018-5767-4](<https://doi.org/10.1007/s10994-018-5767-4>) - 2018-12-19 · `affiliated` · Bayesian optimistic Kullback–Leibler exploration
- [gdm-doi-10.1126-science.aar6404](<https://doi.org/10.1126/science.aar6404>) - 2018-12-06 · `affiliated` · A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play
- [gdm-arxiv-1812.02648](<http://arxiv.org/abs/1812.02648>) - 2018-12-06 · `affiliated` · Deep Reinforcement Learning and the Deadly Triad
- [gdm-doi-10.1038-s41592-018-0261-2](<https://doi.org/10.1038/s41592-018-0261-2>) - 2018-12-04 · `affiliated` · U-Net: deep learning for cell counting, detection, and morphometry
- [gdm-arxiv-1806.03335](<https://papers.nips.cc/paper/2018/file/5a7b238ba0f6502e5d6be14424b20ded-Paper.pdf>) - 2018-12-03 · `affiliated` · Randomized prior functions for deep reinforcement learning
- [gdm-doi-10.1609-aimag.v39i4.2824](<https://doi.org/10.1609/aimag.v39i4.2824>) - 2018-12-01 · `affiliated` · Reports on the 2018 AAAI Spring Symposium Series
- [gdm-arxiv-1712.07040](<https://doi.org/10.1162/tacl_a_00023>) - 2018-12-01 · `affiliated` · The NarrativeQA Reading Comprehension Challenge
- [gdm-doi-10.1093-jamia-ocy175](<https://doi.org/10.1093/jamia/ocy175>) - 2018-11-29 · `affiliated` · The impact of mobile technology on teamwork and communication in hospitals: a systematic review
- [gdm-arxiv-1811.11043](<http://arxiv.org/abs/1811.11043>) - 2018-11-27 · `affiliated` · Rotting bandits are not harder than stochastic ones
- [gdm-doi-10.1523-jneurosci.0706-18.2018](<https://doi.org/10.1523/jneurosci.0706-18.2018>) - 2018-11-19 · `affiliated` · Computing Value from Quality and Quantity in Human Decision-Making
- [gdm-arxiv-1808.03715](<https://doi.org/10.1007/s00521-018-3758-9>) - 2018-11-14 · `affiliated` · This time with feeling: learning expressive musical performance
- [gdm-doi-10.1101-461129](<https://doi.org/10.1101/461129>) - 2018-11-02 · `affiliated` · From predictive models to cognitive models: Separable behavioral processes underlying reward learning in the rat
- [gdm-doi-10.1109-humanoids.2018.8624995](<https://doi.org/10.1109/humanoids.2018.8624995>) - 2018-11-01 · `affiliated` · Learning Robust Task Priorities of QP-Based Whole-Body Torque-Controllers
- [gdm-arxiv-1810.11428](<http://arxiv.org/abs/1810.11428>) - 2018-10-26 · `affiliated` · Resampled Priors for Variational Autoencoders
- [gdm-arxiv-1810.10510](<https://doi.org/10.48550/arxiv.1810.10510>) - 2018-10-24 · `affiliated` · Neighbourhood Consensus Networks
- [gdm-doi-10.1145-3269206.3272922](<https://doi.org/10.1145/3269206.3272922>) - 2018-10-17 · `affiliated` · Teaching Artificial Agents to Understand Language by Modelling Reward
- [gdm-doi-10.1073-pnas.1800755115](<https://doi.org/10.1073/pnas.1800755115>) - 2018-10-15 · `affiliated` · Comparing continual task learning in minds and machines
- [gdm-doi-10.1016-j.neuron.2018.10.002](<https://doi.org/10.1016/j.neuron.2018.10.002>) - 2018-10-01 · `affiliated` · What Is a Cognitive Map? Organizing Knowledge for Flexible Behavior
- [gdm-doi-10.1145-3240323.3240333](<https://doi.org/10.1145/3240323.3240333>) - 2018-09-27 · `affiliated` · DLRS 2018
- [gdm-arxiv-1809.10460](<https://arxiv.org/pdf/1809.10460.pdf>) - 2018-09-27 · `affiliated` · Sample-efficient adaptive text-to-speech
- [gdm-doi-10.1162-jocn_a_01341](<https://doi.org/10.1162/jocn_a_01341>) - 2018-09-21 · `affiliated` · Subgoal- and Goal-related Reward Prediction Errors in Medial Prefrontal Cortex
- [gdm-doi-10.1016-j.conb.2018.08.003](<https://doi.org/10.1016/j.conb.2018.08.003>) - 2018-09-08 · `affiliated` · Dendritic solutions to the credit assignment problem
- [gdm-doi-10.1016-j.neuron.2018.08.009](<https://doi.org/10.1016/j.neuron.2018.08.009>) - 2018-09-01 · `affiliated` · Big-Loop Recurrence within the Hippocampal System Supports Integration of Information across Episodes
- [gdm-doi-10.1038-s41562-018-0401-9](<https://doi.org/10.1038/s41562-018-0401-9>) - 2018-08-29 · `affiliated` · Mental labour
- [gdm-arxiv-1808.09352](<http://arxiv.org/abs/1808.09352>) - 2018-08-28 · `affiliated` · Evaluating Theory of Mind in Question Answering
- [gdm-doi-10.1038-s41583-018-0049-5](<https://doi.org/10.1038/s41583-018-0049-5>) - 2018-08-14 · `affiliated` · Can neocortical feedback alter the sign of plasticity?
- [gdm-arxiv-1808.04468](<http://arxiv.org/abs/1808.04468>) - 2018-08-13 · `affiliated` · Risk-Sensitive Generative Adversarial Imitation Learning
- [gdm-arxiv-1703.05593](<https://doi.org/10.1109/tpami.2018.2865351>) - 2018-08-13 · `affiliated` · Convolutional Neural Network Architecture for Geometric Matching
- [gdm-doi-10.1038-s41591-018-0107-6](<https://doi.org/10.1038/s41591-018-0107-6>) - 2018-08-06 · `affiliated` · Clinically applicable deep learning for diagnosis and referral in retinal disease
- [gdm-arxiv-1808.02078](<http://arxiv.org/abs/1808.02078>) - 2018-08-06 · `affiliated` · Unbiased Implicit Variational Inference
- [gdm-doi-10.1287-ijoc.2017.0798](<https://doi.org/10.1287/ijoc.2017.0798>) - 2018-08-01 · `affiliated` · What Works Best When? A Systematic Evaluation of Heuristics for Max-Cut and QUBO
- [gdm-web-e368bd89fa8d](<https://openalex.org/W2964272379>) - 2018-07-19 · `affiliated` · Multitask Reinforcement Learning for Zero-shot Generalization with Subtask Dependencies
- [gdm-web-6c91776121c8](<https://iovs.arvojournals.org/article.aspx?articleid=2690068>) - 2018-07-13 · `affiliated` · Predicting refractive error from retinal fundus images using deep learning
- [gdm-doi-10.1561-2200000070](<https://doi.org/10.1561/2200000070>) - 2018-07-12 · `affiliated` · A Tutorial on Thompson Sampling
- [gdm-doi-10.1101-365593](<https://doi.org/10.1101/365593>) - 2018-07-10 · `affiliated` · What is a cognitive map? Organising knowledge for flexible behaviour
- [gdm-doi-10.65109-rqxk8220](<https://doi.org/10.65109/rqxk8220>) - 2018-07-09 · `affiliated` · Evolving Coverage Behaviours For MAVs Using NEAT
- [gdm-doi-10.65109-jsrc7365](<https://doi.org/10.65109/jsrc7365>) - 2018-07-09 · `affiliated` · Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward
- [gdm-arxiv-1803.06376](<https://doi.org/10.65109/dbto3701>) - 2018-07-09 · `affiliated` · A Generalised Method for Empirical Game Theoretic Analysis
- [gdm-arxiv-1707.04402](<https://doi.org/10.65109/qdcv6054>) - 2018-07-09 · `affiliated` · Lenient Multi-Agent Deep Reinforcement Learning
- [gdm-arxiv-1806.07917](<https://doi.org/10.1145/3205651.3205763>) - 2018-07-06 · `affiliated` · Meta-learning by the baldwin effect
- [gdm-web-547af9cc2045](<https://openalex.org/W2810080174>) - 2018-07-04 · `affiliated` · Neural signatures of detours, shortcuts and back-tracking during navigation
- [gdm-doi-10.1101-360537](<https://doi.org/10.1101/360537>) - 2018-07-03 · `affiliated` · Episodic Control as Meta-Reinforcement Learning
- [gdm-arxiv-1802.03753](<https://doi.org/10.1109/taslp.2018.2851664>) - 2018-07-03 · `affiliated` · Sample Efficient Deep Reinforcement Learning for Dialogue Systems With Large Action Spaces
- [gdm-doi-10.24963-ijcai.2018-792](<https://doi.org/10.24963/ijcai.2018/792>) - 2018-07-01 · `affiliated` · Learning Explanatory Rules from Noisy Data (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2018-787](<https://doi.org/10.24963/ijcai.2018/787>) - 2018-07-01 · `affiliated` · Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2018-666](<https://doi.org/10.24963/ijcai.2018/666>) - 2018-07-01 · `affiliated` · Organizing Experience: a Deeper Look at Replay Mechanisms for Sample-Based Planning in Continuous State Domains
- [gdm-doi-10.15607-rss.2018.xiv.010](<https://doi.org/10.15607/rss.2018.xiv.010>) - 2018-06-26 · `affiliated` · Sim-to-Real: Learning Agile Locomotion For Quadruped Robots
- [gdm-doi-10.15607-rss.2018.xiv.009](<https://doi.org/10.15607/rss.2018.xiv.009>) - 2018-06-26 · `affiliated` · Reinforcement and Imitation Learning for Diverse Visuomotor Skills
- [gdm-doi-10.1038-s41467-018-04841-1](<https://doi.org/10.1038/s41467-018-04841-1>) - 2018-06-21 · `affiliated` · Dissociable neural mechanisms track evidence accumulation for selection of attention versus action
- [gdm-arxiv-1806.06827](<http://arxiv.org/abs/1806.06827>) - 2018-06-18 · `affiliated` · PAC-Bayes bounds for stable algorithms with instance-dependent priors
- [gdm-doi-10.1109-tpami.2018.2847688](<https://doi.org/10.1109/tpami.2018.2847688>) - 2018-06-15 · `affiliated` · Opening the Black Box: Hierarchical Sampling Optimization for Hand Pose Estimation
- [gdm-doi-10.1126-science.aar6170](<https://doi.org/10.1126/science.aar6170>) - 2018-06-14 · `affiliated` · Neural scene representation and rendering
- [gdm-arxiv-1712.07798](<https://doi.org/10.1167/iovs.18-23887>) - 2018-06-04 · `affiliated` · Deep Learning for Predicting Refractive Error From Retinal Fundus Images
- [gdm-doi-10.17863-cam.42159](<https://www.repository.cam.ac.uk/handle/1810/295082>) - 2018-06-01 · `affiliated` · Learning disentangled representations with semi-supervised deep generative models
- [gdm-doi-10.1109-cvpr.2018.00481](<https://doi.org/10.1109/cvpr.2018.00481>) - 2018-06-01 · `affiliated` · ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans
- [gdm-doi-10.1007-s42113-018-0007-3](<https://doi.org/10.1007/s42113-018-0007-3>) - 2018-06-01 · `affiliated` · Different Physical Intuitions Exist Between Tasks, Not Domains
- [gdm-arxiv-1712.06861](<https://doi.org/10.1109/cvpr.2018.00723>) - 2018-06-01 · `affiliated` · End-to-End Weakly-Supervised Semantic Alignment
- [gdm-arxiv-1711.05908](<https://doi.org/10.1109/cvpr.2018.00958>) - 2018-06-01 · `affiliated` · NISP: Pruning Networks Using Neuron Importance Score Propagation
- [gdm-arxiv-1711.05187](<https://doi.org/10.1109/cvpr.2018.00724>) - 2018-06-01 · `affiliated` · Dynamic Zoom-in Network for Fast Object Detection in Large Images
- [gdm-arxiv-1805.09801](<http://arxiv.org/abs/1805.09801>) - 2018-05-24 · `affiliated` · Meta-Gradient Reinforcement Learning
- [gdm-doi-10.1214-17-ba1093](<https://doi.org/10.1214/17-ba1093>) - 2018-05-19 · `affiliated` · Modeling Population Structure Under Hierarchical Dirichlet Processes
- [gdm-doi-10.1038-s41593-018-0147-8](<https://doi.org/10.1038/s41593-018-0147-8>) - 2018-05-11 · `affiliated` · Prefrontal cortex as a meta-reinforcement learning system
- [gdm-doi-10.1111-ele.12974](<https://doi.org/10.1111/ele.12974>) - 2018-05-09 · `affiliated` · Beyond the fast–slow continuum: demographic dimensions structuring a tropical tree community
- [gdm-doi-10.1007-s00224-018-9865-2](<https://doi.org/10.1007/s00224-018-9865-2>) - 2018-05-05 · `affiliated` · Analyzing Power in Weighted Voting Games with Super-Increasing Weights
- [gdm-doi-10.1109-icra.2018.8461176](<https://doi.org/10.1109/icra.2018.8461176>) - 2018-05-01 · `affiliated` · Distance-Based Multi-Robot Coordination on Pocket Drones
- [gdm-doi-10.1038-s41586-018-0102-6](<https://doi.org/10.1038/s41586-018-0102-6>) - 2018-05-01 · `affiliated` · Vector-based navigation using grid-like representations in artificial agents
- [gdm-arxiv-1710.10044](<https://doi.org/10.1609/aaai.v32i1.11791>) - 2018-04-29 · `affiliated` · Distributional Reinforcement Learning With Quantile Regression
- [gdm-arxiv-1710.02298](<https://doi.org/10.1609/aaai.v32i1.11796>) - 2018-04-29 · `affiliated` · Rainbow: Combining Improvements in Deep Reinforcement Learning
- [gdm-arxiv-1708.00111](<https://ojs.aaai.org/index.php/AAAI/article/view/11806>) - 2018-04-29 · `affiliated` · A Continuous Relaxation of Beam Search for End-to-End Training of Neural Sequence Models
- [gdm-arxiv-1704.03732](<https://doi.org/10.1609/aaai.v32i1.11757>) - 2018-04-29 · `affiliated` · Deep Q-learning From Demonstrations
- [gdm-doi-10.1371-journal.pbio.2004752](<https://doi.org/10.1371/journal.pbio.2004752>) - 2018-04-24 · `affiliated` · Agent-specific learning signals for self–other distinction during mentalising
- [gdm-arxiv-1805.01772](<https://doi.org/10.1145/3190508.3190551>) - 2018-04-18 · `affiliated` · Dynamic control flow in large-scale machine learning
- [gdm-arxiv-1804.06021](<http://arxiv.org/abs/1804.06021>) - 2018-04-17 · `affiliated` · Model-Free Linear Quadratic Control via Reduction to Expert Prediction
- [gdm-doi-10.1007-s12553-018-0228-4](<https://doi.org/10.1007/s12553-018-0228-4>) - 2018-04-11 · `affiliated` · Letter in response to Google DeepMind and healthcare in an age of algorithms
- [gdm-doi-10.1038-s41467-018-03837-1](<https://doi.org/10.1038/s41467-018-03837-1>) - 2018-04-03 · `affiliated` · Stimulus dependent diversity and stereotypy in the output of an olfactory functional unit
- [gdm-doi-10.1109-ispass.2018.00029](<https://doi.org/10.1109/ispass.2018.00029>) - 2018-04-01 · `affiliated` · The Alberta Workloads for the SPEC CPU 2017 Benchmark Suite
- [gdm-doi-10.1109-isbi.2018.8363792](<https://doi.org/10.1109/isbi.2018.8363792>) - 2018-04-01 · `affiliated` · ISOO&lt;inf&gt;DL&lt;/inf&gt;: Instance segmentation of overlapping biological objects using deep learning
- [gdm-doi-10.1109-icassp.2018.8461921](<https://doi.org/10.1109/icassp.2018.8461921>) - 2018-04-01 · `affiliated` · Temporal Modeling Using Dilated Convolution and Gating for Voice-Activity-Detection
- [gdm-arxiv-1804.06557](<http://purl.org/au-research/grants/arc/FT140101229>) - 2018-04-01 · `affiliated` · The limits and potentials of deep learning for robotics
- [gdm-arxiv-1802.04200](<https://doi.org/10.1109/icassp.2018.8461690>) - 2018-04-01 · `affiliated` · End-to-End Automatic Speech Translation of Audiobooks
- [gdm-arxiv-1712.01120](<https://doi.org/10.1109/icassp.2018.8462529>) - 2018-04-01 · `affiliated` · Wavenet Based Low Rate Speech Coding
- [gdm-doi-10.1016-j.automatica.2018.03.009](<https://doi.org/10.1016/j.automatica.2018.03.009>) - 2018-03-21 · `affiliated` · Continuous-action planning for discounted infinite-horizon nonlinear optimal control with Lipschitz values
- [gdm-doi-10.1073-pnas.1712314115](<https://doi.org/10.1073/pnas.1712314115>) - 2018-03-05 · `affiliated` · How cognitive and reactive fear circuits optimize escape decisions in humans
- [gdm-arxiv-1712.02151](<https://doi.org/10.1109/dcc.2018.00033>) - 2018-03-01 · `affiliated` · Generalized Probability Smoothing
- [gdm-doi-10.1038-s41467-018-03068-4](<https://doi.org/10.1038/s41467-018-03068-4>) - 2018-02-28 · `affiliated` · Toward a universal decoder of linguistic meaning from brain activation
- [gdm-doi-10.1016-j.neuroimage.2018.01.071](<https://doi.org/10.1016/j.neuroimage.2018.01.071>) - 2018-02-12 · `affiliated` · A probabilistic approach to discovering dynamic full-brain functional connectivity patterns
- [gdm-arxiv-1711.04574](<https://doi.org/10.1613/jair.5714>) - 2018-01-26 · `affiliated` · Learning Explanatory Rules from Noisy Data
- [gdm-arxiv-1801.08116](<https://deepmind.google/blog/open-sourcing-psychlab/>) - 2018-01-24 · `affiliated` · Psychlab: A Psychology Laboratory for Deep Reinforcement Learning Agents
- [gdm-arxiv-1711.05074](<https://doi.org/10.1038/s41598-018-19194-4>) - 2018-01-11 · `affiliated` · Symmetric Decomposition of Asymmetric Games
- [gdm-doi-10.1101-245829](<https://doi.org/10.1101/245829>) - 2018-01-10 · `affiliated` · Subgoal- and Goal-Related Prediction Errors in Medial Prefrontal Cortex
- [gdm-web-4b30a80174f2](<https://research.monash.edu/en/publications/f37759ff-20a8-4d70-be66-6da4b8cb8520>) - 2018-01-01 · `affiliated` · The Context-Dependent Additive Recurrent Neural Net
- [gdm-url-98d313aea8dd](<https://biblio.vub.ac.be/vubir/learning-to-coordinate-with-coordination-graphs-in-repeated-singlestage-multiagent-decision-problems(f4e3e291-6300-4c2a-a5b4-101d65ef04b3).html>) - 2018-01-01 · `affiliated` · Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems
- [gdm-doi-10.32470-ccn.2018.1125-0](<https://doi.org/10.32470/ccn.2018.1125-0>) - 2018-01-01 · `affiliated` · Corticostriatal signatures of learning efficient internal models for control
- [gdm-doi-10.2139-ssrn.3365416](<https://doi.org/10.2139/ssrn.3365416>) - 2018-01-01 · `affiliated` · Optimal Pricing in Markets with Non-Convex Costs
- [gdm-doi-10.18653-v1-w18-5446](<https://doi.org/10.18653/v1/w18-5446>) - 2018-01-01 · `affiliated` · GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
- [gdm-doi-10.18653-v1-p18-1132](<https://doi.org/10.18653/v1/p18-1132>) - 2018-01-01 · `affiliated` · LSTMs Can Learn Syntax-Sensitive Dependencies Well, But Modeling Structure Makes Them Better
- [gdm-doi-10.18653-v1-n18-1130](<https://doi.org/10.18653/v1/n18-1130>) - 2018-01-01 · `affiliated` · Using Morphological Knowledge in Open-Vocabulary Neural Language Models
- [gdm-doi-10.18653-v1-n18-1086](<https://doi.org/10.18653/v1/n18-1086>) - 2018-01-01 · `affiliated` · Neural Syntactic Generative Models with Exact Marginalization
- [gdm-doi-10.18653-v1-d18-1533](<https://doi.org/10.18653/v1/d18-1533>) - 2018-01-01 · `affiliated` · Recovering Missing Characters in Old Hawaiian Writing
- [gdm-doi-10.1016-b978-0-12-812098-9.00003-6](<https://doi.org/10.1016/b978-0-12-812098-9.00003-6>) - 2018-01-01 · `affiliated` · The Temporal Dynamics of Reward-Based Goal-Directed Decision-Making
- [gdm-arxiv-1808.00300](<https://doi.org/10.1007/978-3-030-01231-1_1>) - 2018-01-01 · `affiliated` · Learning Visual Question Answering by Bootstrapping Hard Attention
- [gdm-arxiv-1806.03863](<https://doi.org/10.1007/978-3-030-01225-0_40>) - 2018-01-01 · `affiliated` · Massively Parallel Video Networks
- [gdm-arxiv-1805.03151](<https://doi.org/10.1007/978-3-319-95582-7_7>) - 2018-01-01 · `affiliated` · A Weakness Measure for GR(1) Formulae
- [gdm-arxiv-1712.06651](<https://doi.org/10.1007/978-3-030-01246-5_27>) - 2018-01-01 · `affiliated` · Objects that Sound
- [gdm-arxiv-1506.02371](<https://doi.org/10.4310/sii.2018.v11.n3.a12>) - 2018-01-01 · `affiliated` · Interpretable selection and visualization of features and interactions using Bayesian forests

### 17.10 2017（73 条）

- [gdm-doi-10.7554-elife.22901](<https://doi.org/10.7554/elife.22901>) - 2017-12-04 · `affiliated` · Towards deep learning with segregated dendrites
- [gdm-arxiv-1707.02747](<https://papers.nips.cc/paper/7116-robust-imitation-of-diverse-behaviors.pdf>) - 2017-12-04 · `affiliated` · Robust imitation of diverse behaviors
- [gdm-arxiv-1711.08028](<http://arxiv.org/abs/1711.08028>) - 2017-11-21 · `affiliated` · Recurrent Relational Networks
- [gdm-arxiv-1711.05282](<http://arxiv.org/abs/1711.05282>) - 2017-11-14 · `affiliated` · C-WSL: Count-guided Weakly Supervised Localization
- [gdm-doi-10.1145-3131286](<https://doi.org/10.1145/3131286>) - 2017-10-24 · `affiliated` · Technical perspective: Solving imperfect information games
- [gdm-doi-10.1016-j.engappai.2017.08.020](<https://doi.org/10.1016/j.engappai.2017.08.020>) - 2017-10-15 · `affiliated` · Optimistic planning with an adaptive number of action switches for near-optimal nonlinear control
- [gdm-doi-10.1038-nn.4650](<https://doi.org/10.1038/nn.4650>) - 2017-10-02 · `affiliated` · The hippocampus as a predictive map
- [gdm-doi-10.1109-iccvw.2017.115](<https://doi.org/10.1109/iccvw.2017.115>) - 2017-10-01 · `affiliated` · Vision-as-Inverse-Graphics: Obtaining a Rich 3D Explanation of a Scene from a Single Image
- [gdm-doi-10.1109-iccv.2017.73](<https://doi.org/10.1109/iccv.2017.73>) - 2017-10-01 · `affiliated` · Look, Listen and Learn
- [gdm-doi-10.1109-iccv.2017.580](<https://doi.org/10.1109/iccv.2017.580>) - 2017-10-01 · `affiliated` · Realistic Dynamic Facial Textures from a Single Image Using GANs
- [gdm-doi-10.1109-iccv.2017.241](<https://doi.org/10.1109/iccv.2017.241>) - 2017-10-01 · `affiliated` · Raster-to-Vector: Revisiting Floorplan Transformation
- [gdm-doi-10.1109-iccv.2017.135](<https://doi.org/10.1109/iccv.2017.135>) - 2017-10-01 · `affiliated` · DeepContext: Context-Encoding Neural Pathways for 3D Holistic Scene Understanding
- [gdm-doi-10.1038-nature24270](<https://doi.org/10.1038/nature24270>) - 2017-10-01 · `affiliated` · Mastering the game of Go without human knowledge
- [gdm-arxiv-1708.07860](<https://doi.org/10.1109/iccv.2017.226>) - 2017-10-01 · `affiliated` · Multi-task Self-Supervised Visual Learning
- [gdm-doi-10.1371-journal.pcbi.1005768](<https://doi.org/10.1371/journal.pcbi.1005768>) - 2017-09-25 · `affiliated` · Predictive representations can link model-based reinforcement learning to model-free mechanisms
- [gdm-arxiv-1608.02117](<https://doi.org/10.1162/coli_a_00301>) - 2017-09-11 · `affiliated` · HyperLex: A Large-Scale Evaluation of Graded Lexical Entailment
- [gdm-doi-10.1101-183632](<https://doi.org/10.1101/183632>) - 2017-09-01 · `affiliated` · Clustering and compositionality of task representations in a neural network trained to perform many cognitive tasks
- [gdm-doi-10.1038-s41562-017-0180-8](<https://doi.org/10.1038/s41562-017-0180-8>) - 2017-08-25 · `affiliated` · The successor representation in human reinforcement learning
- [gdm-doi-10.1145-3109859.3109953](<https://doi.org/10.1145/3109859.3109953>) - 2017-08-24 · `affiliated` · DLRS 2017
- [gdm-arxiv-1708.06845](<http://arxiv.org/abs/1708.06845>) - 2017-08-22 · `affiliated` · Constructing Convex Inner Approximations of Steady-State Security Regions
- [gdm-doi-10.12688-f1000research.11637.2](<https://doi.org/10.12688/f1000research.11637.2>) - 2017-08-07 · `affiliated` · Service evaluation of the implementation of a digitally-enabled care pathway for the recognition and management of acute kidney injury
- [gdm-arxiv-1612.08810](<http://proceedings.mlr.press/v70/silver17a/silver17a.pdf>) - 2017-08-06 · `affiliated` · The predictron: end-to-end learning and planning
- [gdm-doi-10.1101-172387](<https://doi.org/10.1101/172387>) - 2017-08-04 · `affiliated` · Time without clocks: Human time perception based on perceptual classification
- [gdm-doi-10.1038-nn.4613](<https://doi.org/10.1038/nn.4613>) - 2017-07-31 · `affiliated` · Dorsal hippocampus contributes to model-based planning
- [gdm-doi-10.21437-semdial.2017-15](<https://doi.org/10.21437/semdial.2017-15>) - 2017-07-29 · `affiliated` · Online learning and transfer for user adaptation in dialogue systems
- [gdm-doi-10.24963-ijcai.2017-717](<https://doi.org/10.24963/ijcai.2017/717>) - 2017-07-28 · `affiliated` · Approximate Value Iteration with Temporally Extended Actions (Extended Abstract)
- [gdm-doi-10.24963-ijcai.2017-688](<https://doi.org/10.24963/ijcai.2017/688>) - 2017-07-28 · `affiliated` · On Thompson Sampling and Asymptotic Optimality
- [gdm-doi-10.24963-ijcai.2017-385](<https://doi.org/10.24963/ijcai.2017/385>) - 2017-07-28 · `affiliated` · End-to-end optimization of goal-driven and visually grounded dialogue systems
- [gdm-doi-10.1145-3129161.3129163](<https://doi.org/10.1145/3129161.3129163>) - 2017-07-28 · `affiliated` · Evolving brains in evolving environments
- [gdm-arxiv-1705.08417](<https://doi.org/10.24963/ijcai.2017/656>) - 2017-07-28 · `affiliated` · Reinforcement Learning with a Corrupted Reward Channel
- [gdm-arxiv-1707.08475](<http://arxiv.org/abs/1707.08475>) - 2017-07-26 · `affiliated` · DARLA: Improving Zero-Shot Transfer in Reinforcement Learning
- [gdm-arxiv-1707.04175](<http://arxiv.org/abs/1707.04175>) - 2017-07-13 · `affiliated` · Distral: Robust Multitask Reinforcement Learning
- [gdm-web-737d25aa6daf](<https://hal.science/hal-01576347>) - 2017-07-06 · `affiliated` · Faut-il minimiser le résidu de Bellman ou maximiser la valeur moyenne ?
- [gdm-arxiv-1707.00683](<http://arxiv.org/abs/1707.00683>) - 2017-07-02 · `affiliated` · Modulating early visual processing by language
- [gdm-doi-10.1109-cvpr.2017.269](<https://doi.org/10.1109/cvpr.2017.269>) - 2017-07-01 · `affiliated` · Synthesizing 3D Shapes via Modeling Multi-view Depth Maps and Silhouettes with Deep Generative Networks
- [gdm-doi-10.1016-s0262-4079-17-31370-2](<https://doi.org/10.1016/s0262-4079(17)31370-2>) - 2017-07-01 · `affiliated` · Reboot and reform
- [gdm-doi-10.1016-j.neuron.2017.06.011](<https://doi.org/10.1016/j.neuron.2017.06.011>) - 2017-07-01 · `affiliated` · Neuroscience-Inspired Artificial Intelligence
- [gdm-arxiv-1705.07750](<https://doi.org/10.1109/cvpr.2017.502>) - 2017-07-01 · `affiliated` · Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
- [gdm-arxiv-1611.08481](<https://doi.org/10.1109/cvpr.2017.475>) - 2017-07-01 · `affiliated` · GuessWhat?! Visual Object Discovery through Multi-modal Dialogue
- [gdm-arxiv-1706.08606](<http://arxiv.org/abs/1706.08606>) - 2017-06-26 · `affiliated` · Cognitive Psychology for Deep Neural Networks: A Shape Bias Case Study
- [gdm-doi-10.1016-j.tics.2017.05.012](<https://doi.org/10.1016/j.tics.2017.05.012>) - 2017-06-24 · `affiliated` · Mind Games: Game Engines as an Architecture for Intuitive Physics
- [gdm-doi-10.1109-ijcnn.2017.7965820](<https://doi.org/10.1109/ijcnn.2017.7965820>) - 2017-05-01 · `affiliated` · Plenary talks: Frontiers in recurrent neural network research
- [gdm-doi-10.1109-icra.2017.7989385](<https://doi.org/10.1109/icra.2017.7989385>) - 2017-05-01 · `affiliated` · Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates
- [gdm-doi-10.1007-s10590-017-9194-2](<https://doi.org/10.1007/s10590-017-9194-2>) - 2017-04-29 · `affiliated` · The representational geometry of word meanings acquired by neural machine translation models
- [gdm-doi-10.1038-544413a](<https://doi.org/10.1038/544413a>) - 2017-04-25 · `affiliated` · Artificial Intelligence: Chess match of the century
- [gdm-arxiv-1704.01279](<http://arxiv.org/abs/1704.01279>) - 2017-04-05 · `affiliated` · Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders
- [gdm-doi-10.1146-annurev-neuro-072116-031526](<https://doi.org/10.1146/annurev-neuro-072116-031526>) - 2017-04-04 · `affiliated` · Toward a Rational and Mechanistic Account of Mental Effort
- [gdm-doi-10.1145-3093337.3037725](<https://doi.org/10.1145/3093337.3037725>) - 2017-04-04 · `affiliated` · CHERI JNI
- [gdm-arxiv-1703.08520](<http://arxiv.org/abs/1703.08520>) - 2017-03-24 · `affiliated` · Augmented Ensemble MCMC sampling in Factorial Hidden Markov Models
- [gdm-arxiv-1703.05449](<http://arxiv.org/abs/1703.05449>) - 2017-03-16 · `affiliated` · Minimax Regret Bounds for Reinforcement Learning
- [gdm-arxiv-1703.05423](<http://arxiv.org/abs/1703.05423>) - 2017-03-15 · `affiliated` · End-to-end optimization of goal-driven and visually grounded dialogue systems Harm de Vries
- [gdm-arxiv-1703.04933](<http://arxiv.org/abs/1703.04933>) - 2017-03-15 · `affiliated` · Sharp Minima Can Generalize For Deep Nets
- [gdm-arxiv-1703.04813](<http://arxiv.org/abs/1703.04813>) - 2017-03-14 · `affiliated` · Learned Optimizers that Scale and Generalize
- [gdm-arxiv-1612.00796](<https://doi.org/10.1073/pnas.1611835114>) - 2017-03-14 · `affiliated` · Overcoming catastrophic forgetting in neural networks
- [gdm-arxiv-1703.01988](<http://arxiv.org/abs/1703.01988>) - 2017-03-06 · `affiliated` · Neural Episodic Control
- [gdm-arxiv-1703.00956](<http://arxiv.org/abs/1703.00956>) - 2017-03-02 · `affiliated` · A Laplacian Framework for Option Discovery in Reinforcement Learning
- [gdm-arxiv-1702.03037](<http://arxiv.org/abs/1702.03037>) - 2017-02-10 · `affiliated` · Multi-agent Reinforcement Learning in Sequential Social Dilemmas
- [gdm-doi-10.1371-journal.pcbi.1005333](<https://doi.org/10.1371/journal.pcbi.1005333>) - 2017-02-03 · `affiliated` · Insect Bio-inspired Neural Network Provides New Evidence on How Simple Feature Detectors Can Enable Complex Visual Generalization and Stimulus Location Invariance in the Miniature Brain of Honeybees
- [gdm-doi-10.1162-neco_a_00929](<https://doi.org/10.1162/neco_a_00929>) - 2017-01-17 · `affiliated` · Deep Learning with Dynamic Spiking Neurons and Fixed Feedback Weights
- [gdm-doi-10.4467-20838476si.16.004.6185](<https://doi.org/10.4467/20838476si.16.004.6185>) - 2017-01-01 · `affiliated` · On Loss Functions for Deep Neural Networks in Classification
- [gdm-doi-10.3389-frym.2017.00026](<https://doi.org/10.3389/frym.2017.00026>) - 2017-01-01 · `affiliated` · Precommitment: A Way around Temptation
- [gdm-doi-10.18653-v1-w17-3412](<https://doi.org/10.18653/v1/w17-3412>) - 2017-01-01 · `affiliated` · Introducing Structure into Neural Network-Based Semantic Models
- [gdm-doi-10.18653-v1-s17-2157](<https://doi.org/10.18653/v1/s17-2157>) - 2017-01-01 · `affiliated` · Oxford at SemEval-2017 Task 9: Neural AMR Parsing with Pointer-Augmented Attention
- [gdm-doi-10.18653-v1-p17-1137](<https://doi.org/10.18653/v1/p17-1137>) - 2017-01-01 · `affiliated` · Learning to Create and Reuse Words in Open-Vocabulary Neural Language Modeling
- [gdm-doi-10.18653-v1-e17-1117](<https://doi.org/10.18653/v1/e17-1117>) - 2017-01-01 · `affiliated` · What Do Recurrent Neural Network Grammars Learn About Syntax?
- [gdm-doi-10.18653-v1-d17-1197](<https://doi.org/10.18653/v1/d17-1197>) - 2017-01-01 · `affiliated` · Reference-Aware Language Models
- [gdm-doi-10.1109-mprv.2017.2940968](<https://doi.org/10.1109/mprv.2017.2940968>) - 2017-01-01 · `affiliated` · Squeezing Deep Learning into Mobile and Embedded Devices
- [gdm-doi-10.1017-s0140525x17000218](<https://doi.org/10.1017/s0140525x17000218>) - 2017-01-01 · `affiliated` · Understand the cogs to understand cognition
- [gdm-doi-10.1017-s0140525x17000048](<https://doi.org/10.1017/s0140525x17000048>) - 2017-01-01 · `affiliated` · Building machines that learn and think for themselves
- [gdm-doi-10.1007-978-3-662-54345-0_3](<https://doi.org/10.1007/978-3-662-54345-0_3>) - 2017-01-01 · `affiliated` · Invited Talk: U-Net Convolutional Networks for Biomedical Image Segmentation
- [gdm-arxiv-1705.02925](<https://doi.org/10.18653/v1/p17-1191>) - 2017-01-01 · `affiliated` · Ontology-Aware Token Embeddings for Prepositional Phrase Attachment
- [gdm-arxiv-1704.07092](<https://doi.org/10.18653/v1/p17-1112>) - 2017-01-01 · `affiliated` · Robust Incremental Neural Semantic Graph Parsing
- [gdm-arxiv-1704.06970](<https://doi.org/10.18653/v1/p17-2058>) - 2017-01-01 · `affiliated` · Differentiable Scheduled Sampling for Credit Assignment

### 17.11 2016（77 条）

- [gdm-doi-10.7551-mitpress-10761.003.0005](<https://doi.org/10.7551/mitpress/10761.003.0005>) - 2016-12-23 · `affiliated` · Herding as a Learning System with Edge-of-Chaos Dynamics
- [gdm-doi-10.1101-096339](<https://doi.org/10.1101/096339>) - 2016-12-23 · `affiliated` · Identifying Model-Based and Model-Free Patterns in Behavior on Multi-Step Tasks
- [gdm-arxiv-1612.01744](<http://arxiv.org/abs/1612.01744>) - 2016-12-06 · `affiliated` · Listen and Translate: A Proof of Concept for End-to-End Speech-to-Text  Translation
- [gdm-doi-10.1038-nn.4444](<https://doi.org/10.1038/nn.4444>) - 2016-12-05 · `affiliated` · A probabilistic approach to demixing odors
- [gdm-arxiv-1606.01868](<https://proceedings.neurips.cc/paper/2016/file/afda332245e2af431fb7b672a68b659d-Paper.pdf>) - 2016-12-05 · `affiliated` · Unifying count-based exploration and intrinsic motivation
- [gdm-arxiv-1604.06057](<http://hdl.handle.net/1721.1/112755>) - 2016-12-05 · `affiliated` · Hierarchical deep reinforcement learning: integrating temporal abstraction and intrinsic motivation
- [gdm-arxiv-1603.08575](<https://openalex.org/W2963951231>) - 2016-12-05 · `affiliated` · Attend, infer, repeat: fast scene understanding with generative models
- [gdm-doi-10.1016-j.neuron.2016.10.052](<https://doi.org/10.1016/j.neuron.2016.10.052>) - 2016-12-01 · `affiliated` · Computations Underlying Social Hierarchy Learning: Distinct Neural Mechanisms for Updating and Representing Self-Relevant Information
- [gdm-doi-10.1098-rstb.2016.0049](<https://doi.org/10.1098/rstb.2016.0049>) - 2016-11-21 · `affiliated` · Complementary learning systems within the hippocampus: a neural network modelling approach to reconciling episodic memory with statistical learning
- [gdm-doi-10.1038-ncomms13276](<https://doi.org/10.1038/ncomms13276>) - 2016-11-08 · `affiliated` · Random synaptic feedback weights support error backpropagation for deep learning
- [gdm-arxiv-1611.01423](<http://arxiv.org/abs/1611.01423>) - 2016-11-04 · `affiliated` · Learning Continuous Semantic Representations of Symbolic Expressions
- [gdm-arxiv-1610.09027](<http://arxiv.org/abs/1610.09027>) - 2016-10-27 · `affiliated` · Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes
- [gdm-doi-10.1038-nature20101](<https://doi.org/10.1038/nature20101>) - 2016-10-12 · `affiliated` · Hybrid computing using a neural network with dynamic external memory
- [gdm-doi-10.1109-iros.2016.7759546](<https://doi.org/10.1109/iros.2016.7759546>) - 2016-10-01 · `affiliated` · Navigation Among Movable Obstacles with learned dynamic constraints
- [gdm-doi-10.1038-nn.4384](<https://doi.org/10.1038/nn.4384>) - 2016-09-27 · `affiliated` · Dorsal anterior cingulate cortex and the value of control
- [gdm-doi-10.3389-fncom.2016.00094](<https://doi.org/10.3389/fncom.2016.00094>) - 2016-09-13 · `affiliated` · Toward an Integration of Deep Learning and Neuroscience
- [gdm-doi-10.1016-j.media.2016.08.008](<https://doi.org/10.1016/j.media.2016.08.008>) - 2016-09-10 · `affiliated` · Gland segmentation in colon histology images: The glas challenge contest
- [gdm-doi-10.1016-j.cognition.2016.08.012](<https://doi.org/10.1016/j.cognition.2016.08.012>) - 2016-09-03 · `affiliated` · Inferring mass in complex scenes by mental simulation
- [gdm-doi-10.1147-jrd.2016.2586238](<https://doi.org/10.1147/jrd.2016.2586238>) - 2016-09-01 · `affiliated` · Mixed-effects models and tolerance limits for mycotoxin measurements in food-stock lots
- [gdm-doi-10.1109-cig.2016.7860430](<https://doi.org/10.1109/cig.2016.7860430>) - 2016-09-01 · `affiliated` · Analyzing the robustness of general video game playing agents
- [gdm-doi-10.12688-f1000research.9525.1](<https://doi.org/10.12688/f1000research.9525.1>) - 2016-08-30 · `affiliated` · Applying machine learning to automated segmentation of head and neck tumour volumes and organs at risk on radiotherapy planning CT and MRI scans
- [gdm-doi-10.1073-pnas.1610686113](<https://doi.org/10.1073/pnas.1610686113>) - 2016-08-22 · `affiliated` · Semantic representations in the temporal pole predict false memories
- [gdm-arxiv-1608.05343](<http://arxiv.org/abs/1608.05343>) - 2016-08-18 · `affiliated` · Decoupled Neural Interfaces using Synthetic Gradients
- [gdm-doi-10.1038-srep31330](<https://doi.org/10.1038/srep31330>) - 2016-08-11 · `affiliated` · Retrieval-Based Model Accounts for Striking Profile of Episodic Memory and Generalization
- [gdm-doi-10.1145-2939672.2945358](<https://doi.org/10.1145/2939672.2945358>) - 2016-08-08 · `affiliated` · Learning to Learn and Compositionality with Deep Recurrent Neural Networks
- [gdm-doi-10.1101-066282](<https://doi.org/10.1101/066282>) - 2016-07-27 · `affiliated` · fMRI Dependent Components Analysis Reveals Dynamic Relations Between Functional Large Scale Cortical Networks
- [gdm-arxiv-1509.02971](<https://arxiv.org/pdf/1509.02971.pdf>) - 2016-07-22 · `affiliated` · Continuous control with deep reinforcement learning
- [gdm-doi-10.1145-2908812.2908890](<https://doi.org/10.1145/2908812.2908890>) - 2016-07-20 · `affiliated` · Convolution by Evolution
- [gdm-doi-10.1016-j.neuroimage.2016.06.034](<https://doi.org/10.1016/j.neuroimage.2016.06.034>) - 2016-07-16 · `affiliated` · The Neuro Bureau ADHD-200 Preprocessed repository
- [gdm-doi-10.12688-f1000research.8996.1](<https://doi.org/10.12688/f1000research.8996.1>) - 2016-07-05 · `affiliated` · Automated analysis of retinal imaging using machine learning techniques for computer vision
- [gdm-doi-10.1109-acc.2016.7524916](<https://doi.org/10.1109/acc.2016.7524916>) - 2016-07-01 · `affiliated` · Discounted near-optimal control of general continuous-action nonlinear systems using optimistic planning
- [gdm-arxiv-1607.00215](<http://arxiv.org/abs/1607.00215>) - 2016-07-01 · `affiliated` · Why is Posterior Sampling Better than Optimism for Reinforcement  Learning?
- [gdm-arxiv-1606.08718](<http://arxiv.org/abs/1606.08718>) - 2016-06-28 · `affiliated` · Learning Nash Equilibrium for General-Sum Markov Games from Batch Data
- [gdm-arxiv-1606.07636](<http://arxiv.org/abs/1606.07636>) - 2016-06-24 · `affiliated` · Is the Bellman residual a bad proxy?
- [gdm-web-27791719d48f](<http://proceedings.mlr.press/v48/santoro16.pdf>) - 2016-06-19 · `affiliated` · Meta-learning with memory-augmented neural networks
- [gdm-arxiv-1602.06725](<http://proceedings.mlr.press/v48/mnihb16.pdf>) - 2016-06-19 · `affiliated` · Variational inference for Monte Carlo objectives
- [gdm-arxiv-1606.05328](<http://export.arxiv.org/pdf/1606.05328>) - 2016-06-16 · `affiliated` · Conditional Image Generation with PixelCNN Decoders
- [gdm-doi-10.1016-j.tics.2016.05.004](<https://doi.org/10.1016/j.tics.2016.05.004>) - 2016-06-14 · `affiliated` · What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated
- [gdm-arxiv-1606.04080](<http://arxiv.org/abs/1606.04080>) - 2016-06-13 · `affiliated` · Matching Networks for One Shot Learning
- [gdm-arxiv-1606.02580](<http://arxiv.org/abs/1606.02580>) - 2016-06-08 · `affiliated` · Convolution by Evolution: Differentiable Pattern Producing Networks
- [gdm-doi-10.1101-057216](<https://doi.org/10.1101/057216>) - 2016-06-07 · `affiliated` · Decoding of generic mental representations from functional MRI data using word embeddings
- [gdm-arxiv-1606.01128](<http://arxiv.org/abs/1606.01128>) - 2016-06-03 · `affiliated` · Difference of Convex Functions Programming Applied to Control with Expert Data
- [gdm-doi-10.1109-tpami.2016.2574713](<https://doi.org/10.1109/tpami.2016.2574713>) - 2016-06-02 · `affiliated` · Learning Category-Specific Deformable 3D Models for Object Reconstruction
- [gdm-arxiv-1507.06550](<https://doi.org/10.1109/cvpr.2016.512>) - 2016-06-01 · `affiliated` · Human Pose Estimation with Iterative Error Feedback
- [gdm-doi-10.1080-02643294.2016.1176907](<https://doi.org/10.1080/02643294.2016.1176907>) - 2016-05-18 · `affiliated` · A comparative evaluation of off-the-shelf distributed semantic representations for modelling behavioural data
- [gdm-arxiv-1605.02226](<http://arxiv.org/abs/1605.02226>) - 2016-05-07 · `affiliated` · Neural Autoregressive Distribution Estimation
- [gdm-doi-10.1101-051870](<https://doi.org/10.1101/051870>) - 2016-05-06 · `affiliated` · Complementary learning systems within the hippocampus: A neural network modeling approach to reconciling episodic memory with statistical learning
- [gdm-doi-10.1109-tnnls.2016.2543000](<https://doi.org/10.1109/tnnls.2016.2543000>) - 2016-05-04 · `affiliated` · Bridging the Gap Between Imitation Learning and Inverse Reinforcement Learning
- [gdm-doi-10.1162-coli_a_00249](<https://doi.org/10.1162/coli_a_00249>) - 2016-04-27 · `affiliated` · Mining Parallel Corpora from Sina Weibo and Twitter
- [gdm-doi-10.2196-jmir.4854](<https://doi.org/10.2196/jmir.4854>) - 2016-04-06 · `affiliated` · Interprofessional Communication of Clinicians Using a Mobile Phone App: A Randomized Crossover Trial Using Simulated Patients
- [gdm-arxiv-1603.05106](<http://arxiv.org/abs/1603.05106>) - 2016-03-16 · `affiliated` · One-Shot Generalization in Deep Generative Models
- [gdm-arxiv-1603.00748](<http://arxiv.org/abs/1603.00748>) - 2016-03-02 · `affiliated` · Continuous Deep Q-Learning with Model-based Acceleration
- [gdm-arxiv-1509.06461](<https://ojs.aaai.org/index.php/AAAI/article/view/10295>) - 2016-03-02 · `affiliated` · Deep Reinforcement Learning with Double Q-Learning
- [gdm-arxiv-1602.07905](<http://arxiv.org/abs/1602.07905>) - 2016-02-25 · `affiliated` · Thompson Sampling is Asymptotically Optimal in General Environments
- [gdm-arxiv-1602.07714](<http://arxiv.org/abs/1602.07714>) - 2016-02-24 · `affiliated` · Learning values across many orders of magnitude
- [gdm-doi-10.1609-aaai.v30i1.10227](<https://doi.org/10.1609/aaai.v30i1.10227>) - 2016-02-21 · `affiliated` · Generalized Emphatic Temporal Difference Learning: Bias-Variance Analysis
- [gdm-arxiv-1512.04860](<https://ojs.aaai.org/index.php/AAAI/article/view/10303>) - 2016-02-21 · `affiliated` · Increasing the Action Gap: New Operators for Reinforcement Learning
- [gdm-arxiv-1602.04621](<http://arxiv.org/abs/1602.04621>) - 2016-02-15 · `affiliated` · Deep Exploration via Bootstrapped DQN
- [gdm-arxiv-1602.03032](<http://arxiv.org/abs/1602.03032>) - 2016-02-09 · `affiliated` · Associative Long Short-Term Memory
- [gdm-arxiv-1602.02660](<http://arxiv.org/abs/1602.02660>) - 2016-02-08 · `affiliated` · Exploiting Cyclic Symmetry in Convolutional Neural Networks
- [gdm-doi-10.1152-jn.00971.2015](<https://doi.org/10.1152/jn.00971.2015>) - 2016-02-03 · `affiliated` · Primary motor cortex neurons classified in a postural task predict muscle activation patterns in a reaching task
- [gdm-doi-10.1038-nature16961](<https://doi.org/10.1038/nature16961>) - 2016-01-26 · `affiliated` · Mastering the game of Go with deep neural networks and tree search
- [gdm-arxiv-1601.06759](<http://export.arxiv.org/pdf/1601.06759>) - 2016-01-25 · `affiliated` · Pixel Recurrent Neural Networks
- [gdm-arxiv-1507.01526](<https://arxiv.org/pdf/1507.01526>) - 2016-01-07 · `affiliated` · Grid Long Short-Term Memory
- [gdm-web-083a74d237e2](<https://hal.science/hal-01531770>) - 2016-01-01 · `affiliated` · Principles of Systems Biology, No. 11
- [gdm-url-e6281f02f288](<http://hdl.handle.net/20.500.12210/24626>) - 2016-01-01 · `affiliated` · Analysis of Classification-based Policy Iteration Algorithms
- [gdm-url-87e339ff57ac](<https://hal.science/hal-01629651>) - 2016-01-01 · `affiliated` · Batch Policy Iteration Algorithms for Continuous Domains
- [gdm-doi-10.18653-v1-p16-1057](<https://doi.org/10.18653/v1/p16-1057>) - 2016-01-01 · `affiliated` · Latent Predictor Networks for Code Generation
- [gdm-doi-10.18653-v1-d16-1138](<https://doi.org/10.18653/v1/d16-1138>) - 2016-01-01 · `affiliated` · Online Segment to Segment Neural Transduction
- [gdm-doi-10.1109-tsg.2015.2513900](<https://doi.org/10.1109/tsg.2015.2513900>) - 2016-01-01 · `affiliated` · A Sparse Coding Approach to Household Electricity Demand Forecasting in Smart Grids
- [gdm-doi-10.1007-978-3-319-46976-8](<https://doi.org/10.1007/978-3-319-46976-8>) - 2016-01-01 · `affiliated` · Deep Learning and Data Labeling for Medical Applications
- [gdm-arxiv-1609.07561](<https://doi.org/10.18653/v1/d16-1180>) - 2016-01-01 · `affiliated` · Distilling an Ensemble of Greedy Dependency Parsers into One MST Parser
- [gdm-arxiv-1606.06650](<https://doi.org/10.1007/978-3-319-46723-8_49>) - 2016-01-01 · `affiliated` · 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
- [gdm-arxiv-1606.00499](<https://doi.org/10.18653/v1/d16-1124>) - 2016-01-01 · `affiliated` · Generalizing and Hybridizing Count-based and Neural Language Models
- [gdm-arxiv-1602.04951](<https://doi.org/10.1007/978-3-319-46379-7_21>) - 2016-01-01 · `affiliated` · Q&#40;&#36;&#36;&#92;&#92;lambda &#36;&#36;&#41; with Off-Policy Corrections
- [gdm-arxiv-1511.06279](<https://arxiv.org/pdf/1511.06279>) - 2016-01-01 · `affiliated` · Neural Programmer-Interpreters
- [gdm-arxiv-1511.04581](<https://lirias.kuleuven.be/handle/123456789/531614>) - 2016-01-01 · `affiliated` · A Test of Relative Similarity For Model Selection in Generative Models

### 17.12 2015（35 条）

- [gdm-web-5f3f7db138c9](<https://openalex.org/W2963430173>) - 2015-12-07 · `affiliated` · Embed to control: a locally Linear Latent dynamics model for control from raw images
- [gdm-arxiv-1510.09142](<https://arxiv.org/pdf/1510.09142>) - 2015-12-07 · `affiliated` · Learning continuous control policies by stochastic value gradients
- [gdm-arxiv-1509.08731](<https://papers.nips.cc/paper/5668-variational-information-maximisation-for-intrinsically-motivated-reinforcement-learning.pdf>) - 2015-12-07 · `affiliated` · Variational information maximisation for intrinsically motivated reinforcement learning
- [gdm-arxiv-1507.00210](<https://arxiv.org/pdf/1507.00210>) - 2015-12-07 · `affiliated` · Natural Neural Networks
- [gdm-doi-10.1109-iccv.2015.173](<https://doi.org/10.1109/iccv.2015.173>) - 2015-12-01 · `affiliated` · Deep Fried Convnets
- [gdm-arxiv-1511.06581](<http://arxiv.org/abs/1511.06581>) - 2015-11-20 · `affiliated` · Dueling Network Architectures for Deep Reinforcement Learning
- [gdm-arxiv-1511.06295](<http://arxiv.org/abs/1511.06295>) - 2015-11-19 · `affiliated` · Policy Distillation
- [gdm-arxiv-1511.05952](<http://arxiv.org/abs/1511.05952>) - 2015-11-18 · `affiliated` · Prioritized Experience Replay
- [gdm-arxiv-1511.05176](<http://arxiv.org/abs/1511.05176>) - 2015-11-16 · `affiliated` · MuProp: Unbiased Backpropagation for Stochastic Neural Networks
- [gdm-arxiv-1509.06664](<http://arxiv.org/abs/1509.06664>) - 2015-09-22 · `affiliated` · Reasoning about Entailment with Neural Attention
- [gdm-doi-10.1073-pnas.1505483112](<https://doi.org/10.1073/pnas.1505483112>) - 2015-08-31 · `affiliated` · Evidence integration in model-based tree search
- [gdm-doi-10.1016-j.cobeha.2015.08.009](<https://doi.org/10.1016/j.cobeha.2015.08.009>) - 2015-08-28 · `affiliated` · Reinforcement learning, efficient coding, and the statistics of natural tasks
- [gdm-arxiv-1302.2788](<https://doi.org/10.1016/j.jcss.2015.06.011>) - 2015-08-07 · `affiliated` · The complexity of minimum-length path decompositions
- [gdm-web-741865153e39](<http://discovery.ucl.ac.uk/1523606/>) - 2015-07-25 · `affiliated` · Smooth UCT search in computer poker
- [gdm-web-4804ca49de4d](<https://www.ijcai.org/Proceedings/15/Papers/470.pdf>) - 2015-07-25 · `affiliated` · Count-based frequency estimation with bounded memory
- [gdm-web-d64c7a929ce6](<http://proceedings.mlr.press/v37/schaul15.pdf>) - 2015-07-06 · `affiliated` · Universal Value Function Approximators
- [gdm-web-0cd574579d54](<http://proceedings.mlr.press/v37/blundell15.pdf>) - 2015-07-06 · `affiliated` · Weight Uncertainty in Neural Network
- [gdm-web-042cb82c5afe](<https://openalex.org/W2294739201>) - 2015-07-06 · `affiliated` · Cheap Bandits
- [gdm-arxiv-1502.04623](<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.697.4023>) - 2015-07-06 · `affiliated` · DRAW: A Recurrent Neural Network For Image Generation
- [gdm-arxiv-1506.05254](<http://arxiv.org/abs/1506.05254>) - 2015-06-17 · `affiliated` · Gradient Estimation Using Stochastic Computation Graphs
- [gdm-arxiv-1506.03340](<https://openalex.org/W2949615363>) - 2015-06-10 · `affiliated` · Teaching Machines to Read and Comprehend
- [gdm-arxiv-1506.02516](<http://arxiv.org/abs/1506.02516>) - 2015-06-08 · `affiliated` · Learning to Transduce with Unbounded Memory
- [gdm-arxiv-1506.02025](<http://arxiv.org/abs/1506.02025>) - 2015-06-05 · `affiliated` · Spatial Transformer Networks
- [gdm-arxiv-1505.05770](<http://arxiv.org/abs/1505.05770>) - 2015-05-21 · `affiliated` · Variational Inference with Normalizing Flows
- [gdm-doi-10.1371-journal.pone.0123108](<https://doi.org/10.1371/journal.pone.0123108>) - 2015-04-02 · `affiliated` · Investigating the Use of Support Vector Machine Classification on Structural Brain Images of Preterm–Born Teenagers as a Biological Marker
- [gdm-doi-10.1007-s10032-015-0242-2](<https://doi.org/10.1007/s10032-015-0242-2>) - 2015-03-11 · `affiliated` · Automatic diacritization of Arabic text using recurrent neural networks
- [gdm-arxiv-1503.02551](<http://arxiv.org/abs/1503.02551>) - 2015-03-09 · `affiliated` · Kernel-Based Just-In-Time Learning for Passing Expectation Propagation Messages
- [gdm-doi-10.1038-nature14236](<https://doi.org/10.1038/nature14236>) - 2015-02-24 · `affiliated` · Human-level control through deep reinforcement learning
- [gdm-doi-10.1162-coli_a_00209](<https://doi.org/10.1162/coli_a_00209>) - 2015-02-23 · `affiliated` · Concrete Models and Empirical Evaluations for the Categorical Compositional Distributional Model of Meaning
- [gdm-arxiv-1411.5326](<https://ojs.aaai.org/index.php/AAAI/article/view/9600>) - 2015-02-21 · `affiliated` · Compress and Control
- [gdm-arxiv-1502.03509](<http://arxiv.org/abs/1502.03509>) - 2015-02-12 · `affiliated` · MADE: Masked Autoencoder for Distribution Estimation
- [gdm-doi-10.1109-tciaig.2015.2402393](<https://doi.org/10.1109/tciaig.2015.2402393>) - 2015-02-10 · `affiliated` · The 2014 General Video Game Playing Competition
- [gdm-web-bbdfa45d0825](<https://openalex.org/W2182034283>) - 2015-01-01 · `affiliated` · Adaptive strategy for stratified Monte Carlo sampling
- [gdm-web-9b7053670cd5](<http://hutter1.net/publ/ratagentx.pdf>) - 2015-01-01 · `affiliated` · Rationality, optimism and guarantees in general reinforcement learning
- [gdm-doi-10.3115-v1-p15-2142](<https://doi.org/10.3115/v1/p15-2142>) - 2015-01-01 · `affiliated` · Generative Incremental Dependency Parsing with Neural Networks

### 17.13 2014（17 条）

- [gdm-arxiv-1412.7755](<http://arxiv.org/abs/1412.7755>) - 2014-12-24 · `affiliated` · Multiple Object Recognition with Visual Attention
- [gdm-arxiv-1412.6564](<http://arxiv.org/abs/1412.6564>) - 2014-12-20 · `affiliated` · Move Evaluation in Go Using Deep Convolutional Neural Networks
- [gdm-arxiv-1412.5903](<http://arxiv.org/abs/1412.5903>) - 2014-12-18 · `affiliated` · Deep Structured Output Learning for Unconstrained Text Recognition
- [gdm-web-5d4db9a9a511](<http://discovery.ucl.ac.uk/1496709/>) - 2014-12-08 · `affiliated` · Bayes-Adaptive Simulation-based Search with Value Function Approximation
- [gdm-doi-10.1136-archdischild-2014-307384.1069](<https://doi.org/10.1136/archdischild-2014-307384.1069>) - 2014-10-01 · `affiliated` · PO-0427 Investigating The Use Of Support Vector Machine Classification On Structural Brain Images Of Preterm–born Teenagers As A Biological Marker
- [gdm-doi-10.1109-tciaig.2014.2352795](<https://doi.org/10.1109/tciaig.2014.2352795>) - 2014-08-27 · `affiliated` · An Extensible Description Language for Video Games
- [gdm-arxiv-1406.6247](<http://export.arxiv.org/pdf/1406.6247>) - 2014-06-24 · `affiliated` · Recurrent Models of Visual Attention
- [gdm-web-f2d22ebb2d01](<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.664.3860>) - 2014-06-21 · `affiliated` · Towards End-To-End Speech Recognition with Recurrent Neural Networks
- [gdm-web-14eb58f75946](<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.668.2580>) - 2014-06-21 · `affiliated` · Skip Context Tree Switching
- [gdm-arxiv-1406.5298](<https://handle.uba.uva.nl/personal/pure/en/publications/semisupervised-learning-with-deep-generative-models(cd6a2c89-1b67-4e25-b4d6-dd06075a6730).html>) - 2014-06-20 · `affiliated` · Semi-Supervised Learning with Deep Generative Models
- [gdm-arxiv-1406.3070](<http://arxiv.org/abs/1406.3070>) - 2014-06-11 · `affiliated` · Distributed Parameter Estimation in Probabilistic Graphical Models
- [gdm-arxiv-1403.6863](<http://arxiv.org/abs/1403.6863>) - 2014-03-26 · `affiliated` · Online Learning of k-CNF Boolean Functions
- [gdm-arxiv-1402.0030](<http://arxiv.org/abs/1402.0030>) - 2014-01-31 · `affiliated` · Neural Variational Inference and Learning in Belief Networks
- [gdm-arxiv-1401.4082](<http://arxiv.org/abs/1401.4082>) - 2014-01-16 · `affiliated` · Stochastic Backpropagation and Approximate Inference in Deep Generative Models
- [gdm-web-4e1f4e892e4e](<https://inria.hal.science/hal-00938992>) - 2014-01-01 · `affiliated` · Deterministic policy gradient algorithms
- [gdm-doi-10.1007-978-3-319-09165-5_6](<https://doi.org/10.1007/978-3-319-09165-5_6>) - 2014-01-01 · `affiliated` · MoHex 2.0: A Pattern-Based MCTS Hex Player
- [gdm-doi-10.1007-978-3-319-09165-5_4](<https://doi.org/10.1007/978-3-319-09165-5_4>) - 2014-01-01 · `affiliated` · Investigating the Limits of Monte-Carlo Tree Search Methods in Computer Go

### 17.14 2013（6 条）

- [gdm-arxiv-1312.6055](<http://arxiv.org/abs/1312.6055>) - 2013-12-20 · `affiliated` · Unit Tests for Stochastic Optimization
- [gdm-web-4d7d7e1d085c](<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.637.986>) - 2013-12-05 · `affiliated` · Bayesian Hierarchical Community Discovery
- [gdm-web-41c3025df4aa](<http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf>) - 2013-12-05 · `affiliated` · Learning word embeddings efficiently with noise-contrastive estimation
- [gdm-arxiv-1310.8499](<http://arxiv.org/abs/1310.8499>) - 2013-10-31 · `affiliated` · Deep AutoRegressive Networks
- [gdm-doi-10.5281-zenodo.14433924](<https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/73666499d>) - 2013-08-10 · `affiliated` · Parallelized Deep Neural Networks for Distributed Intelligent Systems
- [gdm-arxiv-1109.5951](<https://doi.org/10.1007/978-3-642-44958-1_18>) - 2013-01-01 · `affiliated` · An Approximation of the Universal Intelligence Measure

### 17.15 未标日期（2 条）

- [gdm-url-93a933910a46](<https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf>) - unknown-date · `affiliated` · Veo: a Text-to-Video Generation System
- [gdm-url-49ab4305f7af](<https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf#page=105>) - unknown-date · `core` · Gemini 1.5 Model Card


## 18. 完整性声明与更新规则

本章“完整”的可审计对象是 2026-07-19 冻结的 `gemini.jsonl`：

- **inventory identity**：1,985 条，ID 唯一；1,985 条均有 `primary_url`，共有 1,984 个不同 URL；
- **tier/type**：core 24、direct 18、affiliated 1,943；research paper 1,926、model card 41、technical report 17、
  dataset 1；
- **artifact**：1,625 个公开 PDF 路由；1,543 份本地 PDF/TXT；其余记录仍以官方/出版社/元数据入口保留；
- **appendix contract**：第 17 节的链接文字集合必须与 1,985 个 ID 完全相等，且每个 href 必须等于对应
  `primary_url`；同 URL 双记录不去重；
- **interpretation contract**：core/direct/affiliated 不互相升级；[D]/[C]/[R]/[I]/[U] 不互相替代；
- **temporal contract**：新版本、撤稿、更正、URL 迁移、card 更新、PDF revision 与新 affiliation evidence
  必须进入 inventory/discovery log 后再改正文计数。

它仍不能证明私有、未发表、已静默删除、从未被索引或未写明 affiliation 的工作不存在。[U] 因此最强诚实表述是：
**对定义过、快照化、逐条可回溯的公共来源宇宙做到全覆盖；对不可观察宇宙保持开放缺口。**

更新时先运行 corpus validate/audit/download/extract/report，再重新生成附录，最后执行 ID/URL exact-set、数学、
链接和 `mkdocs build --strict`。任何计数变化都应同时更新本章开头、主题画像、卡片审计、coverage ledger 与
本完整性声明。
