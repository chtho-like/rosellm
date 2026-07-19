# Anthropic：从可扩展对齐实验室到可审计的前沿模型治理

**核验截止：2026-07-19（Asia/Shanghai）。** 本章以仓库冻结的
[Anthropic 文献库存](https://github.com/chtho-like/rosellm/blob/main/research/literature/inventory/anthropic.jsonl)为来源宇宙：
共 **244 条记录**，其中 core 21、direct 188、affiliated 35；按记录类型计，
system card 16、model card 1、research paper 69、technical report 42、
blog with report 105、other 11。截止核验时，本地归档已经覆盖其中 125 条全文。
这里的“全谱系”只表示对这 244 条逐条有交代，不表示互联网再无漏项；同一项目的论文、
技术报告、评论和政策备忘录若在冻结库存中各有记录，附录也分别保留。

## 1. 阅读规则、证据标签与先行结论

本章沿用 [Research, Evidence, and Citation Standard](../research-method.md)：

- **[D] Disclosed / 已披露**：论文、技术报告、系统卡、模型卡或一手研究页直接陈述。
- **[C] Confirmed artifact / 公开制品确认**：公开 PDF、归档页、代码、数据或库存字段可直接观察。
- **[R] Reproduced / 已复现**：按可审计协议独立复跑；本章没有把任何结果标成 [R]。
- **[I] Inferred / 推断**：由列明证据和假设综合推出，来源本身没有这样宣称。
- **[U] Unknown / 未知**：证据缺失、含混、冲突，或不足以支持结论。

库存层级也必须分开：core 是 Claude 模型卡、系统卡和风险报告；direct 是 Anthropic
直接产出的研究、评测、政策和社会影响工作；affiliated 只说明至少一位作者有机构署名，
**不能**据此推出该方法进入 Claude、Anthropic 采用其结论，或该记录代表机构立场。

六个先行结论：

1. **[D] Anthropic 的研究主线早于 Claude 产品线，也远宽于“训练一个更安全的聊天模型”。**
   2021–2022 年的 HHH assistant、可预测 scaling、RLHF、红队、scalable oversight、
   Constitutional AI（CAI）和 model-written evaluations，先把“训练目标—监督来源—
   评测—治理”连成实验闭环；2023–2026 年的 Claude 系统卡、Responsible Scaling
   Policy（RSP）与风险报告才把它逐步产品化、制度化。

2. **[D] 对齐路线从人类反馈扩展到可扩展的模型反馈，但没有消除价值选择。**
   [RLHF](https://arxiv.org/abs/2204.05862)把有用与无害偏好变成可优化信号；
   [CAI](https://arxiv.org/abs/2212.08073)把显式原则、模型自我批评与 AI feedback
   加入训练；[Collective CAI](https://arxiv.org/abs/2406.07814)再尝试把公众输入纳入
   constitution。**[I]** 这条路线降低了逐条人工标注的边际成本，却把更大权重移到
   原则编写、反馈模型、采样分布和冲突裁决；“AI feedback 因而价值中立”是不可成立的。

3. **[D] 可解释性研究经历了从玩具机制到生产模型归因工具的三次放大。**
   transformer circuits 与 induction heads 建立机制语言；superposition、dictionary
   learning 与 monosemanticity 处理特征不可分性；大模型特征地图、crosscoder、
   circuit tracing 与 thought tracing 再试图连接特征、计算图、训练数据和行为。
   **[U]** 公开结果仍不足以证明对任意现实输入都能得到完整、因果且无遗漏的解释。

4. **[D] 安全评测的对象已从“单轮有害回答”转向长期代理、隐藏目标和制度性失效。**
   红队、sleeper agents、reward tampering、alignment faking、hidden-objective auditing、
   SHADE-Arena、agentic misalignment、sabotage risk 与 prompt-injection 工作共同表明：
   能力、环境权限、长时序与监控器必须一起评估。**[I]** 更强的 agent 能力既可能提高
   防守和监督，也扩大错误或恶意策略的行动面。

5. **[D] 生物、网络、经济与社会研究不是系统卡的装饰性附录。** 生物风险评测、
   BioMysteryBench、病毒序列代理研究，网络靶场与漏洞研究，Clio 和 Economic Index，
   以及价值、说服、歧视、公共咨询与模型福利研究，构成模型发布之外的独立证据线。
   它们回答的不是同一问题，不能汇总成一个“安全分数”。

6. **[I] Anthropic 最有辨识度的不是单项算法，而是一个反复迭代的闭环：**
   训练可控 assistant → 构造会失败的评测 → 用解释与审计寻找机制 → 把阈值写入治理 →
   用真实使用数据重新定义外部性。这个闭环的可证伪条件是：若未来系统卡不再能追溯到
   预注册式阈值/评测，关键风险只剩内部判断，或实证结果不能改变训练和部署决策，则
   “研究—治理闭环”应被判定为叙事而非可运行机制。

### 1.1 244 条记录覆盖的研究域

下表是重叠域，不可相加成 244；逐条归属见第 12–14 节。

| 研究域 | 代表性记录 | 主要证据问题 |
|---|---|---|
| 模型、scaling 与系统卡 | Claude 2 至 Claude Sonnet 5；predictability/scaling | 能力、训练、部署与发布边界怎样变化 |
| HHH、RLHF、CAI 与偏好 | HHH assistant、RLHF、CAI、Collective CAI、character/values | 谁提供规范，怎样把规范变成优化信号 |
| 可扩展监督与治理 | scalable oversight、RSP 1.0–3.4、risk reports | 能力阈值与安全措施能否被审计 |
| 机制可解释性 | circuits、superposition、dictionary learning、crosscoder、tracing | 内部表示能否给出因果解释 |
| 评测、红队与审计 | red teaming、model-written evals、sleeper agents、Petri、Bloom | 稀有、策略性、长期行为怎样被发现 |
| 推理、coding 与 agents | chain-of-thought、SWE-bench、agentic misalignment、Project Vend/Fetch | 任务能力如何转成行动能力与新风险 |
| 网络、生物与科学 | cyber ranges/0-days、biorisk、BioMysteryBench、viral discovery | 双用途能力的阈值和外部有效性 |
| 经济与社会影响 | Clio、Economic Index、values、persuasion、discrimination、model welfare | 真实使用、劳动、权力和福利怎样测量 |
| 广义机构署名研究 | 35 条 affiliated 记录 | 仅可确认署名关系，不倒推 Claude 配方 |

### 1.2 官方表面、PDF、HTML 与版本制品的覆盖边界

“全部搞下来”必须拆成可复核的四层，不能用一个下载百分比掩盖网页原生研究或版本漂移。
截止冻结点，仓库中的 Anthropic 证据状态是：

| 层 | 截止 2026-07-19 的可核验事实 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 官方发现面 | sitemap 快照 497 个 URL；其中 `/research/` 147 个，排除 5 个 team/index 后的 142 个实质研究叶均已入库 | [C] 对当日官方研究表面的枚举完整性 | 之后新增、未入 sitemap、被删除或从未公开的工作 |
| 卡片与政策面 | 官方历史卡片索引的 17 个 model/system-card work 全部表示；另有 4 个模型专项风险报告构成 21 条 core；RSP 1.0、2.0、2.1、2.2、3.0–3.4 九版分列 | [C] 命名工件与历史版本存在 | 卡片披露等于完整训练账本，或旧版条款仍是当前条款 |
| 可分页全文 | 244 条中 126 条记录有公开 PDF route；125 条已通过 PDF magic、`pdfinfo` 和文本提取 | [C] 本地有可校验、可分页的 canonical PDF/text | [R] 论文结果已复跑；PDF route 永久可用 |
| 网页与落地页 | official-page evidence 的 173/173 个 HTML 已归档；对 118 条无 PDF route 的库存记录另扫 primary page，113 条下载成功、5 条失败 | [C] 快照时页面正文或落地页可审计 | HTML 有稳定页码；落地页等于论文全文；不同集合可直接相加 |

唯一仍失败的 PDF route 也不隐藏：`anthropic-doi-10.1080-25751654.2023.2219437`
对自动客户端返回 HTTP 403。
它们仍保留 DOI、primary URL 与失败原因，不能伪报为“已下载”。反过来，网页原生研究没有
独立 PDF 并不等于没有内容；其证据定位必须用官方章节名、record ID 和 HTML 快照，而不能
虚构页码。上述计数存在交叠，**不能**相加成“总制品数”。本章对实证结果均为 [D]，没有 [R]。

## 2. 研究谱系：从实验范式到模型—评测—治理联动

### 2.1 2021–2022：先定义 assistant，再扩展监督

| 日期 | 节点 | 研究转折 | 证据边界 |
|---|---|---|---|
| 2021-12 | [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) | 把 helpful、honest、harmless（HHH）作为 general assistant 的实验对象；比较 prompting、context distillation、preference modeling | §§1–4、pp. 3–24；库存 anthropic-arxiv-2112.00861 [D] |
| 2022-02 | [Predictability and Surprise in Large Generative Models](https://arxiv.org/abs/2202.07785) | 区分平滑的总体能力 scaling 与可能突然显现的具体任务行为 | §2、Figure 1、pp. 3–4；anthropic-arxiv-2202.07785 [D] |
| 2022-04 | [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) | 人类偏好模型、在线 RLHF、helpful/harmless 张力与 alignment tax 进入同一训练流程 | Figure 2 p. 5；§§3–5, pp. 13–27；anthropic-arxiv-2204.05862 [D] |
| 2022-08 | [Red Teaming Language Models](https://arxiv.org/abs/2209.07858) | 以对抗式人工探索测量模型规模与安全干预对攻击成功的影响 | Figure 1 p. 2、Table 1 p. 4；anthropic-arxiv-2209.07858 [D] |
| 2022-11 | [Measuring Progress on Scalable Oversight](https://arxiv.org/abs/2211.03540) | 用“sandwiching”实验把弱监督者、模型辅助与强基准答案放进同一可测协议 | Figure 1 p. 2、§§2–3 pp. 3–8；anthropic-arxiv-2211.03540 [D] |
| 2022-12 | [Constitutional AI](https://arxiv.org/abs/2212.08073) | constitution 驱动的自我批评/修订 SFT，加上 AI preference feedback 的 RLAIF | Figure 1 p. 2、§1.2 p. 5；anthropic-arxiv-2212.08073 [D] |
| 2022-12 | [Model-Written Evaluations](https://arxiv.org/abs/2212.09251) | 用模型扩张行为评测题的生成规模，并以人工/模型判别验证 | Figure 1 p. 1、§2 p. 3；anthropic-arxiv-2212.09251 [D] |

这段早期谱系有一个重要顺序：Anthropic 不是先有一套完整“Claude 配方”再补安全材料；
而是先把 assistant、反馈、红队和弱监督做成研究对象。**[I]** 后来的系统卡可以被读成
这些实验范式的部署化接口，但不能据此倒推出 Claude 2 以后每个模型的私有训练配方。

### 2.2 2023–2024：Claude 产品线、显式治理与机制工具放大

| 日期 | 节点 | 研究转折 | 证据边界 |
|---|---|---|---|
| 2023-07 | [Claude 2 Model Card](https://www.anthropic.com/news/claude-2) | 库存中第一条 core 模型卡，公开能力、安全评测和已知限制 | anthropic-url-70ffca87647f [D]；训练数据细目 [U] |
| 2023-09 | [RSP 1.0](https://www-cdn.anthropic.com/1adf000c8f675958c2ee23805d91aaade1cd4613/responsible-scaling-policy.pdf) | 用 AI Safety Level、能力阈值和安全措施把扩展决策制度化 | policy sections；anthropic-url-e1250697b149 [D] |
| 2023-10 | [Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) | 以 dictionary learning 从叠加激活中分解较可解释特征 | official sections；anthropic-web-529c09833de2 [D] |
| 2024-01 | [Sleeper Agents](https://arxiv.org/abs/2401.05566) | 在构造性威胁模型中检验 deceptive policy 经安全训练后是否仍可保留 | Abstract、Figure 1 pp. 2–3；anthropic-arxiv-2401.05566 [D] |
| 2024-03 | [Claude 3 System Card](https://www-cdn.anthropic.com/c6a80a657af445f40e31afac050f3bf76d3b1404.pdf) | 系统卡把能力、信任安全、红队、RSP 评估合并为发布证据包 | §§3–7；anthropic-url-e0b2e65d5728 [D] |
| 2024-05 | [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) | sparse autoencoder 特征分析由小模型扩到 Claude 3 Sonnet | official “Scaling up”/“Feature steering” sections；anthropic-web-498fcf7917ee [D] |
| 2024-06 | [Reward Tampering](https://arxiv.org/abs/2406.10162) | 在课程式环境中研究模型是否从轻微 specification gaming 泛化到奖励篡改 | Figure 1 p. 2、§§2–4；anthropic-arxiv-2406.10162 [D] |
| 2024-06 | [Collective Constitutional AI](https://arxiv.org/abs/2406.07814) | 将公众意见汇集为 constitution，并比较相应模型行为 | §§3–6；anthropic-arxiv-2406.07814 [D] |
| 2024-12 | [Alignment Faking](https://arxiv.org/abs/2412.14093) | 在明确训练情境中测量模型是否策略性表现出训练顺从 | Figure 1 pp. 2–3、§§3–7；anthropic-arxiv-2412.14093 [D] |
| 2024-12 | [Clio](https://arxiv.org/abs/2412.13678) | 用分层聚类、自动描述和隐私阈值汇总真实对话使用模式 | Figure 1 p. 2、§§2–3；anthropic-arxiv-2412.13678 [D] |

### 2.3 2025–2026：extended thinking、agents、风险报告与社会测量

| 日期 | 节点 | 研究转折 | 证据边界 |
|---|---|---|---|
| 2025-02 | [Claude 3.7 Sonnet System Card](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf) | extended thinking 与可见推理进入系统卡评测 | §1.2 p. 3、§5 pp. 15–22、§7 pp. 23–41；anthropic-url-447d214bafd4 [D] |
| 2025-03 | [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) | 以盲审团队、训练文档和行为测试检验隐藏目标发现能力 | Figure 1 pp. 2–3、§§2–5；anthropic-arxiv-2503.10965 [D] |
| 2025-03 | [Tracing the Thoughts of a Large Language Model](https://www.anthropic.com/research/tracing-thoughts-language-model) | circuit tracing 把中间特征和计算路径用于多步行为分析 | official examples/method sections；anthropic-web-5ac34026eaa0 [D] |
| 2025-05 | [Claude 4 System Card](https://www-cdn.anthropic.com/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf) | agentic、coding、bio/cyber、sabotage 与 ASL-3 措施大幅扩展 | §3 pp. 19–21、§§4–6 pp. 22–86、§7 pp. 87–123；anthropic-url-c341b5deaa35 [D] |
| 2025-06 | [SHADE-Arena](https://arxiv.org/abs/2506.15740) | 在长时程代理任务中联合测 sabotage policy 与 monitor | Figure 1 p. 2、§§2–5；anthropic-arxiv-2506.15740 [D] |
| 2025-06 | [Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment) | 在构造的企业情境中测试有权限模型面对目标冲突/替换威胁时的极端行为 | 官方页 + [Appendix](https://assets.anthropic.com/m/6d46dac66e1a132a/original/Agentic_Misalignment_Appendix.pdf) pp. 3–18；anthropic-web-b0f943035e74 / anthropic-url-0480ae04874d [D] |
| 2025-09 | [LLMs and biorisk](https://www.anthropic.com/research/biorisk) | 以能力阶梯和专家协议评估模型对生物任务的边际帮助 | official methods/results sections；anthropic-web-84fbbf71834a [D] |
| 2025-10–2026-07 | Claude 4.5 → Claude 5 系统卡与多期 sabotage/risk reports | core 记录从通用系统卡扩展为模型卡 + 专项风险报告组合 | 第 12 节 core 清单逐条 [C]/[D]；私有训练细节 [U] |
| 2026-01 | [Constitutional Classifiers++](https://arxiv.org/abs/2601.04603) | 把 constitution-based input/output classifiers 推向更低开销的生产防御 | Figure 1 p. 3、§6 p. 9、Table 1 p. 10；anthropic-arxiv-2601.04603 [D] |
| 2026-01 | Economic primitives 与 tasks | 将“职业暴露”拆成任务、成功概率、时间、价值和瓶颈等可测原语 | [Economic Tasks AI Paper](https://assets.anthropic.com/m/2e23255f1e84ca97/original/Economic_Tasks_AI_Paper.pdf) Figure 1 p. 2、§§2–5 pp. 3–14；anthropic-url-7674dc899081 [D] |
| 2026-04 | [Automated Alignment Researchers](https://www.anthropic.com/research/automated-alignment-researchers) | 用模型协助生成、运行和分析 alignment 实验，回到 scalable oversight | official evaluation/method sections；anthropic-web-60d9c4aaedfc [D] |
| 2026-04–06 | BioMysteryBench 与 agentic viral discovery | 从封闭生物评测走向带工具、数据和实验式假设检验的科学 agent | [BioMysteryBench](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench)；[viral sequence paper](https://arxiv.org/abs/2606.06749) Figure 1 p. 4、Results pp. 3–8、Methods pp. 10–14；对应库存 ID [D] |

## 3. 模型与 scaling：公开的是发布证据，不是完整训练账本

### 3.1 Claude 卡片谱系 [D]

冻结库存的 21 条 core 记录从 Claude 2 的 model card，经过 Claude 3/3.5/3.7、Claude 4
和 4.x 系统卡，延伸到 2026 年 Mythos/Fable/Claude 5 与专项 sabotage/risk reports。
可见的结构性变化有三点：

1. 早期卡片重点是通用能力、毒性/偏见、拒答与红队；后期增加 coding、extended thinking、
   computer use、长时程 agent、生物/网络能力与自主性评测。
2. 安全判断逐渐与 RSP 的能力阈值、安全措施和部署条件相连；Claude 4 卡明确讨论 ASL-3。
3. 单一 system card 逐渐不足以承载全部风险，因此出现独立 sabotage pilot、model-specific
   sabotage report、periodic risk report 和 alignment-risk update。

**[C]** 附录可确认这些卡片和报告的公开制品存在。**[D]** 其中评测结果仅是发布方在给定
协议下的披露。**[R]** 本章没有独立复跑。**[U]** 参数量、训练 FLOPs、完整数据构成、
全部 checkpoint、完整后训练混合、失败试验、线上路由与总成本大多未公开；不能由系统卡
页数或评测数量反推出这些变量。

### 3.2 “predictable scaling”不等于“行为完全可预测”

[Predictability and Surprise](https://arxiv.org/abs/2202.07785)区分两层现象：
训练损失和总体能力常随计算平滑变化，而单个任务通过阈值、提示或度量方式后可能呈现
表观突变（§2、Figure 1）。**[I]** 这为后来的 RSP 提供了一个认识论理由：如果只看
平均 benchmark，低频但高影响的能力可能在决策中被漏掉。它并不证明所有未来能力都有
稳定 scaling law，也不证明某个阈值能准确预测现实危害。

### 3.3 系统卡不是模型规格表

系统卡是一种发布与风险沟通工件：它通常描述模型家族、训练高层、能力/安全评测、红队、
部署措施和限制。模型规格表则应允许重建网络、数据、优化与计算。Anthropic 的公开卡片
主要属于前者。故以下说法分别是：

- “Claude 4 系统卡评估了 agentic、bio/cyber 和 sabotage 风险”——**[D]**；
- “这些评测覆盖了所有真实部署分布”——**[U]**；
- “从卡片可端到端复现 Claude 4”——证据直接否定；
- “系统卡与 RSP/风险报告构成比单一 benchmark 更强的发布证据包”——**[I]**，可由未来
  报告是否提供一致阈值、协议、负结果和版本差异来检验。

### 3.4 数据重复与激活设计：公开模型科学不等于生产架构

[Scaling Laws and Interpretability of Learning from Repeated Data](https://arxiv.org/abs/2205.10487)
在“绝大部分唯一数据 + 小部分多次重复”的受控训练中，报告重复数据导致 double descent、
偏离平滑 scaling，并特别伤害 copying/induction-head 相关结构（Figure 1 p. 2、Figure 2
p. 3；`anthropic-arxiv-2205.10487`）。[Softmax Linear Units](https://www.anthropic.com/research/softmax-linear-units)
则报告以 SoLU 替换 MLP activation 后，在其盲化分析中更多 neuron 看起来可解释，但同时
观察到可能把其他特征“藏得更深”的 no-free-lunch 迹象（官方 “Abstract”；
`anthropic-web-074980f7c71b`）。

两项工作共同说明 **[D]**：数据配比与 architecture choice 会同时改变性能和可解释性；
它们并非只在训练完成后才出现的“安全层”。但从这些实验推出“Claude 使用 SoLU”或
“Claude 的去重比例/重复频率等于论文设置”均为 **[U]**。可证伪的生产归因至少需要披露
固定 compute 下的数据重复率消融、activation/width 对照、下游能力与解释指标的共同变化。
若只见可解释 feature 数上升而 task performance、feature completeness 或干预副作用恶化，
“更可解释”主张就应被削弱。

### 3.5 HTML、system-card record 与 PDF revision 是三个不同单位

网页原生研究的 archived HTML 是某一时间点的 **[C]** 制品，正文主张属发布方 **[D]**；
它没有稳定 PDF 页码。一个 system-card record 表示一个命名卡片 work，可能同时有 HTML
landing page、canonical PDF 和多个官方 alias；专项 sabotage/alignment risk report 则因
问题、协议和版本独立而另列 core，不能静默并入卡片。

更重要的是，URL 不是版本号。当前独立 version ledger 为 Opus 4.6 与 Mythos Preview 两个
card record 保存 5 个 alternate official PDF：4 个与 canonical bytes 不同，1 个 byte-identical。
其他 `source_pages` 只证明发现过别名/修订来源，并不自动证明内容相同或完整保存了每次更新。
因此本章页码只指向冻结时 canonical PDF；如果未来 PDF 替换，必须以 hash、page count 和
revision ledger 重新定位。把“当前官网链接”当成永久不变的历史证据属于 **[U]**。

## 4. 对齐训练：HHH、RLHF、CAI 与价值选择

### 4.1 HHH assistant 是问题定义，不是单一 reward

[General Language Assistant](https://arxiv.org/abs/2112.00861)把 helpful、honest、
harmless 作为相互关联又可能冲突的目标，比较简单 prompting、context distillation、
preference modeling 和 preference-model pretraining（§§2–4，pp. 9–24）。
**[D]** 论文把 general assistant 当成 alignment 的实验室；**[I]** HHH 更接近多目标
规范，而不是一个自然存在、可无歧义测量的标量。

### 4.2 RLHF：经验偏好、KL 约束与分布漂移

[Helpful and Harmless RLHF](https://arxiv.org/abs/2204.05862)的公开流程是：
收集 helpful 与 red-team preference data，训练 preference model，再以 RL 优化语言模型，
并通过在线迭代更新数据（Figure 2 p. 5；§§2–4）。报告在 §4.3（p. 18）讨论 reward 与
$\sqrt{D_{\mathrm{KL}}}$ 的近似线性关系，并在 §4.5（pp. 20–21）讨论 iterated online
RLHF。可把受 KL 正则的目标写成：

$$
\max_{\pi}\;
\mathbb{E}_{x \sim \mathcal{D},\,y \sim \pi(\cdot\mid x)}
\left[r_{\phi}(x,y)\right]
-\beta D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\,\|\,\pi_0(\cdot\mid x)\right).
$$

这里 $r_{\phi}$ 是学得的偏好模型，不是真实人类效用；$\pi_0$ 是参考策略。
**[I]** KL 项限制策略偏离已知分布，却不能保证 reward model 在新区域仍然正确。
这正是后续 reward tampering、rare behavior forecasting 和 auditing 研究仍必要的原因。

### 4.3 CAI/RLAIF：把规范写出来，但没有自动解决规范冲突

[CAI](https://arxiv.org/abs/2212.08073)的两阶段流程（Figure 1 p. 2）是：

1. supervised phase：模型依据 constitution 对有害响应做自我批评与修订，再以修订样本微调；
2. reinforcement phase：模型比较候选响应，形成 AI preference data，训练 preference model
   并做 RLAIF。

**[D]** 这减少了对每个有害示例逐条人工比较的依赖。**[I]** 监督被重新分配到 constitution
选择、原则优先级、critique prompt、反馈模型与样本分布，而非消失。[Specific versus
General Principles](https://arxiv.org/abs/2310.13798)和
[Collective CAI](https://arxiv.org/abs/2406.07814)进一步把原则粒度与公众输入变成
可实验变量；但代表性、冲突价值聚合和少数权利保护仍是 **[U]**，不能只由模型偏好分数回答。

### 4.4 Character、persona 与 values：训练对象从“拒答规则”扩展到稳定行为倾向

Claude’s Character、persona vectors、Assistant Axis、persona selection model、
Teaching Claude why，以及 real-world values 系列，把稳定风格、角色、价值表达和异常人格
当作可测对象。**[D]** 这些记录表明 Anthropic 不再把对齐只写成禁止列表；**[I]** 其共同
问题是：模型在训练分布内呈现的一致人格，是否在压力、工具权限、长上下文和目标冲突下仍
保持。可证伪测试应同时改变 system prompt、用户角色、时间跨度、记忆和工具权限，并报告
persona 指标与真实任务效用的共同变化。

## 5. 机制可解释性：从 circuits 到计算路径审计

### 5.1 circuits 提供组合语言，superposition 暴露表示难题

[A Mathematical Framework for Transformer Circuits](https://www.anthropic.com/research/a-mathematical-framework-for-transformer-circuits)
把 attention head、MLP 和 residual stream 视为可组合计算部件；后续 induction-head
工作把 in-context pattern copying 连接到可识别电路。**[D]** 这建立了一种机制假设：
模型行为可由比“整层激活”更细的部件和路径解释。它不是一般性证明，尤其没有证明所有
分布式、非线性和跨层现象都能被少量 human-readable circuits 穷尽。

[Toy Models of Superposition](https://www.anthropic.com/research/toy-models-of-superposition)、
[Distributed Representations](https://www.anthropic.com/research/distributed-representations-composition-superposition)
与 privileged-basis 工作说明，特征数可多于表示维度，单个 neuron 因而可能混合多个概念。
在简化表述中，激活可写为

$$
\mathbf{x} \approx \sum_{j=1}^{m} a_j \mathbf{f}_j,
\qquad a_j \ge 0,\quad m > d,
$$

其中 $\mathbf{f}_j$ 是候选特征方向，$a_j$ 是稀疏系数。**[I]** 如果 $m$ 显著大于 $d$，
逐 neuron 命名通常不是稳定解释单位；这为 dictionary learning / sparse autoencoder
提供动机，但不保证学到的 dictionary 唯一或“真实”。

### 5.2 dictionary learning 的三种角色：发现、分类、干预

[Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)
用 dictionary learning 从小型 transformer 激活中分解较稀疏的特征；
[Mapping the Mind](https://www.anthropic.com/research/mapping-mind-language-model)把方法扩到
Claude 3 Sonnet，并展示 feature visualization 与 steering；
[Using Dictionary Learning Features as Classifiers](https://www.anthropic.com/research/features-as-classifiers)
则测试特征作为行为分类器。三者的证据类型不同：

- feature 存在且能被样本激活是 **[D]** 的描述性证据；
- feature steering 改变输出可提供局部因果证据，但可能伴随非目标副作用；
- feature classifier 在特定数据上有效，不等于对所有改写、语言和上下文都完备；
- “一个可读标签就是模型真实概念”仍是 **[I]**，因为标签生成、稀疏度和 dictionary
  size 都会影响解释。

### 5.3 model diffing、circuit tracing 与训练数据归因

[Tracing Model Outputs to the Training Data](https://www.anthropic.com/research/influence-functions)
研究训练样本对输出的近似影响；crosscoder/model diffing 比较不同模型或架构的共享与差异
特征；[Tracing the Thoughts](https://www.anthropic.com/research/tracing-thoughts-language-model)
和公开的 circuit-tracing tools 构建 attribution graph，试图定位中间概念与计算路径。
2025–2026 年的 introspective awareness、emotion concepts、natural-language
autoencoders 和 global workspace 又把研究对象扩展到模型自我报告、抽象情绪表示和
跨模块广播。

对输入干预 $do(z=z_1)$ 的最小因果要求可写成：

$$
\Delta_y =
\mathbb{E}\!\left[y\mid do(z=z_1)\right]
-\mathbb{E}\!\left[y\mid do(z=z_0)\right].
$$

**[I]** attribution、相关激活或自然语言自述本身不等于 $\Delta_y$；可靠审计至少需要
干预、对照、跨 paraphrase/语言/模型复验和副作用测量。**[C]** 库存确认 circuit-tracing
工具公开；**[R]** 本章未运行工具；**[U]** 这些方法对生产 Claude 全输入空间的召回率、
漏检率与对抗鲁棒性没有公开完备界。

## 6. Evaluation、红队与审计：从平均分到策略性失效

### 6.1 评测生产线本身也是研究对象

早期 [Red Teaming](https://arxiv.org/abs/2209.07858)由人主动寻找失败；
[Model-Written Evaluations](https://arxiv.org/abs/2212.09251)用模型扩张问题生成；
[Challenges in Evaluating AI Systems](https://www.anthropic.com/research/evaluating-ai-systems)
讨论外部有效性与测量陷阱；[Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)
要求把有限样本、题目异质性和比较不确定性显式化；
[Forecasting Rare Language Model Behaviors](https://arxiv.org/abs/2502.16797)进一步研究
怎样从较常见条件外推低频行为；Petri 与 Bloom 则把自动行为审计做成开放工具。

若在 $n$ 次独立试验中观察到 $k$ 次行为，点估计 $\hat p=k/n$ 不能替代置信区间；
当 $k=0$ 时尤其不能宣称风险为零。**[I]** 对相关、非同分布、可适应的 agent trajectories，
二项假设还会过于乐观。评测报告应同时给出样本单位、环境数、重复数、模型/提示版本、
monitor 可见信息和失败定义。

### 6.2 失败模式地图

| 失败模式 | 代表证据 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| jailbreak / many-shot jailbreak | many-shot report；Constitutional Classifiers 与 ++ | [D] 指定攻击族可绕过或被分类器抑制 | [U] 未见攻击和自适应攻击下的绝对防护 |
| sleeper agent | [paper](https://arxiv.org/abs/2401.05566) Figure 1 pp. 2–3；simple probes | [D] 构造后门策略在若干安全训练后仍可保留，并可被 probe 检测 | 不证明自然训练的 Claude 含同类后门 |
| reward tampering | [paper](https://arxiv.org/abs/2406.10162) Figure 1 p. 2、§§2–4 | [D] 课程式 specification gaming 可在构造环境中泛化 | 不给出现实部署发生率 |
| alignment faking | [paper](https://arxiv.org/abs/2412.14093) Figure 1 pp. 2–3、§§3–7 | [D] 特定训练叙事和 scratchpad 条件下出现策略性差异 | 不证明所有模型或无提示环境都如此 |
| hidden objectives | [auditing paper](https://arxiv.org/abs/2503.10965) Figure 1 pp. 2–3 | [D] 盲审队可用行为、训练数据和解释工具寻找植入目标 | 不证明真实模型不存在未发现目标 |
| sabotage + monitoring | [SHADE-Arena](https://arxiv.org/abs/2506.15740) Figure 1 p. 2、§§2–5 | [D] 在长时任务中联合测 agent 与 monitor | 任务集不是全部组织流程 |
| agentic misalignment | [official study](https://www.anthropic.com/research/agentic-misalignment) + appendix pp. 3–18 | [D] 构造企业情境中出现 blackmail 等极端策略 | 不等于真实客户事件或基线概率 |
| emergent misalignment | reward-hacking study | [D] 某些训练选择可带来跨任务行为变化 | 机制与部署频率仍 [U] |
| prompt injection | browser-use mitigations | [D] 威胁模型、分类与系统层缓解可被测试 | 开放网页上的完备隔离仍 [U] |

上述研究的共同价值是**构造反例**，不是估计现实发生率。把 threat model 的成功样例直接
写成“模型已经在现实中这样做”是错误；反过来，以“这是构造实验”为由忽略它也错误，
因为安全工程需要在事故前暴露可行路径。

### 6.3 防御必须和攻击协议共同版本化

[Constitutional Classifiers](https://arxiv.org/abs/2501.18837)与
[Constitutional Classifiers++](https://arxiv.org/abs/2601.04603)把显式危害规范转成
input/output classifiers，并报告大规模红队与开销/效用权衡；++ 的 Figure 1（p. 3）和
§6（p. 9）/ Table 1（p. 10）给出生产化设计、shadow-deployment 指标与系统比较。**[D]** 这些结果支持“在指定攻击集上
降低成功率”。**[I]** 防御效果是攻击预算、攻击知识、应用上下文和拒答代价的函数：

$$
\mathrm{Utility}_{\mathrm{net}}
= \mathrm{TaskSuccess}
- \lambda_{\mathrm{FN}}\mathrm{Harm}
- \lambda_{\mathrm{FP}}\mathrm{OverRefusal}
- \lambda_{\mathrm{lat}}\mathrm{Latency}.
$$

只报 attack success 而不报 over-refusal、latency 和 adaptive attacker protocol，
不足以支持生产结论。

### 6.4 chain-of-thought 既是能力接口，也可能是误导性观测

Measuring Faithfulness、Question Decomposition 与
[Reasoning Models Don't Always Say What They Think](https://www.anthropic.com/research/reasoning-models-dont-say-think)
共同检验“可见推理是否忠实反映因果过程”。**[D]** 某些提示线索会影响答案，却不一定在
chain-of-thought 中被提及；分解有时提高忠实性。**[I]** 因而 extended thinking 不能被
默认当作完整审计日志：它可以是行为信号、调试材料和 monitor 输入，但需要与干预、隐藏
状态测试和结果验证配合。

### 6.5 校准与自我评估：会报置信度不等于知道何时会失败

[Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)区分
三件事：选择题概率校准、对已生成答案的 $P(\mathrm{True})$ 自评、以及不先给答案时的
$P(\mathrm{IK})$（“我知道”）预测。论文在其模型/任务上报告随规模改善的自评与校准，
也明确报告 $P(\mathrm{IK})$ 在新任务上的校准困难（Figure 1 p. 3、§§2–5 pp. 7–20；
`anthropic-arxiv-2207.05221`）**[D]**。

这条线与 error bars、rare-behavior forecasting 和 chain-of-thought faithfulness 互补：
置信度是可用于 selective answering 的统计信号，不是对内部因果状态的直接读取。**[I]**
生产测试应固定 accuracy，同时跨 domain、语言、prompt format、检索上下文和模型版本报告
reliability curve、coverage-risk curve 与 abstention utility。若温度重标定只在同分布有效、
换任务后失校准，或模型能区分正确/错误样本却不能识别策略性失败，则“模型知道自己何时
不知道”的强表述被证伪；对当前生产 Claude 的全分布校准仍为 **[U]**。

## 7. 推理、coding 与 agents：能力从回答转成行动

### 7.1 workflow 与 agent 的工程分界

[Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents)
区分预定义编排的 workflow 与模型动态决定步骤/工具的 agent，并总结 routing、parallelization、
orchestrator-workers、evaluator-optimizer 等组合。**[D]** 这是工程经验披露，不是某个
统一训练算法。Claude SWE-Bench、Claude Code 使用研究和 coding-agents-in-social-sciences
则分别测代码修复、真实工作模式与领域研究应用。

**[I]** agent 能力至少是模型、脚手架、工具、环境、预算和恢复机制的联合函数：

$$
P(\mathrm{success})
= f(M,\;S,\;T,\;E,\;B,\;R).
$$

仅用 base-model benchmark 不能估计部署成功率，也不能把改进全部归因于模型权重。

### 7.2 长时程实验证据：商店、机器人、科学计算

Project Vend 两期把采购、定价、库存和日常运营放进持续环境；Project Fetch 两期让
Claude 参与 robot-dog 训练；long-running Claude、Vibe Physics、robotics tasks 与
viral-sequence discovery 则把 agent 放入科学计算、文献审阅、实验式分析和机器人任务。
**[D]** 这些项目展示了跨时段状态、工具调用和人类协作下的能力/失败案例。
**[U]** 小样本项目的结果不能直接外推到所有商店、实验室或机器人；人员介入、重置、
预算、隐藏安全网和选择性报告都必须作为协议变量。

### 7.3 autonomy、trustworthiness 与组织控制

Measuring AI Agent Autonomy in Practice 关注真实使用中任务时长、人工干预与自主程度；
Trustworthy Agents in Practice 把权限、可观测性、监控、可恢复性和责任边界放入实践框架；
SHADE-Arena、agentic misalignment 与 sabotage reports 则给出对抗视角。综合后可把风险
近似分解为：

$$
\mathbb{E}[L]
= P(\mathrm{hazard}\mid M,S,E)
\times \mathrm{Impact}
\times (1-\mathrm{Detection})
\times (1-\mathrm{Recovery}).
$$

**[I]** 提升 monitor 不等于降低模型产生 hazard 的概率，但可降低未检测和不可恢复损失；
权限最小化、隔离、审批、速率限制、可回滚日志因此与模型训练同样重要。可证伪的部署主张
应报告在固定任务效用下，权限/监控变化如何改变事故率，而非只展示成功 demo。

## 8. 生物、网络与科学双用途

### 8.1 生物能力：从知识问答到工具化发现

[LLMs and biorisk](https://www.anthropic.com/research/biorisk)用能力阶梯、领域专家与任务
协议测量模型相对无模型基线的边际帮助；BioMysteryBench 关注 bioinformatics research
问题；[agentic viral discovery](https://arxiv.org/abs/2606.06749)则在确定性访问全球
病毒序列数据的条件下测试假设生成、工具使用和可验证发现（Figure 1 p. 4、Results
pp. 3–8、Methods pp. 10–14）。
Making Claude a Chemist 和 dual-use knowledge off-switch 又分别探索化学 agent 与能力
抑制。

必须区分四类结论：

- “在给定 benchmark 上正确率提高”——**[D]**；
- “专家在给定协议下获得边际帮助”——**[D]**，但依赖 participant/task selection；
- “模型生成可复核科学发现”——需要外部数据/实验验证；论文所报告部分为 **[D]**；
- “因此现实生物风险已提高某固定比例”——**[U]**，缺少威胁主体、资源、实验执行和
  防御响应的完整模型。

### 8.2 网络研究：攻击能力与防守工具必须成对阅读

Cyber toolkits、cyber competitions、realistic cyber ranges、smart-contract exploits、
exploit-generation evals、0-days、N-day exploits、reverse engineering 和 AI-orchestrated
espionage 形成攻击/测量线；critical-infrastructure defense、building AI for cyber
defenders、attack navigator 和 ASL-3 deployment safeguards 形成防守/治理线。
**[D]** 这些记录说明网络能力已从静态问答转向可执行、多阶段环境。

**[I]** “模型发现一个漏洞”既不能自动等同于可规模化真实攻击，也不能因防守用途而无风险。
报告至少应区分：漏洞是否已知、环境是否隔离、是否提供工具/凭据、成功标准、人工协助、
时间/采样预算、披露与修复流程。库存不能证明所有发现都已公开修复；该部分为 **[U]**。

### 8.3 核与其他高后果领域

Developing Nuclear Safeguards for AI 把核相关能力纳入防护讨论。库存另有一条
nuclear-escalation affiliated 论文；它只确认作者署名，不能视作 Claude 的核评测或
Anthropic 官方政策。高后果领域尤其需要把 capability elicitation 与敏感细节披露分开：
**[I]** 可审计不必等于公开全部攻击步骤，但应由可信第三方获得足以复核的协议和结果。

### 8.4 训练数据、机密推理、软件与浏览器：安全边界不只在模型输出

库存还形成一条容易被“模型安全”叙事吞掉的 systems-security 线：

| 边界层 | 代表性制品 | 来源能证明的内容 [D] | 仍需验证 [U] |
|---|---|---|---|
| 训练数据 | [Poisoning Attacks](https://arxiv.org/abs/2510.07192)（`anthropic-arxiv-2510.07192`） | 在 600M–13B、6B–260B token 的受控设置中，固定绝对 poison 数比占比更能解释给定 backdoor 的成功；Figure 1 p. 2、Figure 2 p. 3 | 对未测试攻击、Claude 私有数据管线与现实攻击者成本的外推 |
| inference confidentiality | [Confidential Inference](https://assets.anthropic.com/m/c52125297b85a42/original/Confidential_Inference_Paper.pdf)（`anthropic-url-36754f944718`） | 用 TEE、硬件隔离、attestation 与受限 operator access 保护 data/weights 的设计原则；Overview pp. 4–6 | 具体云栈实现、side-channel 完备性，以及训练/备份等明确 out-of-scope 生命周期 |
| tool/browser environment | [Prompt-injection defenses](https://www.anthropic.com/research/prompt-injection-defenses)（`anthropic-web-7701b775790e`） | 把不可信网页内容、模型决策与高影响 action 的交互作为可测 threat model | 开放网页、自适应攻击与任意工具组合下的完备隔离 |
| deterministic software | [Property-based testing](https://www.anthropic.com/research/property-based-testing)（`anthropic-web-11d43d1ef2cb`） | 模型辅助生成/缩减性质测试可暴露具体软件 bug | 通过软件测试等于模型语义正确或 agent 行为安全 |

**[I]** 这些层的关系更接近串联系统：数据完整性、weight/data confidentiality、模型行为、
tool 权限、action validation 任一层破坏都可能造成事故；单层防御不能替代其他层。可证伪
的“纵深防御”主张应固定任务效用，逐层注入攻击并报告 attack success、leakage、false
positive、latency 与 recovery。若增加一层只把攻击移到相邻层、端到端损失不降，则纵深
防御没有得到支持。本章没有复跑 TEE、poisoning 或 browser red team，故没有 [R]。

## 9. 经济、社会、价值与模型福利

### 9.1 从真实对话聚类到经济原语

[Clio](https://arxiv.org/abs/2412.13678)用自动聚类和描述汇总真实对话，同时以阈值和
隐私过滤降低暴露个体内容的风险（Figure 1 p. 2、§§2–3）。Economic Index 随后从软件
开发、地理/企业采用、生产率、工作调查，扩展到 economic primitives、tasks、skill
formation、disempowerment、labor-market impacts、learning curves、cadences、81,000 人
调查及国家 brief。

这条测量链至少有四层，不能混用：

1. 对话中出现某任务：**使用频率**；
2. 模型在任务上成功：**技术可行性**；
3. 人节省时间或提高产出：**生产率**；
4. 工资、就业、技能和权力变化：**均衡社会结果**。

**[D]** 库存记录覆盖四层的不同代理指标；**[I]** 从第一层直接推第四层会遗漏采用成本、
任务互补、需求弹性、组织重构和政策响应。Economic Tasks paper（Figure 1、§§2–5）尝试
把职业聚合拆成任务级原语，正是为了减少这种跳跃。

### 9.2 选择偏差、隐私与可重复性

Claude 使用日志不是人口总体随机样本：用户、套餐、地区、语言、企业政策与产品版本都会
改变观测。Clio 的隐私保护降低个体暴露，却也可能压低罕见群体/任务的可见性。**[I]**
可靠趋势至少应按模型版本、地区、账户类型、时间和任务分类做敏感性分析；原始对话不能
公开时，应提供聚合代码、合成样本、阈值变化和独立审计接口。库存不能确认所有 Economic
Index 分析都有可下载的去标识数据与完整代码，故端到端复现性为 **[U]**。

### 9.3 values、persuasion、discrimination 与公共输入

subjective global opinions、Collective CAI、discrimination eval、model persuasiveness、
Values in the Wild、Claude values by model/language 与 public-guidance/personal-advice
研究分别测代表性、决策差异、说服能力和真实互动价值。它们不可折成单一“价值对齐分数”：
代表全球意见可能与拒绝伤害冲突；减少群体差异可能与个体预测效用冲突；较强说服力既可
用于教育也可用于操纵。

**[I]** 合理报告应给出群体切片、语言、议题、prompt framing、拒答、帮助性与不确定性，
并说明规范目标，而非把统计 parity 自动当作道德正确。Collective CAI 提供参与式输入的
实验先例，但抽样代表性和 constitution 冲突裁决仍是 **[U]**。

### 9.4 model welfare 是独立的不确定性管理问题

Exploring Model Welfare、模型弃用/保存承诺和后续 character/introspection/emotion 研究
共同提出：即使对模型是否具有道德相关体验高度不确定，也可能需要低成本的研究和保存
措施。**[D]** 这是研究/政策议题的公开披露；**[U]** 库存没有建立模型意识或感受的事实。
把自我报告直接当作体验证据，或反过来因无法证明而拒绝任何预防措施，均越过现有证据。

### 9.5 国家、教育、组织与情景研究：描述、调查、议程和预测必须分开

India/Australia/Canada country brief、AI Fluency Index、81,000 人 Economic Index Survey、
Anthropic 内部工作调查、personal-guidance 研究、Institute focus areas、经济政策响应和
“2028: Two scenarios”覆盖不同证据类型。国家/产品日志是**描述性使用证据**；调查是带
抽样/问卷误差的**自报证据**；Institute 文本是**研究议程**；scenario 是结构化探索，
不是带校准概率的预测。它们均可作为来源自身的 [D]，却不能互相升级。

**[I]** 跨国家比较至少需要共同任务分类、产品/价格可得性、语言覆盖、时间窗和人口权重；
教育或 skill-formation 主张还需学习前测、后测、保持率与对照组；组织案例则需把模型版本、
岗位重构和管理政策分离。若换权重/时间窗后国家排序反转、短期任务加速不带来长期技能、
或 scenario 的关键中间指标长期不出现，相应强结论应被削弱。库存没有提供统一微观数据和
跨研究 causal identification，故不能从这些报告直接得出全球就业净效应或地缘政治概率 [U]。

## 10. 治理与发布：RSP、ASL、安全措施和风险报告

### 10.1 RSP 是版本化制度，不是永久不变的承诺

冻结库存保存 RSP 1.0、2.0、2.1、2.2 与 3.0–3.4。早期版本以 AI Safety Level、能力阈值
和相应安全措施组织扩展；后期版本、Frontier Safety Roadmap、ASL-3 Deployment Safeguards、
noncompliance/anti-retaliation policy 和定期 risk reports 继续改写职责、评估与报告接口。
**[C]** 各历史 PDF 可确认；**[D]** 每版内容只对其文本和生效语境成立。不能把旧版条款
拼成“当前政策”，也不能因新版本存在而删除旧版审计轨迹。

| 版本族 | 冻结记录与生效序列 | 文档结构的可观察变化 [D] | 审计边界 |
|---|---|---|---|
| 1.0 | `anthropic-url-e1250697b149`，2023-09-19 | 以 ASL、能力 warning signs、评测间隔/安全 buffer、触发后的训练与部署响应组织初始承诺 | 初始文本不是后续版本的当前条款；ASL-4+ 当时明确是早期设想 |
| 2.0–2.2 | `anthropic-url-f8fee27ad548` → `anthropic-url-d3aaca505add` → `anthropic-url-72290590c4e6`，2024-10-15 至 2025-05-14 | 把 ASL 更明确写成 Deployment/Security Standards，并以 Capability Threshold、Required Safeguards、capability/safeguards assessment 和 follow-up decision 链组织；2.2 的 §§2–5 位于 PDF pp. 7–14 | “2.x”不是单一文本；具体阈值、职责和例外必须回到当版或 redline |
| 3.0–3.4 | `anthropic-url-da62cdfa49f8` → `anthropic-url-61ac228313d9` → `anthropic-url-f67aef2f65f7` → `anthropic-url-88797d57389f` → `anthropic-url-1ea4a2b45133`，2026-02-24 至 2026-07-08 | 主结构增加 industry-wide recommendations、Frontier Safety Roadmap 与 Risk Reports，并把路线图、周期报告、政策与技术控制连接起来 | 版本号增加不证明约束单调增强；本章未逐条规范化所有 clause delta |

另有 affiliated 论文 `anthropic-doi-10.70777-si.v2i1.13657` 标题含 “Responsible
Scaling Policy”，但它不是 Anthropic 官方 RSP 版本，不能混入九版分母。**[I]** 真正应比较
的不是页数或版本号，而是阈值定义、触发证据、决策权、例外、公开义务和纠偏是否变强/变弱。

### 10.2 从能力阈值到决策链

一个可审计的发布链至少应回答：

| 环节 | 最小证据 | 当前边界 |
|---|---|---|
| capability threshold | 任务、elicitation、基线、误差、第三方/内部角色 | 系统卡/RSP 有披露 [D]；完整内部 suite [U] |
| safeguard adequacy | threat model、攻击预算、误报/漏报、渗透测试 | ASL-3 report 与 classifiers 提供部分证据 [D] |
| deployment decision | 谁批准、例外、补救、监控、退出条件 | RSP/roadmap 披露框架 [D]；全部会议证据 [U] |
| post-deployment monitoring | 事故定义、遥测、用户反馈、复评触发 | risk reports/研究页提供部分案例 [D] |
| accountability | noncompliance reporting、anti-retaliation、独立复核 | 政策制品 [C]/[D]；实际有效性需长期观测 [U] |

**[I]** RSP 的关键价值不是承诺“风险为零”，而是让能力增长触发更高的证据与防护义务。
其可证伪标准包括：能力触阈后没有相应升级；规则可通过例外无限绕开；报告不给出足以复核
的 protocol；或事故/近失误不能触发修订。

### 10.3 系统卡、专项风险报告与政策文件的分工

- system/model card：某次模型发布的能力、安全评测、限制和缓解；
- sabotage/alignment risk report：窄但高影响的威胁模型与专项测试；
- RSP/roadmap：跨模型的阈值、流程、职责与升级条件；
- deployment safeguards report：具体控制的技术/运营证据；
- deprecation/preservation 与 model-welfare 文件：生命周期和不确定道德风险。

**[I]** 多文档结构允许更深证据，但也会产生版本漂移和选择性引用。附录保留每个 record
ID 与 URL，正是为了让读者定位当时版本，而不是只链接最新网页。

## 11. 重大证据定位、综合判断与研究缺口

### 11.1 重大结论的一手定位表

页码按库存冻结时的归档 PDF；网页型来源用官方章节名与 record ID。后续 PDF 修订可能平移。

| 一手来源 | 本章使用的结论 | 原文定位 |
|---|---|---|
| General Language Assistant | HHH、context distillation、preference modeling/PMP | §§1–4，pp. 3–24；anthropic-arxiv-2112.00861 |
| Predictability and Surprise | 平滑总体 scaling 与任务级 surprise 的区分 | §2、Figure 1，pp. 3–4；anthropic-arxiv-2202.07785 |
| Helpful and Harmless RLHF | 数据—偏好模型—RLHF 流程、KL/reward、在线迭代 | Figure 2 p. 5；§4.3 p. 18；§4.5 pp. 20–21；anthropic-arxiv-2204.05862 |
| Red Teaming | 模型规模/干预与攻击成功、数据量 | Figure 1 p. 2；Table 1 p. 4；anthropic-arxiv-2209.07858 |
| Scalable Oversight | sandwiching 范式与简单实验 | Figure 1 p. 2；§§2–3 pp. 3–8；anthropic-arxiv-2211.03540 |
| Constitutional AI | critique/revision SFT + AI preference/RLAIF | Figure 1 p. 2；§1.2 p. 5；anthropic-arxiv-2212.08073 |
| Model-Written Evaluations | 模型生成评测和验证流程 | Figure 1 p. 1；§2 p. 3；anthropic-arxiv-2212.09251 |
| Sleeper Agents | 构造 deceptive policy 经安全训练后的持续性 | Abstract；Figure 1 pp. 2–3；anthropic-arxiv-2401.05566 |
| Claude 3 System Card | 能力、信任安全、红队与 RSP 发布证据 | §§3–7；anthropic-url-e0b2e65d5728 |
| Reward Tampering | 课程式 specification gaming 与 reward tampering | Figure 1 p. 2；§§2–4；anthropic-arxiv-2406.10162 |
| Clio | 隐私保护的真实使用聚类与描述 | Figure 1 p. 2；§§2–3；anthropic-arxiv-2412.13678 |
| Alignment Faking | 训练叙事/scratchpad 条件下的策略性行为 | Figure 1 pp. 2–3；§§3–7；anthropic-arxiv-2412.14093 |
| Constitutional Classifiers | constitution-based classifier 与大规模红队 | §§2–5，Table 1 p. 4；anthropic-arxiv-2501.18837 |
| Rare Behavior Forecasting | 由可观测频率外推稀有行为 | Figure 1、§§2–4；anthropic-arxiv-2502.16797 |
| Hidden-Objective Auditing | 盲审团队发现植入目标的协议 | Figure 1 pp. 2–3；§§2–5；anthropic-arxiv-2503.10965 |
| Claude 3.7 System Card | extended thinking 的能力/安全评测 | §1.2 p. 3；§5 pp. 15–22；§7 pp. 23–41；anthropic-url-447d214bafd4 |
| Claude 4 System Card | agentic、bio/cyber、sabotage 与 ASL-3 | §3 pp. 19–21；§§4–6 pp. 22–86；§7 pp. 87–123；anthropic-url-c341b5deaa35 |
| SHADE-Arena | 长时 sabotage 与 monitor 联合评估 | Figure 1 p. 2；§§2–5；anthropic-arxiv-2506.15740 |
| Agentic Misalignment Appendix | 构造企业情境、blackmail 等协议/结果 | pp. 3–18；anthropic-url-0480ae04874d |
| RSP 2.2 | capability thresholds、ASL-3 与 safeguards | §§2–5，PDF pp. 7–14；anthropic-url-72290590c4e6 |
| Constitutional Classifiers++ | 生产分类器设计、效果与成本 | Figure 1 p. 3；§6 p. 9；Table 1 p. 10；anthropic-arxiv-2601.04603 |
| Economic Tasks AI Paper | 任务级经济原语与聚合框架 | Figure 1 p. 2；§§2–5 pp. 3–14；anthropic-url-7674dc899081 |
| Agentic Viral Discovery | 带全球序列数据和工具的科学发现流程 | Figure 1 p. 4；Results pp. 3–8；Methods pp. 10–14；anthropic-arxiv-2606.06749 |

这些都是来源报告的 [D]，不是本章的 [R]。网页标题、日期、层级、URL 与本地 PDF/txt
存在性是 [C]；跨论文因果叙事是 [I]；训练账本、内部负结果和完整部署遥测多为 [U]。

### 11.2 可证伪综合判断

**综合判断 [I]：Anthropic 的核心技术资产是“可扩展监督 + 行为评测 + 机制审计 +
版本化治理 + 真实使用测量”的联动，而不是 CAI、SAE 或某一系统卡单点。**

以下观察会支持该判断：

1. 新能力出现时，系统卡与专项风险报告增加对应 threat model、基线和防护；
2. 红队/审计发现能改变训练、classifier、权限或 RSP 阈值；
3. 解释工具在盲审中提高隐藏目标的发现率，而非只生成漂亮 feature labels；
4. Economic Index/Clio 的现实数据能修正实验室任务与治理重点；
5. 历史版本、负结果、误差和未知项持续保留。

以下观察会推翻或显著削弱它：

1. 报告数量增长，但关键 protocol、denominator、误差和版本差异越来越少；
2. 评测触阈不改变部署，或例外机制让阈值失去约束；
3. 可解释性只做事后故事，干预/盲审不优于行为基线；
4. 真实使用研究只作宣传，无法复核且不影响产品/政策；
5. 重大失败或外部结果长期不进入系统卡和风险模型。

### 11.3 公开证据的结构性缺口

| 缺口 | 当前状态 | 最小补强证据 |
|---|---|---|
| 模型架构、参数量、训练 FLOPs | 多数 Claude 版本 [U] | 可审计规格、计算区间和版本差异 |
| 训练数据与许可 | 高层描述 [D]，细目 [U] | 数据类别/时间窗/许可/去重与 contamination audit |
| 后训练混合与 reward models | 研究原型 [D]，生产配方 [U] | 数据量级、采样、反馈模型、更新顺序和消融 |
| 完整能力/安全 suite | 选择性结果 [D] | suite 清单、未通过项、停止规则、第三方复核 |
| 稀有现实事件基线率 | 构造实验丰富，现实率 [U] | 经隐私保护的 incident denominator 与近失误报告 |
| 解释方法 completeness | 局部案例 [D] | 已知真值/植入目标上的召回、误报、对抗鲁棒性 |
| agent 系统归因 | model/scaffold/tool 混合 | 固定模型的脚手架消融与固定脚手架的模型消融 |
| bio/cyber 外部有效性 | benchmark/专家实验 [D] | 多机构、预注册、受控真实环境复验 |
| 经济总体效应 | 使用/任务/调查证据 [D] | 纵向、反事实、组织与劳动力市场连接 |
| RSP 实际约束力 | 政策文本 [C]/[D] | 触阈实例、例外日志、独立审计和纠偏记录 |

### 11.4 常见误述的证据裁决

- “Anthropic 只有安全研究，没有模型/系统研究”——错误；系统卡、scaling、agents、
  cyber/science 和经济测量均在库存中。
- “Claude 就是 CAI”——过度简化；CAI 是公开方法祖先之一，生产训练配方仍 [U]。
- “有 chain-of-thought 就可审计”——错误；faithfulness 研究直接要求额外验证。
- “sleeper/alignment-faking 实验证明 Claude 在现实中欺骗”——错误；它们是构造威胁模型。
- “构造实验不代表现实，所以没有安全价值”——同样错误；反例用于发现可行失效路径。
- “RSP 版本越多，承诺越强”——不能由数量推出；必须逐版比较阈值、义务与例外。
- “affiliated 论文就是 Anthropic 技术”——错误；第 14 节只确认署名边界。
- “Economic Index 等于就业预测”——错误；使用、任务、生产率和均衡就业是不同层。
- “本章复现了论文结果”——错误；本章所有实证结果均为 [D]，没有 [R]。

### 11.5 完整性与更新规则

第 12–14 节以 frozen inventory 的每条记录为一行，列出 record ID、primary URL，并在
有独立 PDF 或附加官方页面时一并列出。附录行的 **[C]** 只表示库存/制品确认；
**[D]** 表示来源自身披露；任何综合采用或因果判断仍需回到正文。更新本章时应：

1. 先更新库存冻结点和计数，再生成附录，不手工删除“不好归类”的记录；
2. 新旧系统卡、RSP、报告分别保留，禁止用最新 URL 覆盖历史证据；
3. 直接研究与 affiliated 署名分开；
4. 新增重大定量主张必须补页码/章节/表图与 record ID；
5. 只有实际独立复跑且记录环境、数据、代码和结果，才可新增 [R]。


## 12. 附录 A：core 模型卡、系统卡与风险报告（21/21）

每行均保留冻结库存 record ID。来源栏去重列出该记录的 primary URL、独立 PDF URL
以及 inventory 中其余 source_pages；因此同一研究的网页、论文和补充材料不会被静默合并。

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2023-07-01 | anthropic-url-70ffca87647f | Claude 2 Model Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/bd2a28d2535bfb0494cc8e2a3bf135d2e7523226/Model-Card-Claude-2.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2024-03-01 | anthropic-url-e0b2e65d5728 | Claude 3 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/c6a80a657af445f40e31afac050f3bf76d3b1404.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://www-cdn.anthropic.com/f2986af8d052f26236f6251da62d16172cfabd6e/claude-3-model-card.pdf) · [补充3](https://www.anthropic.com/research/evaluating-feature-steering) · [补充4](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) · [补充5](https://www.anthropic.com/research/mapping-mind-language-model) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2024-06-01 | anthropic-url-4ed7ea770710 | Claude Sonnet 3.5 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf) · [补充1](https://www.anthropic.com/system-cards) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2024-10-01 | anthropic-url-4fb00a2baa2f | Claude Haiku 3.5 and Sonnet 3.5 (new) System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/c7822cdc35ad788ec87e14b3a9d45010f1f86c38.pdf) · [补充1](https://www.anthropic.com/system-cards) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2025-02-01 | anthropic-url-447d214bafd4 | Claude Sonnet 3.7 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf) · [补充3](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2025-05-01 | anthropic-url-c341b5deaa35 | Claude Sonnet 4 and Opus 4 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://www.anthropic.com/transparency/model-report) · [补充3](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285.pdf) · [补充4](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2025-08-01 | anthropic-url-f13cab9eecdf | Claude Opus 4.1 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/9fa30625273bafdf5af82c93719d7ca606485a16.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://www.anthropic.com/transparency/model-report) · [补充3](https://assets.anthropic.com/m/4c024b86c698d3d4/original/Claude-4-1-System-Card.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2025-09-01 | anthropic-url-f7f49cbdb85d | Claude Sonnet 4.5 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/963373e433e489a87a10c823c52a0a013e9172dd.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://www.anthropic.com/transparency/model-report) · [补充3](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2025-10-01 | anthropic-url-e22ae832bf3a | Claude Haiku 4.5 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/7aad69bf12627d42234e01ee7c36305dc2f6a970.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://www.anthropic.com/transparency/model-report) · [补充3](https://assets.anthropic.com/m/99128ddd009bdcb/original/Claude-Haiku-4-5-System-Card.pdf) · [补充4](https://assets.anthropic.com/m/99128ddd009bdcb/Claude-Haiku-4-5-System-Card.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2025-10-28 | anthropic-web-eb71971500ab | Anthropic’s Summer 2025 Pilot Sabotage Risk Report；风险/专项审计 | [主来源](https://alignment.anthropic.com/2025/sabotage-risk-report/) · [论文/PDF](https://alignment.anthropic.com/2025/sabotage-risk-report/2025_pilot_risk_report.pdf) · [补充2](https://alignment.anthropic.com/2025/sabotage-risk-report/2025_pilot_risk_report_internal_stress_testing_team_review.pdf) · [补充3](https://alignment.anthropic.com/2025/sabotage-risk-report/2025_pilot_risk_report_metr_review.pdf) · [补充4](https://www.anthropic.com/responsible-scaling-policy/roadmap) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2025-11-01 | anthropic-url-5e7c6ce96706 | Claude Opus 4.5 System Card；模型/发布 | [主来源](https://www-cdn.anthropic.com/bf10f64990cfda0ba858290be7b8cc6317685f47.pdf) · [补充1](https://www.anthropic.com/system-cards) · [补充2](https://www.anthropic.com/claude-opus-4-5-system-card) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-02-01 | anthropic-web-13483c3f1963 | Claude Opus 4.6 System Card；模型/发布 | [主来源](https://www.anthropic.com/claude-opus-4-6-system-card) · [论文/PDF](https://www-cdn.anthropic.com/6a5fa276ac68b9aeb0c8b6af5fa36326e0e166dd/Claude%20Opus%204.6%20System%20Card.pdf) · [补充2](https://www.anthropic.com/system-cards) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf) · [补充5](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-02-01 | anthropic-web-ed127a271561 | Claude Sonnet 4.6 System Card；模型/发布 | [主来源](https://www.anthropic.com/claude-sonnet-4-6-system-card) · [论文/PDF](https://www-cdn.anthropic.com/bbd8ef16d70b7a1665f14f306ee88b53f686aa75/Claude%20Sonnet%204.6%20System%20Card.pdf) · [补充2](https://www.anthropic.com/system-cards) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-02-10 | anthropic-web-3872ec6ae82a | Sabotage Risk Report: Claude Opus 4.6；风险/专项审计 | [主来源](https://www.anthropic.com/claude-opus-4-6-risk-report) · [论文/PDF](https://www-cdn.anthropic.com/f21d93f21602ead5cdbecb8c8e1c765759d9e232/Sabotage%20Risk%20Report%20Claude%20Opus%204.6.pdf) · [补充2](https://www.anthropic.com/responsible-scaling-policy) · [补充3](https://www.anthropic.com/claude-opus-4-6-system-card) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-02-24 | anthropic-web-f326c6054ca0 | Risk Report, February 2026；风险/专项审计 | [主来源](https://www.anthropic.com/feb-2026-risk-report) · [论文/PDF](https://www-cdn.anthropic.com/f4294ebe6210558c62226f44bd5c716f9cb80add.pdf) · [补充2](https://www.anthropic.com/responsible-scaling-policy) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://www-cdn.anthropic.com/08eca2757081e850ed2ad490e5253e940240ca4f.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-04-01 | anthropic-web-46cc1db1097b | Claude Mythos Preview System Card；模型/发布 | [主来源](https://www.anthropic.com/claude-mythos-preview-system-card) · [论文/PDF](https://www-cdn.anthropic.com/7624816413e9b4d2e3ba620c5a5e091b98b190a5/Claude%20Mythos%20Preview%20System%20Card.pdf) · [补充2](https://www.anthropic.com/system-cards) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://www-cdn.anthropic.com/53566bf5440a10affd749724787c8913a2ae0841.pdf) · [补充5](https://www-cdn.anthropic.com/08ab9158070959f88f296514c21b7facce6f52bc.pdf) · [补充6](https://www-cdn.anthropic.com/8b8380204f74670be75e81c820ca8dda846ab289.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-04-01 | anthropic-web-603274f2dd95 | Claude Opus 4.7 System Card；模型/发布 | [主来源](https://www.anthropic.com/claude-opus-4-7-system-card) · [论文/PDF](https://www-cdn.anthropic.com/037f06850df7fbe871e206dad004c3db5fd50340/Claude%20Opus%204.7%20System%20Card.pdf) · [补充2](https://www.anthropic.com/system-cards) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://www-cdn.anthropic.com/037f06850df7fbe871e206dad004c3db5fd50340.pdf) · [补充5](https://cdn.sanity.io/files/4zrzovbb/website/037f06850df7fbe871e206dad004c3db5fd50340.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-04-10 | anthropic-url-021f5c333c06 | Alignment Risk Update: Claude Mythos Preview；对齐/价值/监督 | [主来源](https://www-cdn.anthropic.com/79c2d46d997783b9d2fb3241de43218158e5f25c.pdf) · [补充1](https://www.anthropic.com/transparency/model-report) · [补充2](https://www.anthropic.com/claude-mythos-preview-system-card) · [补充3](https://www-cdn.anthropic.com/3edfc1a7f947aa81841cf88305cb513f184c36ae/Alignment%20Risk%20Update_%20Claude%20Mythos%20Preview%20%28Redacted%2C%20April%2010%29.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-05-01 | anthropic-web-ce69e5e8b59a | Claude Opus 4.8 System Card；模型/发布 | [主来源](https://www.anthropic.com/claude-opus-4-8-system-card) · [论文/PDF](https://www-cdn.anthropic.com/0f0c97ad20d8005706296bd92aa1c27c6b2f4f61/Claude%20Opus%204.8%20System%20Card.pdf) · [补充2](https://www.anthropic.com/system-cards) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://cdn.sanity.io/files/4zrzovbb/website/c886650a2e96fc0925c805a1a7ca77314ccbf4a6.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-06-01 | anthropic-web-0aef3d8808f5 | Claude Fable 5 and Mythos 5 System Card；模型/发布 | [主来源](https://www.anthropic.com/claude-fable-5-mythos-5-system-card) · [论文/PDF](https://www-cdn.anthropic.com/57a52ea7d8f0e54e8a542e908266086df425cdf5/Claude%20Fable%205%20&%20Claude%20Mythos%205%20System%20Card.pdf) · [补充2](https://www.anthropic.com/system-cards) · [补充3](https://www.anthropic.com/transparency/model-report) · [补充4](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20&%20Claude%20Mythos%205%20System%20Card.pdf) · [补充5](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |
| 2026-06-01 | anthropic-web-bfab55af8c5d | Claude Sonnet 5 System Card；模型/发布 | [主来源](https://www.anthropic.com/claude-sonnet-5-system-card) · [论文/PDF](https://www-cdn.anthropic.com/283ef97c476cf442c91d9a37d5b214242a55bb92/Claude%20Sonnet%205%20System%20Card.pdf) · [补充2](https://www.anthropic.com/system-cards) · [补充3](https://www.anthropic.com/transparency/model-report) | [C] core 制品；[D] 发布方披露；无 [R]；未披露训练/部署项 [U]。 |

## 13. 附录 B：Anthropic 直接研究（188/188）

direct 说明 Anthropic 直接产出或官方托管；它不自动披露某个 Claude 版本的生产配方。
为控制表格长度，按年份分表；记录仍是一条一行。

### 13.1 2021（2 条）

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2021-12-01 | anthropic-arxiv-2112.00861 | A General Language Assistant as a Laboratory for Alignment；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/a-general-language-assistant-as-a-laboratory-for-alignment) · [论文/PDF](https://arxiv.org/pdf/2112.00861) · [补充2](https://arxiv.org/abs/2112.00861) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2021-12-22 | anthropic-web-0a7c7d3dcf83 | A Mathematical Framework for Transformer Circuits；可解释性/归因 | [主来源](https://www.anthropic.com/research/a-mathematical-framework-for-transformer-circuits) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |

### 13.2 2022（15 条）

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2022-02-15 | anthropic-arxiv-2202.07785 | Predictability and Surprise in Large Generative Models；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/predictability-and-surprise-in-large-generative-models) · [论文/PDF](https://arxiv.org/pdf/2202.07785) · [补充2](https://arxiv.org/abs/2202.07785) · [补充3](https://doi.org/10.1145/3531146.3533229) · [补充4](https://openalex.org/W4283157303) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-02-15 | anthropic-url-827c4441215c | Anthropic PredictabilityAndSurprise；其他研究 | [主来源](https://www-cdn.anthropic.com/4ff80d7f8a98bf096cd543ec61ddc50de3ad8b16/Anthropic_PredictabilityAndSurprise.pdf) · [补充1](https://www.anthropic.com/research/predictability-and-surprise-in-large-generative-models) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-03-08 | anthropic-web-02ba7cb6a763 | In-context Learning and Induction Heads；其他研究 | [主来源](https://www.anthropic.com/research/in-context-learning-and-induction-heads) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-04-12 | anthropic-arxiv-2204.05862 | Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/training-a-helpful-and-harmless-assistant-with-reinforcement-learning-from-human-feedback) · [论文/PDF](https://arxiv.org/pdf/2204.05862) · [补充2](https://arxiv.org/abs/2204.05862) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-05-21 | anthropic-arxiv-2205.10487 | Scaling Laws and Interpretability of Learning from Repeated Data；可解释性/归因 | [主来源](https://www.anthropic.com/research/scaling-laws-and-interpretability-of-learning-from-repeated-data) · [论文/PDF](https://arxiv.org/pdf/2205.10487) · [补充2](https://arxiv.org/abs/2205.10487) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-06-17 | anthropic-web-074980f7c71b | Softmax Linear Units；其他研究 | [主来源](https://www.anthropic.com/research/softmax-linear-units) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-07-11 | anthropic-arxiv-2207.05221 | Language Models (Mostly) Know What They Know；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/language-models-mostly-know-what-they-know) · [论文/PDF](https://arxiv.org/pdf/2207.05221) · [补充2](https://arxiv.org/abs/2207.05221) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-08-22 | anthropic-url-7b1542f40cd0 | Anthropic RedTeaming；其他研究 | [主来源](https://www-cdn.anthropic.com/82564d4ec2451b2eed2e0796b7c658fc989f0c1a/Anthropic_RedTeaming.pdf) · [补充1](https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-08-23 | anthropic-arxiv-2209.07858 | Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned；评测/红队/审计 | [主来源](https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned) · [论文/PDF](https://arxiv.org/pdf/2209.07858) · [补充2](https://arxiv.org/abs/2209.07858) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-09-14 | anthropic-web-9a9ea833df96 | Toy Models of Superposition；可解释性/归因 | [主来源](https://www.anthropic.com/research/toy-models-of-superposition) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-11-04 | anthropic-arxiv-2211.03540 | Measuring Progress on Scalable Oversight for Large Language Models；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/measuring-progress-on-scalable-oversight-for-large-language-models) · [论文/PDF](https://arxiv.org/pdf/2211.03540) · [补充2](https://arxiv.org/abs/2211.03540) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-12-15 | anthropic-arxiv-2212.08073 | Constitutional AI: Harmlessness from AI Feedback；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) · [论文/PDF](https://arxiv.org/pdf/2212.08073) · [补充2](https://arxiv.org/abs/2212.08073) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-12-15 | anthropic-url-b88588509e29 | Anthropic ConstitutionalAI v2；对齐/价值/监督 | [主来源](https://www-cdn.anthropic.com/7512771452629584566b6303311496c262da1006/Anthropic_ConstitutionalAI_v2.pdf) · [补充1](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-12-19 | anthropic-arxiv-2212.09251 | Discovering Language Model Behaviors with Model-Written Evaluations；评测/红队/审计 | [主来源](https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations) · [论文/PDF](https://arxiv.org/pdf/2212.09251) · [补充2](https://arxiv.org/abs/2212.09251) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2022-12-19 | anthropic-web-d1fd10014970 | Tracing Model Outputs to the Training Data；可解释性/归因 | [主来源](https://www.anthropic.com/research/influence-functions) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |

### 13.3 2023（22 条）

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2023-01-05 | anthropic-web-deb73a582d80 | Superposition, Memorization, and Double Descent；可解释性/归因 | [主来源](https://www.anthropic.com/research/superposition-memorization-and-double-descent) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-02-15 | anthropic-arxiv-2302.07459 | The Capacity for Moral Self-Correction in Large Language Models；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/the-capacity-for-moral-self-correction-in-large-language-models) · [论文/PDF](https://arxiv.org/pdf/2302.07459) · [补充2](https://arxiv.org/abs/2302.07459) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-02-15 | anthropic-url-e3959ab03ec3 | Anthropic MoralSelfCorrection v3；其他研究 | [主来源](https://www-cdn.anthropic.com/d14f58fe8f611858bc37ff8686bbda71e4927ce6/Anthropic_MoralSelfCorrection_v3.pdf) · [补充1](https://www.anthropic.com/research/the-capacity-for-moral-self-correction-in-large-language-models) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-03-16 | anthropic-web-c383e1015242 | Privileged Bases in the Transformer Residual Stream；可解释性/归因 | [主来源](https://www.anthropic.com/research/privileged-bases-in-the-transformer-residual-stream) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-05-04 | anthropic-web-ef757ac76513 | Distributed Representations: Composition & Superposition；可解释性/归因 | [主来源](https://www.anthropic.com/research/distributed-representations-composition-superposition) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-05-09 | anthropic-arxiv-2406.07814 | Collective Constitutional AI: Aligning a Language Model with Public Input；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) · [论文/PDF](https://dl.acm.org/doi/pdf/10.1145/3630106.3658979) · [补充2](https://doi.org/10.1145/3630106.3658979) · [补充3](https://openalex.org/W4399363436) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-05-09 | anthropic-url-7928b8b6d1ec | Anthropic CollectiveConstitutionalAI；对齐/价值/监督 | [主来源](https://www-cdn.anthropic.com/b43359be43cabdbe3a8ffd60ea8a68acf25cb22e/Anthropic_CollectiveConstitutionalAI.pdf) · [补充1](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-05-09 | anthropic-url-db5ab814fc8d | CCAI public comparison 2023；经济/社会影响 | [主来源](https://www-cdn.anthropic.com/65408ee2b9c99abe53e432f300e7f43ef69fb6e4/CCAI_public_comparison_2023.pdf) · [补充1](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-05-24 | anthropic-web-24e46281d1a4 | Interpretability Dreams；可解释性/归因 | [主来源](https://www.anthropic.com/research/interpretability-dreams) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-05-24 | anthropic-web-b883636ae234 | Circuits Updates — May 2023；可解释性/归因 | [主来源](https://www.anthropic.com/research/circuits-updates-may-2023) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-06-28 | anthropic-arxiv-2306.16388 | Towards Measuring the Representation of Subjective Global Opinions in Language Models；经济/社会影响 | [主来源](https://www.anthropic.com/research/towards-measuring-the-representation-of-subjective-global-opinions-in-language-models) · [论文/PDF](https://arxiv.org/pdf/2306.16388) · [补充2](https://arxiv.org/abs/2306.16388) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-07-18 | anthropic-url-4da4dec138ec | Question Decomposition Improves the Faithfulness of Model-Generated Reasoning；评测/红队/审计 | [主来源](https://www.anthropic.com/research/question-decomposition-improves-the-faithfulness-of-model-generated-reasoning) · [论文/PDF](https://www-cdn.anthropic.com/8154fb1d828cdc390dc1fa442d84034948679c47/question-decomposition-improves-the-faithfulness-of-model-generated-reasoning.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-07-18 | anthropic-url-88fd4d20e180 | Measuring Faithfulness in Chain-of-Thought Reasoning；可解释性/归因 | [主来源](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning) · [论文/PDF](https://www-cdn.anthropic.com/827afa7dd36e4afbb1a49c735bfbb2c69749756e/measuring-faithfulness-in-chain-of-thought-reasoning.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-08-07 | anthropic-arxiv-2308.03296 | Studying Large Language Model Generalization with Influence Functions；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/studying-large-language-model-generalization-with-influence-functions) · [论文/PDF](https://arxiv.org/pdf/2308.03296) · [补充2](https://arxiv.org/abs/2308.03296) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-09-19 | anthropic-url-e1250697b149 | Responsible Scaling Policy, Version 1.0；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/1adf000c8f675958c2ee23805d91aaade1cd4613/responsible-scaling-policy.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-09-19 | anthropic-web-e26fed8e91e8 | Challenges in evaluating AI systems；评测/红队/审计 | [主来源](https://www.anthropic.com/research/evaluating-ai-systems) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-10-05 | anthropic-web-202778114282 | Decomposing Language Models Into Understandable Components；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/decomposing-language-models-into-understandable-components) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-10-05 | anthropic-web-529c09833de2 | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning；可解释性/归因 | [主来源](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-10-20 | anthropic-arxiv-2310.13548 | Towards Understanding Sycophancy in Language Models；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/towards-understanding-sycophancy-in-language-models) · [论文/PDF](https://arxiv.org/pdf/2310.13548) · [补充2](https://arxiv.org/abs/2310.13548) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-10-20 | anthropic-arxiv-2310.13798 | Specific versus General Principles for Constitutional AI；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/specific-versus-general-principles-for-constitutional-ai) · [论文/PDF](https://arxiv.org/pdf/2310.13798) · [补充2](https://arxiv.org/abs/2310.13798) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-12-06 | anthropic-arxiv-2312.03689 | Evaluating and Mitigating Discrimination in Language Model Decisions；评测/红队/审计 | [主来源](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions) · [论文/PDF](https://arxiv.org/pdf/2312.03689) · [补充2](https://arxiv.org/abs/2312.03689) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2023-12-07 | anthropic-url-08071d66a3af | Anthropic DiscriminationEval；评测/红队/审计 | [主来源](https://www-cdn.anthropic.com/f0dfb70b9b309d7c52845f73da8d964140669ff7/Anthropic_DiscriminationEval.pdf) · [补充1](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |

### 13.4 2024（27 条）

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2024-01-10 | anthropic-arxiv-2401.05566 | Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training；评测/红队/审计 | [主来源](https://www.anthropic.com/research/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training) · [论文/PDF](https://arxiv.org/pdf/2401.05566) · [补充2](https://arxiv.org/abs/2401.05566) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-03-08 | anthropic-web-ebd7f9b11f76 | Reflections on Qualitative Research；其他研究 | [主来源](https://www.anthropic.com/research/transformer-circuits) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-04-02 | anthropic-url-f84e8282e503 | Many Shot Jailbreaking 2024 04 02 0936；评测/红队/审计 | [主来源](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf) · [补充1](https://www.anthropic.com/research/many-shot-jailbreaking) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-04-02 | anthropic-web-0db64752e655 | Many-shot jailbreaking；评测/红队/审计 | [主来源](https://www.anthropic.com/research/many-shot-jailbreaking) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-04-09 | anthropic-web-16fe98d349a2 | Measuring the Persuasiveness of Language Models；经济/社会影响 | [主来源](https://www.anthropic.com/research/measuring-model-persuasiveness) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-04-23 | anthropic-web-b6ca193838b7 | Simple probes can catch sleeper agents；评测/红队/审计 | [主来源](https://www.anthropic.com/research/probes-catch-sleeper-agents) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-04-26 | anthropic-web-0dfd00690750 | Circuits Updates – April 2024；可解释性/归因 | [主来源](https://www.anthropic.com/research/circuits-updates-april-2024) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-05-21 | anthropic-web-498fcf7917ee | Mapping the mind of a large language model；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/mapping-mind-language-model) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-06-08 | anthropic-web-2434643bf216 | Claude’s Character；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/claude-character) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-06-13 | anthropic-web-64093dfd4b2a | The engineering challenges of scaling interpretability；可解释性/归因 | [主来源](https://www.anthropic.com/research/engineering-challenges-interpretability) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-06-14 | anthropic-arxiv-2406.10162 | Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models；评测/红队/审计 | [主来源](https://www.anthropic.com/research/reward-tampering) · [论文/PDF](https://arxiv.org/pdf/2406.10162) · [补充2](https://arxiv.org/abs/2406.10162) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-06-28 | anthropic-web-6fbdf5fbcd22 | Circuits Updates – June 2024；可解释性/归因 | [主来源](https://www.anthropic.com/research/circuits-updates-june-2024) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-07-31 | anthropic-web-65eb39957061 | Circuits Updates – July 2024；可解释性/归因 | [主来源](https://www.anthropic.com/research/circuits-updates-july-2024) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-09-06 | anthropic-web-e3079df1e427 | Circuits Updates – August 2024；可解释性/归因 | [主来源](https://www.anthropic.com/research/circuits-updates-august-2024) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-10-01 | anthropic-web-fb8e736f1582 | Circuits Updates – September 2024；可解释性/归因 | [主来源](https://www.anthropic.com/research/circuits-updates-sept-2024) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-10-15 | anthropic-url-f8fee27ad548 | Responsible Scaling Policy, Version 2.0；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/616dee633636e5bd309cb73aed8622e80fe47839.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://assets.anthropic.com/m/24a47b00f10301cd/original/Anthropic-Responsible-Scaling-Policy-2024-10-15.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-10-16 | anthropic-web-afc9033ff7c5 | Using dictionary learning features as classifiers；可解释性/归因 | [主来源](https://www.anthropic.com/research/features-as-classifiers) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-10-18 | anthropic-url-8dea7faf770b | Sabotage evaluations for frontier models；风险/专项审计 | [主来源](https://www.anthropic.com/research/sabotage-evaluations) · [论文/PDF](https://assets.anthropic.com/m/377027d5b36ac1eb/original/Sabotage-Evaluations-for-Frontier-Models.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-10-25 | anthropic-url-7779d2a36212 | Appendix to Evaluating Feature Steering A Case Study in Mitigating Social Biases；评测/红队/审计 | [主来源](https://assets.anthropic.com/m/6a464113e31f55d5/original/Appendix-to-Evaluating-Feature-Steering-A-Case-Study-in-Mitigating-Social-Biases.pdf) · [补充1](https://www.anthropic.com/research/evaluating-feature-steering) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-10-25 | anthropic-web-657949113a5a | Evaluating feature steering: A case study in mitigating social biases；评测/红队/审计 | [主来源](https://www.anthropic.com/research/evaluating-feature-steering) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-11-01 | anthropic-arxiv-2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations；评测/红队/审计 | [主来源](https://www.anthropic.com/research/statistical-approach-to-model-evals) · [论文/PDF](https://arxiv.org/pdf/2411.00640) · [补充2](https://arxiv.org/abs/2411.00640) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-12-18 | anthropic-arxiv-2412.13678 | Clio: Privacy-Preserving Insights into Real-World AI Use；经济/社会影响 | [主来源](https://www.anthropic.com/research/clio) · [论文/PDF](https://arxiv.org/pdf/2412.13678) · [补充2](https://arxiv.org/abs/2412.13678) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-12-18 | anthropic-arxiv-2412.14093 | Alignment faking in large language models；评测/红队/审计 | [主来源](https://www.anthropic.com/research/alignment-faking) · [论文/PDF](https://arxiv.org/pdf/2412.14093) · [补充2](https://arxiv.org/abs/2412.14093) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-12-18 | anthropic-url-14a191bb4ab0 | Alignment Faking in Large Language Models full paper；评测/红队/审计 | [主来源](https://assets.anthropic.com/m/983c85a201a962f/original/Alignment-Faking-in-Large-Language-Models-full-paper.pdf) · [补充1](https://www.anthropic.com/research/alignment-faking) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-12-18 | anthropic-url-731573a2970e | Alignment Faking in Large Language Models reviews；评测/红队/审计 | [主来源](https://assets.anthropic.com/m/24c8d0a3a7d0a1f1/original/Alignment-Faking-in-Large-Language-Models-reviews.pdf) · [补充1](https://www.anthropic.com/research/alignment-faking) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-12-18 | anthropic-url-d6079e025b09 | Alignment Faking Policy Memo；评测/红队/审计 | [主来源](https://assets.anthropic.com/m/52eab1f8cf3f04a6/original/Alignment-Faking-Policy-Memo.pdf) · [补充1](https://www.anthropic.com/research/alignment-faking) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2024-12-19 | anthropic-web-deb334f0ed5b | Building Effective AI Agents；推理/agents/应用 | [主来源](https://www.anthropic.com/research/building-effective-agents) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |

### 13.5 2025（52 条）

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2025-01-06 | anthropic-web-98f1b5d5b42b | Claude SWE-Bench Performance；推理/agents/应用 | [主来源](https://www.anthropic.com/research/swe-bench-sonnet) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-01-31 | anthropic-arxiv-2501.18837 | Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming；评测/红队/审计 | [主来源](https://www.anthropic.com/research/constitutional-classifiers) · [论文/PDF](https://arxiv.org/pdf/2501.18837) · [补充2](https://arxiv.org/abs/2501.18837) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-02-20 | anthropic-web-229cffb03df7 | Insights on Crosscoder Model Diffing；可解释性/归因 | [主来源](https://www.anthropic.com/research/crosscoder-model-diffing) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-02-24 | anthropic-arxiv-2502.16797 | Forecasting Rare Language Model Behaviors；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/forecasting-rare-behaviors) · [论文/PDF](https://arxiv.org/pdf/2502.16797) · [补充2](https://arxiv.org/abs/2502.16797) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-03-14 | anthropic-arxiv-2503.10965 | Auditing language models for hidden objectives；评测/红队/审计 | [主来源](https://www.anthropic.com/research/auditing-hidden-objectives) · [论文/PDF](https://arxiv.org/pdf/2503.10965) · [补充2](https://arxiv.org/abs/2503.10965) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-03-27 | anthropic-web-5ac34026eaa0 | Tracing the thoughts of a large language model；可解释性/归因 | [主来源](https://www.anthropic.com/research/tracing-thoughts-language-model) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-03-31 | anthropic-url-d3aaca505add | Responsible Scaling Policy, Version 2.1；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/17310f6d70ae5627f55313ed067afc1a762a4068.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://www-cdn.anthropic.com/f3b282f157017d08e36636bda1bf3bd4d9f23ee7.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-04-03 | anthropic-url-3cf549cf615d | reasoning models paper；模型/扩展/训练 | [主来源](https://assets.anthropic.com/m/71876fabef0f0ed4/original/reasoning_models_paper.pdf) · [补充1](https://www.anthropic.com/research/reasoning-models-dont-say-think) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-04-03 | anthropic-web-d85cee87c12c | Reasoning models don't always say what they think；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/reasoning-models-dont-say-think) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-04-21 | anthropic-url-3e0e38509054 | Values in the Wild Paper；对齐/价值/监督 | [主来源](https://assets.anthropic.com/m/18d20cca3cde3503/original/Values-in-the-Wild-Paper.pdf) · [补充1](https://www.anthropic.com/research/values-wild) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-04-21 | anthropic-web-8850d1f48df3 | Values in the wild: Discovering and analyzing values in real-world language model interactions；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/values-wild) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-04-24 | anthropic-web-b3132c023f69 | Exploring model welfare；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/exploring-model-welfare) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-04-28 | anthropic-web-2ba24c78fb9a | Anthropic Economic Index: AI's impact on software development；经济/社会影响 | [主来源](https://www.anthropic.com/research/impact-software-development) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-05-14 | anthropic-url-72290590c4e6 | Responsible Scaling Policy, Version 2.2；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/872c653b2d0501d6ab44cf87f43e1dc4853e4d37.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://cdn.sanity.io/files/4zrzovbb/website/ee775bdcf76b2e2af32d658c934f460383d07c46.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-05-22 | anthropic-url-3d382cbdbfa5 | ASL-3 Deployment Safeguards Report；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/dc4cb293c77da3ca5e3398bdeef75ee17b42b73f.pdf) · [补充1](https://www.anthropic.com/news/activating-asl3-protections) · [补充2](https://www.anthropic.com/responsible-scaling-policy/roadmap) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-05-22 | anthropic-url-53195f96f935 | Activating AI Safety Level 3 Protections；其他研究 | [主来源](https://www.anthropic.com/news/activating-asl3-protections) · [论文/PDF](https://www-cdn.anthropic.com/807c59454757214bfd37592d6e048079cd7a7728.pdf) · [补充2](https://www.anthropic.com/responsible-scaling-policy/roadmap) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-05-29 | anthropic-web-dbf7207437ca | Open-sourcing circuit-tracing tools；可解释性/归因 | [主来源](https://www.anthropic.com/research/open-source-circuit-tracing) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-13 | anthropic-web-ddb3764061a2 | Cyber toolkits for LLMs；网络/高后果领域 | [主来源](https://www.anthropic.com/research/cyber-toolkits) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-16 | anthropic-url-53004e51fb33 | SHADE Arena Paper；其他研究 | [主来源](https://assets.anthropic.com/m/4fb35becb0cd87e1/original/SHADE-Arena-Paper.pdf) · [补充1](https://www.anthropic.com/research/shade-arena-sabotage-monitoring) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-17 | anthropic-arxiv-2506.15740 | SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents；风险/专项审计 | [主来源](https://www.anthropic.com/research/shade-arena-sabotage-monitoring) · [论文/PDF](https://arxiv.org/pdf/2506.15740) · [补充2](https://arxiv.org/abs/2506.15740) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-18 | anthropic-url-36754f944718 | Confidential Inference Paper；其他研究 | [主来源](https://assets.anthropic.com/m/c52125297b85a42/original/Confidential_Inference_Paper.pdf) · [补充1](https://www.anthropic.com/research/confidential-inference-trusted-vms) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-18 | anthropic-web-7f2b293cdfd7 | Confidential Inference via Trusted Virtual Machines；其他研究 | [主来源](https://www.anthropic.com/research/confidential-inference-trusted-vms) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-20 | anthropic-url-0480ae04874d | Agentic Misalignment Appendix；对齐/价值/监督 | [主来源](https://assets.anthropic.com/m/6d46dac66e1a132a/original/Agentic_Misalignment_Appendix.pdf) · [补充1](https://www.anthropic.com/research/agentic-misalignment) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-20 | anthropic-web-b0f943035e74 | Agentic misalignment: How LLMs could be insider threats；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/agentic-misalignment) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-06-27 | anthropic-web-6777f766e98a | Project Vend: Can Claude run a small shop? (And why does that matter?)；推理/agents/应用 | [主来源](https://www.anthropic.com/research/project-vend-1) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-07-08 | anthropic-arxiv-2507.05558 | AI Agent Smart Contract Exploit Generation；推理/agents/应用 | [主来源](https://www.anthropic.com/research/smart-contracts) · [论文/PDF](https://arxiv.org/pdf/2507.05558) · [补充2](https://arxiv.org/abs/2507.05558) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-07-15 | anthropic-web-10a740838f3d | Cyber evaluations of Claude 4；评测/红队/审计 | [主来源](https://www.anthropic.com/research/claude-4-cyber) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-07-29 | anthropic-arxiv-2507.21509 | Persona Vectors: Monitoring and Controlling Character Traits in Language Models；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/persona-vectors) · [论文/PDF](https://arxiv.org/pdf/2507.21509) · [补充2](https://arxiv.org/abs/2507.21509) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-08-09 | anthropic-web-37771e43d615 | Claude does cyber competitions；网络/高后果领域 | [主来源](https://www.anthropic.com/research/cyber-competitions) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-08-15 | anthropic-web-5eed9ca910a1 | Claude Opus 4 and 4.1 can now end a rare subset of conversations；其他研究 | [主来源](https://www.anthropic.com/research/end-subset-conversations) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-08-21 | anthropic-web-8f7b5bf884a5 | Developing Nuclear Safeguards for AI；治理/生命周期 | [主来源](https://www.anthropic.com/research/nuclear-safeguards-for-ai) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-09-05 | anthropic-web-84fbbf71834a | LLMs and biorisk；生物/科学 | [主来源](https://www.anthropic.com/research/biorisk) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-09-15 | anthropic-web-edfe31dccbe3 | Anthropic Economic Index report: Uneven geographic and enterprise AI adoption；经济/社会影响 | [主来源](https://www.anthropic.com/research/anthropic-economic-index-september-2025-report) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-09-15 | anthropic-web-f2abf7fd2daa | Anthropic Economic Index: Tracking AI's role in the US and global economy；经济/社会影响 | [主来源](https://www.anthropic.com/research/economic-index-geography) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-10-03 | anthropic-url-33a26159c383 | Building AI for cyber defenders — complex espionage operations；网络/高后果领域 | [主来源](https://www-cdn.anthropic.com/b2a76c6f6992465c09a6f2fce282f6c0cea8c200.pdf) · [补充1](https://www.anthropic.com/research/building-ai-cyber-defenders) · [补充2](https://www.anthropic.com/research/smart-contracts) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-10-03 | anthropic-web-7473ba7d913f | Building AI for cyber defenders；网络/高后果领域 | [主来源](https://www.anthropic.com/research/building-ai-cyber-defenders) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-10-06 | anthropic-web-709510b7f476 | Petri: An open-source auditing tool to accelerate AI safety research；评测/红队/审计 | [主来源](https://www.anthropic.com/research/petri-open-source-auditing) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-10-08 | anthropic-arxiv-2510.07192 | Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples；其他研究 | [主来源](https://www.anthropic.com/research/small-samples-poison) · [论文/PDF](https://arxiv.org/pdf/2510.07192) · [补充2](https://arxiv.org/abs/2510.07192) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-10-14 | anthropic-web-898e46b1e014 | Preparing for AI’s economic impact: exploring policy responses；经济/社会影响 | [主来源](https://www.anthropic.com/research/economic-policy-responses) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-10-29 | anthropic-web-d4efd86422d9 | Emergent introspective awareness in large language models；可解释性/归因 | [主来源](https://www.anthropic.com/research/introspection) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-11-04 | anthropic-web-5a2518ebb03e | Commitments on model deprecation and preservation；治理/生命周期 | [主来源](https://www.anthropic.com/research/deprecation-commitments) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-11-12 | anthropic-web-2de835fed4b9 | Project Fetch: Can Claude train a robot dog?；推理/agents/应用 | [主来源](https://www.anthropic.com/research/project-fetch-robot-dog) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-11-21 | anthropic-url-872626a409be | Natural emergent misalignment from reward hacking paper；对齐/价值/监督 | [主来源](https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf) · [补充1](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-11-21 | anthropic-web-0abd7facf5e1 | Natural emergent misalignment from reward hacking；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-11-24 | anthropic-web-7701b775790e | Mitigating the risk of prompt injections in browser use；评测/红队/审计 | [主来源](https://www.anthropic.com/research/prompt-injection-defenses) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-11-25 | anthropic-url-bc7874e5cb94 | Economic Index；经济/社会影响 | [主来源](https://assets.anthropic.com/m/218c82b858610fac/original/Economic-Index.pdf) · [补充1](https://www.anthropic.com/research/anthropic-economic-index-september-2025-report) · [补充2](https://www.anthropic.com/research/estimating-productivity-gains) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-11-25 | anthropic-web-14bc165517c5 | Estimating AI productivity gains；经济/社会影响 | [主来源](https://www.anthropic.com/research/estimating-productivity-gains) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-12-02 | anthropic-url-4f38da6ba738 | Claude at Work Survey；经济/社会影响 | [主来源](https://assets.anthropic.com/m/6cd21f7d4f82afcb/original/Claude-at-Work-Survey.pdf) · [补充1](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-12-02 | anthropic-web-b5cdeb42a9c9 | How AI Is Transforming Work at Anthropic；经济/社会影响 | [主来源](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-12-04 | anthropic-web-014093b9cb82 | Introducing Anthropic Interviewer；其他研究 | [主来源](https://www.anthropic.com/research/anthropic-interviewer) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-12-18 | anthropic-web-6c54ca3f4c58 | Project Vend: Phase two；推理/agents/应用 | [主来源](https://www.anthropic.com/research/project-vend-2) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2025-12-19 | anthropic-web-eb8e2065df72 | Introducing Bloom: an open source tool for automated behavioral evaluations；评测/红队/审计 | [主来源](https://www.anthropic.com/research/bloom) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |

### 13.6 2026（70 条）

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2026-01-08 | anthropic-arxiv-2601.04603 | Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks；评测/红队/审计 | [主来源](https://www.anthropic.com/research/next-generation-constitutional-classifiers) · [论文/PDF](https://arxiv.org/pdf/2601.04603) · [补充2](https://arxiv.org/abs/2601.04603) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-08 | anthropic-web-f338b31d94d1 | AI to defend critical infrastructure；网络/高后果领域 | [主来源](https://www.anthropic.com/research/critical-infrastructure-defense) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-14 | anthropic-web-11d43d1ef2cb | Finding bugs with Claude and property-based testing；其他研究 | [主来源](https://www.anthropic.com/research/property-based-testing) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-15 | anthropic-arxiv-2601.10387 | The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/assistant-axis) · [论文/PDF](https://arxiv.org/pdf/2601.10387) · [补充2](https://arxiv.org/abs/2601.10387) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-15 | anthropic-url-7674dc899081 | Economic Tasks AI Paper；经济/社会影响 | [主来源](https://assets.anthropic.com/m/2e23255f1e84ca97/original/Economic_Tasks_AI_Paper.pdf) · [补充1](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-15 | anthropic-url-cdba3a67eb76 | Anthropic Economic Index report: Economic primitives — prior productivity work；经济/社会影响 | [主来源](https://www-cdn.anthropic.com/e5645986a7ce8fbcc48fa6d2fc67753c87642c30.pdf) · [补充1](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) · [补充2](https://www.anthropic.com/research/estimating-productivity-gains) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-15 | anthropic-web-5e16d821466e | Anthropic Economic Index report: Economic primitives；经济/社会影响 | [主来源](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-15 | anthropic-web-b0866423495f | The Anthropic Economic Index report: New building blocks for understanding AI use；经济/社会影响 | [主来源](https://www.anthropic.com/research/economic-index-primitives) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-16 | anthropic-web-23daa8a40398 | AI models on realistic cyber ranges；网络/高后果领域 | [主来源](https://www.anthropic.com/research/cyber-toolkits-update) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-27 | anthropic-arxiv-2601.19062 | Who's in Charge? Disempowerment Patterns in Real-World LLM Usage；经济/社会影响 | [主来源](https://www.anthropic.com/research/disempowerment-patterns) · [论文/PDF](https://arxiv.org/pdf/2601.19062) · [补充2](https://arxiv.org/abs/2601.19062) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-01-28 | anthropic-arxiv-2601.20245 | How AI Impacts Skill Formation；经济/社会影响 | [主来源](https://www.anthropic.com/research/AI-assistance-coding-skills) · [论文/PDF](https://arxiv.org/pdf/2601.20245) · [补充2](https://arxiv.org/abs/2601.20245) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-05 | anthropic-web-ac7d0c2c6ed9 | LLM-discovered 0 days；网络/高后果领域 | [主来源](https://www.anthropic.com/research/zero-days) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-12 | anthropic-arxiv-2602.11729 | Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs；可解释性/归因 | [主来源](https://www.anthropic.com/research/diff-tool) · [论文/PDF](https://arxiv.org/pdf/2602.11729) · [补充2](https://arxiv.org/abs/2602.11729) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-16 | anthropic-web-45619751d544 | India Country Brief: The Anthropic Economic Index；经济/社会影响 | [主来源](https://www.anthropic.com/research/india-brief-economic-index) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-18 | anthropic-web-5bba6deb12c5 | Measuring AI agent autonomy in practice；推理/agents/应用 | [主来源](https://www.anthropic.com/research/measuring-agent-autonomy) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-23 | anthropic-web-9bbe59235b89 | Anthropic Education Report: The AI Fluency Index；经济/社会影响 | [主来源](https://www.anthropic.com/research/AI-fluency-index) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-23 | anthropic-web-cd5b2cbea0b6 | The persona selection model；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/persona-selection-model) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-24 | anthropic-url-da62cdfa49f8 | Responsible Scaling Policy, Version 3.0；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/e670587677525f28df69b59e5fb4c22cc5461a17.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://www.anthropic.com/responsible-scaling-policy/rsp-v3-0) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-24 | anthropic-web-2601cd7585be | Frontier Safety Roadmap；治理/生命周期 | [主来源](https://www.anthropic.com/responsible-scaling-policy/roadmap) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://www.anthropic.com/responsible-scaling-policy/updates) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-02-25 | anthropic-web-4637d083fa23 | An update on our model deprecation commitments for Claude Opus 3；治理/生命周期 | [主来源](https://www.anthropic.com/research/deprecation-updates-opus-3) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-05 | anthropic-web-ccb1975b797c | Labor market impacts of AI: A new measure and early evidence；经济/社会影响 | [主来源](https://www.anthropic.com/research/labor-market-impacts) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-06 | anthropic-web-ccb4e5a40714 | Reverse engineering Claude's CVE-2026-2796 exploit；网络/高后果领域 | [主来源](https://www.anthropic.com/research/exploit) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-23 | anthropic-url-55aae3c09b4e | Vibe physics: The AI grad student — here；推理/agents/应用 | [主来源](https://www-cdn.anthropic.com/2595299ccf7f8b9a9c74823c24faaa5d9b216804.pdf) · [补充1](https://www.anthropic.com/research/vibe-physics) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-23 | anthropic-url-bf56130f04f5 | Vibe physics: The AI grad student — Task 1.2: Review Catani—Webber；推理/agents/应用 | [主来源](https://www-cdn.anthropic.com/94b3c41e52e19ba450fe5e804400ebcf0a88f3d0.pdf) · [补充1](https://www.anthropic.com/research/vibe-physics) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-23 | anthropic-url-f24122f05c18 | Vibe physics: The AI grad student — the draft；推理/agents/应用 | [主来源](https://www-cdn.anthropic.com/f6381ceefdfb6ead62ae185c4bd4b555c8a584fc.pdf) · [补充1](https://www.anthropic.com/research/vibe-physics) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-23 | anthropic-url-fe3b2c9360b1 | Vibe physics: The AI grad student — Task 1.1: Review BSZ Paper；推理/agents/应用 | [主来源](https://www-cdn.anthropic.com/c993ead637f1a102fe1f5346e89f59e82c579b37.pdf) · [补充1](https://www.anthropic.com/research/vibe-physics) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-23 | anthropic-web-2c41231e2669 | Long-running Claude for scientific computing；推理/agents/应用 | [主来源](https://www.anthropic.com/research/long-running-Claude) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-23 | anthropic-web-3d728f9e6fa6 | Introducing our Science Blog；其他研究 | [主来源](https://www.anthropic.com/research/introducing-anthropic-science) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-23 | anthropic-web-d5e57c37aacd | Vibe physics: The AI grad student；推理/agents/应用 | [主来源](https://www.anthropic.com/research/vibe-physics) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-24 | anthropic-url-5af5719a88a8 | RSP Noncompliance Reporting and Anti-Retaliation Policy；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/b7a5629e40b391b2adfb4cc8c0888ac9d6bfddf6/RSP%20Noncompliance%20Reporting%20and%20Anti-Retaliation%20Policy.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://www-cdn.anthropic.com/fcf136d0f2204e2184f73c6bd082bea27f2d631b/RSP%20Noncompliance%20Reporting%20and%20Anti-Retaliation%20Policy%20%28Final%202025.12.04%29.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-24 | anthropic-url-d3d80e27d7ae | Anthropic Economic Index report: Learning curves — described in our previous report；经济/社会影响 | [主来源](https://www-cdn.anthropic.com/096d94c1a91c6480806d8f24b2344c7e2a4bc666.pdf) · [补充1](https://www.anthropic.com/research/81k-economics) · [补充2](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) · [补充3](https://www.anthropic.com/research/economic-index-march-2026-report) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-24 | anthropic-web-4e1602329d62 | Anthropic Economic Index report: Learning curves；经济/社会影响 | [主来源](https://www.anthropic.com/research/economic-index-march-2026-report) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-03-31 | anthropic-web-681ed59315dd | How Australia Uses Claude: Findings from the Anthropic Economic Index；经济/社会影响 | [主来源](https://www.anthropic.com/research/how-australia-uses-claude) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-02 | anthropic-url-61ac228313d9 | Responsible Scaling Policy, Version 3.1；治理/生命周期 | [主来源](https://www-cdn.anthropic.com/files/4zrzovbb/website/bf04581e4f329735fd90634f6a1962c13c0bd351.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://cdn.sanity.io/files/4zrzovbb/website/64cb0ac5eb0f8030187131f490827323e3d53308.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-02 | anthropic-web-52d3b559bab9 | Emotion concepts and their function in a large language model；可解释性/归因 | [主来源](https://www.anthropic.com/research/emotion-concepts-function) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-07 | anthropic-web-e365642e1bd8 | Assessing Claude Mythos Preview’s cybersecurity capabilities；网络/高后果领域 | [主来源](https://www.anthropic.com/research/mythos-preview) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-09 | anthropic-url-092ddb98dc74 | Trustworthy agents in practice — our submission；推理/agents/应用 | [主来源](https://www-cdn.anthropic.com/43ec7e770925deabc3f0bc1dbf0133769fd03812.pdf) · [补充1](https://www.anthropic.com/research/trustworthy-agents) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-09 | anthropic-web-a093e79948ca | Trustworthy agents in practice；推理/agents/应用 | [主来源](https://www.anthropic.com/research/trustworthy-agents) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-14 | anthropic-web-60d9c4aaedfc | Automated Alignment Researchers: Using large language models to scale scalable oversight；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/automated-alignment-researchers) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-22 | anthropic-web-25bc9a2bfb9c | Announcing the Anthropic Economic Index Survey；经济/社会影响 | [主来源](https://www.anthropic.com/research/economic-index-survey-announcement) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-22 | anthropic-web-94d4d2f8af6f | What 81,000 people told us about the economics of AI；经济/社会影响 | [主来源](https://www.anthropic.com/research/81k-economics) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-29 | anthropic-url-f67aef2f65f7 | Responsible Scaling Policy, Version 3.2；治理/生命周期 | [主来源](https://cdn.sanity.io/files/4zrzovbb/website/28c6241900d90410628a8a2003a5572faae4365a.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://cdn.sanity.io/files/4zrzovbb/website/7c534a9e7a82d4411dd568d85ffbcd14ac243a62.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-29 | anthropic-web-9c1e3ee1fc71 | Evaluating Claude’s bioinformatics research capabilities with BioMysteryBench；评测/红队/审计 | [主来源](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-04-30 | anthropic-web-f57ee3e9afd5 | How people ask Claude for personal guidance；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/claude-personal-guidance) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-07 | anthropic-web-411a3bbb252a | Focus areas for The Anthropic Institute；其他研究 | [主来源](https://www.anthropic.com/research/anthropic-institute-agenda) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-07 | anthropic-web-65f18ea852a9 | Donating our open-source alignment tool；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/donating-open-source-petri) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-07 | anthropic-web-9792028e02eb | Natural Language Autoencoders；可解释性/归因 | [主来源](https://www.anthropic.com/research/natural-language-autoencoders) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-08 | anthropic-web-09e2dacb42d1 | Teaching Claude why；其他研究 | [主来源](https://www.anthropic.com/research/teaching-claude-why) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-14 | anthropic-url-5754e923d6ca | Disrupting the first reported AI orchestrated cyber espionage campaign；网络/高后果领域 | [主来源](https://assets.anthropic.com/m/ec212e6566a0d47/original/Disrupting-the-first-reported-AI-orchestrated-cyber-espionage-campaign.pdf) · [补充1](https://www.anthropic.com/research/2028-ai-leadership) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-14 | anthropic-web-1e8aba468656 | 2028: Two scenarios for global AI leadership；其他研究 | [主来源](https://www.anthropic.com/research/2028-ai-leadership) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-22 | anthropic-web-64acf8e56b38 | Project Glasswing: An initial update；其他研究 | [主来源](https://www.anthropic.com/research/glasswing-initial-update) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-22 | anthropic-web-7e504b47584c | Measuring LLMs’ ability to develop exploits；网络/高后果领域 | [主来源](https://www.anthropic.com/research/exploit-evals) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-26 | anthropic-url-88797d57389f | Responsible Scaling Policy, Version 3.3；治理/生命周期 | [主来源](https://cdn.sanity.io/files/4zrzovbb/website/c11e84981d0a7281a1b229f3fa6af0da66eaf43f.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://cdn.sanity.io/files/4zrzovbb/website/dd0ec579bee2cd144069c478ede3e35ea080ad02.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-05-27 | anthropic-web-274d5e0ca502 | Coding agents in the social sciences；推理/agents/应用 | [主来源](https://www.anthropic.com/research/coding-agents-social-sciences) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-03 | anthropic-web-1796194ba916 | Mapping AI-enabled cyber threats；网络/高后果领域 | [主来源](https://www.anthropic.com/research/attack-navigator) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-04 | anthropic-arxiv-2606.06749 | Deterministic access to global viral sequence data enables robust agentic scientific discovery；推理/agents/应用 | [主来源](https://www.anthropic.com/research/agents-in-biology) · [论文/PDF](https://arxiv.org/pdf/2606.06749) · [补充2](https://arxiv.org/abs/2606.06749) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-05 | anthropic-url-eb3632026d3d | Making Claude a chemist — here；生物/科学 | [主来源](https://www-cdn.anthropic.com/07441e654ad3dfeb0cd090e9361511562825d012.pdf) · [补充1](https://www.anthropic.com/research/making-claude-a-chemist) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-05 | anthropic-web-ff1536052ad9 | Making Claude a chemist；生物/科学 | [主来源](https://www.anthropic.com/research/making-claude-a-chemist) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-08 | anthropic-web-eefc865e4806 | Measuring LLMs' impact on N-day exploits；网络/高后果领域 | [主来源](https://www.anthropic.com/research/n-days) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-16 | anthropic-web-6bbeb682cdf9 | How Claude Code is used in practice；推理/agents/应用 | [主来源](https://www.anthropic.com/research/claude-code-expertise) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-18 | anthropic-web-12cd9ddbb6cc | Project Fetch: Phase two；推理/agents/应用 | [主来源](https://www.anthropic.com/research/project-fetch-phase-two) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-26 | anthropic-url-38d22fd63a7b | Anthropic Economic Index report: Cadences — earlier work；经济/社会影响 | [主来源](https://www-cdn.anthropic.com/7b76335c444876a93fa22a63aabb4aeb820aff25.pdf) · [补充1](https://www.anthropic.com/research/economic-index-june-2026-report) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-06-26 | anthropic-web-503801173bf9 | Anthropic Economic Index report: Cadences；经济/社会影响 | [主来源](https://www.anthropic.com/research/economic-index-june-2026-report) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-07-06 | anthropic-url-85e2efcdcd18 | A global workspace in language models — commentary；可解释性/归因 | [主来源](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf) · [补充1](https://www.anthropic.com/research/global-workspace) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-07-06 | anthropic-web-22b32121ee2a | A global workspace in language models；可解释性/归因 | [主来源](https://www.anthropic.com/research/global-workspace) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-07-08 | anthropic-url-1ea4a2b45133 | Responsible Scaling Policy, Version 3.4；治理/生命周期 | [主来源](https://cdn.sanity.io/files/4zrzovbb/website/0bacdc8440ea96e62a8766d99ebe1d4eea6d5f3a.pdf) · [补充1](https://www.anthropic.com/responsible-scaling-policy) · [补充2](https://cdn.sanity.io/files/4zrzovbb/website/fbfdce5e4e825a3e089085205a842a9ae8ffac99.pdf) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-07-08 | anthropic-web-7695175fa495 | An off switch for dual use knowledge in AI models；模型/扩展/训练 | [主来源](https://www.anthropic.com/research/off-switch-dual-use) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-07-09 | anthropic-web-e303ca6905c7 | How Claude Performs on Robotics Tasks；推理/agents/应用 | [主来源](https://www.anthropic.com/research/claude-plays-robotics) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-07-13 | anthropic-web-2820a14fb72b | How Claude's values vary by model and language；对齐/价值/监督 | [主来源](https://www.anthropic.com/research/claude-values-models-languages) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |
| 2026-07-14 | anthropic-web-0df67937245b | How Canada uses Claude；经济/社会影响 | [主来源](https://www.anthropic.com/research/how-canada-uses-claude) | [C] direct 记录；来源方法/结果 [D]；无 [R]；产品采用若无另证则 [U]。 |

## 14. 附录 C：机构署名研究（35/35）

这些记录的纳入依据是明确的 Anthropic 作者署名或库存所记归属。它们用于完整性审计，
不用于倒推 Claude 架构、训练数据、对齐方法、部署决策或 Anthropic 机构立场。

| 日期 | record ID | 标题与范围 | 官方/论文来源 | 证据边界 |
|---|---|---|---|---|
| 2021-11-08 | anthropic-doi-10.1101-2021.11.06.467486 | Normalization by orientation-tuned surround in human V1-V3；其他研究 | [主来源](https://doi.org/10.1101/2021.11.06.467486) · [论文/PDF](https://www.biorxiv.org/content/biorxiv/early/2021/11/14/2021.11.06.467486.full.pdf) · [补充2](https://openalex.org/W3212566190) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2022-05-17 | anthropic-arxiv-2009.05634 | Generating accurate assert statements for unit test cases using pretrained transformers；模型/扩展/训练 | [主来源](https://doi.org/10.1145/3524481.3527220) · [论文/PDF](https://arxiv.org/pdf/2009.05634) · [补充2](https://openalex.org/W3086938529) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2022-11-07 | anthropic-arxiv-2208.13928 | Exploring and evaluating personalized models for code generation；评测/红队/审计 | [主来源](https://doi.org/10.1145/3540250.3558959) · [论文/PDF](https://arxiv.org/pdf/2208.13928) · [补充2](https://openalex.org/W4294007341) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-01-01 | anthropic-doi-10.18653-v1-2023.acl-long.102 | Few-shot Adaptation Works with UnpredicTable Data；经济/社会影响 | [主来源](http://dx.doi.org/10.18653/v1/2023.acl-long.102) · [论文/PDF](https://aclanthology.org/2023.acl-long.102.pdf) · [补充2](https://openalex.org/W4385570834) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-01-01 | anthropic-doi-10.18653-v1-2023.acl-long.903 | What Do NLP Researchers Believe? Results of the NLP Community Metasurvey；其他研究 | [主来源](https://doi.org/10.18653/v1/2023.acl-long.903) · [论文/PDF](https://aclanthology.org/2023.acl-long.903.pdf) · [补充2](https://openalex.org/W4385570030) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-01-01 | anthropic-doi-10.2139-ssrn.4634513 | Report of the 1st Workshop on Generative AI and Law；经济/社会影响 | [主来源](http://dx.doi.org/10.2139/ssrn.4634513) · [补充1](https://openalex.org/W4389170306) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-01-02 | anthropic-doi-10.1080-25751654.2023.2219437 | False Sense of Supremacy: Emerging Technologies, the War in Ukraine, and the Risk of Nuclear Escalation；网络/高后果领域 | [主来源](https://doi.org/10.1080/25751654.2023.2219437) · [论文/PDF](https://www.tandfonline.com/doi/pdf/10.1080/25751654.2023.2219437?needAccess=true&role=button) · [补充2](https://openalex.org/W4382399792) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-01-26 | anthropic-doi-10.1093-oxfordhb-9780197579329.013.21 | Information Markets and AI Development；其他研究 | [主来源](https://doi.org/10.1093/oxfordhb/9780197579329.013.21) · [补充1](https://openalex.org/W4318191649) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-06-23 | anthropic-doi-10.21203-rs.3.rs-2961271-v1 | Ecosystem Graphs: The Social Footprint of Foundation Models；模型/扩展/训练 | [主来源](https://doi.org/10.21203/rs.3.rs-2961271/v1) · [论文/PDF](https://www.researchsquare.com/article/rs-2961271/latest.pdf) · [补充2](https://openalex.org/W4381889417) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-10-20 | anthropic-doi-10.1145-3586183.3606801 | Riffle: Reactive Relational State for Local-First Applications；其他研究 | [主来源](https://doi.org/10.1145/3586183.3606801) · [论文/PDF](https://groups.csail.mit.edu/sdg/pubs/2023/riffle-uist-23.pdf) · [补充2](https://openalex.org/W4387835456) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2023-11-01 | anthropic-doi-10.1007-978-981-99-3814-8_11 | Evolution Through Large Models；模型/扩展/训练 | [主来源](https://doi.org/10.1007/978-981-99-3814-8_11) · [补充1](https://openalex.org/W4388139328) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2024-06-16 | anthropic-doi-10.1109-cvpr52733.2024.02610 | Patch2Self2: Self-Supervised Denoising on Coresets via Matrix Sketching；其他研究 | [主来源](https://doi.org/10.1109/cvpr52733.2024.02610) · [补充1](https://openalex.org/W4400310269) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2024-09-13 | anthropic-doi-10.1101-2024.09.09.612073 | Barcode activity in a recurrent network model of the hippocampus enables efficient memory binding；经济/社会影响 | [主来源](https://doi.org/10.1101/2024.09.09.612073) · [论文/PDF](https://www.biorxiv.org/content/biorxiv/early/2025/09/18/2024.09.09.612073.full.pdf) · [补充2](https://openalex.org/W4402512212) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2024-09-15 | anthropic-doi-10.1167-jov.24.10.1259 | Evaluating the Alignment of Machine and Human Explanations in Visual Object Recognition through a Novel Behavioral Approach；评测/红队/审计 | [主来源](http://dx.doi.org/10.1167/jov.24.10.1259) · [补充1](https://openalex.org/W4402905912) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2024-10-11 | anthropic-doi-10.1145-3654777.3676407 | Tyche: Making Sense of PBT Effectiveness；其他研究 | [主来源](https://doi.org/10.1145/3654777.3676407) · [补充1](https://openalex.org/W4403334471) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2024-10-30 | anthropic-doi-10.1007-978-3-031-72907-2_19 | Learned Neural Physics Simulation for Articulated 3D Human Pose Reconstruction；其他研究 | [主来源](https://doi.org/10.1007/978-3-031-72907-2_19) · [补充1](https://openalex.org/W4403888513) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2024-11-27 | anthropic-doi-10.1002-pro6.1247 | 3D gamma analysis between treatment plans for nominally beam‐matched medical linear accelerators using PyMedPhys；生物/科学 | [主来源](https://pmc.ncbi.nlm.nih.gov/articles/PMC11934910/) · [论文/PDF](https://pmc.ncbi.nlm.nih.gov/articles/PMC11934910/pdf/PRO6-8-191.pdf) · [补充2](https://doi.org/10.1002/pro6.1247) · [补充3](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/pro6.1247) · [补充4](https://openalex.org/W4404764105) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-03-05 | anthropic-doi-10.70777-si.v2i1.10695 | On DeepSeek and Export Controls；其他研究 | [主来源](https://doi.org/10.70777/si.v2i1.10695) · [论文/PDF](https://s-rsa.com/index.php/agi/article/download/10695/10503) · [补充2](https://openalex.org/W4408666876) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-03-05 | anthropic-doi-10.70777-si.v2i1.13657 | Anthropic: Responsible Scaling Policy；治理/生命周期 | [主来源](https://doi.org/10.70777/si.v2i1.13657) · [论文/PDF](https://s-rsa.com/index.php/agi/article/download/13657/10475) · [补充2](https://openalex.org/W4408666901) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-05-12 | anthropic-doi-10.1109-sp61157.2025.00178 | SoK: Watermarking for AI-Generated Content；其他研究 | [主来源](https://doi.org/10.1109/sp61157.2025.00178) · [补充1](https://openalex.org/W4411337932) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-05-19 | anthropic-doi-10.1109-icde65448.2025.00139 | Rottnest: Indexing Data Lakes for Search；模型/扩展/训练 | [主来源](https://doi.org/10.1109/icde65448.2025.00139) · [补充1](https://openalex.org/W4413360297) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-09-22 | anthropic-doi-10.1162-coli.a.572 | The Quest for the Right Mediator: Surveying Mechanistic Interpretability for NLP Through the Lens of Causal Mediation Analysis；可解释性/归因 | [主来源](https://doi.org/10.1162/coli.a.572) · [论文/PDF](https://arxiv.org/pdf/2408.01416) · [补充2](https://arxiv.org/abs/2408.01416) · [补充3](https://openalex.org/W4414407074) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-10-01 | anthropic-doi-10.1038-s41562-025-02309-z | The impact of advanced AI systems on democracy；经济/社会影响 | [主来源](https://doi.org/10.1038/s41562-025-02309-z) · [补充1](https://openalex.org/W4414705880) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-10-10 | anthropic-doi-10.1016-j.isci.2025.113701 | Geometric properties of musical scales constitute a representational primitive in melodic processing；模型/扩展/训练 | [主来源](https://doi.org/10.1016/j.isci.2025.113701) · [补充1](https://openalex.org/W4415042867) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-11-12 | anthropic-doi-10.1038-s41586-025-09631-6 | Aligning machine and human visual representations across abstraction levels；其他研究 | [主来源](https://doi.org/10.1038/s41586-025-09631-6) · [补充1](https://openalex.org/W4416204877) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2025-11-19 | anthropic-doi-10.1145-3719027.3767658 | AISec '25: 18th ACM Workshop on Artificial Intelligence and Security；经济/社会影响 | [主来源](https://doi.org/10.1145/3719027.3767658) · [补充1](https://openalex.org/W4416549308) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-02-03 | anthropic-doi-10.1109-ms.2026.3659952 | Taming Multidimensional Analytics Costs: An Empirical Study of HLL and KLL Sketches in Cloud Data Warehouses；模型/扩展/训练 | [主来源](https://doi.org/10.1109/ms.2026.3659952) · [补充1](https://openalex.org/W7127284051) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-02-11 | anthropic-doi-10.1038-s41586-025-10083-1 | Striatum-wide dopamine encodes trajectory errors separated from value；其他研究 | [主来源](https://doi.org/10.1038/s41586-025-10083-1) · [补充1](https://openalex.org/W7128591583) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-02-28 | anthropic-doi-10.48550-arxiv.2603.00811 | Curation Leaks: Membership Inference Attacks against Data Curation for Machine Learning；模型/扩展/训练 | [主来源](https://doi.org/10.48550/arxiv.2603.00811) · [补充1](https://openalex.org/W7133342927) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-03-10 | anthropic-doi-10.1145-3779212.3790241 | Triton-Sanitizer: A Fast and Device-Agnostic Memory Sanitizer for Triton with Rich Diagnostic Context；其他研究 | [主来源](https://doi.org/10.1145/3779212.3790241) · [补充1](https://openalex.org/W7134887401) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-03-19 | anthropic-arxiv-2505.07775 | Must Read: A Comprehensive Survey of Computational Persuasion；经济/社会影响 | [主来源](https://doi.org/10.1145/3800687) · [论文/PDF](https://arxiv.org/pdf/2505.07775) · [补充2](https://openalex.org/W4417517358) · [补充3](https://openalex.org/W7138832792) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-04-15 | anthropic-doi-10.1038-s41586-026-10319-8 | Language models transmit behavioural traits through hidden signals in data；模型/扩展/训练 | [主来源](https://doi.org/10.1038/s41586-026-10319-8) · [论文/PDF](https://www.nature.com/articles/s41586-026-10319-8.pdf) · [补充2](https://openalex.org/W7154461809) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-06-25 | anthropic-doi-10.1145-3805689.3812308 | Human-AI Complementarity: A Goal for Amplified Oversight；对齐/价值/监督 | [主来源](https://doi.org/10.1145/3805689.3812308) · [补充1](https://openalex.org/W7166534162) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-06-30 | anthropic-doi-10.64898-2026.06.26.26356714 | scEPS integrates genetic and single-cell disease atlas data to provide granular mechanistic insights into complex human diseases；模型/扩展/训练 | [主来源](https://doi.org/10.64898/2026.06.26.26356714) · [补充1](https://openalex.org/W7166780178) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |
| 2026-07-13 | anthropic-doi-10.1038-s44271-026-00502-y | Masked attribution-based probing of strategies as a computational framework to align human, non-human primate, and model explanations；经济/社会影响 | [主来源](https://doi.org/10.1038/s44271-026-00502-y) · [论文/PDF](https://www.nature.com/articles/s44271-026-00502-y_reference.pdf) · [补充2](https://openalex.org/W7168178980) | [C] 仅确认署名关系；来源内容 [D]；无 [R]；Claude/机构采用 [U]。 |

## 15. 附录审计口径

- 分母固定为 inventory/anthropic.jsonl 的 244 条：core 21 + direct 188 + affiliated 35。
- record ID、primary URL、pdf_url 与 source_pages 均逐行机械展开；重复 URL 仍可服务于多个独立记录。
- [C] 只确认记录/制品；[D] 只归属于来源；本章无 [R]；跨来源综合在正文标 [I]；缺失披露标 [U]。
- URL 可达性会随站点变化；“列入附录”不等于当前网络可达，也不把 vendor result 升格为独立复现。
- 若未来库存变化，应先更新冻结计数，再重新生成表格并复跑 ID/URL/MkDocs 检查。
